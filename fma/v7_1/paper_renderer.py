"""Reproducibility-oriented XeLaTeX and full-page rendering for V7.1 papers."""

from __future__ import annotations

import csv
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

from fma.codex_driver import ProcessOutputLimitExceeded, _default_process_runner
from fma.hashing import sha256_value

from .paper_runtime import (
    PaperAttemptPathsV71,
    PaperDeliveryError,
    _read_model,
    _safe_relative_path,
    _sha256_file,
    _verify_review_role_artifacts,
    _write_json,
    assert_paper_attempt_open_v71,
    current_paper_attempt_v71,
    paper_writer_lock_v71,
)
from .paper_schemas import (
    CitationManifestV71,
    FigureManifestV71,
    PaperAuthoringRequestV71,
    PaperBuildReceiptV71,
    PaperClaimLedgerV71,
    PaperContentAuditV71,
    PaperEvidenceBundleV71,
    PaperMetadataV71,
    PaperSemanticReviewV71,
    PaperToolIdentityV71,
    TableManifestV71,
)


_TEMPLATE_PATH = Path(__file__).parent / "templates" / "fma_article_v1.tex"
_PLACEHOLDERS = (
    "%%FMA_TITLE%%",
    "%%FMA_AUTHORS%%",
    "%%FMA_ABSTRACT%%",
    "%%FMA_BODY%%",
    "%%FMA_BIBLIOGRAPHY%%",
)
_LOG_FAILURES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"Overfull \\[hv]box"), "overfull_box"),
    (re.compile(r"Undefined control sequence"), "undefined_control_sequence"),
    (
        re.compile(r"(?:Reference|Citation).+undefined", re.IGNORECASE),
        "undefined_reference_or_citation",
    ),
    (
        re.compile(r"There were undefined references", re.IGNORECASE),
        "undefined_references",
    ),
    (re.compile(r"Missing character:"), "missing_character"),
    (re.compile(r"Float too large", re.IGNORECASE), "float_too_large"),
)
_MAX_TOOL_STDOUT_BYTES = 8 * 1024 * 1024
_MAX_TOOL_STDERR_BYTES = 4 * 1024 * 1024
_VERSION_STREAM_LIMIT_BYTES = 256 * 1024


@dataclass(frozen=True)
class PaperBuildVerificationResultV71:
    ok: bool
    mismatches: list[str]
    build_hash: str | None = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "&": r"\&",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _safe_url(value: str) -> str:
    if any(ord(character) < 32 for character in value) or any(
        character in value for character in "{}\\"
    ):
        raise PaperDeliveryError("citation URL contains unsafe TeX characters")
    return value


def _replace_identifier_macro(
    text: str,
    macro: str,
    resolver: Callable[[str], str],
) -> str:
    pattern = re.compile(
        rf"\\{macro}\{{([A-Za-z][A-Za-z0-9_.:-]{{0,127}})\}}"
    )
    return pattern.sub(lambda match: resolver(match.group(1)), text)


def _render_table(
    root: Path,
    table: object,
) -> str:
    table_id = str(getattr(table, "table_id"))
    source = _safe_relative_path(root, str(getattr(table, "csv_path")))
    if not source.is_file() or _sha256_file(source) != getattr(
        table, "csv_sha256"
    ):
        raise PaperDeliveryError(f"table source changed: {table_id}")
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise PaperDeliveryError(f"table source is empty: {table_id}")
    column_count = len(rows[0])
    if column_count == 0 or any(len(row) != column_count for row in rows):
        raise PaperDeliveryError(f"table is ragged: {table_id}")
    if len(rows) - 1 > getattr(table, "max_rows"):
        raise PaperDeliveryError(f"table exceeds max_rows: {table_id}")
    if column_count > getattr(table, "max_columns"):
        raise PaperDeliveryError(f"table exceeds max_columns: {table_id}")
    columns = "Y" * column_count
    rendered_rows = []
    for index, row in enumerate(rows):
        cells = [_tex_escape(cell) for cell in row]
        rendered = " & ".join(cells) + r" \\"
        if index == 0:
            rendered = r"\textbf{" + r"} & \textbf{".join(cells) + r"} \\"
        rendered_rows.append(rendered)
        if index == 0:
            rendered_rows.append(r"\midrule")
    return (
        "\n"
        r"\begin{table}[H]"
        "\n"
        r"\centering"
        "\n"
        rf"\caption{{{_tex_escape(getattr(table, 'caption'))}}}"
        "\n"
        rf"\label{{tab:{table_id}}}"
        "\n"
        rf"\begin{{tabularx}}{{\linewidth}}{{{columns}}}"
        "\n"
        r"\toprule"
        "\n"
        + "\n".join(rendered_rows)
        + "\n"
        + r"\bottomrule"
        + "\n"
        + r"\end{tabularx}"
        + "\n"
        + r"\end{table}"
        + "\n"
    )


def _copy_figure(
    root: Path,
    build_root: Path,
    figure: object,
) -> tuple[str, str]:
    figure_id = str(getattr(figure, "figure_id"))
    source = _safe_relative_path(root, str(getattr(figure, "artifact_path")))
    if not source.is_file() or _sha256_file(source) != getattr(
        figure, "artifact_sha256"
    ):
        raise PaperDeliveryError(f"figure source changed: {figure_id}")
    if source.suffix.lower() not in {".pdf", ".png", ".jpg", ".jpeg"}:
        raise PaperDeliveryError(
            f"figure uses an unsupported format: {figure_id}"
        )
    relative = (
        Path("assets")
        / "figures"
        / f"{getattr(figure, 'artifact_sha256')}{source.suffix.lower()}"
    )
    destination = build_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if _sha256_file(destination) != getattr(figure, "artifact_sha256"):
        raise PaperDeliveryError(f"figure copy hash mismatch: {figure_id}")
    return figure_id, relative.as_posix()


def _render_bibliography(
    citations: CitationManifestV71, *, language: str
) -> str:
    if not citations.citations:
        return ""
    items: list[str] = []
    for citation in citations.citations:
        authors = ", ".join(_tex_escape(author) for author in citation.authors)
        reference = (
            f"{authors}. {_tex_escape(citation.title)}. "
            f"{_tex_escape(citation.venue)}, {citation.year}."
        )
        locator = citation.url
        if citation.doi:
            locator = "https://doi.org/" + citation.doi
        if locator:
            reference += rf" \url{{{_safe_url(locator)}}}"
        items.append(
            rf"\item \label{{cite:{citation.citation_id}}}{reference}"
        )
    heading = "参考文献" if language == "zh" else "References"
    return (
        rf"\section*{{{heading}}}"
        + "\n"
        + rf"\addcontentsline{{toc}}{{section}}{{{heading}}}"
        + "\n"
        + r"\begin{enumerate}"
        + "\n"
        + "\n".join(items)
        + "\n"
        + r"\end{enumerate}"
    )


def _render_sources(
    root: Path,
    build_root: Path,
    bundle: PaperEvidenceBundleV71,
    metadata: PaperMetadataV71,
    abstract: str,
    body: str,
    citations: CitationManifestV71,
    figures: FigureManifestV71,
    tables: TableManifestV71,
) -> str:
    numeric = {item.token_id: item for item in bundle.numeric_tokens}
    citation_map = {item.citation_id: item for item in citations.citations}
    figure_map = {item.figure_id: item for item in figures.figures}
    table_map = {item.table_id: item for item in tables.tables}
    figure_paths = dict(
        _copy_figure(root, build_root, item) for item in figures.figures
    )

    def render_fragment(fragment: str) -> str:
        def section(match: re.Match[str]) -> str:
            return rf"\section{{{_tex_escape(match.group(2))}}}\label{{sec:{match.group(1)}}}"

        def subsection(match: re.Match[str]) -> str:
            return rf"\subsection{{{_tex_escape(match.group(2))}}}\label{{subsec:{match.group(1)}}}"

        value = re.sub(
            r"\\FMASection\{([A-Za-z][A-Za-z0-9_.:-]{0,127})\}\{([^{}]+)\}",
            section,
            fragment,
        )
        value = re.sub(
            r"\\FMASubsection\{([A-Za-z][A-Za-z0-9_.:-]{0,127})\}\{([^{}]+)\}",
            subsection,
            value,
        )
        value = re.sub(
            r",\s*\\FMAClaim\{[A-Za-z][A-Za-z0-9_.:-]{0,127}\}\s*,",
            ",",
            value,
        )
        value = _replace_identifier_macro(
            value, "FMAClaim", lambda claim_id: ""
        )
        value = _replace_identifier_macro(
            value,
            "FMAValue",
            lambda token_id: numeric[token_id].display_value,
        )
        value = _replace_identifier_macro(
            value,
            "FMACite",
            lambda citation_id: (
                rf"\hyperref[cite:{citation_id}]{{[{list(citation_map).index(citation_id) + 1}]}}"
            ),
        )

        def figure(figure_id: str) -> str:
            item = figure_map[figure_id]
            return (
                "\n"
                r"\begin{figure}[H]"
                "\n"
                r"\centering"
                "\n"
                rf"\includegraphics[width={item.width_fraction:.3f}\linewidth]"
                rf"{{{figure_paths[figure_id]}}}"
                "\n"
                rf"\caption{{{_tex_escape(item.caption)}}}"
                "\n"
                rf"\label{{fig:{figure_id}}}"
                "\n"
                r"\end{figure}"
                "\n"
            )

        figure_label = "Figure" if metadata.language == "en" else "图"
        value = re.sub(
            r"\\FMAFigure\{([A-Za-z][A-Za-z0-9_.:-]{0,127})\}([.!?])",
            lambda match: (
                rf"{figure_label}~\ref{{fig:{match.group(1)}}}"
                + match.group(2)
                + "\n\n"
                + figure(match.group(1))
            ),
            value,
        )
        value = _replace_identifier_macro(value, "FMAFigure", figure)
        table_label = "Table" if metadata.language == "en" else "表"
        value = re.sub(
            r"\\FMATable\{([A-Za-z][A-Za-z0-9_.:-]{0,127})\}([.!?])",
            lambda match: (
                rf"{table_label}~\ref{{tab:{match.group(1)}}}"
                + match.group(2)
                + "\n\n"
                + _render_table(root, table_map[match.group(1)])
            ),
            value,
        )
        value = _replace_identifier_macro(
            value,
            "FMATable",
            lambda table_id: _render_table(root, table_map[table_id]),
        )
        return value

    rendered_abstract = render_fragment(abstract)
    rendered_body = render_fragment(body)
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    if metadata.language == "en":
        template = template.replace(
            r"\textbf{\color{FMAInk}摘\quad 要}",
            r"\textbf{\color{FMAInk}Abstract}",
        )
        template = template.replace(
            "适用边界与失效条件",
            "Applicability Boundary and Failure Conditions",
        )
        template = template.replace(
            r"\begin{document}",
            "\n".join(
                (
                    r"\renewcommand{\figurename}{Figure}",
                    r"\renewcommand{\tablename}{Table}",
                    r"\begin{document}",
                )
            ),
        )
    replacements = {
        "%%FMA_TITLE%%": _tex_escape(metadata.title),
        "%%FMA_AUTHORS%%": r"\quad ".join(
            _tex_escape(author) for author in metadata.authors
        ),
        "%%FMA_ABSTRACT%%": rendered_abstract,
        "%%FMA_BODY%%": rendered_body,
        "%%FMA_BIBLIOGRAPHY%%": _render_bibliography(
            citations, language=metadata.language
        ),
    }
    for placeholder, replacement in replacements.items():
        template = template.replace(placeholder, replacement)
    unresolved = [item for item in _PLACEHOLDERS if item in template]
    if unresolved:
        raise PaperDeliveryError(
            "unresolved template placeholders: " + ", ".join(unresolved)
        )
    return template


def _minimal_environment() -> dict[str, str]:
    allowed = {
        "APPDATA",
        "COMSPEC",
        "HOME",
        "HOMEDRIVE",
        "HOMEPATH",
        "LANG",
        "LC_ALL",
        "LOCALAPPDATA",
        "PATH",
        "PATHEXT",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USERPROFILE",
        "WINDIR",
        "XDG_CACHE_HOME",
    }
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in allowed
    }
    environment.update(
        {
            "openin_any": "p",
            "openout_any": "p",
            "shell_escape": "0",
            "SOURCE_DATE_EPOCH": "946684800",
            "FORCE_SOURCE_DATE": "1",
        }
    )
    return environment


def _run(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    environment: dict[str, str] | None = None,
    max_stdout_bytes: int = _MAX_TOOL_STDOUT_BYTES,
    max_stderr_bytes: int = _MAX_TOOL_STDERR_BYTES,
) -> subprocess.CompletedProcess[bytes]:
    if max_stdout_bytes <= 0 or max_stderr_bytes <= 0:
        raise ValueError("paper command stream limits must be positive")
    try:
        result = _default_process_runner(
            command,
            cwd=cwd,
            input_text=None,
            timeout_seconds=timeout_seconds,
            env=environment or _minimal_environment(),
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )
        return subprocess.CompletedProcess(
            command,
            result.returncode,
            result.stdout.encode("utf-8"),
            result.stderr.encode("utf-8"),
        )
    except FileNotFoundError as exc:
        raise PaperDeliveryError(f"required executable not found: {command[0]}") from exc
    except ProcessOutputLimitExceeded as exc:
        raise PaperDeliveryError(
            f"paper command exceeded its output limit: {command[0]}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise PaperDeliveryError(
            f"paper command timed out after {timeout_seconds:g}s: {command[0]}"
        ) from exc


def _executable(value: str | os.PathLike[str]) -> str:
    raw = os.fspath(value)
    resolved: str | None = None
    if os.name == "nt" and not Path(raw).suffix:
        for directory in os.environ.get("PATH", "").split(os.pathsep):
            if not directory:
                continue
            candidate = Path(directory) / f"{raw}.exe"
            if candidate.is_file():
                resolved = str(candidate)
                break
    if resolved is None:
        resolved = shutil.which(raw)
    if resolved is None:
        candidate = Path(raw)
        if not candidate.is_file():
            raise PaperDeliveryError(f"executable was not found: {raw}")
        resolved = str(candidate.resolve())
    return str(Path(resolved).resolve())


def _tool_version(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    environment: dict[str, str],
) -> str:
    completed = _run(
        command,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        environment=environment,
        max_stdout_bytes=_VERSION_STREAM_LIMIT_BYTES,
        max_stderr_bytes=_VERSION_STREAM_LIMIT_BYTES,
    )
    if completed.returncode != 0:
        raise PaperDeliveryError(
            f"tool version preflight failed with return code "
            f"{completed.returncode}: {command[0]}"
        )
    text = (
        completed.stdout.decode("utf-8", errors="replace")
        + "\n"
        + completed.stderr.decode("utf-8", errors="replace")
    )
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not first_line:
        raise PaperDeliveryError(f"tool version preflight was empty: {command[0]}")
    return first_line[:500]


def _tool_identity(
    tool: str,
    executable: str,
    version: str,
    commands: list[list[str]],
) -> PaperToolIdentityV71:
    return PaperToolIdentityV71(
        tool=tool,
        resolved_path=executable,
        binary_sha256=_sha256_file(Path(executable)),
        version=version,
        argv_hash=sha256_value(commands),
    )


def _relative_to_workspace(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise PaperDeliveryError("paper build path escapes workspace") from exc


def _build_paper_unlocked_v71(
    workspace_root: str | Path,
    *,
    xelatex_command: str | os.PathLike[str] = "xelatex",
    pdfinfo_command: str | os.PathLike[str] = "pdfinfo",
    pdftoppm_command: str | os.PathLike[str] = "pdftoppm",
    timeout_seconds: float = 180.0,
) -> PaperBuildReceiptV71:
    """Build the exact current audited manuscript and render every PDF page."""

    if timeout_seconds <= 0:
        raise PaperDeliveryError("timeout_seconds must be positive")
    root = Path(workspace_root).resolve()
    paths = current_paper_attempt_v71(root)
    assert_paper_attempt_open_v71(paths)
    bundle_path = paths.attempt_root / "evidence_bundle.json"
    request_path = paths.attempt_root / "author_request.json"
    writer_packet_path = paths.attempt_root / "writer_packet.json"
    metadata_path = paths.attempt_root / "metadata.json"
    abstract_path = paths.source_root / "abstract.tex"
    body_path = paths.source_root / "body.tex"
    ledger_path = paths.manifests_root / "claim_ledger.json"
    citation_path = paths.manifests_root / "citations.json"
    figure_path = paths.manifests_root / "figures.json"
    table_path = paths.manifests_root / "tables.json"
    writer_receipt_path = paths.attempt_root / "writer_transport_receipt.json"
    writer_output_path = paths.attempt_root / "writer_output.json"
    writer_role_request_path = paths.attempt_root / "writer_role_request.json"
    audit_path = paths.reviews_root / "content_audit.json"
    semantic_path = paths.reviews_root / "semantic_review.json"
    semantic_transport_path = (
        paths.reviews_root / "semantic_transport_receipt.json"
    )

    bundle = _read_model(bundle_path, PaperEvidenceBundleV71)
    request = _read_model(request_path, PaperAuthoringRequestV71)
    metadata = _read_model(metadata_path, PaperMetadataV71)
    _read_model(ledger_path, PaperClaimLedgerV71)
    citations = _read_model(citation_path, CitationManifestV71)
    figures = _read_model(figure_path, FigureManifestV71)
    tables = _read_model(table_path, TableManifestV71)
    audit = _read_model(audit_path, PaperContentAuditV71)
    semantic = _read_model(semantic_path, PaperSemanticReviewV71)
    if audit.status != "PASS":
        raise PaperDeliveryError("content audit must pass before build")
    if semantic.verdict != "APPROVE":
        raise PaperDeliveryError("semantic review must approve before build")
    if semantic.content_audit_hash != audit.audit_hash:
        raise PaperDeliveryError("semantic review binds another content audit")
    if semantic.bundle_hash != bundle.bundle_hash:
        raise PaperDeliveryError("semantic review binds another evidence bundle")
    if (
        not semantic_transport_path.is_file()
        or _sha256_file(semantic_transport_path)
        != semantic.reviewer_transport_receipt_sha256
    ):
        raise PaperDeliveryError(
            "semantic reviewer transport receipt is missing or changed"
        )
    _verify_review_role_artifacts(paths, semantic, "semantic")
    expected_hashes = {
        request_path: audit.author_request_sha256,
        writer_packet_path: audit.writer_packet_sha256,
        metadata_path: audit.metadata_sha256,
        abstract_path: audit.abstract_sha256,
        body_path: audit.body_sha256,
        ledger_path: audit.claim_ledger_sha256,
        citation_path: audit.citation_manifest_sha256,
        figure_path: audit.figure_manifest_sha256,
        table_path: audit.table_manifest_sha256,
        writer_receipt_path: audit.writer_transport_receipt_sha256,
        writer_output_path: audit.writer_output_sha256,
        writer_role_request_path: audit.writer_role_request_sha256,
    }
    for path, expected in expected_hashes.items():
        if not path.is_file() or _sha256_file(path) != expected:
            raise PaperDeliveryError(
                f"audited paper input changed before build: {path.name}"
            )

    environment = _minimal_environment()
    environment_hash = sha256_value(environment)
    xelatex = _executable(xelatex_command)
    pdfinfo = _executable(pdfinfo_command)
    pdftoppm = _executable(pdftoppm_command)
    xelatex_version_command = [xelatex, "--version"]
    pdfinfo_version_command = [pdfinfo, "-v"]
    pdftoppm_version_command = [pdftoppm, "-v"]
    version_timeout = min(timeout_seconds, 20)
    compiler_version = _tool_version(
        xelatex_version_command,
        cwd=paths.builds_root,
        timeout_seconds=version_timeout,
        environment=environment,
    )
    pdfinfo_version = _tool_version(
        pdfinfo_version_command,
        cwd=paths.builds_root,
        timeout_seconds=version_timeout,
        environment=environment,
    )
    pdftoppm_version = _tool_version(
        pdftoppm_version_command,
        cwd=paths.builds_root,
        timeout_seconds=version_timeout,
        environment=environment,
    )
    compiler_command = [
        xelatex,
        "-no-shell-escape",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        "main.tex",
    ]
    if "miktex" in compiler_version.lower():
        compiler_command.insert(1, "-disable-installer")
    toolchain_input = {
        "environment_hash": environment_hash,
        "xelatex": {
            "resolved_path": xelatex,
            "binary_sha256": _sha256_file(Path(xelatex)),
            "version": compiler_version,
            "compiler_command": compiler_command,
        },
        "pdfinfo": {
            "resolved_path": pdfinfo,
            "binary_sha256": _sha256_file(Path(pdfinfo)),
            "version": pdfinfo_version,
        },
        "pdftoppm": {
            "resolved_path": pdftoppm,
            "binary_sha256": _sha256_file(Path(pdftoppm)),
            "version": pdftoppm_version,
        },
    }
    input_hash = sha256_value(
        {
            "template_sha256": _sha256_file(_TEMPLATE_PATH),
            "bundle_hash": bundle.bundle_hash,
            "request_hash": request.request_hash,
            "author_request_sha256": audit.author_request_sha256,
            "writer_packet_sha256": audit.writer_packet_sha256,
            "metadata_sha256": audit.metadata_sha256,
            "abstract_sha256": audit.abstract_sha256,
            "body_sha256": audit.body_sha256,
            "claim_ledger_sha256": audit.claim_ledger_sha256,
            "citation_manifest_sha256": audit.citation_manifest_sha256,
            "figure_manifest_sha256": audit.figure_manifest_sha256,
            "table_manifest_sha256": audit.table_manifest_sha256,
            "writer_transport_receipt_sha256": (
                audit.writer_transport_receipt_sha256
            ),
            "writer_role_request_sha256": audit.writer_role_request_sha256,
            "writer_output_sha256": audit.writer_output_sha256,
            "content_audit_hash": audit.audit_hash,
            "semantic_review_sha256": _sha256_file(semantic_path),
            "toolchain": toolchain_input,
        }
    )
    build_id = f"build-{input_hash[:20]}"
    final_root = paths.builds_root / build_id
    if final_root.exists():
        receipt_path = final_root / "build_receipt.json"
        if receipt_path.is_file():
            existing = _read_model(receipt_path, PaperBuildReceiptV71)
            verification = verify_paper_build_v71(root, build_id=build_id)
            if verification.ok:
                _write_json(
                    paths.attempt_root / "current_build.json",
                    {
                        "schema_version": "7.1-paper-current-build",
                        "build_id": build_id,
                        "build_hash": existing.build_hash,
                        "projection_only": True,
                    },
                )
                return existing
        raise PaperDeliveryError(
            "content-addressed build path already exists but is not valid"
        )
    scratch = paths.builds_root / f".building-{build_id}-{uuid4().hex[:8]}"
    scratch.mkdir(parents=True, exist_ok=False)
    try:
        abstract = abstract_path.read_text(encoding="utf-8")
        body = body_path.read_text(encoding="utf-8")
        generated = _render_sources(
            root,
            scratch,
            bundle,
            metadata,
            abstract,
            body,
            citations,
            figures,
            tables,
        )
        tex_path = scratch / "main.tex"
        tex_path.write_text(generated, encoding="utf-8", newline="\n")
        logs: list[bytes] = []
        final_pass_output = b""
        for pass_index in (1, 2):
            completed = _run(
                compiler_command,
                cwd=scratch,
                timeout_seconds=timeout_seconds,
                environment=environment,
            )
            logs.extend(
                [
                    f"\n--- pass {pass_index} stdout ---\n".encode(),
                    completed.stdout,
                    f"\n--- pass {pass_index} stderr ---\n".encode(),
                    completed.stderr,
                ]
            )
            if completed.returncode != 0:
                (scratch / "compiler.log").write_bytes(b"".join(logs))
                raise PaperDeliveryError(
                    f"XeLaTeX pass {pass_index} failed with return code "
                    f"{completed.returncode}"
                )
            final_pass_output = completed.stdout + b"\n" + completed.stderr
        log_bytes = b"".join(logs)
        log_text = final_pass_output.decode("utf-8", errors="replace")
        lint = sorted(
            {
                code
                for pattern, code in _LOG_FAILURES
                if pattern.search(log_text)
            }
        )
        if lint:
            raise PaperDeliveryError(
                "XeLaTeX log failed layout lint: " + ", ".join(lint)
            )
        log_path = scratch / "compiler.log"
        log_path.write_bytes(log_bytes)
        pdf_path = scratch / "main.pdf"
        if not pdf_path.is_file() or not pdf_path.read_bytes().startswith(b"%PDF-"):
            raise PaperDeliveryError("XeLaTeX did not produce a valid PDF")

        pdfinfo_run_command = [pdfinfo, "main.pdf"]
        info = _run(
            pdfinfo_run_command,
            cwd=scratch,
            timeout_seconds=min(timeout_seconds, 30),
            environment=environment,
        )
        if info.returncode != 0:
            raise PaperDeliveryError("pdfinfo failed on generated PDF")
        match = re.search(
            r"^Pages:\s+([1-9][0-9]*)\s*$",
            info.stdout.decode("utf-8", errors="replace"),
            flags=re.MULTILINE,
        )
        if not match:
            raise PaperDeliveryError("pdfinfo did not report a page count")
        page_count = int(match.group(1))
        if page_count > request.max_pages:
            raise PaperDeliveryError(
                f"paper has {page_count} pages, exceeding max_pages "
                f"{request.max_pages}"
            )
        pages_root = scratch / "pages"
        pages_root.mkdir()
        page_images: dict[str, str] = {}
        pdftoppm_run_commands: list[list[str]] = []
        for page in range(1, page_count + 1):
            stem = pages_root / f"page-{page:03d}"
            relative_stem = Path("pages") / f"page-{page:03d}"
            render_command = [
                pdftoppm,
                "-png",
                "-r",
                "144",
                "-f",
                str(page),
                "-l",
                str(page),
                "-singlefile",
                "main.pdf",
                str(relative_stem),
            ]
            pdftoppm_run_commands.append(render_command)
            rendered = _run(
                render_command,
                cwd=scratch,
                timeout_seconds=timeout_seconds,
                environment=environment,
            )
            image_path = stem.with_suffix(".png")
            if rendered.returncode != 0 or not image_path.is_file():
                raise PaperDeliveryError(
                    f"failed to render PDF page {page} to PNG"
                )
            relative = (
                Path("pages") / image_path.name
            ).as_posix()
            page_images[relative] = _sha256_file(image_path)

        future_tex_path = final_root / "main.tex"
        future_pdf_path = final_root / "main.pdf"
        page_images = {
            _relative_to_workspace(root, final_root / path): digest
            for path, digest in sorted(page_images.items())
        }
        receipt = PaperBuildReceiptV71.seal(
            bundle_hash=bundle.bundle_hash,
            content_audit_hash=audit.audit_hash,
            semantic_review_sha256=_sha256_file(semantic_path),
            template_sha256=_sha256_file(_TEMPLATE_PATH),
            author_request_sha256=audit.author_request_sha256,
            writer_packet_sha256=audit.writer_packet_sha256,
            metadata_sha256=audit.metadata_sha256,
            abstract_sha256=audit.abstract_sha256,
            body_sha256=audit.body_sha256,
            claim_ledger_sha256=audit.claim_ledger_sha256,
            citation_manifest_sha256=audit.citation_manifest_sha256,
            figure_manifest_sha256=audit.figure_manifest_sha256,
            table_manifest_sha256=audit.table_manifest_sha256,
            writer_transport_receipt_sha256=(
                audit.writer_transport_receipt_sha256
            ),
            writer_role_request_sha256=audit.writer_role_request_sha256,
            writer_output_sha256=audit.writer_output_sha256,
            generated_tex_path=_relative_to_workspace(root, future_tex_path),
            generated_tex_sha256=_sha256_file(tex_path),
            pdf_path=_relative_to_workspace(root, future_pdf_path),
            pdf_sha256=_sha256_file(pdf_path),
            compiler_command=compiler_command,
            compiler_version=compiler_version,
            xelatex_identity=_tool_identity(
                "xelatex",
                xelatex,
                compiler_version,
                [xelatex_version_command, compiler_command, compiler_command],
            ),
            pdfinfo_identity=_tool_identity(
                "pdfinfo",
                pdfinfo,
                pdfinfo_version,
                [pdfinfo_version_command, pdfinfo_run_command],
            ),
            pdftoppm_identity=_tool_identity(
                "pdftoppm",
                pdftoppm,
                pdftoppm_version,
                [pdftoppm_version_command, *pdftoppm_run_commands],
            ),
            environment_hash=environment_hash,
            compiler_log_sha256=_sha256_file(log_path),
            page_images=page_images,
            layout_lint=[],
            built_at=_utc_now(),
        )
        _write_json(
            scratch / "build_receipt.json", receipt.model_dump(mode="json")
        )
        scratch.rename(final_root)
        _write_json(
            paths.attempt_root / "current_build.json",
            {
                "schema_version": "7.1-paper-current-build",
                "build_id": build_id,
                "build_hash": receipt.build_hash,
                "projection_only": True,
            },
        )
        return receipt
    except Exception as exc:
        if scratch.exists():
            _write_json(
                scratch / "build_failure.json",
                {
                    "schema_version": "7.1-paper-build-failure",
                    "input_hash": input_hash,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "scientific_qualification_granted": False,
                    "real_world_action_authorized": False,
                },
            )
        raise


def build_paper_v71(
    workspace_root: str | Path,
    *,
    xelatex_command: str | os.PathLike[str] = "xelatex",
    pdfinfo_command: str | os.PathLike[str] = "pdfinfo",
    pdftoppm_command: str | os.PathLike[str] = "pdftoppm",
    timeout_seconds: float = 180.0,
) -> PaperBuildReceiptV71:
    """Build one paper as a serialized V7.1 read-check-write transaction."""

    root = Path(workspace_root).resolve()
    with paper_writer_lock_v71(
        root, timeout_seconds=max(30.0, timeout_seconds)
    ):
        return _build_paper_unlocked_v71(
            root,
            xelatex_command=xelatex_command,
            pdfinfo_command=pdfinfo_command,
            pdftoppm_command=pdftoppm_command,
            timeout_seconds=timeout_seconds,
        )


def load_current_build_v71(
    paths: PaperAttemptPathsV71,
) -> PaperBuildReceiptV71:
    projection_path = paths.attempt_root / "current_build.json"
    if not projection_path.is_file():
        raise PaperDeliveryError("current paper build projection is missing")
    payload = _read_model_dict(projection_path)
    build_id = payload.get("build_id")
    if not isinstance(build_id, str) or not re.fullmatch(
        r"build-[0-9a-f]{20}", build_id
    ):
        raise PaperDeliveryError("current build projection has invalid build_id")
    return _read_model(
        paths.builds_root / build_id / "build_receipt.json",
        PaperBuildReceiptV71,
    )


def _read_model_dict(path: Path) -> dict[str, object]:
    import json

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PaperDeliveryError(f"invalid JSON projection {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PaperDeliveryError(f"JSON projection must be an object: {path}")
    return value


def _expected_build_input_hash(
    receipt: PaperBuildReceiptV71,
    request: PaperAuthoringRequestV71,
) -> str:
    toolchain_input = {
        "environment_hash": receipt.environment_hash,
        "xelatex": {
            "resolved_path": receipt.xelatex_identity.resolved_path,
            "binary_sha256": receipt.xelatex_identity.binary_sha256,
            "version": receipt.xelatex_identity.version,
            "compiler_command": receipt.compiler_command,
        },
        "pdfinfo": {
            "resolved_path": receipt.pdfinfo_identity.resolved_path,
            "binary_sha256": receipt.pdfinfo_identity.binary_sha256,
            "version": receipt.pdfinfo_identity.version,
        },
        "pdftoppm": {
            "resolved_path": receipt.pdftoppm_identity.resolved_path,
            "binary_sha256": receipt.pdftoppm_identity.binary_sha256,
            "version": receipt.pdftoppm_identity.version,
        },
    }
    return sha256_value(
        {
            "template_sha256": receipt.template_sha256,
            "bundle_hash": receipt.bundle_hash,
            "request_hash": request.request_hash,
            "author_request_sha256": receipt.author_request_sha256,
            "writer_packet_sha256": receipt.writer_packet_sha256,
            "metadata_sha256": receipt.metadata_sha256,
            "abstract_sha256": receipt.abstract_sha256,
            "body_sha256": receipt.body_sha256,
            "claim_ledger_sha256": receipt.claim_ledger_sha256,
            "citation_manifest_sha256": receipt.citation_manifest_sha256,
            "figure_manifest_sha256": receipt.figure_manifest_sha256,
            "table_manifest_sha256": receipt.table_manifest_sha256,
            "writer_transport_receipt_sha256": (
                receipt.writer_transport_receipt_sha256
            ),
            "writer_role_request_sha256": receipt.writer_role_request_sha256,
            "writer_output_sha256": receipt.writer_output_sha256,
            "content_audit_hash": receipt.content_audit_hash,
            "semantic_review_sha256": receipt.semantic_review_sha256,
            "toolchain": toolchain_input,
        }
    )


def _replay_build_and_pages(
    root: Path,
    paths: PaperAttemptPathsV71,
    build_id: str,
    receipt: PaperBuildReceiptV71,
    *,
    timeout_seconds: float = 180.0,
) -> list[str]:
    mismatches: set[str] = set()
    build_root = paths.builds_root / build_id
    environment = _minimal_environment()
    expected_compiler = [
        receipt.xelatex_identity.resolved_path,
        "-no-shell-escape",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        "main.tex",
    ]
    if "miktex" in receipt.compiler_version.lower():
        expected_compiler.insert(1, "-disable-installer")
    if receipt.compiler_command != expected_compiler:
        return ["compiler command differs from the V7.1 allowlisted command"]

    version_commands = (
        (
            receipt.xelatex_identity,
            [receipt.xelatex_identity.resolved_path, "--version"],
            "xetex",
        ),
        (
            receipt.pdfinfo_identity,
            [receipt.pdfinfo_identity.resolved_path, "-v"],
            "pdfinfo",
        ),
        (
            receipt.pdftoppm_identity,
            [receipt.pdftoppm_identity.resolved_path, "-v"],
            "pdftoppm",
        ),
    )
    for identity, command, marker in version_commands:
        try:
            current_version = _tool_version(
                command,
                cwd=build_root,
                timeout_seconds=min(timeout_seconds, 20),
                environment=environment,
            )
        except PaperDeliveryError as exc:
            mismatches.add(str(exc))
            continue
        if current_version != identity.version:
            mismatches.add(f"build tool version changed: {identity.tool}")
        if marker not in current_version.lower():
            mismatches.add(
                f"build tool version is not recognizable as {identity.tool}"
            )

    expected_page_count = len(receipt.page_images)
    pdfinfo_run = [receipt.pdfinfo_identity.resolved_path, "main.pdf"]
    render_commands = [
        [
            receipt.pdftoppm_identity.resolved_path,
            "-png",
            "-r",
            "144",
            "-f",
            str(page),
            "-l",
            str(page),
            "-singlefile",
            "main.pdf",
            str(Path("pages") / f"page-{page:03d}"),
        ]
        for page in range(1, expected_page_count + 1)
    ]
    if receipt.xelatex_identity.argv_hash != sha256_value(
        [
            [receipt.xelatex_identity.resolved_path, "--version"],
            expected_compiler,
            expected_compiler,
        ]
    ):
        mismatches.add("XeLaTeX argv hash is inconsistent")
    if receipt.pdfinfo_identity.argv_hash != sha256_value(
        [[receipt.pdfinfo_identity.resolved_path, "-v"], pdfinfo_run]
    ):
        mismatches.add("pdfinfo argv hash is inconsistent")
    if receipt.pdftoppm_identity.argv_hash != sha256_value(
        [
            [receipt.pdftoppm_identity.resolved_path, "-v"],
            *render_commands,
        ]
    ):
        mismatches.add("pdftoppm argv hash is inconsistent")

    with tempfile.TemporaryDirectory(
        prefix=".verify-", dir=paths.builds_root
    ) as raw:
        replay_root = Path(raw)
        shutil.copyfile(build_root / "main.tex", replay_root / "main.tex")
        assets = build_root / "assets"
        if assets.is_dir():
            shutil.copytree(assets, replay_root / "assets")
        for pass_index in (1, 2):
            try:
                completed = _run(
                    expected_compiler,
                    cwd=replay_root,
                    timeout_seconds=timeout_seconds,
                    environment=environment,
                )
            except PaperDeliveryError as exc:
                mismatches.add(f"XeLaTeX replay failed: {exc}")
                return sorted(mismatches)
            if completed.returncode != 0:
                mismatches.add(
                    f"XeLaTeX replay pass {pass_index} returned "
                    f"{completed.returncode}"
                )
                return sorted(mismatches)
        replay_pdf = replay_root / "main.pdf"
        if (
            not replay_pdf.is_file()
            or not replay_pdf.read_bytes().startswith(b"%PDF-")
        ):
            mismatches.add("XeLaTeX replay did not produce a valid PDF")
            return sorted(mismatches)
        if _sha256_file(replay_pdf) != receipt.pdf_sha256:
            mismatches.add("XeLaTeX replay PDF hash differs from receipt")

        try:
            info = _run(
                pdfinfo_run,
                cwd=replay_root,
                timeout_seconds=min(timeout_seconds, 30),
                environment=environment,
            )
        except PaperDeliveryError as exc:
            mismatches.add(f"pdfinfo replay failed: {exc}")
            return sorted(mismatches)
        match = re.search(
            r"^Pages:\s+([1-9][0-9]*)\s*$",
            info.stdout.decode("utf-8", errors="replace"),
            flags=re.MULTILINE,
        )
        if info.returncode != 0 or not match:
            mismatches.add("pdfinfo replay did not validate the rebuilt PDF")
            return sorted(mismatches)
        if int(match.group(1)) != expected_page_count:
            mismatches.add("rebuilt PDF page count differs from receipt")
            return sorted(mismatches)

        replay_pages = replay_root / "pages"
        replay_pages.mkdir()
        for page, command in enumerate(render_commands, 1):
            try:
                rendered = _run(
                    command,
                    cwd=replay_root,
                    timeout_seconds=timeout_seconds,
                    environment=environment,
                )
            except PaperDeliveryError as exc:
                mismatches.add(f"page replay failed: {exc}")
                continue
            page_path = replay_pages / f"page-{page:03d}.png"
            expected_relative = _relative_to_workspace(
                root, build_root / "pages" / page_path.name
            )
            expected_hash = receipt.page_images.get(expected_relative)
            if (
                rendered.returncode != 0
                or not page_path.is_file()
                or not page_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            ):
                mismatches.add(f"page replay did not produce PNG page {page}")
            elif expected_hash is None:
                mismatches.add(f"receipt is missing expected page {page}")
            elif _sha256_file(page_path) != expected_hash:
                mismatches.add(f"replayed page hash differs for page {page}")
    return sorted(mismatches)


def verify_paper_build_v71(
    workspace_root: str | Path,
    *,
    build_id: str | None = None,
) -> PaperBuildVerificationResultV71:
    root = Path(workspace_root).resolve()
    mismatches: set[str] = set()
    try:
        paths = current_paper_attempt_v71(root)
        explicit_build_id = build_id is not None
        projection: dict[str, object] | None = None
        if build_id is None:
            projection = _read_model_dict(paths.attempt_root / "current_build.json")
            value = projection.get("build_id")
            if not isinstance(value, str):
                raise PaperDeliveryError("current build_id is missing")
            build_id = value
        if not re.fullmatch(r"build-[0-9a-f]{20}", build_id):
            raise PaperDeliveryError("invalid build_id")
        build_root = paths.builds_root / build_id
        receipt = _read_model(
            build_root / "build_receipt.json", PaperBuildReceiptV71
        )
        request = _read_model(
            paths.attempt_root / "author_request.json",
            PaperAuthoringRequestV71,
        )
        expected_input_hash = _expected_build_input_hash(receipt, request)
        if build_id != f"build-{expected_input_hash[:20]}":
            mismatches.add("build_id differs from recomputed build input hash")
        if (
            not explicit_build_id
            and projection is not None
            and projection.get("build_hash") != receipt.build_hash
        ):
            mismatches.add("current build projection hash differs from receipt")
        if receipt.environment_hash != sha256_value(_minimal_environment()):
            mismatches.add("paper build environment differs from receipt")
        for identity in (
            receipt.xelatex_identity,
            receipt.pdfinfo_identity,
            receipt.pdftoppm_identity,
        ):
            executable = Path(identity.resolved_path)
            if not executable.is_file():
                mismatches.add(
                    f"missing build tool executable: {identity.tool}"
                )
            elif _sha256_file(executable) != identity.binary_sha256:
                mismatches.add(
                    f"build tool executable hash mismatch: {identity.tool}"
                )
        checks = {
            _TEMPLATE_PATH: receipt.template_sha256,
            paths.attempt_root
            / "author_request.json": receipt.author_request_sha256,
            paths.attempt_root
            / "writer_packet.json": receipt.writer_packet_sha256,
            paths.attempt_root / "metadata.json": receipt.metadata_sha256,
            paths.source_root / "abstract.tex": receipt.abstract_sha256,
            paths.source_root / "body.tex": receipt.body_sha256,
            paths.manifests_root
            / "claim_ledger.json": receipt.claim_ledger_sha256,
            paths.manifests_root
            / "citations.json": receipt.citation_manifest_sha256,
            paths.manifests_root
            / "figures.json": receipt.figure_manifest_sha256,
            paths.manifests_root
            / "tables.json": receipt.table_manifest_sha256,
            paths.attempt_root
            / "writer_transport_receipt.json": (
                receipt.writer_transport_receipt_sha256
            ),
            paths.attempt_root
            / "writer_role_request.json": receipt.writer_role_request_sha256,
            paths.attempt_root
            / "writer_output.json": receipt.writer_output_sha256,
            paths.reviews_root
            / "semantic_review.json": receipt.semantic_review_sha256,
            _safe_relative_path(
                root, receipt.generated_tex_path
            ): receipt.generated_tex_sha256,
            _safe_relative_path(root, receipt.pdf_path): receipt.pdf_sha256,
            build_root / "compiler.log": receipt.compiler_log_sha256,
        }
        for path, expected in checks.items():
            if not path.is_file():
                mismatches.add(f"missing build-bound artifact: {path.name}")
            elif _sha256_file(path) != expected:
                mismatches.add(f"hash mismatch for build-bound artifact: {path.name}")
        generated_pdf = _safe_relative_path(root, receipt.pdf_path)
        if (
            generated_pdf.is_file()
            and not generated_pdf.read_bytes().startswith(b"%PDF-")
        ):
            mismatches.add("build-bound PDF does not have a PDF signature")
        audit = _read_model(
            paths.reviews_root / "content_audit.json", PaperContentAuditV71
        )
        if audit.audit_hash != receipt.content_audit_hash:
            mismatches.add("build receipt binds another content audit")
        semantic = _read_model(
            paths.reviews_root / "semantic_review.json",
            PaperSemanticReviewV71,
        )
        semantic_transport = (
            paths.reviews_root / "semantic_transport_receipt.json"
        )
        if (
            not semantic_transport.is_file()
            or _sha256_file(semantic_transport)
            != semantic.reviewer_transport_receipt_sha256
        ):
            mismatches.add("semantic reviewer transport receipt mismatch")
        bundle = _read_model(
            paths.attempt_root / "evidence_bundle.json",
            PaperEvidenceBundleV71,
        )
        if bundle.bundle_hash != receipt.bundle_hash:
            mismatches.add("build receipt binds another evidence bundle")
        if receipt.layout_lint:
            mismatches.add("build receipt contains unresolved layout lint")
        for relative, expected in receipt.page_images.items():
            page = _safe_relative_path(root, relative)
            if not page.is_file():
                mismatches.add(f"missing rendered page: {relative}")
            elif _sha256_file(page) != expected:
                mismatches.add(f"rendered page hash mismatch: {relative}")
            elif not page.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
                mismatches.add(f"rendered page is not a PNG: {relative}")
        if not mismatches:
            mismatches.update(
                _replay_build_and_pages(
                    root,
                    paths,
                    build_id,
                    receipt,
                )
            )
        return PaperBuildVerificationResultV71(
            ok=not mismatches,
            mismatches=sorted(mismatches),
            build_hash=receipt.build_hash,
        )
    except (OSError, PaperDeliveryError, ValueError) as exc:
        mismatches.add(str(exc))
        return PaperBuildVerificationResultV71(
            ok=False,
            mismatches=sorted(mismatches),
        )


__all__ = [
    "PaperBuildVerificationResultV71",
    "build_paper_v71",
    "load_current_build_v71",
    "verify_paper_build_v71",
]
