"""Build a paper only from a template and machine-readable result values.

This module establishes build provenance and current-artifact consistency.  A
successful receipt is deliberately *not* evidence that the model, result
values, prose, or decisions are scientifically correct.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Annotated, Literal, Sequence

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel


TEMPLATE_RELATIVE_PATH = "paper/main.template.tex"
RESULTS_RELATIVE_PATH = "results/values.json"
GENERATED_TEX_RELATIVE_PATH = "paper/build/main.tex"
PDF_RELATIVE_PATH = "paper/build/main.pdf"
RECEIPT_RELATIVE_PATH = "paper/build/build_receipt.json"
EVIDENCE_SCOPE = "paper_build_and_artifact_consistency_only"

_SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_RESULT_PLACEHOLDER = re.compile(
    r"\{\{result\.("
    r"[A-Za-z_][A-Za-z0-9_-]*"
    r"(?:\.[A-Za-z_][A-Za-z0-9_-]*)*"
    r")\}\}"
)
_ANY_PLACEHOLDER = re.compile(
    r"\{\{\s*[A-Za-z_][A-Za-z0-9_.-]*\s*\}\}"
)
_NUMERIC_LITERAL = re.compile(
    r"(?<![A-Za-z_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
_INCLUDE_GRAPHICS = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{")


class PaperBuildError(RuntimeError):
    """Raised when rendering, compilation, or consistency verification fails."""


class PaperBuildReceipt(StrictModel):
    """Content-bound evidence for one successful clean PDF build."""

    schema_version: Literal["5.0"] = "5.0"
    artifact_kind: Literal["paper_build_receipt"] = "paper_build_receipt"
    evidence_scope: Literal[
        "paper_build_and_artifact_consistency_only"
    ] = EVIDENCE_SCOPE
    template_path: Literal["paper/main.template.tex"] = TEMPLATE_RELATIVE_PATH
    results_path: Literal["results/values.json"] = RESULTS_RELATIVE_PATH
    generated_tex_path: Literal["paper/build/main.tex"] = (
        GENERATED_TEX_RELATIVE_PATH
    )
    pdf_path: Literal["paper/build/main.pdf"] = PDF_RELATIVE_PATH
    template_sha256: _SHA256
    results_sha256: _SHA256
    generated_tex_sha256: _SHA256
    pdf_sha256: _SHA256
    pdf_size_bytes: Annotated[int, Field(gt=0)]
    command: Annotated[list[str], Field(min_length=4)]
    returncode: Literal[0] = 0
    compiler_stdout_sha256: _SHA256
    compiler_stderr_sha256: _SHA256
    scientific_correctness_established: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    receipt_hash: _SHA256

    @model_validator(mode="after")
    def validate_receipt(self) -> "PaperBuildReceipt":
        if "-halt-on-error" not in self.command:
            raise ValueError("paper build command must contain -halt-on-error")
        expected = sha256_value(
            self.model_dump(mode="json", exclude={"receipt_hash"})
        )
        if self.receipt_hash != expected:
            raise ValueError("receipt_hash does not match receipt content")
        return self


class PaperBuildVerification(StrictModel):
    """Current consistency result; it carries no scientific qualification."""

    ok: bool
    mismatches: list[str]
    receipt_hash: _SHA256 | None = None
    evidence_scope: Literal[
        "paper_build_and_artifact_consistency_only"
    ] = EVIDENCE_SCOPE
    scientific_correctness_established: Literal[False] = False
    real_world_action_authorized: Literal[False] = False


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _workspace_root(path: str | os.PathLike[str]) -> Path:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise PaperBuildError(f"workspace root is not a directory: {root}")
    return root


def _workspace_file(root: Path, relative_path: str) -> Path:
    candidate = root.joinpath(*relative_path.split("/"))
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PaperBuildError(
            f"workspace path escapes root: {relative_path}"
        ) from exc
    return candidate


def _read_workspace_file(root: Path, relative_path: str) -> bytes:
    path = _workspace_file(root, relative_path)
    if path.is_symlink():
        raise PaperBuildError(
            f"workspace evidence file may not be a symlink: {relative_path}"
        )
    if not path.is_file():
        raise PaperBuildError(
            f"required workspace file is missing: {relative_path}"
        )
    return path.read_bytes()


def _reject_nonstandard_number(token: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {token}")


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _assert_finite_numbers(value: object, path: str = "results") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PaperBuildError(f"non-finite result value at {path}")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite_numbers(child, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_numbers(child, f"{path}[{index}]")
        return
    raise PaperBuildError(f"unsupported JSON value at {path}")


def _load_results(payload: bytes) -> dict[str, object]:
    try:
        decoded = payload.decode("utf-8")
        value = json.loads(
            decoded,
            parse_constant=_reject_nonstandard_number,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PaperBuildError(f"invalid results/values.json: {exc}") from exc
    if not isinstance(value, dict):
        raise PaperBuildError("results/values.json must contain a JSON object")
    _assert_finite_numbers(value)
    return value


def _lookup_result(values: dict[str, object], dotted_key: str) -> object:
    current: object = values
    for segment in dotted_key.split("."):
        if not isinstance(current, dict) or segment not in current:
            raise KeyError(dotted_key)
        current = current[segment]
    return current


def _format_result(value: object, dotted_key: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PaperBuildError(
            f"result placeholder {dotted_key!r} must resolve to a number"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise PaperBuildError(
            f"result placeholder {dotted_key!r} is not finite"
        )
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _render_bytes(template: bytes, results: bytes) -> bytes:
    try:
        template_text = template.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PaperBuildError(
            "paper/main.template.tex must be UTF-8"
        ) from exc
    values = _load_results(results)
    placeholders = list(_RESULT_PLACEHOLDER.finditer(template_text))
    if not placeholders:
        raise PaperBuildError(
            "paper template must contain at least one {{result.KEY}} placeholder"
        )
    literal_scan = _RESULT_PLACEHOLDER.sub("", template_text)
    hard_coded = _NUMERIC_LITERAL.search(literal_scan)
    if hard_coded is not None:
        raise PaperBuildError(
            "paper template contains a hard-coded numeric literal; "
            "use a {{result.KEY}} placeholder"
        )
    if _INCLUDE_GRAPHICS.search(template_text):
        raise PaperBuildError(
            "paper figures require a generator-bound figure manifest, "
            "which V5.0 does not yet implement"
        )
    missing: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        try:
            value = _lookup_result(values, key)
        except KeyError:
            missing.add(key)
            return match.group(0)
        return _format_result(value, key)

    rendered = _RESULT_PLACEHOLDER.sub(replace, template_text)
    if missing:
        raise PaperBuildError(
            "missing result value(s): " + ", ".join(sorted(missing))
        )
    unresolved = sorted(set(_ANY_PLACEHOLDER.findall(rendered)))
    if unresolved:
        raise PaperBuildError(
            "unresolved template placeholder(s): " + ", ".join(unresolved)
        )
    return rendered.encode("utf-8")


def _input_bytes(root: Path) -> tuple[bytes, bytes]:
    return (
        _read_workspace_file(root, TEMPLATE_RELATIVE_PATH),
        _read_workspace_file(root, RESULTS_RELATIVE_PATH),
    )


def render_paper(workspace_root: str | os.PathLike[str]) -> Path:
    """Inject result numbers and write ``paper/build/main.tex``.

    Rendering does not compile the PDF or establish a valid build receipt.
    """

    root = _workspace_root(workspace_root)
    template, results = _input_bytes(root)
    generated = _render_bytes(template, results)
    output = _workspace_file(root, GENERATED_TEX_RELATIVE_PATH)
    if output.is_symlink():
        raise PaperBuildError("generated paper path may not be a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(generated)
    return output


def _clean_build_directory(build_dir: Path, root: Path) -> None:
    resolved = build_dir.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PaperBuildError("paper build directory escapes workspace") from exc
    if build_dir.is_symlink():
        raise PaperBuildError("paper build directory may not be a symlink")
    if build_dir.exists() and not build_dir.is_dir():
        raise PaperBuildError("paper/build exists but is not a directory")
    build_dir.mkdir(parents=True, exist_ok=True)
    for child in build_dir.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
        else:
            raise PaperBuildError(f"unsupported build artifact: {child.name}")


def _normalize_command(
    command: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
) -> list[str]:
    if isinstance(command, (str, os.PathLike)):
        normalized = [os.fspath(command)]
    else:
        normalized = [os.fspath(part) for part in command]
    if not normalized or any(not part for part in normalized):
        raise PaperBuildError("pdflatex_command must not be empty")
    return normalized


def _sealed_receipt(**payload: object) -> PaperBuildReceipt:
    receipt_hash = sha256_value(payload)
    return PaperBuildReceipt(**payload, receipt_hash=receipt_hash)


def build_paper(
    workspace_root: str | os.PathLike[str],
    *,
    pdflatex_command: (
        str | os.PathLike[str] | Sequence[str | os.PathLike[str]]
    ) = "pdflatex",
    timeout_seconds: float = 120.0,
) -> PaperBuildReceipt:
    """Perform a clean, halt-on-error PDF build and write its receipt."""

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise PaperBuildError("timeout_seconds must be finite and positive")
    root = _workspace_root(workspace_root)
    template, results = _input_bytes(root)
    generated = _render_bytes(template, results)
    template_hash = _sha256_bytes(template)
    results_hash = _sha256_bytes(results)

    build_dir = _workspace_file(root, "paper/build")
    _clean_build_directory(build_dir, root)
    generated_path = _workspace_file(root, GENERATED_TEX_RELATIVE_PATH)
    generated_path.write_bytes(generated)

    command = [
        *_normalize_command(pdflatex_command),
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        "main.tex",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=build_dir,
            check=False,
            capture_output=True,
            timeout=timeout_seconds,
            shell=False,
        )
    except FileNotFoundError as exc:
        raise PaperBuildError(
            f"pdflatex executable was not found: {command[0]}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise PaperBuildError(
            f"pdflatex timed out after {timeout_seconds:g} seconds"
        ) from exc
    if completed.returncode != 0:
        raise PaperBuildError(
            "pdflatex failed with return code "
            f"{completed.returncode}; stdout_sha256="
            f"{_sha256_bytes(completed.stdout)}; stderr_sha256="
            f"{_sha256_bytes(completed.stderr)}"
        )

    pdf_path = _workspace_file(root, PDF_RELATIVE_PATH)
    if pdf_path.is_symlink() or not pdf_path.is_file():
        raise PaperBuildError("pdflatex returned success without paper/build/main.pdf")
    pdf = pdf_path.read_bytes()
    if not pdf.startswith(b"%PDF-"):
        raise PaperBuildError("paper/build/main.pdf is not a PDF file")

    current_template, current_results = _input_bytes(root)
    if _sha256_bytes(current_template) != template_hash:
        raise PaperBuildError("paper template changed during the build")
    if _sha256_bytes(current_results) != results_hash:
        raise PaperBuildError("result values changed during the build")
    if generated_path.read_bytes() != generated:
        raise PaperBuildError("generated TeX changed during the build")

    payload: dict[str, object] = {
        "schema_version": "5.0",
        "artifact_kind": "paper_build_receipt",
        "evidence_scope": EVIDENCE_SCOPE,
        "template_path": TEMPLATE_RELATIVE_PATH,
        "results_path": RESULTS_RELATIVE_PATH,
        "generated_tex_path": GENERATED_TEX_RELATIVE_PATH,
        "pdf_path": PDF_RELATIVE_PATH,
        "template_sha256": template_hash,
        "results_sha256": results_hash,
        "generated_tex_sha256": _sha256_bytes(generated),
        "pdf_sha256": _sha256_bytes(pdf),
        "pdf_size_bytes": len(pdf),
        "command": command,
        "returncode": completed.returncode,
        "compiler_stdout_sha256": _sha256_bytes(completed.stdout),
        "compiler_stderr_sha256": _sha256_bytes(completed.stderr),
        "scientific_correctness_established": False,
        "real_world_action_authorized": False,
    }
    receipt = _sealed_receipt(**payload)
    receipt_path = _workspace_file(root, RECEIPT_RELATIVE_PATH)
    receipt_path.write_text(
        json.dumps(
            receipt.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return receipt


def verify_paper_build(
    workspace_root: str | os.PathLike[str],
    *,
    receipt_path: str = RECEIPT_RELATIVE_PATH,
) -> PaperBuildVerification:
    """Check that current sources and products still match a sealed receipt."""

    try:
        root = _workspace_root(workspace_root)
    except PaperBuildError as exc:
        return PaperBuildVerification(ok=False, mismatches=[str(exc)])

    try:
        raw_receipt = _read_workspace_file(root, receipt_path)
        parsed = json.loads(
            raw_receipt.decode("utf-8"),
            parse_constant=_reject_nonstandard_number,
            object_pairs_hook=_reject_duplicate_keys,
        )
        receipt = PaperBuildReceipt.model_validate(parsed)
    except (PaperBuildError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return PaperBuildVerification(
            ok=False,
            mismatches=[f"invalid build receipt: {exc}"],
        )

    mismatches: list[str] = []
    expected_hashes = (
        (receipt.template_path, receipt.template_sha256, "template"),
        (receipt.results_path, receipt.results_sha256, "results"),
        (
            receipt.generated_tex_path,
            receipt.generated_tex_sha256,
            "generated_tex",
        ),
        (receipt.pdf_path, receipt.pdf_sha256, "pdf"),
    )
    for relative_path, expected, label in expected_hashes:
        try:
            actual = _sha256_bytes(_read_workspace_file(root, relative_path))
        except PaperBuildError as exc:
            mismatches.append(f"{label}: {exc}")
            continue
        if actual != expected:
            mismatches.append(
                f"{label}_sha256: expected {expected}, got {actual}"
            )

    try:
        template, results = _input_bytes(root)
        expected_generated = _render_bytes(template, results)
        actual_generated = _read_workspace_file(
            root, receipt.generated_tex_path
        )
        if actual_generated != expected_generated:
            mismatches.append(
                "generated_tex is not the rendering of current template and results"
            )
    except PaperBuildError as exc:
        mismatches.append(f"render_consistency: {exc}")

    try:
        pdf = _read_workspace_file(root, receipt.pdf_path)
        if not pdf.startswith(b"%PDF-"):
            mismatches.append("pdf does not have a PDF header")
        if len(pdf) != receipt.pdf_size_bytes:
            mismatches.append(
                "pdf_size_bytes: expected "
                f"{receipt.pdf_size_bytes}, got {len(pdf)}"
            )
    except PaperBuildError:
        # A missing PDF is already reported by the hash loop.
        pass

    return PaperBuildVerification(
        ok=not mismatches,
        mismatches=mismatches,
        receipt_hash=receipt.receipt_hash,
    )


def assert_paper_build_consistent(
    workspace_root: str | os.PathLike[str],
    *,
    receipt_path: str = RECEIPT_RELATIVE_PATH,
) -> PaperBuildReceipt:
    """Return the receipt or raise when current paper evidence is inconsistent."""

    verification = verify_paper_build(
        workspace_root, receipt_path=receipt_path
    )
    if not verification.ok:
        raise PaperBuildError("; ".join(verification.mismatches))
    root = _workspace_root(workspace_root)
    payload = json.loads(
        _read_workspace_file(root, receipt_path).decode("utf-8"),
        parse_constant=_reject_nonstandard_number,
        object_pairs_hook=_reject_duplicate_keys,
    )
    return PaperBuildReceipt.model_validate(payload)
