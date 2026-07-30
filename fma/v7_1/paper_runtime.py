"""Evidence-bound runtime for the post-S6 V7.1 paper projection.

The runtime never mutates S0--S6 authority.  It reads one authenticated current
S6 lineage and creates a content-addressed publication attempt below
``delivery/paper/v71``.  A polished paper remains a publication artifact, not a
scientific gate or an authorization for consequential action.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping
from uuid import uuid4

from fma._file_lock import exclusive_file_lock
from fma.hashing import sha256_value
from fma.v5.stage_workspace import STAGES, StageWorkspaceV50
from fma.v5.workspace_schemas import GateCertificateV50
from fma.v5_1.codex_stage_driver import RoleProcessReceiptV51, RoleRequestV51

from .paper_schemas import (
    CitationManifestV71,
    FigureManifestV71,
    PaperAuthoringRequestV71,
    PaperClaimLedgerV71,
    PaperContentAuditV71,
    PaperCurrentProjectionV71,
    PaperDeliveryReceiptV71,
    PaperDeliveryVerificationV71,
    PaperEvidenceBundleV71,
    PaperEvidenceItemV71,
    PaperLayoutReviewV71,
    PaperMetadataV71,
    PaperNumericTokenV71,
    PaperSemanticReviewV71,
    PaperWriterPacketV71,
    TableManifestV71,
)


PAPER_ROOT = PurePosixPath("delivery/paper/v71")
CURRENT_PATH = PAPER_ROOT / "current.json"
_TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_MAX_WRITER_PACKET_BYTES = 2 * 1024 * 1024
_MAX_TEXT_FILE_BYTES = 256 * 1024
_MAX_NUMERIC_TOKENS = 5_000
_FORBIDDEN_TEX = re.compile(
    r"\\(?:"
    r"documentclass|usepackage|input|include|includegraphics|"
    r"write18|write|openin|openout|read|catcode|def|edef|gdef|xdef|"
    r"newcommand|renewcommand|providecommand|csname|special|"
    r"immediate|directlua|endinput"
    r")\b",
    flags=re.IGNORECASE,
)
_TEX_COMMAND = re.compile(r"\\([A-Za-z@]+)")
_TEX_ENVIRONMENT = re.compile(r"\\(?:begin|end)\{([^{}]+)\}")
_ALLOWED_TEX_COMMANDS = {
    "FMAClaim",
    "FMACite",
    "FMAFigure",
    "FMALimitations",
    "FMAQualifier",
    "FMASection",
    "FMASubsection",
    "FMATable",
    "FMAValue",
    "Big",
    "Bigg",
    "Delta",
    "Gamma",
    "Lambda",
    "Omega",
    "Phi",
    "Pi",
    "Psi",
    "Sigma",
    "Theta",
    "Xi",
    "alpha",
    "approx",
    "argmax",
    "argmin",
    "array",
    "bar",
    "begin",
    "beta",
    "big",
    "bigg",
    "bm",
    "cap",
    "cdot",
    "cdots",
    "chi",
    "cup",
    "delta",
    "dfrac",
    "div",
    "ell",
    "emph",
    "end",
    "epsilon",
    "eqref",
    "eta",
    "exp",
    "footnote",
    "frac",
    "gamma",
    "ge",
    "geq",
    "hat",
    "iint",
    "iiint",
    "in",
    "infty",
    "int",
    "item",
    "kappa",
    "label",
    "lambda",
    "ldots",
    "le",
    "left",
    "leq",
    "lim",
    "ln",
    "log",
    "mapsto",
    "mathbb",
    "mathbf",
    "mathcal",
    "mathit",
    "mathrm",
    "mathsf",
    "mathtt",
    "max",
    "min",
    "mp",
    "mu",
    "nabla",
    "neq",
    "notin",
    "nu",
    "omega",
    "operatorname",
    "overline",
    "pageref",
    "partial",
    "phi",
    "pi",
    "pm",
    "prod",
    "propto",
    "psi",
    "qquad",
    "quad",
    "ref",
    "rho",
    "right",
    "sigma",
    "sim",
    "sqrt",
    "subset",
    "subseteq",
    "sum",
    "tau",
    "tfrac",
    "text",
    "textbf",
    "textit",
    "textnormal",
    "textrm",
    "texttt",
    "theta",
    "times",
    "to",
    "underline",
    "upsilon",
    "varepsilon",
    "varphi",
    "varrho",
    "varsigma",
    "vartheta",
    "vdots",
    "vec",
    "widehat",
    "xi",
    "zeta",
}
_ALLOWED_TEX_ENVIRONMENTS = {
    "FMALimitations",
    "align",
    "align*",
    "bmatrix",
    "cases",
    "description",
    "enumerate",
    "equation",
    "equation*",
    "gather",
    "gather*",
    "itemize",
    "matrix",
    "multline",
    "multline*",
    "pmatrix",
    "smallmatrix",
}
_RAW_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_.:-])"
    r"[-+]?(?:\d{2,}(?:\.\d*)?|\d*\.\d+)(?:[eE][-+]?\d+)?%?"
    r"(?![A-Za-z0-9_.:-])"
)
_FORBIDDEN_WIDENING = re.compile(
    r"\b(?:(?:establish(?:es|ed)?|prove(?:s|d)?|demonstrat(?:e|es|ed)|"
    r"confirm(?:s|ed)?)\s+(?:an?\s+)?causal(?:\s+relationship|\s+effect|"
    r"\s+mechanism)?|causal(?:ly)?\s+(?:determines?|drives?|causes?)|"
    r"globally optimal|proves? the mechanism|guarantee(?:s|d)?\s+"
    r"(?:valid\s+|safe\s+)?(?:extrapolation|generalization|all\s+deployments?))"
    r"\b|(?:证明|证实|确认).{0,12}因果|因果(?:关系|效应|机制)"
    r"(?:成立|已验证|已证实)|全局最优|机制真实性|保证(?:外推|泛化|所有部署)",
    flags=re.IGNORECASE,
)
_MACROS: dict[str, re.Pattern[str]] = {
    "claim": re.compile(r"\\FMAClaim\{([A-Za-z][A-Za-z0-9_.:-]{0,127})\}"),
    "value": re.compile(r"\\FMAValue\{([A-Za-z][A-Za-z0-9_.:-]{0,127})\}"),
    "cite": re.compile(r"\\FMACite\{([A-Za-z][A-Za-z0-9_.:-]{0,127})\}"),
    "figure": re.compile(r"\\FMAFigure\{([A-Za-z][A-Za-z0-9_.:-]{0,127})\}"),
    "table": re.compile(r"\\FMATable\{([A-Za-z][A-Za-z0-9_.:-]{0,127})\}"),
}
_SECTION = re.compile(
    r"\\FMASection\{([A-Za-z][A-Za-z0-9_.:-]{0,127})\}\{[^{}]+\}"
)
_REQUIRED_SECTIONS = {
    "academic_article": {
        "introduction",
        "methods",
        "results",
        "discussion",
        "conclusion",
    },
    "modeling_contest": {
        "problem",
        "assumptions",
        "model",
        "solution",
        "validation",
        "sensitivity",
        "limitations",
        "conclusion",
    },
    "technical_report": {
        "executive_summary",
        "problem",
        "method",
        "results",
        "decision",
        "limitations",
    },
}


class PaperDeliveryError(RuntimeError):
    """A V7.1 publication operation failed closed."""


@dataclass(frozen=True)
class PaperAttemptPathsV71:
    workspace_root: Path
    attempt_id: str
    attempt_root: Path

    @property
    def source_root(self) -> Path:
        return self.attempt_root / "source"

    @property
    def manifests_root(self) -> Path:
        return self.attempt_root / "manifests"

    @property
    def reviews_root(self) -> Path:
        return self.attempt_root / "reviews"

    @property
    def builds_root(self) -> Path:
        return self.attempt_root / "builds"


def assert_paper_attempt_open_v71(paths: PaperAttemptPathsV71) -> None:
    """Reject every mutation after a delivery receipt has been minted."""

    if (paths.attempt_root / "delivery_receipt.json").exists():
        raise PaperDeliveryError(
            "finalized paper attempt is immutable; verify it or create a "
            "different authoring request"
        )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
                default=str,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def paper_writer_lock_v71(
    workspace_root: str | Path, *, timeout_seconds: float = 30.0
) -> Iterator[None]:
    """Serialize every V7.1 mutation for one modelling workspace."""

    root = Path(workspace_root).resolve()
    lock_path = _safe_relative_path(
        root, (PAPER_ROOT / ".writer.lock").as_posix()
    )
    with exclusive_file_lock(lock_path, timeout_seconds=timeout_seconds):
        yield


def _read_json(path: Path) -> object:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PaperDeliveryError(f"invalid JSON artifact {path}: {exc}") from exc


def _read_model(path: Path, model_type: type[Any]) -> Any:
    try:
        return model_type.model_validate(_read_json(path))
    except (ValueError, TypeError) as exc:
        raise PaperDeliveryError(
            f"invalid {model_type.__name__} at {path}: {exc}"
        ) from exc


def _safe_relative_path(root: Path, relative_path: str) -> Path:
    raw = PurePosixPath(relative_path)
    if (
        not relative_path
        or "\\" in relative_path
        or raw.is_absolute()
        or any(part in {"", ".", ".."} for part in raw.parts)
    ):
        raise PaperDeliveryError(f"unsafe relative path: {relative_path}")
    candidate = root.joinpath(*raw.parts)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise PaperDeliveryError(
            f"path escapes workspace: {relative_path}"
        ) from exc
    current = root
    for part in raw.parts:
        current = current / part
        if current.is_symlink():
            raise PaperDeliveryError(
                f"publication inputs may not use symlinks: {relative_path}"
            )
    return candidate


def _current_certificates(
    workspace: StageWorkspaceV50,
) -> dict[str, GateCertificateV50]:
    certificates: dict[str, GateCertificateV50] = {}
    for stage in STAGES:
        gate_hash = workspace.current_gate(stage)
        if gate_hash is None:
            raise PaperDeliveryError(
                f"paper delivery requires a current authenticated {stage} gate"
            )
        certificate = workspace._certificate_for_current_node(stage)
        if (
            certificate is None
            or certificate.certificate_hash != gate_hash
            or not workspace.verify_certificate(certificate)
        ):
            raise PaperDeliveryError(
                f"current {stage} certificate could not be authenticated"
            )
        certificates[stage] = certificate
    return certificates


def _evidence_kind(relative_path: str) -> str:
    prefix = relative_path.split("/", 1)[0]
    return {
        "problem": "problem",
        "data": "data",
        "docs": "model",
        "src": "code",
        "checks": "validation",
        "results": "result",
        "predictions": "result",
        "paper": "paper",
    }.get(prefix, "other")


def _json_pointer(segment: str) -> str:
    return segment.replace("~", "~0").replace("/", "~1")


def _numeric_values(
    value: object, pointer: str = ""
) -> Iterable[tuple[str, int | float]]:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        yield pointer or "/", value
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PaperDeliveryError(
                f"non-finite numeric evidence at {pointer or '/'}"
            )
        yield pointer or "/", value
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            yield from _numeric_values(child, f"{pointer}/{index}")
        return
    if isinstance(value, dict):
        for key in sorted(value):
            child_pointer = f"{pointer}/{_json_pointer(str(key))}"
            yield from _numeric_values(value[key], child_pointer)


def _display_number(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    return format(value, ".12g")


def build_evidence_bundle_v71(
    workspace: StageWorkspaceV50,
    *,
    requested_model: str = "gpt-5.6-sol",
) -> PaperEvidenceBundleV71:
    """Freeze a read-only paper projection over the exact current S0--S6 chain."""

    certificates = _current_certificates(workspace)
    evidence_items: list[PaperEvidenceItemV71] = []
    numeric_tokens: list[PaperNumericTokenV71] = []
    for stage in STAGES:
        certificate = certificates[stage]
        manifest_hash = str(certificate.manifest.manifest_hash)
        gate_hash = str(certificate.certificate_hash)
        for binding in certificate.manifest.files:
            evidence_id = (
                f"E.{stage.lower()}."
                f"{hashlib.sha256(binding.relative_path.encode('utf-8')).hexdigest()[:16]}"
            )
            evidence_items.append(
                PaperEvidenceItemV71(
                    evidence_id=evidence_id,
                    stage=stage,
                    relative_path=binding.relative_path,
                    sha256=binding.sha256,
                    size_bytes=binding.size_bytes,
                    manifest_hash=manifest_hash,
                    gate_hash=gate_hash,
                    kind=_evidence_kind(binding.relative_path),
                )
            )
            path = _safe_relative_path(workspace.root, binding.relative_path)
            if path.suffix.lower() != ".json":
                continue
            payload = path.read_bytes()
            if _sha256_bytes(payload) != binding.sha256:
                raise PaperDeliveryError(
                    f"current evidence changed while bundling: {binding.relative_path}"
                )
            try:
                decoded = json.loads(
                    payload.decode("utf-8"),
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"non-finite JSON constant: {value}")
                    ),
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise PaperDeliveryError(
                    f"invalid JSON evidence {binding.relative_path}: {exc}"
                ) from exc
            for pointer, number in _numeric_values(decoded):
                token_id = (
                    f"N.{stage.lower()}."
                    f"{hashlib.sha256((binding.relative_path + chr(0) + pointer).encode('utf-8')).hexdigest()[:20]}"
                )
                numeric_tokens.append(
                    PaperNumericTokenV71(
                        token_id=token_id,
                        evidence_id=evidence_id,
                        json_pointer=pointer,
                        value=number,
                        display_value=_display_number(number),
                    )
                )
                if len(numeric_tokens) > _MAX_NUMERIC_TOKENS:
                    raise PaperDeliveryError(
                        "paper evidence exceeds the numeric-token budget"
                    )
    evidence_items.sort(key=lambda item: item.evidence_id)
    numeric_tokens.sort(key=lambda item: item.token_id)
    return PaperEvidenceBundleV71.seal(
        workspace_id=workspace.spec.workspace_id,
        workspace_spec_hash=workspace.spec.spec_hash,
        objective=workspace.spec.objective,
        s6_gate_hash=str(certificates["S6"].certificate_hash),
        current_gate_hashes={
            stage: str(certificates[stage].certificate_hash)
            for stage in STAGES
        },
        evidence_items=evidence_items,
        numeric_tokens=numeric_tokens,
        allowed_claim_types=sorted(
            [
                "comparison",
                "decision",
                "limitation",
                "method",
                "model_structure",
                "problem",
                "quantitative",
                "robustness",
            ]
        ),
        forbidden_claim_types=sorted(
            [
                "causal",
                "global_optimality",
                "mechanistic_truth",
                "unsupported_extrapolation",
            ]
        ),
        requested_model=requested_model,
    )


def _writer_packet(
    root: Path,
    bundle: PaperEvidenceBundleV71,
    request: PaperAuthoringRequestV71,
) -> PaperWriterPacketV71:
    included: list[dict[str, object]] = []
    omitted: list[dict[str, str]] = []
    total = 0
    stage_rank = {stage: index for index, stage in enumerate(STAGES)}

    def priority(item: PaperEvidenceItemV71) -> tuple[int, int, str]:
        path = item.relative_path
        if path == "results/values.json":
            rank = 0
        elif path == "results/decision_dossier.json":
            rank = 1
        elif path in {
            "results/verification_summary.json",
            "results/uq_summary.json",
            "results/index.json",
        }:
            rank = 2
        elif path in {
            "docs/model_spec.json",
            "docs/validation_plan.json",
            "docs/assumptions.json",
            "docs/symbols.json",
        }:
            rank = 3
        elif path.startswith("problem/"):
            rank = 4
        elif path.startswith(("data/", "src/", "checks/")):
            rank = 5
        else:
            rank = 6
        return rank, -stage_rank[item.stage], item.evidence_id

    seen_files: set[tuple[str, str]] = set()
    selected_evidence_ids: set[str] = set()
    for item in sorted(bundle.evidence_items, key=priority):
        if item.kind == "paper" or item.relative_path.startswith("paper/"):
            omitted.append(
                {
                    "evidence_id": item.evidence_id,
                    "reason": "upstream_publication_projection",
                }
            )
            continue
        identity = (item.relative_path, item.sha256)
        if identity in seen_files:
            omitted.append(
                {
                    "evidence_id": item.evidence_id,
                    "reason": "duplicate_projection",
                }
            )
            continue
        seen_files.add(identity)
        path = _safe_relative_path(root, item.relative_path)
        if path.suffix.lower() not in _TEXT_SUFFIXES:
            omitted.append(
                {"evidence_id": item.evidence_id, "reason": "non_text_artifact"}
            )
            continue
        payload = path.read_bytes()
        if len(payload) > _MAX_TEXT_FILE_BYTES:
            omitted.append(
                {"evidence_id": item.evidence_id, "reason": "file_budget"}
            )
            continue
        if _sha256_bytes(payload) != item.sha256:
            raise PaperDeliveryError(
                f"evidence changed during writer packet: {item.relative_path}"
            )
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            omitted.append(
                {"evidence_id": item.evidence_id, "reason": "not_utf8"}
            )
            continue
        encoded_size = len(text.encode("utf-8"))
        if total + encoded_size > _MAX_WRITER_PACKET_BYTES:
            omitted.append(
                {"evidence_id": item.evidence_id, "reason": "packet_budget"}
            )
            continue
        total += encoded_size
        included.append(
            {
                "evidence_id": item.evidence_id,
                "stage": item.stage,
                "relative_path": item.relative_path,
                "sha256": item.sha256,
                "content": text,
            }
        )
        selected_evidence_ids.add(item.evidence_id)
    included.sort(key=lambda item: str(item["evidence_id"]))
    omitted.sort(key=lambda item: item["evidence_id"])
    packet_tokens = [
        item
        for item in bundle.numeric_tokens
        if item.evidence_id in selected_evidence_ids
    ]
    return PaperWriterPacketV71.seal(
        bundle_hash=bundle.bundle_hash,
        request_hash=request.request_hash,
        objective=bundle.objective,
        allowed_claim_types=bundle.allowed_claim_types,
        forbidden_claim_types=bundle.forbidden_claim_types,
        numeric_tokens=packet_tokens,
        included_evidence=included,
        omitted_evidence=omitted,
        narrative_questions=[
            "what must actually be decided",
            "where the information difficulty lies",
            "how the model addresses that difficulty",
            "why the result should be trusted",
            "under what conditions the conclusion holds",
            "when the model must be redone",
        ],
        required_macros=[
            r"\FMAClaim{claim_id}",
            r"\FMAValue{numeric_token_id}",
            r"\FMACite{citation_id}",
            r"\FMAFigure{figure_id}",
            r"\FMATable{table_id}",
            r"\FMASection{section_id}{title}",
        ],
        required_section_ids=sorted(_REQUIRED_SECTIONS[request.venue_profile]),
        authority_denials=[
            "cannot authorize real-world action",
            "cannot grant scientific qualification",
            "cannot invent evidence or results",
            "cannot review own manuscript",
            (
                "claim-ledger statement and scope fields are plain text with "
                "no digits or TeX; values bind only through numeric_token_ids"
            ),
        ],
    )


def load_validated_writer_packet_v71(
    paths: PaperAttemptPathsV71,
) -> PaperWriterPacketV71:
    """Recompute the exact public writer view before any model may consume it."""

    bundle = _read_model(
        paths.attempt_root / "evidence_bundle.json", PaperEvidenceBundleV71
    )
    request = _read_model(
        paths.attempt_root / "author_request.json", PaperAuthoringRequestV71
    )
    expected_attempt = (
        f"paper-{str(bundle.bundle_hash)[:16]}-"
        f"{str(request.request_hash)[:12]}"
    )
    if paths.attempt_id != expected_attempt:
        raise PaperDeliveryError(
            "paper attempt_id does not match its bundle and author request"
        )
    packet = _read_model(
        paths.attempt_root / "writer_packet.json", PaperWriterPacketV71
    )
    rebuilt = _writer_packet(paths.workspace_root, bundle, request)
    if packet != rebuilt:
        raise PaperDeliveryError(
            "writer packet differs from its current bundle/request projection"
        )
    return packet


def _prepare_paper_delivery_v71_unlocked(
    workspace: StageWorkspaceV50,
    *,
    title_hint: str,
    authors: list[str],
    language: str = "zh",
    venue_profile: str = "academic_article",
    requested_model: str = "gpt-5.6-sol",
    max_pages: int = 24,
    max_revision_rounds: int = 2,
) -> PaperAttemptPathsV71:
    """Create or reuse one idempotent content-addressed paper attempt."""

    bundle = build_evidence_bundle_v71(
        workspace, requested_model=requested_model
    )
    request = PaperAuthoringRequestV71.seal(
        bundle_hash=bundle.bundle_hash,
        language=language,
        venue_profile=venue_profile,
        requested_model=requested_model,
        title_hint=title_hint,
        authors=authors,
        max_pages=max_pages,
        max_revision_rounds=max_revision_rounds,
    )
    attempt_id = (
        f"paper-{str(bundle.bundle_hash)[:16]}-"
        f"{str(request.request_hash)[:12]}"
    )
    attempt_root = _safe_relative_path(
        workspace.root,
        (PAPER_ROOT / "attempts" / attempt_id).as_posix(),
    )
    paths = PaperAttemptPathsV71(
        workspace_root=workspace.root,
        attempt_id=attempt_id,
        attempt_root=attempt_root,
    )
    bundle_path = attempt_root / "evidence_bundle.json"
    request_path = attempt_root / "author_request.json"
    finalized = False
    if attempt_root.exists():
        existing_bundle = _read_model(bundle_path, PaperEvidenceBundleV71)
        existing_request = _read_model(request_path, PaperAuthoringRequestV71)
        if existing_bundle != bundle or existing_request != request:
            raise PaperDeliveryError(
                "content-addressed paper attempt already exists with other inputs"
            )
        finalized = (attempt_root / "delivery_receipt.json").exists()
    else:
        for directory in (
            paths.source_root,
            paths.manifests_root,
            paths.reviews_root,
            paths.builds_root,
        ):
            directory.mkdir(parents=True, exist_ok=False)
        _write_json(bundle_path, bundle.model_dump(mode="json"))
        _write_json(request_path, request.model_dump(mode="json"))
        _write_json(
            attempt_root / "writer_packet.json",
            _writer_packet(workspace.root, bundle, request).model_dump(
                mode="json"
            ),
        )
    load_validated_writer_packet_v71(paths)
    if finalized:
        if current_paper_attempt_v71(workspace.root) != paths:
            raise PaperDeliveryError(
                "finalized paper attempt is not the current projection"
            )
        verification = verify_paper_delivery_v71(workspace)
        if not verification.ok:
            raise PaperDeliveryError(
                "existing finalized paper attempt failed verification: "
                + "; ".join(verification.mismatches)
            )
        return paths
    _write_json(
        _safe_relative_path(workspace.root, CURRENT_PATH.as_posix()),
        PaperCurrentProjectionV71(
            attempt_id=attempt_id,
            bundle_hash=bundle.bundle_hash,
            request_hash=request.request_hash,
            status="NEEDS_REVISION",
        ).model_dump(mode="json"),
    )
    return paths


def prepare_paper_delivery_v71(
    workspace: StageWorkspaceV50,
    *,
    title_hint: str,
    authors: list[str],
    language: str = "zh",
    venue_profile: str = "academic_article",
    requested_model: str = "gpt-5.6-sol",
    max_pages: int = 24,
    max_revision_rounds: int = 2,
) -> PaperAttemptPathsV71:
    """Atomically create or reuse one content-addressed paper attempt."""

    with paper_writer_lock_v71(workspace.root):
        return _prepare_paper_delivery_v71_unlocked(
            workspace,
            title_hint=title_hint,
            authors=authors,
            language=language,
            venue_profile=venue_profile,
            requested_model=requested_model,
            max_pages=max_pages,
            max_revision_rounds=max_revision_rounds,
        )


def current_paper_attempt_v71(
    workspace_root: str | Path,
) -> PaperAttemptPathsV71:
    root = Path(workspace_root).resolve()
    current_path = _safe_relative_path(root, CURRENT_PATH.as_posix())
    if not current_path.is_file():
        raise PaperDeliveryError("no current V7.1 paper attempt")
    projection = _read_model(current_path, PaperCurrentProjectionV71)
    attempt_id = projection.attempt_id
    attempt_root = _safe_relative_path(
        root, (PAPER_ROOT / "attempts" / attempt_id).as_posix()
    )
    if not attempt_root.is_dir() or attempt_root.is_symlink():
        raise PaperDeliveryError("current paper attempt directory is missing")
    paths = PaperAttemptPathsV71(root, attempt_id, attempt_root)
    bundle = _read_model(
        attempt_root / "evidence_bundle.json", PaperEvidenceBundleV71
    )
    request = _read_model(
        attempt_root / "author_request.json", PaperAuthoringRequestV71
    )
    expected_attempt = (
        f"paper-{str(bundle.bundle_hash)[:16]}-"
        f"{str(request.request_hash)[:12]}"
    )
    if (
        projection.bundle_hash != bundle.bundle_hash
        or projection.request_hash != request.request_hash
        or attempt_id != expected_attempt
    ):
        raise PaperDeliveryError(
            "current paper projection does not bind its exact attempt inputs"
        )
    return paths


def project_paper_status_v71(
    workspace_root: str | Path,
    status: str,
) -> PaperCurrentProjectionV71:
    """Persist a non-authoritative workflow stop without minting delivery."""

    if status not in {"NEEDS_REVISION", "HUMAN"}:
        raise PaperDeliveryError(
            "only NEEDS_REVISION or HUMAN may be projected without finalization"
        )
    with paper_writer_lock_v71(workspace_root):
        paths = current_paper_attempt_v71(workspace_root)
        assert_paper_attempt_open_v71(paths)
        bundle = _read_model(
            paths.attempt_root / "evidence_bundle.json",
            PaperEvidenceBundleV71,
        )
        request = _read_model(
            paths.attempt_root / "author_request.json",
            PaperAuthoringRequestV71,
        )
        projection = PaperCurrentProjectionV71(
            attempt_id=paths.attempt_id,
            bundle_hash=bundle.bundle_hash,
            request_hash=request.request_hash,
            status=status,
        )
        _write_json(
            _safe_relative_path(
                Path(workspace_root).resolve(), CURRENT_PATH.as_posix()
            ),
            projection.model_dump(mode="json"),
        )
        return projection


def _artifact_hash(
    errors: set[str], root: Path, relative_path: str, expected: str, label: str
) -> None:
    try:
        path = _safe_relative_path(root, relative_path)
        if not path.is_file():
            errors.add(f"{label}: missing file {relative_path}")
        elif _sha256_file(path) != expected:
            errors.add(f"{label}: hash mismatch for {relative_path}")
    except (OSError, PaperDeliveryError) as exc:
        errors.add(f"{label}: {exc}")


def _expected_citation_snapshot(citation: object) -> dict[str, object]:
    return {
        "schema_version": "7.1-citation-source-snapshot",
        "title": getattr(citation, "title"),
        "authors": getattr(citation, "authors"),
        "year": getattr(citation, "year"),
        "venue": getattr(citation, "venue"),
        "doi": getattr(citation, "doi"),
        "url": getattr(citation, "url"),
    }


def _collect_macros(text: str, name: str) -> list[str]:
    return _MACROS[name].findall(text)


def _meaningful_paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    for raw in re.split(r"\n\s*\n", text):
        value = _SECTION.sub("", raw)
        value = re.sub(
            r"\\FMASubsection\{[A-Za-z][A-Za-z0-9_.:-]{0,127}\}\{[^{}]+\}",
            "",
            value,
        )
        value = re.sub(r"\\(?:begin|end)\{[^{}]+\}", "", value).strip()
        if len(value) < 40:
            continue
        if value.startswith((r"\[", r"\]", "%")):
            continue
        paragraphs.append(value)
    return paragraphs


def _audit_paper_content_v71_unlocked(
    workspace: StageWorkspaceV50,
) -> PaperContentAuditV71:
    """Run deterministic provenance and manuscript-closure checks."""

    paths = current_paper_attempt_v71(workspace.root)
    assert_paper_attempt_open_v71(paths)
    bundle_path = paths.attempt_root / "evidence_bundle.json"
    request_path = paths.attempt_root / "author_request.json"
    metadata_path = paths.attempt_root / "metadata.json"
    abstract_path = paths.source_root / "abstract.tex"
    body_path = paths.source_root / "body.tex"
    ledger_path = paths.manifests_root / "claim_ledger.json"
    citation_path = paths.manifests_root / "citations.json"
    figure_path = paths.manifests_root / "figures.json"
    table_path = paths.manifests_root / "tables.json"
    writer_receipt_path = paths.attempt_root / "writer_transport_receipt.json"
    writer_output_path = paths.attempt_root / "writer_output.json"
    writer_packet_path = paths.attempt_root / "writer_packet.json"
    writer_role_request_path = paths.attempt_root / "writer_role_request.json"
    required = [
        bundle_path,
        request_path,
        metadata_path,
        abstract_path,
        body_path,
        ledger_path,
        citation_path,
        figure_path,
        table_path,
        writer_receipt_path,
        writer_output_path,
        writer_packet_path,
        writer_role_request_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise PaperDeliveryError(
            "paper authoring artifacts are incomplete: " + ", ".join(missing)
        )

    bundle = _read_model(bundle_path, PaperEvidenceBundleV71)
    request = _read_model(request_path, PaperAuthoringRequestV71)
    metadata = _read_model(metadata_path, PaperMetadataV71)
    ledger = _read_model(ledger_path, PaperClaimLedgerV71)
    citations = _read_model(citation_path, CitationManifestV71)
    figures = _read_model(figure_path, FigureManifestV71)
    tables = _read_model(table_path, TableManifestV71)
    writer_receipt = _read_model(
        writer_receipt_path, RoleProcessReceiptV51
    )
    writer_output = _read_json(writer_output_path)
    writer_role_request = _read_model(
        writer_role_request_path, RoleRequestV51
    )
    errors: set[str] = set()
    warnings: set[str] = set()
    try:
        writer_packet = load_validated_writer_packet_v71(paths)
    except PaperDeliveryError as exc:
        errors.add(str(exc))
        writer_packet = None
    abstract = abstract_path.read_text(encoding="utf-8")
    body = body_path.read_text(encoding="utf-8")
    manuscript = abstract + "\n\n" + body

    try:
        current_bundle = build_evidence_bundle_v71(
            workspace, requested_model=bundle.requested_model
        )
        if current_bundle != bundle:
            errors.add("evidence bundle is stale relative to current S0-S6")
    except PaperDeliveryError as exc:
        errors.add(f"current S0-S6 authority unavailable: {exc}")
    if request.bundle_hash != bundle.bundle_hash:
        errors.add("author request binds another evidence bundle")
    for label, bound_hash in (
        ("metadata", metadata.bundle_hash),
        ("claim ledger", ledger.bundle_hash),
        ("citation manifest", citations.bundle_hash),
        ("figure manifest", figures.bundle_hash),
        ("table manifest", tables.bundle_hash),
    ):
        if bound_hash != bundle.bundle_hash:
            errors.add(f"{label} binds another evidence bundle")
    if metadata.requested_model != request.requested_model:
        errors.add("metadata requested_model differs from author request")
    if metadata.language != request.language:
        errors.add("metadata language differs from author request")
    if metadata.venue_profile != request.venue_profile:
        errors.add("metadata venue_profile differs from author request")
    if metadata.authors != request.authors:
        errors.add("metadata authors differ from author request")
    auxiliary_surfaces = {
        "paper title": metadata.title,
        **{
            f"figure {item.figure_id} caption": item.caption
            for item in figures.figures
        },
        **{
            f"figure {item.figure_id} alt text": item.alt_text
            for item in figures.figures
        },
        **{
            f"table {item.table_id} caption": item.caption
            for item in tables.tables
        },
        **{
            f"claim {item.claim_id} statement": item.statement
            for item in ledger.claims
        },
        **{
            f"claim {item.claim_id} scope": item.scope_qualifier
            for item in ledger.claims
        },
    }
    for label, value in auxiliary_surfaces.items():
        if label.startswith("claim ") and re.search(r"\d", value):
            errors.add(
                f"{label} contains a raw numeric literal; claim-ledger prose "
                "must be number-free and bind values only through "
                "numeric_token_ids"
            )
        elif _RAW_NUMBER.search(value):
            errors.add(
                f"{label} contains a raw numeric literal; auxiliary prose "
                "must omit exact values and render them only in manuscript "
                "text through FMAValue"
            )
        if _FORBIDDEN_WIDENING.search(value):
            errors.add(f"{label} contains forbidden claim-widening language")
        if _TEX_COMMAND.search(value) or "^^" in value:
            errors.add(f"{label} may not contain authored TeX commands")
    if writer_receipt.requested_model != request.requested_model:
        errors.add("writer transport requested another model")
    if writer_receipt.transport != "codex_cli":
        errors.add("writer transport is fixture-only, not native Codex CLI")
    if writer_receipt.receipt_hash is None:
        errors.add("writer transport receipt is not sealed")
    if not writer_receipt.scratch_unchanged or writer_receipt.tool_event_count:
        errors.add("writer transport was not tool-free and scratch-stable")
    try:
        writer_role_request.assert_sealed()
    except ValueError as exc:
        errors.add(f"writer role request is not sealed: {exc}")
    if (
        writer_role_request.request_hash != writer_receipt.request_hash
        or writer_role_request.run_id != writer_receipt.run_id
        or writer_role_request.context_id != writer_receipt.context_id
        or writer_role_request.role_name != "paper_writer_v71"
        or writer_role_request.role_kind != "generator"
        or writer_role_request.stage != "S6"
        or writer_role_request.subject_id != paths.attempt_id
        or writer_receipt.role_name != "paper_writer_v71"
        or writer_receipt.role_kind != "generator"
    ):
        errors.add("writer role request and transport receipt disagree")
    if (
        writer_role_request.public_inputs.get("author_request")
        != request.model_dump(mode="json")
    ):
        errors.add("writer role request contains another author request")
    if writer_packet is not None and (
        writer_role_request.public_inputs.get("writer_packet")
        != writer_packet.model_dump(mode="json")
    ):
        errors.add("writer role request contains another writer packet")
    revision_context = writer_role_request.public_inputs.get(
        "revision_context"
    )
    if not isinstance(revision_context, dict) or set(revision_context) != {
        "revision_round",
        "feedback",
        "previous_output_sha256",
        "previous_output",
    }:
        errors.add("writer role request has an invalid revision context")
    else:
        revision_round = revision_context.get("revision_round")
        feedback = revision_context.get("feedback")
        previous_hash = revision_context.get("previous_output_sha256")
        previous_output = revision_context.get("previous_output")
        if (
            not isinstance(revision_round, int)
            or isinstance(revision_round, bool)
            or revision_round < 0
            or revision_round > request.max_revision_rounds
            or not isinstance(feedback, list)
            or len(feedback) > 128
            or any(
                not isinstance(item, str)
                or not item.strip()
                or len(item) > 4_000
                for item in feedback
            )
        ):
            errors.add("writer revision round or feedback is invalid")
        elif revision_round == 0 and (
            feedback or previous_hash is not None or previous_output is not None
        ):
            errors.add("initial writer round contains revision-only state")
        elif revision_round > 0 and (
            not feedback
            or not isinstance(previous_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", previous_hash)
            or not isinstance(previous_output, dict)
            or sha256_value(previous_output) != previous_hash
        ):
            errors.add("writer revision does not bind its prior output")
    format_retry = writer_role_request.public_inputs.get("format_retry")
    if (
        not isinstance(format_retry, dict)
        or set(format_retry)
        != {"format_attempt", "previous_contract_error"}
        or format_retry.get("format_attempt") not in {0, 1}
    ):
        errors.add("writer role request has an invalid format-retry contract")
    else:
        previous_error = format_retry.get("previous_contract_error")
        if (
            format_retry["format_attempt"] == 0
            and previous_error is not None
        ) or (
            format_retry["format_attempt"] == 1
            and (
                not isinstance(previous_error, str)
                or not previous_error.strip()
                or len(previous_error) > 4_000
            )
        ):
            errors.add("writer format-retry lineage is inconsistent")
    if metadata.writer_context_ids != [writer_receipt.context_id]:
        errors.add("metadata does not bind the exact writer context")
    if not isinstance(writer_output, dict):
        errors.add("writer output must be a JSON object")
    else:
        if sha256_value(writer_output) != writer_receipt.output_hash:
            errors.add("writer output hash differs from transport receipt")
        if writer_output.get("request_hash") != writer_receipt.request_hash:
            errors.add("writer output binds another role request")
        component_checks = {
            "metadata": metadata.model_dump(mode="json"),
            "abstract_tex": abstract,
            "body_tex": body,
            "claim_ledger": ledger.model_dump(mode="json"),
            "citations": citations.model_dump(mode="json"),
            "figures": figures.model_dump(mode="json"),
            "tables": tables.model_dump(mode="json"),
        }
        for component, expected in component_checks.items():
            if writer_output.get(component) != expected:
                errors.add(
                    f"projected author artifact differs from writer output: "
                    f"{component}"
                )

    if _FORBIDDEN_TEX.search(manuscript):
        errors.add("authored TeX contains a forbidden control sequence")
    if _FORBIDDEN_WIDENING.search(manuscript):
        errors.add("manuscript contains forbidden claim-widening language")
    unknown_commands = sorted(
        set(_TEX_COMMAND.findall(manuscript)) - _ALLOWED_TEX_COMMANDS
    )
    if unknown_commands:
        errors.add(
            "authored TeX contains non-allowlisted commands: "
            + ", ".join(unknown_commands)
        )
    unknown_environments = sorted(
        set(_TEX_ENVIRONMENT.findall(manuscript))
        - _ALLOWED_TEX_ENVIRONMENTS
    )
    if unknown_environments:
        errors.add(
            "authored TeX contains non-allowlisted environments: "
            + ", ".join(unknown_environments)
        )
    if any(token in manuscript for token in ("%%FMA_", "[[", "]]", "{{", "}}")):
        errors.add("manuscript contains an unresolved placeholder")
    if "^^" in manuscript or any(
        ord(character) < 32 and character not in "\n\r\t"
        for character in manuscript
    ):
        errors.add("manuscript contains forbidden TeX/control encoding")
    if not _collect_macros(abstract, "claim"):
        errors.add("abstract must cite at least one registered claim")
    for index, paragraph in enumerate(_meaningful_paragraphs(manuscript), 1):
        if not _collect_macros(paragraph, "claim"):
            errors.add(
                f"meaningful manuscript paragraph {index} lacks an FMAClaim anchor"
            )
    raw_number_scan = _MACROS["value"].sub("", manuscript)
    raw_number = _RAW_NUMBER.search(raw_number_scan)
    if raw_number:
        errors.add(
            "manuscript contains a raw multi-digit/decimal numeric literal: "
            + raw_number.group(0)
        )

    required_sections = _REQUIRED_SECTIONS[metadata.venue_profile]
    actual_sections = set(_SECTION.findall(body))
    for section_id in sorted(required_sections - actual_sections):
        errors.add(f"required section is missing: {section_id}")

    evidence_ids = {item.evidence_id for item in bundle.evidence_items}
    evidence_files = {
        (item.relative_path, item.sha256) for item in bundle.evidence_items
    }
    disclosed_evidence_files = (
        {
            (item.relative_path, item.sha256)
            for item in writer_packet.included_evidence
        }
        if writer_packet is not None
        else set()
    )
    numeric_ids = {item.token_id for item in bundle.numeric_tokens}
    claim_ids = {item.claim_id for item in ledger.claims}
    citation_ids = {item.citation_id for item in citations.citations}
    figure_ids = {item.figure_id for item in figures.figures}
    table_ids = {item.table_id for item in tables.tables}
    used_claims = set(_collect_macros(manuscript, "claim"))
    used_values = set(_collect_macros(manuscript, "value"))
    used_citations = set(_collect_macros(manuscript, "cite"))
    used_figures = set(_collect_macros(manuscript, "figure"))
    used_tables = set(_collect_macros(manuscript, "table"))

    for label, used, known in (
        ("claim", used_claims, claim_ids),
        ("numeric token", used_values, numeric_ids),
        ("citation", used_citations, citation_ids),
        ("figure", used_figures, figure_ids),
        ("table", used_tables, table_ids),
    ):
        for unknown in sorted(used - known):
            errors.add(f"unknown {label} referenced by manuscript: {unknown}")
    for missing_claim in sorted(claim_ids - used_claims):
        errors.add(f"registered claim is absent from manuscript: {missing_claim}")
    for missing_id in sorted(citation_ids - used_citations):
        errors.add(f"registered citation is unused: {missing_id}")
    for missing_id in sorted(figure_ids - used_figures):
        errors.add(f"registered figure is unused: {missing_id}")
    for missing_id in sorted(table_ids - used_tables):
        errors.add(f"registered table is unused: {missing_id}")

    claims_by_id = {item.claim_id: item for item in ledger.claims}
    for claim in ledger.claims:
        if claim.claim_type in bundle.forbidden_claim_types:
            errors.add(
                f"claim {claim.claim_id} uses forbidden type {claim.claim_type}"
            )
        for evidence_id in claim.evidence_ids:
            if evidence_id not in evidence_ids:
                errors.add(
                    f"claim {claim.claim_id} references unknown evidence {evidence_id}"
                )
        for numeric_id in claim.numeric_token_ids:
            if numeric_id not in numeric_ids:
                errors.add(
                    f"claim {claim.claim_id} references unknown numeric token {numeric_id}"
                )
            elif numeric_id not in used_values:
                errors.add(
                    f"claim {claim.claim_id} numeric token is absent from manuscript: "
                    f"{numeric_id}"
                )
        for citation_id in claim.citation_ids:
            if citation_id not in citation_ids:
                errors.add(
                    f"claim {claim.claim_id} references unknown citation {citation_id}"
                )
            else:
                citation = next(
                    item
                    for item in citations.citations
                    if item.citation_id == citation_id
                )
                if claim.claim_id not in citation.supports_claim_ids:
                    errors.add(
                        f"claim {claim.claim_id} citation support is not reciprocal "
                        f"with {citation_id}"
                    )

    for citation in citations.citations:
        for claim_id in citation.supports_claim_ids:
            if claim_id not in claim_ids:
                errors.add(
                    f"citation {citation.citation_id} supports unknown claim {claim_id}"
                )
            elif citation.citation_id not in claims_by_id[claim_id].citation_ids:
                errors.add(
                    f"citation {citation.citation_id} support is not reciprocal "
                    f"with claim {claim_id}"
                )
        if citation.verification_status != "SNAPSHOT_BOUND":
            errors.add(f"citation remains HUMAN: {citation.citation_id}")
        if citation.source_snapshot_path and citation.source_snapshot_sha256:
            _artifact_hash(
                errors,
                workspace.root,
                citation.source_snapshot_path,
                citation.source_snapshot_sha256,
                f"citation {citation.citation_id}",
            )
            if (
                citation.source_snapshot_path,
                citation.source_snapshot_sha256,
            ) not in evidence_files:
                errors.add(
                    f"citation {citation.citation_id} source is not S0-S6 evidence"
                )
            if (
                citation.source_snapshot_path,
                citation.source_snapshot_sha256,
            ) not in disclosed_evidence_files:
                errors.add(
                    f"citation {citation.citation_id} source snapshot was not "
                    "disclosed to the cold semantic reviewer"
                )
            try:
                citation_snapshot = _read_json(
                    _safe_relative_path(
                        workspace.root, citation.source_snapshot_path
                    )
                )
            except PaperDeliveryError as exc:
                errors.add(
                    f"citation {citation.citation_id} snapshot is invalid: {exc}"
                )
            else:
                if citation_snapshot != _expected_citation_snapshot(citation):
                    errors.add(
                        f"citation {citation.citation_id} metadata differs from "
                        "its frozen source snapshot"
                    )

    for figure in figures.figures:
        for evidence_id in figure.evidence_ids:
            if evidence_id not in evidence_ids:
                errors.add(
                    f"figure {figure.figure_id} references unknown evidence "
                    f"{evidence_id}"
                )
        for claim_id in figure.claim_ids:
            if claim_id not in claim_ids:
                errors.add(
                    f"figure {figure.figure_id} references unknown claim "
                    f"{claim_id}"
                )
        _artifact_hash(
            errors,
            workspace.root,
            figure.artifact_path,
            figure.artifact_sha256,
            f"figure {figure.figure_id}",
        )
        if (figure.artifact_path, figure.artifact_sha256) not in evidence_files:
            errors.add(
                f"figure {figure.figure_id} is not an S0-S6 evidence artifact"
            )
        if figure.generator_path and figure.generator_sha256:
            _artifact_hash(
                errors,
                workspace.root,
                figure.generator_path,
                figure.generator_sha256,
                f"figure generator {figure.figure_id}",
            )
            if (
                figure.generator_path,
                figure.generator_sha256,
            ) not in evidence_files:
                errors.add(
                    f"figure generator {figure.figure_id} is not S0-S6 evidence"
                )

    for table in tables.tables:
        for evidence_id in table.evidence_ids:
            if evidence_id not in evidence_ids:
                errors.add(
                    f"table {table.table_id} references unknown evidence "
                    f"{evidence_id}"
                )
        for claim_id in table.claim_ids:
            if claim_id not in claim_ids:
                errors.add(
                    f"table {table.table_id} references unknown claim "
                    f"{claim_id}"
                )
        _artifact_hash(
            errors,
            workspace.root,
            table.csv_path,
            table.csv_sha256,
            f"table {table.table_id}",
        )
        if (table.csv_path, table.csv_sha256) not in evidence_files:
            errors.add(
                f"table {table.table_id} is not an S0-S6 evidence artifact"
            )

    if not citations.citations:
        warnings.add("paper contains no registered literature citations")
    if not figures.figures:
        warnings.add("paper contains no registered figures")
    if not tables.tables:
        warnings.add("paper contains no registered tables")

    audit = PaperContentAuditV71.seal(
        bundle_hash=bundle.bundle_hash,
        attempt_id=paths.attempt_id,
        status="PASS" if not errors else "FAIL",
        errors=sorted(errors),
        warnings=sorted(warnings),
        metadata_sha256=_sha256_file(metadata_path),
        author_request_sha256=_sha256_file(request_path),
        writer_packet_sha256=_sha256_file(writer_packet_path),
        abstract_sha256=_sha256_file(abstract_path),
        body_sha256=_sha256_file(body_path),
        claim_ledger_sha256=_sha256_file(ledger_path),
        citation_manifest_sha256=_sha256_file(citation_path),
        figure_manifest_sha256=_sha256_file(figure_path),
        table_manifest_sha256=_sha256_file(table_path),
        writer_transport_receipt_sha256=_sha256_file(writer_receipt_path),
        writer_role_request_sha256=_sha256_file(writer_role_request_path),
        writer_output_sha256=_sha256_file(writer_output_path),
        checked_at=_utc_now(),
    )
    _write_json(
        paths.reviews_root / "content_audit.json",
        audit.model_dump(mode="json"),
    )
    return audit


def audit_paper_content_v71(
    workspace: StageWorkspaceV50,
) -> PaperContentAuditV71:
    """Atomically audit and project the current authored manuscript."""

    with paper_writer_lock_v71(workspace.root):
        return _audit_paper_content_v71_unlocked(workspace)


def record_semantic_review_v71(
    workspace_root: str | Path,
    review: PaperSemanticReviewV71 | Mapping[str, object],
) -> PaperSemanticReviewV71:
    with paper_writer_lock_v71(workspace_root):
        paths = current_paper_attempt_v71(workspace_root)
        assert_paper_attempt_open_v71(paths)
        parsed = (
            review
            if isinstance(review, PaperSemanticReviewV71)
            else PaperSemanticReviewV71.model_validate(review)
        )
        _verify_review_role_artifacts(paths, parsed, "semantic")
        _write_json(
            paths.reviews_root / "semantic_review.json",
            parsed.model_dump(mode="json"),
        )
        return parsed


def record_layout_review_v71(
    workspace_root: str | Path,
    review: PaperLayoutReviewV71 | Mapping[str, object],
) -> PaperLayoutReviewV71:
    with paper_writer_lock_v71(workspace_root):
        paths = current_paper_attempt_v71(workspace_root)
        assert_paper_attempt_open_v71(paths)
        parsed = (
            review
            if isinstance(review, PaperLayoutReviewV71)
            else PaperLayoutReviewV71.model_validate(review)
        )
        _verify_review_role_artifacts(paths, parsed, "layout")
        _write_json(
            paths.reviews_root / "layout_review.json",
            parsed.model_dump(mode="json"),
        )
        return parsed


def _verify_review_role_artifacts(
    paths: PaperAttemptPathsV71,
    review: PaperSemanticReviewV71 | PaperLayoutReviewV71,
    kind: str,
) -> None:
    if kind not in {"semantic", "layout"}:
        raise ValueError("review role kind must be semantic or layout")
    request_path = paths.reviews_root / f"{kind}_role_request.json"
    draft_path = paths.reviews_root / f"{kind}_role_draft.json"
    receipt_path = paths.reviews_root / f"{kind}_transport_receipt.json"
    if (
        not request_path.is_file()
        or not draft_path.is_file()
        or not receipt_path.is_file()
    ):
        raise PaperDeliveryError(f"{kind} review role evidence is incomplete")
    if _sha256_file(request_path) != review.reviewer_request_sha256:
        raise PaperDeliveryError(f"{kind} review request hash mismatch")
    if _sha256_file(draft_path) != review.reviewer_draft_sha256:
        raise PaperDeliveryError(f"{kind} review draft hash mismatch")
    if (
        _sha256_file(receipt_path)
        != review.reviewer_transport_receipt_sha256
    ):
        raise PaperDeliveryError(f"{kind} review transport hash mismatch")
    request = _read_model(request_path, RoleRequestV51)
    receipt = _read_model(receipt_path, RoleProcessReceiptV51)
    draft = _read_json(draft_path)
    if not isinstance(draft, dict):
        raise PaperDeliveryError(f"{kind} review draft must be an object")
    try:
        request.assert_sealed()
    except ValueError as exc:
        raise PaperDeliveryError(
            f"{kind} review request is not sealed: {exc}"
        ) from exc
    expected_role = (
        "paper_semantic_reviewer_v71"
        if kind == "semantic"
        else "paper_layout_reviewer_v71"
    )
    if receipt.receipt_hash is None:
        raise PaperDeliveryError(f"{kind} transport receipt is not sealed")
    if (
        request.request_hash != receipt.request_hash
        or request.run_id != receipt.run_id
        or request.context_id != receipt.context_id
        or request.context_id != review.reviewer_context_id
        or request.role_name != expected_role
        or receipt.role_name != expected_role
        or request.role_kind != "reviewer"
        or receipt.role_kind != "reviewer"
        or request.stage != "S6"
        or request.subject_id != paths.attempt_id
        or receipt.requested_model != review.requested_model
    ):
        raise PaperDeliveryError(
            f"{kind} request, receipt, and review projection disagree"
        )
    if not receipt.scratch_unchanged or receipt.tool_event_count:
        raise PaperDeliveryError(
            f"{kind} reviewer was not tool-free and scratch-stable"
        )
    format_retry = request.public_inputs.get("format_retry")
    if (
        not isinstance(format_retry, dict)
        or set(format_retry)
        != {"format_attempt", "previous_contract_error"}
        or format_retry.get("format_attempt") not in {0, 1}
    ):
        raise PaperDeliveryError(
            f"{kind} reviewer has an invalid format-retry contract"
        )
    previous_error = format_retry.get("previous_contract_error")
    if (
        format_retry["format_attempt"] == 0
        and previous_error is not None
    ) or (
        format_retry["format_attempt"] == 1
        and (
            not isinstance(previous_error, str)
            or not previous_error.strip()
            or len(previous_error) > 4_000
        )
    ):
        raise PaperDeliveryError(
            f"{kind} reviewer format-retry lineage is inconsistent"
        )
    if (
        sha256_value(draft) != receipt.output_hash
        or draft.get("request_hash") != request.request_hash
        or draft.get("role_name") != expected_role
        or draft.get("verdict") != review.verdict
    ):
        raise PaperDeliveryError(
            f"{kind} raw draft differs from its transport/projection"
        )
    if kind == "semantic":
        assert isinstance(review, PaperSemanticReviewV71)
        draft_claims = draft.get("reviewed_claim_ids")
        draft_findings = draft.get("findings")
        if (
            not isinstance(draft_claims, list)
            or sorted(set(draft_claims)) != review.reviewed_claim_ids
            or not isinstance(draft_findings, list)
            or sorted(
                draft_findings,
                key=lambda item: str(item.get("finding_id"))
                if isinstance(item, dict)
                else "",
            )
            != [
                item.model_dump(mode="json") for item in review.findings
            ]
        ):
            raise PaperDeliveryError(
                "semantic raw draft differs from review projection"
            )
        content_audit = request.public_inputs.get("content_audit")
        if (
            not isinstance(content_audit, dict)
            or content_audit.get("audit_hash") != review.content_audit_hash
        ):
            raise PaperDeliveryError(
                "semantic request binds another content audit"
            )
    else:
        assert isinstance(review, PaperLayoutReviewV71)
        from .paper_renderer import load_current_build_v71

        build = load_current_build_v71(paths)
        author_request = _read_model(
            paths.attempt_root / "author_request.json",
            PaperAuthoringRequestV71,
        )
        draft_pages = draft.get("pages_reviewed")
        draft_findings = draft.get("findings")
        if (
            not isinstance(draft_pages, list)
            or sorted(set(draft_pages)) != review.pages_reviewed
            or not isinstance(draft_findings, list)
            or sorted(set(draft_findings)) != review.findings
            or request.public_inputs.get("build_hash") != build.build_hash
            or request.public_inputs.get("page_images") != build.page_images
            or request.public_inputs.get("expected_page_count")
            != len(build.page_images)
            or request.public_inputs.get("venue_profile")
            != author_request.venue_profile
            or request.public_inputs.get("max_pages")
            != author_request.max_pages
            or review.build_hash != build.build_hash
        ):
            raise PaperDeliveryError(
                "layout request/draft differs from the reviewed build projection"
            )


def _require_native_codex_receipt(
    path: Path,
    *,
    label: str,
    requested_model: str,
) -> RoleProcessReceiptV51:
    receipt = _read_model(path, RoleProcessReceiptV51)
    if (
        receipt.receipt_hash is None
        or receipt.transport != "codex_cli"
        or receipt.provider.strip().lower() == "fixture"
        or receipt.requested_model != requested_model
        or not receipt.scratch_unchanged
        or receipt.tool_event_count != 0
    ):
        raise PaperDeliveryError(
            f"{label} is not a qualifying native Codex CLI receipt"
        )
    return receipt


def _finalize_paper_delivery_v71_unlocked(
    workspace: StageWorkspaceV50,
) -> PaperDeliveryReceiptV71:
    """Bind passing content, semantic, build, and full-page layout artifacts."""

    from .paper_renderer import load_current_build_v71, verify_paper_build_v71

    paths = current_paper_attempt_v71(workspace.root)
    current_path = _safe_relative_path(
        workspace.root, CURRENT_PATH.as_posix()
    )
    starting_projection = _read_model(
        current_path, PaperCurrentProjectionV71
    )
    bundle = _read_model(
        paths.attempt_root / "evidence_bundle.json", PaperEvidenceBundleV71
    )
    request = _read_model(
        paths.attempt_root / "author_request.json",
        PaperAuthoringRequestV71,
    )
    audit = _read_model(
        paths.reviews_root / "content_audit.json", PaperContentAuditV71
    )
    semantic = _read_model(
        paths.reviews_root / "semantic_review.json", PaperSemanticReviewV71
    )
    layout = _read_model(
        paths.reviews_root / "layout_review.json", PaperLayoutReviewV71
    )
    _verify_review_role_artifacts(paths, semantic, "semantic")
    _verify_review_role_artifacts(paths, layout, "layout")
    _require_native_codex_receipt(
        paths.attempt_root / "writer_transport_receipt.json",
        label="paper author transport",
        requested_model=request.requested_model,
    )
    _require_native_codex_receipt(
        paths.reviews_root / "semantic_transport_receipt.json",
        label="semantic reviewer transport",
        requested_model=request.requested_model,
    )
    _require_native_codex_receipt(
        paths.reviews_root / "layout_transport_receipt.json",
        label="layout reviewer transport",
        requested_model=request.requested_model,
    )
    ledger = _read_model(
        paths.manifests_root / "claim_ledger.json", PaperClaimLedgerV71
    )
    build = load_current_build_v71(paths)
    build_verification = verify_paper_build_v71(workspace.root)
    if not build_verification.ok:
        raise PaperDeliveryError(
            "paper build verification failed: "
            + "; ".join(build_verification.mismatches)
        )
    if audit.status != "PASS":
        raise PaperDeliveryError("content audit has not passed")
    if semantic.verdict != "APPROVE":
        raise PaperDeliveryError("semantic review has not approved the paper")
    if layout.verdict != "APPROVE":
        raise PaperDeliveryError("layout review has not approved the paper")
    if semantic.bundle_hash != bundle.bundle_hash:
        raise PaperDeliveryError("semantic review binds another bundle")
    if semantic.content_audit_hash != audit.audit_hash:
        raise PaperDeliveryError("semantic review binds another content audit")
    if set(semantic.reviewed_claim_ids) != {
        item.claim_id for item in ledger.claims
    }:
        raise PaperDeliveryError("semantic review did not cover every claim")
    if layout.build_hash != build.build_hash:
        raise PaperDeliveryError("layout review binds another build")
    semantic_transport = paths.reviews_root / "semantic_transport_receipt.json"
    layout_transport = paths.reviews_root / "layout_transport_receipt.json"
    if (
        not semantic_transport.is_file()
        or _sha256_file(semantic_transport)
        != semantic.reviewer_transport_receipt_sha256
    ):
        raise PaperDeliveryError("semantic reviewer transport receipt mismatch")
    if (
        not layout_transport.is_file()
        or _sha256_file(layout_transport)
        != layout.reviewer_transport_receipt_sha256
    ):
        raise PaperDeliveryError("layout reviewer transport receipt mismatch")
    expected_pages = list(range(1, len(build.page_images) + 1))
    if layout.pages_reviewed != expected_pages:
        raise PaperDeliveryError("layout review did not inspect every PDF page")
    current_s6 = workspace.current_gate("S6")
    if current_s6 != bundle.s6_gate_hash:
        raise PaperDeliveryError("paper evidence is stale relative to S6")

    receipt = PaperDeliveryReceiptV71.seal(
        status="DRAFT_READY",
        bundle_hash=bundle.bundle_hash,
        s6_gate_hash=bundle.s6_gate_hash,
        content_audit_hash=audit.audit_hash,
        semantic_review_sha256=_sha256_file(
            paths.reviews_root / "semantic_review.json"
        ),
        build_hash=build.build_hash,
        layout_review_sha256=_sha256_file(
            paths.reviews_root / "layout_review.json"
        ),
        pdf_path=build.pdf_path,
        pdf_sha256=build.pdf_sha256,
        created_at=_utc_now(),
    )
    receipt_path = paths.attempt_root / "delivery_receipt.json"
    _write_json(receipt_path, receipt.model_dump(mode="json"))
    if (
        _read_model(current_path, PaperCurrentProjectionV71)
        != starting_projection
    ):
        raise PaperDeliveryError(
            "current paper projection changed during finalization"
        )
    _write_json(
        current_path,
        PaperCurrentProjectionV71(
            attempt_id=paths.attempt_id,
            bundle_hash=bundle.bundle_hash,
            request_hash=_read_model(
                paths.attempt_root / "author_request.json",
                PaperAuthoringRequestV71,
            ).request_hash,
            delivery_hash=receipt.delivery_hash,
            status=receipt.status,
        ).model_dump(mode="json"),
    )
    return receipt


def finalize_paper_delivery_v71(
    workspace: StageWorkspaceV50,
) -> PaperDeliveryReceiptV71:
    """Atomically bind the approved build and advance the current projection."""

    with paper_writer_lock_v71(workspace.root):
        return _finalize_paper_delivery_v71_unlocked(workspace)


def verify_paper_delivery_v71(
    workspace: StageWorkspaceV50,
) -> PaperDeliveryVerificationV71:
    """Pure-read verification of the current publication projection."""

    mismatches: set[str] = set()
    try:
        paths = current_paper_attempt_v71(workspace.root)
    except PaperDeliveryError as exc:
        return PaperDeliveryVerificationV71(
            ok=False,
            status="NEEDS_REVISION",
            mismatches=[str(exc)],
        )
    bundle: PaperEvidenceBundleV71 | None = None
    try:
        bundle = _read_model(
            paths.attempt_root / "evidence_bundle.json",
            PaperEvidenceBundleV71,
        )
    except PaperDeliveryError as exc:
        mismatches.add(str(exc))
    current_s6 = workspace.current_gate("S6")
    status = "NEEDS_REVISION"
    if bundle is not None:
        if current_s6 != bundle.s6_gate_hash:
            mismatches.add("current S6 gate differs from paper evidence bundle")
            status = "STALE"
        else:
            try:
                rebuilt = build_evidence_bundle_v71(
                    workspace, requested_model=bundle.requested_model
                )
                if rebuilt != bundle:
                    mismatches.add(
                        "paper evidence bundle differs from current S0-S6"
                    )
                    status = "STALE"
            except PaperDeliveryError as exc:
                mismatches.add(str(exc))
                status = "STALE"
    receipt_path = paths.attempt_root / "delivery_receipt.json"
    receipt: PaperDeliveryReceiptV71 | None = None
    projection = _read_model(
        _safe_relative_path(workspace.root, CURRENT_PATH.as_posix()),
        PaperCurrentProjectionV71,
    )
    if not receipt_path.is_file():
        mismatches.add("delivery receipt is missing")
    else:
        try:
            receipt = _read_model(receipt_path, PaperDeliveryReceiptV71)
            request = _read_model(
                paths.attempt_root / "author_request.json",
                PaperAuthoringRequestV71,
            )
            if (
                projection.status != receipt.status
                or projection.delivery_hash != receipt.delivery_hash
            ):
                mismatches.add(
                    "current projection differs from delivery receipt"
                )
            if bundle and receipt.bundle_hash != bundle.bundle_hash:
                mismatches.add("delivery receipt binds another evidence bundle")
            if current_s6 and receipt.s6_gate_hash != current_s6:
                mismatches.add("delivery receipt binds another S6 gate")
                status = "STALE"
            audit = _read_model(
                paths.reviews_root / "content_audit.json",
                PaperContentAuditV71,
            )
            semantic_path = paths.reviews_root / "semantic_review.json"
            semantic = _read_model(
                semantic_path, PaperSemanticReviewV71
            )
            layout_path = paths.reviews_root / "layout_review.json"
            layout = _read_model(layout_path, PaperLayoutReviewV71)
            _verify_review_role_artifacts(paths, semantic, "semantic")
            _verify_review_role_artifacts(paths, layout, "layout")
            _require_native_codex_receipt(
                paths.attempt_root / "writer_transport_receipt.json",
                label="paper author transport",
                requested_model=request.requested_model,
            )
            _require_native_codex_receipt(
                paths.reviews_root / "semantic_transport_receipt.json",
                label="semantic reviewer transport",
                requested_model=request.requested_model,
            )
            _require_native_codex_receipt(
                paths.reviews_root / "layout_transport_receipt.json",
                label="layout reviewer transport",
                requested_model=request.requested_model,
            )
            if audit.audit_hash != receipt.content_audit_hash:
                mismatches.add("delivery receipt binds another content audit")
            if _sha256_file(semantic_path) != receipt.semantic_review_sha256:
                mismatches.add("semantic review hash mismatch")
            if _sha256_file(layout_path) != receipt.layout_review_sha256:
                mismatches.add("layout review hash mismatch")
            semantic_transport = (
                paths.reviews_root / "semantic_transport_receipt.json"
            )
            layout_transport = (
                paths.reviews_root / "layout_transport_receipt.json"
            )
            if (
                not semantic_transport.is_file()
                or _sha256_file(semantic_transport)
                != semantic.reviewer_transport_receipt_sha256
            ):
                mismatches.add("semantic reviewer transport receipt mismatch")
            if (
                not layout_transport.is_file()
                or _sha256_file(layout_transport)
                != layout.reviewer_transport_receipt_sha256
            ):
                mismatches.add("layout reviewer transport receipt mismatch")
            from .paper_renderer import (
                load_current_build_v71,
                verify_paper_build_v71,
            )

            build = load_current_build_v71(paths)
            build_verification = verify_paper_build_v71(workspace.root)
            if not build_verification.ok:
                mismatches.update(build_verification.mismatches)
            if build.build_hash != receipt.build_hash:
                mismatches.add("delivery receipt binds another build")
            if semantic.verdict != "APPROVE":
                mismatches.add("semantic review is not APPROVE")
            if layout.verdict != "APPROVE":
                mismatches.add("layout review is not APPROVE")
            if layout.build_hash != build.build_hash:
                mismatches.add("layout review binds another build")
            if layout.pages_reviewed != list(
                range(1, len(build.page_images) + 1)
            ):
                mismatches.add("layout review did not cover every page")
            pdf = _safe_relative_path(workspace.root, receipt.pdf_path)
            if not pdf.is_file() or _sha256_file(pdf) != receipt.pdf_sha256:
                mismatches.add("delivered PDF hash mismatch")
        except PaperDeliveryError as exc:
            mismatches.add(str(exc))
    if not mismatches and receipt is not None:
        status = receipt.status
    return PaperDeliveryVerificationV71(
        ok=not mismatches and status == "DRAFT_READY",
        status=status,
        bundle_hash=bundle.bundle_hash if bundle else None,
        current_s6_gate_hash=current_s6,
        mismatches=sorted(mismatches),
    )


__all__ = [
    "PaperAttemptPathsV71",
    "PaperDeliveryError",
    "assert_paper_attempt_open_v71",
    "audit_paper_content_v71",
    "build_evidence_bundle_v71",
    "current_paper_attempt_v71",
    "finalize_paper_delivery_v71",
    "load_validated_writer_packet_v71",
    "paper_writer_lock_v71",
    "prepare_paper_delivery_v71",
    "project_paper_status_v71",
    "record_layout_review_v71",
    "record_semantic_review_v71",
    "verify_paper_delivery_v71",
]
