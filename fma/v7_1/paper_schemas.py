"""Strict schemas for the post-S6 V7.1 paper-delivery projection.

These artifacts describe publication readiness only.  They never grant
scientific qualification or real-world action authority.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, StrictFloat, StrictInt, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
PaperId = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")]
StageId = Literal["S0", "S1", "S2", "S3", "S4", "S5", "S6"]
EvidenceKind = Literal[
    "problem",
    "data",
    "model",
    "code",
    "validation",
    "result",
    "decision",
    "paper",
    "other",
]
AllowedClaimType = Literal[
    "problem",
    "method",
    "model_structure",
    "quantitative",
    "comparison",
    "robustness",
    "decision",
    "limitation",
]
ForbiddenClaimType = Literal[
    "causal",
    "mechanistic_truth",
    "global_optimality",
    "unsupported_extrapolation",
]
ClaimType = AllowedClaimType | ForbiddenClaimType
DeliveryStatus = Literal["DRAFT_READY", "NEEDS_REVISION", "HUMAN", "STALE"]

_STAGES = ("S0", "S1", "S2", "S3", "S4", "S5", "S6")


def _safe_relative(value: str, field: str) -> None:
    if not value or "\\" in value:
        raise ValueError(f"{field} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} is unsafe")


def _sorted_unique(values: list[str], field: str) -> None:
    if values != sorted(set(values)):
        raise ValueError(f"{field} must be sorted and unique")


class PaperEvidenceItemV71(StrictModel):
    evidence_id: PaperId
    stage: StageId
    relative_path: Annotated[str, Field(min_length=1, max_length=512)]
    sha256: Sha256
    size_bytes: Annotated[int, Field(ge=0, le=33_554_432)]
    manifest_hash: Sha256
    gate_hash: Sha256
    kind: EvidenceKind

    @model_validator(mode="after")
    def validate_item(self) -> "PaperEvidenceItemV71":
        _safe_relative(self.relative_path, "relative_path")
        return self


class PaperNumericTokenV71(StrictModel):
    token_id: PaperId
    evidence_id: PaperId
    json_pointer: Annotated[str, Field(min_length=1, max_length=1024)]
    value: StrictInt | Annotated[StrictFloat, Field(allow_inf_nan=False)]
    display_value: Annotated[str, Field(min_length=1, max_length=64)]

    @model_validator(mode="after")
    def validate_token(self) -> "PaperNumericTokenV71":
        if not self.json_pointer.startswith("/"):
            raise ValueError("json_pointer must start with '/'")
        if isinstance(self.value, bool):
            raise ValueError("numeric token value may not be boolean")
        return self


class PaperEvidenceBundleV71(StrictModel):
    schema_version: Literal["7.1-paper-evidence-bundle"] = (
        "7.1-paper-evidence-bundle"
    )
    workspace_id: PaperId
    workspace_spec_hash: Sha256
    objective: Annotated[str, Field(min_length=5, max_length=4000)]
    s6_gate_hash: Sha256
    current_gate_hashes: dict[StageId, Sha256]
    evidence_items: Annotated[list[PaperEvidenceItemV71], Field(min_length=1)]
    numeric_tokens: list[PaperNumericTokenV71] = Field(default_factory=list)
    allowed_claim_types: list[AllowedClaimType]
    forbidden_claim_types: list[ForbiddenClaimType]
    requested_model: Annotated[str, Field(min_length=1, max_length=128)]
    served_model_attested: Literal[False] = False
    claim_scope: Literal["publication_projection_only"] = (
        "publication_projection_only"
    )
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    bundle_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bundle(self) -> "PaperEvidenceBundleV71":
        if tuple(self.current_gate_hashes) != _STAGES:
            raise ValueError("current_gate_hashes must contain ordered S0-S6")
        evidence_ids = [item.evidence_id for item in self.evidence_items]
        _sorted_unique(evidence_ids, "evidence_items")
        token_ids = [item.token_id for item in self.numeric_tokens]
        _sorted_unique(token_ids, "numeric_tokens")
        known = set(evidence_ids)
        if any(item.evidence_id not in known for item in self.numeric_tokens):
            raise ValueError("numeric token references unknown evidence")
        _sorted_unique(self.allowed_claim_types, "allowed_claim_types")
        _sorted_unique(self.forbidden_claim_types, "forbidden_claim_types")
        if self.bundle_hash and self.bundle_hash != self.content_hash():
            raise ValueError("bundle_hash does not match evidence bundle")
        return self

    def content_hash(self) -> str:
        return sha256_value(
            self.model_dump(mode="json", exclude={"bundle_hash"})
        )

    @classmethod
    def seal(cls, **data: object) -> "PaperEvidenceBundleV71":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"bundle_hash"}),
            bundle_hash=draft.content_hash(),
        )


class PaperAuthoringRequestV71(StrictModel):
    schema_version: Literal["7.1-paper-authoring-request"] = (
        "7.1-paper-authoring-request"
    )
    bundle_hash: Sha256
    language: Literal["zh", "en"] = "zh"
    venue_profile: Literal[
        "academic_article", "modeling_contest", "technical_report"
    ] = "academic_article"
    requested_model: Annotated[str, Field(min_length=1, max_length=128)] = (
        "gpt-5.6-sol"
    )
    title_hint: Annotated[str, Field(min_length=3, max_length=300)]
    authors: Annotated[list[str], Field(min_length=1, max_length=20)]
    max_pages: Annotated[int, Field(ge=2, le=100)] = 24
    max_revision_rounds: Annotated[int, Field(ge=0, le=5)] = 2
    served_model_attested: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    request_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_request(self) -> "PaperAuthoringRequestV71":
        if any(not author.strip() for author in self.authors):
            raise ValueError("authors may not be blank")
        if self.request_hash and self.request_hash != self.content_hash():
            raise ValueError("request_hash does not match authoring request")
        return self

    def content_hash(self) -> str:
        return sha256_value(
            self.model_dump(mode="json", exclude={"request_hash"})
        )

    @classmethod
    def seal(cls, **data: object) -> "PaperAuthoringRequestV71":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"request_hash"}),
            request_hash=draft.content_hash(),
        )


class PaperWriterEvidenceV71(StrictModel):
    evidence_id: PaperId
    stage: StageId
    relative_path: Annotated[str, Field(min_length=1, max_length=512)]
    sha256: Sha256
    content: Annotated[str, Field(max_length=262_144)]

    @model_validator(mode="after")
    def validate_evidence(self) -> "PaperWriterEvidenceV71":
        _safe_relative(self.relative_path, "relative_path")
        if (
            hashlib.sha256(self.content.encode("utf-8")).hexdigest()
            != self.sha256
        ):
            raise ValueError("writer evidence content hash mismatch")
        return self


class PaperWriterOmissionV71(StrictModel):
    evidence_id: PaperId
    reason: Literal[
        "duplicate_projection",
        "non_text_artifact",
        "file_budget",
        "not_utf8",
        "packet_budget",
        "upstream_publication_projection",
    ]


class PaperWriterPacketV71(StrictModel):
    schema_version: Literal["7.1-paper-writer-packet"] = (
        "7.1-paper-writer-packet"
    )
    bundle_hash: Sha256
    request_hash: Sha256
    objective: Annotated[str, Field(min_length=5, max_length=4000)]
    allowed_claim_types: list[AllowedClaimType]
    forbidden_claim_types: list[ForbiddenClaimType]
    numeric_tokens: list[PaperNumericTokenV71]
    included_evidence: list[PaperWriterEvidenceV71]
    omitted_evidence: list[PaperWriterOmissionV71]
    narrative_questions: Annotated[list[str], Field(min_length=6, max_length=6)]
    required_macros: Annotated[list[str], Field(min_length=6, max_length=6)]
    required_section_ids: Annotated[list[PaperId], Field(min_length=1)]
    authority_denials: Annotated[list[str], Field(min_length=4)]
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    packet_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_packet(self) -> "PaperWriterPacketV71":
        _sorted_unique(self.allowed_claim_types, "allowed_claim_types")
        _sorted_unique(self.forbidden_claim_types, "forbidden_claim_types")
        _sorted_unique(
            [item.token_id for item in self.numeric_tokens], "numeric_tokens"
        )
        _sorted_unique(
            [item.evidence_id for item in self.included_evidence],
            "included_evidence",
        )
        _sorted_unique(
            [item.evidence_id for item in self.omitted_evidence],
            "omitted_evidence",
        )
        _sorted_unique(self.required_section_ids, "required_section_ids")
        if set(item.evidence_id for item in self.included_evidence) & set(
            item.evidence_id for item in self.omitted_evidence
        ):
            raise ValueError("writer packet evidence may not be included and omitted")
        if self.packet_hash and self.packet_hash != self.content_hash():
            raise ValueError("packet_hash does not match writer packet")
        return self

    def content_hash(self) -> str:
        return sha256_value(
            self.model_dump(mode="json", exclude={"packet_hash"})
        )

    @classmethod
    def seal(cls, **data: object) -> "PaperWriterPacketV71":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"packet_hash"}),
            packet_hash=draft.content_hash(),
        )


class PaperCurrentProjectionV71(StrictModel):
    schema_version: Literal["7.1-paper-current-projection"] = (
        "7.1-paper-current-projection"
    )
    attempt_id: Annotated[
        str, Field(pattern=r"^paper-[0-9a-f]{16}-[0-9a-f]{12}$")
    ]
    bundle_hash: Sha256
    request_hash: Sha256
    status: DeliveryStatus
    delivery_hash: Sha256 | None = None
    projection_only: Literal[True] = True
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_projection(self) -> "PaperCurrentProjectionV71":
        if self.status == "DRAFT_READY" and self.delivery_hash is None:
            raise ValueError("DRAFT_READY projection requires delivery_hash")
        if self.status != "DRAFT_READY" and self.delivery_hash is not None:
            raise ValueError(
                "only DRAFT_READY projection may bind a delivery_hash"
            )
        return self


class PaperMetadataV71(StrictModel):
    schema_version: Literal["7.1-paper-metadata"] = "7.1-paper-metadata"
    bundle_hash: Sha256
    title: Annotated[str, Field(min_length=3, max_length=300)]
    authors: Annotated[list[str], Field(min_length=1, max_length=20)]
    language: Literal["zh", "en"] = "zh"
    venue_profile: Literal[
        "academic_article", "modeling_contest", "technical_report"
    ] = "academic_article"
    requested_model: Annotated[str, Field(min_length=1, max_length=128)]
    served_model_attested: Literal[False] = False
    writer_context_ids: Annotated[list[PaperId], Field(min_length=1)]
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_metadata(self) -> "PaperMetadataV71":
        if any(not item.strip() for item in self.authors):
            raise ValueError("authors may not be blank")
        _sorted_unique(self.writer_context_ids, "writer_context_ids")
        return self


class PaperClaimV71(StrictModel):
    claim_id: PaperId
    claim_type: ClaimType
    statement: Annotated[
        str,
        Field(
            min_length=5,
            max_length=2000,
            description=(
                "Plain-text semantic summary with no digits and no TeX. "
                "Bind exact values only through numeric_token_ids; render them "
                "with FMAValue only in abstract_tex or body_tex."
            ),
        ),
    ]
    scope_qualifier: Annotated[
        str,
        Field(
            min_length=3,
            max_length=1000,
            description=(
                "Plain-text scope with no digits and no TeX. Bind any exact "
                "value through numeric_token_ids and render it only in the "
                "manuscript."
            ),
        ),
    ]
    evidence_ids: Annotated[list[PaperId], Field(min_length=1)]
    numeric_token_ids: list[PaperId] = Field(default_factory=list)
    citation_ids: list[PaperId] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_claim(self) -> "PaperClaimV71":
        _sorted_unique(self.evidence_ids, "evidence_ids")
        _sorted_unique(self.numeric_token_ids, "numeric_token_ids")
        _sorted_unique(self.citation_ids, "citation_ids")
        if self.claim_type == "quantitative" and not self.numeric_token_ids:
            raise ValueError("quantitative claim requires numeric_token_ids")
        return self


class PaperClaimLedgerV71(StrictModel):
    schema_version: Literal["7.1-paper-claim-ledger"] = (
        "7.1-paper-claim-ledger"
    )
    bundle_hash: Sha256
    claims: Annotated[list[PaperClaimV71], Field(min_length=1)]
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_ledger(self) -> "PaperClaimLedgerV71":
        _sorted_unique([item.claim_id for item in self.claims], "claims")
        return self


class CitationRecordV71(StrictModel):
    citation_id: PaperId
    title: Annotated[str, Field(min_length=3, max_length=1000)]
    authors: Annotated[list[str], Field(min_length=1, max_length=100)]
    year: Annotated[int, Field(ge=1000, le=2200)]
    venue: Annotated[str, Field(min_length=1, max_length=500)]
    doi: Annotated[str, Field(min_length=3, max_length=300)] | None = None
    url: Annotated[str, Field(min_length=8, max_length=2000)] | None = None
    source_snapshot_path: Annotated[str, Field(max_length=512)] | None = None
    source_snapshot_sha256: Sha256 | None = None
    supports_claim_ids: Annotated[list[PaperId], Field(min_length=1)]
    verification_status: Literal["SNAPSHOT_BOUND", "HUMAN"]

    @model_validator(mode="after")
    def validate_citation(self) -> "CitationRecordV71":
        if not self.doi and not self.url:
            raise ValueError("citation requires doi or url")
        if self.doi and not re.fullmatch(r"10\.\d{4,9}/\S+", self.doi):
            raise ValueError("citation DOI is malformed")
        if self.url and not self.url.startswith(("https://", "http://")):
            raise ValueError("citation URL must be HTTP(S)")
        if (self.source_snapshot_path is None) != (
            self.source_snapshot_sha256 is None
        ):
            raise ValueError("citation snapshot path and hash must coexist")
        if self.verification_status == "SNAPSHOT_BOUND" and not (
            self.source_snapshot_path and self.source_snapshot_sha256
        ):
            raise ValueError(
                "SNAPSHOT_BOUND citation requires a frozen source snapshot"
            )
        if self.source_snapshot_path:
            _safe_relative(self.source_snapshot_path, "source_snapshot_path")
            if PurePosixPath(self.source_snapshot_path).suffix.lower() != ".json":
                raise ValueError("citation source snapshot must be JSON")
        _sorted_unique(self.supports_claim_ids, "supports_claim_ids")
        return self


class CitationManifestV71(StrictModel):
    schema_version: Literal["7.1-paper-citation-manifest"] = (
        "7.1-paper-citation-manifest"
    )
    bundle_hash: Sha256
    citations: list[CitationRecordV71] = Field(default_factory=list)
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_manifest(self) -> "CitationManifestV71":
        _sorted_unique(
            [item.citation_id for item in self.citations], "citations"
        )
        return self


class FigureBindingV71(StrictModel):
    figure_id: PaperId
    artifact_path: Annotated[str, Field(min_length=1, max_length=512)]
    artifact_sha256: Sha256
    caption: Annotated[str, Field(min_length=3, max_length=2000)]
    alt_text: Annotated[str, Field(min_length=3, max_length=2000)]
    evidence_ids: Annotated[list[PaperId], Field(min_length=1)]
    claim_ids: Annotated[list[PaperId], Field(min_length=1)]
    generator_path: Annotated[str, Field(max_length=512)] | None = None
    generator_sha256: Sha256 | None = None
    width_fraction: Annotated[float, Field(ge=0.2, le=1.0)] = 0.9

    @model_validator(mode="after")
    def validate_figure(self) -> "FigureBindingV71":
        _safe_relative(self.artifact_path, "artifact_path")
        if (self.generator_path is None) != (self.generator_sha256 is None):
            raise ValueError("figure generator path and hash must coexist")
        if self.generator_path:
            _safe_relative(self.generator_path, "generator_path")
        _sorted_unique(self.evidence_ids, "evidence_ids")
        _sorted_unique(self.claim_ids, "claim_ids")
        return self


class FigureManifestV71(StrictModel):
    schema_version: Literal["7.1-paper-figure-manifest"] = (
        "7.1-paper-figure-manifest"
    )
    bundle_hash: Sha256
    figures: list[FigureBindingV71] = Field(default_factory=list)
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_manifest(self) -> "FigureManifestV71":
        _sorted_unique([item.figure_id for item in self.figures], "figures")
        return self


class TableBindingV71(StrictModel):
    table_id: PaperId
    csv_path: Annotated[str, Field(min_length=1, max_length=512)]
    csv_sha256: Sha256
    caption: Annotated[str, Field(min_length=3, max_length=2000)]
    evidence_ids: Annotated[list[PaperId], Field(min_length=1)]
    claim_ids: Annotated[list[PaperId], Field(min_length=1)]
    max_rows: Annotated[int, Field(ge=1, le=100)] = 30
    max_columns: Annotated[int, Field(ge=1, le=12)] = 8

    @model_validator(mode="after")
    def validate_table(self) -> "TableBindingV71":
        _safe_relative(self.csv_path, "csv_path")
        _sorted_unique(self.evidence_ids, "evidence_ids")
        _sorted_unique(self.claim_ids, "claim_ids")
        return self


class TableManifestV71(StrictModel):
    schema_version: Literal["7.1-paper-table-manifest"] = (
        "7.1-paper-table-manifest"
    )
    bundle_hash: Sha256
    tables: list[TableBindingV71] = Field(default_factory=list)
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_manifest(self) -> "TableManifestV71":
        _sorted_unique([item.table_id for item in self.tables], "tables")
        return self


class PaperContentAuditV71(StrictModel):
    schema_version: Literal["7.1-paper-content-audit"] = (
        "7.1-paper-content-audit"
    )
    bundle_hash: Sha256
    attempt_id: PaperId
    status: Literal["PASS", "FAIL"]
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata_sha256: Sha256
    author_request_sha256: Sha256
    writer_packet_sha256: Sha256
    abstract_sha256: Sha256
    body_sha256: Sha256
    claim_ledger_sha256: Sha256
    citation_manifest_sha256: Sha256
    figure_manifest_sha256: Sha256
    table_manifest_sha256: Sha256
    writer_role_request_sha256: Sha256
    writer_transport_receipt_sha256: Sha256
    writer_output_sha256: Sha256
    checked_at: datetime
    audit_scope: Literal["content_and_binding_only"] = "content_and_binding_only"
    scientific_correctness_established: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    audit_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_audit(self) -> "PaperContentAuditV71":
        _sorted_unique(self.errors, "errors")
        _sorted_unique(self.warnings, "warnings")
        if (self.status == "PASS") != (not self.errors):
            raise ValueError("content audit status and errors disagree")
        if self.audit_hash and self.audit_hash != self.content_hash():
            raise ValueError("audit_hash does not match content audit")
        return self

    def content_hash(self) -> str:
        return sha256_value(self.model_dump(mode="json", exclude={"audit_hash"}))

    @classmethod
    def seal(cls, **data: object) -> "PaperContentAuditV71":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"audit_hash"}),
            audit_hash=draft.content_hash(),
        )


class PaperReviewFindingV71(StrictModel):
    finding_id: PaperId
    severity: Literal["ERROR", "WARNING", "HUMAN"]
    claim_ids: list[PaperId] = Field(default_factory=list)
    evidence_ids: list[PaperId] = Field(default_factory=list)
    message: Annotated[str, Field(min_length=5, max_length=2000)]

    @model_validator(mode="after")
    def validate_finding(self) -> "PaperReviewFindingV71":
        _sorted_unique(self.claim_ids, "claim_ids")
        _sorted_unique(self.evidence_ids, "evidence_ids")
        return self


class PaperSemanticReviewV71(StrictModel):
    schema_version: Literal["7.1-paper-semantic-review"] = (
        "7.1-paper-semantic-review"
    )
    bundle_hash: Sha256
    content_audit_hash: Sha256
    writer_context_ids: Annotated[list[PaperId], Field(min_length=1)]
    reviewer_context_id: PaperId
    context_isolated: bool
    reviewed_claim_ids: Annotated[list[PaperId], Field(min_length=1)]
    verdict: Literal["APPROVE", "REJECT", "HUMAN"]
    findings: list[PaperReviewFindingV71] = Field(default_factory=list)
    reviewer_request_sha256: Sha256
    reviewer_draft_sha256: Sha256
    reviewer_transport_receipt_sha256: Sha256
    requested_model: Annotated[str, Field(min_length=1, max_length=128)]
    served_model_attested: Literal[False] = False
    review_scope: Literal["paper_claim_support_and_consistency"] = (
        "paper_claim_support_and_consistency"
    )
    scientific_correctness_established: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_review(self) -> "PaperSemanticReviewV71":
        _sorted_unique(self.writer_context_ids, "writer_context_ids")
        _sorted_unique(self.reviewed_claim_ids, "reviewed_claim_ids")
        _sorted_unique(
            [item.finding_id for item in self.findings], "findings"
        )
        if self.context_isolated and (
            self.reviewer_context_id in self.writer_context_ids
        ):
            raise ValueError("isolated reviewer must differ from every writer")
        if self.verdict == "APPROVE" and (
            not self.context_isolated or self.findings
        ):
            raise ValueError(
                "APPROVE requires isolated context and no findings"
            )
        if self.verdict == "REJECT" and not any(
            item.severity == "ERROR" for item in self.findings
        ):
            raise ValueError("REJECT requires an ERROR finding")
        if self.verdict == "HUMAN" and not any(
            item.severity == "HUMAN" for item in self.findings
        ):
            raise ValueError("HUMAN requires a HUMAN finding")
        return self


class PaperLayoutReviewV71(StrictModel):
    schema_version: Literal["7.1-paper-layout-review"] = (
        "7.1-paper-layout-review"
    )
    build_hash: Sha256
    writer_context_ids: Annotated[list[PaperId], Field(min_length=1)]
    reviewer_context_id: PaperId
    context_isolated: bool
    pages_reviewed: Annotated[list[int], Field(min_length=1)]
    verdict: Literal["APPROVE", "REJECT", "HUMAN"]
    findings: list[str] = Field(default_factory=list)
    reviewer_request_sha256: Sha256
    reviewer_draft_sha256: Sha256
    reviewer_transport_receipt_sha256: Sha256
    requested_model: Annotated[str, Field(min_length=1, max_length=128)]
    served_model_attested: Literal[False] = False
    review_scope: Literal["layout_and_readability_only"] = (
        "layout_and_readability_only"
    )
    scientific_correctness_established: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_review(self) -> "PaperLayoutReviewV71":
        _sorted_unique(self.writer_context_ids, "writer_context_ids")
        if self.pages_reviewed != sorted(set(self.pages_reviewed)):
            raise ValueError("pages_reviewed must be sorted and unique")
        if any(page < 1 for page in self.pages_reviewed):
            raise ValueError("pages_reviewed must be positive")
        _sorted_unique(self.findings, "findings")
        if self.context_isolated and (
            self.reviewer_context_id in self.writer_context_ids
        ):
            raise ValueError("isolated reviewer must differ from every writer")
        if self.verdict == "APPROVE" and (
            not self.context_isolated or self.findings
        ):
            raise ValueError(
                "APPROVE requires isolated context and no findings"
            )
        if self.verdict != "APPROVE" and not self.findings:
            raise ValueError("REJECT/HUMAN layout review requires findings")
        return self


class PaperToolIdentityV71(StrictModel):
    tool: Literal["xelatex", "pdfinfo", "pdftoppm"]
    resolved_path: Annotated[str, Field(min_length=1, max_length=2048)]
    binary_sha256: Sha256
    version: Annotated[str, Field(min_length=1, max_length=500)]
    argv_hash: Sha256

    @model_validator(mode="after")
    def validate_identity(self) -> "PaperToolIdentityV71":
        if any(ord(character) < 32 for character in self.resolved_path):
            raise ValueError("tool resolved_path contains control characters")
        return self


class PaperBuildReceiptV71(StrictModel):
    schema_version: Literal["7.1-paper-build-receipt"] = (
        "7.1-paper-build-receipt"
    )
    bundle_hash: Sha256
    content_audit_hash: Sha256
    semantic_review_sha256: Sha256
    template_sha256: Sha256
    author_request_sha256: Sha256
    writer_packet_sha256: Sha256
    metadata_sha256: Sha256
    abstract_sha256: Sha256
    body_sha256: Sha256
    claim_ledger_sha256: Sha256
    citation_manifest_sha256: Sha256
    figure_manifest_sha256: Sha256
    table_manifest_sha256: Sha256
    writer_role_request_sha256: Sha256
    writer_transport_receipt_sha256: Sha256
    writer_output_sha256: Sha256
    generated_tex_path: Annotated[str, Field(min_length=1, max_length=512)]
    generated_tex_sha256: Sha256
    pdf_path: Annotated[str, Field(min_length=1, max_length=512)]
    pdf_sha256: Sha256
    compiler_command: Annotated[list[str], Field(min_length=1)]
    compiler_version: Annotated[str, Field(min_length=1, max_length=500)]
    xelatex_identity: PaperToolIdentityV71
    pdfinfo_identity: PaperToolIdentityV71
    pdftoppm_identity: PaperToolIdentityV71
    environment_hash: Sha256
    compiler_log_sha256: Sha256
    page_images: dict[str, Sha256]
    layout_lint: list[str] = Field(default_factory=list)
    built_at: datetime
    build_scope: Literal["publication_build_only"] = "publication_build_only"
    scientific_correctness_established: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    build_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "PaperBuildReceiptV71":
        _safe_relative(self.generated_tex_path, "generated_tex_path")
        _safe_relative(self.pdf_path, "pdf_path")
        if not self.page_images:
            raise ValueError("paper build requires rendered page images")
        if list(self.page_images) != sorted(self.page_images):
            raise ValueError("page_images must be sorted")
        for path in self.page_images:
            _safe_relative(path, "page image path")
        _sorted_unique(self.layout_lint, "layout_lint")
        if (
            self.xelatex_identity.tool != "xelatex"
            or self.pdfinfo_identity.tool != "pdfinfo"
            or self.pdftoppm_identity.tool != "pdftoppm"
        ):
            raise ValueError("paper build tool identities are assigned incorrectly")
        if self.compiler_command[0] != self.xelatex_identity.resolved_path:
            raise ValueError("compiler command does not use the bound XeLaTeX")
        if self.compiler_version != self.xelatex_identity.version:
            raise ValueError("compiler version differs from XeLaTeX identity")
        if self.build_hash and self.build_hash != self.content_hash():
            raise ValueError("build_hash does not match build receipt")
        return self

    def content_hash(self) -> str:
        return sha256_value(self.model_dump(mode="json", exclude={"build_hash"}))

    @classmethod
    def seal(cls, **data: object) -> "PaperBuildReceiptV71":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"build_hash"}),
            build_hash=draft.content_hash(),
        )


class PaperDeliveryReceiptV71(StrictModel):
    schema_version: Literal["7.1-paper-delivery-receipt"] = (
        "7.1-paper-delivery-receipt"
    )
    status: DeliveryStatus
    bundle_hash: Sha256
    s6_gate_hash: Sha256
    content_audit_hash: Sha256
    semantic_review_sha256: Sha256
    build_hash: Sha256
    layout_review_sha256: Sha256
    pdf_path: Annotated[str, Field(min_length=1, max_length=512)]
    pdf_sha256: Sha256
    created_at: datetime
    claim_scope: Literal["publication_projection_only"] = (
        "publication_projection_only"
    )
    scientific_correctness_established: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    delivery_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_delivery(self) -> "PaperDeliveryReceiptV71":
        _safe_relative(self.pdf_path, "pdf_path")
        if self.delivery_hash and self.delivery_hash != self.content_hash():
            raise ValueError("delivery_hash does not match delivery receipt")
        return self

    def content_hash(self) -> str:
        return sha256_value(
            self.model_dump(mode="json", exclude={"delivery_hash"})
        )

    @classmethod
    def seal(cls, **data: object) -> "PaperDeliveryReceiptV71":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"delivery_hash"}),
            delivery_hash=draft.content_hash(),
        )


class PaperDeliveryVerificationV71(StrictModel):
    schema_version: Literal["7.1-paper-delivery-verification"] = (
        "7.1-paper-delivery-verification"
    )
    ok: bool
    status: DeliveryStatus
    bundle_hash: Sha256 | None = None
    current_s6_gate_hash: Sha256 | None = None
    mismatches: list[str] = Field(default_factory=list)
    scientific_correctness_established: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_verification(self) -> "PaperDeliveryVerificationV71":
        _sorted_unique(self.mismatches, "mismatches")
        if self.ok and self.mismatches:
            raise ValueError("successful verification cannot contain mismatches")
        if not self.ok and not self.mismatches:
            raise ValueError("failed verification requires mismatches")
        return self
