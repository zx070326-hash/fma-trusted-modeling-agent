from __future__ import annotations

from pydantic import ValidationError
import pytest

from fma.v7_1.paper_schemas import (
    CitationRecordV71,
    PaperEvidenceBundleV71,
    PaperEvidenceItemV71,
    PaperLayoutReviewV71,
    PaperNumericTokenV71,
    PaperSemanticReviewV71,
)


HASH = "1" * 64
OTHER_HASH = "2" * 64
STAGE_HASHES = {f"S{index}": str(index + 1) * 64 for index in range(7)}


def _evidence_item() -> PaperEvidenceItemV71:
    return PaperEvidenceItemV71(
        evidence_id="E.s0.problem",
        stage="S0",
        relative_path="problem/contract.json",
        sha256=HASH,
        size_bytes=10,
        manifest_hash=OTHER_HASH,
        gate_hash=HASH,
        kind="problem",
    )


def _bundle_payload() -> dict[str, object]:
    return {
        "workspace_id": "fixture",
        "workspace_spec_hash": HASH,
        "objective": "Test a bounded publication projection.",
        "s6_gate_hash": OTHER_HASH,
        "current_gate_hashes": STAGE_HASHES,
        "evidence_items": [_evidence_item()],
        "numeric_tokens": [],
        "allowed_claim_types": [
            "comparison",
            "decision",
            "limitation",
            "method",
            "model_structure",
            "problem",
            "quantitative",
            "robustness",
        ],
        "forbidden_claim_types": [
            "causal",
            "global_optimality",
            "mechanistic_truth",
            "unsupported_extrapolation",
        ],
        "requested_model": "gpt-5.6-sol",
    }


def test_paper_bundle_authority_flags_are_fail_closed() -> None:
    payload = _bundle_payload()

    for field in (
        "served_model_attested",
        "scientific_qualification_granted",
        "real_world_action_authorized",
    ):
        with pytest.raises(ValidationError):
            PaperEvidenceBundleV71(**payload, **{field: True})

    bundle = PaperEvidenceBundleV71.seal(**payload)
    assert bundle.claim_scope == "publication_projection_only"
    assert bundle.served_model_attested is False
    assert bundle.scientific_qualification_granted is False
    assert bundle.real_world_action_authorized is False


def test_paper_bundle_rejects_numeric_token_for_unknown_evidence() -> None:
    payload = _bundle_payload()
    payload["numeric_tokens"] = [
        PaperNumericTokenV71(
            token_id="N.s0.unknown",
            evidence_id="E.s0.unknown",
            json_pointer="/forecast",
            value=0.25,
            display_value="0.25",
        )
    ]

    with pytest.raises(ValidationError, match="unknown evidence"):
        PaperEvidenceBundleV71(**payload)


def test_final_reviews_require_context_isolation() -> None:
    with pytest.raises(ValidationError, match="isolated reviewer"):
        PaperSemanticReviewV71(
            bundle_hash=HASH,
            content_audit_hash=OTHER_HASH,
            writer_context_ids=["writer-context"],
            reviewer_context_id="writer-context",
            context_isolated=True,
            reviewed_claim_ids=["claim.one"],
            verdict="APPROVE",
            findings=[],
            reviewer_request_sha256=OTHER_HASH,
            reviewer_draft_sha256=HASH,
            reviewer_transport_receipt_sha256=HASH,
            requested_model="gpt-5.6-sol",
        )

    with pytest.raises(ValidationError, match="isolated reviewer"):
        PaperLayoutReviewV71(
            build_hash=HASH,
            writer_context_ids=["writer-context"],
            reviewer_context_id="writer-context",
            context_isolated=True,
            pages_reviewed=[1],
            verdict="APPROVE",
            findings=[],
            reviewer_request_sha256=HASH,
            reviewer_draft_sha256=OTHER_HASH,
            reviewer_transport_receipt_sha256=OTHER_HASH,
            requested_model="gpt-5.6-sol",
        )


def test_snapshot_bound_citation_requires_typed_json_snapshot() -> None:
    with pytest.raises(ValidationError, match="must be JSON"):
        CitationRecordV71(
            citation_id="cite.fixture",
            title="Fixture citation metadata",
            authors=["F. Author"],
            year=2025,
            venue="Fixture Journal",
            doi="10.1234/fixture",
            source_snapshot_path="docs/citation.txt",
            source_snapshot_sha256=HASH,
            supports_claim_ids=["claim.fixture"],
            verification_status="SNAPSHOT_BOUND",
        )
