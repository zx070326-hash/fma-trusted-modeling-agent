from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from fma.hashing import sha256_value
from fma.v6.recovery_kernel import (
    FailureDiagnosisV60,
    RecoveryPlanV60,
    RecoveryTransitionReceiptV60,
)
from fma.v6.s1_review_recovery import (
    S1_BOUNDED_REPAIR_CONTEXT_PATH_V67,
    S1_FORMALIZATION_FAILURE_CODE_V67,
    S1_FORMALIZATION_REJECTION_EVIDENCE_PATH_V67,
    S1BoundedRepairContextV67,
    S1FormalizationRejectionEvidenceV67,
    S1FormalizationRejectionHandoffV67,
    build_s1_bounded_repair_context_v67,
    build_s1_formalization_rejection_evidence_v67,
    build_s1_formalization_rejection_handoff_v67,
    normalize_s1_review_findings_v67,
    s1_recovery_evidence_refs_v67,
    s1_review_failure_signature_v67,
    s1_reviewer_finding_signature_v67,
)


WORKSPACE_HASH = "a" * 64
S0_GATE_HASH = "b" * 64
PROTOCOL_HASH = "c" * 64
REVIEWER_RECEIPT_HASH = "d" * 64


def _recovery_chain(
    *,
    failure_code: str = S1_FORMALIZATION_FAILURE_CODE_V67,
    successor_attempt: int = 2,
):
    failure_signature = sha256_value(
        {
            "failed_stage": "S1",
            "category": "review_rejection",
            "failure_code": failure_code,
        }
    )
    diagnosis = FailureDiagnosisV60.seal(
        workspace_spec_hash=WORKSPACE_HASH,
        failed_stage="S1",
        category="review_rejection",
        failure_code=failure_code,
        evidence_refs=["1" * 64],
        failure_signature=failure_signature,
        earliest_affected_stage="S1",
        retryable=True,
        candidate_change_required=False,
        data_change_required=False,
        holdout_exposed=False,
        private_evidence_used=False,
    )
    plan = RecoveryPlanV60.seal(
        plan_id="s1-review-patch",
        diagnosis_hash=diagnosis.diagnosis_hash,
        failure_signature=diagnosis.failure_signature,
        action="PATCH",
        revoke_from="S1",
        automatic_execution_permitted=True,
        expected_information_gain=0.5,
        forbidden_evidence_refs=[],
        stop_conditions=[
            "Stop after one successor repair or a repeated failure signature."
        ],
        attempt_budget_remaining=1,
    )
    receipt = RecoveryTransitionReceiptV60.seal(
        diagnosis_hash=diagnosis.diagnosis_hash,
        plan_hash=plan.plan_hash,
        before_graph_state_hash="2" * 64,
        after_graph_state_hash="3" * 64,
        status="ATTEMPT_CREATED",
        failed_stage="S1",
        revoke_from="S1",
        predecessor_attempt=1,
        successor_attempt=successor_attempt,
        affected_node_hashes=["4" * 64],
        quarantined_file_hashes={
            "docs/model_spec.json": "5" * 64,
        },
    )
    return diagnosis, plan, receipt


def _evidence(
    findings: list[str] | None = None,
) -> S1FormalizationRejectionEvidenceV67:
    diagnosis, plan, receipt = _recovery_chain()
    return build_s1_formalization_rejection_evidence_v67(
        workspace_spec_hash=WORKSPACE_HASH,
        s0_gate_hash=S0_GATE_HASH,
        predata_protocol_hash=PROTOCOL_HASH,
        reviewer_receipt_hash=REVIEWER_RECEIPT_HASH,
        findings=findings
        or [
            "Candidate family semantics contradict the frozen protocol.",
            "  The selected equation loosens the registered state boundary.  ",
        ],
        diagnosis=diagnosis,
        plan=plan,
        recovery_receipt=receipt,
    )


def _handoff(
    *,
    existing_repair_context_hash: str | None = None,
) -> S1FormalizationRejectionHandoffV67:
    return build_s1_formalization_rejection_handoff_v67(
        workspace_spec_hash=WORKSPACE_HASH,
        s0_gate_hash=S0_GATE_HASH,
        predata_protocol_hash=PROTOCOL_HASH,
        reviewer_receipt_hash=REVIEWER_RECEIPT_HASH,
        findings=[
            "Candidate family semantics contradict the frozen protocol."
        ],
        predecessor_attempt=2 if existing_repair_context_hash else 1,
        existing_repair_context_hash=existing_repair_context_hash,
    )


def test_write_ahead_handoff_freezes_recovery_inputs_before_mutation() -> None:
    handoff = _handoff()
    handoff.assert_sealed()

    assert handoff.recovery_disposition == "bounded_patch"
    assert handoff.expected_information_gain == 0.5
    assert handoff.existing_repair_context_hash is None
    assert handoff.predecessor_attempt == 1
    assert handoff.base_evidence_refs == sorted(
        {
            WORKSPACE_HASH,
            S0_GATE_HASH,
            PROTOCOL_HASH,
            REVIEWER_RECEIPT_HASH,
            handoff.reviewer_finding_signature,
        }
    )
    recovery_refs = s1_recovery_evidence_refs_v67(
        handoff,
        handoff_artifact_hash="e" * 64,
    )
    assert recovery_refs == sorted(
        {*handoff.base_evidence_refs, "e" * 64}
    )
    assert handoff.observation_data_included is False
    assert handoff.private_evidence_used is False
    assert handoff.holdout_exposed is False
    assert handoff.scientific_qualification_granted is False
    assert handoff.real_world_action_authorized is False


def test_repeated_rejection_handoff_preserves_terminal_disposition() -> None:
    handoff = _handoff(existing_repair_context_hash="f" * 64)
    handoff.assert_sealed()

    assert handoff.recovery_disposition == "terminal_human"
    assert handoff.expected_information_gain == 0.0
    assert handoff.existing_repair_context_hash == "f" * 64
    assert handoff.predecessor_attempt == 2


def test_handoff_tampering_or_unbound_artifact_hash_fails_closed() -> None:
    handoff = _handoff()
    payload = handoff.model_dump(mode="json")
    payload["expected_information_gain"] = 0.0
    with pytest.raises(ValidationError, match="bounded patch"):
        S1FormalizationRejectionHandoffV67.model_validate(payload)

    with pytest.raises(ValueError, match="not SHA-256"):
        s1_recovery_evidence_refs_v67(
            handoff,
            handoff_artifact_hash="not-a-hash",
        )


def test_rejection_evidence_binds_review_and_executed_attempt_lineage() -> None:
    evidence = _evidence()
    evidence.assert_sealed()

    assert evidence.failure_code == (
        "s1_formalization_review_rejected"
    )
    assert evidence.failure_signature == s1_review_failure_signature_v67()
    assert evidence.workspace_spec_hash == WORKSPACE_HASH
    assert evidence.s0_gate_hash == S0_GATE_HASH
    assert evidence.predata_protocol_hash == PROTOCOL_HASH
    assert evidence.reviewer_receipt_hash == REVIEWER_RECEIPT_HASH
    assert evidence.predecessor_attempt == 1
    assert evidence.successor_attempt == 2
    assert len(evidence.diagnosis_hash) == 64
    assert evidence.diagnosis_evidence_refs == ["1" * 64]
    assert len(evidence.recovery_plan_hash) == 64
    assert len(evidence.recovery_receipt_hash) == 64
    assert evidence.rollback_root_selected_by_evidence is False
    assert evidence.gate_certificate_issued is False
    assert evidence.scientific_failure_established is False
    assert evidence.scientific_qualification_granted is False
    assert evidence.real_world_action_authorized is False
    assert S1_FORMALIZATION_REJECTION_EVIDENCE_PATH_V67.startswith(
        "checks/"
    )


def test_free_text_changes_finding_commitment_but_not_failure_class() -> None:
    first = _evidence(
        ["Candidate contradicts the exact registered family."]
    )
    second = _evidence(
        ["The equation is incompatible with the frozen family semantics."]
    )

    assert first.reviewer_finding_signature != (
        second.reviewer_finding_signature
    )
    assert first.evidence_hash != second.evidence_hash
    assert first.failure_signature == second.failure_signature
    assert first.failure_signature == sha256_value(
        {
            "failed_stage": "S1",
            "category": "review_rejection",
            "failure_code": "s1_formalization_review_rejected",
        }
    )
    assert "Candidate contradicts" not in json.dumps(
        {
            "failed_stage": "S1",
            "category": "review_rejection",
            "failure_code": S1_FORMALIZATION_FAILURE_CODE_V67,
        }
    )


def test_bounded_repair_context_discloses_only_necessary_findings() -> None:
    evidence = _evidence()
    context = build_s1_bounded_repair_context_v67(evidence)
    context.assert_sealed()
    payload = context.model_dump(mode="json")

    assert context.source_rejection_evidence_hash == evidence.evidence_hash
    assert context.recovery_receipt_hash == evidence.recovery_receipt_hash
    assert context.predecessor_attempt == 1
    assert context.successor_attempt == 2
    assert context.maximum_model_repair_attempts == 1
    assert context.remaining_model_repair_attempts == 1
    assert [item.normalized_finding for item in context.findings] == [
        item.normalized_finding for item in evidence.findings
    ]
    assert context.protocol_change_permitted is False
    assert context.adapter_change_permitted is False
    assert context.threshold_change_permitted is False
    assert context.observation_data_included is False
    assert context.reviewer_rationale_included is False
    assert context.reviewer_uncertainties_included is False
    assert context.private_evidence_included is False
    assert context.holdout_evidence_included is False
    assert context.rollback_root_included is False
    assert context.gate_signing_authority is False
    assert context.scientific_qualification_granted is False
    assert context.real_world_action_authorized is False
    assert not {
        "reviewer_rationale",
        "reviewer_uncertainties",
        "revoke_from",
        "rollback_root",
        "observation_values",
    } & set(payload)
    assert S1_BOUNDED_REPAIR_CONTEXT_PATH_V67.startswith("checks/")


def test_tampering_fails_closed_for_evidence_and_repair_projection() -> None:
    evidence = _evidence()
    evidence_payload = evidence.model_dump(mode="json")
    evidence_payload["predata_protocol_hash"] = "e" * 64
    with pytest.raises(ValidationError, match="evidence hash differs"):
        S1FormalizationRejectionEvidenceV67.model_validate(
            evidence_payload
        )

    context = build_s1_bounded_repair_context_v67(evidence)
    context_payload = context.model_dump(
        mode="json",
        exclude={"context_hash"},
    )
    context_payload["findings"][0]["source_finding_hash"] = "f" * 64
    with pytest.raises(
        ValidationError,
        match="sealed source projection",
    ):
        S1BoundedRepairContextV67.model_validate(context_payload)


def test_builder_rejects_wrong_failure_code_or_nonadjacent_successor() -> None:
    diagnosis, plan, receipt = _recovery_chain(
        failure_code="model_selected_failure_class"
    )
    with pytest.raises(ValueError, match="diagnosis differs"):
        build_s1_formalization_rejection_evidence_v67(
            workspace_spec_hash=WORKSPACE_HASH,
            s0_gate_hash=S0_GATE_HASH,
            predata_protocol_hash=PROTOCOL_HASH,
            reviewer_receipt_hash=REVIEWER_RECEIPT_HASH,
            findings=["Candidate conflicts with the frozen protocol."],
            diagnosis=diagnosis,
            plan=plan,
            recovery_receipt=receipt,
        )

    diagnosis, plan, receipt = _recovery_chain(successor_attempt=3)
    with pytest.raises(ValueError, match="graph transition"):
        build_s1_formalization_rejection_evidence_v67(
            workspace_spec_hash=WORKSPACE_HASH,
            s0_gate_hash=S0_GATE_HASH,
            predata_protocol_hash=PROTOCOL_HASH,
            reviewer_receipt_hash=REVIEWER_RECEIPT_HASH,
            findings=["Candidate conflicts with the frozen protocol."],
            diagnosis=diagnosis,
            plan=plan,
            recovery_receipt=receipt,
        )


def test_runtime_neutral_finding_signature_matches_normalization_contract() -> None:
    normalized = normalize_s1_review_findings_v67(
        [
            "  SAME Finding ",
            "same   finding",
            "Another protocol finding",
        ]
    )

    assert normalized == (
        "another protocol finding",
        "same finding",
    )
    assert s1_reviewer_finding_signature_v67(normalized) == sha256_value(
        {
            "schema_version": "6.7-s1-formalization-rejection",
            "stage": "S1",
            "recovery_category": "review_rejection",
            "normalized_findings": list(normalized),
        }
    )
    assert normalize_s1_review_findings_v67([]) == (
        "reviewer rejected without a normalized finding",
    )

    diagnosis, plan, receipt = _recovery_chain()
    with pytest.raises(ValueError, match="signature differs"):
        build_s1_formalization_rejection_evidence_v67(
            workspace_spec_hash=WORKSPACE_HASH,
            s0_gate_hash=S0_GATE_HASH,
            predata_protocol_hash=PROTOCOL_HASH,
            reviewer_receipt_hash=REVIEWER_RECEIPT_HASH,
            findings=list(normalized),
            reviewer_finding_signature="f" * 64,
            diagnosis=diagnosis,
            plan=plan,
            recovery_receipt=receipt,
        )
