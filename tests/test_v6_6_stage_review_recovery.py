from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from fma.hashing import canonical_json, sha256_value
from fma.v5_1.codex_stage_driver import RoleRequestV51
from fma.v5_2.ode_system import ODEThresholdsV52
from fma.v5_7.adaptive_positive_series import AdaptiveThresholdsV57
from fma.v6.scientific_success import ScientificSuccessThresholdsV61
from fma.v6.stage_driver import role_draft_schema_v66
from fma.v6.stage_review_recovery import (
    ADAPTIVE_THRESHOLDS_HASH_V66,
    AUTO_REPAIRABLE_S0_CODES_V66,
    ODE_THRESHOLDS_HASH_V66,
    SCIENTIFIC_SUCCESS_THRESHOLDS_HASH_V66,
    DecisionFunctionDraftV66,
    RegimeDiagnosisDraftV66,
    S0_CANARY_TOLERANCE_V66,
    S0_EVALUATION_PROFILE_PATH_V66,
    S0_ROLE_ARTIFACT_MAX_CHARACTERS_V66,
    S0EvaluationProfileV66,
    S0RepairContextV66,
    S0ReviewDefectCodeV66,
    S0ReviewFindingSetV66,
    authorize_s0_semantic_repair_v66,
    build_s0_repair_context_v66,
    frozen_s0_evaluation_profile_v66,
    infer_s0_review_finding_drafts_v66,
    materialize_decision_function_v66,
    materialize_regime_diagnosis_v66,
    seal_s0_review_findings_v66,
)


def _hash(label: str) -> str:
    return sha256_value({"label": label})


def _decision_payload() -> dict[str, object]:
    return {
        "schema_version": "6.6-s0-decision-draft",
        "function_id": "report.absolute-error",
        "input_names": ["actual", "forecast"],
        "expression": "abs(actual - forecast)",
        "sense": "minimize",
        "output_unit": "index_point",
        "canaries": [
            {
                "canary_id": "exact",
                "input_values": [1.0, 1.0],
                "expected": 0.0,
            },
            {
                "canary_id": "offset",
                "input_values": [1.0, 1.05],
                "expected": 0.05,
            },
        ],
    }


def _regime_payload() -> dict[str, object]:
    profile = frozen_s0_evaluation_profile_v66()
    return {
        "schema_version": "6.6-s0-regime-draft",
        "system_boundary": (
            "The system is one positive scalar series within the stated years."
        ),
        "state_and_memory": (
            "The observable state is scalar, while memory remains a tested "
            "model assumption."
        ),
        "uncertainty_and_data": (
            "Only public observations enter development and uncertainty is "
            "reported diagnostically."
        ),
        "decision_and_loss": (
            "The workflow reports forecast error and authorizes no external "
            "decision."
        ),
        "query_type": "prediction",
        "downstream_decision": (
            "A human may compare the bounded forecast report with alternatives."
        ),
        "decision_function_id": "report.absolute-error",
        "computable_decision_function": (
            "The absolute difference between actual and forecast values is "
            "minimized."
        ),
        "evidence_hashes": sorted([_hash("objective"), profile.profile_hash]),
        "limitations": [
            "The local series does not establish a causal mechanism.",
            "Diagnostic intervals do not provide finite-sample coverage.",
        ],
        "evaluation_profile_hash": profile.profile_hash,
    }


def _adaptive_thresholds() -> AdaptiveThresholdsV57:
    return AdaptiveThresholdsV57.seal(
        split_fraction=0.7,
        minimum_points_per_slice=8,
        maximum_validation_relative_rmse=0.15,
        minimum_persistence_relative_improvement=0.1,
        maximum_innovation_absolute_lag1_correlation=0.35,
        maximum_absolute_growth_ar1_phi=0.95,
        minimum_growth_ar1_validation_relative_improvement=0.05,
        maximum_growth_phi_window_range=0.3,
        maximum_growth_drift_window_range_standardized=1.0,
        maximum_innovation_mean_shift_standardized=1.5,
        maximum_single_innovation_standardized=5.0,
        minimum_validation_interval_coverage=0.5,
        maximum_absolute_mean_log_growth=0.5,
        selection_complexity_penalty_per_parameter=0.002,
        bootstrap_replicates=40,
        bootstrap_seed=155921,
        minimum_bootstrap_success_fraction=0.8,
        maximum_forecast_interval_relative_width=2.0,
        maximum_window_sensitivity_relative_range=1.0,
    )


def _seal(
    reviewer_codes: list[str | S0ReviewDefectCodeV66],
    *,
    regime_payload: object | None = None,
    decision_payload: object | None = None,
    attempt_id: str = "attempt-a1",
) -> S0ReviewFindingSetV66:
    profile = frozen_s0_evaluation_profile_v66()
    return seal_s0_review_findings_v66(
        task_id="task-v66",
        attempt_id=attempt_id,
        reviewer_receipt_hash=_hash(f"review-{attempt_id}"),
        reviewer_codes=reviewer_codes,
        regime_payload=(
            _regime_payload() if regime_payload is None else regime_payload
        ),
        decision_payload=(
            _decision_payload() if decision_payload is None else decision_payload
        ),
        evaluation_profile_payload=profile.model_dump(mode="json"),
    )


def test_frozen_profile_binds_exact_historical_thresholds() -> None:
    profile = frozen_s0_evaluation_profile_v66()
    profile.assert_sealed()

    assert profile.ode_thresholds_hash == ODEThresholdsV52.seal().threshold_hash
    assert profile.adaptive_thresholds_hash == _adaptive_thresholds().threshold_hash
    assert (
        profile.scientific_success_thresholds_hash
        == ScientificSuccessThresholdsV61.seal().thresholds_hash
    )
    assert profile.ode_thresholds_hash == ODE_THRESHOLDS_HASH_V66
    assert profile.adaptive_thresholds_hash == ADAPTIVE_THRESHOLDS_HASH_V66
    assert (
        profile.scientific_success_thresholds_hash
        == SCIENTIFIC_SUCCESS_THRESHOLDS_HASH_V66
    )
    assert profile.candidate_registry_stage == "S1"
    assert profile.private_feedback_permitted is False
    assert profile.scientific_qualification_granted is False
    assert profile.real_world_action_authorized is False
    assert S0_EVALUATION_PROFILE_PATH_V66 == (
        "docs/s0_evaluation_profile_v66.json"
    )


def test_profile_is_deterministic_and_tamper_evident() -> None:
    first = frozen_s0_evaluation_profile_v66()
    second = frozen_s0_evaluation_profile_v66()
    assert first == second
    assert canonical_json(first) == canonical_json(second)

    payload = first.model_dump(mode="json")
    payload["profile_hash"] = _hash("forged")
    with pytest.raises(ValidationError, match="profile hash differs"):
        S0EvaluationProfileV66.model_validate(payload)

    payload = first.model_dump(mode="json")
    payload["ode_thresholds_hash"] = _hash("replacement")
    with pytest.raises(ValidationError, match="ODE threshold hash differs"):
        S0EvaluationProfileV66.model_validate(payload)


def test_v66_drafts_fit_envelope_and_harness_injects_tolerance() -> None:
    decision = DecisionFunctionDraftV66.model_validate(_decision_payload())
    regime = RegimeDiagnosisDraftV66.model_validate(_regime_payload())

    assert len(canonical_json(decision)) < S0_ROLE_ARTIFACT_MAX_CHARACTERS_V66
    assert len(canonical_json(regime)) < S0_ROLE_ARTIFACT_MAX_CHARACTERS_V66

    materialized_decision = materialize_decision_function_v66(decision)
    materialized_regime = materialize_regime_diagnosis_v66(regime)
    assert all(
        item.tolerance == S0_CANARY_TOLERANCE_V66
        for item in materialized_decision.canaries
    )
    assert (
        materialized_regime.decision_function_id
        == materialized_decision.function_id
    )
    assert regime.evaluation_profile_hash in materialized_regime.evidence_hashes


def test_model_cannot_supply_canary_tolerance_or_incomplete_sentences() -> None:
    decision = _decision_payload()
    decision["canaries"][0]["tolerance"] = 1.0
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DecisionFunctionDraftV66.model_validate(decision)

    regime = _regime_payload()
    regime["system_boundary"] = "The generated field stops before completion"
    with pytest.raises(ValidationError, match="terminal punctuation"):
        RegimeDiagnosisDraftV66.model_validate(regime)

    regime = _regime_payload()
    regime["limitations"] = ["The field trails away..."]
    with pytest.raises(ValidationError, match="ellipsis"):
        RegimeDiagnosisDraftV66.model_validate(regime)


def test_regime_requires_exact_profile_binding_and_evidence_commitment() -> None:
    regime = _regime_payload()
    regime["evaluation_profile_hash"] = _hash("other-profile")
    with pytest.raises(ValidationError, match="not bound"):
        RegimeDiagnosisDraftV66.model_validate(regime)

    regime = _regime_payload()
    regime["evidence_hashes"] = [_hash("objective")]
    with pytest.raises(ValidationError, match="included in evidence_hashes"):
        RegimeDiagnosisDraftV66.model_validate(regime)


def test_materialized_v5_regime_preserves_profile_binding_in_evidence() -> None:
    profile = frozen_s0_evaluation_profile_v66()
    regime = materialize_regime_diagnosis_v66(
        RegimeDiagnosisDraftV66.model_validate(_regime_payload())
    )
    findings = infer_s0_review_finding_drafts_v66(
        regime_payload=regime.model_dump(mode="json"),
        decision_payload=_decision_payload(),
        evaluation_profile_payload=profile.model_dump(mode="json"),
    )

    assert profile.profile_hash in regime.evidence_hashes
    assert all(
        item.code
        not in {
            S0ReviewDefectCodeV66.EVALUATION_PROFILE_BINDING_MISSING,
            S0ReviewDefectCodeV66.EVALUATION_PROFILE_BINDING_MISMATCH,
        }
        for item in findings
    )


def test_v66_transport_constrains_s0_review_to_fixed_codes() -> None:
    request = RoleRequestV51.seal(
        request_id="request-v66-review",
        task_id="task-v66",
        stage="S0",
        role_name="s0_referee",
        role_kind="reviewer",
        subject_id="s0_problem_contract",
        objective="Independently evaluate the frozen S0 workflow contract.",
        public_inputs={},
        allowed_candidate_ids=[],
        authority_denials=["cannot_sign_gate"],
    )

    schema = role_draft_schema_v66(request)
    items = schema["properties"]["findings"]["items"]

    assert set(items["enum"]) == {
        item.value for item in S0ReviewDefectCodeV66
    }
    assert items["type"] == "string"
    assert schema["properties"]["proposed_artifacts"]["maxItems"] == 0


def test_mechanical_inference_detects_truncation_and_tolerance() -> None:
    profile = frozen_s0_evaluation_profile_v66()
    regime = _regime_payload()
    regime["system_boundary"] = "x" * 200
    decision = _decision_payload()
    decision["canaries"][0]["tolerance"] = 1.0

    findings = infer_s0_review_finding_drafts_v66(
        regime_payload=regime,
        decision_payload=decision,
        evaluation_profile_payload=profile.model_dump(mode="json"),
    )
    keys = {
        (item.code, item.artifact_path, item.json_pointer)
        for item in findings
    }
    assert (
        S0ReviewDefectCodeV66.FIELD_TRUNCATED,
        "docs/regime.json",
        "/system_boundary",
    ) in keys
    assert (
        S0ReviewDefectCodeV66.MODEL_CONTROLLED_CANARY_TOLERANCE,
        "problem/decision_function.json",
        "/canaries/0",
    ) in keys


def test_claimed_auto_code_requires_current_artifact_evidence() -> None:
    unsupported = _seal(
        [S0ReviewDefectCodeV66.FIELD_TRUNCATED],
    )
    assert len(unsupported.findings) == 1
    finding = unsupported.findings[0]
    assert finding.code == S0ReviewDefectCodeV66.OTHER_UNCLASSIFIED_REJECT
    assert finding.verification == "UNCLASSIFIED"
    assert finding.severity == "HUMAN"
    assert finding.auto_repairable is False

    regime = _regime_payload()
    regime["system_boundary"] = "x" * 200
    supported = _seal(
        [S0ReviewDefectCodeV66.FIELD_TRUNCATED],
        regime_payload=regime,
    )
    finding = supported.findings[0]
    assert finding.code == S0ReviewDefectCodeV66.FIELD_TRUNCATED
    assert finding.verification == "MECHANICALLY_VERIFIED"
    assert finding.auto_repairable is True
    assert finding.code in AUTO_REPAIRABLE_S0_CODES_V66


def test_unknown_and_semantic_codes_are_human_only() -> None:
    unknown = _seal(["REVIEWER_FREE_TEXT_CODE"])
    assert unknown.findings[0].code == (
        S0ReviewDefectCodeV66.OTHER_UNCLASSIFIED_REJECT
    )
    assert unknown.findings[0].auto_repairable is False

    semantic = _seal(
        [S0ReviewDefectCodeV66.SEMANTIC_BOUNDARY_UNRESOLVED]
    )
    assert semantic.findings[0].verification == "HUMAN_ONLY"
    assert semantic.findings[0].severity == "HUMAN"
    decision = authorize_s0_semantic_repair_v66(
        semantic,
        repair_attempts_used=0,
    )
    assert decision.authorized is False
    assert decision.reason == "NON_AUTOREPAIRABLE_FINDING"


def test_finding_ids_and_failure_signature_are_harness_derived() -> None:
    regime = _regime_payload()
    regime["system_boundary"] = "x" * 200
    first = _seal(
        [S0ReviewDefectCodeV66.FIELD_TRUNCATED],
        regime_payload=regime,
        attempt_id="attempt-a1",
    )
    second = _seal(
        [S0ReviewDefectCodeV66.FIELD_TRUNCATED],
        regime_payload=regime,
        attempt_id="attempt-a2",
    )

    assert first.failure_signature == second.failure_signature
    assert first.finding_set_hash != second.finding_set_hash
    assert first.findings[0].finding_id == second.findings[0].finding_id

    payload = first.model_dump(mode="json")
    payload["failure_signature"] = _hash("forged-signature")
    with pytest.raises(ValidationError, match="failure signature"):
        S0ReviewFindingSetV66.model_validate(payload)


def test_single_semantic_repair_budget_and_repeat_stop() -> None:
    regime = _regime_payload()
    regime["system_boundary"] = "x" * 200
    findings = _seal(
        [S0ReviewDefectCodeV66.FIELD_TRUNCATED],
        regime_payload=regime,
    )

    first = authorize_s0_semantic_repair_v66(
        findings,
        repair_attempts_used=0,
    )
    assert first.authorized is True
    assert first.reason == "AUTHORIZED"
    first.assert_sealed()

    repeated = authorize_s0_semantic_repair_v66(
        findings,
        repair_attempts_used=0,
        previous_failure_signatures=[findings.failure_signature],
    )
    assert repeated.authorized is False
    assert repeated.reason == "REPEATED_FAILURE_SIGNATURE"

    exhausted = authorize_s0_semantic_repair_v66(
        findings,
        repair_attempts_used=1,
    )
    assert exhausted.authorized is False
    assert exhausted.reason == "REPAIR_BUDGET_EXHAUSTED"


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        ("holdout", "HOLDOUT_EXPOSED"),
        ("private", "PRIVATE_EVIDENCE_USED"),
    ],
)
def test_private_or_holdout_exposure_stops_auto_repair(
    flag: str,
    expected: str,
) -> None:
    regime = _regime_payload()
    regime["system_boundary"] = "x" * 200
    findings = _seal(
        [S0ReviewDefectCodeV66.FIELD_TRUNCATED],
        regime_payload=regime,
    )
    decision = authorize_s0_semantic_repair_v66(
        findings,
        repair_attempts_used=0,
        holdout_exposed=flag == "holdout",
        private_evidence_used=flag == "private",
    )
    assert decision.authorized is False
    assert decision.reason == expected


def test_repair_context_contains_only_codes_pointers_and_hashes() -> None:
    profile = frozen_s0_evaluation_profile_v66()
    regime = _regime_payload()
    regime["system_boundary"] = "x" * 200
    findings = _seal(
        [S0ReviewDefectCodeV66.FIELD_TRUNCATED],
        regime_payload=regime,
    )
    decision = authorize_s0_semantic_repair_v66(
        findings,
        repair_attempts_used=0,
    )
    context = build_s0_repair_context_v66(
        finding_set=findings,
        authorization=decision,
        new_attempt_id="attempt-a2",
    )
    context.assert_sealed()
    assert context.evaluation_profile_hash == profile.profile_hash
    assert context.reviewer_rationale_included is False
    assert context.private_evidence_included is False
    assert context.holdout_evidence_included is False
    assert context.targets[0].repair_operator == "COMPLETE_SENTENCE"

    serialized = json.loads(context.model_dump_json())
    assert serialized["reviewer_rationale_included"] is False
    assert "private_evidence" not in serialized
    assert "holdout_evidence" not in serialized

    payload = context.model_dump(mode="json")
    payload["context_hash"] = _hash("forged-context")
    with pytest.raises(ValidationError, match="context hash differs"):
        S0RepairContextV66.model_validate(payload)


def test_non_authorized_repair_cannot_build_context() -> None:
    findings = _seal(
        [S0ReviewDefectCodeV66.HUMAN_DECISION_REQUIRED]
    )
    decision = authorize_s0_semantic_repair_v66(
        findings,
        repair_attempts_used=0,
    )
    with pytest.raises(ValueError, match="not authorized"):
        build_s0_repair_context_v66(
            finding_set=findings,
            authorization=decision,
            new_attempt_id="attempt-a2",
        )
