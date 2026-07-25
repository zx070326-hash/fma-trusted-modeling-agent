from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from fma.hashing import sha256_value
from fma.v3.controlled_dynamics_loop import (
    default_controlled_dynamics_exploratory_spec_v31,
    default_controlled_dynamics_policies_v31,
    run_controlled_dynamics_worldpack_v31,
    verify_controlled_dynamics_run_v31,
    _permission_v31,
)
from fma.v3.controlled_dynamics_loop_v311 import (
    default_controlled_dynamics_exploratory_spec_v311,
    default_controlled_dynamics_policies_v311,
    run_controlled_dynamics_worldpack_v311,
    verify_controlled_dynamics_run_v311,
)
from fma.v3.experiment_ir import (
    ExperimentConstraintEnvelopeV31,
    KnownActuatorMapV31,
    PiecewiseConstantInputActionV31,
    validate_action_against_envelope_v31,
)


AT31 = datetime(2026, 7, 22, 5, 0, tzinfo=timezone.utc)
AT311 = datetime(2026, 7, 22, 6, 0, tzinfo=timezone.utc)
PRIOR_EPISTEMIC = sha256_value("v301-qualified")
PRIOR_ACTIVE = sha256_value("v261-qualified")
METHOD = sha256_value("iteration10-method-evidence")


@pytest.fixture(scope="module")
def v31_failure(tmp_path_factory):
    root = tmp_path_factory.mktemp("controlled_v31")
    baseline, candidate = default_controlled_dynamics_policies_v31(
        prior_epistemic_qualification_hash=PRIOR_EPISTEMIC,
        prior_active_design_qualification_hash=PRIOR_ACTIVE,
        method_evidence_hash=METHOD,
    )
    spec = default_controlled_dynamics_exploratory_spec_v31(
        baseline_policy_hash=baseline.policy_hash,
        candidate_policy_hash=candidate.policy_hash,
        method_evidence_hash=METHOD,
        prior_epistemic_qualification_hash=PRIOR_EPISTEMIC,
        prior_active_design_qualification_hash=PRIOR_ACTIVE,
        frozen_at=AT31,
    )
    return run_controlled_dynamics_worldpack_v31(
        root,
        spec=spec,
        baseline_policy=baseline,
        candidate_policy=candidate,
        evaluated_at=AT31,
        run_id="v31-failure",
    )


@pytest.fixture(scope="module")
def v311_failure(tmp_path_factory, v31_failure):
    root = tmp_path_factory.mktemp("controlled_v311")
    prior_failure = v31_failure.report.report_hash
    baseline, candidate = default_controlled_dynamics_policies_v311(
        prior_v31_failure_report_hash=prior_failure,
        prior_epistemic_qualification_hash=PRIOR_EPISTEMIC,
        prior_active_design_qualification_hash=PRIOR_ACTIVE,
        method_evidence_hash=METHOD,
    )
    spec = default_controlled_dynamics_exploratory_spec_v311(
        baseline_policy_hash=baseline.policy_hash,
        candidate_policy_hash=candidate.policy_hash,
        method_evidence_hash=METHOD,
        prior_epistemic_qualification_hash=PRIOR_EPISTEMIC,
        prior_active_design_qualification_hash=PRIOR_ACTIVE,
        prior_v31_failure_report_hash=prior_failure,
        frozen_at=AT311,
    )
    return run_controlled_dynamics_worldpack_v311(
        root,
        spec=spec,
        baseline_policy=baseline,
        candidate_policy=candidate,
        evaluated_at=AT311,
        run_id="v311-failure",
    )


def test_controlled_input_ir_recomputes_physical_budget() -> None:
    actuator = KnownActuatorMapV31.seal(
        actuator_id="known_actuator",
        state_names=["x"],
        input_names=["u"],
        matrix=[[1.0]],
        source_ref="fixture:known_actuator",
    )
    values = [[-0.35], [-0.35], [0.35], [-0.35], [0.35], [0.35]]
    action = PiecewiseConstantInputActionV31.seal(
        action_id="bounded_input",
        actuator_hash=actuator.actuator_hash,
        segment_duration=0.32,
        input_values=values,
        peak_amplitude=0.35,
        total_energy=0.32 * 6 * 0.35**2,
        switch_count=3,
    )
    envelope = ExperimentConstraintEnvelopeV31.seal(
        envelope_id="bounded_envelope",
        actuator_hash=actuator.actuator_hash,
        state_lower_bounds=[-2.0],
        state_upper_bounds=[2.0],
        required_peak_amplitude=0.35,
        required_total_energy=0.32 * 6 * 0.35**2,
        required_switch_count=3,
        maximum_empirical_prediction_risk=0.25,
    )
    assert validate_action_against_envelope_v31(action, actuator, envelope) == []
    with pytest.raises(ValidationError):
        PiecewiseConstantInputActionV31.seal(
            action_id="false_energy",
            actuator_hash=actuator.actuator_hash,
            segment_duration=0.32,
            input_values=values,
            peak_amplitude=0.35,
            total_energy=1.0,
            switch_count=3,
        )


def test_unknown_actuator_cannot_enter_v31() -> None:
    with pytest.raises(ValidationError):
        KnownActuatorMapV31.seal(
            actuator_id="unknown_actuator",
            state_names=["x"],
            input_names=["u"],
            matrix=[[1.0]],
            source_ref="fixture:unknown",
            actuator_is_known=False,
        )


def test_no_admissible_action_returns_code_level_abstention() -> None:
    decision = _permission_v31(
        sha256_value("acquisition"),
        sha256_value("envelope"),
        data_quality_passed=True,
        admissible=False,
        budget_before=2,
        decided_at=AT31,
    )
    assert decision.decision == "abstain"
    assert decision.policy_rule == "abstain_no_admissible_action"
    assert decision.budget_after == decision.budget_before


def test_v31_records_loss_and_router_failure_without_qualification(v31_failure) -> None:
    report = v31_failure.report
    assert report.status == "exploratory_only_v31"
    assert report.macro_absolute_loss_improvement < 0
    assert report.material_negative_transfer_count == 6
    assert report.routing_accuracy == 0.75
    assert not report.gates["macro_improvement_lower_bound"]
    assert not report.gates["routing_accuracy"]
    assert v31_failure.qualification is None
    assert verify_controlled_dynamics_run_v31(v31_failure.store.run_directory)


def test_v31_data_gate_prevents_every_controlled_input(v31_failure) -> None:
    data_cases = {
        case.public_case.case_id
        for case in v31_failure.private_pack.cases
        if "data_layer" in case.expected_issue_routes
    }
    assert data_cases
    for receipt in v31_failure.candidate_bundle.case_receipts:
        if receipt.case_id in data_cases:
            assert receipt.selected_action_ids == []
            assert "data_layer" in receipt.issue_routes
            assert receipt.abstention_count == 1
    assert v31_failure.report.data_gate_experiment_count == 0


def test_v31_all_catalog_actions_have_equal_physical_budget(v31_failure) -> None:
    for case in v31_failure.private_pack.cases:
        actions = case.public_case.action_catalog
        assert len({action.peak_amplitude for action in actions}) == 1
        assert len({action.total_energy for action in actions}) == 1
        assert len({action.switch_count for action in actions}) == 1
        assert all(not action.real_world_action_authorized for action in actions)


def test_v311_changes_only_horizon_and_preserves_failure(v311_failure, v31_failure) -> None:
    evolution = v311_failure.evolution_report
    report = evolution.base_adjudication_report
    assert evolution.prior_v31_failure_report_hash == v31_failure.report.report_hash
    assert evolution.single_component_delta == "action_horizon_2_to_3_only"
    assert not evolution.estimator_changed
    assert not evolution.acquisition_changed
    assert not evolution.risk_gate_changed
    assert not evolution.model_router_changed
    assert report.macro_absolute_loss_improvement > 0
    assert report.macro_improvement_ci_low < 0
    assert report.material_negative_transfer_count == 6
    assert report.routing_accuracy == 0.75
    assert v311_failure.qualification is None
    assert verify_controlled_dynamics_run_v311(v311_failure.store.run_directory)


def test_v311_dimension_aware_horizon_trace(v311_failure) -> None:
    private_by_id = {
        case.public_case.case_id: case for case in v311_failure.private_pack.cases
    }
    candidate = next(
        receipt
        for receipt in v311_failure.candidate_bundle.case_receipts
        if private_by_id[receipt.case_id].performance_eligible
        and private_by_id[receipt.case_id].target_was_underspecified
    )
    baseline = next(
        receipt
        for receipt in v311_failure.baseline_bundle.case_receipts
        if receipt.case_id == candidate.case_id
    )
    assert [step.action_kind for step in candidate.steps] == [
        "clarify_target", "controlled_experiment", "controlled_experiment"
    ]
    assert [step.action_kind for step in baseline.steps] == [
        "controlled_experiment", "controlled_experiment", "controlled_experiment"
    ]
    assert candidate.action_budget_consumed == baseline.action_budget_consumed == 3

