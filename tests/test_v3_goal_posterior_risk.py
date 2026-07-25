from __future__ import annotations

from datetime import datetime, timezone

import pytest

from fma.hashing import sha256_value
from fma.v3.controlled_dynamics_loop_v32 import (
    default_controlled_dynamics_exploratory_spec_v32,
    default_controlled_dynamics_policies_v32,
    run_controlled_dynamics_worldpack_v32,
    verify_controlled_dynamics_run_v32,
)


AT32 = datetime(2026, 7, 22, 7, 30, tzinfo=timezone.utc)
PRIOR_V311 = sha256_value("v311-failure")
PRIOR_EPISTEMIC = sha256_value("v301-qualified")
PRIOR_ACTIVE = sha256_value("v261-qualified")
METHOD = sha256_value("iteration11-goal-posterior-risk-evidence")


@pytest.fixture(scope="module")
def v32_outcome(tmp_path_factory):
    root = tmp_path_factory.mktemp("controlled_v32")
    baseline, candidate = default_controlled_dynamics_policies_v32(
        prior_v311_evolution_report_hash=PRIOR_V311,
        prior_epistemic_qualification_hash=PRIOR_EPISTEMIC,
        prior_active_design_qualification_hash=PRIOR_ACTIVE,
        method_evidence_hash=METHOD,
    )
    spec = default_controlled_dynamics_exploratory_spec_v32(
        baseline_policy_hash=baseline.policy_hash,
        candidate_policy_hash=candidate.policy_hash,
        method_evidence_hash=METHOD,
        prior_epistemic_qualification_hash=PRIOR_EPISTEMIC,
        prior_active_design_qualification_hash=PRIOR_ACTIVE,
        prior_v311_evolution_report_hash=PRIOR_V311,
        frozen_at=AT32,
    )
    return run_controlled_dynamics_worldpack_v32(
        root,
        spec=spec,
        baseline_policy=baseline,
        candidate_policy=candidate,
        evaluated_at=AT32,
        run_id="v32-goal-risk",
    )


def test_v32_changes_only_acquisition_and_cannot_qualify(v32_outcome) -> None:
    evolution = v32_outcome.evolution_report
    assert evolution.single_component_delta == (
        "heuristic_utility_to_goal_posterior_risk_only"
    )
    assert not evolution.estimator_changed
    assert not evolution.action_catalog_changed
    assert not evolution.action_horizon_changed
    assert not evolution.risk_gate_changed
    assert not evolution.statistical_gate_changed
    assert not evolution.model_router_changed
    assert not evolution.overall_qualification_permitted
    assert not evolution.confirmation_permitted


def test_v32_candidate_selects_maximum_admissible_goal_risk_score(v32_outcome) -> None:
    for case in v32_outcome.candidate_bundle.case_receipts:
        for step in case.steps:
            if step.action_kind != "controlled_experiment":
                continue
            admissible = [item for item in step.acquisition_receipts if item.admissible]
            selected = next(
                item for item in admissible
                if item.action_hash == step.selected_action_hash
            )
            assert selected.ranking_score == max(
                item.ranking_score for item in admissible
            )
            assert selected.ranking_score == (
                selected.robust_fractional_goal_risk_reduction
            )


def test_v32_receipts_are_goal_bound_and_exclude_private_probe(v32_outcome) -> None:
    receipt = next(
        item
        for case in v32_outcome.candidate_bundle.case_receipts
        for step in case.steps
        for item in step.acquisition_receipts
    )
    payload = receipt.model_dump(mode="json")
    assert payload["belief_precision_hash"]
    assert payload["goal_operator_hash"]
    assert payload["goal_feature_row_count"] > 0
    assert payload["predicted_goal_posterior_risk"] <= (
        payload["current_goal_posterior_risk"]
    )
    assert not payload["goal_operator_uses_private_probe"]
    serialized = receipt.model_dump_json()
    assert "probe_clean_states" not in serialized
    assert "hidden_parameters" not in serialized


def test_v32_preserves_equal_total_action_budget(v32_outcome) -> None:
    baseline = {
        item.case_id: item for item in v32_outcome.baseline_bundle.case_receipts
    }
    candidate = {
        item.case_id: item for item in v32_outcome.candidate_bundle.case_receipts
    }
    private = {
        item.public_case.case_id: item for item in v32_outcome.private_pack.cases
    }
    assert baseline.keys() == candidate.keys()
    for case_id in baseline:
        if "data_layer" in private[case_id].expected_issue_routes:
            assert baseline[case_id].selected_action_ids == []
            assert candidate[case_id].selected_action_ids == []
        else:
            assert baseline[case_id].action_budget_consumed == (
                candidate[case_id].action_budget_consumed
            ) == 3


def test_v32_run_replays_independently(v32_outcome) -> None:
    assert verify_controlled_dynamics_run_v32(v32_outcome.store.run_directory)
