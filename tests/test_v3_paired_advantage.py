from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from fma.hashing import sha256_value
from fma.v3.controlled_dynamics_loop_v332 import (
    _paired_goal_risk_advantage_v332,
    default_controlled_dynamics_exploratory_spec_v332,
    default_controlled_dynamics_policies_v332,
    run_controlled_dynamics_worldpack_v332,
    verify_controlled_dynamics_run_v332,
)


AT332 = datetime(2026, 7, 22, 13, 0, tzinfo=timezone.utc)
PRIOR_V331 = sha256_value("v331-failure")
PRIOR_EPISTEMIC = sha256_value("v301-qualified")
PRIOR_ACTIVE = sha256_value("v261-qualified")
METHOD = sha256_value("iteration11-paired-advantage-evidence")


@pytest.fixture(scope="module")
def v332_outcome(tmp_path_factory):
    root = tmp_path_factory.mktemp("controlled_v332")
    baseline, candidate = default_controlled_dynamics_policies_v332(
        prior_v331_failure_report_hash=PRIOR_V331,
        prior_epistemic_qualification_hash=PRIOR_EPISTEMIC,
        prior_active_design_qualification_hash=PRIOR_ACTIVE,
        method_evidence_hash=METHOD,
    )
    spec = default_controlled_dynamics_exploratory_spec_v332(
        baseline_policy_hash=baseline.policy_hash,
        candidate_policy_hash=candidate.policy_hash,
        method_evidence_hash=METHOD,
        prior_epistemic_qualification_hash=PRIOR_EPISTEMIC,
        prior_active_design_qualification_hash=PRIOR_ACTIVE,
        prior_v331_failure_report_hash=PRIOR_V331,
        frozen_at=AT332,
    )
    return run_controlled_dynamics_worldpack_v332(
        root,
        spec=spec,
        baseline_policy=baseline,
        candidate_policy=candidate,
        evaluated_at=AT332,
        run_id="v332-paired-advantage",
    )


def test_v332_preserves_resources_contracts_and_shared_anchors(v332_outcome) -> None:
    evolution = v332_outcome.evolution_report
    assert evolution.resource_entitlement_parity
    assert evolution.resource_use_parity
    assert evolution.target_contract_parity
    assert evolution.shared_anchor_parity
    assert evolution.trust_receipts_complete
    baseline = {
        item.case_id: item for item in v332_outcome.baseline_bundle.case_receipts
    }
    for candidate in v332_outcome.candidate_bundle.case_receipts:
        left = baseline[candidate.case_id]
        assert left.resource_ledger == candidate.resource_ledger
        assert left.final_contract.contract_hash == candidate.final_contract.contract_hash
        if len(candidate.selected_action_ids) >= 2:
            assert left.selected_action_ids[:2] == candidate.selected_action_ids[:2]


def test_v332_recomputes_paired_member_advantage_and_controls_third_action(
    v332_outcome,
) -> None:
    baseline = {
        item.case_id: item for item in v332_outcome.baseline_bundle.case_receipts
    }
    private = {
        item.public_case.case_id: item for item in v332_outcome.private_pack.cases
    }
    for candidate in v332_outcome.candidate_bundle.case_receipts:
        decision = candidate.paired_advantage_trust_decision
        if len(candidate.selected_action_ids) < 3:
            assert decision is None
            continue
        assert decision is not None
        case = private[candidate.case_id]
        actions = {
            action.action_hash: action for action in case.public_case.action_catalog
        }
        observations = [
            case.action_observations[action_id]
            for action_id in candidate.selected_action_ids[:2]
        ]
        seed, paired = _paired_goal_risk_advantage_v332(
            v332_outcome.spec,
            case.public_case,
            observations,
            candidate.final_contract.decision_target,
            candidate.steps[-1].step_index,
            actions[decision.active_action_hash],
            actions[decision.fallback_action_hash],
        )
        assert decision.ensemble_seed == seed
        assert decision.paired_advantage_distribution_hash == sha256_value({
            "case_id": candidate.case_id,
            "step_index": candidate.steps[-1].step_index,
            "ensemble_seed": seed,
            "active_action_hash": decision.active_action_hash,
            "fallback_action_hash": decision.fallback_action_hash,
            "paired_member_advantages": paired.tolist(),
        })
        assert decision.paired_advantage_q20 == pytest.approx(
            np.quantile(paired, v332_outcome.spec.robust_goal_gain_quantile),
            abs=1e-12,
        )
        passes = (
            decision.cross_excitation_nrmse
            <= decision.maximum_cross_excitation_nrmse
            and decision.paired_advantage_q20
            >= decision.minimum_paired_goal_risk_advantage
        )
        assert (decision.decision == "use_goal_risk") == passes
        public_hashes = {
            action.action_id: action.action_hash
            for action in case.public_case.action_catalog
        }
        assert decision.fallback_action_hash == public_hashes[
            baseline[candidate.case_id].selected_action_ids[2]
        ]
        assert decision.selected_action_hash == public_hashes[
            candidate.selected_action_ids[2]
        ]
        assert not decision.hidden_probe_used
        assert not decision.calibrated_probability_claimed
        assert not decision.theoretical_selection_guarantee_claimed


def test_v332_changes_only_pairwise_acquisition_trust_and_cannot_qualify(
    v332_outcome,
) -> None:
    evolution = v332_outcome.evolution_report
    assert evolution.single_component_delta == (
        "unpaired_quantile_margin_to_paired_bootstrap_advantage_only"
    )
    assert evolution.acquisition_changed
    assert not evolution.estimator_changed
    assert not evolution.action_catalog_changed
    assert not evolution.risk_gate_changed
    assert not evolution.statistical_gate_changed
    assert not evolution.model_router_changed
    assert not evolution.budget_model_changed
    assert not evolution.overall_qualification_permitted
    assert not evolution.confirmation_permitted


def test_v332_run_replays_independently(v332_outcome) -> None:
    assert verify_controlled_dynamics_run_v332(v332_outcome.store.run_directory)
