from __future__ import annotations

from datetime import datetime, timezone

import pytest

from fma.hashing import sha256_value
from fma.v3.controlled_dynamics_loop_v331 import (
    default_controlled_dynamics_exploratory_spec_v331,
    default_controlled_dynamics_policies_v331,
    run_controlled_dynamics_worldpack_v331,
    verify_controlled_dynamics_run_v331,
)


AT331 = datetime(2026, 7, 22, 11, 0, tzinfo=timezone.utc)
PRIOR_V33 = sha256_value("v33-failure")
PRIOR_EPISTEMIC = sha256_value("v301-qualified")
PRIOR_ACTIVE = sha256_value("v261-qualified")
METHOD = sha256_value("iteration11-acquisition-trust-gate-evidence")


@pytest.fixture(scope="module")
def v331_outcome(tmp_path_factory):
    root = tmp_path_factory.mktemp("controlled_v331")
    baseline, candidate = default_controlled_dynamics_policies_v331(
        prior_v33_failure_report_hash=PRIOR_V33,
        prior_epistemic_qualification_hash=PRIOR_EPISTEMIC,
        prior_active_design_qualification_hash=PRIOR_ACTIVE,
        method_evidence_hash=METHOD,
    )
    spec = default_controlled_dynamics_exploratory_spec_v331(
        baseline_policy_hash=baseline.policy_hash,
        candidate_policy_hash=candidate.policy_hash,
        method_evidence_hash=METHOD,
        prior_epistemic_qualification_hash=PRIOR_EPISTEMIC,
        prior_active_design_qualification_hash=PRIOR_ACTIVE,
        prior_v33_failure_report_hash=PRIOR_V33,
        frozen_at=AT331,
    )
    return run_controlled_dynamics_worldpack_v331(
        root,
        spec=spec,
        baseline_policy=baseline,
        candidate_policy=candidate,
        evaluated_at=AT331,
        run_id="v331-acquisition-trust-gate",
    )


def test_v331_preserves_resources_contracts_and_shared_anchors(v331_outcome) -> None:
    evolution = v331_outcome.evolution_report
    assert evolution.resource_entitlement_parity
    assert evolution.resource_use_parity
    assert evolution.target_contract_parity
    assert evolution.shared_anchor_parity
    assert evolution.trust_receipts_complete

    baseline = {
        item.case_id: item for item in v331_outcome.baseline_bundle.case_receipts
    }
    for candidate in v331_outcome.candidate_bundle.case_receipts:
        left = baseline[candidate.case_id]
        assert left.resource_ledger == candidate.resource_ledger
        assert left.final_contract.contract_hash == candidate.final_contract.contract_hash
        if len(candidate.selected_action_ids) >= 2:
            assert left.selected_action_ids[:2] == candidate.selected_action_ids[:2]


def test_v331_trust_gate_mechanically_controls_only_third_action(v331_outcome) -> None:
    baseline = {
        item.case_id: item for item in v331_outcome.baseline_bundle.case_receipts
    }
    public = {
        item.public_case.case_id: item.public_case
        for item in v331_outcome.private_pack.cases
    }
    decisions = []
    for candidate in v331_outcome.candidate_bundle.case_receipts:
        if len(candidate.selected_action_ids) < 3:
            assert candidate.acquisition_trust_decision is None
            continue
        decision = candidate.acquisition_trust_decision
        assert decision is not None
        decisions.append(decision.decision)
        left = baseline[candidate.case_id]
        hashes = {
            action.action_id: action.action_hash
            for action in public[candidate.case_id].action_catalog
        }
        assert decision.anchor_action_hashes == [
            hashes[action_id] for action_id in candidate.selected_action_ids[:2]
        ]
        assert decision.fallback_action_hash == hashes[left.selected_action_ids[2]]
        assert decision.selected_action_hash == hashes[candidate.selected_action_ids[2]]
        passes = (
            decision.cross_excitation_nrmse
            <= decision.maximum_cross_excitation_nrmse
            and decision.goal_risk_margin >= decision.minimum_goal_risk_margin
        )
        assert (decision.decision == "use_goal_risk") == passes
        assert not decision.hidden_probe_used
        assert not decision.calibrated_probability_claimed
    assert decisions


def test_v331_changes_only_acquisition_trust_and_cannot_qualify(v331_outcome) -> None:
    evolution = v331_outcome.evolution_report
    assert evolution.single_component_delta == (
        "unconditional_goal_risk_to_cross_excitation_trust_gate_only"
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


def test_v331_run_replays_independently(v331_outcome) -> None:
    assert verify_controlled_dynamics_run_v331(v331_outcome.store.run_directory)
