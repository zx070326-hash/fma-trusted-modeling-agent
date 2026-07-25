from __future__ import annotations

from datetime import datetime, timezone

import pytest

from fma.hashing import sha256_value
from fma.v3.controlled_dynamics_loop_v341 import (
    default_controlled_dynamics_exploratory_spec_v341,
    default_controlled_dynamics_policies_v341,
    run_controlled_dynamics_worldpack_v341,
    verify_controlled_dynamics_run_v341,
)


AT341 = datetime(2026, 7, 22, 17, 0, tzinfo=timezone.utc)
PRIOR_V34 = sha256_value("v34-failure")
PRIOR_V332 = sha256_value("v332-failure")
PRIOR_EPISTEMIC = sha256_value("v301-qualified")
PRIOR_ACTIVE = sha256_value("v261-qualified")
METHOD = sha256_value("iteration12-persistent-mismatch-evidence")


@pytest.fixture(scope="module")
def v341_outcome(tmp_path_factory):
    root = tmp_path_factory.mktemp("controlled_v341")
    baseline, candidate = default_controlled_dynamics_policies_v341(
        prior_v34_failure_report_hash=PRIOR_V34,
        prior_v332_failure_report_hash=PRIOR_V332,
        prior_epistemic_qualification_hash=PRIOR_EPISTEMIC,
        prior_active_design_qualification_hash=PRIOR_ACTIVE,
        method_evidence_hash=METHOD,
    )
    spec = default_controlled_dynamics_exploratory_spec_v341(
        baseline_policy_hash=baseline.policy_hash,
        candidate_policy_hash=candidate.policy_hash,
        method_evidence_hash=METHOD,
        prior_epistemic_qualification_hash=PRIOR_EPISTEMIC,
        prior_active_design_qualification_hash=PRIOR_ACTIVE,
        prior_v332_failure_report_hash=PRIOR_V332,
        prior_v34_failure_report_hash=PRIOR_V34,
        frozen_at=AT341,
    )
    return run_controlled_dynamics_worldpack_v341(
        root,
        spec=spec,
        baseline_policy=baseline,
        candidate_policy=candidate,
        evaluated_at=AT341,
        run_id="v341-persistent-mismatch",
    )


def test_v341_interrupts_only_after_two_consecutive_exceedances(v341_outcome) -> None:
    for receipt in v341_outcome.candidate_bundle.case_receipts:
        if receipt.executed_intervention is None:
            continue
        switches = [
            item for item in receipt.segment_receipts
            if item.decision_after in {
                "switch_to_zero_fallback", "terminate_switch_budget"
            }
        ]
        if not switches:
            continue
        assert len(switches) == 1
        current = switches[0]
        assert current.segment_index >= 2
        previous = receipt.segment_receipts[current.segment_index - 2]
        assert previous.mismatch_nrmse > previous.mismatch_threshold
        assert current.mismatch_nrmse > current.mismatch_threshold


def test_v341_preserves_proposals_noise_and_exposure_dominance(v341_outcome) -> None:
    report = v341_outcome.evolution_report.base_adapter_report
    for gate in (
        "proposal_target_anchor_parity",
        "trust_decision_parity",
        "common_noise_schedule_parity",
        "paired_abstention_parity",
        "segment_and_exposure_receipts_complete",
        "candidate_exposure_dominated_by_baseline",
    ):
        assert report.gates[gate]


def test_v341_is_one_trigger_delta_and_cannot_broaden_authority(v341_outcome) -> None:
    report = v341_outcome.evolution_report
    assert report.single_component_delta == (
        "single_segment_exceedance_to_two_consecutive_exceedances_only"
    )
    assert not report.proposer_changed
    assert not report.estimator_changed
    assert report.reality_adapter_trigger_changed
    assert not report.exposure_model_changed
    assert not report.statistical_gates_changed
    assert not report.router_changed
    assert not report.overall_qualification_permitted
    assert not report.confirmation_permitted
    assert not report.real_world_authorization_permitted
    assert report.persistent_adapter_ready == all(
        report.base_adapter_report.gates.values()
    )


def test_v341_run_replays_independently(v341_outcome) -> None:
    assert verify_controlled_dynamics_run_v341(v341_outcome.store.run_directory)
