from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from fma.hashing import sha256_value
from fma.v3.controlled_dynamics_loop_v35 import (
    default_controlled_dynamics_exploratory_spec_v35,
    default_controlled_dynamics_policies_v35,
    run_controlled_dynamics_worldpack_v35,
    verify_controlled_dynamics_run_v35,
)


AT35 = datetime(2026, 7, 22, 19, 0, tzinfo=timezone.utc)
HASH = sha256_value("prior-evidence")
METHOD = sha256_value("iteration13-guarded-acquisition-factorial")


@pytest.fixture(scope="module")
def v35_outcome(tmp_path_factory):
    root = tmp_path_factory.mktemp("controlled_v35")
    random, unguarded, guarded = default_controlled_dynamics_policies_v35(
        prior_v341_adapter_report_hash=HASH,
        prior_v34_failure_report_hash=HASH,
        prior_v332_failure_report_hash=HASH,
        prior_epistemic_qualification_hash=HASH,
        prior_active_design_qualification_hash=HASH,
        method_evidence_hash=METHOD,
    )
    spec = default_controlled_dynamics_exploratory_spec_v35(
        baseline_policy_hash=random.policy_hash,
        diagnostic_policy_hash=unguarded.policy_hash,
        candidate_policy_hash=guarded.policy_hash,
        method_evidence_hash=METHOD,
        prior_epistemic_qualification_hash=HASH,
        prior_active_design_qualification_hash=HASH,
        prior_v332_failure_report_hash=HASH,
        prior_v34_failure_report_hash=HASH,
        prior_v341_adapter_report_hash=HASH,
        frozen_at=AT35,
    )
    return run_controlled_dynamics_worldpack_v35(
        root,
        spec=spec,
        baseline_policy=random,
        diagnostic_policy=unguarded,
        candidate_policy=guarded,
        evaluated_at=AT35,
        run_id="v35-guarded-acquisition-factorial",
    )


def test_v35_three_arms_share_context_but_bind_distinct_actions(v35_outcome) -> None:
    random = {x.case_id: x for x in v35_outcome.baseline_bundle.case_receipts}
    unguarded = {x.case_id: x for x in v35_outcome.diagnostic_bundle.case_receipts}
    guarded = {x.case_id: x for x in v35_outcome.candidate_bundle.case_receipts}
    for case_id, left in random.items():
        middle = unguarded[case_id]
        right = guarded[case_id]
        assert left.anchor_action_hashes == middle.anchor_action_hashes
        assert left.anchor_action_hashes == right.anchor_action_hashes
        assert left.anchor_observation_hashes == middle.anchor_observation_hashes
        assert left.noise_schedule_hash == middle.noise_schedule_hash
        assert left.noise_schedule_hash == right.noise_schedule_hash
        assert left.abstention_reason == middle.abstention_reason
        assert left.abstention_reason == right.abstention_reason
        if not left.plan_admissible:
            continue
        assert left.trust_decision is not None
        assert middle.trust_decision is not None
        assert right.trust_decision is not None
        assert left.selected_action_hash == left.trust_decision.fallback_action_hash
        assert middle.selected_action_hash == middle.trust_decision.selected_action_hash
        assert right.selected_action_hash == right.trust_decision.selected_action_hash
        assert middle.selected_action_hash == right.selected_action_hash


def test_v35_factorial_estimands_recompute_from_private_outer_losses(v35_outcome) -> None:
    random = {x.case_id: x for x in v35_outcome.baseline_bundle.case_receipts}
    unguarded = {x.case_id: x for x in v35_outcome.diagnostic_bundle.case_receipts}
    guarded = {x.case_id: x for x in v35_outcome.candidate_bundle.case_receipts}
    ids = [case_id for case_id, item in random.items() if item.target_loss is not None]
    r = np.asarray([random[x].target_loss for x in ids])
    a = np.asarray([unguarded[x].target_loss for x in ids])
    g = np.asarray([guarded[x].target_loss for x in ids])
    report = v35_outcome.evolution_report
    assert report.random_vs_unguarded_macro_improvement == pytest.approx(
        np.mean(r - a), abs=1e-12
    )
    assert report.unguarded_vs_guarded_macro_improvement == pytest.approx(
        np.mean(a - g), abs=1e-12
    )
    assert report.package_macro_improvement == pytest.approx(
        np.mean(r - g), abs=1e-12
    )


def test_v35_guard_needs_persistent_mismatch_and_cannot_expand_exposure(
    v35_outcome,
) -> None:
    random = {x.case_id: x for x in v35_outcome.baseline_bundle.case_receipts}
    unguarded = {x.case_id: x for x in v35_outcome.diagnostic_bundle.case_receipts}
    for guarded in v35_outcome.candidate_bundle.case_receipts:
        if not guarded.plan_admissible:
            continue
        assert guarded.exposure_ledger is not None
        for comparison in (random[guarded.case_id], unguarded[guarded.case_id]):
            assert comparison.exposure_ledger is not None
            assert guarded.exposure_ledger.used_duration <= (
                comparison.exposure_ledger.used_duration + 1e-12
            )
            assert guarded.exposure_ledger.used_energy <= (
                comparison.exposure_ledger.used_energy + 1e-12
            )
            assert guarded.exposure_ledger.used_switch_count <= (
                comparison.exposure_ledger.used_switch_count
            )
        switches = [
            item for item in guarded.segment_receipts
            if item.decision_after in {
                "switch_to_zero_fallback", "terminate_switch_budget"
            }
        ]
        for current in switches:
            assert current.segment_index >= 2
            previous = guarded.segment_receipts[current.segment_index - 2]
            assert previous.mismatch_nrmse > previous.mismatch_threshold
            assert current.mismatch_nrmse > current.mismatch_threshold


def test_v35_readiness_is_package_scoped_and_cannot_qualify(v35_outcome) -> None:
    report = v35_outcome.evolution_report
    assert report.guarded_acquisition_ready == all(report.gates.values())
    assert report.router_experiment_permitted == report.guarded_acquisition_ready
    assert not report.router_changed
    assert not report.overall_qualification_permitted
    assert not report.confirmation_permitted
    assert not report.real_world_authorization_permitted


def test_v35_run_replays_independently(v35_outcome) -> None:
    assert verify_controlled_dynamics_run_v35(v35_outcome.store.run_directory)
