from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from fma.hashing import sha256_value
from fma.v3.controlled_dynamics_loop_v36 import (
    OutcomeCalibrationRowV36,
    default_controlled_dynamics_exploratory_spec_v36,
    default_controlled_dynamics_policies_v36,
    run_controlled_dynamics_worldpack_v36,
    seal_outcome_calibration_ledger_v36,
    verify_controlled_dynamics_run_v36,
)


AT36 = datetime(2026, 7, 22, 20, 30, tzinfo=timezone.utc)
V332 = sha256_value("v332-report")
V35 = sha256_value("v35-report")
HASH = sha256_value("prior-evidence")
METHOD = sha256_value("iteration13-outcome-calibration")


@pytest.fixture(scope="module")
def calibration_ledger():
    pairs = (
        [(0.04, -0.05)] * 13
        + [(0.07, 0.01)] * 9
        + [(0.10, value) for value in (-0.03, 0.05, 0.05, 0.05)]
        + [(0.13, 0.10)] * 2
        + [(0.16, 0.12)] * 2
    )
    rows = []
    for index, (q20, gain) in enumerate(pairs):
        version = "v332" if index < 15 else "v35"
        rows.append(OutcomeCalibrationRowV36(
            source_version=version,
            source_report_hash=V332 if version == "v332" else V35,
            case_id=f"calibration_case_{index:02d}",
            paired_advantage_q20=q20,
            realized_acquisition_gain=gain,
            material_negative_transfer=gain < -0.02,
        ))
    return seal_outcome_calibration_ledger_v36(
        rows=rows,
        source_v332_report_hash=V332,
        source_v35_report_hash=V35,
        source_bundle_hashes=[sha256_value(f"bundle-{i}") for i in range(4)],
        built_at=AT36,
    )


@pytest.fixture(scope="module")
def v36_outcome(tmp_path_factory, calibration_ledger):
    root = tmp_path_factory.mktemp("controlled_v36")
    random, original, calibrated = default_controlled_dynamics_policies_v36(
        ledger=calibration_ledger,
        prior_v35_failure_report_hash=V35,
        prior_v341_adapter_report_hash=HASH,
        prior_v332_failure_report_hash=V332,
        method_evidence_hash=METHOD,
    )
    spec = default_controlled_dynamics_exploratory_spec_v36(
        ledger=calibration_ledger,
        baseline_policy_hash=random.policy_hash,
        diagnostic_policy_hash=original.policy_hash,
        candidate_policy_hash=calibrated.policy_hash,
        method_evidence_hash=METHOD,
        prior_epistemic_qualification_hash=HASH,
        prior_active_design_qualification_hash=HASH,
        prior_v332_failure_report_hash=V332,
        prior_v34_failure_report_hash=HASH,
        prior_v341_adapter_report_hash=HASH,
        prior_v35_failure_report_hash=V35,
        frozen_at=AT36,
    )
    return run_controlled_dynamics_worldpack_v36(
        root,
        ledger=calibration_ledger,
        spec=spec,
        baseline_policy=random,
        diagnostic_policy=original,
        candidate_policy=calibrated,
        evaluated_at=AT36,
        run_id="v36-outcome-calibration",
    )


def test_v36_ledger_recomputes_grid_and_selects_lowest_eligible_cutoff(
    calibration_ledger,
) -> None:
    assert calibration_ledger.selected_q20_cutoff == 0.12
    eligible = [
        item.q20_cutoff for item in calibration_ledger.cutoff_summaries
        if item.bootstrap_ci_low > 0
        and item.material_negative_transfer_count == 0
    ]
    assert eligible[0] == calibration_ledger.selected_q20_cutoff
    assert not calibration_ledger.exchangeability_assumed
    assert not calibration_ledger.conformal_guarantee_claimed


def test_v36_candidate_follows_ledger_while_context_stays_paired(v36_outcome) -> None:
    random = {x.case_id: x for x in v36_outcome.baseline_bundle.case_receipts}
    original = {x.case_id: x for x in v36_outcome.diagnostic_bundle.case_receipts}
    calibrated = {x.case_id: x for x in v36_outcome.candidate_bundle.case_receipts}
    cutoff = v36_outcome.ledger.selected_q20_cutoff
    for case_id, left in random.items():
        middle = original[case_id]
        right = calibrated[case_id]
        assert left.anchor_action_hashes == middle.anchor_action_hashes
        assert left.anchor_action_hashes == right.anchor_action_hashes
        assert left.noise_schedule_hash == middle.noise_schedule_hash
        assert left.noise_schedule_hash == right.noise_schedule_hash
        if not left.plan_admissible:
            continue
        trust = left.trust_decision
        assert trust is not None
        assert left.selected_action_hash == trust.fallback_action_hash
        assert middle.selected_action_hash == trust.selected_action_hash
        expected = (
            trust.active_action_hash
            if trust.decision == "use_goal_risk"
            and trust.paired_advantage_q20 >= cutoff
            else trust.fallback_action_hash
        )
        assert right.selected_action_hash == expected


def test_v36_report_recomputes_primary_package_mean(v36_outcome) -> None:
    random = {x.case_id: x for x in v36_outcome.baseline_bundle.case_receipts}
    calibrated = {x.case_id: x for x in v36_outcome.candidate_bundle.case_receipts}
    values = [
        random[case_id].target_loss - item.target_loss
        for case_id, item in calibrated.items()
        if item.target_loss is not None
    ]
    assert v36_outcome.evolution_report.calibrated_package_macro_improvement == (
        pytest.approx(np.mean(values), abs=1e-12)
    )


def test_v36_readiness_remains_package_scoped(v36_outcome) -> None:
    report = v36_outcome.evolution_report
    assert report.outcome_calibrated_package_ready == all(report.gates.values())
    assert report.router_experiment_permitted == report.outcome_calibrated_package_ready
    assert not report.probability_calibration_claimed
    assert not report.router_changed
    assert not report.overall_qualification_permitted
    assert not report.confirmation_permitted
    assert not report.real_world_authorization_permitted


def test_v36_run_replays_independently(v36_outcome) -> None:
    assert verify_controlled_dynamics_run_v36(v36_outcome.store.run_directory)
