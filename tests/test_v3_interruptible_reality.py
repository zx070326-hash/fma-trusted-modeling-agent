from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from fma.hashing import sha256_value
from fma.v3.controlled_dynamics_loop_v34 import (
    OnlineMismatchCalibrationReceiptV34,
    default_controlled_dynamics_exploratory_spec_v34,
    default_controlled_dynamics_policies_v34,
    run_controlled_dynamics_worldpack_v34,
    verify_controlled_dynamics_run_v34,
)


AT34 = datetime(2026, 7, 22, 15, 0, tzinfo=timezone.utc)
PRIOR_V332 = sha256_value("v332-failure")
PRIOR_EPISTEMIC = sha256_value("v301-qualified")
PRIOR_ACTIVE = sha256_value("v261-qualified")
METHOD = sha256_value("iteration12-interruptible-reality-evidence")


@pytest.fixture(scope="module")
def v34_outcome(tmp_path_factory):
    root = tmp_path_factory.mktemp("controlled_v34")
    baseline, candidate = default_controlled_dynamics_policies_v34(
        prior_v332_failure_report_hash=PRIOR_V332,
        prior_epistemic_qualification_hash=PRIOR_EPISTEMIC,
        prior_active_design_qualification_hash=PRIOR_ACTIVE,
        method_evidence_hash=METHOD,
    )
    spec = default_controlled_dynamics_exploratory_spec_v34(
        baseline_policy_hash=baseline.policy_hash,
        candidate_policy_hash=candidate.policy_hash,
        method_evidence_hash=METHOD,
        prior_epistemic_qualification_hash=PRIOR_EPISTEMIC,
        prior_active_design_qualification_hash=PRIOR_ACTIVE,
        prior_v332_failure_report_hash=PRIOR_V332,
        frozen_at=AT34,
    )
    return run_controlled_dynamics_worldpack_v34(
        root,
        spec=spec,
        baseline_policy=baseline,
        candidate_policy=candidate,
        evaluated_at=AT34,
        run_id="v34-interruptible-reality",
    )


def test_v34_changes_only_reality_adapter_and_preserves_proposals(v34_outcome) -> None:
    report = v34_outcome.evolution_report
    assert report.proposer_changed is False
    assert report.estimator_changed is False
    assert report.target_evaluator_changed is False
    assert report.reality_adapter_changed is True
    assert report.router_changed is False
    baseline = {
        item.case_id: item for item in v34_outcome.baseline_bundle.case_receipts
    }
    for candidate in v34_outcome.candidate_bundle.case_receipts:
        left = baseline[candidate.case_id]
        assert left.anchor_action_hashes == candidate.anchor_action_hashes
        assert left.anchor_observation_hashes == candidate.anchor_observation_hashes
        assert left.selected_action_hash == candidate.selected_action_hash
        assert left.noise_schedule_hash == candidate.noise_schedule_hash
        assert (
            None if left.trust_decision is None else left.trust_decision.trust_hash
        ) == (
            None if candidate.trust_decision is None
            else candidate.trust_decision.trust_hash
        )


def test_v34_calibration_is_public_leave_one_anchor_out_maximum(v34_outcome) -> None:
    for receipt in v34_outcome.candidate_bundle.case_receipts:
        if not receipt.plan_admissible:
            assert receipt.calibration is None
            continue
        assert receipt.calibration is not None
        calibration = receipt.calibration
        assert calibration.mismatch_threshold == max(
            calibration.segment_nrmse_values
        )
        assert not calibration.calibrated_probability_claimed
        assert not calibration.hidden_probe_used
        tampered = calibration.model_dump(exclude={"calibration_hash"})
        tampered["mismatch_threshold"] += 1e-3
        with pytest.raises(ValidationError):
            OnlineMismatchCalibrationReceiptV34(**tampered)


def test_v34_authority_is_monotone_and_exposure_never_exceeds_baseline(
    v34_outcome,
) -> None:
    baseline = {
        item.case_id: item for item in v34_outcome.baseline_bundle.case_receipts
    }
    for candidate in v34_outcome.candidate_bundle.case_receipts:
        left = baseline[candidate.case_id]
        if not candidate.plan_admissible:
            assert candidate.exposure_ledger is None
            continue
        assert candidate.exposure_ledger is not None
        assert left.exposure_ledger is not None
        assert candidate.exposure_ledger.used_duration <= (
            left.exposure_ledger.used_duration + 1e-12
        )
        assert candidate.exposure_ledger.used_energy <= (
            left.exposure_ledger.used_energy + 1e-12
        )
        assert candidate.exposure_ledger.used_peak_amplitude <= (
            left.exposure_ledger.used_peak_amplitude + 1e-12
        )
        assert candidate.exposure_ledger.used_switch_count <= (
            left.exposure_ledger.used_switch_count
        )
        authorities = [
            item.authority_before for item in candidate.segment_receipts
        ] + [candidate.segment_receipts[-1].authority_after]
        assert authorities == sorted(authorities, reverse=True)
        assert all(not item.reescalation_permitted for item in candidate.segment_receipts)


def test_v34_remains_exploratory_even_if_adapter_gates_pass(v34_outcome) -> None:
    report = v34_outcome.evolution_report
    assert not report.overall_qualification_permitted
    assert not report.confirmation_permitted
    assert not report.real_world_authorization_permitted
    assert report.adapter_candidate_ready == all(report.gates.values())


def test_v34_run_replays_independently(v34_outcome) -> None:
    assert verify_controlled_dynamics_run_v34(v34_outcome.store.run_directory)
