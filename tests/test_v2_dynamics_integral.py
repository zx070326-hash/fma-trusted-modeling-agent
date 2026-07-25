from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from fma.v2.dynamics_integral import (
    assert_single_component_estimator_ablation_v25,
    default_estimator_policy_v25,
    fit_dynamics_candidate_v25,
    select_dynamics_candidate_v25,
    simulate_dynamics_model_v25,
)
from fma.v2.dynamics_ir import DynamicsDataSnapshotV24


NOW = datetime(2026, 7, 22, tzinfo=timezone.utc)
KNOWLEDGE_HASH = "a" * 64
FAILURE_HASH = "b" * 64


def _decay_snapshot(count: int = 101) -> DynamicsDataSnapshotV24:
    times = [0.05 * index for index in range(count)]
    values = [[2.0 * math.exp(-0.5 * time)] for time in times]
    return DynamicsDataSnapshotV24.seal(
        snapshot_id="decay_v25_fixture",
        declared_state_names=["x"],
        observed_state_names=["x"],
        times=times,
        values=values,
    )


def _policies():
    point = default_estimator_policy_v25(
        "point_savgol",
        knowledge_bundle_hash=KNOWLEDGE_HASH,
        failure_evidence_hash=FAILURE_HASH,
    )
    integral = default_estimator_policy_v25(
        "window_integral_matching",
        knowledge_bundle_hash=KNOWLEDGE_HASH,
        failure_evidence_hash=FAILURE_HASH,
    )
    return point, integral


def test_v25_policies_are_a_frozen_single_component_ablation() -> None:
    point, integral = _policies()
    assert_single_component_estimator_ablation_v25(point, integral)
    assert point.policy_hash != integral.policy_hash
    assert point.candidates[0].candidate_id == integral.candidates[0].candidate_id
    assert point.candidates[0].polynomial_degree == integral.candidates[0].polynomial_degree


@pytest.mark.parametrize("arm", ["point_savgol", "window_integral_matching"])
def test_v25_estimators_recover_and_replay_fully_observed_decay(arm: str) -> None:
    snapshot = _decay_snapshot()
    policy = default_estimator_policy_v25(
        arm,
        knowledge_bundle_hash=KNOWLEDGE_HASH,
        failure_evidence_hash=FAILURE_HASH,
    )
    definition = policy.candidates[0]
    fit = fit_dynamics_candidate_v25(snapshot, definition, policy, fitted_at=NOW)
    assert fit.status == "fit_succeeded"
    assert fit.design_diagnostic.status == "trajectory_identifiable"
    assert fit.design_diagnostic.structural_identifiability_proven is False
    assert fit.design_diagnostic.row_independence_proven is False
    assert fit.model is not None
    term_index = next(
        index for index, term in enumerate(fit.model.basis_terms) if term.term_id == "x"
    )
    assert fit.model.coefficient_matrix[0][term_index] == pytest.approx(-0.5, abs=3e-3)
    simulated = simulate_dynamics_model_v25(fit.model, [2.0], snapshot.times)
    assert simulated[-1][0] == pytest.approx(snapshot.values[-1][0], rel=3e-3)


def test_v25_integral_selection_is_bound_and_replayable() -> None:
    snapshot = _decay_snapshot()
    _, policy = _policies()
    receipt = select_dynamics_candidate_v25(
        snapshot,
        policy,
        training_points=61,
        selected_at=NOW,
    )
    assert receipt.status == "selected"
    assert receipt.estimator_arm == "window_integral_matching"
    assert receipt.policy_hash == policy.policy_hash
    replay = select_dynamics_candidate_v25(
        snapshot,
        policy,
        training_points=61,
        selected_at=NOW,
    )
    assert replay.receipt_hash == receipt.receipt_hash


def test_v25_both_estimators_abstain_on_partial_observation() -> None:
    times = [0.05 * index for index in range(101)]
    snapshot = DynamicsDataSnapshotV24.seal(
        snapshot_id="partial_v25_fixture",
        declared_state_names=["observed", "latent"],
        observed_state_names=["observed"],
        times=times,
        values=[[1.0 + 0.1 * time] for time in times],
    )
    for policy in _policies():
        fit = fit_dynamics_candidate_v25(
            snapshot, policy.candidates[0], policy, fitted_at=NOW
        )
        assert fit.status == "needs_evidence"
        assert fit.reason_codes == ["partial_state_observation"]
        assert fit.design_diagnostic.status == "partial_observation"
