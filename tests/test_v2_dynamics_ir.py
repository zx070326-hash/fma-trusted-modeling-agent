from __future__ import annotations

from datetime import datetime, timezone

import pytest

from fma.v2.dynamics_ir import (
    DynamicsDataSnapshotV24,
    default_dynamics_arm_policy,
    fit_dynamics_candidate,
    select_dynamics_candidate,
    simulate_dynamics_model,
)


NOW = datetime(2026, 7, 22, tzinfo=timezone.utc)


def _decay_snapshot(count: int = 101) -> DynamicsDataSnapshotV24:
    times = [0.05 * index for index in range(count)]
    values = [[2.0 * __import__("math").exp(-0.5 * time)] for time in times]
    return DynamicsDataSnapshotV24.seal(
        snapshot_id="decay_fixture",
        declared_state_names=["x"],
        observed_state_names=["x"],
        times=times,
        values=values,
    )


def test_dynamics_ir_fits_and_replays_a_fully_observed_decay() -> None:
    snapshot = _decay_snapshot()
    policy = default_dynamics_arm_policy("direct_generation")
    definition = policy.candidates[0]
    result = fit_dynamics_candidate(snapshot, definition, policy, fitted_at=NOW)
    assert result.status == "fit_succeeded"
    assert result.identifiability.status == "trajectory_identifiable"
    assert result.identifiability.structural_identifiability_proven is False
    assert result.model is not None
    term_index = next(
        index for index, term in enumerate(result.model.basis_terms) if term.term_id == "x"
    )
    assert result.model.coefficient_matrix[0][term_index] == pytest.approx(-0.5, abs=2e-3)
    simulated = simulate_dynamics_model(result.model, [2.0], snapshot.times)
    assert simulated[-1][0] == pytest.approx(snapshot.values[-1][0], rel=2e-3)


def test_partial_observation_abstains_before_parameter_fitting() -> None:
    times = [0.05 * index for index in range(101)]
    snapshot = DynamicsDataSnapshotV24.seal(
        snapshot_id="partial_fixture",
        declared_state_names=["observed", "latent"],
        observed_state_names=["observed"],
        times=times,
        values=[[1.0 + 0.1 * time] for time in times],
    )
    policy = default_dynamics_arm_policy("direct_generation")
    fit = fit_dynamics_candidate(snapshot, policy.candidates[0], policy, fitted_at=NOW)
    assert fit.status == "needs_evidence"
    assert fit.reason_codes == ["partial_state_observation"]
    assert fit.identifiability.status == "partial_observation"


def test_rank_deficient_trajectory_causes_policy_level_abstention() -> None:
    times = [0.05 * index for index in range(101)]
    snapshot = DynamicsDataSnapshotV24.seal(
        snapshot_id="rank_fixture",
        declared_state_names=["x", "y"],
        observed_state_names=["x", "y"],
        times=times,
        values=[[1.0 + time, 2.0 + 2.0 * time] for time in times],
    )
    policy = default_dynamics_arm_policy("direct_generation")
    receipt = select_dynamics_candidate(
        snapshot,
        policy,
        training_points=61,
        selected_at=NOW,
    )
    assert receipt.status == "needs_evidence"
    assert receipt.inner_simulation_count == 0
    assert all(score.status == "needs_evidence" for score in receipt.scores)


def test_dynamics_selection_is_bound_to_public_data_and_exact_policy() -> None:
    snapshot = _decay_snapshot()
    policy = default_dynamics_arm_policy("direct_generation")
    receipt = select_dynamics_candidate(
        snapshot,
        policy,
        training_points=61,
        selected_at=NOW,
    )
    assert receipt.status == "selected"
    assert receipt.public_data_hash == snapshot.snapshot_hash
    assert receipt.arm_policy_hash == policy.policy_hash
    replay = select_dynamics_candidate(
        snapshot,
        policy,
        training_points=61,
        selected_at=NOW,
    )
    assert replay.receipt_hash == receipt.receipt_hash
