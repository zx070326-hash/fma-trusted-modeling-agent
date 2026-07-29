from __future__ import annotations

import math

import pytest

from fma.v5_2.ode_system import ODETimeSeriesSnapshotV52
from fma.v6.decision_value import (
    DecisionValueContractV62,
    evaluate_decision_value_v62,
)
from fma.v6.scientific_success import (
    ScientificSuccessContractV61,
    default_scientific_success_contract_v61,
)


WORKSPACE_HASH = "a" * 64


def _snapshot(
    *,
    task_id: str,
    observations: list[float],
    state_unit: str = "count",
) -> ODETimeSeriesSnapshotV52:
    return ODETimeSeriesSnapshotV52.seal(
        task_id=task_id,
        time_unit="day",
        state_unit=state_unit,
        times=[index * 0.5 for index in range(len(observations))],
        observations=observations,
        source_id=f"{task_id}-fixture-source",
        fixture_only=True,
    )


def _logistic_observations(count: int = 36) -> list[float]:
    return [
        100.0
        / (1.0 + 19.0 * math.exp(-0.45 * index * 0.5))
        * (1.0 + 0.006 * math.sin(index * 1.7))
        for index in range(count)
    ]


def _success_contract(
    workspace_spec_hash: str = WORKSPACE_HASH,
) -> ScientificSuccessContractV61:
    return default_scientific_success_contract_v61(
        workspace_spec_hash=workspace_spec_hash,
        adapter_id="scalar_autonomous_ode_v52",
    )


def _decision_contract(
    *,
    success_contract: ScientificSuccessContractV61,
    workspace_spec_hash: str = WORKSPACE_HASH,
    success_contract_hash: str | None = None,
    action_unit: str = "count",
) -> DecisionValueContractV62:
    return DecisionValueContractV62.seal(
        workspace_spec_hash=workspace_spec_hash,
        success_contract_hash=(
            success_contract_hash or success_contract.contract_hash
        ),
        decision_id="capacity-plan",
        action_unit=action_unit,
        underage_unit_cost=1.0,
        overage_unit_cost=1.0,
    )


def test_contract_hash_binding_and_action_unit_fail_closed() -> None:
    success = _success_contract()
    snapshot = _snapshot(
        task_id="contract-binding",
        observations=_logistic_observations(),
    )
    contract = _decision_contract(success_contract=success)

    stale_contract = contract.model_copy(
        update={"underage_unit_cost": 2.0}
    )
    with pytest.raises(ValueError, match="contract is not sealed"):
        evaluate_decision_value_v62(
            snapshot=snapshot,
            success_contract=success,
            decision_contract=stale_contract,
        )

    stale_success = success.model_copy(
        update={"workspace_spec_hash": "b" * 64}
    )
    with pytest.raises(ValueError, match="success contract is not sealed"):
        evaluate_decision_value_v62(
            snapshot=snapshot,
            success_contract=stale_success,
            decision_contract=contract,
        )

    wrong_success_binding = _decision_contract(
        success_contract=success,
        success_contract_hash="b" * 64,
    )
    with pytest.raises(ValueError, match="binds another success contract"):
        evaluate_decision_value_v62(
            snapshot=snapshot,
            success_contract=success,
            decision_contract=wrong_success_binding,
        )

    wrong_workspace = _decision_contract(
        success_contract=success,
        workspace_spec_hash="b" * 64,
    )
    with pytest.raises(ValueError, match="belongs to another workspace"):
        evaluate_decision_value_v62(
            snapshot=snapshot,
            success_contract=success,
            decision_contract=wrong_workspace,
        )

    wrong_unit = _decision_contract(
        success_contract=success,
        action_unit="person",
    )
    with pytest.raises(ValueError, match="action unit differs"):
        evaluate_decision_value_v62(
            snapshot=snapshot,
            success_contract=success,
            decision_contract=wrong_unit,
        )


def test_short_series_is_not_run_and_retains_input_binding() -> None:
    success = _success_contract()
    decision = _decision_contract(success_contract=success)
    snapshot = _snapshot(
        task_id="decision-too-short",
        observations=_logistic_observations(12),
    )

    evidence = evaluate_decision_value_v62(
        snapshot=snapshot,
        success_contract=success,
        decision_contract=decision,
    )

    assert evidence.status == "NOT_RUN"
    assert evidence.reason_codes == ["insufficient_observations"]
    assert evidence.thresholds["minimum_observation_count"] == 23
    assert evidence.completed_fold_count == 0
    assert evidence.training_snapshot_hashes == []
    assert evidence.snapshot_hash == snapshot.snapshot_hash
    assert evidence.fixture_only is True
    assert evidence.local_retrospective_only is True
    assert evidence.prospective_trial_completed is False
    assert evidence.real_world_action_authorized is False


def test_stable_logistic_control_has_local_decision_value_only() -> None:
    success = _success_contract()
    decision = _decision_contract(success_contract=success)
    snapshot = _snapshot(
        task_id="stable-decision-control",
        observations=_logistic_observations(),
    )

    evidence = evaluate_decision_value_v62(
        snapshot=snapshot,
        success_contract=success,
        decision_contract=decision,
    )

    assert evidence.status == "PASS"
    assert evidence.completed_fold_count == 6
    assert evidence.admissible_fold_count == 6
    assert evidence.completed_origin_indices == [30, 31, 32, 33, 34, 35]
    assert len(evidence.training_snapshot_hashes) == 6
    assert len(evidence.model_action_hashes) == 6
    assert evidence.metrics["relative_loss_improvement"] >= 0.05
    assert evidence.metrics["mean_normalized_regret"] <= 0.20
    assert evidence.fixture_only is True
    assert evidence.local_retrospective_only is True
    assert evidence.prospective_trial_completed is False
    assert evidence.real_world_action_authorized is False


def test_oscillating_negative_control_cannot_pass_decision_gate() -> None:
    success = _success_contract()
    decision = _decision_contract(success_contract=success)
    snapshot = _snapshot(
        task_id="oscillating-decision-negative",
        observations=[
            25.0 if index % 2 == 0 else 125.0 for index in range(36)
        ],
    )

    evidence = evaluate_decision_value_v62(
        snapshot=snapshot,
        success_contract=success,
        decision_contract=decision,
    )

    assert evidence.status == "FAIL"
    assert "mean_normalized_regret_bounded" in evidence.reason_codes
    assert not all(evidence.checks.values())
    assert evidence.metrics["mean_normalized_regret"] > 0.20
    assert evidence.real_world_action_authorized is False


def test_future_mutation_does_not_change_prior_fold_inputs_or_actions() -> None:
    success = _success_contract()
    decision = _decision_contract(success_contract=success)
    observations = _logistic_observations()
    original = _snapshot(
        task_id="future-mutation",
        observations=observations,
    )
    mutated_observations = observations.copy()
    mutated_observations[33] *= 1.5
    mutated = _snapshot(
        task_id="future-mutation",
        observations=mutated_observations,
    )

    before = evaluate_decision_value_v62(
        snapshot=original,
        success_contract=success,
        decision_contract=decision,
    )
    after = evaluate_decision_value_v62(
        snapshot=mutated,
        success_contract=success,
        decision_contract=decision,
    )

    assert before.completed_origin_indices == [30, 31, 32, 33, 34, 35]
    assert after.completed_origin_indices == before.completed_origin_indices
    assert (
        after.training_snapshot_hashes[:4]
        == before.training_snapshot_hashes[:4]
    )
    assert after.model_action_hashes[:4] == before.model_action_hashes[:4]
    assert (
        after.training_snapshot_hashes[4:]
        != before.training_snapshot_hashes[4:]
    )
    assert after.actual_values_hash != before.actual_values_hash
