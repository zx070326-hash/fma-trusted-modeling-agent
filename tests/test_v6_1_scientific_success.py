from __future__ import annotations

import math

import numpy as np

from fma.v5_2.ode_system import ODETimeSeriesSnapshotV52
from fma.v6.scientific_success import (
    default_scientific_success_contract_v61,
    evaluate_rolling_confirmation_v61,
)


WORKSPACE_HASH = "a" * 64


def _snapshot(
    *,
    task_id: str,
    observations: list[float],
) -> ODETimeSeriesSnapshotV52:
    return ODETimeSeriesSnapshotV52.seal(
        task_id=task_id,
        time_unit="day",
        state_unit="count",
        times=[index * 0.5 for index in range(len(observations))],
        observations=observations,
        source_id=f"{task_id}-fixture-source",
        fixture_only=True,
    )


def _logistic_observations(count: int = 36) -> list[float]:
    times = [index * 0.5 for index in range(count)]
    return [
        100.0
        / (1.0 + 19.0 * math.exp(-0.45 * time))
        * (1.0 + 0.006 * math.sin(index * 1.7))
        for index, time in enumerate(times)
    ]


def test_success_contract_is_hash_bound_and_claim_relative() -> None:
    contract = default_scientific_success_contract_v61(
        workspace_spec_hash=WORKSPACE_HASH,
        adapter_id="scalar_autonomous_ode_v52",
    )

    assert contract.claim_kind == "predictive"
    assert contract.confirmation_method == (
        "nested_rolling_origin_one_step"
    )
    assert contract.required_dimension_ids == [
        "data_provenance",
        "leakage_safe_confirmation",
        "local_adapter_checks",
        "workflow_integrity",
    ]
    assert contract.contract_hash == contract.content_hash()
    contract.thresholds.assert_sealed()


def test_twelve_points_are_executable_but_not_confirmation_eligible() -> None:
    contract = default_scientific_success_contract_v61(
        workspace_spec_hash=WORKSPACE_HASH,
        adapter_id="scalar_autonomous_ode_v52",
    )

    confirmation = evaluate_rolling_confirmation_v61(
        snapshot=_snapshot(
            task_id="too-short",
            observations=_logistic_observations(12),
        ),
        contract=contract,
    )

    assert confirmation.status == "NOT_RUN"
    assert confirmation.reason_codes == ["insufficient_observations"]
    assert confirmation.thresholds["minimum_observation_count"] == 23


def test_nested_rolling_confirmation_passes_stable_logistic_control() -> None:
    contract = default_scientific_success_contract_v61(
        workspace_spec_hash=WORKSPACE_HASH,
        adapter_id="scalar_autonomous_ode_v52",
    )

    confirmation = evaluate_rolling_confirmation_v61(
        snapshot=_snapshot(
            task_id="stable-logistic",
            observations=_logistic_observations(),
        ),
        contract=contract,
    )

    assert confirmation.status == "PASS"
    assert confirmation.completed_fold_count == 6
    assert confirmation.admissible_fold_count == 6
    assert confirmation.selected_model_ids == ["logistic"] * 6
    assert (
        confirmation.metrics["persistence_relative_improvement"] >= 0.10
    )
    assert confirmation.actual_values_hash
    assert confirmation.prediction_values_hash
    assert confirmation.persistence_values_hash


def test_nested_confirmation_rejects_oscillating_false_success() -> None:
    contract = default_scientific_success_contract_v61(
        workspace_spec_hash=WORKSPACE_HASH,
        adapter_id="scalar_autonomous_ode_v52",
    )
    observations = [
        25.0 if index % 2 == 0 else 125.0 for index in range(36)
    ]

    confirmation = evaluate_rolling_confirmation_v61(
        snapshot=_snapshot(
            task_id="oscillating-negative",
            observations=observations,
        ),
        contract=contract,
    )

    assert confirmation.status == "FAIL"
    assert confirmation.reason_codes
    assert not all(confirmation.checks.values())
    assert np.isfinite(
        float(confirmation.metrics["confirmation_relative_rmse"])
    )
