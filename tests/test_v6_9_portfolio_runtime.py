from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from fma.v5_2.ode_system import ODEThresholdsV52
from fma.v6.capability_catalog_v68 import (
    default_development_capability_registry_v68,
)
from fma.v6.capability_sdk_v68 import (
    CapabilityQueryV68,
    MeasurementSignatureV68,
)
from fma.v6.portfolio_protocol_v68 import (
    BranchBudgetV68,
    PortfolioBudgetV68,
)
from fma.v6 import portfolio_runtime_v69 as runtime_v69
from fma.v6.portfolio_runtime_v69 import (
    MAXIMUM_ROLLING_ORIGINS_V69,
    PersistenceBaselinePolicyV69,
    PositiveSeriesSnapshotV69,
    build_common_rolling_origin_plan_v69,
    compile_default_development_portfolio_protocol_v69,
    execute_development_portfolio_v69,
    verify_development_portfolio_run_v69,
)
from fma.v6.positive_log_increment_v68 import (
    PositiveLogIncrementThresholdsV68,
)
from fma.v6.recovery_kernel import ProblemSignatureV60


def _query() -> CapabilityQueryV68:
    return CapabilityQueryV68.seal(
        workspace_spec_hash="a" * 64,
        s0_gate_hash="b" * 64,
        problem_signature=ProblemSignatureV60(
            state_kind="scalar",
            time_kind="continuous",
            dynamics_kind="autonomous",
            observation_kind="complete",
            task_kind="prediction",
            observation_count=35,
            positive_observations=True,
            strictly_increasing_time=True,
        ),
        claim_kind="predictive",
        measurement=MeasurementSignatureV68(
            measurement_contract_hash="c" * 64,
            scale_type="ratio",
            study_design_type="time_series",
            missingness_policy="reject_incomplete_series",
            measurement_unit="registered_positive_state",
            time_basis="day",
            minimum_planned_observations=35,
        ),
    )


def _protocol(*, tie_tolerance: float = 1e-12):
    branch_budget = BranchBudgetV68.seal(
        max_wall_seconds=60,
        max_cpu_seconds=60,
        max_memory_megabytes=512,
        max_artifact_bytes=2_000_000,
        max_model_calls=0,
        max_tool_calls=2,
    )
    return compile_default_development_portfolio_protocol_v69(
        query=_query(),
        registry=default_development_capability_registry_v68(),
        branch_budget=branch_budget,
        portfolio_budget=PortfolioBudgetV68.seal(
            max_parallel_branches=2,
            total_wall_seconds=120,
            total_cpu_seconds=120,
            total_memory_megabytes=1024,
            total_artifact_bytes=4_000_000,
            total_model_calls=0,
            total_tool_calls=4,
        ),
        tie_tolerance=tie_tolerance,
    )


def _snapshot(
    task_id: str,
    values: np.ndarray,
) -> PositiveSeriesSnapshotV69:
    return PositiveSeriesSnapshotV69.seal(
        task_id=task_id,
        time_unit="day",
        state_unit="registered_positive_state",
        times=np.arange(len(values), dtype=float).tolist(),
        observations=values.tolist(),
        source_id=f"{task_id}-public-source",
    )


def _thresholds() -> tuple[
    ODEThresholdsV52,
    PositiveLogIncrementThresholdsV68,
]:
    return (
        ODEThresholdsV52.seal(bootstrap_replicates=20),
        PositiveLogIncrementThresholdsV68.seal(bootstrap_replicates=20),
    )


def _execute(snapshot: PositiveSeriesSnapshotV69, *, tie_tolerance=1e-12):
    protocol = _protocol(tie_tolerance=tie_tolerance)
    plan = build_common_rolling_origin_plan_v69(snapshot)
    ode_thresholds, log_thresholds = _thresholds()
    policy = PersistenceBaselinePolicyV69.seal(
        minimum_relative_improvement=0.01,
    )
    run = execute_development_portfolio_v69(
        protocol=protocol,
        snapshot=snapshot,
        origin_plan=plan,
        ode_thresholds=ode_thresholds,
        log_increment_thresholds=log_thresholds,
        baseline_policy=policy,
    )
    return protocol, plan, ode_thresholds, log_thresholds, policy, run


def test_v69_real_two_pack_vertical_slice_is_hash_bound_and_guarded() -> None:
    rng = np.random.default_rng(103)
    growths = np.zeros(34, dtype=float)
    growths[0] = 0.04
    for index in range(1, len(growths)):
        growths[index] = (
            0.04
            + 0.85 * (growths[index - 1] - 0.04)
            + rng.normal(0.0, 0.02)
        )
    values = 100.0 * np.exp(
        np.concatenate(([0.0], np.cumsum(growths)))
    )
    snapshot = _snapshot("v69-happy", values)
    (
        protocol,
        plan,
        ode_thresholds,
        log_thresholds,
        policy,
        run,
    ) = _execute(snapshot)

    assert protocol.runtime_mode == "development_sandbox"
    assert {item.capability_pack_id for item in protocol.branches} == {
        "positive_log_increment_v68",
        "scalar_autonomous_ode_v52",
    }
    assert plan.initial_training_count == 34
    assert [item.training_count for item in plan.origins] == [34]
    assert run.inner_selection.decision == "SELECT"
    assert run.final_decision == "SELECT"
    assert run.persistence_relative_improvement is not None
    assert run.persistence_relative_improvement > (
        policy.minimum_relative_improvement
    )
    assert run.scientific_qualification_granted is False
    assert run.real_world_action_authorized is False
    assert run.run_is_stage_evidence is False
    assert all(item.execution_status == "PASS" for item in run.evaluations)
    assert all(
        item.execution_evidence_hash == receipt.receipt_hash
        for item, receipt in zip(
            run.evaluations,
            run.branch_receipts,
            strict=True,
        )
    )
    assert all(
        receipt.parameter_count
        == max(
            int(item.parameter_count)
            for item in receipt.origin_receipts
            if item.parameter_count is not None
        )
        for receipt in run.branch_receipts
    )
    assert all(
        item.scientific_qualification_granted is False
        and item.real_world_action_authorized is False
        for receipt in run.branch_receipts
        for item in receipt.origin_receipts
    )
    assert verify_development_portfolio_run_v69(
        protocol=protocol,
        snapshot=snapshot,
        origin_plan=plan,
        ode_thresholds=ode_thresholds,
        log_increment_thresholds=log_thresholds,
        baseline_policy=policy,
        run=run,
    )

    tampered_receipt = run.branch_receipts[0].model_copy(
        update={"parameter_count": 99}
    )
    with pytest.raises(ValueError, match="not sealed|inconsistent"):
        tampered_receipt.assert_sealed()
    forged_run = run.model_copy(update={"run_hash": "f" * 64})
    assert not verify_development_portfolio_run_v69(
        protocol=protocol,
        snapshot=snapshot,
        origin_plan=plan,
        ode_thresholds=ode_thresholds,
        log_increment_thresholds=log_thresholds,
        baseline_policy=policy,
        run=forged_run,
    )


def test_v69_both_packs_losing_to_persistence_forces_abstention() -> None:
    growths = np.concatenate((np.full(33, 0.03), [-0.20]))
    values = 100.0 * np.exp(
        np.concatenate(([0.0], np.cumsum(growths)))
    )
    snapshot = _snapshot("v69-break", values)
    *_, run = _execute(snapshot, tie_tolerance=0.0)

    assert run.inner_selection.decision == "SELECT"
    assert all(
        loss >= run.persistence_baseline.rmse
        for loss in run.inner_selection.common_loss_by_branch.values()
    )
    assert run.final_decision == "ABSTAIN"
    assert run.selected_branch_id is None
    assert run.persistence_relative_improvement is not None
    assert run.persistence_relative_improvement < 0
    assert run.reason_code == "persistence-baseline-not-beaten"


def test_v69_controlled_pack_failure_is_retained_and_forces_abstention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_log_pack(**_kwargs):
        raise RuntimeError("deterministic injected optimizer failure")

    monkeypatch.setattr(
        runtime_v69,
        "execute_positive_log_increment_ir_v68",
        fail_log_pack,
    )
    times = np.arange(35, dtype=float)
    snapshot = _snapshot(
        "v69-pack-failure",
        20.0 * np.exp(0.025 * times),
    )
    *_, run = _execute(snapshot)

    by_pack = {
        item.capability_pack_id: item for item in run.branch_receipts
    }
    failed = by_pack["positive_log_increment_v68"]
    assert failed.execution_status == "FAIL"
    assert failed.predictions == []
    assert failed.failure_reason == "one-or-more-origins-failed"
    assert failed.origin_receipts[0].failure_reason == (
        "controlled-pack-runtime-failure"
    )
    assert failed.origin_receipts[0].failure_phase == "pack-execution"
    assert failed.origin_receipts[0].output_bundle_hash is None
    assert failed.origin_receipts[0].verifier_evidence_hash is None
    assert failed.origin_receipts[0].failure_evidence_hash is not None
    assert by_pack["scalar_autonomous_ode_v52"].execution_status == "PASS"
    assert run.inner_selection.decision == "ABSTAIN"
    assert run.inner_selection.reason_code == "insufficient_completed_branches"
    assert run.final_decision == "ABSTAIN"
    assert run.reason_code == "inner-selection-abstained"


def test_v69_unknown_runtime_integrity_failure_is_not_downgraded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_integrity(**_kwargs):
        raise RuntimeError("authority binding corrupted")

    monkeypatch.setattr(
        runtime_v69,
        "execute_positive_log_increment_ir_v68",
        fail_integrity,
    )
    times = np.arange(35, dtype=float)
    snapshot = _snapshot(
        "v69-integrity-failure",
        20.0 * np.exp(0.025 * times),
    )

    with pytest.raises(RuntimeError, match="authority binding corrupted"):
        _execute(snapshot)


def test_v69_inapplicable_log_cadence_cannot_enter_outer_selection() -> None:
    times = np.arange(35, dtype=float)
    times[10:] += 0.5
    values = 20.0 * np.exp(0.025 * times)
    snapshot = PositiveSeriesSnapshotV69.seal(
        task_id="v69-irregular-cadence",
        time_unit="day",
        state_unit="registered_positive_state",
        times=times.tolist(),
        observations=values.tolist(),
        source_id="v69-irregular-cadence-public-source",
    )
    *_, run = _execute(snapshot)
    by_pack = {
        item.capability_pack_id: item for item in run.branch_receipts
    }

    log_receipt = by_pack["positive_log_increment_v68"]
    assert log_receipt.execution_status == "FAIL"
    assert log_receipt.origin_receipts[0].failure_reason == (
        "pack-level-obligation-not-passed"
    )
    assert run.inner_selection.decision == "ABSTAIN"
    assert run.final_decision == "ABSTAIN"
    assert run.selected_branch_id is None


def test_v69_default_origin_plan_has_a_code_owned_hard_cap() -> None:
    snapshot = _snapshot("v69-origin-cap", np.full(100, 100.0))
    plan = build_common_rolling_origin_plan_v69(snapshot)

    assert len(plan.origins) == MAXIMUM_ROLLING_ORIGINS_V69
    with pytest.raises(ValueError, match="maximum_origins"):
        build_common_rolling_origin_plan_v69(
            snapshot,
            maximum_origins=MAXIMUM_ROLLING_ORIGINS_V69 + 1,
        )


def test_v69_rejects_mismatched_data_view_before_execution() -> None:
    times = np.arange(35, dtype=float)
    original = _snapshot("v69-view-a", 20.0 * np.exp(0.02 * times))
    different = _snapshot("v69-view-b", 20.0 * np.exp(0.03 * times))
    plan = build_common_rolling_origin_plan_v69(original)
    ode_thresholds, log_thresholds = _thresholds()

    with pytest.raises(ValueError, match="different snapshot data view"):
        execute_development_portfolio_v69(
            protocol=_protocol(),
            snapshot=different,
            origin_plan=plan,
            ode_thresholds=ode_thresholds,
            log_increment_thresholds=log_thresholds,
            baseline_policy=PersistenceBaselinePolicyV69.seal(),
        )


def test_v69_rejects_tampered_and_short_snapshots() -> None:
    times = np.arange(35, dtype=float)
    snapshot = _snapshot("v69-tamper", 30.0 * np.exp(0.01 * times))
    tampered = snapshot.model_copy(
        update={"observations": [*snapshot.observations[:-1], 999.0]}
    )
    with pytest.raises(ValueError, match="snapshot (hash differs|is not sealed)"):
        build_common_rolling_origin_plan_v69(tampered)

    with pytest.raises(ValidationError):
        _snapshot("v69-short", np.full(34, 100.0))
    with pytest.raises(ValidationError, match="positive"):
        _snapshot(
            "v69-nonpositive",
            np.concatenate((np.full(34, 100.0), [0.0])),
        )
