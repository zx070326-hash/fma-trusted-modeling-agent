from __future__ import annotations

import numpy as np
import pytest

import fma.v6.portfolio_transaction_v69 as transaction_module
from fma.v5_2.ode_system import ODEThresholdsV52
from fma.v5.scaffold import scaffold_task_workspace
from fma.v5.stage_workspace import StageWorkspaceV50
from fma.v5.workspace_schemas import TaskWorkspaceSpecV50, WorkflowProfileV50
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
from fma.v6.portfolio_runtime_v69 import (
    PersistenceBaselinePolicyV69,
    PositiveSeriesSnapshotV69,
)
from fma.v6.portfolio_transaction_v69 import (
    PORTFOLIO_BRANCH_RECEIPT_KIND_V69,
    PORTFOLIO_COMPLETION_KIND_V69,
    PORTFOLIO_RUN_INTENT_KIND_V69,
    PORTFOLIO_RUN_KIND_V69,
    DevelopmentPortfolioTransactionV69,
)
from fma.v6.positive_log_increment_v68 import (
    PositiveLogIncrementThresholdsV68,
)
from fma.v6.recovery_kernel import ProblemSignatureV60
from tests.test_v5_stage_workspace import (
    AUTHORITY_KEY,
    AUTHORITY_KEY_ID,
    _open_stage,
    _write_s0,
    _write_s1,
)


def _workspace(tmp_path, *, evidence_scope: str = "development"):
    root = scaffold_task_workspace(
        tmp_path / "task",
        "fixture",
        "Evaluate a synthetic forecast while preserving evidence boundaries",
    )
    spec = TaskWorkspaceSpecV50.seal(
        workspace_id="fixture",
        graph_id="v5-fixture",
        objective="Evaluate a synthetic forecast while preserving evidence boundaries",
        mission_hash="1" * 64,
        evidence_snapshot_hash="2" * 64,
        evaluator_epoch="pytest-v69",
        profile=WorkflowProfileV50.seal(),
        evidence_scope=evidence_scope,
        max_nodes=128,
        max_outcomes=128,
    )
    workspace = StageWorkspaceV50.create(
        root,
        spec,
        authority_key=AUTHORITY_KEY,
        authority_key_id=AUTHORITY_KEY_ID,
    )
    _write_s0(root)
    _open_stage(workspace, "S0", actor="model")
    assert workspace.current_gate("S0") is not None
    return root, workspace


def _query(workspace) -> CapabilityQueryV68:
    assert workspace.spec.spec_hash is not None
    assert workspace.current_gate("S0") is not None
    return CapabilityQueryV68.seal(
        workspace_spec_hash=workspace.spec.spec_hash,
        s0_gate_hash=workspace.current_gate("S0"),
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


def _snapshot() -> PositiveSeriesSnapshotV69:
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
    return PositiveSeriesSnapshotV69.seal(
        task_id="v69-transaction",
        time_unit="day",
        state_unit="registered_positive_state",
        times=np.arange(35, dtype=float).tolist(),
        observations=values.tolist(),
        source_id="v69-transaction-public-source",
    )


def _transaction(root, *, fault_hook=None):
    return DevelopmentPortfolioTransactionV69(
        root,
        authority_key=AUTHORITY_KEY,
        authority_key_id=AUTHORITY_KEY_ID,
        _fault_hook=fault_hook,
    )


def _freeze(transaction, workspace):
    return transaction.freeze(
        query=_query(workspace),
        registry=default_development_capability_registry_v68(),
        branch_budget=BranchBudgetV68.seal(
            max_wall_seconds=60,
            max_cpu_seconds=60,
            max_memory_megabytes=512,
            max_artifact_bytes=2_000_000,
            max_model_calls=0,
            max_tool_calls=2,
        ),
        portfolio_budget=PortfolioBudgetV68.seal(
            max_parallel_branches=2,
            total_wall_seconds=120,
            total_cpu_seconds=120,
            total_memory_megabytes=1024,
            total_artifact_bytes=4_000_000,
            total_model_calls=0,
            total_tool_calls=4,
        ),
        time_unit="day",
        baseline_policy=PersistenceBaselinePolicyV69.seal(
            minimum_relative_improvement=0.01,
        ),
        ode_thresholds=ODEThresholdsV52.seal(bootstrap_replicates=20),
        log_increment_thresholds=PositiveLogIncrementThresholdsV68.seal(
            bootstrap_replicates=20,
        ),
    )


def test_v69_freeze_data_run_are_separate_and_idempotent(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, workspace = _workspace(tmp_path)
    transaction = _transaction(root)
    s0_before = workspace.current_gate("S0")

    frozen = _freeze(transaction, workspace)
    assert frozen.status == "FROZEN"
    assert frozen.intent is not None
    assert frozen.intent.observation_values_accessed is False
    assert frozen.intent.planned_observation_count == 35
    assert frozen.intent.state_unit == "registered_positive_state"
    assert frozen.intent.time_unit == "day"

    staged = transaction.stage_snapshot(_snapshot())
    assert staged.status == "DATA_STAGED"
    assert staged.run_intent is not None
    assert staged.run_intent.run_started is False
    reopened = transaction._open()
    assert len(
        reopened._artifacts_of_kind(PORTFOLIO_RUN_INTENT_KIND_V69)
    ) == 1
    assert not reopened._artifacts_of_kind(
        PORTFOLIO_BRANCH_RECEIPT_KIND_V69
    )
    assert not reopened._artifacts_of_kind(PORTFOLIO_RUN_KIND_V69)
    assert not reopened._artifacts_of_kind(PORTFOLIO_COMPLETION_KIND_V69)

    run = transaction.execute()
    assert run.final_decision == "SELECT"
    completed = transaction.project()
    assert completed.status == "COMPLETED"
    assert completed.protocol_hash == run.protocol_hash
    assert completed.snapshot_hash == run.snapshot_hash
    assert completed.plan_hash == run.origin_plan_hash
    assert completed.branch_statuses == {
        "branch-log-increment": "PASS",
        "branch-scalar-ode": "PASS",
    }
    assert completed.final_decision == "SELECT"
    assert completed.run_hash == run.run_hash
    counts = {
        kind: len(transaction._open()._artifacts_of_kind(kind))
        for kind in (
            PORTFOLIO_RUN_INTENT_KIND_V69,
            PORTFOLIO_BRANCH_RECEIPT_KIND_V69,
            PORTFOLIO_RUN_KIND_V69,
            PORTFOLIO_COMPLETION_KIND_V69,
        )
    }
    assert transaction.execute() == run
    replay_calls = 0
    original_execute = transaction_module.execute_development_portfolio_v69

    def counted_replay(**kwargs):
        nonlocal replay_calls
        replay_calls += 1
        return original_execute(**kwargs)

    monkeypatch.setattr(
        transaction_module,
        "execute_development_portfolio_v69",
        counted_replay,
    )
    assert transaction.reconcile().status == "COMPLETED"
    assert replay_calls == 1
    assert {
        kind: len(transaction._open()._artifacts_of_kind(kind))
        for kind in counts
    } == counts
    assert transaction._open().current_gate("S0") == s0_before


@pytest.mark.parametrize(
    ("fault_phase", "expected_status", "expected_receipts", "has_run"),
    [
        ("after_run_intent", "DATA_STAGED", 0, False),
        ("after_branch_1", "RECOVERY_PENDING", 1, False),
        ("after_run", "RECOVERY_PENDING", 2, True),
    ],
)
def test_v69_reconcile_recovers_each_partial_commit(
    tmp_path,
    fault_phase: str,
    expected_status: str,
    expected_receipts: int,
    has_run: bool,
) -> None:
    root, workspace = _workspace(tmp_path)

    def fault(phase: str) -> None:
        if phase == fault_phase:
            raise RuntimeError(f"injected-{phase}")

    transaction = _transaction(root, fault_hook=fault)
    _freeze(transaction, workspace)
    if fault_phase == "after_run_intent":
        with pytest.raises(RuntimeError, match=fault_phase):
            transaction.stage_snapshot(_snapshot())
    else:
        transaction.stage_snapshot(_snapshot())
        with pytest.raises(RuntimeError, match=fault_phase):
            transaction.execute()

    recovery = _transaction(root)
    partial = recovery.project()
    assert partial.status == expected_status
    assert len(partial.branch_receipts) == expected_receipts
    assert (partial.run is not None) is has_run

    completed = recovery.reconcile()
    assert completed.status == "COMPLETED"
    assert len(completed.branch_receipts) == 2
    assert completed.run is not None
    assert completed.completion is not None
    assert recovery.reconcile().run_hash == completed.run_hash


def test_v69_s0_change_makes_pending_transaction_stale(tmp_path) -> None:
    root, workspace = _workspace(tmp_path)
    transaction = _transaction(root)
    _freeze(transaction, workspace)
    before = transaction.project()
    assert before.status == "FROZEN"

    workspace.invalidate_from(
        "S0",
        reason="test successor S0 invalidates the frozen portfolio",
    )
    assert workspace.current_gate("S0") is None
    stale = transaction.project()
    assert stale.status == "STALE_PENDING"
    assert transaction.stage_snapshot(_snapshot()).status == "STALE_PENDING"
    assert transaction.reconcile().status == "STALE_PENDING"
    assert not transaction._open()._artifacts_of_kind(
        PORTFOLIO_RUN_INTENT_KIND_V69
    )


def test_v69_stage_snapshot_enforces_frozen_count_and_units(tmp_path) -> None:
    root, workspace = _workspace(tmp_path)
    transaction = _transaction(root)
    _freeze(transaction, workspace)
    snapshot = _snapshot()

    wrong_time = snapshot.model_dump(
        mode="json",
        exclude={"time_unit", "snapshot_hash"},
    )
    with pytest.raises(ValueError, match="frozen units/count"):
        transaction.stage_snapshot(
            PositiveSeriesSnapshotV69.seal(**wrong_time, time_unit="year")
        )
    too_long = snapshot.model_dump(
        mode="json",
        exclude={"times", "observations", "snapshot_hash"},
    )
    with pytest.raises(ValueError, match="frozen units/count"):
        transaction.stage_snapshot(
            PositiveSeriesSnapshotV69.seal(
                **too_long,
                times=[*snapshot.times, 35.0],
                observations=[*snapshot.observations, snapshot.observations[-1]],
            )
        )


def test_v69_rejects_non_development_scope_and_open_s1(tmp_path) -> None:
    nondevelopment_root, _ = _workspace(
        tmp_path / "nondevelopment",
        evidence_scope="synthetic_fixture",
    )
    with pytest.raises(PermissionError, match="development evidence scope"):
        _transaction(nondevelopment_root).project()

    root, workspace = _workspace(tmp_path / "opened-s1")
    transaction = _transaction(root)
    _freeze(transaction, workspace)
    _write_s1(root, workspace)
    _open_stage(workspace, "S1", actor="model")
    assert workspace.current_gate("S1") is not None
    with pytest.raises(PermissionError, match="cannot run after S1 opens"):
        transaction.project()
