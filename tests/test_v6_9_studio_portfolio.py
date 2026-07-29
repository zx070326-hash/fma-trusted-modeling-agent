from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from fma.studio.server import StudioHTTPServer
from fma.studio.service import (
    StudioBridgeError,
    StudioConflictError,
    StudioValidationError,
)
from fma.v6.portfolio_transaction_v69 import (
    PORTFOLIO_BRANCH_RECEIPT_KIND_V69,
    PORTFOLIO_COMPLETION_KIND_V69,
    PORTFOLIO_RUN_INTENT_KIND_V69,
    PORTFOLIO_RUN_KIND_V69,
    PORTFOLIO_TRANSACTION_INTENT_KIND_V69,
    DevelopmentPortfolioTransactionV69,
)
from tests.test_studio_bridge import (
    AUTHORITY_KEY,
    BRIDGE_TOKEN,
    OBJECTIVE,
    _service,
    _s1_draft,
)


_PORTFOLIO_KINDS = (
    PORTFOLIO_TRANSACTION_INTENT_KIND_V69,
    PORTFOLIO_RUN_INTENT_KIND_V69,
    PORTFOLIO_BRANCH_RECEIPT_KIND_V69,
    PORTFOLIO_RUN_KIND_V69,
    PORTFOLIO_COMPLETION_KIND_V69,
)


def _open_v69_task(service, task_id: str) -> dict:
    service.create_task(
        {
            "objective": OBJECTIVE,
            "workspace_id": task_id,
            "evidence_scope": "development",
            "workflow_mode": "legacy",
        }
    )
    return service.run_s0(task_id)


def _prepare_payload(*, count: int = 35) -> dict:
    return {
        "schema_version": "6.9-studio-portfolio-prepare",
        "planned_observation_count": count,
        "state_unit": "registered_positive_state",
        "time_unit": "day",
        "initial_training_count": 34,
        "min_relative_improvement": 0.01,
    }


def _series_payload(*, count: int = 35) -> dict:
    rng = np.random.default_rng(103)
    growths = np.zeros(count - 1, dtype=float)
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
    return {
        "schema_version": "6.9-studio-portfolio-series",
        "times": np.arange(count, dtype=float).tolist(),
        "observations": values.tolist(),
        "source_id": "caller-declared-public-v69-series",
        "public_data_only": True,
    }


def _gate_hashes(workspace) -> dict[str, str | None]:
    return {
        stage: workspace.current_gate(stage)
        for stage in ("S0", "S1", "S2", "S3", "S4", "S5", "S6")
    }


def _portfolio_artifact_counts(workspace) -> dict[str, int]:
    return {
        kind: len(workspace._artifacts_of_kind(kind))
        for kind in _PORTFOLIO_KINDS
    }


def test_v69_studio_runs_development_side_lane_without_stage_authority(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    after_s0 = _open_v69_task(service, "studio-v69")
    workspace = service._workspace("studio-v69")
    gates_before = _gate_hashes(workspace)

    assert after_s0["portfolio_v69"]["available"] is True
    assert after_s0["portfolio_v69"]["transaction_status"] == "NOT_STARTED"
    assert "prepare_portfolio_v69" in after_s0["next_valid_actions"]

    prepared = service.prepare_portfolio_v69(
        "studio-v69",
        _prepare_payload(),
    )
    assert prepared["portfolio_v69"]["transaction_status"] == "PREPARED"
    assert prepared["next_valid_actions"] == [
        "inspect_s0",
        "ingest_portfolio_v69",
    ]
    with pytest.raises(StudioConflictError, match="S1 is blocked"):
        service.run_s1("studio-v69")
    with pytest.raises(StudioConflictError, match="S1 is blocked"):
        service.start_s1("studio-v69")

    staged = service.ingest_portfolio_series_v69(
        "studio-v69",
        _series_payload(),
    )
    assert staged["portfolio_v69"]["transaction_status"] == "DATA_READY"
    assert staged["portfolio_v69"]["branch_statuses"] == {}
    assert staged["next_valid_actions"] == [
        "inspect_s0",
        "run_portfolio_v69",
    ]

    accepted = service.start_portfolio_v69("studio-v69")
    assert accepted["portfolio_v69"]["transaction_status"] in {
        "DATA_READY",
        "RUN_PENDING",
        "COMPLETED",
    }
    for _ in range(500):
        completed = service.snapshot("studio-v69")
        if completed["activity"] not in {"accepted", "running"}:
            break
        time.sleep(0.01)
    else:
        pytest.fail("V6.9 portfolio worker did not become terminal")
    portfolio = completed["portfolio_v69"]
    assert portfolio["transaction_status"] == "COMPLETED"
    assert portfolio["decision"] == "SELECT"
    assert portfolio["selected_branch_id"] is not None
    assert portfolio["baseline_guard_status"] == "PASS"
    assert portfolio["branch_statuses"] == {
        "branch-log-increment": "PASS",
        "branch-scalar-ode": "PASS",
    }
    assert portfolio["protocol_hash"]
    assert portfolio["snapshot_hash"]
    assert portfolio["outer_origin_plan_hash"]
    assert portfolio["run_hash"] == portfolio["decision_hash"]
    assert portfolio["engineering_status"] == "COMPLETED"
    assert portfolio["scientific_evidence_status"] == "NOT_RUN"
    assert portfolio["claim_ceiling"] == "development_protocol_only"
    assert portfolio["s1_s6_gates_touched"] is False
    assert portfolio["scientific_qualification_granted"] is False
    assert portfolio["real_world_action_authorized"] is False

    workspace = service._workspace("studio-v69")
    assert _gate_hashes(workspace) == gates_before
    assert gates_before["S0"] is not None
    assert all(gates_before[stage] is None for stage in ("S1", "S2", "S3", "S4", "S5", "S6"))
    assert workspace.verify()

    counts = _portfolio_artifact_counts(workspace)
    completion_events = [
        event
        for event in completed["events"]
        if event["event_type"] == "portfolio_v69_run_completed"
    ]
    service.prepare_portfolio_v69("studio-v69", _prepare_payload())
    service.ingest_portfolio_series_v69("studio-v69", _series_payload())
    service.run_portfolio_v69("studio-v69")
    replay = service.reconcile_portfolio_v69("studio-v69")
    assert _portfolio_artifact_counts(service._workspace("studio-v69")) == counts
    assert len(
        [
            event
            for event in replay["events"]
            if event["event_type"] == "portfolio_v69_run_completed"
        ]
    ) == len(completion_events) == 1


def test_v69_studio_rejects_other_modes_and_stale_s0(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    service.create_task(
        {
            "objective": OBJECTIVE,
            "workspace_id": "v69-v67",
            "evidence_scope": "development",
            "workflow_mode": "v67",
        }
    )
    service.create_task(
        {
            "objective": OBJECTIVE,
            "workspace_id": "v69-public",
            "evidence_scope": "public_data",
            "workflow_mode": "legacy",
        }
    )
    assert service.snapshot("v69-v67")["portfolio_v69"]["available"] is False
    assert service.snapshot("v69-public")["portfolio_v69"]["available"] is False
    with pytest.raises(StudioConflictError, match="development-scope legacy"):
        service.prepare_portfolio_v69("v69-v67", _prepare_payload())
    with pytest.raises(StudioConflictError, match="development-scope legacy"):
        service.prepare_portfolio_v69("v69-public", _prepare_payload())

    _open_v69_task(service, "v69-stale")
    service.prepare_portfolio_v69("v69-stale", _prepare_payload())
    workspace = service._workspace("v69-stale")
    workspace.invalidate_from(
        "S0",
        reason="test successor authority makes the V6.9 intent stale",
    )
    counts = _portfolio_artifact_counts(workspace)
    assert service.snapshot("v69-stale")["portfolio_v69"][
        "transaction_status"
    ] == "STALE_PENDING"
    with pytest.raises(StudioConflictError, match="stale S0"):
        service.ingest_portfolio_series_v69("v69-stale", _series_payload())
    with pytest.raises(StudioConflictError, match="stale S0"):
        service.run_portfolio_v69("v69-stale")
    with pytest.raises(StudioConflictError, match="stale S0"):
        service.reconcile_portfolio_v69("v69-stale")
    assert _portfolio_artifact_counts(service._workspace("v69-stale")) == counts


def test_v69_studio_reconciles_partial_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    _open_v69_task(service, "v69-recovery")
    service.prepare_portfolio_v69("v69-recovery", _prepare_payload())
    service.ingest_portfolio_series_v69("v69-recovery", _series_payload())
    fault_fired = False

    def transaction_with_one_fault(workspace):
        def fault(phase: str) -> None:
            nonlocal fault_fired
            if phase == "after_branch_1" and not fault_fired:
                fault_fired = True
                raise RuntimeError("injected portfolio interruption")

        return DevelopmentPortfolioTransactionV69(
            workspace.root,
            authority_key=AUTHORITY_KEY,
            authority_key_id="studio-test-v1",
            _fault_hook=fault,
        )

    monkeypatch.setattr(
        service,
        "_portfolio_transaction_v69",
        transaction_with_one_fault,
    )
    with pytest.raises(RuntimeError, match="injected portfolio interruption"):
        service.run_portfolio_v69("v69-recovery")
    partial = service.snapshot("v69-recovery")
    assert partial["portfolio_v69"]["transaction_status"] == "RUN_PENDING"
    assert partial["portfolio_v69"]["recovery_available"] is True

    monkeypatch.setattr(
        service,
        "_portfolio_transaction_v69",
        lambda workspace: DevelopmentPortfolioTransactionV69(
            workspace.root,
            authority_key=AUTHORITY_KEY,
            authority_key_id="studio-test-v1",
        ),
    )
    completed = service.reconcile_portfolio_v69("v69-recovery")
    assert completed["portfolio_v69"]["transaction_status"] == "COMPLETED"
    assert completed["portfolio_v69"]["decision"] == "SELECT"


def test_v69_freeze_and_s1_race_cannot_create_mixed_authority(
    tmp_path: Path,
) -> None:
    portfolio_service = _service(tmp_path, _s1_draft)
    _open_v69_task(portfolio_service, "v69-race")
    s1_service = _service(tmp_path, _s1_draft)
    barrier = threading.Barrier(2)

    def invoke(action) -> str:
        barrier.wait(timeout=5)
        try:
            action()
        except StudioConflictError:
            return "CONFLICT"
        return "SUCCEEDED"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = [
            future.result(timeout=30)
            for future in (
                pool.submit(
                    invoke,
                    lambda: portfolio_service.prepare_portfolio_v69(
                        "v69-race",
                        _prepare_payload(),
                    ),
                ),
                pool.submit(
                    invoke,
                    lambda: s1_service.run_s1("v69-race"),
                ),
            )
        ]

    assert sorted(outcomes) == ["CONFLICT", "SUCCEEDED"]
    workspace = portfolio_service._workspace("v69-race")
    portfolio_frozen = bool(
        workspace._artifacts_of_kind(PORTFOLIO_TRANSACTION_INTENT_KIND_V69)
    )
    s1_open = workspace.current_gate("S1") is not None
    assert portfolio_frozen is not s1_open
    assert workspace.verify()


def test_v69_studio_errors_do_not_echo_series_or_malformed_authority(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    _open_v69_task(service, "v69-safe-errors")
    service.prepare_portfolio_v69("v69-safe-errors", _prepare_payload())
    invalid = _series_payload()
    secret_value = -987654321.12345
    invalid["observations"][7] = secret_value
    with pytest.raises(StudioValidationError) as rejected:
        service.ingest_portfolio_series_v69("v69-safe-errors", invalid)
    assert str(secret_value) not in str(rejected.value)

    _open_v69_task(service, "v69-malformed-authority")
    workspace = service._workspace("v69-malformed-authority")
    secret_marker = "do-not-echo-malformed-authority-payload"
    workspace.commit_evidence(
        PORTFOLIO_TRANSACTION_INTENT_KIND_V69,
        {"secret_marker": secret_marker},
    )
    with pytest.raises(
        StudioBridgeError,
        match="integrity verification failed closed",
    ) as integrity_failure:
        service.snapshot("v69-malformed-authority")
    assert secret_marker not in str(integrity_failure.value)


def test_v69_http_routes_preserve_auth_and_action_shapes() -> None:
    service = MagicMock()
    service.prepare_portfolio_v69.return_value = {"action": "prepare"}
    service.ingest_portfolio_series_v69.return_value = {"action": "data"}
    service.start_portfolio_v69.return_value = {"action": "run"}
    service.reconcile_portfolio_v69.return_value = {"action": "reconcile"}
    server = StudioHTTPServer(
        ("127.0.0.1", 0),
        service,
        token=BRIDGE_TOKEN,
        allowed_origins={"http://localhost:3001"},
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    headers = {
        "Content-Type": "application/json",
        "Origin": "http://localhost:3001",
        "X-FMA-Bridge-Token": BRIDGE_TOKEN,
    }
    try:
        denied = urllib.request.Request(
            base + "/api/v1/tasks/http-v69/portfolio-v69/prepare",
            data=json.dumps(_prepare_payload()).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as unauthenticated:
            urllib.request.urlopen(denied, timeout=5)
        assert unauthenticated.value.code == 401

        actions = [
            (
                "prepare",
                _prepare_payload(),
                201,
            ),
            (
                "data",
                _series_payload(),
                201,
            ),
            (
                "run",
                None,
                202,
            ),
            (
                "reconcile",
                None,
                200,
            ),
        ]
        for action, payload, expected_status in actions:
            request = urllib.request.Request(
                base + f"/api/v1/tasks/http-v69/portfolio-v69/{action}",
                data=(
                    json.dumps(payload).encode("utf-8")
                    if payload is not None
                    else None
                ),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                assert response.status == expected_status
                assert json.load(response)["action"] == action

        service.prepare_portfolio_v69.assert_called_once_with(
            "http-v69",
            _prepare_payload(),
        )
        service.ingest_portfolio_series_v69.assert_called_once_with(
            "http-v69",
            _series_payload(),
        )
        service.start_portfolio_v69.assert_called_once_with("http-v69")
        service.reconcile_portfolio_v69.assert_called_once_with("http-v69")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
