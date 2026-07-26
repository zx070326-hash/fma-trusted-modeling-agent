from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from fma.hashing import canonical_json
from fma.v5_1.codex_stage_driver import FixtureStageRoleTransportV51
from fma.studio.server import StudioHTTPServer
from fma.studio.service import (
    StudioTaskService,
    StudioValidationError,
)


AUTHORITY_KEY = b"studio-test-authority-key-" + b"k" * 32
BRIDGE_TOKEN = "studio-test-bridge-token-123456"
OBJECTIVE = (
    "Forecast weekly emergency visits for twelve weeks so staffing can be "
    "planned, while treating understaffing as the larger error."
)


def _valid_draft(request):
    if request.role_kind == "reviewer":
        return {
            "schema_version": "5.1",
            "request_hash": request.request_hash,
            "role_name": request.role_name,
            "selected_candidate_id": None,
            "verdict": "APPROVE",
            "rationale": "The S0 task is bounded, computable, and honest.",
            "assumptions": [],
            "findings": [],
            "uncertainties": ["No empirical data has been ingested yet."],
            "proposed_artifacts": [],
            "authority_claimed": False,
        }
    decision = {
        "schema_version": "5.0",
        "function_id": "asymmetric_staffing_loss",
        "input_names": ["prediction", "target"],
        "expression": (
            "2 * max(target - prediction, 0) + "
            "max(prediction - target, 0)"
        ),
        "sense": "minimize",
        "output_unit": "visit_count",
        "canaries": [
            {
                "canary_id": "exact",
                "inputs": {"prediction": 10.0, "target": 10.0},
                "expected": 0.0,
                "tolerance": 1e-09,
            },
            {
                "canary_id": "under",
                "inputs": {"prediction": 9.0, "target": 10.0},
                "expected": 2.0,
                "tolerance": 1e-09,
            },
        ],
        "function_hash": None,
    }
    regime = {
        "schema_version": "5.0",
        "system_boundary": (
            "One emergency department and its weekly aggregate visit demand."
        ),
        "state_and_memory": (
            "Observed weekly visit counts with trend, seasonality, and lagged state."
        ),
        "uncertainty_and_data": (
            "No data has been ingested; sampling, drift, and missingness remain open."
        ),
        "decision_and_loss": (
            "A report-only forecast scored by asymmetric staffing loss."
        ),
        "query_type": "prediction",
        "downstream_decision": "Prepare a human-reviewed staffing draft.",
        "decision_function_id": "asymmetric_staffing_loss",
        "computable_decision_function": "asymmetric absolute staffing loss",
        "evidence_hashes": [request.public_inputs["evidence_snapshot_hash"]],
        "limitations": [
            "No forecast is usable until data provenance and validation pass."
        ],
        "diagnosis_hash": None,
    }
    return {
        "schema_version": "5.1",
        "request_hash": request.request_hash,
        "role_name": request.role_name,
        "selected_candidate_id": None,
        "verdict": "NOT_APPLICABLE",
        "rationale": "Two typed S0 artifacts are proposed for harness validation.",
        "assumptions": ["Weekly aggregation is meaningful."],
        "findings": [],
        "uncertainties": ["No empirical data has been ingested yet."],
        "proposed_artifacts": [
            {
                "artifact_type": "decision_function",
                "content": canonical_json(decision),
            },
            {
                "artifact_type": "regime_diagnosis",
                "content": canonical_json(regime),
            },
        ],
        "authority_claimed": False,
    }


def _service(tmp_path: Path, draft_factory=_valid_draft) -> StudioTaskService:
    return StudioTaskService(
        tmp_path / "tasks",
        authority_key=AUTHORITY_KEY,
        authority_key_id="studio-test-v1",
        role_transport_factory=lambda _: FixtureStageRoleTransportV51(
            draft_factory
        ),
    )


def test_create_task_is_idempotent_and_starts_at_s0(tmp_path: Path) -> None:
    service = _service(tmp_path)
    first = service.create_task(
        {
            "objective": OBJECTIVE,
            "workspace_id": "emergency-visits",
            "evidence_scope": "development",
        }
    )
    second = service.create_task(
        {
            "objective": OBJECTIVE,
            "workspace_id": "emergency-visits",
            "evidence_scope": "development",
        }
    )

    assert first["task_id"] == second["task_id"] == "emergency-visits"
    assert first["workflow"]["frontier_stages"] == ["S0"]
    assert first["workflow"]["stage_statuses"]["S0"] == "frontier"
    assert first["scientific_qualification_granted"] is False
    assert first["real_world_action_authorized"] is False
    assert first["events"][0]["event_type"] == "task_created"


def test_s0_runs_generator_reviewer_check_and_gate(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.create_task(
        {"objective": OBJECTIVE, "workspace_id": "emergency-visits"}
    )

    result = service.run_s0("emergency-visits")

    assert result["workflow"]["stage_statuses"]["S0"] == "gate_open"
    assert result["workflow"]["frontier_stages"] == ["S1"]
    assert result["next_valid_actions"] == ["inspect_s0", "continue_s1"]
    assert result["scientific_qualification_granted"] is False
    assert result["real_world_action_authorized"] is False
    assert result["events"][-1]["event_type"] == "s0_gate_evaluated"
    assert result["events"][-1]["details"]["decision"] == "OPEN"
    root = tmp_path / "tasks" / "emergency-visits"
    contract = json.loads(
        (root / "problem" / "contract.json").read_text(encoding="utf-8")
    )
    assert contract["question"] == OBJECTIVE
    assert (root / "problem" / "decision_function.json").is_file()
    assert (root / "docs" / "regime.json").is_file()


def test_invalid_agent_artifacts_fail_before_stage_files_exist(
    tmp_path: Path,
) -> None:
    def invalid_draft(request):
        payload = _valid_draft(request)
        if request.role_kind == "generator":
            payload["proposed_artifacts"] = []
        return payload

    service = _service(tmp_path, invalid_draft)
    service.create_task({"objective": OBJECTIVE, "workspace_id": "bad-s0"})

    with pytest.raises(StudioValidationError):
        service.run_s0("bad-s0")

    root = tmp_path / "tasks" / "bad-s0"
    assert not (root / "problem" / "contract.json").exists()
    assert not (root / "problem" / "decision_function.json").exists()
    assert not (root / "docs" / "regime.json").exists()
    assert service.snapshot("bad-s0")["workflow"]["frontier_stages"] == ["S0"]


def test_s0_uses_one_bounded_repair_after_typed_rejection(
    tmp_path: Path,
) -> None:
    generator_calls = 0

    def repairing_draft(request):
        nonlocal generator_calls
        payload = _valid_draft(request)
        if request.role_kind == "generator":
            generator_calls += 1
            if generator_calls == 1:
                payload["proposed_artifacts"] = []
        return payload

    service = _service(tmp_path, repairing_draft)
    service.create_task({"objective": OBJECTIVE, "workspace_id": "repair-s0"})

    result = service.run_s0("repair-s0")

    assert generator_calls == 2
    assert result["workflow"]["stage_statuses"]["S0"] == "gate_open"
    assert "s0_generator_rejected" in [
        event["event_type"] for event in result["events"]
    ]
    completed = next(
        event
        for event in result["events"]
        if event["event_type"] == "s0_generator_completed"
    )
    assert completed["details"]["generator_attempts"] == 2


def test_http_bridge_requires_token_for_mutation(tmp_path: Path) -> None:
    service = _service(tmp_path)
    server = StudioHTTPServer(
        ("127.0.0.1", 0),
        service,
        token=BRIDGE_TOKEN,
        allowed_origins={"http://localhost:3001"},
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        health = urllib.request.Request(
            base + "/api/v1/health",
            headers={"Origin": "http://localhost:3001"},
        )
        with urllib.request.urlopen(health, timeout=5) as response:
            assert json.load(response)["authority_key_exposed"] is False

        body = json.dumps(
            {"objective": OBJECTIVE, "workspace_id": "http-task"}
        ).encode("utf-8")
        unauthenticated = urllib.request.Request(
            base + "/api/v1/tasks",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Origin": "http://localhost:3001",
            },
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as denied:
            urllib.request.urlopen(unauthenticated, timeout=5)
        assert denied.value.code == 401

        authenticated = urllib.request.Request(
            base + "/api/v1/tasks",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Origin": "http://localhost:3001",
                "X-FMA-Bridge-Token": BRIDGE_TOKEN,
            },
            method="POST",
        )
        with urllib.request.urlopen(authenticated, timeout=5) as response:
            payload = json.load(response)
        assert payload["task_id"] == "http-task"
        assert payload["workflow"]["frontier_stages"] == ["S0"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
