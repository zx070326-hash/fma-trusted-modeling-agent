from __future__ import annotations

import json
import threading
import urllib.request
from unittest.mock import MagicMock

from fma.studio.server import StudioHTTPServer


BRIDGE_TOKEN = "operator-http-test-token"
ORIGIN = "http://localhost:3001"


def _request(
    url: str,
    *,
    method: str = "GET",
) -> urllib.request.Request:
    headers = {
        "Origin": ORIGIN,
        "X-FMA-Bridge-Token": BRIDGE_TOKEN,
    }
    data = None
    if method == "POST":
        headers.update(
            {
                "Content-Type": "application/json",
            }
        )
        data = b"{}"
    return urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )


def test_v70_http_routes_are_thin_service_projections() -> None:
    service = MagicMock()
    service.operator_doctor_v70.return_value = {
        "status": "PASS",
        "scientific_qualification_granted": False,
        "real_world_action_authorized": False,
    }
    service.project_next_packet_v70.return_value = {
        "schema_version": "7.0-operator-packet",
        "action": "run_s0",
        "claim_scope": "workflow_control_only",
    }
    service.reconcile_operator_v70.return_value = {
        "status": "success",
        "authority_reconciled_work_ids": [],
    }
    service.create_task_from_intake_v70.return_value = {
        "status": "success",
        "task_id": "http-intake-task",
    }
    server = StudioHTTPServer(
        ("127.0.0.1", 0),
        service,
        token=BRIDGE_TOKEN,
        allowed_origins={ORIGIN},
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(
            _request(base + "/api/v1/doctor"),
            timeout=5,
        ) as response:
            doctor = json.load(response)
        assert doctor["status"] == "PASS"

        with urllib.request.urlopen(
            _request(base + "/api/v1/tasks/http-task/next-packet"),
            timeout=5,
        ) as response:
            projected = json.load(response)
        assert projected["packet"]["action"] == "run_s0"
        assert projected["scientific_qualification_granted"] is False

        with urllib.request.urlopen(
            _request(base + "/api/v1/operator/reconcile", method="POST"),
            timeout=5,
        ) as response:
            reconciled = json.load(response)
        assert reconciled["authority_reconciled_work_ids"] == []

        with urllib.request.urlopen(
            _request(
                base + "/api/v1/intakes/intake-0123456789abcdef01234567/create-task",
                method="POST",
            ),
            timeout=5,
        ) as response:
            created = json.load(response)
            assert response.status == 201
        assert created["task_id"] == "http-intake-task"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    service.operator_doctor_v70.assert_called_once_with()
    service.project_next_packet_v70.assert_called_once_with("http-task")
    service.reconcile_operator_v70.assert_called_once_with()
    service.create_task_from_intake_v70.assert_called_once_with(
        "intake-0123456789abcdef01234567"
    )
