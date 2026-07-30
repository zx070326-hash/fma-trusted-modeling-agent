from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

from fma.operator_cli_v70 import main


def test_operator_cli_intake_and_unkeyed_doctor_are_stable_json(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    task_root = tmp_path / "tasks"
    attachment = tmp_path / "brief.txt"
    attachment.write_text("untrusted attachment body", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fma-ops",
            "--task-root",
            str(task_root),
            "intake",
            "--idempotency-key",
            "cli-intake-case",
            "--objective",
            "Model the attached system with explicit uncertainty checks.",
            "--attachment",
            str(attachment),
        ],
    )
    assert main() == 0
    intake = json.loads(capsys.readouterr().out)
    assert intake["status"] == "success"
    assert intake["task"] is None
    assert intake["intake"]["attachments"][0]["logical_name"] == "brief.txt"
    assert "untrusted attachment body" not in json.dumps(intake)
    assert str(attachment.parent) not in json.dumps(intake)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fma-ops",
            "--task-root",
            str(task_root),
            "doctor",
        ],
    )
    assert main() == 0
    doctor = json.loads(capsys.readouterr().out)
    assert doctor["status"] == "NOT_RUN"
    assert doctor["operational"]["status"] == "PASS"
    assert doctor["authority"]["status"] == "NOT_RUN"


def test_operator_cli_can_create_and_project_keyed_task(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    task_root = tmp_path / "tasks"
    key_file = tmp_path / "authority.key"
    key_file.write_bytes(b"operator-cli-authority-key-000000000000000")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fma-ops",
            "--task-root",
            str(task_root),
            "intake",
            "--idempotency-key",
            "cli-create-case",
            "--objective",
            "Model the supplied public dynamics under a frozen protocol.",
            "--workspace-id",
            "cli-created",
            "--create-task",
            "--authority-key-file",
            str(key_file),
        ],
    )
    assert main() == 0
    created = json.loads(capsys.readouterr().out)
    assert created["task"]["task_id"] == "cli-created"
    assert created["task"]["workflow"]["stage_statuses"]["S0"] == "frontier"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fma-ops",
            "--task-root",
            str(task_root),
            "next",
            "--task-id",
            "cli-created",
            "--authority-key-file",
            str(key_file),
        ],
    )
    assert main() == 0
    projected = json.loads(capsys.readouterr().out)
    assert projected["packet"]["action"] == "run_s0"
    assert projected["packet"]["claim_scope"] == "workflow_control_only"
    assert projected["scientific_qualification_granted"] is False


def test_operator_cli_doctor_returns_nonzero_for_corrupt_operational_state(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    task_root = tmp_path / "tasks"
    store_args = [
        "fma-ops",
        "--task-root",
        str(task_root),
        "status",
    ]
    monkeypatch.setattr(sys, "argv", store_args)
    assert main() == 0
    capsys.readouterr()

    database = task_root / ".fma-op-v70" / "operator.sqlite3"
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(
            """
            INSERT INTO operator_events(
                work_id,event_type,payload_json,recorded_at,
                previous_event_hash,event_hash
            ) VALUES(NULL,'corrupt.event','not-json',
                     '2026-07-30T00:00:00+00:00',NULL,?)
            """,
            ("0" * 64,),
        )
        connection.commit()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fma-ops",
            "--task-root",
            str(task_root),
            "doctor",
        ],
    )
    assert main() == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "FAIL"
    assert report["operational"]["status"] == "FAIL"


def test_operator_cli_doctor_returns_nonzero_for_recovery_pending(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    task_root = tmp_path / "tasks"
    attachment = tmp_path / "brief.txt"
    attachment.write_text("recovery-pending intake", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fma-ops",
            "--task-root",
            str(task_root),
            "intake",
            "--idempotency-key",
            "cli-recovery-case",
            "--objective",
            "Model the supplied dynamics with a recoverable intake.",
            "--attachment",
            str(attachment),
        ],
    )
    assert main() == 0
    capsys.readouterr()
    (task_root / ".fma-op-v70" / "current_intake.json").unlink()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fma-ops",
            "--task-root",
            str(task_root),
            "doctor",
        ],
    )
    assert main() == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "RECOVERY_PENDING"
    assert report["operational"]["status"] == "RECOVERY_PENDING"
    assert report["scientific_qualification_granted"] is False
    assert report["real_world_action_authorized"] is False
