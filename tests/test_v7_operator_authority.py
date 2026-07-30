from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from fma.hashing import sha256_value
from fma.operator_v70 import (
    IntakeManifestV70,
    OperatorAuthorityBindingV70,
    OperatorSubmissionV70,
    capture_file_manifest,
)
from fma.studio.service import StudioConflictError, StudioTaskService


AUTHORITY_KEY = b"v7-operator-authority-test-key-000000000000"


def _service(task_root: Path) -> StudioTaskService:
    return StudioTaskService(
        task_root,
        authority_key=AUTHORITY_KEY,
        authority_key_id="operator-test-key",
    )


def test_next_packet_is_a_pure_graph_bound_projection(tmp_path: Path) -> None:
    service = _service(tmp_path / "tasks")
    service.create_task(
        {
            "objective": "Estimate a trustworthy dynamic model from public observations.",
            "workspace_id": "packet-projection",
        }
    )
    assert service.operator_store.list_work("packet-projection") == []

    first = service.project_next_packet_v70("packet-projection")
    second = service.project_next_packet_v70("packet-projection")
    assert first == second
    assert first is not None
    assert first["action"] == "run_s0"
    assert first["claim_scope"] == "workflow_control_only"
    assert first["authority_binding"]["stage_statuses"]["S0"] == "frontier"
    assert service.operator_store.list_work("packet-projection") == []


def test_operator_database_cannot_mint_a_stage_gate_or_scientific_authority(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path / "tasks")
    snapshot = service.create_task(
        {
            "objective": "Estimate a trustworthy dynamic model from public observations.",
            "workspace_id": "no-authority-forgery",
        }
    )
    packet = service._operator_packet_v70(
        service._workspace("no-authority-forgery"),
        "run_s0",
    )
    work = service.operator_store.ensure_work(packet)
    with closing(
        sqlite3.connect(service.operator_store.database_path)
    ) as connection:
        connection.execute(
            """
            UPDATE work_items SET status='ACCEPTED',
              authority_projection_json='{"status":"ACCEPTED"}'
            WHERE work_id=?
            """,
            (work["work_id"],),
        )
        connection.commit()

    workspace = service._workspace("no-authority-forgery")
    assert workspace.current_gate("S0") is None
    after = service.snapshot("no-authority-forgery")
    assert after["workflow"]["stage_statuses"]["S0"] == "frontier"
    assert after["scientific_qualification_granted"] is False
    assert after["real_world_action_authorized"] is False
    assert snapshot["scientific_success"]["scientific_success_status"] == "NOT_RUN"
    assert service.operator_store.doctor()["status"] == "FAIL"


def test_intake_bound_task_is_atomically_installed_and_hash_bound(
    tmp_path: Path,
) -> None:
    task_root = tmp_path / "tasks"
    service = _service(task_root)
    attachment = tmp_path / "problem.md"
    attachment.write_text(
        "Fit competing growth laws and report predictive uncertainty.",
        encoding="utf-8",
    )
    published = service.publish_intake_v70(
        idempotency_key="bound-intake-case",
        objective="Fit and compare growth models using the attached public brief.",
        attachment_paths=(attachment,),
        workspace_id="intake-bound",
        evidence_scope="public_data",
        workflow_mode="v67",
    )
    manifest = IntakeManifestV70.model_validate(published["intake"])
    snapshot = service.create_task_from_intake_v70(manifest.intake_id)

    task = task_root / "intake-bound"
    assert task.is_dir()
    assert (task / "problem" / "intake" / "manifest.json").is_file()
    installed = IntakeManifestV70.model_validate_json(
        (task / "problem" / "intake" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert installed.manifest_hash == manifest.manifest_hash
    assert snapshot["workflow"]["stage_statuses"]["S0"] == "frontier"
    assert snapshot["next_packet_v70"]["action"] == "run_s0"
    assert snapshot["scientific_qualification_granted"] is False
    assert service.operator_store.doctor()["status"] == "PASS"


def test_intake_task_replay_uses_the_original_binding_after_graph_progress(
    tmp_path: Path,
) -> None:
    task_root = tmp_path / "tasks"
    service = _service(task_root)
    attachment = tmp_path / "problem.md"
    attachment.write_text(
        "Fit competing growth laws and report predictive uncertainty.",
        encoding="utf-8",
    )
    published = service.publish_intake_v70(
        idempotency_key="bound-intake-progress",
        objective="Fit and compare growth models using the attached public brief.",
        attachment_paths=(attachment,),
        workspace_id="intake-progress",
        evidence_scope="public_data",
        workflow_mode="v67",
    )
    manifest = IntakeManifestV70.model_validate(published["intake"])
    service.create_task_from_intake_v70(manifest.intake_id)
    workspace = service._workspace("intake-progress")
    original = service.operator_store.get_intake_binding(
        manifest.intake_id,
        workspace_id="intake-progress",
    )
    workspace.commit_evidence(
        "operator_replay_progress_test",
        {"status": "development_only"},
    )

    replay = service.create_task_from_intake_v70(manifest.intake_id)
    after = service.operator_store.get_intake_binding(
        manifest.intake_id,
        workspace_id="intake-progress",
    )
    assert replay["task_id"] == "intake-progress"
    assert after == original


def test_studio_doctor_reports_corrupt_task_instead_of_hiding_it(
    tmp_path: Path,
) -> None:
    task_root = tmp_path / "tasks"
    service = _service(task_root)
    corrupt = task_root / "corrupt-task" / ".fma"
    corrupt.mkdir(parents=True)
    listing = service.list_tasks()
    assert listing["health"] == "degraded"
    assert listing["corrupt_items"] == [
        {
            "task_id": "corrupt-task",
            "health": "corrupt",
            "reason_code": "workspace_spec_missing",
        }
    ]
    doctor = service.operator_doctor_v70()
    assert doctor["status"] == "FAIL"
    assert doctor["authority"]["errors"]["corrupt-task"] == (
        "workspace_spec_missing"
    )


def test_workspace_intake_drift_blocks_next_packet_and_authority_doctor(
    tmp_path: Path,
) -> None:
    task_root = tmp_path / "tasks"
    service = _service(task_root)
    attachment = tmp_path / "problem.md"
    attachment.write_text(
        "Fit competing growth laws and report predictive uncertainty.",
        encoding="utf-8",
    )
    published = service.publish_intake_v70(
        idempotency_key="bound-intake-drift",
        objective="Fit and compare growth models using the attached public brief.",
        attachment_paths=(attachment,),
        workspace_id="intake-drift",
        evidence_scope="public_data",
        workflow_mode="v67",
    )
    manifest = IntakeManifestV70.model_validate(published["intake"])
    service.create_task_from_intake_v70(manifest.intake_id)
    installed = (
        task_root
        / "intake-drift"
        / "problem"
        / "intake"
        / "attachments"
        / "problem.md"
    )
    installed.write_text("changed after binding", encoding="utf-8")

    with pytest.raises(StudioConflictError, match="changed workspace intake"):
        service.project_next_packet_v70("intake-drift")
    doctor = service.operator_doctor_v70()
    assert doctor["status"] == "FAIL"
    assert doctor["authority"]["errors"]["intake-drift"] == (
        "StudioConflictError"
    )


def test_reconcile_projects_only_an_exact_submitted_authority_state(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path / "tasks")
    service.create_task(
        {
            "objective": "Estimate a trustworthy model from public observations.",
            "workspace_id": "submitted-reconcile",
        }
    )
    workspace = service._workspace("submitted-reconcile")
    packet = service._operator_packet_v70(workspace, "run_s0")
    work = service.operator_store.ensure_work(packet)
    lease = service.operator_store.claim(
        work["work_id"],
        worker_id="crashed-after-submit",
    )
    output_binding = service._operator_authority_binding_v70(workspace)
    submission = OperatorSubmissionV70.seal(
        work_id=lease.work_id,
        packet_hash=packet.packet_hash,
        input_binding_hash=packet.authority_binding.binding_hash,
        output_binding=output_binding,
        before_manifest_hash=sha256_value({}),
        after_manifest_hash=sha256_value(
            capture_file_manifest(workspace.root)
        ),
        changed_paths=(),
        result_summary={"status": "submitted-before-projection"},
        submitted_at="2026-07-30T00:00:00+00:00",
    )
    service.operator_store.submit(lease, submission)

    report = service.reconcile_operator_v70()
    recovered = service.operator_store.get_work(work["work_id"])
    assert report["authority_reconciled_work_ids"] == [work["work_id"]]
    assert report["authority_ambiguous_work_ids"] == []
    assert recovered["status"] == "REJECTED"
    assert recovered["authority_projection"]["worker_submission_recovered"] is True
    assert recovered["authority_projection"]["reason_codes"] == [
        "s0_gate_not_open"
    ]
    assert recovered["authority_projection"]["scientific_qualification_granted"] is False


def test_reconcile_refuses_a_submission_not_matching_current_graph(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path / "tasks")
    service.create_task(
        {
            "objective": "Estimate a trustworthy model from public observations.",
            "workspace_id": "stale-reconcile",
        }
    )
    workspace = service._workspace("stale-reconcile")
    packet = service._operator_packet_v70(workspace, "run_s0")
    work = service.operator_store.ensure_work(packet)
    lease = service.operator_store.claim(
        work["work_id"],
        worker_id="stale-submitter",
    )
    current = service._operator_authority_binding_v70(workspace)
    stale = OperatorAuthorityBindingV70.seal(
        workspace_id=current.workspace_id,
        graph_id=current.graph_id,
        workspace_spec_hash=current.workspace_spec_hash,
        graph_snapshot_hash=sha256_value({"stale": True}),
        frontier_node_hashes=current.frontier_node_hashes,
        stage_statuses=current.stage_statuses,
        current_gate_hashes=current.current_gate_hashes,
        frontier_stages=current.frontier_stages,
        operator_policy_hash=current.operator_policy_hash,
    )
    submission = OperatorSubmissionV70.seal(
        work_id=lease.work_id,
        packet_hash=packet.packet_hash,
        input_binding_hash=packet.authority_binding.binding_hash,
        output_binding=stale,
        before_manifest_hash=sha256_value({}),
        after_manifest_hash=sha256_value(
            capture_file_manifest(workspace.root)
        ),
        changed_paths=(),
        result_summary={"status": "stale-submitted-state"},
        submitted_at="2026-07-30T00:00:00+00:00",
    )
    service.operator_store.submit(lease, submission)

    report = service.reconcile_operator_v70()
    unresolved = service.operator_store.get_work(work["work_id"])
    assert report["authority_reconciled_work_ids"] == []
    assert report["authority_ambiguous_work_ids"] == [work["work_id"]]
    assert unresolved["status"] == "SUBMITTED"


def test_reconcile_refuses_a_changed_filesystem_postcondition(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path / "tasks")
    service.create_task(
        {
            "objective": "Estimate a trustworthy model from public observations.",
            "workspace_id": "manifest-reconcile",
        }
    )
    workspace = service._workspace("manifest-reconcile")
    packet = service._operator_packet_v70(workspace, "run_s0")
    work = service.operator_store.ensure_work(packet)
    lease = service.operator_store.claim(
        work["work_id"],
        worker_id="manifest-submitter",
    )
    output_binding = service._operator_authority_binding_v70(workspace)
    submission = OperatorSubmissionV70.seal(
        work_id=lease.work_id,
        packet_hash=packet.packet_hash,
        input_binding_hash=packet.authority_binding.binding_hash,
        output_binding=output_binding,
        before_manifest_hash=sha256_value({}),
        after_manifest_hash=sha256_value(
            capture_file_manifest(workspace.root)
        ),
        changed_paths=(),
        result_summary={"status": "submitted-before-file-change"},
        submitted_at="2026-07-30T00:00:00+00:00",
    )
    service.operator_store.submit(lease, submission)
    (workspace.root / "docs" / "late-change.txt").write_text(
        "changed after submission",
        encoding="utf-8",
    )

    report = service.reconcile_operator_v70()
    unresolved = service.operator_store.get_work(work["work_id"])
    assert report["authority_reconciled_work_ids"] == []
    assert report["authority_ambiguous_work_ids"] == [work["work_id"]]
    assert unresolved["status"] == "SUBMITTED"
