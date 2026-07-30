from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from fma.operator_v70 import (
    IntakeAttachmentV70,
    OperatorConflictError,
    OperatorIntegrityError,
    OperatorStoreV70,
)


def test_transactional_intake_hashes_copies_and_replays_exactly(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    brief = source / "brief.md"
    data = source / "series.csv"
    brief.write_text("# Problem\nEstimate a robust dynamic model.", encoding="utf-8")
    data.write_text("t,y\n0,1\n1,2\n", encoding="utf-8")
    task_root = tmp_path / "tasks"
    store = OperatorStoreV70(task_root)

    manifest = store.publish_intake(
        idempotency_key="intake-case-001",
        objective="Estimate and validate the attached dynamic system.",
        attachment_paths=(brief, data),
        requested_workspace_id="case-001",
        evidence_scope="public_data",
        workflow_mode="v67",
    )
    replay = store.publish_intake(
        idempotency_key="intake-case-001",
        objective="Estimate and validate the attached dynamic system.",
        attachment_paths=(brief, data),
        requested_workspace_id="case-001",
        evidence_scope="public_data",
        workflow_mode="v67",
    )
    assert replay.manifest_hash == manifest.manifest_hash
    assert [item.logical_name for item in manifest.attachments] == [
        "brief.md",
        "series.csv",
    ]
    assert store.verify_intake(manifest.intake_id)
    current = store.root / "current_intake.json"
    assert current.is_file()
    current.unlink()
    repaired = store.publish_intake(
        idempotency_key="intake-case-001",
        objective="Estimate and validate the attached dynamic system.",
        attachment_paths=(brief, data),
        requested_workspace_id="case-001",
        evidence_scope="public_data",
        workflow_mode="v67",
    )
    assert repaired.manifest_hash == manifest.manifest_hash
    assert json.loads(current.read_text(encoding="utf-8"))["manifest_hash"] == (
        manifest.manifest_hash
    )

    installed = store.materialize_intake(
        manifest.intake_id,
        tmp_path / "workspace",
    )
    assert (installed / "attachments" / "brief.md").read_bytes() == (
        brief.read_bytes()
    )


def test_intake_same_key_different_content_is_a_conflict(tmp_path: Path) -> None:
    attachment = tmp_path / "brief.txt"
    attachment.write_text("first version", encoding="utf-8")
    store = OperatorStoreV70(tmp_path / "tasks")
    store.publish_intake(
        idempotency_key="same-network-request",
        objective="Model the attached system with explicit uncertainty.",
        attachment_paths=(attachment,),
    )
    attachment.write_text("different version", encoding="utf-8")
    with pytest.raises(OperatorConflictError, match="different content"):
        store.publish_intake(
            idempotency_key="same-network-request",
            objective="Model the attached system with explicit uncertainty.",
            attachment_paths=(attachment,),
        )


def test_publication_failure_never_moves_current_pointer_or_creates_formal_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attachment = tmp_path / "brief.txt"
    attachment.write_text("bounded failure injection", encoding="utf-8")
    task_root = tmp_path / "tasks"
    store = OperatorStoreV70(task_root)
    original_replace = os.replace

    def fail_intake_publish(source: object, target: object) -> None:
        target_path = Path(target)
        if target_path.parent == store.intakes_root:
            raise OSError("injected publication crash")
        original_replace(source, target)

    monkeypatch.setattr("fma.operator_v70.os.replace", fail_intake_publish)
    with pytest.raises(OSError, match="injected publication crash"):
        store.publish_intake(
            idempotency_key="crash-case",
            objective="Model the attached system after a transactional intake.",
            attachment_paths=(attachment,),
            requested_workspace_id="formal-task",
        )
    assert not (store.root / "current_intake.json").exists()
    assert not (task_root / "formal-task").exists()
    report = store.doctor()
    assert report["status"] == "RECOVERY_PENDING"
    assert any("publication incomplete" in warning for warning in report["warnings"])


def test_doctor_detects_mutated_committed_attachment(tmp_path: Path) -> None:
    attachment = tmp_path / "brief.txt"
    attachment.write_text("immutable content", encoding="utf-8")
    store = OperatorStoreV70(tmp_path / "tasks")
    manifest = store.publish_intake(
        idempotency_key="tamper-case",
        objective="Model the attached system and preserve exact provenance.",
        attachment_paths=(attachment,),
    )
    committed = (
        store.intakes_root
        / manifest.intake_id
        / "attachments"
        / "brief.txt"
    )
    committed.write_text("mutated", encoding="utf-8")
    with pytest.raises(OperatorIntegrityError):
        store.verify_intake(manifest.intake_id)
    report = store.doctor()
    assert report["status"] == "FAIL"
    assert any(manifest.intake_id in error for error in report["errors"])


def test_intake_schema_rejects_path_names_and_symlink_sources(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        IntakeAttachmentV70(
            logical_name="../escape.txt",
            sha256="0" * 64,
            size_bytes=0,
            media_type="text/plain",
            blob_ref=f"sha256/{'0' * 64}",
        )

    source = tmp_path / "source.txt"
    source.write_text("source body", encoding="utf-8")
    link = tmp_path / "source-link.txt"
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip("symbolic links are unavailable on this Windows host")
    store = OperatorStoreV70(tmp_path / "tasks")
    with pytest.raises(ValueError, match="symbolic link"):
        store.publish_intake(
            idempotency_key="symlink-source-case",
            objective="Model the supplied source without following symbolic links.",
            attachment_paths=(link,),
        )


def test_database_and_disk_intake_manifests_must_match(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first body", encoding="utf-8")
    second.write_text("second body", encoding="utf-8")
    store = OperatorStoreV70(tmp_path / "tasks")
    first_manifest = store.publish_intake(
        idempotency_key="manifest-db-first",
        objective="Model the first committed intake under an exact manifest.",
        attachment_paths=(first,),
    )
    second_manifest = store.publish_intake(
        idempotency_key="manifest-db-second",
        objective="Model the second committed intake under an exact manifest.",
        attachment_paths=(second,),
    )
    with closing(sqlite3.connect(store.database_path)) as connection:
        connection.execute(
            """
            UPDATE intakes SET manifest_hash=?,manifest_json=?
            WHERE intake_id=?
            """,
            (
                second_manifest.manifest_hash,
                second_manifest.model_dump_json(),
                first_manifest.intake_id,
            ),
        )
        connection.commit()
    with pytest.raises(OperatorIntegrityError, match="manifests differ"):
        store.get_intake(first_manifest.intake_id)
    assert store.doctor()["status"] == "FAIL"


def test_materialized_intake_detects_mutation_and_extra_files(
    tmp_path: Path,
) -> None:
    attachment = tmp_path / "brief.txt"
    attachment.write_text("immutable workspace copy", encoding="utf-8")
    store = OperatorStoreV70(tmp_path / "tasks")
    manifest = store.publish_intake(
        idempotency_key="workspace-copy-case",
        objective="Model the attached brief from an exact workspace copy.",
        attachment_paths=(attachment,),
    )
    workspace = tmp_path / "workspace"
    installed = store.materialize_intake(manifest.intake_id, workspace)
    (installed / "attachments" / "brief.txt").write_text(
        "changed",
        encoding="utf-8",
    )
    with pytest.raises(OperatorIntegrityError, match="differs"):
        store.verify_materialized_intake(manifest.intake_id, workspace)
    (installed / "attachments" / "brief.txt").write_bytes(
        attachment.read_bytes()
    )
    (installed / "attachments" / "unexpected.txt").write_text(
        "not in manifest",
        encoding="utf-8",
    )
    with pytest.raises(OperatorIntegrityError, match="unexpected"):
        store.verify_materialized_intake(manifest.intake_id, workspace)


def test_materialized_intake_rejects_a_linked_intake_directory(
    tmp_path: Path,
) -> None:
    attachment = tmp_path / "brief.txt"
    attachment.write_text("immutable workspace copy", encoding="utf-8")
    store = OperatorStoreV70(tmp_path / "tasks")
    manifest = store.publish_intake(
        idempotency_key="workspace-link-case",
        objective="Model the attached brief from an exact workspace copy.",
        attachment_paths=(attachment,),
    )
    workspace = tmp_path / "workspace"
    installed = store.materialize_intake(manifest.intake_id, workspace)
    external = tmp_path / "external-intake"
    installed.rename(external)
    try:
        installed.symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("directory symbolic links are unavailable on this host")

    with pytest.raises(
        OperatorIntegrityError,
        match="symbolic link|junction|outside",
    ):
        store.verify_materialized_intake(manifest.intake_id, workspace)
