from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from fma.hashing import sha256_value
from fma.operator_v70 import (
    OperatorAuthorityBindingV70,
    OperatorConflictError,
    OperatorLeaseError,
    OperatorPacketV70,
    OperatorStoreV70,
    OperatorSubmissionV70,
    assert_changed_paths_owned,
    owned_paths_overlap,
)


def _binding(
    workspace_id: str = "task-one",
    *,
    graph_snapshot_hash: str | None = None,
) -> OperatorAuthorityBindingV70:
    return OperatorAuthorityBindingV70.seal(
        workspace_id=workspace_id,
        graph_id=f"graph-{workspace_id}",
        workspace_spec_hash=sha256_value({"workspace": workspace_id}),
        graph_snapshot_hash=graph_snapshot_hash
        or sha256_value({"graph": workspace_id}),
        frontier_node_hashes=(sha256_value({"frontier": workspace_id}),),
        stage_statuses={"S0": "frontier"},
        current_gate_hashes={},
        frontier_stages=("S0",),
        operator_policy_hash=sha256_value({"policy": "v70"}),
    )


def _packet(
    *,
    workspace_id: str = "task-one",
    action: str = "run_s0",
    idempotency_key: str = "semantic-key-for-task-one",
    write_paths: tuple[str, ...] = (".",),
    graph_snapshot_hash: str | None = None,
) -> OperatorPacketV70:
    return OperatorPacketV70.seal(
        workspace_id=workspace_id,
        action=action,
        purpose="Run one bounded operator test action.",
        authority_binding=_binding(
            workspace_id,
            graph_snapshot_hash=graph_snapshot_hash,
        ),
        read_paths=(".",),
        write_paths=write_paths,
        allowed_tool_profile="test_profile",
        expected_outputs=("typed receipt",),
        max_attempts=3,
        lease_seconds=30,
        max_wall_seconds=60,
        idempotency_key=idempotency_key,
    )


def _submission(
    lease,
    packet: OperatorPacketV70,
    *,
    changed_paths: tuple[str, ...] = (),
) -> OperatorSubmissionV70:
    return OperatorSubmissionV70.seal(
        work_id=lease.work_id,
        packet_hash=packet.packet_hash,
        input_binding_hash=packet.authority_binding.binding_hash,
        output_binding=packet.authority_binding,
        before_manifest_hash=sha256_value({}),
        after_manifest_hash=sha256_value({}),
        changed_paths=changed_paths,
        result_summary={"status": "test-only"},
        submitted_at="2026-07-30T00:00:00+00:00",
    )


def test_operator_semantic_idempotency_binds_packet_hash(tmp_path: Path) -> None:
    store = OperatorStoreV70(tmp_path)
    packet = _packet()
    first = store.ensure_work(packet)
    replay = store.ensure_work(packet)
    assert replay["work_id"] == first["work_id"]
    assert len(store.list_work()) == 1

    changed = _packet(
        action="run_s1",
        idempotency_key=packet.idempotency_key,
    )
    with pytest.raises(OperatorConflictError, match="different operator packet"):
        store.ensure_work(changed)


def test_expired_lease_is_fenced_and_quarantined_before_retry(
    tmp_path: Path,
) -> None:
    store = OperatorStoreV70(tmp_path)
    packet = _packet()
    work = store.ensure_work(packet)
    first = store.claim(work["work_id"], worker_id="worker-a", lease_seconds=30)

    assert store.reconcile_expired(
        now_epoch=first.lease_until_epoch + 1
    ) == [work["work_id"]]
    assert store.get_work(work["work_id"])["status"] == "RECOVERY_PENDING"
    with pytest.raises(OperatorConflictError, match="RECOVERY_PENDING"):
        store.claim(
            work["work_id"],
            worker_id="worker-b",
            lease_seconds=30,
        )

    with pytest.raises(OperatorLeaseError, match="fenced|active"):
        store.heartbeat(first)
    with pytest.raises(OperatorLeaseError, match="fenced|active"):
        store.submit(first, _submission(first, packet))
    assert store.doctor()["status"] == "RECOVERY_PENDING"


def test_controlled_failure_can_retry_with_a_new_fence(tmp_path: Path) -> None:
    store = OperatorStoreV70(tmp_path)
    packet = _packet()
    work = store.ensure_work(packet)
    first = store.claim(work["work_id"], worker_id="worker-a")
    store.fail(first, error_type="ControlledFailure", message="safe retry")
    second = store.claim(work["work_id"], worker_id="worker-b")
    assert second.attempt_epoch == first.attempt_epoch + 1
    assert second.fencing_token == first.fencing_token + 1
    with pytest.raises(OperatorLeaseError, match="fenced|active"):
        store.heartbeat(first)
    submitted = store.submit(second, _submission(second, packet))
    assert submitted["status"] == "SUBMITTED"
    accepted = store.project_authority_decision(
        work["work_id"],
        accepted=True,
        authority_receipt_hash=sha256_value({"gate": "S0"}),
    )
    assert accepted["status"] == "ACCEPTED"
    assert accepted["authority_projection"]["claim_scope"] == (
        "workflow_control_only"
    )
    assert accepted["authority_projection"]["scientific_qualification_granted"] is False


def test_component_ownership_and_write_scope_are_fail_closed() -> None:
    assert owned_paths_overlap("docs", "docs/model.json")
    assert owned_paths_overlap(r"DOCS\Model.json", "docs")
    assert not owned_paths_overlap("docs", "results")
    with pytest.raises(ValueError, match="workspace-relative|escapes"):
        owned_paths_overlap("../private", "docs")
    with pytest.raises(ValueError, match="unsafe component"):
        owned_paths_overlap("docs/result.json:secret", "docs")
    with pytest.raises(ValueError, match="unsafe component"):
        owned_paths_overlap("docs/trailing. ", "docs")
    with pytest.raises(OperatorConflictError, match="outside declared ownership"):
        assert_changed_paths_owned(("results/output.json",), ("docs",))
    assert_changed_paths_owned(("docs/output.json",), ("docs",))


def test_active_parent_child_ownership_conflict_is_cross_process_durable(
    tmp_path: Path,
) -> None:
    first_store = OperatorStoreV70(tmp_path)
    second_store = OperatorStoreV70(tmp_path)
    first_packet = _packet(
        workspace_id="same-task",
        idempotency_key="same-task-docs-parent",
        write_paths=("docs",),
    )
    second_packet = _packet(
        workspace_id="same-task",
        action="run_s1",
        idempotency_key="same-task-docs-child",
        write_paths=("docs/model.json",),
    )
    first = first_store.ensure_work(first_packet)
    second = second_store.ensure_work(second_packet)
    first_store.claim(first["work_id"], worker_id="worker-a")
    with pytest.raises(OperatorConflictError, match="ownership conflicts"):
        second_store.claim(second["work_id"], worker_id="worker-b")


def test_operator_doctor_verifies_event_chain_and_detects_tamper(
    tmp_path: Path,
) -> None:
    store = OperatorStoreV70(tmp_path)
    store.ensure_work(_packet())
    assert store.doctor()["status"] == "PASS"

    with closing(sqlite3.connect(store.database_path)) as connection:
        connection.execute(
            "UPDATE operator_events SET payload_json='{}' WHERE sequence=1"
        )
        connection.commit()
    report = store.doctor()
    assert report["status"] == "FAIL"
    assert any("event chain differs" in error for error in report["errors"])


def test_operator_doctor_replays_state_instead_of_trusting_work_row(
    tmp_path: Path,
) -> None:
    store = OperatorStoreV70(tmp_path)
    work = store.ensure_work(_packet())
    with closing(sqlite3.connect(store.database_path)) as connection:
        connection.execute(
            "UPDATE work_items SET status='FAILED' WHERE work_id=?",
            (work["work_id"],),
        )
        connection.commit()

    report = store.doctor()
    assert report["status"] == "FAIL"
    assert any(
        "row state differs from event-derived state" in error
        for error in report["errors"]
    )


def test_operator_doctor_binds_routing_columns_to_the_sealed_packet(
    tmp_path: Path,
) -> None:
    store = OperatorStoreV70(tmp_path)
    work = store.ensure_work(_packet())
    with closing(sqlite3.connect(store.database_path)) as connection:
        connection.execute(
            "UPDATE work_items SET workspace_id='wrong-route' WHERE work_id=?",
            (work["work_id"],),
        )
        connection.commit()

    report = store.doctor()
    assert report["status"] == "FAIL"
    assert any(
        "routing fields differ from packet" in error
        for error in report["errors"]
    )


def test_operator_doctor_reports_malformed_event_payload_without_throwing(
    tmp_path: Path,
) -> None:
    store = OperatorStoreV70(tmp_path)
    store.ensure_work(_packet())
    with closing(sqlite3.connect(store.database_path)) as connection:
        connection.execute(
            "UPDATE operator_events SET payload_json='not-json' WHERE sequence=1"
        )
        connection.commit()

    report = store.doctor()
    assert report["status"] == "FAIL"
    assert any(
        "event payload is unreadable" in error for error in report["errors"]
    )


def test_operator_doctor_binds_terminal_projection_to_row_and_event(
    tmp_path: Path,
) -> None:
    store = OperatorStoreV70(tmp_path)
    packet = _packet()
    work = store.ensure_work(packet)
    lease = store.claim(work["work_id"], worker_id="projection-worker")
    store.submit(lease, _submission(lease, packet))
    store.project_authority_decision(
        work["work_id"],
        accepted=True,
        authority_receipt_hash=sha256_value({"gate": "S0"}),
    )
    with closing(sqlite3.connect(store.database_path)) as connection:
        projection = connection.execute(
            "SELECT authority_projection_json FROM work_items WHERE work_id=?",
            (work["work_id"],),
        ).fetchone()[0]
        payload = json.loads(projection)
        payload["status"] = "REJECTED"
        connection.execute(
            "UPDATE work_items SET authority_projection_json=? WHERE work_id=?",
            (json.dumps(payload), work["work_id"]),
        )
        connection.commit()

    report = store.doctor()
    assert report["status"] == "FAIL"
    assert any(
        "authority projection status differs" in error
        for error in report["errors"]
    )
    assert any(
        "authority projection differs from event" in error
        for error in report["errors"]
    )
