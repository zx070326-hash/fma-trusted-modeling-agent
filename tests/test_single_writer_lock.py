from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from multiprocessing.synchronize import Barrier
from pathlib import Path

import pytest

from fma.hashing import sha256_value
from fma.storage import RunStore
from fma.v4 import GraphLoopContractV40, GraphLoopStoreV40, GraphNodeV40
from fma.v5.external_harness import ExternalHarnessV50


GRAPH_NOW = datetime(2026, 7, 24, tzinfo=timezone.utc)


def _emit_from_stale_run_handle(
    run_directory: str, rendezvous: Barrier, writer: str
) -> None:
    store = RunStore.open_existing(run_directory)
    rendezvous.wait(timeout=10)
    store.emit("concurrent_test_event", {"writer": writer})


def _append_external_event(
    harness_root: str, rendezvous: Barrier, writer: str
) -> None:
    # Delay the first event hash until both processes have entered the append
    # path.  An implementation without a writer lock deterministically reads
    # the same tip twice; the locked implementation serializes at that point.
    import fma.v5.external_harness as harness_module

    original_sha256_value = harness_module.sha256_value
    waited = False

    def synchronized_sha256(value: object) -> str:
        nonlocal waited
        if (
            not waited
            and isinstance(value, dict)
            and value.get("event_type") == "ablation_assessed"
        ):
            waited = True
            try:
                rendezvous.wait(timeout=1)
            except threading.BrokenBarrierError:
                pass
        return original_sha256_value(value)

    harness_module.sha256_value = synchronized_sha256
    try:
        ExternalHarnessV50(harness_root)._append_event(
            "ablation_assessed",
            "case.concurrent",
            {"writer": writer},
        )
    finally:
        harness_module.sha256_value = original_sha256_value


def _record_frontier_outcome(
    graph_directory: str,
    node_hash: str,
    rendezvous: Barrier,
    results: object,
    writer: str,
) -> None:
    graph = GraphLoopStoreV40.open_existing(graph_directory)
    output = graph.put_output(
        f"concurrent_output_{writer}",
        {"writer": writer, "node_hash": node_hash},
    )
    rendezvous.wait(timeout=10)
    try:
        graph.record_outcome(
            node_hash,
            actor="harness",
            status="succeeded",
            output_artifacts=[output],
            summary=f"writer {writer} completed the frontier node",
            outcome_id=f"concurrent_outcome_{writer}",
            started_at=GRAPH_NOW,
            finished_at=GRAPH_NOW,
        )
    except Exception as exc:
        results.put(("failed", type(exc).__name__, str(exc)))
    else:
        results.put(("succeeded", "", ""))


def _assert_processes_succeeded(processes: list[object]) -> None:
    for process in processes:
        process.join(timeout=20)
        assert not process.is_alive()
        assert process.exitcode == 0


def _frontier_graph(tmp_path: Path, graph_id: str) -> tuple[GraphLoopStoreV40, str]:
    contract = GraphLoopContractV40.seal(
        graph_id=graph_id,
        layer="modeling",
        evaluator_epoch="single-writer-v1",
        objective="exercise one serialized graph frontier outcome",
        created_at=GRAPH_NOW,
    )
    graph = GraphLoopStoreV40(tmp_path, contract)
    node = GraphNodeV40.seal(
        node_id="frontier_execution",
        layer="modeling",
        node_kind="execution",
        executor="harness",
        created_by="harness",
        artifact_hash=sha256_value({"node": "frontier_execution"}),
        purpose="exercise a single-writer frontier transition",
        created_at=GRAPH_NOW,
    )
    graph.add_node(node)
    assert node.node_hash is not None
    return graph, node.node_hash


def test_two_stale_run_handles_refresh_tip_for_emit_and_put_artifact(
    tmp_path: Path,
) -> None:
    created = RunStore(tmp_path, run_id="stale-handles")
    first = RunStore.open_existing(created.run_directory)
    second = RunStore.open_existing(created.run_directory)

    first.emit("first_writer", {"value": 1})
    reference = second.put_artifact("second_writer", {"value": 2})
    first.put_artifact("first_writer_artifact", {"value": 3})

    assert RunStore.open_existing(created.run_directory).verify_event_chain()
    records = [
        json.loads(line)
        for line in created.event_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["sequence"] for record in records] == [1, 2, 3, 4]
    assert records[2]["payload"]["sha256"] == reference.sha256


def test_cross_process_stale_run_handles_form_one_chain(tmp_path: Path) -> None:
    import multiprocessing

    created = RunStore(tmp_path, run_id="cross-process")
    context = multiprocessing.get_context("spawn")
    rendezvous = context.Barrier(2)
    processes = [
        context.Process(
            target=_emit_from_stale_run_handle,
            args=(str(created.run_directory), rendezvous, writer),
        )
        for writer in ("a", "b")
    ]
    for process in processes:
        process.start()
    _assert_processes_succeeded(processes)

    reopened = RunStore.open_existing(created.run_directory)
    assert reopened.verify_event_chain()
    records = [
        json.loads(line)
        for line in created.event_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["sequence"] for record in records] == [1, 2, 3]
    assert {record["payload"].get("writer") for record in records[1:]} == {"a", "b"}


def test_cross_process_external_appends_form_one_chain(tmp_path: Path) -> None:
    import multiprocessing

    harness = ExternalHarnessV50(tmp_path / "external")
    context = multiprocessing.get_context("spawn")
    rendezvous = context.Barrier(2)
    processes = [
        context.Process(
            target=_append_external_event,
            args=(str(harness.root), rendezvous, writer),
        )
        for writer in ("a", "b")
    ]
    for process in processes:
        process.start()
    _assert_processes_succeeded(processes)

    assert harness.verify_event_chain()
    events = harness._read_events()
    assert [event.sequence for event in events] == [1, 2]
    assert {event.payload["writer"] for event in events} == {"a", "b"}


def test_two_stale_graph_handles_cannot_both_finish_one_frontier(
    tmp_path: Path,
) -> None:
    graph, node_hash = _frontier_graph(tmp_path, "stale_graph_handles")
    first = GraphLoopStoreV40.open_existing(graph.run_directory)
    second = GraphLoopStoreV40.open_existing(graph.run_directory)
    first_output = first.put_output("first_output", {"writer": "first"})
    second_output = second.put_output("second_output", {"writer": "second"})

    first.record_outcome(
        node_hash,
        actor="harness",
        status="succeeded",
        output_artifacts=[first_output],
        summary="first stale handle won the frontier",
        outcome_id="first_outcome",
        started_at=GRAPH_NOW,
        finished_at=GRAPH_NOW,
    )
    with pytest.raises(RuntimeError, match="executable frontier"):
        second.record_outcome(
            node_hash,
            actor="harness",
            status="succeeded",
            output_artifacts=[second_output],
            summary="second stale handle must not duplicate the outcome",
            outcome_id="second_outcome",
            started_at=GRAPH_NOW,
            finished_at=GRAPH_NOW,
        )

    assert graph.verify()
    reopened = GraphLoopStoreV40.open_existing(graph.run_directory)
    assert reopened.verify()
    assert reopened.project_state().snapshot.outcome_count == 1


def test_cross_process_graph_frontier_has_exactly_one_winner(
    tmp_path: Path,
) -> None:
    import multiprocessing

    graph, node_hash = _frontier_graph(tmp_path, "process_graph_race")
    context = multiprocessing.get_context("spawn")
    rendezvous = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(
            target=_record_frontier_outcome,
            args=(
                str(graph.run_directory),
                node_hash,
                rendezvous,
                results,
                writer,
            ),
        )
        for writer in ("a", "b")
    ]
    for process in processes:
        process.start()
    _assert_processes_succeeded(processes)

    outcomes = [results.get(timeout=5) for _ in processes]
    assert [item[0] for item in outcomes].count("succeeded") == 1
    failures = [item for item in outcomes if item[0] == "failed"]
    assert failures == [
        ("failed", "RuntimeError", "node is not on the executable frontier")
    ]
    assert graph.verify()
    reopened = GraphLoopStoreV40.open_existing(graph.run_directory)
    assert reopened.verify()
    assert reopened.store.verify_event_chain()
    assert reopened.project_state().snapshot.outcome_count == 1
