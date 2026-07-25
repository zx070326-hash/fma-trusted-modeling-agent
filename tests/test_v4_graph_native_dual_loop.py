from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from fma.hashing import sha256_value
from fma.v4 import (
    GraphEdgeV40,
    GraphLoopContractV40,
    GraphLoopStoreV40,
    GraphNodeV40,
    import_cross_layer_bridge_v40,
    reconcile_cross_layer_bridge_v40,
)


NOW = datetime(2026, 7, 23, tzinfo=timezone.utc)


def _contract(graph_id: str, layer: str, **overrides: object) -> GraphLoopContractV40:
    data: dict[str, object] = {
        "graph_id": graph_id,
        "layer": layer,
        "evaluator_epoch": "anchor-v1",
        "objective": f"exercise the {layer} graph-native loop",
        "created_at": NOW,
    }
    data.update(overrides)
    return GraphLoopContractV40.seal(**data)


def _node(
    node_id: str,
    layer: str,
    kind: str,
    executor: str,
    *,
    created_by: str = "harness",
) -> GraphNodeV40:
    return GraphNodeV40.seal(
        node_id=node_id,
        layer=layer,
        node_kind=kind,
        executor=executor,
        created_by=created_by,
        artifact_hash=sha256_value({"node": node_id}),
        purpose=f"execute frozen {node_id} responsibility",
        created_at=NOW,
    )


def _edge(
    edge_id: str,
    layer: str,
    source: GraphNodeV40,
    target: GraphNodeV40,
    relation: str,
) -> GraphEdgeV40:
    return GraphEdgeV40.seal(
        edge_id=edge_id,
        layer=layer,
        source_node_hash=source.node_hash,
        target_node_hash=target.node_hash,
        relation=relation,
        rationale=f"bind {source.node_id} to {target.node_id}",
        created_at=NOW,
    )


def _finish(
    store: GraphLoopStoreV40,
    node: GraphNodeV40,
    actor: str,
    *,
    status: str = "succeeded",
) -> None:
    output = store.put_output(
        f"{node.node_id}_output",
        {"node_hash": node.node_hash, "status": status},
    )
    store.record_outcome(
        node.node_hash,
        actor=actor,
        status=status,
        output_artifacts=[output],
        summary=f"{node.node_id} reached {status}",
        outcome_id=f"{node.node_id}_outcome",
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=1),
    )


def test_modeling_graph_frontier_private_promotion_and_recovery(tmp_path) -> None:
    store = GraphLoopStoreV40(
        tmp_path,
        _contract("modeling_graph", "modeling"),
    )
    candidate = _node(
        "candidate", "modeling", "model_candidate", "model", created_by="model"
    )
    evaluator = _node("private_eval", "modeling", "evaluation", "verifier")
    store.add_node(candidate)
    store.add_node(evaluator)
    store.add_edge(
        _edge(
            "candidate_evaluated_by_private",
            "modeling",
            candidate,
            evaluator,
            "evaluated_by",
        )
    )

    assert store.project_state().snapshot.frontier_node_hashes == [candidate.node_hash]
    _finish(store, candidate, "model")
    assert store.project_state().snapshot.frontier_node_hashes == [evaluator.node_hash]
    _finish(store, evaluator, "verifier")

    with pytest.raises(ValidationError, match="private scientific gate"):
        store.decide_promotion(
            candidate.node_hash,
            evaluator.node_hash,
            evidence_node_hashes=[evaluator.node_hash],
            decision="qualified",
            authority="verifier",
            independent_gate_passed=True,
            private_scientific_gate_passed=False,
            scope="synthetic graph-kernel test only",
            promotion_id="unsafe_qualification",
            decided_at=NOW,
        )

    store.decide_promotion(
        candidate.node_hash,
        evaluator.node_hash,
        evidence_node_hashes=[candidate.node_hash, evaluator.node_hash],
        decision="qualified",
        authority="verifier",
        independent_gate_passed=True,
        private_scientific_gate_passed=True,
        scope="synthetic graph-kernel test only",
        promotion_id="candidate_qualified",
        decided_at=NOW,
    )
    assert store.project_state().snapshot.node_statuses[candidate.node_hash] == "qualified"
    store.decide_promotion(
        candidate.node_hash,
        evaluator.node_hash,
        evidence_node_hashes=[candidate.node_hash, evaluator.node_hash],
        decision="active",
        authority="human",
        independent_gate_passed=True,
        private_scientific_gate_passed=True,
        scope="synthetic graph-kernel test only",
        promotion_id="candidate_activated",
        decided_at=NOW,
    )
    assert store.project_state().snapshot.node_statuses[candidate.node_hash] == "active"
    assert store.verify()
    reopened = GraphLoopStoreV40.open_existing(store.run_directory)
    assert reopened.project_state().snapshot == store.project_state().snapshot


def test_model_cannot_take_verifier_or_harness_authority(tmp_path) -> None:
    with pytest.raises(ValidationError, match="cannot execute node kind evaluation"):
        _node("bad_eval", "modeling", "evaluation", "model", created_by="model")

    store = GraphLoopStoreV40(tmp_path, _contract("authority_graph", "modeling"))
    execution = _node("solver_run", "modeling", "execution", "harness")
    store.add_node(execution)
    output = store.put_output("solver_output", {"status": "ok"})
    with pytest.raises(PermissionError, match="does not own"):
        store.record_outcome(
            execution.node_hash,
            actor="model",
            status="succeeded",
            output_artifacts=[output],
            summary="model tried to self-certify execution",
            outcome_id="unauthorized_outcome",
            started_at=NOW,
            finished_at=NOW,
        )


def test_development_loop_promotes_patch_then_imports_only_pending_runtime(tmp_path) -> None:
    development = GraphLoopStoreV40(
        tmp_path / "development",
        _contract("development_graph", "development"),
    )
    patch = _node("patch", "development", "patch", "model", created_by="model")
    review = _node("review", "development", "review", "verifier")
    release = _node("release", "development", "release", "human")
    for node in [patch, review, release]:
        development.add_node(node)
    development.add_edge(
        _edge("patch_review", "development", patch, review, "evaluated_by")
    )
    development.add_edge(
        _edge("patch_release", "development", patch, release, "requires_active")
    )

    _finish(development, patch, "model")
    assert review.node_hash in development.project_state().snapshot.frontier_node_hashes
    assert release.node_hash not in development.project_state().snapshot.frontier_node_hashes
    _finish(development, review, "verifier")
    development.decide_promotion(
        patch.node_hash,
        review.node_hash,
        evidence_node_hashes=[patch.node_hash, review.node_hash],
        decision="qualified",
        authority="verifier",
        independent_gate_passed=True,
        private_scientific_gate_passed=False,
        scope="local V4 kernel regression suite",
        promotion_id="patch_qualified",
        decided_at=NOW,
    )
    development.decide_promotion(
        patch.node_hash,
        review.node_hash,
        evidence_node_hashes=[patch.node_hash, review.node_hash],
        decision="active",
        authority="human",
        independent_gate_passed=True,
        private_scientific_gate_passed=False,
        scope="local V4 kernel regression suite",
        promotion_id="patch_activated",
        decided_at=NOW,
    )
    assert release.node_hash in development.project_state().snapshot.frontier_node_hashes
    _finish(development, release, "human")

    modeling = GraphLoopStoreV40(
        tmp_path / "modeling",
        _contract("target_modeling_graph", "modeling"),
    )
    bridge, runtime = import_cross_layer_bridge_v40(
        development,
        modeling,
        release.node_hash,
        target_node_id="runtime_candidate",
        sanitized_payload={"package_hash": sha256_value({"release": "v4"})},
        bridge_id="release_bridge",
        approved_by="human",
        created_at=NOW,
    )
    target_state = modeling.project_state()
    assert bridge.scientific_validity_granted is False
    assert bridge.private_acceptance_data_exposed is False
    assert target_state.snapshot.node_statuses[runtime.node_hash] == "pending"
    assert runtime.node_hash in target_state.snapshot.frontier_node_hashes
    runtime_output = modeling.put_output("runtime_probe", {"status": "not_executed"})
    with pytest.raises(RuntimeError, match="bridge reconciliation"):
        modeling.record_outcome(
            runtime.node_hash,
            actor="harness",
            status="succeeded",
            output_artifacts=[runtime_output],
            summary="must not execute before source-snapshot reconciliation",
            outcome_id="premature_runtime_execution",
            started_at=NOW,
            finished_at=NOW,
        )
    reconciliation, reconciliation_ref = reconcile_cross_layer_bridge_v40(
        development,
        modeling,
        runtime.node_hash,
        reconciliation_id="release_current",
        checked_by="verifier",
        checked_at=NOW,
    )
    assert reconciliation.decision == "valid"
    modeling.record_outcome(
        runtime.node_hash,
        actor="harness",
        status="succeeded",
        output_artifacts=[runtime_output],
        summary="reconciled runtime release was evaluated locally",
        outcome_id="reconciled_runtime_execution",
        started_at=NOW,
        finished_at=NOW,
        bridge_source=development,
        bridge_reconciliation_ref=reconciliation_ref,
    )

    development.revoke_node(
        patch.node_hash,
        authority="human",
        reason="anchored regression later invalidated the patch",
        revocation_id="patch_revoked",
        revoked_at=NOW,
    )
    source_state = development.project_state()
    assert source_state.snapshot.node_statuses[release.node_hash] == "revoked"
    assert bridge.source_snapshot_hash != source_state.snapshot.snapshot_hash
    invalid, _ = reconcile_cross_layer_bridge_v40(
        development,
        modeling,
        runtime.node_hash,
        reconciliation_id="release_revoked",
        checked_by="verifier",
        checked_at=NOW,
    )
    assert invalid.decision == "invalid"
    assert "release_is_not_succeeded" in invalid.reasons
    assert modeling.project_state().snapshot.node_statuses[runtime.node_hash] == "revoked"


def test_failure_edge_opens_learning_frontier_and_budget_stops(tmp_path) -> None:
    store = GraphLoopStoreV40(
        tmp_path,
        _contract(
            "failure_learning_graph",
            "modeling",
            max_outcomes=2,
            max_failures=2,
        ),
    )
    execution = _node("execution", "modeling", "execution", "harness")
    failure = _node("failure", "modeling", "failure", "verifier")
    experience = _node("experience", "modeling", "experience", "harness")
    for node in [execution, failure, experience]:
        store.add_node(node)
    store.add_edge(
        _edge(
            "execution_failure",
            "modeling",
            execution,
            failure,
            "learned_from_failure",
        )
    )
    store.add_edge(
        _edge(
            "failure_experience",
            "modeling",
            failure,
            experience,
            "requires_terminal",
        )
    )
    _finish(store, execution, "harness", status="failed")
    assert store.project_state().snapshot.frontier_node_hashes == [failure.node_hash]
    _finish(store, failure, "verifier")
    snapshot = store.project_state().snapshot
    assert snapshot.budget_exhausted
    assert snapshot.stop_reason == "budget_exhausted"
    assert snapshot.frontier_node_hashes == []


def test_cycle_and_event_tampering_fail_closed(tmp_path) -> None:
    store = GraphLoopStoreV40(tmp_path, _contract("tamper_graph", "development"))
    first = _node("first", "development", "design", "model", created_by="model")
    second = _node("second", "development", "patch", "model", created_by="model")
    store.add_node(first)
    store.add_node(second)
    store.add_edge(_edge("first_second", "development", first, second, "requires_success"))
    with pytest.raises(ValueError, match="cycle"):
        store.add_edge(
            _edge("second_first", "development", second, first, "requires_success")
        )

    records = store.store.event_path.read_text(encoding="utf-8").splitlines()
    event = json.loads(records[-1])
    event["payload"]["kind"] = "tampered"
    records[-1] = json.dumps(event)
    store.store.event_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    assert not store.verify()
