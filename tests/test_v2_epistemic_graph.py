from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fma.hashing import sha256_value
from fma.v2.epistemic_graph import (
    EpistemicEdgeV22,
    EpistemicGraphStore,
    EpistemicNodeV22,
    register_method_learning_run,
    register_official_shadow_run,
    register_worldpack_confirmation_run,
    register_worldpack_run,
)
from fma.v2.method_knowledge import MethodFetchResponse, MethodSourceContractV22, capture_method_candidate
from fma.v2.worldpack import default_worldpack_spec, run_worldpack_ablation


NOW = datetime(2026, 7, 22, tzinfo=timezone.utc)


def _node(name: str, kind: str = "evidence") -> EpistemicNodeV22:
    return EpistemicNodeV22.seal(
        node_id=name,
        node_kind=kind,
        artifact_hash=sha256_value({"artifact": name}),
        intended_uses=["test only"],
        created_at=NOW,
    )


def _edge(
    edge_id: str,
    source: EpistemicNodeV22,
    target: EpistemicNodeV22,
    relation: str = "derived_from",
) -> EpistemicEdgeV22:
    return EpistemicEdgeV22.seal(
        edge_id=edge_id,
        source_node_hash=source.node_hash,
        target_node_hash=target.node_hash,
        relation=relation,
        rationale="frozen test dependency",
        created_at=NOW,
    )


def test_revocation_cascades_only_through_derivation_relations(tmp_path) -> None:
    graph = EpistemicGraphStore(tmp_path, "epistemic_cascade")
    source = _node("source", "source")
    snapshot = _node("snapshot", "data_snapshot")
    validation = _node("validation", "validation_report")
    skill = _node("skill", "skill_report")
    decision = _node("decision", "decision_report")
    operator = _node("operator", "evolution_operator")
    refuter = _node("refuter", "method_claim")
    for node in [source, snapshot, validation, skill, decision, operator, refuter]:
        graph.add_node(node)
    for edge in [
        _edge("source_to_snapshot", source, snapshot),
        _edge("snapshot_to_validation", snapshot, validation, "evaluates"),
        _edge("validation_to_skill", validation, skill),
        _edge("skill_to_decision", skill, decision, "justifies_use"),
        _edge("skill_to_operator", skill, operator, "learned_from"),
        _edge("refuter_to_decision", refuter, decision, "refutes"),
    ]:
        graph.add_edge(edge)

    receipt = graph.revoke_node(
        source.node_hash,
        reason="upstream source was withdrawn",
        receipt_id="source_revocation",
        revoked_at=NOW,
    )
    assert receipt.affected_node_hashes == sorted(
        [
            source.node_hash,
            snapshot.node_hash,
            validation.node_hash,
            skill.node_hash,
            decision.node_hash,
            operator.node_hash,
        ]
    )
    state = graph.project_state()
    assert state.snapshot.node_statuses[refuter.node_hash] == "active"
    for node in [source, snapshot, validation, skill, decision, operator]:
        assert state.snapshot.node_statuses[node.node_hash] == "revoked"
    assert graph.verify()
    assert EpistemicGraphStore.open_existing(graph.run_directory).verify()


def test_refutation_and_supersession_change_status_without_revocation_cascade(
    tmp_path,
) -> None:
    graph = EpistemicGraphStore(tmp_path, "epistemic_semantics")
    old_claim = _node("old_claim", "method_claim")
    new_claim = _node("new_claim", "method_claim")
    challenged = _node("challenged_claim", "method_claim")
    refuter = _node("refuting_claim", "method_claim")
    for node in [old_claim, new_claim, challenged, refuter]:
        graph.add_node(node)
    graph.add_edge(_edge("new_supersedes_old", new_claim, old_claim, "supersedes"))
    graph.add_edge(_edge("refuter_refutes_claim", refuter, challenged, "refutes"))

    before = graph.project_state().snapshot.node_statuses
    assert before[old_claim.node_hash] == "superseded"
    assert before[challenged.node_hash] == "refuted"
    graph.revoke_node(
        refuter.node_hash,
        reason="refuting claim evidence was withdrawn",
        receipt_id="refuter_revocation",
        revoked_at=NOW,
    )
    after = graph.project_state().snapshot.node_statuses
    assert after[refuter.node_hash] == "revoked"
    assert after[challenged.node_hash] == "refuted"


def test_cycle_and_duplicate_node_ids_are_rejected(tmp_path) -> None:
    graph = EpistemicGraphStore(tmp_path, "epistemic_cycles")
    first = _node("first")
    second = _node("second")
    graph.add_node(first)
    graph.add_node(second)
    graph.add_edge(_edge("first_to_second", first, second))
    with pytest.raises(ValueError, match="cycle"):
        graph.add_edge(_edge("second_to_first", second, first))
    duplicate_id = EpistemicNodeV22.seal(
        node_id="first",
        node_kind="evidence",
        artifact_hash=sha256_value({"different": True}),
        intended_uses=["test only"],
        created_at=NOW,
    )
    with pytest.raises(ValueError, match="node_id"):
        graph.add_node(duplicate_id)


def test_artifact_and_event_tampering_fail_verification(tmp_path) -> None:
    artifact_graph = EpistemicGraphStore(tmp_path / "artifact", "tamper_artifact")
    node = _node("artifact_node")
    ref = artifact_graph.add_node(node)
    path = artifact_graph.run_directory / ref.relative_path
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["intended_uses"] = ["tampered use"]
    path.write_text(json.dumps(envelope), encoding="utf-8")
    assert not artifact_graph.verify()

    event_graph = EpistemicGraphStore(tmp_path / "event", "tamper_event")
    event_graph.add_node(_node("event_node"))
    records = event_graph.store.event_path.read_text(encoding="utf-8").splitlines()
    event = json.loads(records[-1])
    event["payload"]["kind"] = "wrong_kind"
    records[-1] = json.dumps(event)
    event_graph.store.event_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    assert not event_graph.verify()


def test_method_source_revocation_overrides_hidden_benchmark_verdict(tmp_path) -> None:
    url = "https://otexts.com/fpp3/ses.html"
    contract = MethodSourceContractV22.seal(
        source_id="fpp3_graph_fixture",
        source_url=url,
        max_bytes=1024,
        approved_claim_scope=["simple exponential smoothing recurrence and limits"],
        created_at=NOW,
    )

    def fetcher(source_url: str, max_bytes: int) -> MethodFetchResponse:
        return MethodFetchResponse(
            200,
            source_url,
            {"content-type": "text/html"},
            b"<html><body>SES fixture evidence</body></html>",
        )

    method = capture_method_candidate(
        tmp_path / "method",
        contract=contract,
        claim_id="ses_graph_claim",
        statement="SES geometrically discounts observations from the distant past.",
        applicability_conditions=["a changing local level is plausible"],
        exclusions=["trend and seasonality need separate components"],
        proposed_operator="exponential_smoothing",
        frozen_parameters={"alpha": 0.3},
        fetcher=fetcher,
        captured_at=NOW,
        run_id="method_graph_fixture",
    )
    worldpack = run_worldpack_ablation(
        tmp_path / "worldpack",
        spec=default_worldpack_spec(frozen_at=NOW),
        evaluated_at=NOW,
        run_id="worldpack_graph_fixture",
    )
    graph = EpistemicGraphStore(tmp_path / "graph", "method_worldpack_graph")
    method_nodes = register_method_learning_run(graph, method.store.run_directory)
    benchmark_nodes = register_worldpack_run(graph, worldpack.store.run_directory)
    graph.add_edge(
        EpistemicEdgeV22.seal(
            edge_id="ses_component_supports_memory_policy",
            source_node_hash=method_nodes["operator_memory"],
            target_node_hash=benchmark_nodes["worldpack_memory_policy"],
            relation="supports",
            rationale="SES is an explicit component of the tested memory policy",
            created_at=NOW,
        )
    )
    state = graph.project_state()
    assert state.snapshot.node_statuses[method_nodes["operator_memory"]] == "active"
    assert (
        state.snapshot.node_statuses[benchmark_nodes["worldpack_memory_policy"]]
        == "refuted"
    )
    assert state.snapshot.node_statuses[benchmark_nodes["worldpack_report"]] == "active"

    graph.revoke_node(
        method_nodes["method_source"],
        reason="captured web source was withdrawn",
        receipt_id="method_source_withdrawal",
        revoked_at=NOW,
    )
    after = graph.project_state().snapshot.node_statuses
    assert after[method_nodes["method_source"]] == "revoked"
    assert after[method_nodes["method_claim"]] == "revoked"
    assert after[method_nodes["operator_memory"]] == "revoked"
    assert after[benchmark_nodes["worldpack_memory_policy"]] == "revoked"
    assert after[benchmark_nodes["worldpack_report"]] == "active"


def test_verified_official_run_revocation_invalidates_downstream_memory(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    matches = list(
        (root / "experiments" / "iteration_02" / "confirmation_v22").glob(
            "official-shadow-bls_private_weekly_hours-*"
        )
    )
    assert len(matches) == 1
    graph = EpistemicGraphStore(tmp_path, "official_evidence_graph")
    nodes = register_official_shadow_run(graph, matches[0])
    assert graph.verify()
    graph.revoke_node(
        nodes["raw_source"],
        reason="official provider revised the captured response",
        receipt_id="official_revision",
        revoked_at=NOW,
    )
    statuses = graph.project_state().snapshot.node_statuses
    for label in [
        "raw_source",
        "source_receipt",
        "data_snapshot",
        "validation",
        "skill",
        "shift",
        "decision",
        "operator_memory",
    ]:
        assert statuses[nodes[label]] == "revoked"
    assert statuses[nodes["model_portfolio"]] == "active"


def test_only_passing_exact_policy_creates_revocable_qualification(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    graph = EpistemicGraphStore(tmp_path, "qualification_graph")
    exploratory = register_worldpack_run(
        graph,
        root
        / "experiments"
        / "iteration_03"
        / "worldpack"
        / "worldpack-ablation-v22",
    )
    rejected = register_worldpack_confirmation_run(
        graph,
        root
        / "experiments"
        / "iteration_04"
        / "worldpack_confirmation"
        / "worldpack-confirmation-v23",
        existing_memory_policy_node_hash=exploratory["worldpack_memory_policy"],
    )
    assert "qualified_policy" not in rejected
    assert (
        graph.project_state().snapshot.node_statuses[
            exploratory["worldpack_memory_policy"]
        ]
        == "refuted"
    )

    passing = register_worldpack_confirmation_run(
        graph,
        root
        / "experiments"
        / "iteration_05"
        / "safe_policy_confirmation"
        / "safe-policy-confirmation-v23",
    )
    statuses = graph.project_state().snapshot.node_statuses
    assert statuses[passing["memory_policy"]] == "active"
    assert statuses[passing["qualified_policy"]] == "active"
    graph.revoke_node(
        passing["confirmation_report"],
        reason="simulated benchmark invalidation drill",
        receipt_id="qualification_report_drill",
        revoked_at=NOW,
    )
    after = graph.project_state().snapshot.node_statuses
    assert after[passing["confirmation_report"]] == "revoked"
    assert after[passing["qualified_policy"]] == "revoked"
    assert after[passing["memory_policy"]] == "active"
