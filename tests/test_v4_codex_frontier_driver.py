from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fma.codex_driver import CodexCLIConfig, ProcessResult
from fma.hashing import sha256_value
from fma.v4 import (
    CodexCLIFrontierTransportV40,
    CodexFrontierDriverV40,
    FixtureFrontierTransportV40,
    FrontierNodeViewV40,
    FrontierProposalV40,
    FrontierRequestV40,
    GraphEdgeV40,
    GraphLoopContractV40,
    GraphLoopStoreV40,
    GraphNodeV40,
)


NOW = datetime(2026, 7, 23, tzinfo=timezone.utc)


class _FrontierCodexRunner:
    def __init__(self) -> None:
        self.exec_argv: list[str] = []
        self.prompt = ""

    def __call__(
        self,
        argv: list[str],
        *,
        cwd: Path,
        input_text: str | None,
        timeout_seconds: int,
        env: dict[str, str],
        max_stdout_bytes: int,
        max_stderr_bytes: int,
    ) -> ProcessResult:
        del cwd, timeout_seconds, env, max_stdout_bytes, max_stderr_bytes
        if argv[-1] == "--version":
            return ProcessResult(0, "codex-cli 0.144.6\n", "")
        if "login" in argv and "status" in argv:
            return ProcessResult(0, "Logged in using ChatGPT\n", "")
        if "mcp" in argv and "list" in argv:
            disabled = any("mcp_servers." in value for value in argv)
            return ProcessResult(
                0,
                json.dumps([{"name": "node_repl", "enabled": not disabled}]),
                "",
            )
        assert "exec" in argv
        assert input_text is not None
        self.exec_argv = list(argv)
        self.prompt = input_text
        public = json.loads(input_text.split("INPUT_JSON\n", 1)[1])
        proposal = FrontierProposalV40(
            request_hash=public["request_hash"],
            action="execute",
            selected_node_hash=public["candidate_nodes"][0]["node_hash"],
            draft="bounded candidate from fake Codex transport",
            rationale="select the only public model-owned node",
        )
        events = [
            {"type": "thread.started", "thread_id": "frontier-thread"},
            {"type": "turn.started"},
            {
                "type": "item.completed",
                "item": {
                    "id": "message-1",
                    "type": "agent_message",
                    "text": proposal.model_dump_json(),
                },
            },
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 0,
                    "output_tokens": 40,
                    "reasoning_output_tokens": 5,
                },
            },
        ]
        return ProcessResult(
            0,
            "\n".join(json.dumps(event) for event in events) + "\n",
            "",
        )


def _graph(tmp_path) -> tuple[GraphLoopStoreV40, GraphNodeV40, GraphNodeV40]:
    contract = GraphLoopContractV40.seal(
        graph_id="frontier_driver_graph",
        layer="modeling",
        evaluator_epoch="anchor-v1",
        objective="test model-only frontier transport",
        created_at=NOW,
    )
    graph = GraphLoopStoreV40(tmp_path, contract)
    candidate = GraphNodeV40.seal(
        node_id="draft_candidate",
        layer="modeling",
        node_kind="model_candidate",
        executor="model",
        created_by="harness",
        artifact_hash=sha256_value({"candidate": 1}),
        purpose="draft a bounded public model candidate",
        created_at=NOW,
    )
    evaluator = GraphNodeV40.seal(
        node_id="private_evaluator",
        layer="modeling",
        node_kind="evaluation",
        executor="verifier",
        created_by="harness",
        artifact_hash=sha256_value({"evaluator": 1}),
        purpose="evaluate candidate with hidden acceptance cases",
        created_at=NOW,
    )
    graph.add_node(candidate)
    graph.add_node(evaluator)
    graph.add_edge(
        GraphEdgeV40.seal(
            edge_id="candidate_private_evaluation",
            layer="modeling",
            source_node_hash=candidate.node_hash,
            target_node_hash=evaluator.node_hash,
            relation="evaluated_by",
            rationale="private evaluator follows the untrusted draft",
            created_at=NOW,
        )
    )
    return graph, candidate, evaluator


def test_frontier_driver_executes_only_model_node_then_pauses_for_verifier(
    tmp_path,
) -> None:
    graph, candidate, evaluator = _graph(tmp_path)

    def proposal(request):
        assert not request.private_evidence_exposed
        assert not request.authority_fields_exposed
        assert [item.node_hash for item in request.candidate_nodes] == [
            candidate.node_hash
        ]
        return FrontierProposalV40(
            request_hash=request.request_hash,
            action="execute",
            selected_node_hash=candidate.node_hash,
            draft="candidate family: bounded symbolic dynamics",
            rationale="the only model-owned node is ready",
        )

    driver = CodexFrontierDriverV40(FixtureFrontierTransportV40(proposal))
    first = driver.run_once(graph, receipt_id="driver_step_1", created_at=NOW)
    assert first.receipt.status == "executed"
    assert first.receipt.selected_node_hash == candidate.node_hash
    assert first.receipt.next_required_actors == ["verifier"]
    state = graph.project_state()
    assert state.snapshot.node_statuses[candidate.node_hash] == "succeeded"
    assert state.snapshot.frontier_node_hashes == [evaluator.node_hash]

    second = driver.run_once(graph, receipt_id="driver_step_2", created_at=NOW)
    assert second.receipt.status == "waiting_for_non_model_executor"
    assert second.receipt.next_required_actors == ["verifier"]


def test_frontier_driver_rejects_selection_outside_model_frontier(tmp_path) -> None:
    graph, _, evaluator = _graph(tmp_path)

    def proposal(request):
        return FrontierProposalV40(
            request_hash=request.request_hash,
            action="execute",
            selected_node_hash=evaluator.node_hash,
            draft="attempt to claim evaluator authority",
            rationale="malicious authority escalation attempt",
        )

    driver = CodexFrontierDriverV40(FixtureFrontierTransportV40(proposal))
    outcome = driver.run_once(graph, receipt_id="driver_denial", created_at=NOW)
    assert outcome.receipt.status == "transport_error"
    assert "outside its frontier" in outcome.receipt.error
    assert all(
        status == "pending"
        for status in graph.project_state().snapshot.node_statuses.values()
    )


def test_codex_cli_transport_is_tool_free_ephemeral_and_public_only(tmp_path) -> None:
    fake_cli = tmp_path / "codex.exe"
    fake_cli.write_bytes(b"fake-codex-cli")
    runner = _FrontierCodexRunner()
    request = FrontierRequestV40.seal(
        request_id="frontier_transport_request",
        graph_id="frontier_transport_graph",
        layer="modeling",
        graph_snapshot_hash=sha256_value({"snapshot": 1}),
        candidate_nodes=[
            FrontierNodeViewV40(
                node_id="candidate",
                node_hash=sha256_value({"candidate": 1}),
                node_kind="model_candidate",
                purpose="draft one bounded public candidate",
            )
        ],
        created_at=NOW,
    )
    transport = CodexCLIFrontierTransportV40(
        tmp_path / "transport",
        CodexCLIConfig(executable=fake_cli, timeout_seconds=30),
        process_runner=runner,
        cli_locator=lambda explicit: fake_cli,
    )
    try:
        proposal = transport.propose(request)
    finally:
        transport.close()

    assert proposal.request_hash == request.request_hash
    assert proposal.selected_node_hash == request.candidate_nodes[0].node_hash
    public = json.loads(runner.prompt.split("INPUT_JSON\n", 1)[1])
    assert public["private_evidence_exposed"] is False
    assert public["authority_fields_exposed"] is False
    assert "acceptance_tests" not in runner.prompt
    assert "--ephemeral" in runner.exec_argv
    assert "--ignore-user-config" in runner.exec_argv
    assert "read-only" in runner.exec_argv
    assert "never" in runner.exec_argv
    schemas = list((tmp_path / "transport").glob(
        "frontier-transport-*/cli_calls/*/frontier-output.schema.json"
    ))
    assert len(schemas) == 1
    schema = json.loads(schemas[0].read_text(encoding="utf-8"))
    assert set(schema["required"]) == set(schema["properties"])
    assert all("default" not in item for item in schema["properties"].values())
