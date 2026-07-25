from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fma.codex_driver import CodexCLIConfig, ProcessResult
from fma.hashing import sha256_value
from fma.v4.codex_open_evolution import (
    CodexCLIOpenEvolutionTransportV42,
    GeneratedModelDraftV42,
    OpenEvolutionCandidateViewV42,
    OpenEvolutionFailureViewV42,
    OpenEvolutionGenerationRequestV42,
    OpenEvolutionGenerationResponseV42,
    generation_call_evidence_v42,
)
from fma.v4.event_process_open_evolution import (
    _single_hawkes_candidate_v42,
    event_process_open_grammar_v42,
)


NOW = datetime(2026, 7, 23, tzinfo=timezone.utc)


class _OpenEvolutionCodexRunner:
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
        response = OpenEvolutionGenerationResponseV42(
            request_hash=public["request_hash"],
            action="stop",
            proposals=[],
            rationale="no bounded structure proposal is needed for transport testing",
        )
        events = [
            {"type": "thread.started", "thread_id": "open-evolution-thread"},
            {"type": "turn.started"},
            {
                "type": "item.completed",
                "item": {
                    "id": "message-1",
                    "type": "agent_message",
                    "text": response.model_dump_json(),
                },
            },
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 200,
                    "cached_input_tokens": 0,
                    "output_tokens": 50,
                    "reasoning_output_tokens": 5,
                },
            },
        ]
        return ProcessResult(
            0,
            "\n".join(json.dumps(event) for event in events) + "\n",
            "",
        )


def _request() -> OpenEvolutionGenerationRequestV42:
    grammar = event_process_open_grammar_v42()
    candidate = _single_hawkes_candidate_v42()
    return OpenEvolutionGenerationRequestV42.seal(
        request_id="open_evolution_transport_request",
        objective="Generate one bounded public event-process structure draft.",
        grammar_id=grammar.grammar_id,
        grammar_hash=grammar.grammar_hash,
        allowed_primitives=grammar.primitives,
        executable_adapter_id=grammar.executable_adapter_ids[0],
        max_symbols=grammar.max_symbols,
        max_applications=grammar.max_applications,
        current_candidate=OpenEvolutionCandidateViewV42(
            family=candidate.family,
            symbols=candidate.symbols,
            applications=candidate.applications,
            assumptions=candidate.assumptions,
            rationale=candidate.rationale,
        ),
        failure=OpenEvolutionFailureViewV42(
            failed_gates=["parameter_interior"],
            diagnostic_codes=["parameter_boundary"],
            sanitized_summary="The fitted decay parameter reached a frozen boundary.",
        ),
        development_metrics={"candidate_training_bic": 123.0},
        adapter_guidance=["Return an acyclic intensity expression."],
        created_at=NOW,
    )


def test_codex_open_evolution_transport_is_strict_ephemeral_and_public(
    tmp_path,
) -> None:
    fake_cli = tmp_path / "codex.exe"
    fake_cli.write_bytes(b"fake-codex-cli")
    runner = _OpenEvolutionCodexRunner()
    transport = CodexCLIOpenEvolutionTransportV42(
        tmp_path / "transport",
        CodexCLIConfig(executable=fake_cli, timeout_seconds=30),
        process_runner=runner,
        cli_locator=lambda explicit: fake_cli,
    )
    request = _request()
    try:
        response = transport.propose(request)
        evidence = generation_call_evidence_v42(
            request, response, transport
        )
    finally:
        transport.close()

    assert response.request_hash == request.request_hash
    assert evidence.transport == "codex_cli"
    public = json.loads(runner.prompt.split("INPUT_JSON\n", 1)[1])
    assert public["private_evidence_exposed"] is False
    assert public["authority_fields_exposed"] is False
    assert public["tools_permitted"] is False
    assert "confirmation_snapshot" not in runner.prompt
    assert "acceptance_tests" not in runner.prompt
    assert "--ephemeral" in runner.exec_argv
    assert "--ignore-user-config" in runner.exec_argv
    assert "read-only" in runner.exec_argv
    assert "never" in runner.exec_argv
    schemas = list(
        (tmp_path / "transport").glob(
            "open-evolution-transport-*/cli_calls/*/"
            "open-evolution-output.schema.json"
        )
    )
    assert len(schemas) == 1
    schema = json.loads(schemas[0].read_text(encoding="utf-8"))
    assert set(schema["required"]) == set(schema["properties"])
    assert set(schema["$defs"]["GeneratedModelDraftV42"]["required"]) == set(
        schema["$defs"]["GeneratedModelDraftV42"]["properties"]
    )
    assert all(
        "default" not in value
        for value in schema["$defs"]["GeneratedModelDraftV42"][
            "properties"
        ].values()
    )
