from __future__ import annotations

import json
import sys
from pathlib import Path

from fma.codex_driver import CodexCLIConfig, ProcessResult
from fma.v2.capacity_planning import (
    capacity_brief_snapshot,
    capacity_codex_discovery_mission,
    run_codex_capacity_discovery_fixture,
)
from fma.v2.codex_discovery import (
    DISCOVERY_PROTOCOL_VERSION,
    CodexProblemDiscoveryExplorer,
)
from fma.v2.discovery_store import DiscoveryRunStore


def _jsonl(final: str, *, forbidden_item: str | None = None) -> str:
    events: list[dict[str, object]] = [
        {"type": "thread.started", "thread_id": "discovery-thread"},
        {"type": "turn.started"},
    ]
    if forbidden_item:
        events.append(
            {
                "type": "item.started",
                "item": {"id": "tool-1", "type": forbidden_item, "status": "in_progress"},
            }
        )
    events.extend(
        [
            {
                "type": "item.completed",
                "item": {"id": "message-1", "type": "agent_message", "text": final},
            },
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
        ]
    )
    return "\n".join(json.dumps(event) for event in events) + "\n"


class FakeCodexDiscoveryRunner:
    def __init__(self, response_factory, *, forbidden_item: str | None = None) -> None:
        self.response_factory = response_factory
        self.forbidden_item = forbidden_item
        self.prompts: list[str] = []
        self.exec_argv: list[list[str]] = []
        self.exec_envs: list[dict[str, str]] = []

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
        del cwd, timeout_seconds, max_stdout_bytes, max_stderr_bytes
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
        assert "exec" in argv and input_text is not None
        self.prompts.append(input_text)
        self.exec_argv.append(list(argv))
        self.exec_envs.append(dict(env))
        response = self.response_factory(input_text)
        return ProcessResult(
            0,
            _jsonl(json.dumps(response), forbidden_item=self.forbidden_item),
            "",
        )


def _proposed_response(prompt: str) -> dict[str, object]:
    payload = json.loads(prompt.split("INPUT_JSON\n", 1)[1])
    return {
        "protocol_version": DISCOVERY_PROTOCOL_VERSION,
        "request_id": payload["request_id"],
        "status": "proposed",
        "draft": {
            "draft_id": "codex_capacity_draft",
            "mission_spec_hash": payload["mission_spec_hash"],
            "evidence_snapshot_hashes": [
                payload["evidence"]["evidence_snapshot_hash"]
            ],
            "statement": "Regular capacity may be insufficient to meet stated demand in one period.",
            "observed_symptoms": [
                "The brief gives five units of demand and three regular units of capacity"
            ],
            "proposed_value": "Determine whether a bounded cost-and-capacity model is needed",
            "assumptions": ["The brief describes one planning period"],
            "open_questions": ["Whether inventory or backlog is allowed is unresolved"],
        },
        "notes": [],
    }


def _prepared_store(tmp_path: Path) -> tuple[DiscoveryRunStore, object]:
    mission = capacity_codex_discovery_mission()
    snapshot = capacity_brief_snapshot()
    store = DiscoveryRunStore(tmp_path, run_id="codex-discovery")
    store.start(mission)
    store.ingest_evidence(snapshot)
    return store, snapshot


def test_codex_problem_discovery_is_draft_only_and_bound_to_its_observation(
    tmp_path: Path, monkeypatch
) -> None:
    fake_cli = tmp_path / "codex.exe"
    fake_cli.write_bytes(b"fake-codex-cli")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-child-process")
    runner = FakeCodexDiscoveryRunner(_proposed_response)
    store, snapshot = _prepared_store(tmp_path / "run")
    explorer = CodexProblemDiscoveryExplorer(
        store,
        CodexCLIConfig(executable=fake_cli, max_candidates=1, timeout_seconds=30),
        process_runner=runner,
        cli_locator=lambda explicit: fake_cli,
    )

    proposal = explorer.propose(store.build_problem_discovery_context(snapshot))
    assert proposal.status == "proposed"
    assert proposal.draft is not None
    assert proposal.provider_observation_ref is not None
    admission = store.submit_and_admit(
        snapshot,
        proposal.draft,
        provider_observation_ref=proposal.provider_observation_ref,
    )

    assert admission.status == "admitted"
    state = store.project_state()
    assert state.event_count == 5
    assert len(state.provider_observation_hashes) == 1
    assert store.verify()
    assert "untrusted data" in runner.prompts[0]
    assert "known_global_optimum" not in runner.prompts[0]
    assert "read-only" in runner.exec_argv[0]
    assert "--ephemeral" in runner.exec_argv[0]
    assert "shell_tool" in runner.exec_argv[0]
    assert "OPENAI_API_KEY" not in runner.exec_envs[0]
    explorer.close()


def test_codex_no_result_is_recorded_without_a_draft(tmp_path: Path) -> None:
    def no_result(prompt: str) -> dict[str, object]:
        payload = json.loads(prompt.split("INPUT_JSON\n", 1)[1])
        return {
            "protocol_version": DISCOVERY_PROTOCOL_VERSION,
            "request_id": payload["request_id"],
            "status": "no_result",
            "no_result_reason": "The brief does not identify a bounded modeling question.",
            "notes": [],
        }

    fake_cli = tmp_path / "codex.exe"
    fake_cli.write_bytes(b"fake-codex-cli")
    store, snapshot = _prepared_store(tmp_path / "run")
    explorer = CodexProblemDiscoveryExplorer(
        store,
        CodexCLIConfig(executable=fake_cli, max_candidates=1),
        process_runner=FakeCodexDiscoveryRunner(no_result),
        cli_locator=lambda explicit: fake_cli,
    )

    proposal = explorer.propose(store.build_problem_discovery_context(snapshot))

    assert proposal.status == "no_result"
    assert proposal.draft is None
    assert proposal.terminal_code == "model_no_result"
    assert store.project_state().event_count == 3
    assert store.verify()
    explorer.close()


def test_observed_codex_tool_event_fails_closed_and_cannot_create_a_draft(
    tmp_path: Path,
) -> None:
    fake_cli = tmp_path / "codex.exe"
    fake_cli.write_bytes(b"fake-codex-cli")
    store, snapshot = _prepared_store(tmp_path / "run")
    explorer = CodexProblemDiscoveryExplorer(
        store,
        CodexCLIConfig(executable=fake_cli, max_candidates=1),
        process_runner=FakeCodexDiscoveryRunner(
            _proposed_response, forbidden_item="command_execution"
        ),
        cli_locator=lambda explicit: fake_cli,
    )

    proposal = explorer.propose(store.build_problem_discovery_context(snapshot))

    assert proposal.status == "error"
    assert proposal.draft is None
    assert proposal.terminal_code == "policy_violation"
    assert store.project_state().event_count == 3
    assert store.verify()
    explorer.close()


def test_extra_authority_in_codex_discovery_output_is_a_terminal_error(
    tmp_path: Path,
) -> None:
    def authority_attempt(prompt: str) -> dict[str, object]:
        response = _proposed_response(prompt)
        assert isinstance(response["draft"], dict)
        response["draft"]["promotion_status"] = "validated"
        return response

    fake_cli = tmp_path / "codex.exe"
    fake_cli.write_bytes(b"fake-codex-cli")
    store, snapshot = _prepared_store(tmp_path / "run")
    explorer = CodexProblemDiscoveryExplorer(
        store,
        CodexCLIConfig(executable=fake_cli, max_candidates=1),
        process_runner=FakeCodexDiscoveryRunner(authority_attempt),
        cli_locator=lambda explicit: fake_cli,
    )

    proposal = explorer.propose(store.build_problem_discovery_context(snapshot))

    assert proposal.status == "error"
    assert proposal.terminal_code == "output_schema_or_transport_error"
    assert store.project_state().event_count == 3
    assert store.verify()
    explorer.close()


def test_codex_capacity_fixture_wraps_only_discovery_and_admission(tmp_path: Path) -> None:
    fake_cli = tmp_path / "codex.exe"
    fake_cli.write_bytes(b"fake-codex-cli")
    store, proposal, outcome = run_codex_capacity_discovery_fixture(
        tmp_path / "run",
        CodexCLIConfig(executable=fake_cli, max_candidates=1),
        process_runner=FakeCodexDiscoveryRunner(_proposed_response),
        cli_locator=lambda explicit: fake_cli,
    )

    assert proposal.status == "proposed"
    assert outcome is not None and outcome.status == "admitted"
    assert store.project_state().event_count == 5
    assert store.verify()


def test_live_codex_cli_command_requires_an_explicit_flag(
    tmp_path: Path, monkeypatch
) -> None:
    from fma.__main__ import main

    output = tmp_path / "would-be-live-output"
    monkeypatch.setattr(
        sys,
        "argv",
        ["fma", "v2-codex-discovery-fixture", "--output", str(output)],
    )

    try:
        main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("live Codex discovery command must require --live")
    assert not output.exists()
