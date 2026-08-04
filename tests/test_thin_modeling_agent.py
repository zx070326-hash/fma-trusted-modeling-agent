from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

import modeling_agent.model as model_module
import modeling_agent.tools as tools_module
import modeling_agent.verification as verification_module
from modeling_agent.ablation import freeze_ablation, record_result
from modeling_agent.cli import _command_exit_code, _operator_projection, build_parser, main
from modeling_agent.engine import ModelingEngine, run_status
from modeling_agent.model import ScriptedModel
from modeling_agent.research import ResearchStore
from modeling_agent.sources import SOURCE_GATE_NOT_RUN_PREFIX, SourceGate
from modeling_agent.storage import (
    RunLayout,
    RunStore,
    atomic_write_json,
    content_hash,
    file_hash,
    run_lock,
    safe_path,
)
from modeling_agent.tools import ToolRegistry
from modeling_agent.verification import (
    build_external_review_packet,
    candidate_fingerprint,
    default_contract,
    project_promotion,
    validate_contract,
    validate_external_review_bundle,
    validate_manifest,
)


@pytest.fixture(autouse=True)
def _direct_python_is_a_unit_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tools_module,
        "_python_command",
        lambda workspace, command: (command, "unit-test-fixture"),
    )


def _supported_review(claim_id: str = "computed-value") -> dict[str, object]:
    return {
        "verdict": "SUPPORTED",
        "claim_verdicts": [
            {
                "claim_id": claim_id,
                "verdict": "SUPPORTED",
                "max_authority": "E4",
                "findings": [],
            }
        ],
        "findings": [],
        "max_authority": "E4",
        "delivery_verdict": "SUPPORTED",
        "delivery_findings": [],
    }


def _two_claim_review(
    *,
    first_authority: str = "E4",
    second_authority: str = "E4",
    second_verdict: str = "SUPPORTED",
    overall_authority: str = "E4",
) -> dict[str, object]:
    return {
        "verdict": "SUPPORTED",
        "claim_verdicts": [
            {
                "claim_id": "computed-value",
                "verdict": "SUPPORTED",
                "max_authority": first_authority,
                "findings": [],
            },
            {
                "claim_id": "derived-value",
                "verdict": second_verdict,
                "max_authority": second_authority,
                "findings": [] if second_verdict == "SUPPORTED" else ["not supported"],
            },
        ],
        "findings": [],
        "max_authority": overall_authority,
        "delivery_verdict": "SUPPORTED",
        "delivery_findings": [],
    }


class _Researcher:
    def __init__(self, modes: list[str]):
        self.modes = deque(modes)
        self.calls: list[dict[str, object]] = []

    def run(
        self,
        prompt: str,
        *,
        role: str,
        workspace: Path,
        trace_path: Path,
        timeout_seconds: int,
        network_mode: str = "offline-compute",
    ) -> dict[str, object]:
        mode = self.modes.popleft()
        self.calls.append(
            {
                "prompt": prompt,
                "role": role,
                "workspace": workspace,
                "trace_path": trace_path,
                "network_mode": network_mode,
            }
        )
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("{}\n", encoding="utf-8")
        if mode == "branch-request":
            research = workspace / "research"
            research.mkdir(parents=True, exist_ok=True)
            (research / "branch_requests.json").write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "requests": [
                            {
                                "id": "falsifier",
                                "question": "Can a simpler route falsify the proposed structure?",
                                "purpose": "independent falsification",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            return {"role": role, "observable_tool_calls": 1}
        self._write_submission(workspace, mode=mode)
        if mode == "mutate-control":
            control = workspace.parent / ".modeling-agent" / "task-contract.json"
            control.write_text('{"schema":2,"objective":"tampered"}', encoding="utf-8")
        return {"role": role, "observable_tool_calls": 4, "network_mode": network_mode}

    @staticmethod
    def _write_submission(workspace: Path, *, mode: str) -> None:
        contract = json.loads((workspace / "task-contract.json").read_text(encoding="utf-8"))
        for directory in ("src", "checks", "data", "results", "paper", "research"):
            (workspace / directory).mkdir(parents=True, exist_ok=True)
        delivery_relative = contract.get("delivery_artifact", "paper/final.md")
        delivery_path = workspace / delivery_relative
        delivery_path.parent.mkdir(parents=True, exist_ok=True)
        generator = (
            "from pathlib import Path\n"
            "Path('results').mkdir(exist_ok=True)\n"
            "Path('results/value.json').write_text("
            "'{\\\"value\\\":42}\\n', encoding='utf-8')\n"
        )
        if mode == "undeclared-input":
            generator = (
                "import json\n"
                "from pathlib import Path\n"
                "value=int(Path('data/secret.txt').read_text(encoding='utf-8'))\n"
                "Path('results').mkdir(exist_ok=True)\n"
                "Path('results/value.json').write_text(json.dumps({'value':value}), encoding='utf-8')\n"
            )
        checker = (
            "import json\n"
            "from pathlib import Path\n"
            "value=json.loads(Path('results/value.json').read_text(encoding='utf-8'))['value']\n"
            "manifest=json.loads(Path('submission_manifest.json').read_text(encoding='utf-8'))\n"
            "assert manifest['schema'] == 2\n"
            + ("assert value == 41\n" if mode == "bad-check" else "assert value == 42\n")
        )
        (workspace / "src" / "solve.py").write_text(generator, encoding="utf-8")
        (workspace / "checks" / "check_value.py").write_text(checker, encoding="utf-8")
        (workspace / "results" / "value.json").write_text('{"value":42}\n', encoding="utf-8")
        if mode == "undeclared-input":
            (workspace / "data" / "secret.txt").write_text("42\n", encoding="utf-8")
        delivery_path.write_text(
            "# Result\n\nThe bounded local computation returns 42.\n", encoding="utf-8"
        )
        manifest = {
            "schema": 2,
            "contract_hash": content_hash(contract),
            "final_answer": "The bounded local computation returns 42.",
            "final_claim_ids": ["computed-value"],
            "claims": [
                {
                    "id": "computed-value",
                    "statement": "The bounded local computation returns 42.",
                    "claim_type": "computational",
                    "scope": "the supplied deterministic local fixture",
                    "dependencies": [],
                    "artifact_paths": ["results/value.json"],
                    "source_ids": [],
                    "required_check_ids": ["value-check"],
                    "baseline": "not applicable to this finite identity",
                    "falsifiers": ["a replay result other than 42"],
                    "decision_critical": True,
                    "requested_authority": "E4",
                }
            ],
            "limitations": ["This does not establish external validity."],
            "artifacts": [
                {"path": delivery_relative, "role": "paper"},
                {"path": "src/solve.py", "role": "generator"},
                {"path": "checks/check_value.py", "role": "check"},
                {"path": "results/value.json", "role": "result"},
            ],
            "generators": [
                {
                    "script": "src/solve.py",
                    "args": [],
                    "input_paths": [],
                    "expected_outputs": ["results/value.json"],
                    "timeout": 30,
                }
            ],
            "checks": [
                {
                    "id": "value-check",
                    "kind": "python_check",
                    "arguments": {"script": "checks/check_value.py"},
                    "claim_ids": ["computed-value"],
                }
            ],
        }
        if mode == "undeclared-input":
            manifest["artifacts"].append({"path": "data/secret.txt", "role": "input"})
        if mode == "self-seeded-output":
            manifest["generators"][0]["input_paths"] = ["results/value.json"]
        if mode == "ungenerated-claim-artifact":
            (workspace / "data" / "claimed.json").write_text(
                '{"value":42}\n', encoding="utf-8"
            )
            manifest["artifacts"].append(
                {"path": "data/claimed.json", "role": "result"}
            )
            manifest["claims"][0]["artifact_paths"] = ["data/claimed.json"]
        if mode == "long-paper":
            delivery_path.write_text(
                "# Result\n\n" + ("bounded text " * 9_000) + "\nunsupported tail claim\n",
                encoding="utf-8",
            )
        if mode == "paper-tail":
            delivery_path.write_text(
                "# Result\n\nThe bounded result is 42.\nUNIQUE_PAPER_TAIL_CLAIM\n",
                encoding="utf-8",
            )
        if mode == "exact-limit-paper":
            delivery_path.write_text("x" * 100_000, encoding="utf-8")
        if mode == "malformed-generator-list":
            manifest["generators"][0]["input_paths"] = [[]]
        if mode in {"two-claims", "dependent-claims"}:
            second = json.loads(json.dumps(manifest["claims"][0]))
            second.update(
                {
                    "id": "derived-value",
                    "statement": "The derived bounded value is also 42.",
                    "decision_critical": False,
                    "dependencies": (
                        ["computed-value"] if mode == "dependent-claims" else []
                    ),
                }
            )
            manifest["claims"].append(second)
            manifest["checks"][0]["claim_ids"].append("derived-value")
            manifest["final_claim_ids"].append("derived-value")
        if mode == "cyclic-claim":
            manifest["claims"][0]["dependencies"] = ["computed-value"]
        if mode == "paper-only":
            return
        (workspace / "submission_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )


class _BranchResearcher:
    def run(
        self,
        prompt: str,
        *,
        role: str,
        workspace: Path,
        trace_path: Path,
        timeout_seconds: int,
        network_mode: str = "offline-compute",
    ) -> dict[str, object]:
        assert (workspace / "branch_packet.json").is_file()
        (workspace / "branch_summary.json").write_text(
            json.dumps(
                {
                    "schema": 1,
                    "status": "challenged",
                    "conclusion": "The simpler baseline exposes an unsupported assumption.",
                    "observations": ["baseline matches the nominal result"],
                    "falsifiers": ["out-of-support stress test"],
                    "recommended_action": "narrow the main claim",
                }
            ),
            encoding="utf-8",
        )
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("{}\n", encoding="utf-8")
        return {"role": role, "observable_tool_calls": 1}


class _FailingBranchResearcher:
    def run(self, *args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("branch fixture failed")


class _TamperFailResearcher:
    def run(self, *args: object, **kwargs: object) -> dict[str, object]:
        workspace = Path(kwargs["workspace"])
        control = workspace.parent / ".modeling-agent" / "task-contract.json"
        control.write_text('{"schema":2,"objective":"tampered"}', encoding="utf-8")
        raise RuntimeError("failed after tampering")


class _InterruptResearcher:
    def run(self, *args: object, **kwargs: object) -> dict[str, object]:
        raise KeyboardInterrupt("simulated abrupt interruption")


class _SlowResearcher(_Researcher):
    def __init__(self, modes: list[str], delay: float):
        super().__init__(modes)
        self.delay = delay

    def run(self, *args: object, **kwargs: object) -> dict[str, object]:
        time.sleep(self.delay)
        return super().run(*args, **kwargs)


class _SlowVerifier(ScriptedModel):
    def __init__(self, responses: list[dict[str, object]], delay: float):
        super().__init__(responses)
        self.delay = delay

    def complete(self, *args: object, **kwargs: object) -> dict[str, object]:
        time.sleep(self.delay)
        return super().complete(*args, **kwargs)


class _InterruptVerifier:
    model = "scripted"

    def complete(self, *args: object, **kwargs: object) -> dict[str, object]:
        raise KeyboardInterrupt


def _engine(tmp_path: Path, researcher: _Researcher, verifier: ScriptedModel, **kwargs: object) -> ModelingEngine:
    return ModelingEngine(
        tmp_path,
        researcher=researcher,
        verifier=verifier,
        source_reviewer=ScriptedModel([]),
        model_requested=str(kwargs.pop("model_requested", "scripted")),
        max_attempts=int(kwargs.pop("max_attempts", 2)),
        max_seconds=int(kwargs.pop("max_seconds", 60)),
        **kwargs,
    )


def _operator_candidate(tmp_path: Path) -> tuple[dict[str, object], RunLayout]:
    contract = default_contract(
        "Review one bounded computation.", network_mode="offline-compute"
    )
    engine = ModelingEngine(
        tmp_path,
        researcher=None,
        verifier=None,
        source_reviewer=None,
        model_requested="current-codex-test",
        max_attempts=2,
        max_seconds=60,
    )
    engine.prepare(contract)
    layout = RunLayout.open(tmp_path)
    _Researcher._write_submission(layout.work, mode="valid")
    submitted = engine.submit(contract)
    assert submitted.status == "candidate"
    return contract, layout


def test_research_graph_projects_attempts_counterexamples_and_dangling_edges(
    tmp_path: Path,
) -> None:
    store = ResearchStore(tmp_path / "research.jsonl")
    store.append(
        {
            "id": "q1",
            "kind": "question",
            "statement": "Which mechanism explains the observation?",
        }
    )
    store.append(
        {
            "id": "a1",
            "kind": "attempt",
            "statement": "Fit the simplest competing mechanism.",
            "depends_on": ["q1"],
        }
    )
    store.append(
        {
            "id": "x1",
            "kind": "counterexample",
            "statement": "The candidate fails an extreme-case check.",
            "depends_on": ["a1", "missing-observation"],
        }
    )

    graph = store.project_graph()

    assert set(graph["nodes"]) == {"q1", "a1", "x1"}
    assert graph["nodes"]["a1"]["kind"] == "attempt"
    assert graph["nodes"]["x1"]["kind"] == "counterexample"
    assert graph["dangling_dependencies"] == ["missing-observation"]


def test_candidate_fingerprint_covers_nonpaper_declared_artifacts(
    tmp_path: Path,
) -> None:
    contract, layout = _operator_candidate(tmp_path)
    manifest, errors = validate_manifest(
        layout.work, contract, content_hash(contract)
    )
    assert manifest is not None and errors == []
    before = candidate_fingerprint(layout.work, manifest)

    (layout.work / "results" / "value.json").write_text(
        '{"value":43}\n', encoding="utf-8"
    )

    assert candidate_fingerprint(layout.work, manifest) != before


def test_external_review_import_preserves_freedom_but_caps_authority(
    tmp_path: Path,
) -> None:
    contract, layout = _operator_candidate(tmp_path)
    packet = build_external_review_packet(
        layout,
        contract,
        producer_context_id="producer-task",
        execution_mode="local-diagnostic",
    )
    assert packet["mechanical"]["reproduction_status"] == "REPRODUCED_LOCAL"
    assert packet["mechanical"]["execution_isolation"] == "NOT_PROVEN"
    packet_path = layout.control / "external-review" / "packet.json"
    result_path = layout.control / "external-review" / "result.json"
    atomic_write_json(packet_path, packet)
    atomic_write_json(
        result_path,
        {
            "schema": 1,
            "packet_sha256": file_hash(packet_path),
            "producer_context_id": "producer-task",
            "reviewer_context_id": "fresh-reviewer-task",
            "independent_context": True,
            "review": _supported_review(),
        },
    )
    frozen, external = validate_external_review_bundle(
        layout, contract, packet_path, result_path
    )
    engine = ModelingEngine(
        tmp_path,
        researcher=None,
        verifier=None,
        source_reviewer=None,
        model_requested="current-codex-test",
        max_attempts=2,
        max_seconds=60,
        mechanical_override=frozen["mechanical"],
        external_review=external,
    )

    result = engine.qualify(contract)

    assert result.status == "completed"
    records = RunStore(layout).evidence()
    assert len(records) == 1
    assert records[0]["authority"] == "E2"
    assert records[0]["verifier_receipt"]["transport"] == "external-codex-task"


def test_external_review_rejects_self_attested_same_context(tmp_path: Path) -> None:
    contract, layout = _operator_candidate(tmp_path)
    packet = build_external_review_packet(
        layout,
        contract,
        producer_context_id="same-task",
        execution_mode="local-diagnostic",
    )
    packet_path = layout.control / "external-review" / "packet.json"
    result_path = layout.control / "external-review" / "result.json"
    atomic_write_json(packet_path, packet)
    atomic_write_json(
        result_path,
        {
            "schema": 1,
            "packet_sha256": file_hash(packet_path),
            "producer_context_id": "same-task",
            "reviewer_context_id": "same-task",
            "independent_context": True,
            "review": _supported_review(),
        },
    )

    with pytest.raises(ValueError, match="distinct fresh context"):
        validate_external_review_bundle(layout, contract, packet_path, result_path)


def test_cli_exposes_explicit_external_review_transport() -> None:
    parser = build_parser()

    exported = parser.parse_args(
        [
            "review-export",
            "--workspace",
            "run",
            "--producer-context-id",
            "producer-task",
            "--local-replay",
        ]
    )
    imported = parser.parse_args(
        [
            "review-import",
            "--workspace",
            "run",
            "--packet",
            "packet.json",
            "--result",
            "result.json",
        ]
    )

    assert exported.command == "review-export" and exported.local_replay is True
    assert imported.command == "review-import"


def test_operator_does_not_offer_import_for_a_stale_external_result(
    tmp_path: Path,
) -> None:
    contract, layout = _operator_candidate(tmp_path)
    state = json.loads(layout.state_path.read_text(encoding="utf-8"))
    packet = build_external_review_packet(
        layout,
        contract,
        producer_context_id="producer-task",
        execution_mode="local-diagnostic",
    )
    packet_path = layout.control / "external-review" / "packet.json"
    result_path = layout.control / "external-review" / "result.json"
    atomic_write_json(packet_path, packet)
    atomic_write_json(
        result_path,
        {
            "schema": 1,
            "packet_sha256": "0" * 64,
            "producer_context_id": "producer-task",
            "reviewer_context_id": "old-reviewer-task",
            "independent_context": True,
            "review": _supported_review(),
        },
    )

    projection = _operator_projection(layout, contract, state)

    assert projection["action"] == "review-external"
    assert projection["reason"] == "external_review_result_stale"


def test_level_projection_uses_positive_facts() -> None:
    manifest = {
        "final_claim_ids": ["c1"],
        "claims": [
            {
                "id": "c1",
                "claim_type": "computational",
                "dependencies": [],
                "source_ids": [],
                "required_check_ids": ["check-c1"],
            }
        ],
    }
    mechanical = {
        "replay_ok": True,
        "checks_ok": True,
        "checks": [{"id": "check-c1", "ok": True}],
    }
    review = {
        "verdict": "SUPPORTED",
        "max_authority": "E4",
        "delivery_verdict": "SUPPORTED",
        "claim_verdicts": [
            {
                "claim_id": "c1",
                "verdict": "SUPPORTED",
                "max_authority": "E4",
                "findings": [],
            }
        ],
    }

    assert project_promotion(None)["delivery_level"] == "WORKING"
    assert project_promotion(manifest)["delivery_level"] == "CANDIDATE"
    assert project_promotion(manifest, mechanical=mechanical)["delivery_level"] == "CHECKED"
    supported = project_promotion(
        manifest,
        mechanical=mechanical,
        review=review,
    )
    assert supported["delivery_level"] == "SUPPORTED"
    assert supported["claim_levels"] == {"c1": "SUPPORTED"}
    assert "QUALIFIED" not in supported["claim_levels"].values()

    failed_mechanical = {
        **mechanical,
        "checks_ok": False,
        "checks": [{"id": "check-c1", "ok": False}],
    }
    assert project_promotion(
        manifest,
        mechanical=failed_mechanical,
        review=review,
    )["delivery_level"] == "CANDIDATE"


def test_cli_separates_research_qualification_and_scientific_success() -> None:
    parser = build_parser()
    start = parser.parse_args(
        ["start", "--workspace", "run", "--objective", "problem"]
    )
    next_action = parser.parse_args(["next", "--workspace", "run"])
    submit = parser.parse_args(
        ["submit", "--workspace", "run", "--objective", "problem"]
    )
    research = parser.parse_args(
        ["research", "--workspace", "run", "--objective", "problem"]
    )
    solve = parser.parse_args(
        ["solve", "--workspace", "run", "--objective", "problem"]
    )
    qualify = parser.parse_args(["qualify", "--workspace", "run"])

    assert start.producer_model is None
    assert next_action.command == "next"
    assert not hasattr(submit, "model_timeout")
    assert not hasattr(research, "codex_bin")
    assert solve.model_timeout == 300
    assert qualify.model == "gpt-5.6-sol"
    candidate = SimpleNamespace(
        status="candidate",
        state={"qualification": {"status": "PARTIALLY_SUPPORTED"}},
    )
    not_ready = SimpleNamespace(
        status="candidate",
        state={"qualification": {"status": "NOT_READY"}},
    )
    assert _command_exit_code("submit", candidate) == 0
    assert _command_exit_code("research", candidate) == 0
    assert _command_exit_code("solve", candidate) == 2
    assert _command_exit_code("qualify", candidate) == 0
    assert _command_exit_code("qualify", not_ready) == 2


def test_promotion_requires_declared_sources_and_weakest_dependency() -> None:
    manifest = {
        "final_claim_ids": ["child"],
        "claims": [
            {
                "id": "base",
                "claim_type": "causal",
                "dependencies": [],
                "source_ids": ["secondary"],
                "required_check_ids": ["base-check"],
            },
            {
                "id": "child",
                "claim_type": "computational",
                "dependencies": ["base"],
                "source_ids": ["missing"],
                "required_check_ids": ["child-check"],
            },
        ],
    }
    mechanical = {
        "replay_ok": True,
        "checks_ok": True,
        "contract_checks_ok": True,
        "checks": [
            {"id": "base-check", "ok": True},
            {"id": "child-check", "ok": True},
        ],
    }
    sources = {
        "secondary": {
            "authority": "E1",
            "review": {
                "verdict": "SUPPORTED",
                "source_kind": "secondary",
                "supports_claim_ids": ["base"],
            },
        }
    }

    promotion = project_promotion(
        manifest,
        mechanical=mechanical,
        source_records=sources,
    )

    assert promotion["claim_levels"] == {
        "base": "CANDIDATE",
        "child": "CANDIDATE",
    }
    assert promotion["delivery_level"] == "CANDIDATE"


def test_local_check_failure_only_demotes_linked_claim() -> None:
    manifest = {
        "final_claim_ids": ["c1", "c2"],
        "claims": [
            {
                "id": "c1",
                "claim_type": "computational",
                "dependencies": [],
                "source_ids": [],
                "required_check_ids": ["check-c1"],
            },
            {
                "id": "c2",
                "claim_type": "computational",
                "dependencies": [],
                "source_ids": [],
                "required_check_ids": ["check-c2"],
            },
        ],
    }
    mechanical = {
        "replay_ok": True,
        "checks_ok": False,
        "contract_checks_ok": True,
        "checks": [
            {"id": "check-c1", "ok": True},
            {"id": "check-c2", "ok": False},
        ],
    }

    promotion = project_promotion(manifest, mechanical=mechanical)

    assert promotion["claim_levels"] == {
        "c1": "CHECKED",
        "c2": "CANDIDATE",
    }
    assert promotion["delivery_level"] == "CANDIDATE"


def test_open_research_does_not_require_qualification(tmp_path: Path) -> None:
    researcher = _Researcher(["valid"])
    verifier = ScriptedModel([])

    result = _engine(
        tmp_path,
        researcher,
        verifier,
        max_attempts=1,
    ).research(default_contract("Deliver an open candidate.", network_mode="offline-compute"))

    assert result.status == "candidate"
    assert len(researcher.calls) == 1
    assert verifier.calls == []
    assert result.state["delivery"]["level"] == "CANDIDATE"
    assert result.state["qualification"]["status"] == "NOT_REQUESTED"
    assert result.state["delivery"]["paper"] == "work/paper/final.md"
    assert result.state["delivery"]["final_answer"]
    assert not (tmp_path / ".modeling-agent" / "evidence.jsonl").exists()


def test_open_research_preserves_paper_without_qualification_packet(
    tmp_path: Path,
) -> None:
    result = _engine(
        tmp_path,
        _Researcher(["paper-only"]),
        ScriptedModel([]),
        max_attempts=1,
    ).research(default_contract("Deliver the paper first.", network_mode="offline-compute"))

    assert result.status == "candidate"
    assert result.state["delivery"]["level"] == "CANDIDATE"
    assert result.state["delivery"]["final_answer"] is None
    assert result.state["qualification"]["status"] == "NOT_READY"
    assert "missing manifest" in " ".join(
        result.state["qualification"]["errors"]
    )
    assert (tmp_path / "work" / "paper" / "final.md").is_file()


def test_verifier_not_run_preserves_checked_delivery(tmp_path: Path) -> None:
    researcher = _Researcher(["valid"])
    verifier = ScriptedModel([])

    result = _engine(
        tmp_path,
        researcher,
        verifier,
        max_attempts=1,
    ).solve(default_contract("Keep a checked candidate.", network_mode="offline-compute"))

    assert result.status == "candidate"
    assert len(researcher.calls) == 1
    assert len(verifier.calls) == 1
    assert result.state["delivery"]["level"] == "CHECKED"
    assert result.state["qualification"]["status"] == "NOT_RUN"
    assert (tmp_path / "work" / "paper" / "final.md").is_file()
    assert result.state["delivery"]["final_answer"]
    assert not (tmp_path / ".modeling-agent" / "evidence.jsonl").exists()


def test_mixed_review_promotes_only_independently_supported_claims(
    tmp_path: Path,
) -> None:
    result = _engine(
        tmp_path,
        _Researcher(["two-claims"]),
        ScriptedModel([_two_claim_review(second_verdict="UNSUPPORTED")]),
        max_attempts=1,
    ).solve(default_contract("Promote bounded claims independently.", network_mode="offline-compute"))

    assert result.status == "candidate"
    assert result.state["qualification"]["status"] == "PARTIALLY_SUPPORTED"
    assert result.state["qualification"]["claim_levels"] == {
        "computed-value": "SUPPORTED",
        "derived-value": "CHECKED",
    }
    assert result.state["delivery"]["level"] == "CHECKED"
    evidence = RunStore(RunLayout.open(tmp_path)).evidence()
    assert {item["claim_id"] for item in evidence if item["kind"] == "claim"} == {
        "computed-value"
    }


def test_solve_revises_after_content_qualification_failure(
    tmp_path: Path,
) -> None:
    researcher = _Researcher(["two-claims", "valid"])
    verifier = ScriptedModel(
        [
            _two_claim_review(second_verdict="UNSUPPORTED"),
            _supported_review(),
        ]
    )

    result = _engine(
        tmp_path,
        researcher,
        verifier,
        max_attempts=2,
    ).solve(default_contract("Revise a weak model.", network_mode="offline-compute"))

    assert result.status == "completed"
    assert len(researcher.calls) == 2
    assert len(verifier.calls) == 2
    assert any(
        record["kind"] == "revocation"
        for record in RunStore(RunLayout.open(tmp_path)).evidence()
    )


def test_research_resume_preserves_partial_qualification(tmp_path: Path) -> None:
    contract = default_contract(
        "Preserve admitted claims.", network_mode="offline-compute"
    )
    first = _engine(
        tmp_path,
        _Researcher(["two-claims"]),
        ScriptedModel([_two_claim_review(second_verdict="UNSUPPORTED")]),
        max_attempts=1,
    ).solve(contract)
    assert first.state["qualification"]["status"] == "PARTIALLY_SUPPORTED"

    resumed = _engine(
        tmp_path,
        _Researcher([]),
        ScriptedModel([]),
        max_attempts=1,
    ).research(contract)

    assert resumed.status == "candidate"
    assert resumed.state["qualification"]["status"] == "PARTIALLY_SUPPORTED"
    assert resumed.state["evidence_count"] == 1


def test_partial_evidence_is_revoked_when_an_artifact_changes(
    tmp_path: Path,
) -> None:
    contract = default_contract(
        "Revoke stale partial evidence.", network_mode="offline-compute"
    )
    first = _engine(
        tmp_path,
        _Researcher(["two-claims"]),
        ScriptedModel([_two_claim_review(second_verdict="UNSUPPORTED")]),
        max_attempts=1,
    ).solve(contract)
    assert first.status == "candidate"
    (tmp_path / "work" / "results" / "value.json").write_text(
        '{"value":99}', encoding="utf-8"
    )
    assert run_status(tmp_path)["status"] == "stale"

    resumed = _engine(
        tmp_path,
        _Researcher([]),
        ScriptedModel([]),
        max_attempts=1,
    ).research(contract)

    assert resumed.status == "candidate"
    assert resumed.state["qualification"]["status"] == "STALE"
    assert RunStore(RunLayout.open(tmp_path)).evidence()[-1]["kind"] == "revocation"


def test_paper_only_candidate_does_not_regress_on_resume(tmp_path: Path) -> None:
    contract = default_contract(
        "Keep the paper candidate.", network_mode="offline-compute"
    )
    first = _engine(
        tmp_path,
        _Researcher(["paper-only"]),
        ScriptedModel([]),
        max_attempts=1,
    ).research(contract)
    assert first.status == "candidate"
    researcher = _Researcher([])

    resumed = _engine(
        tmp_path,
        researcher,
        ScriptedModel([]),
        max_attempts=1,
    ).research(contract)

    assert resumed.status == "candidate"
    assert resumed.state["qualification"]["status"] == "NOT_READY"
    assert researcher.calls == []


def test_direct_qualification_refreshes_candidate_metadata(tmp_path: Path) -> None:
    contract = default_contract(
        "Refresh candidate metadata.", network_mode="offline-compute"
    )
    first = _engine(
        tmp_path,
        _Researcher(["valid"]),
        ScriptedModel([]),
        max_attempts=1,
    ).research(contract)
    assert first.status == "candidate"
    paper = tmp_path / "work" / "paper" / "final.md"
    paper.write_text("# Revised candidate\n\nThe bounded result is 42.\n", encoding="utf-8")

    result = ModelingEngine(
        tmp_path,
        researcher=None,
        verifier=ScriptedModel([_supported_review()]),
        source_reviewer=ScriptedModel([]),
        model_requested="scripted",
        max_attempts=1,
        max_seconds=60,
    ).qualify(contract)

    assert result.status == "completed"
    assert result.state["delivery"]["paper_sha256"] == file_hash(paper)
    assert run_status(tmp_path)["status"] == "completed"


def test_interrupted_qualification_is_charged_and_number_is_not_reused(
    tmp_path: Path,
) -> None:
    contract = default_contract(
        "Resume an interrupted qualification.", network_mode="offline-compute"
    )
    _engine(
        tmp_path,
        _Researcher(["valid"]),
        ScriptedModel([]),
        max_attempts=1,
    ).research(contract)
    interrupted = ModelingEngine(
        tmp_path,
        researcher=None,
        verifier=_InterruptVerifier(),
        source_reviewer=ScriptedModel([]),
        model_requested="scripted",
        max_attempts=1,
        max_seconds=60,
    )
    with pytest.raises(KeyboardInterrupt):
        interrupted.qualify(contract)

    resumed = ModelingEngine(
        tmp_path,
        researcher=None,
        verifier=ScriptedModel([_supported_review()]),
        source_reviewer=ScriptedModel([]),
        model_requested="scripted",
        max_attempts=1,
        max_seconds=60,
    ).qualify(contract)

    assert resumed.status == "completed"
    assert resumed.state["qualification_attempts"] == 2
    assert (tmp_path / ".modeling-agent" / "verdicts" / "attempt-2.json").is_file()


def test_start_never_discovers_or_spawns_codex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("start must not discover or spawn a nested Codex")

    monkeypatch.setattr(model_module, "discover_codex_cli", forbidden)
    monkeypatch.setattr(model_module.subprocess, "run", forbidden)

    exit_code = main(
        [
            "start",
            "--workspace",
            str(tmp_path),
            "--objective",
            "Build and test a useful mathematical model.",
            "--network",
            "offline-compute",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["status"] == "ready"
    assert output["next"]["action"] == "research"
    assert (tmp_path / ".modeling-agent" / "run-state.json").is_file()
    assert not output.get("attempts")


@pytest.mark.parametrize(
    ("sandbox", "profile", "network_mode", "tool_free", "expected_profile"),
    [
        ("workspace-write", "modeling-workspace-only", "offline-compute", False, "modeling-workspace-only"),
        ("workspace-write", ":workspace", "research-search", False, ":workspace"),
        ("read-only", None, "source-review", True, ":read-only"),
        ("read-only", None, "delivery", True, ":read-only"),
    ],
)
def test_codex_argv_policy_matrix_is_noninteractive_and_bounded(
    tmp_path: Path,
    sandbox: str,
    profile: str | None,
    network_mode: str,
    tool_free: bool,
    expected_profile: str,
) -> None:
    workspace = (tmp_path / "work").resolve()
    argv = [
        *model_module._base_argv(
            tmp_path / "codex.exe",
            workspace=workspace,
            model="test-model",
            sandbox=sandbox,
            sandbox_profile=profile,
            network_mode=network_mode,
            tool_free=tool_free,
        ),
        "exec",
    ]

    assert argv[argv.index("--ask-for-approval") + 1] == "never"
    assert argv.index("--ask-for-approval") < argv.index("exec")
    assert 'approval_policy="never"' in argv
    if model_module.os.name == "nt":
        assert 'windows.sandbox="unelevated"' in argv
    if expected_profile == "modeling-workspace-only":
        assert f'default_permissions="{expected_profile}"' in argv
        assert "--sandbox" not in argv
    else:
        assert not any(str(item).startswith("default_permissions=") for item in argv)
        assert argv[argv.index("--sandbox") + 1] == sandbox
        assert argv.index("--sandbox") < argv.index("exec")
    assert argv[argv.index("-C") + 1] == str(workspace)
    rendered = " ".join(str(item) for item in argv)
    for forbidden in (
        "danger-full-access",
        "--dangerously-bypass-approvals-and-sandbox",
        "--yolo",
    ):
        assert forbidden not in rendered
    expected_search = network_mode in {"research-search", "source-review"}
    assert ("--search" in argv) is expected_search
    assert ('web_search="live"' in argv) is expected_search
    assert ('web_search="disabled"' in argv) is (not expected_search)


def test_base_argv_rejects_unapproved_permission_profile(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="permission profile"):
        model_module._base_argv(
            tmp_path / "codex.exe",
            workspace=tmp_path,
            model="test-model",
            sandbox="workspace-write",
            sandbox_profile="danger-full-access",
            network_mode="offline-compute",
            tool_free=False,
        )


def test_sandbox_preflight_forces_unelevated_windows_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "codex.exe"
    executable.write_bytes(b"")
    workspace = tmp_path / "work"
    workspace.mkdir()
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(argv)
        target = Path(argv[-1])
        if target.parent == workspace:
            return SimpleNamespace(
                returncode=0,
                stdout=target.read_text(encoding="utf-8"),
                stderr="",
            )
        return SimpleNamespace(returncode=1, stdout="", stderr="denied")

    monkeypatch.setattr(model_module.subprocess, "run", fake_run)

    model_module.ensure_workspace_only_sandbox(
        executable,
        workspace,
        timeout_seconds=30,
    )

    assert len(calls) == 2
    if model_module.os.name == "nt":
        assert all('windows.sandbox="unelevated"' in call for call in calls)


def test_trace_counts_interaction_request_as_contract_violation(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        '{"type":"item.started","item":{"id":"p1","type":"request_permissions"}}\n',
        encoding="utf-8",
    )

    assert model_module._trace_counts(trace)["observable_interaction_requests"] == 1


def test_submit_with_incomplete_packet_never_discovers_or_spawns_codex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("incomplete submission must return to the current Codex")

    monkeypatch.setattr(model_module, "discover_codex_cli", forbidden)
    monkeypatch.setattr(model_module.subprocess, "run", forbidden)

    exit_code = main(
        [
            "submit",
            "--workspace",
            str(tmp_path),
            "--objective",
            "Build and test a useful mathematical model.",
            "--network",
            "offline-compute",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert output["status"] == "ready"
    assert output["next"]["action"] == "research"
    state = json.loads(
        (tmp_path / ".modeling-agent" / "run-state.json").read_text(encoding="utf-8")
    )
    assert state["attempts"] == []

    solve_exit = main(
        [
            "solve",
            "--workspace",
            str(tmp_path),
            "--objective",
            "Build and test a useful mathematical model.",
            "--network",
            "offline-compute",
        ]
    )
    solve_output = json.loads(capsys.readouterr().out)
    assert solve_exit == 2
    assert solve_output["next"]["action"] == "research"

    qualify_exit = main(["qualify", "--workspace", str(tmp_path)])
    qualify_output = json.loads(capsys.readouterr().out)
    assert qualify_exit == 2
    assert qualify_output["summary"]["qualification"]["status"] == "NOT_READY"


def test_submit_admits_paper_candidate_without_spawning_codex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("candidate admission must not spawn a nested Codex")

    monkeypatch.setattr(model_module, "discover_codex_cli", forbidden)
    monkeypatch.setattr(model_module.subprocess, "run", forbidden)
    objective = "Build and test a useful mathematical model."
    assert main(
        [
            "start",
            "--workspace",
            str(tmp_path),
            "--objective",
            objective,
            "--network",
            "offline-compute",
        ]
    ) == 0
    capsys.readouterr()
    paper = tmp_path / "work" / "paper" / "final.md"
    paper.parent.mkdir(parents=True, exist_ok=True)
    paper.write_text("# Candidate\n\nA bounded modeling result.\n", encoding="utf-8")

    exit_code = main(
        [
            "submit",
            "--workspace",
            str(tmp_path),
            "--objective",
            objective,
            "--network",
            "offline-compute",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["status"] == "candidate"
    assert output["reason"] == "operator_candidate_qualification_not_ready"
    assert output["next"]["action"] == "repair"
    state = json.loads(
        (tmp_path / ".modeling-agent" / "run-state.json").read_text(encoding="utf-8")
    )
    assert state["attempts"] == []
    assert state["qualification"]["status"] == "NOT_READY"


@pytest.mark.parametrize(
    "receipt_override",
    [
        {"approval_policy": "on-request"},
        {"sandbox_profile": "danger-full-access"},
        {"workspace": "../outside"},
        {"windows_sandbox": "elevated"},
        {"interactive": True},
        {"observable_interaction_requests": 1},
    ],
)
def test_interactive_or_unbounded_verifier_receipt_cannot_admit(
    tmp_path: Path, receipt_override: dict[str, object]
) -> None:
    verifier = ScriptedModel(
        [_supported_review()], receipts=[receipt_override]
    )

    result = _engine(
        tmp_path,
        _Researcher(["valid"]),
        verifier,
        max_attempts=1,
    ).solve(default_contract("Reject an untrusted verifier receipt.", network_mode="offline-compute"))

    assert result.status == "candidate"
    assert result.state["qualification"]["status"] == "NOT_RUN"
    assert not (tmp_path / ".modeling-agent" / "evidence.jsonl").exists()


def test_policy_denied_replay_is_not_run_not_refuted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = ScriptedModel([_supported_review()])

    def deny_execute(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "status": "denied",
            "message": "Python policy denied the script",
            "error_type": "permission_denied",
            "data": {"findings": ["network import is forbidden"]},
        }

    monkeypatch.setattr(
        verification_module.ToolRegistry,
        "execute",
        deny_execute,
    )

    result = _engine(
        tmp_path,
        _Researcher(["valid"]),
        verifier,
        max_attempts=1,
    ).solve(default_contract("Do not call a denied replay a refutation.", network_mode="offline-compute"))

    assert result.status == "candidate"
    assert result.state["qualification"]["status"] == "NOT_RUN"
    assert verifier.calls == []
    assert not (tmp_path / ".modeling-agent" / "evidence.jsonl").exists()


def test_single_engine_completes_with_control_work_separation(tmp_path: Path) -> None:
    researcher = _Researcher(["valid"])
    result = _engine(
        tmp_path, researcher, ScriptedModel([_supported_review()]), max_attempts=1
    ).solve(default_contract("Compute a reproducible value.", network_mode="offline-compute"))

    assert result.status == "completed"
    assert researcher.calls[0]["workspace"] == tmp_path / "work"
    assert not (tmp_path / "work" / ".modeling-agent").exists()
    assert (tmp_path / ".modeling-agent" / "evidence.jsonl").is_file()
    assert result.state["delivery"]["claim_ceiling"] == "E4"
    assert run_status(tmp_path)["evidence_integrity"] == {"computed-value": "current"}


def test_mechanical_failure_vetoes_fresh_verifier(tmp_path: Path) -> None:
    verifier = ScriptedModel([])
    result = _engine(tmp_path, _Researcher(["bad-check"]), verifier, max_attempts=1).solve(
        default_contract("Reject a failed check.", network_mode="offline-compute")
    )

    assert result.status == "candidate"
    assert result.state["qualification"]["status"] == "UNSUPPORTED"
    assert result.state["delivery"]["level"] == "CANDIDATE"
    assert verifier.calls == []


def test_completed_artifact_mutation_is_reported_stale_and_revoked_on_resume(
    tmp_path: Path,
) -> None:
    contract = default_contract("Detect stale evidence.", network_mode="offline-compute")
    first = _engine(
        tmp_path, _Researcher(["valid"]), ScriptedModel([_supported_review()]), max_attempts=1
    ).solve(contract)
    assert first.status == "completed"
    (tmp_path / "work" / "results" / "value.json").write_text('{"value":99}', encoding="utf-8")
    assert run_status(tmp_path)["status"] == "stale"

    resumed = _engine(
        tmp_path,
        _Researcher(["valid"]),
        ScriptedModel([_supported_review()]),
        max_attempts=2,
        allow_budget_amendment=True,
    ).solve(contract)
    assert resumed.status == "completed"
    evidence = (tmp_path / ".modeling-agent" / "evidence.jsonl").read_text(encoding="utf-8")
    assert '"kind":"revocation"' in evidence


def test_legacy_completed_state_projects_supported_level(tmp_path: Path) -> None:
    contract = default_contract(
        "Project a legacy completed run.", network_mode="offline-compute"
    )
    first = _engine(
        tmp_path,
        _Researcher(["valid"]),
        ScriptedModel([_supported_review()]),
        max_attempts=1,
    ).solve(contract)
    assert first.status == "completed"
    layout = RunLayout.open(tmp_path)
    state = json.loads(layout.state_path.read_text(encoding="utf-8"))
    state.pop("qualification", None)
    state["delivery"].pop("level", None)
    atomic_write_json(layout.state_path, state)

    status = run_status(tmp_path)
    resumed = _engine(
        tmp_path,
        _Researcher([]),
        ScriptedModel([]),
        max_attempts=1,
    ).solve(contract)

    assert status["promotion_level"] == "SUPPORTED"
    assert status["qualification"]["status"] == "SUPPORTED"
    assert resumed.state["qualification"]["status"] == "SUPPORTED"


def test_control_plane_tampering_is_detected_before_review(tmp_path: Path) -> None:
    verifier = ScriptedModel([])
    result = _engine(
        tmp_path, _Researcher(["mutate-control"]), verifier, max_attempts=1
    ).solve(default_contract("Protect authority files.", network_mode="offline-compute"))

    assert result.status == "stopped"
    errors = result.state["attempts"][0]["verdict"]["errors"]
    assert "researcher modified the harness-owned control plane" in errors
    assert verifier.calls == []
    restored = json.loads((tmp_path / ".modeling-agent" / "task-contract.json").read_text(encoding="utf-8"))
    assert restored["objective"] == "Protect authority files."


def test_on_demand_branch_publishes_working_knowledge_for_next_attempt(
    tmp_path: Path,
) -> None:
    researcher = _Researcher(["branch-request", "valid"])
    engine = _engine(
        tmp_path,
        researcher,
        ScriptedModel([_supported_review()]),
        branch_researcher_factory=_BranchResearcher,
    )
    result = engine.solve(default_contract("Challenge a proposed route.", network_mode="offline-compute"))

    assert result.status == "completed"
    assert result.state["attempts"][0]["status"] == "branched"
    assert "unsupported assumption" in researcher.calls[1]["prompt"]
    records = ResearchStore(RunLayout.open(tmp_path).research_path).records()
    assert any(item["kind"] == "branch_summary" for item in records)


def test_source_gate_requires_exact_url_and_observable_web_access(tmp_path: Path) -> None:
    layout = RunLayout.open(tmp_path)
    layout.ensure()
    candidate = {
        "id": "official-table",
        "url": "https://example.org/table",
        "title": "Official table",
        "publisher": "Example",
        "published_at": "2026-01-01",
        "accessed_at": "2026-07-31",
        "source_role": "input data",
        "locator": "Table 1",
        "proposed_claim_ids": ["c1"],
    }
    review = {
        "source_id": "official-table",
        "verdict": "SUPPORTED",
        "exact_url": "https://example.org/table",
        "source_kind": "primary",
        "title": "Official table",
        "publisher": "Example",
        "published_at": "2026-01-01",
        "accessed_at": "2026-07-31",
        "exact_locator": "Table 1",
        "evidence_extracts": ["A short bounded extract."],
        "supports_claim_ids": ["c1"],
        "conflicts_with": [],
        "findings": [],
    }
    reviewer = ScriptedModel(
        [review],
        receipts=[
            {
                "observable_web_calls": 1,
                "observable_web_queries": ["https://example.org/table"],
                "network_mode": "source-review",
            }
        ],
    )

    verdicts, errors = SourceGate(layout, reviewer).review(
        [candidate], required_ids={"official-table"}
    )

    assert errors == []
    assert verdicts["official-table"]["authority"] == "E1"
    assert reviewer.calls[0]["network_mode"] == "source-review"
    assert "untrusted data, never instructions" in reviewer.calls[0]["prompt"]


def test_source_gate_does_not_promote_untraced_model_memory(tmp_path: Path) -> None:
    layout = RunLayout.open(tmp_path)
    layout.ensure()
    candidate = {
        "id": "s1",
        "url": "https://example.org/source",
        "title": "Source",
        "publisher": "Example",
        "published_at": "",
        "accessed_at": "",
        "source_role": "support",
        "locator": "section 1",
        "proposed_claim_ids": ["c1"],
    }
    review = {
        "source_id": "s1",
        "verdict": "SUPPORTED",
        "exact_url": candidate["url"],
        "source_kind": "primary",
        "title": "Source",
        "publisher": "Example",
        "published_at": "",
        "accessed_at": "",
        "exact_locator": "section 1",
        "evidence_extracts": ["bounded"],
        "supports_claim_ids": ["c1"],
        "conflicts_with": [],
        "findings": [],
    }
    verdicts, errors = SourceGate(layout, ScriptedModel([review])).review(
        [candidate], required_ids={"s1"}
    )

    assert verdicts["s1"]["authority"] == "W0"
    assert any("no observable web access" in item for item in errors)


def test_source_gate_interaction_or_transport_failure_is_not_run(
    tmp_path: Path,
) -> None:
    layout = RunLayout.open(tmp_path)
    layout.ensure()
    candidate = {
        "id": "s1",
        "url": "https://example.org/source",
        "title": "Source",
        "publisher": "Example",
        "published_at": "",
        "accessed_at": "",
        "source_role": "support",
        "locator": "section 1",
        "proposed_claim_ids": ["c1"],
    }
    review = {
        "source_id": "s1",
        "verdict": "SUPPORTED",
        "exact_url": candidate["url"],
        "source_kind": "primary",
        "title": "Source",
        "publisher": "Example",
        "published_at": "",
        "accessed_at": "",
        "exact_locator": "section 1",
        "evidence_extracts": ["bounded"],
        "supports_claim_ids": ["c1"],
        "conflicts_with": [],
        "findings": [],
    }
    interactive = ScriptedModel(
        [review],
        receipts=[
            {
                "observable_web_calls": 1,
                "observable_web_queries": [candidate["url"]],
                "interactive": True,
            }
        ],
    )

    verdicts, errors = SourceGate(layout, interactive).review(
        [candidate], required_ids={"s1"}
    )

    assert verdicts["s1"]["authority"] == "W0"
    assert any(item.startswith(SOURCE_GATE_NOT_RUN_PREFIX) for item in errors)

    missing_response = ScriptedModel([])
    verdicts, errors = SourceGate(layout, missing_response).review(
        [candidate], required_ids={"s1"}, review_tag=2
    )

    assert verdicts == {}
    assert any(item.startswith(SOURCE_GATE_NOT_RUN_PREFIX) for item in errors)


def test_predictive_claim_contract_requires_baseline_and_falsifier(tmp_path: Path) -> None:
    contract = validate_contract(
        default_contract("Validate a prediction.", network_mode="offline-compute")
    )
    layout = RunLayout.open(tmp_path)
    layout.ensure()
    atomic_write_json(layout.work_contract_path, contract)
    _Researcher(["valid"]).run(
        "",
        role="fixture",
        workspace=layout.work,
        trace_path=tmp_path / "trace.jsonl",
        timeout_seconds=10,
    )
    manifest_path = tmp_path / "work" / "submission_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["claims"][0]["claim_type"] = "predictive"
    manifest["claims"][0]["baseline"] = ""
    manifest["claims"][0]["falsifiers"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _, errors = validate_manifest(tmp_path / "work", contract, content_hash(contract))

    assert "claim computed-value: predictive claim requires a baseline" in errors
    assert "claim computed-value: predictive claim requires falsifiers or stress tests" in errors


def test_workspace_escape_and_generated_network_code_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes workspace"):
        safe_path(tmp_path, "../outside.txt")

    tools = ToolRegistry(tmp_path)
    assert tools.execute(
        "write_text",
        {"path": "src/network.py", "content": "import requests\n"},
    )["status"] == "success"
    result = tools.execute(
        "python_compute",
        {
            "script": "src/network.py",
            "args": [],
            "timeout": 10,
            "expected_outputs": [],
        },
    )
    assert result["status"] == "denied"
    assert result["error_type"] == "permission_denied"


def test_ablation_freezes_new_component_arms_and_is_append_only(tmp_path: Path) -> None:
    path = tmp_path / "ablation.json"
    manifest = freeze_ablation(
        path,
        objective="A private unseen modeling task.",
        model="gpt-5.6-sol",
        max_model_turns=8,
        max_tool_calls=24,
        max_wall_seconds=900,
    )
    assert [item["id"] for item in manifest["arms"]] == [
        "raw_codex",
        "codex_web",
        "source_gate",
        "hard_eval",
        "elastic_memory",
    ]
    record_result(path, "hard_eval", {"status": "stopped"})
    with pytest.raises(ValueError, match="already recorded"):
        record_result(path, "hard_eval", {"status": "completed"})


def test_partial_review_preserves_delivery_but_never_reports_success(tmp_path: Path) -> None:
    review = {
        "verdict": "PARTIALLY_SUPPORTED",
        "claim_verdicts": [
            {
                "claim_id": "computed-value",
                "verdict": "PARTIALLY_SUPPORTED",
                "max_authority": "E2",
                "findings": ["scope is not justified"],
            }
        ],
        "findings": ["narrow the scope"],
        "max_authority": "E2",
        "delivery_verdict": "PARTIALLY_SUPPORTED",
        "delivery_findings": ["scope is not justified"],
    }
    result = _engine(
        tmp_path, _Researcher(["valid"]), ScriptedModel([review]), max_attempts=1
    ).solve(default_contract("Preserve an unverified delivery.", network_mode="offline-compute"))

    assert result.status == "candidate"
    assert result.state["delivery"]["status"] == "candidate"
    assert result.state["delivery"]["level"] == "CHECKED"
    assert not (tmp_path / ".modeling-agent" / "evidence.jsonl").exists()


def test_branch_failure_is_preserved_without_destroying_main_route(tmp_path: Path) -> None:
    researcher = _Researcher(["branch-request", "valid"])
    result = _engine(
        tmp_path,
        researcher,
        ScriptedModel([_supported_review()]),
        branch_researcher_factory=_FailingBranchResearcher,
    ).solve(default_contract("Survive one failed branch.", network_mode="offline-compute"))

    assert result.status == "completed"
    branch = result.state["waves"][0]["branches"][0]
    assert branch["status"] == "failed"
    assert "branch fixture failed" in branch["error"]


def test_cumulative_wall_budget_cannot_be_reset_by_resume(tmp_path: Path) -> None:
    contract = default_contract("Respect cumulative time.", network_mode="offline-compute")
    first = _engine(
        tmp_path,
        _Researcher(["bad-check"]),
        ScriptedModel([]),
        max_attempts=1,
    ).solve(contract)
    assert first.status == "candidate"
    layout = RunLayout.open(tmp_path)
    state = json.loads(layout.state_path.read_text(encoding="utf-8"))
    state["elapsed_seconds"] = state["budgets"]["max_seconds"]
    atomic_write_json(layout.state_path, state)
    researcher = _Researcher(["valid"])

    resumed = _engine(
        tmp_path,
        researcher,
        ScriptedModel([_supported_review()]),
        max_attempts=2,
        allow_budget_amendment=True,
    ).solve(contract)

    assert resumed.status == "candidate"
    assert resumed.state["qualification"]["status"] == "UNSUPPORTED"
    assert researcher.calls == []


def test_source_record_hash_participates_in_completed_claim_integrity(
    tmp_path: Path,
) -> None:
    layout = RunLayout.open(tmp_path)
    store = RunStore(layout)
    contract = default_contract("Synthetic integrity fixture.", network_mode="offline-compute")
    atomic_write_json(layout.contract_path, contract)
    source = layout.sources / "official" / "review-1.json"
    atomic_write_json(source, {"verdict": "SUPPORTED"})
    store.save(
        {
            "schema": 1,
            "run_id": "run",
            "status": "completed",
            "contract_hash": content_hash(contract),
            "attempts": [],
            "waves": [],
            "delivery": {"status": "verified"},
        }
    )
    store.admit(
        {
            "schema": 1,
            "kind": "claim",
            "claim_id": "c1",
            "artifact_records": [],
            "source_records": [
                {
                    "source_id": "official",
                    "path": "sources/official/review-1.json",
                    "sha256": file_hash(source),
                    "snapshot_hash": "snapshot",
                }
            ],
        }
    )
    store.admit(
        {
            "schema": 1,
            "kind": "claim",
            "claim_id": "c2",
            "dependencies": ["c1"],
            "artifact_records": [],
            "source_records": [],
        }
    )
    assert run_status(tmp_path)["status"] == "completed"
    atomic_write_json(source, {"verdict": "CONFLICTING"})
    status = run_status(tmp_path)
    assert status["status"] == "stale"
    assert status["evidence_integrity"] == {"c1": "stale", "c2": "stale"}


def test_unsupported_noncritical_claim_does_not_block_supported_claim(
    tmp_path: Path,
) -> None:
    result = _engine(
        tmp_path,
        _Researcher(["two-claims"]),
        ScriptedModel([_two_claim_review(second_verdict="UNSUPPORTED")]),
        max_attempts=1,
    ).solve(default_contract("Reject one unsupported claim.", network_mode="offline-compute"))

    assert result.status == "candidate"
    assert result.state["qualification"]["status"] == "PARTIALLY_SUPPORTED"
    records = RunStore(RunLayout.open(tmp_path)).evidence()
    assert {record["claim_id"] for record in records} == {"computed-value"}


def test_overall_and_dependency_authority_cap_claims(tmp_path: Path) -> None:
    result = _engine(
        tmp_path,
        _Researcher(["dependent-claims"]),
        ScriptedModel(
            [
                _two_claim_review(
                    first_authority="E2",
                    second_authority="E4",
                    overall_authority="E4",
                )
            ]
        ),
        max_attempts=1,
    ).solve(default_contract("Propagate evidence authority.", network_mode="offline-compute"))

    assert result.status == "completed"
    records = RunStore(RunLayout.open(tmp_path)).evidence()
    authority = {item["claim_id"]: item["authority"] for item in records}
    assert authority == {"computed-value": "E2", "derived-value": "E2"}
    assert result.state["delivery"]["claim_ceiling"] == "E2"


def test_working_only_overall_review_cannot_complete(tmp_path: Path) -> None:
    review = _supported_review()
    review["max_authority"] = "W0"
    result = _engine(
        tmp_path,
        _Researcher(["valid"]),
        ScriptedModel([review]),
        max_attempts=1,
    ).solve(default_contract("Do not promote W0.", network_mode="offline-compute"))

    assert result.status == "candidate"
    assert any(
        "working-only" in item
        for item in result.state["qualification"]["errors"]
    )


def test_claim_dependency_cycle_is_rejected_before_review(tmp_path: Path) -> None:
    contract = validate_contract(default_contract("Reject cycles.", network_mode="offline-compute"))
    layout = RunLayout.open(tmp_path)
    layout.ensure()
    atomic_write_json(layout.work_contract_path, contract)
    _Researcher(["cyclic-claim"]).run(
        "",
        role="fixture",
        workspace=layout.work,
        trace_path=tmp_path / "trace.jsonl",
        timeout_seconds=10,
    )

    _, errors = validate_manifest(layout.work, contract, content_hash(contract))

    assert any("dependency graph contains a cycle" in item for item in errors)


def test_replay_cannot_read_undeclared_generator_input(tmp_path: Path) -> None:
    verifier = ScriptedModel([])
    result = _engine(
        tmp_path,
        _Researcher(["undeclared-input"]),
        verifier,
        max_attempts=1,
    ).solve(default_contract("Enforce replay inputs.", network_mode="offline-compute"))

    assert result.status == "candidate"
    assert verifier.calls == []
    assert "did not reproduce" in " ".join(
        result.state["qualification"]["errors"]
    )


@pytest.mark.parametrize(
    "relative",
    [
        "results/value.json",
        "paper/final.md",
        "src/solve.py",
        "checks/check_value.py",
        "submission_manifest.json",
    ],
)
def test_any_reviewed_provenance_mutation_stales_delivery(
    tmp_path: Path, relative: str
) -> None:
    contract = default_contract("Bind all reviewed provenance.", network_mode="offline-compute")
    result = _engine(
        tmp_path,
        _Researcher(["valid"]),
        ScriptedModel([_supported_review()]),
        max_attempts=1,
    ).solve(contract)
    assert result.status == "completed"
    path = tmp_path / "work" / relative
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    assert run_status(tmp_path)["status"] == "stale"


def test_evidence_edit_and_deletion_are_not_reported_completed(tmp_path: Path) -> None:
    contract = default_contract("Detect evidence tampering.", network_mode="offline-compute")
    result = _engine(
        tmp_path,
        _Researcher(["valid"]),
        ScriptedModel([_supported_review()]),
        max_attempts=1,
    ).solve(contract)
    assert result.status == "completed"
    evidence_path = tmp_path / ".modeling-agent" / "evidence.jsonl"
    original = evidence_path.read_text(encoding="utf-8")
    record = json.loads(original)
    record["statement"] = "tampered"
    evidence_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert run_status(tmp_path)["status"] == "corrupt"

    evidence_path.unlink()
    assert run_status(tmp_path)["status"] == "corrupt"


def test_delivery_requires_its_own_supported_review(tmp_path: Path) -> None:
    review = _supported_review()
    review["delivery_verdict"] = "UNSUPPORTED"
    review["delivery_findings"] = ["paper introduces an unsupported conclusion"]
    result = _engine(
        tmp_path,
        _Researcher(["valid"]),
        ScriptedModel([review]),
        max_attempts=1,
    ).solve(default_contract("Review the final delivery.", network_mode="offline-compute"))

    assert result.status == "candidate"
    assert result.state["delivery"]["level"] == "CHECKED"
    records = RunStore(RunLayout.open(tmp_path)).evidence()
    assert len(records) == 1
    assert records[0]["final_answer"] is None


def test_source_support_must_name_the_proposed_claim(tmp_path: Path) -> None:
    layout = RunLayout.open(tmp_path)
    layout.ensure()
    candidate = {
        "id": "official",
        "url": "https://example.org/source",
        "title": "Official",
        "publisher": "Example",
        "published_at": "",
        "accessed_at": "",
        "source_role": "support",
        "locator": "section 1",
        "proposed_claim_ids": ["c1"],
    }
    review = {
        "source_id": "official",
        "verdict": "SUPPORTED",
        "exact_url": candidate["url"],
        "source_kind": "primary",
        "title": "Official",
        "publisher": "Example",
        "published_at": "",
        "accessed_at": "",
        "exact_locator": "section 1",
        "evidence_extracts": ["bounded"],
        "supports_claim_ids": ["different-claim"],
        "conflicts_with": [],
        "findings": [],
    }
    reviewer = ScriptedModel(
        [review],
        receipts=[
            {
                "observable_web_calls": 1,
                "observable_web_queries": [candidate["url"]],
            }
        ],
    )

    verdicts, errors = SourceGate(layout, reviewer).review(
        [candidate], required_ids={"official"}
    )

    assert verdicts["official"]["authority"] == "W0"
    assert any("not proposed" in item or "every proposed claim" in item for item in errors)


def test_deadline_exhaustion_after_review_cannot_admit(tmp_path: Path) -> None:
    result = _engine(
        tmp_path,
        _SlowResearcher(["valid"], 0.55),
        _SlowVerifier([_supported_review()], 0.55),
        max_attempts=1,
        max_seconds=1,
    ).solve(default_contract("Enforce the final deadline.", network_mode="offline-compute"))

    assert result.status == "candidate"
    assert result.state["qualification"]["status"] == "NOT_RUN"
    assert not (tmp_path / ".modeling-agent" / "evidence.jsonl").exists()


def test_budget_and_model_provenance_are_frozen_on_resume(tmp_path: Path) -> None:
    contract = default_contract("Freeze runtime provenance.", network_mode="offline-compute")
    first = _engine(
        tmp_path,
        _Researcher(["bad-check"]),
        ScriptedModel([]),
        max_attempts=1,
        model_requested="model-a",
    ).solve(contract)
    assert first.status == "candidate"

    with pytest.raises(ValueError, match="budgets are frozen"):
        _engine(
            tmp_path,
            _Researcher(["valid"]),
            ScriptedModel([_supported_review()]),
            max_attempts=2,
            model_requested="model-a",
        ).solve(contract)
    with pytest.raises(ValueError, match="model_requested"):
        _engine(
            tmp_path,
            _Researcher(["valid"]),
            ScriptedModel([_supported_review()]),
            max_attempts=1,
            model_requested="model-b",
        ).solve(contract)


def test_failed_requalification_downgrades_old_delivery(tmp_path: Path) -> None:
    contract = default_contract("Revoke stale delivery.", network_mode="offline-compute")
    first = _engine(
        tmp_path,
        _Researcher(["valid"]),
        ScriptedModel([_supported_review()]),
        max_attempts=1,
    ).solve(contract)
    assert first.status == "completed"
    (tmp_path / "work" / "results" / "value.json").write_text(
        '{"value":99}', encoding="utf-8"
    )

    resumed = _engine(
        tmp_path,
        _Researcher([]),
        ScriptedModel([]),
        max_attempts=1,
    ).solve(contract)

    assert resumed.status == "candidate"
    assert resumed.state["delivery"]["status"] == "candidate"
    assert resumed.state["qualification"]["status"] == "STALE"
    assert resumed.state["delivery"]["claim_ceiling"] == "W0"


def test_control_snapshot_is_restored_when_researcher_raises(tmp_path: Path) -> None:
    contract = default_contract("Restore failed tampering.", network_mode="offline-compute")
    result = ModelingEngine(
        tmp_path,
        researcher=_TamperFailResearcher(),
        verifier=ScriptedModel([]),
        source_reviewer=ScriptedModel([]),
        model_requested="scripted",
        max_attempts=1,
        max_seconds=60,
    ).solve(contract)

    assert result.status == "stopped"
    assert result.state["attempts"][0]["control_tamper_restored"] is True
    restored = json.loads(
        (tmp_path / ".modeling-agent" / "task-contract.json").read_text(encoding="utf-8")
    )
    assert restored["objective"] == "Restore failed tampering."


def test_run_lock_rejects_a_second_writer(tmp_path: Path) -> None:
    layout = RunLayout.open(tmp_path)
    with run_lock(layout):
        with pytest.raises(RuntimeError, match="already owns"):
            with run_lock(layout):
                pass


def test_invalid_working_memory_does_not_hide_verified_status(tmp_path: Path) -> None:
    contract = default_contract("Keep W0 non-authoritative.", network_mode="offline-compute")
    result = _engine(
        tmp_path,
        _Researcher(["valid"]),
        ScriptedModel([_supported_review()]),
        max_attempts=1,
    ).solve(contract)
    assert result.status == "completed"
    research = tmp_path / "work" / "research" / "records.jsonl"
    research.write_text('{"kind":"unknown","statement":"bad"}\n', encoding="utf-8")

    status = run_status(tmp_path)

    assert status["status"] == "completed"
    assert status["research_graph"]["incomplete"] is True
    assert status["research_errors"]


@pytest.mark.parametrize(
    ("mode", "expected_error"),
    [
        ("self-seeded-output", "cannot preload its expected outputs"),
        ("ungenerated-claim-artifact", "without generator provenance"),
    ],
)
def test_invalid_generator_provenance_blocks_qualification_not_delivery(
    tmp_path: Path, mode: str, expected_error: str
) -> None:
    result = _engine(
        tmp_path,
        _Researcher([mode]),
        ScriptedModel([]),
        max_attempts=1,
    ).solve(default_contract("Require genuine replay provenance.", network_mode="offline-compute"))

    assert result.status == "candidate"
    assert result.state["qualification"]["status"] == "NOT_READY"
    assert any(
        expected_error in item
        for item in result.state["qualification"]["errors"]
    )
    assert not (tmp_path / ".modeling-agent" / "evidence.jsonl").exists()


def test_fresh_review_fails_closed_when_paper_tail_is_omitted(tmp_path: Path) -> None:
    verifier = ScriptedModel([])
    result = _engine(
        tmp_path,
        _Researcher(["long-paper"]),
        verifier,
        max_attempts=1,
    ).solve(default_contract("Review the complete paper.", network_mode="offline-compute"))

    assert result.status == "candidate"
    assert verifier.calls == []
    assert any(
        "cannot contain complete text" in item
        for item in result.state["qualification"]["errors"]
    )


def test_review_packet_exact_boundary_does_not_hide_later_code(tmp_path: Path) -> None:
    verifier = ScriptedModel([])
    result = _engine(
        tmp_path,
        _Researcher(["exact-limit-paper"]),
        verifier,
        max_attempts=1,
    ).solve(default_contract("Review every selected text artifact.", network_mode="offline-compute"))

    assert result.status == "candidate"
    assert verifier.calls == []
    assert any(
        "cannot contain complete text" in item
        for item in result.state["qualification"]["errors"]
    )


def test_custom_delivery_artifact_is_reviewed_and_projected(tmp_path: Path) -> None:
    contract = default_contract("Review the configured delivery.", network_mode="offline-compute")
    contract["delivery_artifact"] = "deliverable/report.md"
    contract["required_artifacts"] = []
    verifier = ScriptedModel([_supported_review()])

    result = _engine(
        tmp_path,
        _Researcher(["paper-tail"]),
        verifier,
        max_attempts=1,
    ).solve(contract)

    assert result.status == "completed"
    assert "UNIQUE_PAPER_TAIL_CLAIM" in verifier.calls[0]["prompt"]
    assert result.state["delivery"]["paper"] == "work/deliverable/report.md"


def test_malformed_manifest_is_structurally_rejected(tmp_path: Path) -> None:
    verifier = ScriptedModel([])
    result = _engine(
        tmp_path,
        _Researcher(["malformed-generator-list"]),
        verifier,
        max_attempts=1,
    ).solve(default_contract("Reject malformed arrays.", network_mode="offline-compute"))

    assert result.status == "candidate"
    assert result.state["qualification"]["status"] == "NOT_READY"
    assert verifier.calls == []
    assert any(
        "input_paths must be a string array" in item
        for item in result.state["qualification"]["errors"]
    )


def test_interrupted_attempt_is_charged_and_consumed_on_resume(tmp_path: Path) -> None:
    contract = default_contract("Persist interrupted work.", network_mode="offline-compute")
    engine = ModelingEngine(
        tmp_path,
        researcher=_InterruptResearcher(),
        verifier=ScriptedModel([]),
        source_reviewer=ScriptedModel([]),
        model_requested="scripted",
        max_attempts=1,
        max_seconds=60,
    )

    with pytest.raises(KeyboardInterrupt):
        engine.solve(contract)

    resumed = _engine(
        tmp_path,
        _Researcher([]),
        ScriptedModel([]),
        max_attempts=1,
        max_seconds=60,
    ).solve(contract)

    assert resumed.status == "stopped"
    assert resumed.state["attempts"][0]["status"] == "interrupted"
    assert resumed.state["attempts"][0]["charged_seconds"] >= 0


@pytest.mark.parametrize("target", ["contract", "verifier-trace"])
def test_control_provenance_mutation_invalidates_status(
    tmp_path: Path, target: str
) -> None:
    result = _engine(
        tmp_path,
        _Researcher(["valid"]),
        ScriptedModel([_supported_review()]),
        max_attempts=1,
    ).solve(default_contract("Bind control provenance.", network_mode="offline-compute"))
    assert result.status == "completed"

    if target == "contract":
        (tmp_path / ".modeling-agent" / "task-contract.json").write_text(
            '{"schema":2,"objective":"tampered"}', encoding="utf-8"
        )
        assert run_status(tmp_path)["status"] == "corrupt"
    else:
        (tmp_path / ".modeling-agent" / "traces" / "verifier-1.jsonl").unlink()
        assert run_status(tmp_path)["status"] == "stale"
