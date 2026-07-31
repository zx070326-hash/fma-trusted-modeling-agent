from __future__ import annotations

import json
from pathlib import Path

import pytest

from modeling_agent.ablation import freeze_ablation, record_result
from modeling_agent.core import (
    StateStore,
    content_hash,
    delivery_projection,
    file_hash,
    new_state,
    node_states,
    state_planes,
    upsert_nodes,
)
from modeling_agent.loop import ModelingLoop
from modeling_agent.model import ACTION_SCHEMA, ScriptedModel
from modeling_agent.sidecar import (
    NativeSidecar,
    default_contract,
    native_status,
    validate_contract,
)
from modeling_agent.tools import ToolRegistry, run_check


def _action(
    *,
    summary: str = "work",
    upsert_nodes: list[dict[str, object]] | None = None,
    tool_calls: list[dict[str, object]] | None = None,
    candidate_claims: list[dict[str, object]] | None = None,
    final: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "summary": summary,
        "upsert_nodes": upsert_nodes or [],
        "focus_node": "root",
        "tool_calls": tool_calls or [],
        "candidate_claims": candidate_claims or [],
        "final": final,
    }


def _approve() -> dict[str, object]:
    return {
        "verdict": "APPROVE",
        "findings": [],
        "claim_strength": "locally_supported",
    }


def _write_json(value: int) -> dict[str, object]:
    return {
        "call_id": "write-result",
        "name": "write_text",
        "arguments_json": json.dumps(
            {"path": "artifacts/result.json", "content": json.dumps({"value": value})}
        ),
    }


def _claim(value: int) -> dict[str, object]:
    return {
        "id": "e-result",
        "node_id": "root",
        "statement": f"The computed value is {value}.",
        "artifact": "artifacts/result.json",
        "supporting_artifacts": [],
        "checks": [
            {
                "kind": "numeric_assertion",
                "arguments_json": json.dumps(
                    {
                        "path": "artifacts/result.json",
                        "field": "value",
                        "operator": "==",
                        "value": value,
                        "tolerance": 0,
                    }
                ),
            }
        ],
    }


def test_problem_graph_is_open_and_revisions_revoke_downstream() -> None:
    state = new_state(
        "Estimate a coupled nonlinear system.",
        max_steps=4,
        max_tool_calls=8,
        max_seconds=60,
    )
    upsert_nodes(
        state,
        [
            {
                "id": "mechanism",
                "question": "Which mechanism is identifiable?",
                "depends_on": [],
            },
            {
                "id": "forecast",
                "question": "Does the selected mechanism forecast?",
                "depends_on": ["mechanism"],
            },
        ],
    )
    state["evidence"]["e-mechanism"] = {
        "id": "e-mechanism",
        "node_id": "mechanism",
        "status": "verified",
    }
    state["evidence"]["e-forecast"] = {
        "id": "e-forecast",
        "node_id": "forecast",
        "status": "verified",
    }

    upsert_nodes(
        state,
        [
            {
                "id": "mechanism",
                "question": "Which mechanism survives the new diagnostic?",
                "depends_on": [],
            }
        ],
    )

    assert node_states(state)["mechanism"] == "open"
    assert node_states(state)["forecast"] == "blocked"
    assert state["evidence"]["e-mechanism"]["status"] == "revoked"
    assert state["evidence"]["e-forecast"]["status"] == "revoked"

    with pytest.raises(ValueError, match="cycle"):
        upsert_nodes(
            state,
            [
                {"id": "a", "question": "a?", "depends_on": ["b"]},
                {"id": "b", "question": "b?", "depends_on": ["a"]},
            ],
        )


def test_three_planes_project_support_without_duplicate_workflow_state() -> None:
    state = new_state(
        "Test a falsifiable hypothesis.",
        max_steps=3,
        max_tool_calls=4,
        max_seconds=60,
    )

    before = state_planes(state)
    assert set(before) == {"research", "execution", "evidence"}
    assert before["research"]["nodes"]["root"]["support"] == "open"
    assert "state" not in state["nodes"]["root"]
    assert "attempts" not in state["nodes"]["root"]
    assert "delivery" not in state

    state["evidence"]["e-root"] = {
        "id": "e-root",
        "node_id": "root",
        "status": "verified",
    }

    after = state_planes(state)
    assert after["research"]["nodes"]["root"]["support"] == "supported"
    assert after["evidence"]["counts"]["verified"] == 1


def test_legacy_projection_caches_migrate_to_source_facts(tmp_path: Path) -> None:
    state = new_state(
        "Resume an older thin run.",
        max_steps=3,
        max_tool_calls=4,
        max_seconds=60,
    )
    state["nodes"]["root"]["state"] = "supported"
    state["nodes"]["root"]["attempts"] = 2
    state["delivery"] = {
        "answer": "An older exploratory answer.",
        "evidence_ids": [],
        "limitations": ["It was not qualified."],
        "captured_at": "2026-01-01T00:00:00+00:00",
        "qualification_error": "legacy qualification failed",
    }
    store = StateStore(tmp_path)
    store.initialize(state)

    loaded = store.load()

    assert "state" not in loaded["nodes"]["root"]
    assert "attempts" not in loaded["nodes"]["root"]
    assert "delivery" not in loaded
    assert node_states(loaded)["root"] == "open"
    assert delivery_projection(loaded)["qualification_error"] == (
        "legacy qualification failed"
    )


def test_legacy_evidence_without_admission_keeps_claim_semantics(
    tmp_path: Path,
) -> None:
    state = new_state(
        "Load evidence written before selective review existed.",
        max_steps=3,
        max_tool_calls=4,
        max_seconds=60,
    )
    state["evidence"]["legacy-result"] = {
        "id": "legacy-result",
        "node_id": "root",
        "status": "verified",
    }
    store = StateStore(tmp_path)
    store.initialize(state)

    loaded = store.load()

    assert loaded["evidence"]["legacy-result"]["admission"] == "claim"


def test_one_step_run_promotes_checked_independently_reviewed_evidence(
    tmp_path: Path,
) -> None:
    modeler = ScriptedModel(
        [
            _action(
                summary="Compute a directly checkable result.",
                tool_calls=[_write_json(4)],
                candidate_claims=[_claim(4)],
                final={
                    "answer": "The answer is 4.",
                    "evidence_ids": ["e-result"],
                    "limitations": ["This establishes only the stated finite computation."],
                },
            )
        ]
    )
    verifier = ScriptedModel([_approve(), _approve()])

    result = ModelingLoop(
        workspace=tmp_path,
        modeler=modeler,
        verifier=verifier,
        max_steps=1,
    ).run("Compute 2 + 2 and support the answer with an artifact.")

    assert result.status == "completed"
    assert node_states(result.state)["root"] == "supported"
    assert result.state["evidence"]["e-result"]["status"] == "verified"
    assert "state" not in result.state["nodes"]["root"]
    assert "delivery" not in result.state
    assert delivery_projection(result.state, tmp_path)["status"] == "verified"
    assert (tmp_path / "paper" / "final.md").is_file()
    assert len(verifier.calls) == 2
    trace = (tmp_path / ".modeling-agent" / "trace.jsonl").read_text(
        encoding="utf-8"
    )
    assert "tool.result" in trace
    assert "evidence.result" in trace
    assert "run.completed" in trace

    (tmp_path / "artifacts" / "result.json").write_text(
        '{"value": 5}', encoding="utf-8"
    )
    stale = delivery_projection(result.state, tmp_path)
    assert stale["status"] == "best_effort_unverified"
    assert stale["evidence_integrity"]["e-result"] == "stale"
    assert result.state["evidence"]["e-result"]["status"] == "verified"


def test_empty_checks_never_reach_the_verifier(tmp_path: Path) -> None:
    claim = _claim(4)
    claim["checks"] = []
    modeler = ScriptedModel(
        [
            _action(
                tool_calls=[_write_json(4)],
                candidate_claims=[claim],
            )
        ]
    )
    verifier = ScriptedModel([])

    result = ModelingLoop(
        workspace=tmp_path,
        modeler=modeler,
        verifier=verifier,
        max_steps=1,
    ).run("Make an unsupported claim.")

    assert result.status == "stopped"
    assert result.state["evidence"]["e-result"]["status"] == "rejected"
    assert "requires at least one check" in result.state["evidence"]["e-result"]["error"]
    assert verifier.calls == []


def test_failed_attempt_can_change_direction_and_recover(tmp_path: Path) -> None:
    first_claim = _claim(5)
    first_claim["checks"] = [
        {
            "kind": "numeric_assertion",
            "arguments_json": json.dumps(
                {
                    "path": "artifacts/result.json",
                    "field": "value",
                    "operator": "==",
                    "value": 5,
                    "tolerance": 0,
                }
            ),
        }
    ]
    modeler = ScriptedModel(
        [
            _action(
                summary="First hypothesis fails its own check.",
                tool_calls=[_write_json(4)],
                candidate_claims=[first_claim],
            ),
            _action(
                summary="Revise the claim instead of forcing the first answer.",
                candidate_claims=[_claim(4)],
                final={
                    "answer": "The corrected result is 4.",
                    "evidence_ids": ["e-result"],
                    "limitations": [],
                },
            ),
        ]
    )
    verifier = ScriptedModel([_approve(), _approve()])

    result = ModelingLoop(
        workspace=tmp_path,
        modeler=modeler,
        verifier=verifier,
        max_steps=2,
    ).run("Compute a checked result and recover from a failed hypothesis.")

    assert result.status == "completed"
    assert "attempts" not in result.state["nodes"]["root"]
    assert result.state["evidence"]["e-result"]["revision"] == 2
    assert result.state["evidence"]["e-result"]["status"] == "verified"


def test_working_candidate_defers_review_and_reaches_next_context(
    tmp_path: Path,
) -> None:
    working = _claim(4)
    working["admission"] = "working"
    modeler = ScriptedModel(
        [
            _action(
                summary="Keep a checked intermediate result without blocking.",
                tool_calls=[_write_json(4)],
                candidate_claims=[working],
            ),
            _action(summary="Use the working result to choose the next experiment."),
        ]
    )
    verifier = ScriptedModel([])

    result = ModelingLoop(
        workspace=tmp_path,
        modeler=modeler,
        verifier=verifier,
        max_steps=2,
    ).run("Explore a checked result before deciding whether it supports a claim.")

    record = result.state["evidence"]["e-result"]
    assert result.status == "stopped"
    assert record["status"] == "candidate"
    assert record["admission"] == "working"
    assert record["review_deferred"] is True
    assert verifier.calls == []
    assert state_planes(result.state)["evidence"]["working_candidates"] == 1
    second_prompt = modeler.calls[1]["prompt"]
    assert '"admission": "working"' in second_prompt
    assert '"usable_for_exploration": true' in second_prompt
    assert '"admitted_for_claim": false' in second_prompt
    trace = (tmp_path / ".modeling-agent" / "trace.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"review_requested":false' in trace


def test_action_schema_requires_explicit_admission() -> None:
    candidate_schema = ACTION_SCHEMA["properties"]["candidate_claims"]["items"]

    assert "admission" in candidate_schema["required"]
    assert candidate_schema["properties"]["admission"]["enum"] == [
        "working",
        "claim",
    ]


def test_working_candidate_is_reviewed_only_when_promoted_to_claim(
    tmp_path: Path,
) -> None:
    working = _claim(4)
    working["admission"] = "working"
    promoted = _claim(4)
    promoted["admission"] = "claim"
    modeler = ScriptedModel(
        [
            _action(
                summary="Record a checked working result.",
                tool_calls=[_write_json(4)],
                candidate_claims=[working],
            ),
            _action(
                summary="Promote the decision-critical result.",
                candidate_claims=[promoted],
                final={
                    "answer": "The promoted answer is 4.",
                    "evidence_ids": ["e-result"],
                    "limitations": ["This establishes only the local computation."],
                },
            ),
        ]
    )
    verifier = ScriptedModel([_approve(), _approve()])

    result = ModelingLoop(
        workspace=tmp_path,
        modeler=modeler,
        verifier=verifier,
        max_steps=2,
    ).run("Explore first, then qualify only the decision-critical result.")

    record = result.state["evidence"]["e-result"]
    assert result.status == "completed"
    assert record["status"] == "verified"
    assert record["admission"] == "claim"
    assert record["revision"] == 2
    assert len(verifier.calls) == 2
    assert '"admission": "claim"' in verifier.calls[0]["prompt"]


def test_invalid_admission_fails_before_independent_review(tmp_path: Path) -> None:
    claim = _claim(4)
    claim["admission"] = "automatic"
    verifier = ScriptedModel([])

    result = ModelingLoop(
        workspace=tmp_path,
        modeler=ScriptedModel(
            [
                _action(
                    tool_calls=[_write_json(4)],
                    candidate_claims=[claim],
                )
            ]
        ),
        verifier=verifier,
        max_steps=1,
    ).run("Reject an undefined evidence admission level.")

    record = result.state["evidence"]["e-result"]
    assert record["status"] == "rejected"
    assert record["admission"] == "claim"
    assert record["requested_admission"] == "automatic"
    assert "invalid evidence admission" in record["error"]
    assert verifier.calls == []


def test_problem_revision_revokes_working_candidate() -> None:
    state = new_state(
        "Test a provisional mechanism.",
        max_steps=3,
        max_tool_calls=4,
        max_seconds=60,
    )
    state["evidence"]["working-mechanism"] = {
        "id": "working-mechanism",
        "node_id": "root",
        "statement": "The provisional mechanism fits the current question.",
        "artifact": "artifacts/mechanism.json",
        "status": "candidate",
        "admission": "working",
    }

    upsert_nodes(
        state,
        [
            {
                "id": "root",
                "question": "Test a materially revised mechanism.",
                "depends_on": [],
                "priority": 1.0,
            }
        ],
    )

    record = state["evidence"]["working-mechanism"]
    assert record["status"] == "revoked"
    assert record["revocation_reason"] == "problem graph dependency changed"


def test_tool_calls_always_return_structured_results(tmp_path: Path) -> None:
    tools = ToolRegistry(tmp_path)

    malformed = tools.execute("write_text", {"path": "artifacts/a.txt"})
    unknown = tools.execute("delete_everything", {})

    assert malformed["status"] == "error"
    assert malformed["error_type"] == "ValueError"
    assert unknown["status"] == "error"
    assert unknown["error_type"] == "unknown_tool"


def test_write_files_batches_related_files_after_full_validation(
    tmp_path: Path,
) -> None:
    tools = ToolRegistry(tmp_path)

    result = tools.execute(
        "write_files",
        {
            "files": [
                {"path": "src/model.py", "content": "print('model')\n"},
                {"path": "checks/check_model.py", "content": "assert True\n"},
            ]
        },
    )

    assert result["status"] == "success"
    assert len(result["data"]["files"]) == 2
    assert (tmp_path / "src" / "model.py").is_file()
    assert (tmp_path / "checks" / "check_model.py").is_file()


def test_read_files_batches_related_inputs_with_total_bound(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    tools = ToolRegistry(tmp_path)

    result = tools.execute(
        "read_files",
        {"paths": ["a.txt", "b.txt"], "max_chars": 20_000},
    )

    assert result["status"] == "success"
    assert [item["content"] for item in result["data"]["files"]] == [
        "alpha",
        "beta",
    ]
    assert result["data"]["total_chars"] == 9


def test_standard_root_submission_is_writable_but_inputs_are_not(
    tmp_path: Path,
) -> None:
    tools = ToolRegistry(tmp_path)

    delivery = tools.execute(
        "write_text", {"path": "submission.json", "content": "{}\n"}
    )
    input_overwrite = tools.execute(
        "write_text", {"path": "task.md", "content": "changed\n"}
    )

    assert delivery["status"] == "success"
    assert input_overwrite["status"] == "error"
    assert (tmp_path / "submission.json").is_file()
    assert not (tmp_path / "task.md").exists()


def test_context_keeps_moderate_batched_observation_intact(tmp_path: Path) -> None:
    loop = ModelingLoop(
        tmp_path,
        modeler=ScriptedModel([]),
        verifier=ScriptedModel([]),
    )
    state = new_state(
        "objective",
        max_steps=12,
        max_tool_calls=30,
        max_seconds=1800,
    )
    content = "x" * 15_000
    state["observations"].append(
        {
            "kind": "tool.result",
            "result": {"status": "success", "data": {"content": content}},
        }
    )

    observation = loop._context(state)["recent_observations"][0]

    assert observation["result"]["data"]["content"] == content
    assert "truncated" not in observation


def test_context_marks_changed_verified_evidence_stale(tmp_path: Path) -> None:
    artifact = tmp_path / "results" / "value.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"value": 1}\n', encoding="utf-8")
    loop = ModelingLoop(
        tmp_path,
        modeler=ScriptedModel([]),
        verifier=ScriptedModel([]),
    )
    state = new_state(
        "objective",
        max_steps=12,
        max_tool_calls=30,
        max_seconds=1800,
    )
    state["evidence"]["value_v1"] = {
        "id": "value_v1",
        "node_id": "root",
        "statement": "value is one",
        "artifact": "results/value.json",
        "artifact_sha256": "not-the-current-hash",
        "supporting_artifacts": [],
        "checks": [],
        "status": "verified",
        "review": {"verdict": "APPROVE"},
    }

    context = loop._context(state)

    assert context["evidence"]["value_v1"]["integrity"] == "stale"


def test_stale_working_evidence_is_not_usable_for_exploration(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "results" / "working.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"value": 1}\n', encoding="utf-8")
    loop = ModelingLoop(
        tmp_path,
        modeler=ScriptedModel([]),
        verifier=ScriptedModel([]),
    )
    state = new_state(
        "Use current working results but reject stale ones.",
        max_steps=3,
        max_tool_calls=4,
        max_seconds=60,
    )
    state["evidence"]["working-result"] = {
        "id": "working-result",
        "node_id": "root",
        "statement": "The provisional value is one.",
        "artifact": "results/working.json",
        "artifact_sha256": file_hash(artifact),
        "supporting_artifacts": [],
        "checks": [],
        "status": "candidate",
        "admission": "working",
    }

    current = loop._context(state)["evidence"]["working-result"]
    artifact.write_text('{"value": 2}\n', encoding="utf-8")
    stale = loop._context(state)["evidence"]["working-result"]

    assert current["usable_for_exploration"] is True
    assert current["admitted_for_claim"] is False
    assert stale["integrity"] == "stale"
    assert stale["usable_for_exploration"] is False


def test_python_compute_denies_obvious_network_code(tmp_path: Path) -> None:
    tools = ToolRegistry(tmp_path)
    write = tools.execute(
        "write_text",
        {"path": "src/network.py", "content": "import socket\nprint('no')\n"},
    )
    assert write["status"] == "success"

    run = tools.execute(
        "python_compute",
        {"script": "src/network.py", "timeout": 5},
    )
    assert run["status"] == "denied"
    assert run["error_type"] == "permission_denied"


def test_malformed_mechanical_check_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "value.json").write_text(
        '{"value": 4}\n', encoding="utf-8"
    )

    result = run_check(
        tmp_path,
        "numeric_assertion",
        {
            "path": "artifacts/value.json",
            "operator": "==",
            "value": 4,
        },
    )

    assert result["ok"] is False
    assert result["actual"] is None
    json.dumps(result, allow_nan=False)


def test_ablation_contract_is_frozen_and_results_are_append_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ablation.json"
    manifest = freeze_ablation(
        path,
        objective="An unseen modeling problem.",
        model="gpt-5.6-sol",
        max_model_turns=8,
        max_tool_calls=24,
        max_wall_seconds=900,
    )

    assert [arm["id"] for arm in manifest["arms"]] == [
        "raw_codex",
        "thin_harness",
        "native_sidecar",
    ]
    assert manifest["schema"] == 2
    assert manifest["results"] == {}
    with pytest.raises(FileExistsError):
        freeze_ablation(
            path,
            objective="Do not overwrite.",
            model="gpt-5.6-sol",
            max_model_turns=8,
            max_tool_calls=24,
            max_wall_seconds=900,
        )

    record_result(
        path,
        "thin_harness",
        {"status": "completed", "artifact": "result.json"},
    )
    with pytest.raises(ValueError, match="already recorded"):
        record_result(
            path,
            "thin_harness",
            {"status": "changed"},
        )


def test_stopped_run_can_resume_with_a_larger_budget(tmp_path: Path) -> None:
    first = ScriptedModel([_action(summary="Need another step.")])
    stopped = ModelingLoop(
        workspace=tmp_path,
        modeler=first,
        verifier=ScriptedModel([]),
        max_steps=1,
    ).run("Use two attempts.")
    assert stopped.status == "stopped"

    resumed = ModelingLoop(
        workspace=tmp_path,
        modeler=ScriptedModel(
            [
                _action(
                    tool_calls=[_write_json(4)],
                    candidate_claims=[_claim(4)],
                    final={
                        "answer": "Recovered.",
                        "evidence_ids": ["e-result"],
                        "limitations": [],
                    },
                )
            ]
        ),
        verifier=ScriptedModel([_approve(), _approve()]),
        max_steps=2,
    ).run("Use two attempts.")

    assert resumed.status == "completed"
    state = StateStore(tmp_path).load()
    assert state["budgets"]["max_steps"] == 2


def test_one_turn_can_compute_and_send_complete_bundle_to_reviewer(
    tmp_path: Path,
) -> None:
    compute_source = (
        "import json\n"
        "from pathlib import Path\n"
        "Path('artifacts/result.json').parent.mkdir(exist_ok=True)\n"
        "Path('artifacts/result.json').write_text("
        "json.dumps({'value': 4}), encoding='utf-8')\n"
    )
    check_source = (
        "import json\n"
        "from pathlib import Path\n"
        "value = json.loads(Path('artifacts/result.json').read_text("
        "encoding='utf-8'))['value']\n"
        "assert value == 4\n"
    )
    tool_calls = [
        {
            "call_id": "write-compute",
            "name": "write_text",
            "arguments_json": json.dumps(
                {"path": "src/compute.py", "content": compute_source}
            ),
        },
        {
            "call_id": "write-check",
            "name": "write_text",
            "arguments_json": json.dumps(
                {"path": "checks/check_result.py", "content": check_source}
            ),
        },
        {
            "call_id": "run-compute",
            "name": "python_compute",
            "arguments_json": json.dumps(
                {
                    "script": "src/compute.py",
                    "args": [],
                    "timeout": 10,
                    "expected_outputs": ["artifacts/result.json"],
                }
            ),
        },
    ]
    claim = {
        "id": "e-computed",
        "node_id": "root",
        "statement": "The executable computation returns 4.",
        "artifact": "artifacts/result.json",
        "supporting_artifacts": ["src/compute.py"],
        "checks": [
            {
                "kind": "python_check",
                "arguments_json": json.dumps(
                    {"script": "checks/check_result.py"}
                ),
            }
        ],
    }
    modeler = ScriptedModel(
        [
            _action(
                tool_calls=tool_calls,
                candidate_claims=[claim],
                final={
                    "answer": "The checked computation returns 4.",
                    "evidence_ids": ["e-computed"],
                    "limitations": ["This is a finite local computation."],
                },
            )
        ]
    )
    verifier = ScriptedModel([_approve(), _approve()])

    result = ModelingLoop(
        workspace=tmp_path,
        modeler=modeler,
        verifier=verifier,
        max_steps=1,
    ).run("Compute and independently check 2 + 2.")

    assert result.status == "completed"
    assert result.state["tool_calls"] == 3
    assert len(modeler.calls) == 1
    candidate_prompt = verifier.calls[0]["prompt"]
    assert "src/compute.py" in candidate_prompt
    assert "Path('artifacts/result.json')" in candidate_prompt
    assert "checks/check_result.py" in candidate_prompt
    assert "assert value == 4" in candidate_prompt
    final_prompt = verifier.calls[1]["prompt"]
    assert "src/compute.py" in final_prompt
    assert "checks/check_result.py" in final_prompt


def test_last_turn_requires_and_preserves_best_effort_delivery(
    tmp_path: Path,
) -> None:
    modeler = ScriptedModel(
        [
            _action(
                summary="Deliver what is known without claiming qualification.",
                final={
                    "answer": "A plausible answer exists, but it is not verified.",
                    "evidence_ids": [],
                    "limitations": ["No candidate evidence passed qualification."],
                },
            )
        ]
    )

    result = ModelingLoop(
        workspace=tmp_path,
        modeler=modeler,
        verifier=ScriptedModel([]),
        max_steps=1,
    ).run("Attempt a difficult problem under a one-turn budget.")

    assert result.status == "stopped"
    assert result.reason == "step_budget_reached_with_best_effort_delivery"
    assert result.state["final"] is None
    delivery = delivery_projection(result.state, tmp_path)
    assert delivery["status"] == "best_effort_unverified"
    assert delivery["claim_ceiling"] == "exploratory"
    assert "root problem and its declared prerequisites are not supported" in delivery[
        "qualification_error"
    ]
    call = modeler.calls[0]
    assert call["schema"]["properties"]["final"]["type"] == "object"
    assert "BUDGET PRESSURE" in call["prompt"]
    assert "advisory, not a stage gate" in call["prompt"]
    assert "FINAL RESERVED TURN" in call["prompt"]


def test_changed_artifact_blocks_qualification_but_keeps_delivery(
    tmp_path: Path,
) -> None:
    modeler = ScriptedModel(
        [
            _action(
                tool_calls=[_write_json(4)],
                candidate_claims=[_claim(4)],
            ),
            _action(
                tool_calls=[_write_json(5)],
                final={
                    "answer": "The answer is 4.",
                    "evidence_ids": ["e-result"],
                    "limitations": [],
                },
            ),
        ]
    )
    verifier = ScriptedModel([_approve()])

    result = ModelingLoop(
        workspace=tmp_path,
        modeler=modeler,
        verifier=verifier,
        max_steps=2,
    ).run("Do not qualify a result after its artifact changes.")

    assert result.status == "stopped"
    delivery = delivery_projection(result.state, tmp_path)
    assert delivery["status"] == "best_effort_unverified"
    assert "changed after review" in delivery["qualification_error"]
    assert "best_effort_unverified" in (
        tmp_path / "paper" / "final.md"
    ).read_text(encoding="utf-8")
    assert len(verifier.calls) == 1


def test_declared_compute_outputs_get_safe_parent_directories(
    tmp_path: Path,
) -> None:
    tools = ToolRegistry(tmp_path)
    source = (
        "from pathlib import Path\n"
        "Path('results/nested/value.json').write_text("
        "'{\\\"value\\\": 4}', encoding='utf-8')\n"
    )
    assert tools.execute(
        "write_text", {"path": "src/generate.py", "content": source}
    )["status"] == "success"

    result = tools.execute(
        "python_compute",
        {
            "script": "src/generate.py",
            "args": [],
            "timeout": 10,
            "expected_outputs": ["results/nested/value.json"],
        },
    )

    assert result["status"] == "success"
    assert result["data"]["outputs"][0]["exists"] is True


def test_verified_prerequisites_feed_next_claim_and_final_closes_root(
    tmp_path: Path,
) -> None:
    model_claim = {
        "id": "model-evidence",
        "node_id": "model",
        "statement": "The model calculation equals 4.",
        "artifact": "artifacts/model.json",
        "supporting_artifacts": [],
        "checks": [
            {
                "kind": "numeric_assertion",
                "arguments_json": json.dumps(
                    {
                        "path": "artifacts/model.json",
                        "field": "value",
                        "operator": "==",
                        "value": 4,
                        "tolerance": 0,
                    }
                ),
            }
        ],
    }
    decision_claim = {
        "id": "decision-evidence",
        "node_id": "decision",
        "statement": "Given the checked model, the declared policy is maintain.",
        "artifact": "artifacts/decision.json",
        "supporting_artifacts": [],
        "checks": [
            {
                "kind": "file_nonempty",
                "arguments_json": json.dumps(
                    {"path": "artifacts/decision.json"}
                ),
            }
        ],
    }
    modeler = ScriptedModel(
        [
            _action(
                upsert_nodes=[
                    {
                        "id": "root",
                        "question": "Synthesize the model and decision.",
                        "depends_on": ["model", "decision"],
                        "priority": 1.0,
                    },
                    {
                        "id": "model",
                        "question": "What does the model compute?",
                        "depends_on": [],
                        "priority": 1.0,
                    },
                    {
                        "id": "decision",
                        "question": "What follows from the model?",
                        "depends_on": ["model"],
                        "priority": 1.0,
                    },
                ],
                tool_calls=[
                    {
                        "call_id": "write-model",
                        "name": "write_text",
                        "arguments_json": json.dumps(
                            {
                                "path": "artifacts/model.json",
                                "content": '{"value": 4}',
                            }
                        ),
                    },
                    {
                        "call_id": "write-decision",
                        "name": "write_text",
                        "arguments_json": json.dumps(
                            {
                                "path": "artifacts/decision.json",
                                "content": '{"policy": "maintain"}',
                            }
                        ),
                    },
                ],
                candidate_claims=[model_claim, decision_claim],
                final={
                    "answer": "The checked policy is maintain.",
                    "evidence_ids": ["model-evidence", "decision-evidence"],
                    "limitations": ["This is a local toy decision."],
                },
            )
        ]
    )
    verifier = ScriptedModel([_approve(), _approve(), _approve()])

    result = ModelingLoop(
        workspace=tmp_path,
        modeler=modeler,
        verifier=verifier,
        max_steps=1,
    ).run("Solve a decomposed model and decision problem.")

    assert result.status == "completed"
    assert node_states(result.state)["root"] == "supported"
    assert len(verifier.calls) == 3
    decision_prompt = verifier.calls[1]["prompt"]
    assert "verified_prerequisite_evidence" in decision_prompt
    assert "model-evidence" in decision_prompt


def test_delivery_projection_preserves_model_authored_paper(
    tmp_path: Path,
) -> None:
    authored = "# Scientific report\n\nA detailed model-authored paper.\n"
    modeler = ScriptedModel(
        [
            _action(
                tool_calls=[
                    _write_json(4),
                    {
                        "call_id": "write-paper",
                        "name": "write_text",
                        "arguments_json": json.dumps(
                            {"path": "paper/final.md", "content": authored}
                        ),
                    },
                ],
                candidate_claims=[_claim(4)],
                final={
                    "answer": "The answer is 4.",
                    "evidence_ids": ["e-result"],
                    "limitations": [],
                },
            )
        ]
    )

    result = ModelingLoop(
        workspace=tmp_path,
        modeler=modeler,
        verifier=ScriptedModel([_approve(), _approve()]),
        max_steps=1,
    ).run("Preserve a richer authored paper.")

    assert result.status == "completed"
    assert (tmp_path / "paper" / "final.md").read_text(encoding="utf-8") == authored
    assert "locally_supported" in (
        tmp_path / "paper" / "delivery.md"
    ).read_text(encoding="utf-8")


class _ScriptedNativeResearcher:
    def __init__(self, modes: list[str]):
        self.modes = list(modes)
        self.prompts: list[str] = []

    def run(
        self,
        prompt: str,
        *,
        role: str,
        workspace: Path,
        trace_path: Path,
        timeout_seconds: int,
    ) -> dict[str, object]:
        self.prompts.append(prompt)
        mode = self.modes.pop(0)
        contract_path = workspace / ".modeling-agent" / "task-contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract_hash = content_hash(contract)
        for directory in ("src", "checks", "results", "paper"):
            (workspace / directory).mkdir(parents=True, exist_ok=True)
        generator = (
            "from pathlib import Path\n"
            "Path('results').mkdir(exist_ok=True)\n"
            "Path('results/value.json').write_text("
            "'{\\\"value\\\":42}\\n', encoding='utf-8')\n"
        )
        checker = (
            "import json\n"
            "from pathlib import Path\n"
            "value=json.loads(Path('results/value.json').read_text("
            "encoding='utf-8'))['value']\n"
            "assert value == 42\n"
            "print('value verified')\n"
        )
        (workspace / "src" / "solve.py").write_text(generator, encoding="utf-8")
        (workspace / "checks" / "check_value.py").write_text(
            checker, encoding="utf-8"
        )
        (workspace / "results" / "value.json").write_text(
            '{"value":42}\n', encoding="utf-8"
        )
        if mode != "missing-paper":
            (workspace / "paper" / "final.md").write_text(
                "# Result\n\nThe locally computed value is 42.\n",
                encoding="utf-8",
            )
        manifest = {
            "schema": 1,
            "contract_hash": contract_hash,
            "final_answer": "The locally computed value is 42.",
            "claims": [
                {
                    "id": "computed-value",
                    "statement": "The locally computed value is 42.",
                    "artifact_paths": ["results/value.json", "paper/final.md"],
                    "decision_critical": True,
                }
            ],
            "limitations": ["This is a local deterministic fixture."],
            "artifacts": [
                {"path": "paper/final.md", "role": "paper"},
                {"path": "src/solve.py", "role": "generator"},
                {"path": "checks/check_value.py", "role": "check"},
                {"path": "results/value.json", "role": "result"},
            ],
            "generators": [
                {
                    "script": "src/solve.py",
                    "args": [],
                    "expected_outputs": ["results/value.json"],
                    "timeout": 30,
                }
            ],
            "checks": [
                {
                    "kind": "python_check",
                    "arguments": {"script": "checks/check_value.py"},
                }
            ],
        }
        (workspace / "submission_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        if mode == "mutate-contract":
            contract["objective"] = "tampered"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            '{"type":"item.completed","item":{"id":"write","type":"file_change"}}\n',
            encoding="utf-8",
        )
        return {
            "role": role,
            "duration_seconds": 0.01,
            "observable_tool_calls": 1,
            "timeout_seconds": timeout_seconds,
        }


def test_native_sidecar_completes_after_replay_checks_and_fresh_review(
    tmp_path: Path,
) -> None:
    researcher = _ScriptedNativeResearcher(["valid"])
    sidecar = NativeSidecar(
        tmp_path,
        researcher=researcher,
        verifier=ScriptedModel([_approve()]),
        model_requested="scripted",
        max_attempts=2,
        max_seconds=60,
    )

    state = sidecar.run(default_contract("Compute a reproducible value."))

    assert state["status"] == "completed"
    assert state["attempts"][0]["status"] == "verified"
    assert state["attempts"][0]["mechanical"]["replay_ok"] is True
    assert state["attempts"][0]["mechanical"]["checks_ok"] is True
    assert state["delivery"]["verified"] is True
    assert state["delivery"]["claim_ceiling"] == "locally_supported"
    assert native_status(tmp_path)["status"] == "completed"
    assert "Work natively inside the current project" in researcher.prompts[0]


def test_native_sidecar_repairs_a_mechanical_contract_failure(
    tmp_path: Path,
) -> None:
    researcher = _ScriptedNativeResearcher(["missing-paper", "valid"])
    sidecar = NativeSidecar(
        tmp_path,
        researcher=researcher,
        verifier=ScriptedModel([_approve()]),
        model_requested="scripted",
        max_attempts=2,
        max_seconds=60,
    )

    state = sidecar.run(default_contract("Repair a missing delivery."))

    assert state["status"] == "completed"
    assert [item["status"] for item in state["attempts"]] == [
        "mechanical_failed",
        "verified",
    ]
    assert "artifact is missing or empty: paper/final.md" in researcher.prompts[1]


def test_native_sidecar_uses_review_findings_for_one_bounded_repair(
    tmp_path: Path,
) -> None:
    researcher = _ScriptedNativeResearcher(["valid", "valid"])
    rejected = {
        "verdict": "REJECT",
        "findings": ["Add an explicit local claim boundary."],
        "claim_strength": "unsupported",
    }
    sidecar = NativeSidecar(
        tmp_path,
        researcher=researcher,
        verifier=ScriptedModel([rejected, _approve()]),
        model_requested="scripted",
        max_attempts=2,
        max_seconds=60,
    )

    state = sidecar.run(default_contract("Repair a rejected final claim."))

    assert state["status"] == "completed"
    assert [item["status"] for item in state["attempts"]] == [
        "review_rejected",
        "verified",
    ]
    assert "Add an explicit local claim boundary." in researcher.prompts[1]


def test_native_sidecar_detects_contract_tampering_and_preserves_delivery(
    tmp_path: Path,
) -> None:
    sidecar = NativeSidecar(
        tmp_path,
        researcher=_ScriptedNativeResearcher(["mutate-contract"]),
        verifier=ScriptedModel([]),
        model_requested="scripted",
        max_attempts=1,
        max_seconds=60,
    )

    state = sidecar.run(default_contract("Do not rewrite the task contract."))

    assert state["status"] == "stopped"
    assert state["attempts"][0]["status"] == "mechanical_failed"
    assert "researcher modified the immutable task contract" in state["attempts"][0][
        "contract_errors"
    ]
    assert state["delivery"]["verified"] is False
    assert state["delivery"]["final_answer"]


def test_native_contract_rejects_workspace_escape() -> None:
    contract = default_contract("Stay inside the workspace.")
    contract["manifest_path"] = "../escape.json"

    with pytest.raises(ValueError, match="escapes workspace"):
        validate_contract(contract)
