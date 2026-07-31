"""The complete thin loop: propose, compute, check, review, and continue."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import (
    StateStore,
    canonical_json,
    content_hash,
    delivery_projection,
    evidence_integrity,
    file_hash,
    new_state,
    node_states,
    now,
    safe_path,
    state_planes,
    upsert_nodes,
    validate_identifier,
)
from .model import ACTION_SCHEMA, REVIEW_SCHEMA, SYNTHESIS_ACTION_SCHEMA, ModelAdapter
from .tools import CHECK_DESCRIPTIONS, ToolRegistry, run_check


MODELER_INSTRUCTIONS = """\
You are the research modeler inside a lightweight mathematical-modeling loop.

You may propose any mathematical structure, representation, approximation,
algorithm, experiment, or counterexample. There is no registered model-family
allowlist. Prefer the lowest-cost observation that can distinguish competing
explanations, but do not avoid a complex model when the evidence requires it.

The harness controls only tool side effects and admission to verified evidence.
You may create or revise problem nodes at any time. A node's depends_on list
contains prerequisites that must be supported first. To decompose the root
question, add child nodes and update root.depends_on.

Available tools are project-local batched read/write and bounded Python
computation.
Tool calls execute in listed order before candidate claims are reviewed, so one
action may write a script, run it, register its result, and draft the answer.
Numerical or empirical claims must reference a primary artifact, declare the
source, checker, and necessary input files as supporting_artifacts, and include
at least one meaningful mechanical check. An empty check list is rejected. You
cannot approve your own claim.

Evidence has two admission levels. Use `admission=working` for a mechanically
checked intermediate result that should inform later branches without spending
an independent review. It remains visibly unverified and cannot support the
Problem Graph or final answer. Use `admission=claim` only when a result is
decision-critical, intended for the final answer, or would justify an external
action; this requests a fresh independent review. Promote useful working
evidence later by resubmitting the same id with `admission=claim`. Do not review
every intermediate calculation merely because it exists.

At each turn choose the uncertainty most likely to change the final answer.
Prefer `read_files` for related inputs and `write_files` when source and
checker belong to one computation. A
candidate's mechanical checks run automatically during evidence admission and
do not consume the tool budget; do not also call python_compute on the checker
unless debugging a prior failure. A meaningful checker must independently
recompute the claimed quantity from frozen inputs, not merely assert fields
copied into the result. Its assertions must cover every material component of
the candidate statement, including any claimed baseline or alternative-policy
comparison; otherwise narrow the statement. Group outputs that share one
computation and one
falsification boundary instead of serializing every deliverable into a node.
When a thresholded decision depends on predictions outside observed support,
propagate that uncertainty into the decision: compare a structurally distinct
extrapolation or explicit stress envelope. Do not leave extrapolation only as
prose while a nominal point forecast silently determines no action.
When six or fewer tool calls remain, prioritize complete submission artifacts
and preserve at least two calls for one repair and rerun.
As the remaining budget falls, prefer the work with the highest expected value
for the final decision. You may still open a new branch when it is more
informative than immediate synthesis; a workflow label never forbids a research
move. On the final reserved turn you must return a final answer even when
evidence is incomplete: cite only evidence ids that exist and state every
unresolved issue in limitations. Such an answer is preserved as exploratory
delivery, never promoted as verified.
For a verified final, cite the smallest current terminal evidence set. A
terminal evidence packet already carries reviewed prerequisite evidence; do not
also cite older evidence whose artifacts were later revised.
Return only the structured action requested by the response schema. The summary
must be a short operational explanation, not hidden chain-of-thought.
"""


REVIEWER_INSTRUCTIONS = """\
You are a fresh skeptical verifier. Assume the candidate is wrong.
Review only the supplied question, claim, complete bounded artifact bundle,
mechanical receipts, and any already-verified prerequisite evidence. The bundle
distinguishes the primary result from declared source, checker, and input
artifacts. Prerequisite evidence may support only its reviewed statement and
claim ceiling. APPROVE only when the checks are relevant, passed, and the
statement does not exceed what the packet establishes. Use locally_supported
only for an approval. Reject persuasive prose, empty validation, circular
checks, unsupported mechanism or causality, and silent extrapolation.
Apply only criteria entailed by the supplied question and candidate statement.
Do not invent extra admission requirements such as preregistration, benchmark
policy, or production readiness unless the packet explicitly claims them.
"""


@dataclass(frozen=True)
class RunResult:
    status: str
    reason: str
    workspace: Path
    state: dict[str, Any]


def _parse_json_object(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise ValueError(f"{label} must be a JSON string")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must decode to an object")
    return value


def _artifact_excerpt(
    path: Path, *, display_path: str | None = None, limit: int = 24_000
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": display_path or path.name,
        "bytes": path.stat().st_size,
        "sha256": file_hash(path),
    }
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        record["content"] = "<binary or unreadable text>"
        return record
    record["content"] = text[:limit]
    record["truncated"] = len(text) > limit
    return record


class ModelingLoop:
    """One modeler and one fresh verifier sharing only durable public state."""

    def __init__(
        self,
        workspace: Path,
        *,
        modeler: ModelAdapter,
        verifier: ModelAdapter,
        max_steps: int = 12,
        max_tool_calls: int = 30,
        max_seconds: int = 1800,
    ):
        self.workspace = workspace.resolve()
        self.modeler = modeler
        self.verifier = verifier
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.max_seconds = max_seconds
        self.store = StateStore(self.workspace)
        self.tools = ToolRegistry(self.workspace)

    def initialize(self, objective: str) -> dict[str, Any]:
        state = new_state(
            objective,
            max_steps=self.max_steps,
            max_tool_calls=self.max_tool_calls,
            max_seconds=self.max_seconds,
        )
        self.store.initialize(state)
        return state

    def _context(self, state: dict[str, Any]) -> dict[str, Any]:
        turns_left = max(
            0, state["budgets"]["max_steps"] - state["step"] + 1
        )
        planes = state_planes(state, self.workspace)
        evidence = {}
        for key, value in state["evidence"].items():
            integrity = evidence_integrity(self.workspace, value)
            status = value["status"]
            admission = value.get("admission", "claim")
            evidence[key] = {
                "status": value["status"],
                "admission": admission,
                "node_id": value["node_id"],
                "statement": value["statement"],
                "artifact": value["artifact"],
                "checks": value.get("checks", []),
                "review": value.get("review"),
                "integrity": integrity,
                "usable_for_exploration": (
                    status in {"candidate", "verified"} and integrity == "current"
                ),
                "admitted_for_claim": (
                    status == "verified" and integrity == "current"
                ),
            }
        observations_reversed = []
        remaining_observation_chars = 72_000
        for item in reversed(state["observations"][-12:]):
            encoded = canonical_json(item)
            allowance = min(48_000, remaining_observation_chars)
            if allowance <= 0:
                break
            if len(encoded) <= allowance:
                observations_reversed.append(item)
                remaining_observation_chars -= len(encoded)
            else:
                observations_reversed.append(
                    {
                        "kind": item.get("kind"),
                        "summary": encoded[:allowance],
                        "truncated": True,
                    }
                )
                remaining_observation_chars = 0
        observations = list(reversed(observations_reversed))
        return {
            "research": planes["research"],
            "execution": {
                **planes["execution"],
                "model_turns_left_including_current": turns_left,
                "tool_calls_left": max(
                    0,
                    state["budgets"]["max_tool_calls"] - state["tool_calls"],
                ),
                "tool_repair_reserve": 2,
            },
            "evidence": evidence,
            "recent_observations": observations,
            "tools": self.tools.descriptions,
            "mechanical_checks": CHECK_DESCRIPTIONS,
            "completion_rule": (
                "The final answer must cite verified evidence. It may close the "
                "root directly when every declared root prerequisite is supported; "
                "a separate root-synthesis candidate is not required."
            ),
            "delivery_rule": (
                "A final answer is still required on the last turn when scientific "
                "qualification is incomplete; it will be retained with an "
                "exploratory claim ceiling."
            ),
            "admission_policy": {
                "working": (
                    "Passed mechanical checks; may guide exploration, branching, "
                    "and repair, but does not support graph closure or final claims."
                ),
                "claim": (
                    "Requests fresh independent review and may become verified. "
                    "Required before final or external-action use."
                ),
                "promotion": (
                    "Resubmit the same working evidence id with admission=claim "
                    "when it becomes decision-critical."
                ),
            },
            "orchestration_hint": (
                "Use read_files for related inputs and write_files for related "
                "source/checker files. Then run only the generator and let "
                "candidate checks execute during admission."
            ),
        }

    def _model_prompt(self, state: dict[str, Any]) -> str:
        context = self._context(state)
        directive = ""
        turns_left = context["execution"]["model_turns_left_including_current"]
        if turns_left <= min(2, self.max_steps):
            directive = (
                "\n\nBUDGET PRESSURE\n"
                "A deliverable is due soon. Prefer the highest-value remaining "
                "research or synthesis action; this is an advisory, not a stage gate."
            )
        if turns_left == 1:
            directive += (
                "\nFINAL RESERVED TURN: final must be non-null. Give the best "
                "decision-useful answer possible and label unsupported parts in "
                "limitations."
            )
        return (
            MODELER_INSTRUCTIONS
            + "\n\nCURRENT DURABLE STATE\n"
            + json.dumps(context, ensure_ascii=False, indent=2)
            + directive
        )

    def _artifact_bundle(
        self,
        primary: str,
        supporting: list[str],
        check_arguments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        paths = [primary, *supporting]
        for arguments in check_arguments:
            for key in ("script", "path"):
                value = arguments.get(key)
                if isinstance(value, str):
                    paths.append(value)
        unique = list(dict.fromkeys(paths))
        if len(unique) > 16:
            raise ValueError("candidate artifact bundle exceeds 16 files")
        bundle = []
        for relative in unique:
            path = safe_path(self.workspace, relative)
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"artifact bundle file is missing or empty: {relative}")
            bundle.append(_artifact_excerpt(path, display_path=relative))
        return bundle

    def _record_model_receipt(self, state: dict[str, Any], role: str) -> None:
        receipt = getattr(
            self.modeler if role == "modeler" else self.verifier,
            "last_receipt",
            None,
        )
        if receipt:
            self.store.event(f"{role}.receipt", receipt)

    def _apply_nodes(self, state: dict[str, Any], action: dict[str, Any]) -> None:
        proposals = action.get("upsert_nodes", [])
        try:
            changed = upsert_nodes(state, proposals)
            if changed:
                observation = {"kind": "graph.updated", "nodes": changed}
                state["observations"].append(observation)
                self.store.event("graph.updated", observation)
        except (TypeError, ValueError) as exc:
            observation = {"kind": "graph.rejected", "error": str(exc)}
            state["observations"].append(observation)
            self.store.event("graph.rejected", observation)

    def _execute_tools(self, state: dict[str, Any], action: dict[str, Any]) -> None:
        for raw in action.get("tool_calls", []):
            call_id = raw.get("call_id") if isinstance(raw, dict) else None
            name = raw.get("name") if isinstance(raw, dict) else None
            if state["tool_calls"] >= state["budgets"]["max_tool_calls"]:
                result = {
                    "status": "error",
                    "summary": "tool-call budget reached",
                    "data": {},
                    "error_type": "budget_exceeded",
                }
            else:
                try:
                    arguments = _parse_json_object(
                        raw.get("arguments_json"), "tool arguments"
                    )
                    result = self.tools.execute(str(name), arguments)
                except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    result = {
                        "status": "error",
                        "summary": str(exc),
                        "data": {},
                        "error_type": "invalid_arguments",
                    }
                state["tool_calls"] += 1
            observation = {
                "kind": "tool.result",
                "call_id": call_id,
                "tool": name,
                "result": result,
            }
            state["observations"].append(observation)
            self.store.event("tool.result", observation)

    def _review_candidate(
        self,
        *,
        node: dict[str, Any],
        candidate: dict[str, Any],
        artifact_bundle: list[dict[str, Any]],
        check_records: list[dict[str, Any]],
        prerequisite_evidence: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = (
            REVIEWER_INSTRUCTIONS
            + "\n\nCANDIDATE REVIEW PACKET\n"
            + json.dumps(
                {
                    "question": node["question"],
                    "candidate": {
                        "id": candidate["id"],
                        "admission": candidate["admission"],
                        "statement": candidate["statement"],
                        "artifact": candidate["artifact"],
                    },
                    "artifact_bundle": artifact_bundle,
                    "mechanical_checks": check_records,
                    "verified_prerequisite_evidence": prerequisite_evidence,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        review = self.verifier.complete(
            prompt,
            REVIEW_SCHEMA,
            role="candidate-verifier",
            workspace=self.workspace,
        )
        self._record_model_receipt({}, "verifier")
        verdict = review.get("verdict")
        findings = review.get("findings")
        strength = review.get("claim_strength")
        if (
            verdict not in {"APPROVE", "REJECT"}
            or not isinstance(findings, list)
            or strength not in {"exploratory", "locally_supported", "unsupported"}
        ):
            raise ValueError("verifier returned an invalid review object")
        return review

    def _candidate_claims(
        self, state: dict[str, Any], action: dict[str, Any]
    ) -> None:
        for raw in action.get("candidate_claims", []):
            candidate: dict[str, Any] = {
                "id": raw.get("id") if isinstance(raw, dict) else None,
                "node_id": raw.get("node_id") if isinstance(raw, dict) else None,
                "admission": (
                    raw.get("admission", "claim")
                    if isinstance(raw, dict)
                    else "claim"
                ),
                "statement": raw.get("statement") if isinstance(raw, dict) else None,
                "artifact": raw.get("artifact") if isinstance(raw, dict) else None,
                "supporting_artifacts": [],
                "checks": [],
                "status": "candidate",
                "created_at": now(),
            }
            try:
                evidence_id = validate_identifier(candidate["id"])
                node_id = validate_identifier(candidate["node_id"])
                existing = state["evidence"].get(evidence_id)
                if existing and existing.get("status") == "verified":
                    raise ValueError(
                        "verified evidence is immutable; revise the problem node "
                        "or use a new evidence id"
                    )
                if node_id not in state["nodes"]:
                    raise ValueError(f"candidate references unknown node: {node_id}")
                if candidate["admission"] not in {"working", "claim"}:
                    requested_admission = candidate["admission"]
                    candidate["requested_admission"] = requested_admission
                    candidate["admission"] = "claim"
                    raise ValueError(
                        f"invalid evidence admission: {requested_admission}"
                    )
                if not isinstance(candidate["statement"], str) or not candidate[
                    "statement"
                ].strip():
                    raise ValueError("candidate statement must not be empty")
                artifact = safe_path(self.workspace, candidate["artifact"])
                if not artifact.is_file() or artifact.stat().st_size == 0:
                    raise ValueError("candidate artifact is missing or empty")
                raw_supporting = raw.get("supporting_artifacts", [])
                if (
                    not isinstance(raw_supporting, list)
                    or len(raw_supporting) > 12
                    or not all(isinstance(item, str) for item in raw_supporting)
                ):
                    raise ValueError(
                        "supporting_artifacts must be an array of at most 12 paths"
                    )
                raw_checks = raw.get("checks", [])
                if not isinstance(raw_checks, list) or not raw_checks:
                    raise ValueError("candidate evidence requires at least one check")
                check_records = []
                check_arguments = []
                for check in raw_checks:
                    arguments = _parse_json_object(
                        check.get("arguments_json"), "check arguments"
                    )
                    check_arguments.append(arguments)
                    record = {
                        **run_check(self.workspace, check.get("kind"), arguments),
                        "arguments": arguments,
                    }
                    check_records.append(record)
                candidate["checks"] = check_records
                candidate["artifact_sha256"] = file_hash(artifact)
                if not all(item.get("ok") is True for item in check_records):
                    raise ValueError("one or more mechanical checks failed")
                artifact_bundle = self._artifact_bundle(
                    candidate["artifact"], raw_supporting, check_arguments
                )
                candidate["supporting_artifacts"] = [
                    {
                        "path": item["path"],
                        "sha256": item["sha256"],
                        "bytes": item["bytes"],
                    }
                    for item in artifact_bundle
                    if item["path"] != candidate["artifact"]
                ]
                candidate["checked_at"] = now()
                if candidate["admission"] == "claim":
                    dependency_ids = set(state["nodes"][node_id]["depends_on"])
                    prerequisite_evidence = {
                        evidence_id: {
                            "node_id": record["node_id"],
                            "statement": record["statement"],
                            "artifact": record["artifact"],
                            "artifact_sha256": record.get("artifact_sha256"),
                            "review": record.get("review"),
                        }
                        for evidence_id, record in state["evidence"].items()
                        if record.get("status") == "verified"
                        and record.get("node_id") in dependency_ids
                    }
                    review = self._review_candidate(
                        node=state["nodes"][node_id],
                        candidate=candidate,
                        artifact_bundle=artifact_bundle,
                        check_records=check_records,
                        prerequisite_evidence=prerequisite_evidence,
                    )
                    candidate["review"] = review
                    if (
                        review["verdict"] == "APPROVE"
                        and review["claim_strength"] == "locally_supported"
                    ):
                        candidate["status"] = "verified"
                        candidate["verified_at"] = now()
                    else:
                        candidate["status"] = "rejected"
                else:
                    candidate["review_deferred"] = True
            except (
                AttributeError,
                OSError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                RuntimeError,
                TimeoutError,
            ) as exc:
                candidate["status"] = "rejected"
                candidate["error"] = str(exc)
            previous = state["evidence"].get(candidate.get("id"))
            if previous:
                candidate["revision"] = int(previous.get("revision", 1)) + 1
                candidate["previous_hash"] = content_hash(previous)
            else:
                candidate["revision"] = 1
            if candidate.get("id"):
                state["evidence"][candidate["id"]] = candidate
            observation = {
                "kind": "evidence.result",
                "evidence_id": candidate.get("id"),
                "node_id": candidate.get("node_id"),
                "admission": candidate.get("admission"),
                "status": candidate["status"],
                "review_requested": candidate.get("admission") == "claim",
                "usable_for_exploration": candidate["status"] in {
                    "candidate",
                    "verified",
                },
                "error": candidate.get("error"),
                "review": candidate.get("review"),
            }
            state["observations"].append(observation)
            self.store.event("evidence.result", observation)

    def _review_final(
        self, state: dict[str, Any], final: dict[str, Any]
    ) -> dict[str, Any]:
        evidence = {}
        for key in final["evidence_ids"]:
            record = state["evidence"][key]
            supporting = [
                item["path"] for item in record.get("supporting_artifacts", [])
            ]
            check_arguments = [
                item["arguments"]
                for item in record.get("checks", [])
                if isinstance(item.get("arguments"), dict)
            ]
            bundle = self._artifact_bundle(
                record["artifact"], supporting, check_arguments
            )
            expected_hashes = {
                record["artifact"]: record.get("artifact_sha256"),
                **{
                    item["path"]: item.get("sha256")
                    for item in record.get("supporting_artifacts", [])
                },
            }
            changed = [
                item["path"]
                for item in bundle
                if expected_hashes.get(item["path"]) not in {None, item["sha256"]}
            ]
            if changed:
                raise ValueError(
                    f"verified evidence artifacts changed after review: {changed}"
                )
            evidence[key] = {
                "statement": record["statement"],
                "artifact_bundle": bundle,
                "mechanical_checks": record.get("checks", []),
                "candidate_review": record.get("review"),
            }
        prompt = (
            REVIEWER_INSTRUCTIONS
            + "\n\nFINAL ANSWER REVIEW PACKET\n"
            + json.dumps(
                {
                    "objective": state["objective"],
                    "answer": final["answer"],
                    "limitations": final["limitations"],
                    "verified_evidence": evidence,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return self.verifier.complete(
            prompt,
            REVIEW_SCHEMA,
            role="final-verifier",
            workspace=self.workspace,
        )

    def _capture_submission(
        self, state: dict[str, Any], final: Any
    ) -> bool:
        if not isinstance(final, dict):
            return False
        answer = final.get("answer")
        evidence_ids = final.get("evidence_ids")
        limitations = final.get("limitations")
        if (
            not isinstance(answer, str)
            or not answer.strip()
            or not isinstance(evidence_ids, list)
            or not all(isinstance(item, str) for item in evidence_ids)
            or not isinstance(limitations, list)
            or not all(isinstance(item, str) for item in limitations)
        ):
            return False
        submission = {
            "answer": answer,
            "evidence_ids": evidence_ids,
            "limitations": limitations,
            "captured_at": now(),
        }
        state["submission"] = submission
        projected = delivery_projection(state, self.workspace)
        if projected is not None:
            self._write_delivery_projection(projected)
        self.store.event(
            "final.proposed",
            {
                "submission_hash": content_hash(submission),
                "evidence_ids": evidence_ids,
                "claim_ceiling": "exploratory",
            },
        )
        return True

    def _write_delivery_projection(self, delivery: dict[str, Any]) -> None:
        status_path = safe_path(self.workspace, "paper/delivery.md")
        final_path = safe_path(self.workspace, "paper/final.md")
        status_path.parent.mkdir(parents=True, exist_ok=True)
        evidence = "\n".join(
            f"- `{item}`: {delivery.get('evidence_status', {}).get(item, 'missing')}"
            for item in delivery["evidence_ids"]
        ) or "- None"
        limitations = "\n".join(
            f"- {item}" for item in delivery["limitations"]
        ) or "- None stated"
        qualification = delivery.get("qualification_error")
        qualification_section = (
            f"\n\n## Qualification boundary\n\n{qualification}" if qualification else ""
        )
        text = (
            "<!-- generated-by: thin-harness-delivery -->\n"
            "# Modeling result\n\n"
            f"- Status: `{delivery['status']}`\n"
            f"- Claim ceiling: `{delivery['claim_ceiling']}`\n\n"
            "## Answer\n\n"
            f"{delivery['answer'].strip()}\n\n"
            "## Evidence\n\n"
            f"{evidence}\n\n"
            "## Limitations\n\n"
            f"{limitations}"
            f"{qualification_section}\n"
        )
        status_path.write_text(text, encoding="utf-8", newline="\n")
        existing = ""
        if final_path.is_file():
            try:
                existing = final_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                existing = ""
        if not existing or existing.startswith(
            "<!-- generated-by: thin-harness-delivery -->"
        ):
            final_path.write_text(text, encoding="utf-8", newline="\n")

    @staticmethod
    def _root_ready_for_final(state: dict[str, Any]) -> bool:
        root = state["nodes"]["root"]
        states = node_states(state)
        if states["root"] == "supported":
            return True
        dependencies = root.get("depends_on", [])
        return bool(dependencies) and all(
            states[item] == "supported"
            for item in dependencies
        )

    def _finalize(self, state: dict[str, Any], action: dict[str, Any]) -> bool:
        final = action.get("final")
        if final is None:
            return False
        self._capture_submission(state, final)
        try:
            if not isinstance(final, dict):
                raise ValueError("final must be an object or null")
            if not self._root_ready_for_final(state):
                raise ValueError(
                    "root problem and its declared prerequisites are not supported"
                )
            evidence_ids = final.get("evidence_ids")
            if not isinstance(evidence_ids, list) or not evidence_ids:
                raise ValueError("final answer must cite verified evidence")
            bad = [
                item
                for item in evidence_ids
                if state["evidence"].get(item, {}).get("status") != "verified"
            ]
            if bad:
                raise ValueError(f"final cites unverified evidence: {bad}")
            if not isinstance(final.get("answer"), str) or not final["answer"].strip():
                raise ValueError("final answer is empty")
            if not isinstance(final.get("limitations"), list):
                raise ValueError("final limitations must be a list")
            review = self._review_final(state, final)
            self._record_model_receipt(state, "verifier")
            if (
                review.get("verdict") != "APPROVE"
                or review.get("claim_strength") != "locally_supported"
            ):
                raise ValueError(f"final verifier rejected answer: {review}")
            state["status"] = "completed"
            state["final"] = {
                **final,
                "review": review,
                "completed_at": now(),
            }
            projected = delivery_projection(state, self.workspace)
            if projected is not None:
                self._write_delivery_projection(projected)
            self.store.event(
                "run.completed",
                {"evidence_ids": evidence_ids, "review": review},
            )
            return True
        except (TypeError, ValueError, RuntimeError, TimeoutError) as exc:
            if isinstance(state.get("submission"), dict):
                state["submission"]["qualification_error"] = str(exc)
            observation = {"kind": "final.rejected", "error": str(exc)}
            state["observations"].append(observation)
            self.store.event("final.rejected", observation)
            projected = delivery_projection(state, self.workspace)
            if projected is not None:
                self._write_delivery_projection(projected)
            return False

    def run(self, objective: str | None = None) -> RunResult:
        if self.store.exists():
            state = self.store.load()
            if objective and objective.strip() != state["objective"]:
                raise ValueError("objective differs from the existing run")
            if state["status"] == "completed":
                return RunResult(
                    "completed", "already_completed", self.workspace, state
                )
            if state["status"] == "stopped":
                state["status"] = "running"
                state.pop("stop_reason", None)
                state["budgets"]["max_steps"] = max(
                    state["budgets"]["max_steps"], self.max_steps
                )
                state["budgets"]["max_tool_calls"] = max(
                    state["budgets"]["max_tool_calls"], self.max_tool_calls
                )
                state["budgets"]["max_seconds"] = max(
                    state["budgets"]["max_seconds"], self.max_seconds
                )
                self.store.event("run.resumed", {"step": state["step"]})
        elif objective:
            state = self.initialize(objective)
        else:
            raise ValueError("objective is required for a new run")
        started = time.monotonic()
        while state["status"] == "running":
            elapsed = time.monotonic() - started
            if state["step"] >= state["budgets"]["max_steps"]:
                reason = "step_budget_reached"
                break
            if elapsed >= state["budgets"]["max_seconds"]:
                reason = "wall_time_budget_reached"
                break
            state["step"] += 1
            try:
                turns_left = (
                    state["budgets"]["max_steps"] - state["step"] + 1
                )
                schema = (
                    SYNTHESIS_ACTION_SCHEMA
                    if turns_left == 1
                    else ACTION_SCHEMA
                )
                action = self.modeler.complete(
                    self._model_prompt(state),
                    schema,
                    role="modeler",
                    workspace=self.workspace,
                )
                self._record_model_receipt(state, "modeler")
            except (RuntimeError, TimeoutError, OSError) as exc:
                observation = {"kind": "model.error", "error": str(exc)}
                state["observations"].append(observation)
                self.store.event("model.error", observation)
                reason = "model_error"
                break
            summary = action.get("summary")
            self.store.event(
                "model.action",
                {
                    "step": state["step"],
                    "summary": summary if isinstance(summary, str) else "",
                    "action_hash": content_hash(action),
                    "action": action,
                },
            )
            self._apply_nodes(state, action)
            self._execute_tools(state, action)
            self._candidate_claims(state, action)
            if self._finalize(state, action):
                self.store.save(state)
                return RunResult("completed", "verified_final", self.workspace, state)
            self.store.save(state)
        else:
            reason = "completed"
        if state["status"] != "completed":
            state["status"] = "stopped"
            projected = delivery_projection(state, self.workspace)
            if projected and projected["status"] == "best_effort_unverified":
                reason = f"{reason}_with_best_effort_delivery"
            state["stop_reason"] = reason
            self.store.event("run.stopped", {"reason": reason})
            self.store.save(state)
        return RunResult(state["status"], reason, self.workspace, state)
