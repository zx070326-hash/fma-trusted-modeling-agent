"""Small durable state and graph primitives for the modeling loop."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


STATE_DIRECTORY = ".modeling-agent"
STATE_FILE = "state.json"
TRACE_FILE = "trace.jsonl"
NODE_STATES = {"open", "active", "supported", "blocked"}
EVIDENCE_STATES = {"candidate", "verified", "rejected", "revoked"}
EVIDENCE_ADMISSIONS = {"working", "claim"}


def now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    payload = value if isinstance(value, bytes) else canonical_json(value).encode()
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(workspace: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError("path must be a non-empty string")
    value = Path(relative.replace("\\", "/"))
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"path escapes workspace: {relative}")
    resolved = (workspace / value).resolve()
    if not resolved.is_relative_to(workspace.resolve()):
        raise ValueError(f"path escapes workspace: {relative}")
    return resolved


def evidence_integrity(workspace: Path, record: dict[str, Any]) -> str:
    expected = [(record.get("artifact"), record.get("artifact_sha256"))]
    expected.extend(
        (item.get("path"), item.get("sha256"))
        for item in record.get("supporting_artifacts", [])
        if isinstance(item, dict)
    )
    try:
        for relative, digest in expected:
            if not isinstance(relative, str) or not isinstance(digest, str):
                return "unknown"
            path = safe_path(workspace, relative)
            if not path.is_file() or file_hash(path) != digest:
                return "stale"
    except (OSError, TypeError, ValueError):
        return "stale"
    return "current"


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_node_id(node_id: Any) -> str:
    if not isinstance(node_id, str) or not node_id:
        raise ValueError("node id must be a non-empty string")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-")
    if len(node_id) > 96 or node_id[0].isdigit() or any(x not in allowed for x in node_id):
        raise ValueError(f"invalid node id: {node_id}")
    return node_id


def validate_identifier(value: Any) -> str:
    return _validate_node_id(value)


def _assert_acyclic(nodes: dict[str, dict[str, Any]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ValueError(f"problem graph cycle at {node_id}")
        if node_id in visited:
            return
        visiting.add(node_id)
        for dependency in nodes[node_id]["depends_on"]:
            if dependency not in nodes:
                raise ValueError(f"unknown dependency: {node_id} -> {dependency}")
            visit(dependency)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in nodes:
        visit(node_id)


def new_state(
    objective: str,
    *,
    max_steps: int,
    max_tool_calls: int,
    max_seconds: int,
) -> dict[str, Any]:
    if not objective.strip():
        raise ValueError("objective must not be empty")
    if min(max_steps, max_tool_calls, max_seconds) <= 0:
        raise ValueError("budgets must be positive")
    created = now()
    return {
        "schema": 1,
        "run_id": uuid.uuid4().hex,
        "objective": objective.strip(),
        "status": "running",
        "created_at": created,
        "updated_at": created,
        "step": 0,
        "tool_calls": 0,
        "budgets": {
            "max_steps": max_steps,
            "max_tool_calls": max_tool_calls,
            "max_seconds": max_seconds,
        },
        "nodes": {
            "root": {
                "id": "root",
                "question": objective.strip(),
                "depends_on": [],
                "priority": 1.0,
                "revision": 1,
            }
        },
        "evidence": {},
        "observations": [],
        "submission": None,
        "final": None,
    }


def validate_state(state: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(state, dict) or state.get("schema") != 1:
        raise ValueError("state schema must be 1")
    if state.get("status") not in {"running", "completed", "stopped"}:
        raise ValueError("invalid run status")
    nodes = state.get("nodes")
    if not isinstance(nodes, dict) or "root" not in nodes:
        raise ValueError("state must contain a root problem node")
    for key, node in nodes.items():
        if _validate_node_id(key) != node.get("id"):
            raise ValueError(f"node key/id mismatch: {key}")
        if not isinstance(node.get("question"), str) or not node["question"].strip():
            raise ValueError(f"node question is empty: {key}")
        if not isinstance(node.get("depends_on"), list):
            raise ValueError(f"node dependencies must be a list: {key}")
        if "state" in node and node.get("state") not in NODE_STATES:
            raise ValueError(f"invalid node state: {key}")
    _assert_acyclic(nodes)
    evidence = state.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("state evidence must be an object")
    for evidence_id, record in evidence.items():
        _validate_node_id(evidence_id)
        if record.get("status") not in EVIDENCE_STATES:
            raise ValueError(f"invalid evidence state: {evidence_id}")
        if record.get("admission", "claim") not in EVIDENCE_ADMISSIONS:
            raise ValueError(f"invalid evidence admission: {evidence_id}")
        if record.get("node_id") not in nodes:
            raise ValueError(f"evidence points to unknown node: {evidence_id}")
    delivery = state.get("delivery")
    if (
        "delivery" in state
        and delivery is not None
        and not isinstance(delivery, dict)
    ):
        raise ValueError("state delivery must be an object or null")
    if state.get("submission") is not None and not isinstance(
        state["submission"], dict
    ):
        raise ValueError("state submission must be an object or null")
    return state


def upsert_nodes(
    state: dict[str, Any], proposals: list[dict[str, Any]]
) -> list[str]:
    if not isinstance(proposals, list):
        raise ValueError("upsert_nodes must be a list")
    candidate = json.loads(canonical_json(state["nodes"]))
    changed: list[str] = []
    for proposal in proposals:
        if not isinstance(proposal, dict):
            raise ValueError("node proposal must be an object")
        node_id = _validate_node_id(proposal.get("id"))
        question = proposal.get("question")
        dependencies = proposal.get("depends_on", [])
        priority = proposal.get("priority", 1.0)
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"node question is empty: {node_id}")
        if not isinstance(dependencies, list) or not all(
            isinstance(item, str) for item in dependencies
        ):
            raise ValueError(f"invalid dependencies: {node_id}")
        if (
            not isinstance(priority, (int, float))
            or isinstance(priority, bool)
            or priority <= 0
        ):
            raise ValueError(f"priority must be positive: {node_id}")
        previous = candidate.get(node_id)
        semantic = {
            "question": question.strip(),
            "depends_on": list(dict.fromkeys(dependencies)),
            "priority": float(priority),
        }
        if previous and all(previous.get(key) == value for key, value in semantic.items()):
            continue
        candidate[node_id] = {
            "id": node_id,
            **semantic,
            "revision": int(previous.get("revision", 0)) + 1 if previous else 1,
        }
        changed.append(node_id)
    _assert_acyclic(candidate)
    state["nodes"] = candidate
    if changed:
        affected = set(changed)
        grew = True
        while grew:
            grew = False
            for node_id, node in state["nodes"].items():
                if node_id not in affected and any(
                    dependency in affected for dependency in node["depends_on"]
                ):
                    affected.add(node_id)
                    grew = True
        for record in state["evidence"].values():
            if (
                record["node_id"] in affected
                and record["status"] in {"candidate", "verified"}
            ):
                record["status"] = "revoked"
                record["revoked_at"] = now()
                record["revocation_reason"] = "problem graph dependency changed"
    return changed


def node_states(state: dict[str, Any]) -> dict[str, str]:
    """Project graph support from evidence; never store it as a second truth."""

    nodes = state["nodes"]
    verified_nodes = {
        record["node_id"]
        for record in state["evidence"].values()
        if record.get("status") == "verified"
    }
    support_cache: dict[str, bool] = {}

    def supported(node_id: str) -> bool:
        if node_id in support_cache:
            return support_cache[node_id]
        dependencies = nodes[node_id]["depends_on"]
        dependencies_supported = all(supported(item) for item in dependencies)
        value = (
            node_id in verified_nodes and dependencies_supported
        ) or (
            node_id == "root"
            and bool(dependencies)
            and dependencies_supported
        )
        support_cache[node_id] = value
        return value

    projected: dict[str, str] = {}
    for node_id, node in nodes.items():
        if supported(node_id):
            projected[node_id] = "supported"
        elif all(supported(item) for item in node["depends_on"]):
            projected[node_id] = "open"
        else:
            projected[node_id] = "blocked"
    return projected


def frontier(state: dict[str, Any]) -> list[dict[str, Any]]:
    states = node_states(state)
    ready = [
        {**node, "state": states[node_id]}
        for node_id, node in state["nodes"].items()
        if states[node_id] == "open"
    ]
    return sorted(ready, key=lambda item: (-item["priority"], item["id"]))


def delivery_projection(
    state: dict[str, Any], workspace: Path | None = None
) -> dict[str, Any] | None:
    """Render the latest answer from submission and admitted evidence."""

    final = state.get("final")
    if isinstance(final, dict):
        proposal = final
        captured_at = final.get("completed_at")
        qualification_error = None
    else:
        proposal = state.get("submission")
        if not isinstance(proposal, dict):
            return None
        captured_at = proposal.get("captured_at")
        qualification_error = proposal.get("qualification_error")

    evidence_ids = proposal.get("evidence_ids", [])
    statuses = {
        item: state["evidence"].get(item, {}).get("status", "missing")
        for item in evidence_ids
        if isinstance(item, str)
    }
    integrity = (
        {
            item: evidence_integrity(workspace, state["evidence"][item])
            for item in evidence_ids
            if item in state["evidence"]
        }
        if workspace is not None
        else {}
    )
    review = final.get("review") if isinstance(final, dict) else None
    qualified = bool(
        final
        and isinstance(review, dict)
        and review.get("verdict") == "APPROVE"
        and review.get("claim_strength") == "locally_supported"
        and statuses
        and all(status == "verified" for status in statuses.values())
        and (not integrity or all(value == "current" for value in integrity.values()))
    )
    if final and not qualified:
        qualification_error = "accepted answer no longer has current admitted evidence"
    return {
        "answer": proposal.get("answer", ""),
        "evidence_ids": evidence_ids,
        "evidence_status": statuses,
        **({"evidence_integrity": integrity} if integrity else {}),
        "limitations": proposal.get("limitations", []),
        "status": (
            "verified"
            if qualified
            else "best_effort_unverified"
            if qualification_error
            else "proposed"
        ),
        "claim_ceiling": "locally_supported" if qualified else "exploratory",
        "captured_at": captured_at,
        **({"review": review} if review else {}),
        **(
            {"qualification_error": qualification_error}
            if qualification_error
            else {}
        ),
    }


def state_planes(
    state: dict[str, Any], workspace: Path | None = None
) -> dict[str, Any]:
    """Expose three authorities; workflow and delivery are projections only."""

    states = node_states(state)
    return {
        "research": {
            "objective": state["objective"],
            "nodes": {
                node_id: {
                    **{
                        key: value
                        for key, value in node.items()
                        if key not in {"state", "attempts"}
                    },
                    "support": states[node_id],
                }
                for node_id, node in state["nodes"].items()
            },
            "frontier": [node["id"] for node in frontier(state)],
        },
        "execution": {
            "status": state["status"],
            "step": state["step"],
            "tool_calls": state["tool_calls"],
            "budgets": state["budgets"],
            "stop_reason": state.get("stop_reason"),
            "has_submission": isinstance(state.get("submission"), dict),
        },
        "evidence": {
            "counts": {
                name: sum(
                    1
                    for record in state["evidence"].values()
                    if record["status"] == name
                )
                for name in ("candidate", "verified", "rejected", "revoked")
            },
            "working_candidates": sum(
                1
                for record in state["evidence"].values()
                if record["status"] == "candidate"
                and record.get("admission", "claim") == "working"
            ),
            **(
                {
                    "integrity": {
                        evidence_id: evidence_integrity(workspace, record)
                        for evidence_id, record in state["evidence"].items()
                        if record["status"] == "verified"
                    }
                }
                if workspace is not None
                else {}
            ),
        },
    }


def normalize_state(state: dict[str, Any]) -> dict[str, Any]:
    """Drop legacy caches; retain only the answer submission they represented."""

    for node in state["nodes"].values():
        node.pop("state", None)
        node.pop("attempts", None)
    for record in state["evidence"].values():
        record.setdefault("admission", "claim")
    legacy = state.pop("delivery", None)
    if (
        isinstance(legacy, dict)
        and not state.get("final")
        and not state.get("submission")
    ):
        state["submission"] = {
            "answer": legacy.get("answer", ""),
            "evidence_ids": legacy.get("evidence_ids", []),
            "limitations": legacy.get("limitations", []),
            "captured_at": legacy.get("captured_at"),
            **(
                {"qualification_error": legacy["qualification_error"]}
                if legacy.get("qualification_error")
                else {}
            ),
        }
    state.setdefault("submission", None)
    return state


class StateStore:
    """One materialized state file plus one append-only operational trace."""

    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.directory = self.workspace / STATE_DIRECTORY
        self.state_path = self.directory / STATE_FILE
        self.trace_path = self.directory / TRACE_FILE

    def exists(self) -> bool:
        return self.state_path.is_file()

    def load(self) -> dict[str, Any]:
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read modeling state: {exc}") from exc
        return validate_state(normalize_state(validate_state(state)))

    def save(self, state: dict[str, Any]) -> None:
        state["updated_at"] = now()
        atomic_write_json(
            self.state_path, validate_state(normalize_state(state))
        )

    def initialize(self, state: dict[str, Any]) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        if self.exists():
            raise FileExistsError(f"modeling state already exists: {self.state_path}")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.save(state)
        self.event("run.started", {"run_id": state["run_id"]})

    def event(self, kind: str, payload: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        record = {"time": now(), "kind": kind, "payload": payload}
        with self.trace_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(record) + "\n")
