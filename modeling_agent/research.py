"""Model-authored working memory and bounded elastic branch packets."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from .storage import RunLayout, append_jsonl, atomic_write_json, now, read_jsonl, safe_path


RECORD_KINDS = {
    "question",
    "hypothesis",
    "attempt",
    "observation",
    "counterexample",
    "dead_end",
    "decision",
    "branch_summary",
}
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")


class ResearchStore:
    """Working knowledge is useful context, never admitted claim evidence."""

    def __init__(self, path: Path):
        self.path = path

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(record, dict) or record.get("kind") not in RECORD_KINDS:
            raise ValueError("invalid research record kind")
        statement = record.get("statement")
        if not isinstance(statement, str) or not statement.strip():
            raise ValueError("research record statement must not be empty")
        normalized = {
            **record,
            "id": record.get("id") or f"r-{uuid.uuid4().hex[:12]}",
            "statement": statement.strip(),
            "recorded_at": record.get("recorded_at") or now(),
            "authority": "working",
        }
        append_jsonl(self.path, normalized)
        return normalized

    def records(self) -> list[dict[str, Any]]:
        records = read_jsonl(self.path)
        for index, record in enumerate(records, 1):
            if record.get("kind") not in RECORD_KINDS:
                raise ValueError(f"invalid research record kind at line {index}")
            if not isinstance(record.get("statement"), str) or not record["statement"].strip():
                raise ValueError(f"empty research record statement at line {index}")
        return records

    def context(self, *, max_chars: int = 48_000) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        remaining = max_chars
        for record in reversed(self.records()):
            encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            if len(encoded) > remaining:
                continue
            selected.append(record)
            remaining -= len(encoded)
        return list(reversed(selected))

    def project_graph(self) -> dict[str, Any]:
        nodes: dict[str, dict[str, Any]] = {}
        for record in self.records():
            identifier = str(record.get("node_id") or record["id"])
            nodes[identifier] = {
                "id": identifier,
                "kind": record["kind"],
                "statement": record["statement"],
                "depends_on": record.get("depends_on", []),
                "authority": "working",
            }
        dangling = sorted(
            {
                dependency
                for node in nodes.values()
                for dependency in node["depends_on"]
                if dependency not in nodes
            }
        )
        return {
            "nodes": nodes,
            "derived": True,
            "dangling_dependencies": dangling,
        }


def load_branch_requests(layout: RunLayout, *, max_branches: int) -> list[dict[str, str]]:
    path = layout.work / "research" / "branch_requests.json"
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != 1:
        raise ValueError("branch request schema must be 1")
    requests = value.get("requests")
    if not isinstance(requests, list) or len(requests) > max_branches:
        raise ValueError(f"branch requests must contain at most {max_branches} items")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(requests):
        if not isinstance(item, dict):
            raise ValueError(f"branch request {index} must be an object")
        identifier = item.get("id")
        question = item.get("question")
        purpose = item.get("purpose")
        if not isinstance(identifier, str) or not _IDENTIFIER.fullmatch(identifier):
            raise ValueError(f"invalid branch id: {identifier}")
        if identifier in seen:
            raise ValueError(f"duplicate branch id: {identifier}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"branch {identifier} question must not be empty")
        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError(f"branch {identifier} purpose must not be empty")
        seen.add(identifier)
        normalized.append(
            {"id": identifier, "question": question.strip(), "purpose": purpose.strip()}
        )
    return normalized


def prepare_branch(
    layout: RunLayout,
    request: dict[str, str],
    *,
    objective: str,
    wave: int,
    context: list[dict[str, Any]],
) -> Path:
    branch_root = safe_path(layout.branches, request["id"])
    work = branch_root / "work"
    work.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        work / "branch_packet.json",
        {
            "schema": 1,
            "wave": wave,
            "objective": objective,
            "branch": request,
            "published_working_context": context,
            "authority_boundary": (
                "This packet is working knowledge. Return a falsifiable branch summary; "
                "do not claim evidence admission."
            ),
        },
    )
    return work


def load_branch_summary(branch_work: Path, branch_id: str) -> dict[str, Any]:
    path = branch_work / "branch_summary.json"
    if not path.is_file():
        raise ValueError(f"branch {branch_id} did not produce branch_summary.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != 1:
        raise ValueError(f"branch {branch_id} summary schema must be 1")
    status = value.get("status")
    if status not in {"supported", "challenged", "failed", "inconclusive"}:
        raise ValueError(f"branch {branch_id} has invalid status")
    conclusion = value.get("conclusion")
    if not isinstance(conclusion, str) or not conclusion.strip():
        raise ValueError(f"branch {branch_id} conclusion must not be empty")
    return {
        "branch_id": branch_id,
        "status": status,
        "conclusion": conclusion.strip(),
        "observations": value.get("observations", []),
        "falsifiers": value.get("falsifiers", []),
        "recommended_action": value.get("recommended_action", ""),
    }


def publish_branch_summary(layout: RunLayout, summary: dict[str, Any]) -> dict[str, Any]:
    published = layout.work / "research" / "published"
    published.mkdir(parents=True, exist_ok=True)
    atomic_write_json(published / f"{summary['branch_id']}.json", summary)
    return ResearchStore(layout.research_path).append(
        {
            "kind": "branch_summary",
            "node_id": summary["branch_id"],
            "statement": summary["conclusion"],
            "status": summary["status"],
            "falsifiers": summary.get("falsifiers", []),
            "recommended_action": summary.get("recommended_action", ""),
        }
    )
