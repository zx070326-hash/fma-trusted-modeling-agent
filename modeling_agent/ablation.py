"""Frozen equal-budget ablation manifests for raw, thin, and native sidecar."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from .core import atomic_write_json, content_hash, now


def freeze_ablation(
    destination: Path,
    *,
    objective: str,
    model: str,
    max_model_turns: int,
    max_tool_calls: int,
    max_wall_seconds: int,
) -> dict[str, Any]:
    if not objective.strip():
        raise ValueError("objective must not be empty")
    if min(max_model_turns, max_tool_calls, max_wall_seconds) <= 0:
        raise ValueError("ablation budgets must be positive")
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"ablation manifest already exists: {destination}")
    stable = {
        "objective": objective.strip(),
        "model_requested": model,
        "budget": {
            "max_model_turns": max_model_turns,
            "max_tool_calls": max_tool_calls,
            "max_wall_seconds": max_wall_seconds,
        },
        "arms": [
            {
                "id": "raw_codex",
                "description": "One fresh model response; no harness loop or evidence promotion.",
            },
            {
                "id": "thin_harness",
                "description": (
                    "Open problem graph, local tools, fail-closed checks, and fresh review."
                ),
            },
            {
                "id": "native_sidecar",
                "description": (
                    "Native project-local Codex research with a frozen task contract, "
                    "deterministic replay, and one fresh final verifier."
                ),
            },
        ],
        "frozen_outcomes": [
            "task_solution_quality",
            "simple_baseline_delta",
            "falsification_quality",
            "evidence_reproducibility",
            "recovery_after_rejection",
            "human_interventions",
            "wall_time_seconds",
            "model_turns",
            "tool_calls",
        ],
        "grading_rule": (
            "Operational metrics may be computed by the harness. Scientific quality "
            "must be scored by a task-specific frozen grader or independent reviewer. "
            "A workflow-complete flag is not a quality score."
        ),
    }
    manifest = {
        "schema": 2,
        "experiment_id": uuid.uuid4().hex,
        "created_at": now(),
        **stable,
        "contract_hash": content_hash(stable),
        "results": {},
    }
    atomic_write_json(destination, manifest)
    return manifest


def record_result(
    manifest_path: Path, arm_id: str, result: dict[str, Any]
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    known = {item["id"] for item in manifest["arms"]}
    if arm_id not in known:
        raise ValueError(f"unknown ablation arm: {arm_id}")
    if arm_id in manifest["results"]:
        raise ValueError(f"ablation arm is already recorded: {arm_id}")
    manifest["results"][arm_id] = {
        "recorded_at": now(),
        "result": result,
        "result_hash": content_hash(result),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest
