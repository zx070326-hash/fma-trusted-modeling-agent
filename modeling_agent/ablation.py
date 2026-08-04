"""Frozen equal-budget component ablations for the single THIN engine."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from .storage import atomic_write_json, content_hash, now


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
                "id": "codex_web",
                "description": "Raw Codex plus native public-web research; no source admission.",
            },
            {
                "id": "source_gate",
                "description": (
                    "Codex web research plus exact-URL source review and frozen source records."
                ),
            },
            {
                "id": "hard_eval",
                "description": (
                    "Source Gate plus isolated replay, claim obligations, mechanical veto, "
                    "and one fresh read-only verifier."
                ),
            },
            {
                "id": "elastic_memory",
                "description": (
                    "Full engine with working research memory and bounded on-demand branches."
                ),
            },
        ],
        "frozen_outcomes": [
            "task_solution_quality",
            "simple_baseline_delta",
            "source_entailment",
            "critical_claim_coverage",
            "falsification_quality",
            "evidence_reproducibility",
            "false_success_rate",
            "correct_abstention_rate",
            "route_change_quality",
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
        "contamination_rule": (
            "Primary evaluation tasks must remain private and unseen until contracts, "
            "budgets, graders, stop rules, and network policies are frozen. Public-answer "
            "retrieval invalidates a capability comparison."
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
