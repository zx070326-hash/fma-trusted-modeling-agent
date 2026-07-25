"""Command-line facade for the V5 task-workspace protocol.

Authority-bearing commands require a key held outside the task workspace.
Set ``FMA_V5_AUTHORITY_KEY_FILE`` or ``FMA_V5_AUTHORITY_KEY_HEX``, or pass
``--key-file``.  The CLI never creates a default secret inside the workspace.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .paper import build_paper
from .scaffold import scaffold_task_workspace, validate_task_scaffold
from .stage_workspace import STAGES, StageWorkspaceV50
from .workspace_schemas import TaskWorkspaceSpecV50, WorkflowProfileV50


def _json(payload: object) -> None:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")  # type: ignore[union-attr]
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
            default=str,
        )
    )


def _decode_key(payload: bytes) -> bytes:
    stripped = payload.strip()
    if len(stripped) == 64:
        try:
            return bytes.fromhex(stripped.decode("ascii"))
        except (UnicodeDecodeError, ValueError):
            pass
    return stripped


def _authority(args: argparse.Namespace) -> tuple[bytes, str]:
    key_file = getattr(args, "key_file", None) or os.environ.get(
        "FMA_V5_AUTHORITY_KEY_FILE"
    )
    key_hex = os.environ.get("FMA_V5_AUTHORITY_KEY_HEX")
    if key_file:
        path = Path(key_file).expanduser().resolve()
        key = _decode_key(path.read_bytes())
    elif key_hex:
        try:
            key = bytes.fromhex(key_hex)
        except ValueError as exc:
            raise ValueError("FMA_V5_AUTHORITY_KEY_HEX is not valid hex") from exc
    else:
        raise RuntimeError(
            "an external authority key is required; set "
            "FMA_V5_AUTHORITY_KEY_FILE or pass --key-file"
        )
    if len(key) < 32:
        raise ValueError("authority key must contain at least 32 bytes")
    key_id = getattr(args, "key_id", None) or os.environ.get(
        "FMA_V5_AUTHORITY_KEY_ID", "external-harness-v1"
    )
    return key, key_id


def _workspace(args: argparse.Namespace) -> StageWorkspaceV50:
    key, key_id = _authority(args)
    return StageWorkspaceV50.open_existing(
        Path(args.workspace),
        authority_key=key,
        authority_key_id=key_id,
    )


def _stage(value: str) -> str:
    normalized = value.upper()
    if normalized not in STAGES:
        raise argparse.ArgumentTypeError("stage must be S0 through S6")
    return normalized


def _add_authority_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--key-file")
    parser.add_argument("--key-id")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fma.v5",
        description=(
            "Graph-native S0-S6 workflow facade. Stage gates are workflow "
            "evidence only, never scientific qualification."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="scaffold and start a new V5 task graph")
    init.add_argument("--workspace", required=True)
    init.add_argument("--workspace-id", required=True)
    init.add_argument("--objective", required=True)
    init.add_argument("--mission-hash", required=True)
    init.add_argument("--evidence-snapshot-hash", required=True)
    init.add_argument("--evaluator-epoch", default="v5-evaluator-1")
    init.add_argument(
        "--evidence-scope",
        choices=(
            "structural",
            "synthetic_fixture",
            "development",
            "public_data",
            "private_holdout",
            "real_world",
        ),
        default="development",
    )
    _add_authority_flags(init)

    status = sub.add_parser("status", help="verify and project current graph state")
    status.add_argument("--workspace", required=True)
    _add_authority_flags(status)

    submit = sub.add_parser("submit", help="commit one stage artifact manifest")
    submit.add_argument("--workspace", required=True)
    submit.add_argument("--stage", required=True, type=_stage)
    submit.add_argument("--actor", choices=("model", "harness"), required=True)
    submit.add_argument("--extra-path", action="append", default=[])
    _add_authority_flags(submit)

    freeze_raw = sub.add_parser(
        "freeze-raw",
        help="harness-freeze data/raw for the current S2 attempt",
    )
    freeze_raw.add_argument("--workspace", required=True)
    _add_authority_flags(freeze_raw)

    checks = sub.add_parser(
        "checks", help="run structural checks (not scientific substitutes)"
    )
    checks.add_argument("--workspace", required=True)
    checks.add_argument("--stage", type=_stage)
    _add_authority_flags(checks)

    gate = sub.add_parser("gate", help="evaluate a stage gate fail-closed")
    gate.add_argument("--workspace", required=True)
    gate.add_argument("--stage", required=True, type=_stage)
    _add_authority_flags(gate)

    invalidate = sub.add_parser(
        "invalidate", help="revoke a stage and its exact downstream closure"
    )
    invalidate.add_argument("--workspace", required=True)
    invalidate.add_argument("--stage", required=True, type=_stage)
    invalidate.add_argument(
        "--reason", default="operator requested explicit stage rework"
    )
    _add_authority_flags(invalidate)

    paper = sub.add_parser(
        "paper", help="build a PDF from structured result placeholders"
    )
    paper.add_argument("--workspace", required=True)
    _add_authority_flags(paper)

    verify = sub.add_parser("verify", help="verify event, artifact, and auth chains")
    verify.add_argument("--workspace", required=True)
    _add_authority_flags(verify)
    return parser


def _run(args: argparse.Namespace) -> int:
    if args.command == "init":
        key, key_id = _authority(args)
        root = Path(args.workspace).resolve()
        if not root.exists() or not any(root.iterdir()):
            scaffold_task_workspace(root, args.workspace_id, args.objective)
        else:
            validate_task_scaffold(root, args.workspace_id, args.objective)
        profile = WorkflowProfileV50.seal()
        spec = TaskWorkspaceSpecV50.seal(
            workspace_id=args.workspace_id,
            graph_id=f"v5-{args.workspace_id}",
            objective=args.objective,
            mission_hash=args.mission_hash,
            evidence_snapshot_hash=args.evidence_snapshot_hash,
            evaluator_epoch=args.evaluator_epoch,
            profile=profile,
            evidence_scope=args.evidence_scope,
        )
        workspace = StageWorkspaceV50.create(
            root,
            spec,
            authority_key=key,
            authority_key_id=key_id,
        )
        _json(workspace.status())
        return 0
    if args.command == "paper":
        workspace = _workspace(args)
        if workspace.current_gate("S4") is None:
            raise RuntimeError("paper build is locked until the S4 gate is current")
        _json(build_paper(args.workspace))
        return 0

    workspace = _workspace(args)
    if args.command == "status":
        _json(workspace.status())
    elif args.command == "freeze-raw":
        _json(workspace.freeze_raw_inputs(actor="harness"))
    elif args.command == "submit":
        _json(
            workspace.submit_stage(
                args.stage, actor=args.actor, extra_paths=args.extra_path
            )
        )
    elif args.command == "checks":
        if args.stage:
            _json(workspace.run_mechanical_check(args.stage))
        else:
            status = workspace.status()
            runnable = [
                stage
                for stage, value in status.stage_statuses.items()
                if value == "awaiting_gate_evidence"
            ]
            _json([workspace.run_mechanical_check(stage) for stage in runnable])
    elif args.command == "gate":
        evaluation = workspace.evaluate_gate(args.stage)
        _json(evaluation)
        return 0 if evaluation.decision == "OPEN" else 3
    elif args.command == "invalidate":
        _json(
            {
                "stage": args.stage,
                "affected_node_hashes": workspace.invalidate_from(
                    args.stage, reason=args.reason
                ),
            }
        )
    elif args.command == "verify":
        ok = workspace.verify()
        _json(
            {
                "ok": ok,
                "scientific_qualification_granted": False,
                "real_world_action_authorized": False,
            }
        )
        return 0 if ok else 4
    return 0


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        return _run(args)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "scientific_qualification_granted": False,
                    "real_world_action_authorized": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
