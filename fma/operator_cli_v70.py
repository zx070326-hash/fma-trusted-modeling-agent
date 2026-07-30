"""Unified JSON CLI for the FMA V7.0 operator plane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fma.codex_driver import CodexCLIConfig
from fma.operator_v70 import OperatorStoreV70
from fma.studio.service import StudioTaskService
from fma.v5.__main__ import _decode_key


def _json(payload: object) -> None:
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


def _add_authority_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--authority-key-file", type=Path)
    parser.add_argument("--authority-key-id", default="studio-local-v1")


def _service(args: argparse.Namespace, *, required: bool) -> StudioTaskService | None:
    key_path = getattr(args, "authority_key_file", None)
    if key_path is None:
        if required:
            raise ValueError("--authority-key-file is required for this command")
        return None
    key = _decode_key(key_path.expanduser().resolve(strict=True).read_bytes())
    return StudioTaskService(
        args.task_root,
        authority_key=key,
        authority_key_id=args.authority_key_id,
        codex_config=CodexCLIConfig(),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fma-ops",
        description=(
            "Thin FMA operational control plane. Operator success never grants "
            "scientific qualification or real-world authority."
        ),
    )
    parser.add_argument("--task-root", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    intake = subparsers.add_parser(
        "intake",
        help="transactionally publish a content-addressed untrusted task intake",
    )
    intake.add_argument("--objective", required=True)
    intake.add_argument("--attachment", type=Path, action="append", default=[])
    intake.add_argument("--workspace-id")
    intake.add_argument(
        "--evidence-scope",
        choices=("development", "public_data"),
        default="development",
    )
    intake.add_argument(
        "--workflow-mode",
        choices=("legacy", "v67"),
        default="legacy",
    )
    intake.add_argument("--idempotency-key", required=True)
    intake.add_argument("--create-task", action="store_true")
    _add_authority_arguments(intake)

    status = subparsers.add_parser(
        "status",
        help="show operational state, or the combined authority snapshot with a key",
    )
    status.add_argument("--task-id")
    _add_authority_arguments(status)

    next_packet = subparsers.add_parser(
        "next",
        help="project the next read-only, graph-bound work packet",
    )
    next_packet.add_argument("--task-id", required=True)
    _add_authority_arguments(next_packet)

    doctor = subparsers.add_parser(
        "doctor",
        help="verify the whole operator ledger and, when keyed, every authority task",
    )
    _add_authority_arguments(doctor)

    reconcile = subparsers.add_parser(
        "reconcile",
        help=(
            "quarantine expired leases and reconcile only exact submitted "
            "authority effects"
        ),
    )
    _add_authority_arguments(reconcile)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        if args.command == "intake":
            service = _service(args, required=args.create_task)
            if service is None:
                store = OperatorStoreV70(args.task_root)
                manifest = store.publish_intake(
                    idempotency_key=args.idempotency_key,
                    objective=args.objective,
                    attachment_paths=args.attachment,
                    requested_workspace_id=args.workspace_id,
                    evidence_scope=args.evidence_scope,
                    workflow_mode=args.workflow_mode,
                )
                payload: dict[str, Any] = {
                    "status": "success",
                    "intake": manifest.model_dump(mode="json"),
                    "task": None,
                    "next_action": "create_task_from_intake",
                    "claim_scope": "workflow_control_only",
                    "scientific_qualification_granted": False,
                    "real_world_action_authorized": False,
                }
            else:
                payload = service.publish_intake_v70(
                    idempotency_key=args.idempotency_key,
                    objective=args.objective,
                    attachment_paths=args.attachment,
                    workspace_id=args.workspace_id,
                    evidence_scope=args.evidence_scope,
                    workflow_mode=args.workflow_mode,
                )
                payload["task"] = (
                    service.create_task_from_intake_v70(
                        payload["intake"]["intake_id"]
                    )
                    if args.create_task
                    else None
                )
            _json(payload)
            return 0

        if args.command == "status":
            service = _service(args, required=False)
            if service is not None:
                payload = (
                    service.snapshot(args.task_id)
                    if args.task_id
                    else service.list_tasks()
                )
            else:
                store = OperatorStoreV70(args.task_root)
                payload = (
                    store.operational_summary(args.task_id)
                    if args.task_id
                    else {
                        "status": "success",
                        "work_items": store.list_work(),
                        "authority_status": "NOT_RUN",
                        "scientific_qualification_granted": False,
                        "real_world_action_authorized": False,
                    }
                )
            _json(payload)
            return 0

        if args.command == "next":
            service = _service(args, required=True)
            assert service is not None
            _json(
                {
                    "status": "success",
                    "packet": service.project_next_packet_v70(args.task_id),
                    "scientific_qualification_granted": False,
                    "real_world_action_authorized": False,
                }
            )
            return 0

        if args.command == "doctor":
            service = _service(args, required=False)
            if service is not None:
                payload = service.operator_doctor_v70()
            else:
                operational = OperatorStoreV70(args.task_root).doctor()
                payload = {
                    "status": (
                        operational["status"]
                        if operational["status"] != "PASS"
                        else "NOT_RUN"
                    ),
                    "operational": operational,
                    "authority": {
                        "status": "NOT_RUN",
                        "reason": "authority key was not supplied",
                    },
                    "scientific_qualification_granted": False,
                    "real_world_action_authorized": False,
                }
            _json(payload)
            observed_statuses = {
                payload.get("status"),
                (payload.get("operational") or {}).get("status"),
            }
            return (
                1
                if observed_statuses.intersection({"FAIL", "RECOVERY_PENDING"})
                else 0
            )

        if args.command == "reconcile":
            service = _service(args, required=True)
            assert service is not None
            _json(service.reconcile_operator_v70())
            return 0
    except Exception as exc:
        _json(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "scientific_qualification_granted": False,
                "real_world_action_authorized": False,
            }
        )
        return 2
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
