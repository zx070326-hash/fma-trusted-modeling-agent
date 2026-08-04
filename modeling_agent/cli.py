"""Command line interface for THIN research and qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import __version__
from .ablation import freeze_ablation
from .engine import ModelingEngine, operator_prompt, run_status
from .model import CodexCLIModel
from .storage import RunLayout, atomic_write_json, file_hash, safe_path
from .verification import (
    build_external_review_packet,
    default_contract,
    load_contract,
    validate_contract,
    validate_external_review_bundle,
    validate_manifest,
)


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _add_model_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_model: str | None = "gpt-5.6-sol",
    include_timeout: bool = True,
) -> None:
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--codex-bin")
    if include_timeout:
        parser.add_argument("--model-timeout", type=int, default=300)


def _add_research_arguments(
    parser: argparse.ArgumentParser, *, include_verifier: bool
) -> None:
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--objective")
    parser.add_argument("--contract", type=Path)
    parser.add_argument(
        "--network",
        choices=("research-search", "offline-compute"),
        default="research-search",
    )
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--max-seconds", type=int)
    parser.add_argument("--max-branches", type=int, default=3)
    parser.add_argument("--max-waves", type=int, default=2)
    parser.add_argument(
        "--amend-budget",
        action="store_true",
        help="explicitly raise the frozen attempt/wall budget for an existing run",
    )
    parser.add_argument(
        "--producer-model",
        help="current Codex model label for provenance; defaults to unattested",
    )
    if include_verifier:
        _add_model_arguments(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modeling-agent",
        description="Codex-native mathematical modeling with thin evidence authority.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    start = commands.add_parser(
        "start", help="initialize a run for the current Codex task"
    )
    _add_research_arguments(start, include_verifier=False)

    next_action = commands.add_parser(
        "next", help="show the next useful action without launching a model"
    )
    next_action.add_argument("--workspace", type=Path, required=True)

    submit = commands.add_parser(
        "submit", help="check and register artifacts made by the current Codex task"
    )
    _add_research_arguments(submit, include_verifier=False)

    solve = commands.add_parser(
        "solve",
        help="register current-task artifacts, then attempt independent qualification",
    )
    _add_research_arguments(solve, include_verifier=True)

    research = commands.add_parser(
        "research", help="compatibility alias for submit; never launches a producer"
    )
    _add_research_arguments(research, include_verifier=False)

    qualify = commands.add_parser(
        "qualify", help="independently assess an existing candidate"
    )
    qualify.add_argument("--workspace", type=Path, required=True)
    qualify.add_argument("--max-seconds", type=int)
    qualify.add_argument(
        "--amend-budget",
        action="store_true",
        help="explicitly raise the frozen wall budget before qualification",
    )
    _add_model_arguments(qualify)

    review_export = commands.add_parser(
        "review-export",
        help="freeze a bounded packet for review in another Codex task",
    )
    review_export.add_argument("--workspace", type=Path, required=True)
    review_export.add_argument("--producer-context-id", required=True)
    review_export.add_argument(
        "--output",
        default=".modeling-agent/external-review/packet.json",
        help="workspace-relative packet path",
    )
    review_export.add_argument("--codex-bin")
    review_export.add_argument(
        "--local-replay",
        action="store_true",
        help="diagnostic only: replay without proving OS isolation",
    )

    review_import = commands.add_parser(
        "review-import",
        help="validate and admit a result returned by another Codex task",
    )
    review_import.add_argument("--workspace", type=Path, required=True)
    review_import.add_argument("--packet", required=True)
    review_import.add_argument("--result", required=True)
    review_import.add_argument("--max-seconds", type=int)
    review_import.add_argument(
        "--amend-budget",
        action="store_true",
        help="explicitly raise the frozen wall budget before import",
    )

    status = commands.add_parser("status", help="derive status from run facts")
    status.add_argument("--workspace", type=Path, required=True)

    evaluate = commands.add_parser("eval", help="freeze a component-ablation contract")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--objective", required=True)
    evaluate.add_argument("--model", default="gpt-5.6-sol")
    evaluate.add_argument("--max-model-turns", type=int, default=12)
    evaluate.add_argument("--max-tool-calls", type=int, default=30)
    evaluate.add_argument("--max-wall-seconds", type=int, default=1800)
    return parser


def _solve_contract(args: argparse.Namespace) -> dict[str, Any]:
    layout = RunLayout.open(args.workspace)
    if args.contract is not None:
        return load_contract(args.contract, args.objective, network_mode=args.network)
    if layout.contract_path.is_file():
        contract = validate_contract(
            json.loads(layout.contract_path.read_text(encoding="utf-8"))
        )
        if args.objective and args.objective.strip() != contract["objective"]:
            raise ValueError("objective differs from the existing task contract")
        return contract
    if not args.objective:
        raise ValueError("objective is required for a new run")
    contract = default_contract(args.objective, network_mode=args.network)
    contract["max_branches"] = args.max_branches
    contract["max_waves"] = args.max_waves
    return validate_contract(contract)


def _run_budgets(args: argparse.Namespace, layout: RunLayout) -> tuple[int, int]:
    if layout.state_path.is_file():
        frozen_state = json.loads(layout.state_path.read_text(encoding="utf-8"))
        frozen_budgets = frozen_state["budgets"]
        return (
            args.max_attempts
            if args.max_attempts is not None
            else frozen_budgets["max_attempts"],
            args.max_seconds
            if args.max_seconds is not None
            else frozen_budgets["max_seconds"],
        )
    return (
        args.max_attempts if args.max_attempts is not None else 3,
        args.max_seconds if args.max_seconds is not None else 1800,
    )


def _producer_model(args: argparse.Namespace, layout: RunLayout) -> str:
    requested = getattr(args, "producer_model", None)
    if layout.state_path.is_file():
        frozen = json.loads(layout.state_path.read_text(encoding="utf-8"))[
            "model_requested"
        ]
        if requested and requested != frozen:
            raise ValueError("producer_model differs from the frozen run provenance")
        return frozen
    return requested or "current-codex-unattested"


def _operator_projection(
    layout: RunLayout, contract: dict[str, Any], state: dict[str, Any]
) -> dict[str, Any]:
    paper = layout.work / contract["delivery_artifact"]
    qualification = state.get("qualification") or {}
    qualification_status = qualification.get("status", "NOT_REQUESTED")
    if state.get("status") == "completed":
        return {"action": "done", "reason": "delivery_supported"}
    if not paper.is_file() or paper.stat().st_size == 0:
        return {
            "action": "research",
            "reason": "delivery_artifact_missing",
            "workdir": str(layout.work),
            "instructions": operator_prompt(contract, state),
        }
    manifest, errors = validate_manifest(
        layout.work, contract, state["contract_hash"]
    )
    if manifest is None or errors:
        return {
            "action": "repair",
            "reason": "qualification_packet_not_ready",
            "workdir": str(layout.work),
            "findings": errors,
            "instructions": operator_prompt(contract, state),
        }
    delivery = state.get("delivery") or {}
    admitted = (
        state.get("status") == "candidate"
        and delivery.get("paper_sha256") == file_hash(paper)
    )
    if not admitted:
        return {
            "action": "submit",
            "reason": "artifacts_ready_for_mechanical_admission",
            "argv": ["modeling-agent", "submit", "--workspace", str(layout.root)],
        }
    if qualification_status in {"UNSUPPORTED", "PARTIALLY_SUPPORTED", "STALE"}:
        return {
            "action": "repair",
            "reason": f"qualification_{qualification_status.lower()}",
            "workdir": str(layout.work),
            "findings": qualification.get("errors", []),
            "instructions": operator_prompt(contract, state),
        }
    external_packet = layout.control / "external-review" / "packet.json"
    external_result = layout.control / "external-review" / "result.json"
    if external_packet.is_file():
        try:
            packet = json.loads(external_packet.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            packet = {}
        if packet.get("candidate_sha256") == delivery.get("candidate_sha256"):
            if external_result.is_file():
                try:
                    external = json.loads(
                        external_result.read_text(encoding="utf-8")
                    )
                except (OSError, ValueError):
                    external = {}
                if external.get("packet_sha256") == file_hash(external_packet):
                    return {
                        "action": "review-import",
                        "reason": "external_review_result_ready",
                        "argv": [
                            "modeling-agent",
                            "review-import",
                            "--workspace",
                            str(layout.root),
                            "--packet",
                            ".modeling-agent/external-review/packet.json",
                            "--result",
                            ".modeling-agent/external-review/result.json",
                        ],
                    }
            return {
                "action": "review-external",
                "reason": (
                    "external_review_result_stale"
                    if external_result.is_file()
                    else "external_review_packet_ready"
                ),
                "packet": str(external_packet),
                "instructions": (
                    "Open the packet in a different fresh Codex task and write "
                    ".modeling-agent/external-review/result.json according to "
                    "result_contract and review_schema."
                ),
            }
    return {
        "action": "qualify",
        "reason": f"candidate_{qualification_status.lower()}",
        "argv": ["modeling-agent", "qualify", "--workspace", str(layout.root)],
    }


def _command_exit_code(command: str, result: Any) -> int:
    if command == "solve":
        return 0 if result.status == "completed" else 2
    if command in {"research", "submit"}:
        return 0 if result.status in {"candidate", "completed"} else 2
    qualification_status = (result.state.get("qualification") or {}).get(
        "status"
    )
    return (
        2
        if result.status == "stopped"
        or qualification_status in {"NOT_RUN", "NOT_READY"}
        else 0
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "status":
            _print(run_status(args.workspace))
            return 0
        if args.command == "eval":
            _print(
                freeze_ablation(
                    args.output,
                    objective=args.objective,
                    model=args.model,
                    max_model_turns=args.max_model_turns,
                    max_tool_calls=args.max_tool_calls,
                    max_wall_seconds=args.max_wall_seconds,
                )
            )
            return 0
        if args.command == "next":
            layout = RunLayout.open(args.workspace)
            contract = validate_contract(
                json.loads(layout.contract_path.read_text(encoding="utf-8"))
            )
            state = json.loads(layout.state_path.read_text(encoding="utf-8"))
            displayed_status = (
                "ready"
                if state["status"] == "running" and not state.get("attempts")
                else state["status"]
            )
            _print(
                {
                    "status": displayed_status,
                    "workspace": str(layout.root),
                    "next": _operator_projection(layout, contract, state),
                    "summary": run_status(layout.root),
                }
            )
            return 0
        if args.command == "review-export":
            layout = RunLayout.open(args.workspace)
            contract = validate_contract(
                json.loads(layout.contract_path.read_text(encoding="utf-8"))
            )
            output_path = safe_path(layout.root, str(args.output))
            packet = build_external_review_packet(
                layout,
                contract,
                producer_context_id=args.producer_context_id,
                execution_mode=(
                    "local-diagnostic" if args.local_replay else "isolated"
                ),
                codex_executable=args.codex_bin,
            )
            atomic_write_json(output_path, packet)
            _print(
                {
                    "status": "review_packet_ready",
                    "packet": str(output_path),
                    "packet_sha256": file_hash(output_path),
                    "candidate_sha256": packet["candidate_sha256"],
                    "reproduction_status": packet["mechanical"].get(
                        "reproduction_status"
                    ),
                    "execution_isolation": packet["mechanical"].get(
                        "execution_isolation"
                    ),
                    "result_contract": {
                        **packet["result_contract"],
                        "packet_sha256": file_hash(output_path),
                    },
                }
            )
            return 0
        if args.command == "review-import":
            layout = RunLayout.open(args.workspace)
            state = json.loads(layout.state_path.read_text(encoding="utf-8"))
            contract = validate_contract(
                json.loads(layout.contract_path.read_text(encoding="utf-8"))
            )
            packet_path = safe_path(layout.root, str(args.packet))
            result_path = safe_path(layout.root, str(args.result))
            packet, external_review = validate_external_review_bundle(
                layout, contract, packet_path, result_path
            )
            max_seconds = args.max_seconds or state["budgets"]["max_seconds"]
            engine = ModelingEngine(
                args.workspace,
                researcher=None,
                verifier=None,
                source_reviewer=None,
                model_requested=state["model_requested"],
                max_attempts=state["budgets"]["max_attempts"],
                max_seconds=max_seconds,
                allow_budget_amendment=args.amend_budget,
                mechanical_override=packet["mechanical"],
                external_review=external_review,
            )
            result = engine.qualify(contract)
            _print(
                {
                    "status": result.status,
                    "reason": result.reason,
                    "workspace": str(result.workspace),
                    "review_transport": "external-codex-task",
                    "summary": run_status(result.workspace),
                }
            )
            return _command_exit_code(args.command, result)
        if args.command == "start":
            contract = _solve_contract(args)
            layout = RunLayout.open(args.workspace)
            max_attempts, max_seconds = _run_budgets(args, layout)
            producer_model = _producer_model(args, layout)
            engine = ModelingEngine(
                args.workspace,
                researcher=None,
                verifier=None,
                source_reviewer=None,
                model_requested=producer_model,
                max_attempts=max_attempts,
                max_seconds=max_seconds,
                allow_budget_amendment=args.amend_budget,
            )
            result = engine.prepare(contract)
            _print(
                {
                    "status": result.status,
                    "reason": result.reason,
                    "workspace": str(result.workspace),
                    "next": _operator_projection(layout, contract, result.state),
                }
            )
            return 0
        if args.command in {"solve", "submit", "research"}:
            contract = _solve_contract(args)
            layout = RunLayout.open(args.workspace)
            max_attempts, max_seconds = _run_budgets(args, layout)
            producer_model = _producer_model(args, layout)
            engine = ModelingEngine(
                args.workspace,
                researcher=None,
                verifier=None,
                source_reviewer=None,
                model_requested=producer_model,
                max_attempts=max_attempts,
                max_seconds=max_seconds,
                allow_budget_amendment=args.amend_budget,
            )
            submission = engine.submit(contract)
            if args.command != "solve":
                _print(
                    {
                        "status": submission.status,
                        "reason": submission.reason,
                        "workspace": str(submission.workspace),
                        "next": _operator_projection(
                            layout, contract, submission.state
                        ),
                        "summary": run_status(submission.workspace),
                    }
                )
                return _command_exit_code(args.command, submission)
            qualification_status = (
                submission.state.get("qualification") or {}
            ).get("status")
            if submission.status != "candidate" or qualification_status == "NOT_READY":
                _print(
                    {
                        "status": submission.status,
                        "reason": submission.reason,
                        "workspace": str(submission.workspace),
                        "next": _operator_projection(
                            layout, contract, submission.state
                        ),
                        "summary": run_status(submission.workspace),
                    }
                )
                return 2
            verifier = CodexCLIModel(
                model=args.model,
                executable=args.codex_bin,
                timeout_seconds=args.model_timeout,
            )
            qualification_engine = ModelingEngine(
                args.workspace,
                researcher=None,
                verifier=verifier,
                source_reviewer=CodexCLIModel(
                    model=args.model,
                    executable=verifier.executable,
                    timeout_seconds=args.model_timeout,
                ),
                model_requested=producer_model,
                max_attempts=max_attempts,
                max_seconds=max_seconds,
                allow_budget_amendment=args.amend_budget,
            )
            result = qualification_engine.qualify(contract)
            _print(
                {
                    "status": result.status,
                    "reason": result.reason,
                    "workspace": str(result.workspace),
                    "next": _operator_projection(layout, contract, result.state),
                    "summary": run_status(result.workspace),
                }
            )
            return _command_exit_code(args.command, result)
        if args.command == "qualify":
            layout = RunLayout.open(args.workspace)
            state = json.loads(layout.state_path.read_text(encoding="utf-8"))
            contract = validate_contract(
                json.loads(layout.contract_path.read_text(encoding="utf-8"))
            )
            model = args.model
            max_seconds = args.max_seconds or state["budgets"]["max_seconds"]
            manifest, manifest_errors = validate_manifest(
                layout.work, contract, state["contract_hash"]
            )
            if manifest is None or manifest_errors:
                engine = ModelingEngine(
                    args.workspace,
                    researcher=None,
                    verifier=None,
                    source_reviewer=None,
                    model_requested=state["model_requested"],
                    max_attempts=state["budgets"]["max_attempts"],
                    max_seconds=max_seconds,
                    allow_budget_amendment=args.amend_budget,
                )
                result = engine.qualify(contract)
                _print(
                    {
                        "status": result.status,
                        "reason": result.reason,
                        "workspace": str(result.workspace),
                        "summary": run_status(result.workspace),
                    }
                )
                return _command_exit_code(args.command, result)
            verifier = CodexCLIModel(
                model=model,
                executable=args.codex_bin,
                timeout_seconds=args.model_timeout,
            )
            engine = ModelingEngine(
                args.workspace,
                researcher=None,
                verifier=verifier,
                source_reviewer=CodexCLIModel(
                    model=model,
                    executable=verifier.executable,
                    timeout_seconds=args.model_timeout,
                ),
                model_requested=state["model_requested"],
                max_attempts=state["budgets"]["max_attempts"],
                max_seconds=max_seconds,
                allow_budget_amendment=args.amend_budget,
            )
            result = engine.qualify(contract)
            _print(
                {
                    "status": result.status,
                    "reason": result.reason,
                    "workspace": str(result.workspace),
                    "summary": run_status(result.workspace),
                }
            )
            return _command_exit_code(args.command, result)
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        TimeoutError,
        ValueError,
    ) as exc:
        _print({"status": "error", "type": type(exc).__name__, "message": str(exc)})
        return 1
    return 1
