"""Command line interface for the single THIN modeling engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import __version__
from .ablation import freeze_ablation
from .engine import ModelingEngine, run_status
from .model import CodexCLIModel, NativeCodexResearcher
from .storage import RunLayout
from .verification import default_contract, load_contract, validate_contract


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--codex-bin")
    parser.add_argument("--model-timeout", type=int, default=300)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modeling-agent",
        description="Codex-native mathematical modeling with thin evidence authority.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    solve = commands.add_parser("solve", help="start or resume one modeling run")
    solve.add_argument("--workspace", type=Path, required=True)
    solve.add_argument("--objective")
    solve.add_argument("--contract", type=Path)
    solve.add_argument(
        "--network",
        choices=("research-search", "offline-compute"),
        default="research-search",
    )
    solve.add_argument("--max-attempts", type=int)
    solve.add_argument("--max-seconds", type=int)
    solve.add_argument("--max-branches", type=int, default=3)
    solve.add_argument("--max-waves", type=int, default=2)
    solve.add_argument(
        "--amend-budget",
        action="store_true",
        help="explicitly raise the frozen attempt/wall budget for an existing run",
    )
    _add_model_arguments(solve)

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
        if args.command == "solve":
            contract = _solve_contract(args)
            layout = RunLayout.open(args.workspace)
            if layout.state_path.is_file():
                frozen_state = json.loads(layout.state_path.read_text(encoding="utf-8"))
                frozen_budgets = frozen_state["budgets"]
                max_attempts = (
                    args.max_attempts
                    if args.max_attempts is not None
                    else frozen_budgets["max_attempts"]
                )
                max_seconds = (
                    args.max_seconds
                    if args.max_seconds is not None
                    else frozen_budgets["max_seconds"]
                )
            else:
                max_attempts = args.max_attempts if args.max_attempts is not None else 3
                max_seconds = args.max_seconds if args.max_seconds is not None else 1800
            researcher = NativeCodexResearcher(
                model=args.model,
                executable=args.codex_bin,
            )
            executable = researcher.executable

            def branch_factory() -> NativeCodexResearcher:
                return NativeCodexResearcher(model=args.model, executable=executable)

            engine = ModelingEngine(
                args.workspace,
                researcher=researcher,
                verifier=CodexCLIModel(
                    model=args.model,
                    executable=executable,
                    timeout_seconds=args.model_timeout,
                ),
                source_reviewer=CodexCLIModel(
                    model=args.model,
                    executable=executable,
                    timeout_seconds=args.model_timeout,
                ),
                model_requested=args.model,
                max_attempts=max_attempts,
                max_seconds=max_seconds,
                branch_researcher_factory=branch_factory,
                allow_budget_amendment=args.amend_budget,
            )
            result = engine.solve(contract)
            _print(
                {
                    "status": result.status,
                    "reason": result.reason,
                    "workspace": str(result.workspace),
                    "summary": run_status(result.workspace),
                }
            )
            return 0 if result.status == "completed" else 2
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
