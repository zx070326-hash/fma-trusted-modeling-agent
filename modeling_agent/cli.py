"""Command line interface for the lightweight modeling agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import __version__
from .ablation import freeze_ablation
from .core import StateStore, atomic_write_json, delivery_projection, state_planes
from .loop import ModelingLoop
from .model import RAW_SCHEMA, CodexCLIModel, NativeCodexResearcher
from .sidecar import NativeSidecar, load_contract, native_status


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _model(args: argparse.Namespace) -> CodexCLIModel:
    return CodexCLIModel(
        model=args.model,
        executable=args.codex_bin,
        timeout_seconds=args.model_timeout,
    )


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--codex-bin")
    parser.add_argument("--model-timeout", type=int, default=300)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modeling-agent",
        description="Open-ended modeling loop with thin evidence controls.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="start or resume a modeling run")
    run.add_argument("--workspace", type=Path, required=True)
    run.add_argument("--objective")
    run.add_argument("--max-steps", type=int, default=12)
    run.add_argument("--max-tool-calls", type=int, default=30)
    run.add_argument("--max-seconds", type=int, default=1800)
    _add_model_arguments(run)

    native = commands.add_parser(
        "native",
        help="run native Codex research with a thin contract/review sidecar",
    )
    native.add_argument("--workspace", type=Path, required=True)
    native.add_argument("--objective")
    native.add_argument("--contract", type=Path)
    native.add_argument("--max-attempts", type=int, default=2)
    native.add_argument("--max-seconds", type=int, default=1800)
    _add_model_arguments(native)

    status = commands.add_parser("status", help="inspect durable modeling state")
    status.add_argument("--workspace", type=Path, required=True)

    raw = commands.add_parser("raw", help="run the raw-model ablation arm")
    raw.add_argument("--workspace", type=Path, required=True)
    raw.add_argument("--objective", required=True)
    _add_model_arguments(raw)

    ablation = commands.add_parser(
        "ablation-init", help="freeze equal-budget raw/thin/native-sidecar arms"
    )
    ablation.add_argument("--output", type=Path, required=True)
    ablation.add_argument("--objective", required=True)
    ablation.add_argument("--model", default="gpt-5.6-sol")
    ablation.add_argument("--max-model-turns", type=int, default=12)
    ablation.add_argument("--max-tool-calls", type=int, default=30)
    ablation.add_argument("--max-wall-seconds", type=int, default=1800)
    return parser


def _status(workspace: Path) -> dict[str, Any]:
    native_path = workspace.resolve() / ".modeling-agent" / "native-state.json"
    legacy_path = workspace.resolve() / ".modeling-agent" / "state.json"
    if native_path.is_file() and not legacy_path.is_file():
        return native_status(workspace)
    state = StateStore(workspace).load()
    return {
        "run_id": state["run_id"],
        "planes": state_planes(state, workspace),
        "delivery": delivery_projection(state, workspace),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "status":
            _print(_status(args.workspace))
            return 0
        if args.command == "ablation-init":
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
        if args.command == "raw":
            workspace = args.workspace.resolve()
            workspace.mkdir(parents=True, exist_ok=True)
            model = _model(args)
            prompt = (
                "Solve the following mathematical-modeling problem directly. "
                "You have no harness tools in this ablation arm. State assumptions "
                "and limitations honestly.\n\n"
                + args.objective
            )
            result = model.complete(
                prompt, RAW_SCHEMA, role="raw-model", workspace=workspace
            )
            atomic_write_json(workspace / "raw-result.json", result)
            _print({"status": "completed", "result": result})
            return 0
        if args.command == "native":
            contract = load_contract(args.contract, args.objective)
            researcher = NativeCodexResearcher(
                model=args.model,
                executable=args.codex_bin,
            )
            verifier = _model(args)
            sidecar = NativeSidecar(
                args.workspace,
                researcher=researcher,
                verifier=verifier,
                model_requested=args.model,
                max_attempts=args.max_attempts,
                max_seconds=args.max_seconds,
            )
            state = sidecar.run(contract)
            _print(native_status(args.workspace))
            return 0 if state["status"] == "completed" else 2
        if args.command == "run":
            modeler = _model(args)
            verifier = _model(args)
            loop = ModelingLoop(
                args.workspace,
                modeler=modeler,
                verifier=verifier,
                max_steps=args.max_steps,
                max_tool_calls=args.max_tool_calls,
                max_seconds=args.max_seconds,
            )
            result = loop.run(args.objective)
            _print(
                {
                    "status": result.status,
                    "reason": result.reason,
                    "workspace": str(result.workspace),
                    "summary": _status(result.workspace),
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
