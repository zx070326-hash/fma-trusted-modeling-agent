"""CLI for the explicitly versioned V6.7 pre-data campaign runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fma.codex_driver import DEFAULT_EXPECTED_CLI_VERSION, CodexCLIConfig
from fma.hashing import canonical_json
from fma.v5.__main__ import _decode_key

from .real_local_campaign import (
    CampaignReconciliationRequired,
    CampaignRetryRequired,
    LiveExecutionNotAuthorized,
    RealLocalCampaignError,
)
from .real_local_campaign_v67 import (
    RealLocalCampaignRunnerV67,
    RealLocalCampaignSpecV67,
    build_codex_runtime_contract_v67,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fma.v6.real_local_campaign_v67_cli",
        description=(
            "Prepare, execute, replay, or verify one explicit V6.7 campaign "
            "whose source, measurement, and protocol are frozen before S1."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help=(
            "Freeze a V6.7 campaign without model inference or network use; "
            "optional runtime freezing invokes only local codex --version."
        ),
    )
    prepare.add_argument("--campaign-root", required=True)
    prepare.add_argument("--spec-file", required=True)
    prepare.add_argument("--authority-key-file", required=True)
    prepare.add_argument("--authority-key-id", default="real-local-v67")
    prepare.add_argument("--freeze-codex-runtime", action="store_true")
    prepare.add_argument("--codex-bin")
    prepare.add_argument("--model")
    prepare.add_argument(
        "--expected-codex-cli-version",
        default=DEFAULT_EXPECTED_CLI_VERSION,
    )

    status = subparsers.add_parser(
        "status",
        help="Read the V6.7 campaign projection without live calls.",
    )
    status.add_argument("--campaign-root", required=True)
    status.add_argument("--authority-key-file", required=True)
    status.add_argument("--authority-key-id", default="real-local-v67")

    for command in ("execute", "verify"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--campaign-root", required=True)
        sub.add_argument("--authority-key-file", required=True)
        sub.add_argument("--authority-key-id", default="real-local-v67")
        if command == "execute":
            sub.add_argument("--execute-live", action="store_true")
            sub.add_argument("--retry-failed", action="store_true")
            sub.add_argument("--reconcile-human", action="store_true")
        else:
            sub.add_argument(
                "--allow-control",
                action="store_true",
                help=(
                    "Verify an explicitly controlled/fixture campaign without "
                    "promoting it to real-local evidence."
                ),
            )
        sub.add_argument("--codex-bin")
        sub.add_argument("--model")
        sub.add_argument(
            "--expected-codex-cli-version",
            default=DEFAULT_EXPECTED_CLI_VERSION,
        )
    return parser


def _runner_from_args(args: argparse.Namespace) -> RealLocalCampaignRunnerV67:
    key_path = Path(args.authority_key_file).expanduser().resolve(strict=True)
    authority_key = _decode_key(key_path.read_bytes())
    config = CodexCLIConfig(
        executable=(
            Path(args.codex_bin).expanduser().resolve(strict=True)
            if getattr(args, "codex_bin", None)
            else None
        ),
        requested_model=getattr(args, "model", None),
        expected_cli_version=getattr(
            args,
            "expected_codex_cli_version",
            DEFAULT_EXPECTED_CLI_VERSION,
        ),
    )
    return RealLocalCampaignRunnerV67(
        args.campaign_root,
        authority_key=authority_key,
        authority_key_id=args.authority_key_id,
        codex_config=config,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            payload = json.loads(
                Path(args.spec_file)
                .expanduser()
                .resolve(strict=True)
                .read_text(encoding="utf-8")
            )
            if not isinstance(payload, dict):
                raise ValueError(
                    "V6.7 campaign spec file must contain one JSON object"
                )
            supplied_hash = payload.pop("spec_hash", None)
            if args.freeze_codex_runtime:
                if supplied_hash is not None:
                    raise ValueError(
                        "runtime freezing requires an unsealed V6.7 input"
                    )
                request = payload.get("world_bank_request")
                if not isinstance(request, dict):
                    raise ValueError(
                        "V6.7 campaign input lacks world_bank_request"
                    )
                config = CodexCLIConfig(
                    executable=(
                        Path(args.codex_bin).expanduser().resolve(strict=True)
                        if args.codex_bin
                        else None
                    ),
                    requested_model=args.model,
                    expected_cli_version=args.expected_codex_cli_version,
                )
                contract = build_codex_runtime_contract_v67(
                    config=config,
                    source_adapter_id=str(request.get("adapter_id", "")),
                )
                payload["codex_runtime_contract"] = contract.model_dump(
                    mode="json"
                )
            spec = RealLocalCampaignSpecV67.seal(**payload)
            if supplied_hash is not None and supplied_hash != spec.spec_hash:
                raise ValueError("supplied V6.7 campaign spec hash differs")
            runner = _runner_from_args(args)
            runner.prepare(spec)
            output = runner.status()
        elif args.command == "status":
            output = _runner_from_args(args).status()
        else:
            runner = _runner_from_args(args)
            if args.command == "execute":
                receipt = runner.execute(
                    execute_live=args.execute_live,
                    retry_failed=args.retry_failed,
                    reconcile_human=args.reconcile_human,
                )
                output = receipt.model_dump(mode="json")
            else:
                require_real = not args.allow_control
                output = {
                    "schema_version": (
                        "6.7-real-local-campaign-verification"
                    ),
                    "verification_scope": (
                        "real_local"
                        if require_real
                        else "control_protocol_only"
                    ),
                    "verified": runner.verify(require_real=require_real),
                    "scientific_qualification_granted": False,
                    "real_world_action_authorized": False,
                }
    except CampaignReconciliationRequired as exc:
        print(
            canonical_json(
                {
                    "schema_version": "6.7-real-local-campaign-cli-error",
                    "status": "human_reconciliation_required",
                    "terminal_status": "HUMAN_RECONCILIATION_REQUIRED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "scientific_qualification_granted": False,
                    "real_world_action_authorized": False,
                }
            )
        )
        return 3
    except (
        OSError,
        ValueError,
        RealLocalCampaignError,
        LiveExecutionNotAuthorized,
        CampaignRetryRequired,
    ) as exc:
        print(
            canonical_json(
                {
                    "schema_version": "6.7-real-local-campaign-cli-error",
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "scientific_qualification_granted": False,
                    "real_world_action_authorized": False,
                }
            )
        )
        return 2
    print(canonical_json(output))
    if (
        args.command == "status"
        and output.get("reconciliation_required") is True
    ):
        return 3
    if args.command == "verify" and not output["verified"]:
        return 1
    if args.command == "execute":
        if output["terminal_status"] == "FAILED":
            return 1
        if output["terminal_status"] == "HUMAN_RECONCILIATION_REQUIRED":
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
