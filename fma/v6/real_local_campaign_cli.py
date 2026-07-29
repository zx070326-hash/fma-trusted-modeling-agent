"""CLI for the opt-in V6.5 real-local Studio campaign runner."""

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
    RealLocalCampaignRunnerV65,
    RealLocalCampaignSpecV65,
    build_codex_runtime_contract_v65,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fma.v6.real_local_campaign_cli",
        description=(
            "Prepare, explicitly execute, replay, or verify one claim-limited "
            "real-local FMA Studio campaign."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help=(
            "Freeze a campaign spec without model inference or network use; "
            "optional runtime freezing invokes only local codex --version."
        ),
    )
    prepare.add_argument("--campaign-root", required=True)
    prepare.add_argument("--spec-file", required=True)
    prepare.add_argument("--authority-key-file", required=True)
    prepare.add_argument("--authority-key-id", default="real-local-v65")
    prepare.add_argument(
        "--freeze-codex-runtime",
        action="store_true",
        help=(
            "Inspect and bind the exact local Codex binary, model, budgets, "
            "runtime adapter, and code manifest before sealing the spec."
        ),
    )
    prepare.add_argument("--codex-bin")
    prepare.add_argument("--model")
    prepare.add_argument(
        "--expected-codex-cli-version",
        default=DEFAULT_EXPECTED_CLI_VERSION,
    )

    status = subparsers.add_parser(
        "status",
        help="Read the local campaign projection without live calls.",
    )
    status.add_argument("--campaign-root", required=True)
    status.add_argument("--authority-key-file", required=True)
    status.add_argument("--authority-key-id", default="real-local-v65")

    for command in ("execute", "verify"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--campaign-root", required=True)
        sub.add_argument("--authority-key-file", required=True)
        sub.add_argument("--authority-key-id", default="real-local-v65")
        if command == "execute":
            sub.add_argument(
                "--execute-live",
                action="store_true",
                help=(
                    "Permit this invocation to use Codex and World Bank only "
                    "when the frozen spec contains both matching opt-ins."
                ),
            )
            sub.add_argument("--retry-failed", action="store_true")
            sub.add_argument("--reconcile-human", action="store_true")
        sub.add_argument("--codex-bin")
        sub.add_argument("--model")
        sub.add_argument(
            "--expected-codex-cli-version",
            default=DEFAULT_EXPECTED_CLI_VERSION,
        )
    return parser


def _runner_from_args(args: argparse.Namespace) -> RealLocalCampaignRunnerV65:
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
    return RealLocalCampaignRunnerV65(
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
                raise ValueError("campaign spec file must contain one JSON object")
            supplied_hash = payload.pop("spec_hash", None)
            if args.freeze_codex_runtime:
                if supplied_hash is not None:
                    raise ValueError(
                        "runtime freezing requires an unsealed campaign input"
                    )
                request = payload.get("world_bank_request")
                if not isinstance(request, dict):
                    raise ValueError(
                        "campaign input lacks world_bank_request"
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
                contract = build_codex_runtime_contract_v65(
                    config=config,
                    source_adapter_id=str(request.get("adapter_id", "")),
                )
                payload["codex_runtime_contract"] = contract.model_dump(
                    mode="json"
                )
            spec = RealLocalCampaignSpecV65.seal(**payload)
            if supplied_hash is not None and supplied_hash != spec.spec_hash:
                raise ValueError("supplied campaign spec hash differs")
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
                output = {
                    "schema_version": "6.5-real-local-campaign-verification",
                    "verified": runner.verify(require_real=True),
                    "scientific_qualification_granted": False,
                    "real_world_action_authorized": False,
                }
    except CampaignReconciliationRequired as exc:
        print(
            canonical_json(
                {
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
