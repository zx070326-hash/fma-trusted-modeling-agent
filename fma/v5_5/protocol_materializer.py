"""Coordinator CLI for create-once V5.5 public launch artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from fma.hashing import canonical_json

from .campaign_protocol import (
    ProspectiveCampaignProtocolV55,
    materialize_public_launch_v55,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--eligibility-contract-id", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    protocol = ProspectiveCampaignProtocolV55.model_validate_json(
        Path(args.protocol).read_text(encoding="utf-8")
    )
    policy, contract, binding = materialize_public_launch_v55(
        protocol=protocol,
        task_id=args.task_id,
        eligibility_contract_id=args.eligibility_contract_id,
    )
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("public launch output directory already exists")
    artifacts = {
        "candidate_selection_policy_v55.json": policy,
        "public_eligibility_contract_v54.json": contract,
        "public_launch_binding_v55.json": binding,
    }
    output.mkdir(parents=True)
    for name, artifact in artifacts.items():
        with (output / name).open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(artifact) + "\n")
    print(
        canonical_json(
            {
                "schema_version": "5.5-protocol-materializer-output",
                "task_id": binding.task_id,
                "protocol_hash": protocol.protocol_hash,
                "baseline_id": binding.baseline_id,
                "candidate_policy_hash": policy.policy_hash,
                "public_eligibility_contract_hash": contract.contract_hash,
                "public_launch_binding_hash": binding.binding_hash,
                "private_evaluation_performed": False,
                "scientific_qualification_granted": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
