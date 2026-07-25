"""External-host CLI for creating a private capsule and public attestation.

Run this only on the separately administered custodian.  The private target
file, generated capsule, secrecy canary, and Ed25519 private key must never be
copied into the generation/coordinator workspace.
"""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

from fma.hashing import canonical_json
from fma.v5.external_harness import PrivateTargetV50

from .custody import (
    PrivateScoreContractV53,
    create_external_capsule_and_attestation_v53,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-contract", required=True)
    parser.add_argument("--private-targets", required=True)
    parser.add_argument("--private-source-manifest-hash", required=True)
    parser.add_argument("--external-anchor-receipt-hash", required=True)
    parser.add_argument("--custodian-host-id", required=True)
    parser.add_argument("--coordinator-host-id", required=True)
    parser.add_argument("--generator-host-id", required=True)
    parser.add_argument("--attestation-id", required=True)
    parser.add_argument("--attester-key-id", required=True)
    parser.add_argument("--private-key", required=True)
    parser.add_argument("--private-capsule-output", required=True)
    parser.add_argument("--public-attestation-output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = PrivateScoreContractV53.model_validate_json(
        Path(args.score_contract).read_text(encoding="utf-8")
    )
    target_payload = json.loads(Path(args.private_targets).read_text(encoding="utf-8"))
    targets = [
        PrivateTargetV50.model_validate(item) for item in target_payload["targets"]
    ]
    capsule, capsule_bytes, attestation = create_external_capsule_and_attestation_v53(
        score_contract=contract,
        private_targets=targets,
        secrecy_canary="fma-v53-" + secrets.token_hex(32),
        private_source_manifest_hash=args.private_source_manifest_hash,
        external_anchor_receipt_hash=args.external_anchor_receipt_hash,
        custodian_host_id=args.custodian_host_id,
        coordinator_host_id=args.coordinator_host_id,
        generator_host_id=args.generator_host_id,
        attestation_id=args.attestation_id,
        attester_key_id=args.attester_key_id,
        private_key_pem=Path(args.private_key).read_bytes(),
    )
    capsule_path = Path(args.private_capsule_output).resolve()
    attestation_path = Path(args.public_attestation_output).resolve()
    if capsule_path == attestation_path:
        raise ValueError("private capsule and public attestation paths must differ")
    capsule_path.parent.mkdir(parents=True, exist_ok=True)
    attestation_path.parent.mkdir(parents=True, exist_ok=True)
    with capsule_path.open("xb") as handle:
        handle.write(capsule_bytes)
    with attestation_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(attestation) + "\n")
    print(
        canonical_json(
            {
                "schema_version": "5.3-custodian-worker-output",
                "case_id": capsule.case_id,
                "capsule_commitment": capsule.capsule_hash,
                "attestation_hash": attestation.attestation_hash,
                "private_values_disclosed": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
