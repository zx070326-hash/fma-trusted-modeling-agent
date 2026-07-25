"""Host-management CLI that signs one external worker runtime binding."""

from __future__ import annotations

import argparse
from pathlib import Path

from fma.hashing import canonical_json

from .external_private import (
    ExternalPrivateWorkerReceiptV53,
    sign_worker_host_attestation_v53,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-receipt", required=True)
    parser.add_argument("--worker-public-key", required=True)
    parser.add_argument("--coordinator-host-id", required=True)
    parser.add_argument("--generator-host-id", required=True)
    parser.add_argument("--attestation-id", required=True)
    parser.add_argument("--host-attester-key-id", required=True)
    parser.add_argument("--host-attester-private-key", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    worker_receipt = ExternalPrivateWorkerReceiptV53.model_validate_json(
        Path(args.worker_receipt).read_text(encoding="utf-8")
    )
    attestation = sign_worker_host_attestation_v53(
        worker_receipt=worker_receipt,
        worker_public_key_pem=Path(args.worker_public_key).read_bytes(),
        coordinator_host_id=args.coordinator_host_id,
        generator_host_id=args.generator_host_id,
        attestation_id=args.attestation_id,
        host_attester_key_id=args.host_attester_key_id,
        host_attester_private_key_pem=Path(args.host_attester_private_key).read_bytes(),
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(attestation) + "\n")
    print(
        canonical_json(
            {
                "schema_version": "5.3-host-attester-output",
                "worker_receipt_hash": worker_receipt.receipt_hash,
                "host_attestation_hash": attestation.attestation_hash,
                "scientific_qualification_granted": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
