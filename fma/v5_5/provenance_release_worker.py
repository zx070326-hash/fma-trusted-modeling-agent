"""Closeout worker that can decrypt only the V5.5 provenance envelope."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from fma.hashing import canonical_json

from .campaign_protocol import ProspectiveCampaignProtocolV55
from .split_custody import (
    CampaignCloseoutAuthorizationV55,
    EncryptedCustodyEnvelopeV55,
    SplitCustodyAttestationV55,
    release_source_provenance_v55,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--source-provenance-envelope", required=True)
    parser.add_argument("--split-custody-attestation", required=True)
    parser.add_argument("--closeout-authorization", required=True)
    parser.add_argument("--terminal-evidence", required=True)
    parser.add_argument("--source-provenance-key", required=True)
    parser.add_argument("--custody-public-key", required=True)
    parser.add_argument("--closeout-authority-public-key", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    protocol = ProspectiveCampaignProtocolV55.model_validate_json(
        Path(args.protocol).read_text(encoding="utf-8")
    )
    envelope = EncryptedCustodyEnvelopeV55.model_validate_json(
        Path(args.source_provenance_envelope).read_text(encoding="utf-8")
    )
    attestation = SplitCustodyAttestationV55.model_validate_json(
        Path(args.split_custody_attestation).read_text(encoding="utf-8")
    )
    authorization = CampaignCloseoutAuthorizationV55.model_validate_json(
        Path(args.closeout_authorization).read_text(encoding="utf-8")
    )
    record, receipt = release_source_provenance_v55(
        protocol=protocol,
        source_provenance_envelope=envelope,
        attestation=attestation,
        authorization=authorization,
        terminal_evidence_hash=hashlib.sha256(
            Path(args.terminal_evidence).read_bytes()
        ).hexdigest(),
        source_provenance_key=Path(args.source_provenance_key).read_bytes(),
        custody_public_key_pem=Path(args.custody_public_key).read_bytes(),
        closeout_public_keys={
            authorization.closeout_authority_key_id: Path(
                args.closeout_authority_public_key
            ).read_bytes()
        },
    )
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("provenance release output directory already exists")
    output.mkdir(parents=True)
    artifacts = {
        "source_provenance_record_v55.json": record,
        "source_provenance_disclosure_receipt_v55.json": receipt,
    }
    for name, artifact in artifacts.items():
        with (output / name).open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(artifact) + "\n")
    print(
        canonical_json(
            {
                "schema_version": "5.5-provenance-release-worker-output",
                "case_id": record.case_id,
                "source_record_hash": record.record_hash,
                "disclosure_receipt_hash": receipt.receipt_hash,
                "private_target_envelope_accessed": False,
                "private_target_key_accessed": False,
                "private_evaluation_performed": False,
                "scientific_qualification_granted": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
