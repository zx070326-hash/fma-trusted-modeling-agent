"""Independent closeout-authority CLI for provenance-only release."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from fma.hashing import canonical_json

from .campaign_protocol import ProspectiveCampaignProtocolV55
from .split_custody import (
    CampaignCloseoutAuthorizationV55,
    SplitCustodyAttestationV55,
    sign_campaign_closeout_authorization_v55,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--split-custody-attestation", required=True)
    parser.add_argument(
        "--terminal-status",
        required=True,
        choices=["ABSTAIN", "PRIVATE_EVALUATION_COMPLETE", "TERMINATED"],
    )
    parser.add_argument("--terminal-evidence", required=True)
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--closeout-authority-key-id", required=True)
    parser.add_argument("--closeout-authority-private-key", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    protocol = ProspectiveCampaignProtocolV55.model_validate_json(
        Path(args.protocol).read_text(encoding="utf-8")
    )
    attestation = SplitCustodyAttestationV55.model_validate_json(
        Path(args.split_custody_attestation).read_text(encoding="utf-8")
    )
    authorization: CampaignCloseoutAuthorizationV55 = (
        sign_campaign_closeout_authorization_v55(
            protocol=protocol,
            attestation=attestation,
            terminal_status=args.terminal_status,
            terminal_evidence_hash=hashlib.sha256(
                Path(args.terminal_evidence).read_bytes()
            ).hexdigest(),
            authorization_id=args.authorization_id,
            closeout_authority_key_id=args.closeout_authority_key_id,
            closeout_authority_private_key_pem=Path(
                args.closeout_authority_private_key
            ).read_bytes(),
        )
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(authorization) + "\n")
    print(
        canonical_json(
            {
                "schema_version": "5.5-closeout-authority-output",
                "case_id": authorization.case_id,
                "terminal_status": authorization.terminal_status,
                "terminal_evidence_hash": authorization.terminal_evidence_hash,
                "authorization_hash": authorization.authorization_hash,
                "release_source_provenance": True,
                "private_target_release_authorized": False,
                "scientific_qualification_granted": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
