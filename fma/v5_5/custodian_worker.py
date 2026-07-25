"""External custodian CLI for V5.5 split target/provenance encryption.

Run only in the custodian environment.  AES keys, private targets, source
metadata, source canary, target canary, and the custody signing key remain
outside the coordinator and generator contexts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fma.hashing import canonical_json
from fma.v5.external_harness import PrivateTargetV50
from fma.v5_3.custody import PrivateScoreContractV53

from .campaign_protocol import ProspectiveCampaignProtocolV55
from .split_custody import (
    SourceProvenanceDraftV55,
    create_split_custody_envelopes_v55,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--score-contract", required=True)
    parser.add_argument("--private-targets", required=True)
    parser.add_argument("--source-provenance", required=True)
    parser.add_argument("--private-target-key-id", required=True)
    parser.add_argument("--private-target-key", required=True)
    parser.add_argument("--source-provenance-key-id", required=True)
    parser.add_argument("--source-provenance-key", required=True)
    parser.add_argument("--custodian-host-id", required=True)
    parser.add_argument("--coordinator-host-id", required=True)
    parser.add_argument("--generator-host-id", required=True)
    parser.add_argument("--attestation-id", required=True)
    parser.add_argument("--custody-key-id", required=True)
    parser.add_argument("--custody-private-key", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    protocol = ProspectiveCampaignProtocolV55.model_validate_json(
        Path(args.protocol).read_text(encoding="utf-8")
    )
    score_contract = PrivateScoreContractV53.model_validate_json(
        Path(args.score_contract).read_text(encoding="utf-8")
    )
    target_payload = json.loads(Path(args.private_targets).read_text(encoding="utf-8"))
    targets = [
        PrivateTargetV50.model_validate(item) for item in target_payload["targets"]
    ]
    provenance = SourceProvenanceDraftV55.model_validate_json(
        Path(args.source_provenance).read_text(encoding="utf-8")
    )
    (
        _,
        _,
        target_envelope,
        provenance_envelope,
        attestation,
    ) = create_split_custody_envelopes_v55(
        protocol=protocol,
        score_contract=score_contract,
        private_targets=targets,
        source_provenance=provenance,
        private_target_envelope_id=f"{score_contract.case_id}-private-targets",
        source_provenance_envelope_id=f"{score_contract.case_id}-source-provenance",
        private_target_key_id=args.private_target_key_id,
        private_target_key=Path(args.private_target_key).read_bytes(),
        source_provenance_key_id=args.source_provenance_key_id,
        source_provenance_key=Path(args.source_provenance_key).read_bytes(),
        custodian_host_id=args.custodian_host_id,
        coordinator_host_id=args.coordinator_host_id,
        generator_host_id=args.generator_host_id,
        attestation_id=args.attestation_id,
        custody_key_id=args.custody_key_id,
        custody_private_key_pem=Path(args.custody_private_key).read_bytes(),
    )
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("split custody output directory already exists")
    artifacts = {
        "private_target_envelope_v55.json": target_envelope,
        "source_provenance_envelope_v55.json": provenance_envelope,
        "split_custody_attestation_v55.json": attestation,
    }
    output.mkdir(parents=True)
    for name, artifact in artifacts.items():
        with (output / name).open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(artifact) + "\n")
    print(
        canonical_json(
            {
                "schema_version": "5.5-split-custodian-worker-output",
                "case_id": score_contract.case_id,
                "private_target_envelope_hash": target_envelope.envelope_hash,
                "source_provenance_envelope_hash": provenance_envelope.envelope_hash,
                "split_custody_attestation_hash": attestation.attestation_hash,
                "distinct_encryption_key_domains": True,
                "private_target_values_disclosed": False,
                "source_provenance_disclosed": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
