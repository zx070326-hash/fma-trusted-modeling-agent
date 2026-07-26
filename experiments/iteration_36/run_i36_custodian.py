"""Run the frozen I36 source custodian without exposing source identity."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

from fma.hashing import canonical_json, sha256_value
from fma.v5_2.ode_system import ODEThresholdsV52
from fma.v5_5.campaign_protocol import ProspectiveCampaignProtocolV55
from fma.v5_5.world_bank_custodian import WorldBankSelectionSpecV55
from fma.v5_6.unseen_source import UnseenSourceRegistryV56
from fma.v5_7.unseen_source import (
    materialize_unseen_world_bank_campaign_v57,
)


ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--selection-spec", required=True)
    parser.add_argument("--source-registry", required=True)
    parser.add_argument("--ode-thresholds", required=True)
    parser.add_argument("--selection-seed", required=True)
    parser.add_argument("--private-target-key", required=True)
    parser.add_argument("--source-provenance-key", required=True)
    parser.add_argument("--custody-private-key", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execution-receipt", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    execution_receipt = Path(args.execution_receipt).resolve()
    if execution_receipt.exists():
        raise FileExistsError(
            f"execution receipt already exists: {execution_receipt}"
        )
    protocol = ProspectiveCampaignProtocolV55.model_validate_json(
        Path(args.protocol).read_text(encoding="utf-8")
    )
    selection_spec = WorldBankSelectionSpecV55.model_validate_json(
        Path(args.selection_spec).read_text(encoding="utf-8")
    )
    registry = UnseenSourceRegistryV56.model_validate_json(
        Path(args.source_registry).read_text(encoding="utf-8")
    )
    thresholds = ODEThresholdsV52.model_validate_json(
        Path(args.ode_thresholds).read_text(encoding="utf-8")
    )
    summary = materialize_unseen_world_bank_campaign_v57(
        protocol=protocol,
        selection_spec=selection_spec,
        source_registry=registry,
        ode_thresholds=thresholds,
        selection_seed=Path(args.selection_seed).read_bytes(),
        private_target_key_id="i36-local-private-target-key",
        private_target_key=Path(args.private_target_key).read_bytes(),
        source_provenance_key_id="i36-local-source-provenance-key",
        source_provenance_key=Path(args.source_provenance_key).read_bytes(),
        custodian_host_id="same-host-logical-custodian-i36",
        coordinator_host_id="same-host-logical-coordinator-i36",
        generator_host_id="same-host-logical-generator-i36",
        custody_key_id="i36-local-custody-signing-key",
        custody_private_key_pem=Path(args.custody_private_key).read_bytes(),
        output_dir=output_dir,
    )
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    receipt = {
        "schema_version": "5.7-i36-custodian-execution",
        "task_id": summary.task_id,
        "source_commit": source_commit,
        "runner_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "unseen_source_adapter_sha256": hashlib.sha256(
            (ROOT / "fma" / "v5_7" / "unseen_source.py").read_bytes()
        ).hexdigest(),
        "unseen_source_core_sha256": hashlib.sha256(
            (ROOT / "fma" / "v5_6" / "unseen_source.py").read_bytes()
        ).hexdigest(),
        "world_bank_custodian_sha256": hashlib.sha256(
            (
                ROOT
                / "fma"
                / "v5_5"
                / "world_bank_custodian.py"
            ).read_bytes()
        ).hexdigest(),
        "source_registry_hash": registry.registry_hash,
        "selection_spec_hash": summary.selection_spec_hash,
        "selection_seed_commitment": summary.selection_seed_commitment,
        "public_manifest_hash": summary.public_manifest_hash,
        "source_identity_disclosed": False,
        "private_target_values_disclosed": False,
        "same_host_logical_custody_only": True,
        "external_host_established": False,
        "scientific_qualification_granted": False,
        "real_world_action_authorized": False,
    }
    receipt["receipt_hash"] = sha256_value(receipt)
    execution_receipt.parent.mkdir(parents=True, exist_ok=True)
    with execution_receipt.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(receipt) + "\n")
    print(canonical_json(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
