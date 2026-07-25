"""Create-once local custody material for a non-qualifying campaign rehearsal."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Literal

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fma.hashing import canonical_json
from fma.schemas import StrictModel
from fma.v2.schemas import Sha256

from .split_custody import signing_key_fingerprint_v55


class LocalCustodyMaterialSummaryV55(StrictModel):
    schema_version: Literal["5.5-local-custody-material-summary"] = (
        "5.5-local-custody-material-summary"
    )
    selection_seed_commitment: Sha256
    private_target_key_fingerprint: Sha256
    source_provenance_key_fingerprint: Sha256
    custody_public_key_fingerprint: Sha256
    same_host_logical_custody_only: Literal[True] = True
    external_host_established: Literal[False] = False
    independent_management_key_control_established: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False


def _write_secret(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def generate_local_custody_material_v55(
    *,
    output_dir: Path,
) -> LocalCustodyMaterialSummaryV55:
    """Generate independent bytes but make no external-control claim."""

    root = output_dir.resolve()
    if root.exists():
        raise FileExistsError(f"output already exists: {root}")
    root.mkdir(parents=True)
    selection_seed = os.urandom(32)
    private_target_key = os.urandom(32)
    source_provenance_key = os.urandom(32)
    while source_provenance_key == private_target_key:
        source_provenance_key = os.urandom(32)
    custody_private = Ed25519PrivateKey.generate()
    custody_private_pem = custody_private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    custody_public_pem = custody_private.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    _write_secret(root / "selection_seed.bin", selection_seed)
    _write_secret(root / "private_target_aes256.key", private_target_key)
    _write_secret(
        root / "source_provenance_aes256.key",
        source_provenance_key,
    )
    _write_secret(root / "custody_ed25519_private.pem", custody_private_pem)
    _write_secret(root / "custody_ed25519_public.pem", custody_public_pem)
    return LocalCustodyMaterialSummaryV55(
        selection_seed_commitment=hashlib.sha256(selection_seed).hexdigest(),
        private_target_key_fingerprint=hashlib.sha256(
            private_target_key
        ).hexdigest(),
        source_provenance_key_fingerprint=hashlib.sha256(
            source_provenance_key
        ).hexdigest(),
        custody_public_key_fingerprint=signing_key_fingerprint_v55(
            custody_public_pem
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = generate_local_custody_material_v55(
        output_dir=Path(args.output_dir)
    )
    print(canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LocalCustodyMaterialSummaryV55",
    "generate_local_custody_material_v55",
]
