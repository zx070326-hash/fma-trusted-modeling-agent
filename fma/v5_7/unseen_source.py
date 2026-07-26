"""V5.7 source-exclusion custody entrypoint for the I36 campaign.

The custody and verification artifact schemas remain V5.6.  This additive
entrypoint changes only the code-owned iteration namespace to ``i36`` so the
new campaign cannot silently reuse an I35 task identity.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from fma.v5_2.ode_system import ODEThresholdsV52
from fma.v5_5.campaign_protocol import ProspectiveCampaignProtocolV55
from fma.v5_5.world_bank_custodian import (
    FetcherV55,
    WorldBankCustodianSummaryV55,
    WorldBankSelectionSpecV55,
    _default_fetcher,
    materialize_world_bank_campaign_v55,
)
from fma.v5_6.unseen_source import (
    UnseenCampaignArtifactV56,
    UnseenCampaignManifestV56,
    UnseenSourceRegistryV56,
    _issue_receipt,
    _json_bytes,
    _select_unseen_source,
    _write_new,
    verify_unseen_world_bank_campaign_v56,
)


def materialize_unseen_world_bank_campaign_v57(
    *,
    protocol: ProspectiveCampaignProtocolV55,
    selection_spec: WorldBankSelectionSpecV55,
    source_registry: UnseenSourceRegistryV56,
    ode_thresholds: ODEThresholdsV52,
    selection_seed: bytes,
    private_target_key_id: str,
    private_target_key: bytes,
    source_provenance_key_id: str,
    source_provenance_key: bytes,
    custodian_host_id: str,
    coordinator_host_id: str,
    generator_host_id: str,
    custody_key_id: str,
    custody_private_key_pem: bytes,
    output_dir: Path,
    fetcher: FetcherV55 = _default_fetcher,
    retrieved_at: datetime | None = None,
) -> WorldBankCustodianSummaryV55:
    """Materialize I36 after code-enforced exclusion of every prior source."""

    protocol.assert_sealed()
    selection_spec.assert_sealed()
    source_registry.assert_sealed()
    ode_thresholds.assert_sealed()
    if source_registry.fixture_only != selection_spec.fixture_only:
        raise ValueError("source registry and selection spec fixture flags differ")
    if (
        selection_spec.prior_campaign_exclusion_hashes
        != source_registry.exclusion_hashes()
    ):
        raise ValueError("selection spec does not bind typed source exclusions")
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    if len(selection_seed) != 32:
        raise ValueError("selection seed must contain exactly 32 bytes")

    (
        _values,
        _country_name,
        _indicator_name,
        selected_url,
        selected_raw_bytes,
        probes,
        cache,
    ) = _select_unseen_source(
        spec=selection_spec,
        registry=source_registry,
        selection_seed=selection_seed,
        fetcher=fetcher,
    )
    selected_probe = probes[-1]

    def cached_fetcher(url: str) -> bytes:
        if url not in cache or cache[url] is None:
            raise ValueError("source excluded or unavailable in frozen first pass")
        payload = cache[url]
        if not isinstance(payload, bytes):
            raise TypeError("cached source response is not bytes")
        return payload

    final_output = output_dir.resolve()
    final_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{final_output.name}.tmp-",
            dir=final_output.parent,
        )
    )
    try:
        inner = temporary / "campaign_public_v55"
        summary = materialize_world_bank_campaign_v55(
            protocol=protocol,
            selection_spec=selection_spec,
            ode_thresholds=ode_thresholds,
            selection_seed=selection_seed,
            private_target_key_id=private_target_key_id,
            private_target_key=private_target_key,
            source_provenance_key_id=source_provenance_key_id,
            source_provenance_key=source_provenance_key,
            custodian_host_id=custodian_host_id,
            coordinator_host_id=coordinator_host_id,
            generator_host_id=generator_host_id,
            custody_key_id=custody_key_id,
            custody_private_key_pem=custody_private_key_pem,
            output_dir=inner,
            fetcher=cached_fetcher,
            retrieved_at=retrieved_at,
            task_iteration_id="i36",
        )
        if (
            selected_url not in cache
            or cache[selected_url] != selected_raw_bytes
            or not summary.task_id.startswith("i36-")
        ):
            raise ValueError("V5.5 inner materialization changed selected source")
        receipt = _issue_receipt(
            custody_private_key_pem=custody_private_key_pem,
            receipt_id=f"{summary.task_id}-source-selection",
            task_id=summary.task_id,
            source_registry_hash=source_registry.registry_hash,
            selection_spec_hash=selection_spec.selection_spec_hash,
            selection_seed_commitment=hashlib.sha256(selection_seed).hexdigest(),
            public_manifest_hash=summary.public_manifest_hash,
            probes=probes,
            selected_source_identity_hash=(
                selected_probe.source_identity_hash
            ),
            selected_source_artifact_sha256=(
                selected_probe.response_artifact_sha256
            ),
            custody_key_id=custody_key_id,
        )
        _write_new(
            temporary / "source_exclusion_registry_v56.json",
            _json_bytes(source_registry),
        )
        _write_new(
            temporary / "source_selection_receipt_v56.json",
            _json_bytes(receipt),
        )
        files = [
            UnseenCampaignArtifactV56(
                path=path.relative_to(temporary).as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size_bytes=path.stat().st_size,
            )
            for path in sorted(
                temporary.rglob("*"),
                key=lambda item: item.relative_to(temporary).as_posix(),
            )
            if path.is_file()
        ]
        manifest = UnseenCampaignManifestV56.seal(
            task_id=summary.task_id,
            source_registry_hash=source_registry.registry_hash,
            source_selection_receipt_hash=receipt.receipt_hash,
            inner_public_manifest_hash=summary.public_manifest_hash,
            files=files,
            fixture_only=selection_spec.fixture_only,
        )
        _write_new(
            temporary / "unseen_campaign_manifest_v56.json",
            _json_bytes(manifest),
        )
        os.rename(temporary, final_output)
        return summary
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = [
    "materialize_unseen_world_bank_campaign_v57",
    "verify_unseen_world_bank_campaign_v56",
]
