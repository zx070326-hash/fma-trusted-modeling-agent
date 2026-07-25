"""V5.6 source-exclusion custody for genuinely new World Bank campaigns.

V5.5 recorded prior-campaign hashes but did not interpret their namespace or
enforce them during source selection.  This additive wrapper gives those
exclusions typed semantics, skips excluded identities before network access,
rejects excluded response bytes after access, and binds the result to a
custodian-signed public receipt.
"""

from __future__ import annotations

import base64
import hashlib
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import Field, model_validator

from fma.hashing import canonical_json, sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5_2.ode_system import ODEThresholdsV52
from fma.v5_5.campaign_protocol import ProspectiveCampaignProtocolV55
from fma.v5_5.public_ode_campaign import verify_public_launch_v55
from fma.v5_5.world_bank_custodian import (
    FetcherV55,
    WorldBankCustodianSummaryV55,
    WorldBankSelectionSpecV55,
    _candidate_order,
    _default_fetcher,
    _parse_complete_series,
    _source_url,
    materialize_world_bank_campaign_v55,
)


ProbeStatusV56 = Literal[
    "PRIOR_IDENTITY_EXCLUDED",
    "PRIOR_ARTIFACT_EXCLUDED",
    "FETCH_FAILED",
    "DATA_INELIGIBLE",
    "SELECTED",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def world_bank_source_identity_hash_v56(
    *,
    api_base: str,
    country_code: str,
    indicator_code: str,
    period_start: int,
    period_end: int,
) -> str:
    """Hash source identity independently of response formatting or retrieval."""

    return sha256_value(
        {
            "schema_version": "5.6-world-bank-source-identity",
            "source_authority": "World Bank Indicators API v2",
            "api_base": api_base,
            "country_code": country_code,
            "indicator_code": indicator_code,
            "period_start": period_start,
            "period_end": period_end,
        }
    )


class PriorSourceExclusionV56(StrictModel):
    schema_version: Literal["5.6-prior-source-exclusion"] = (
        "5.6-prior-source-exclusion"
    )
    campaign_id: Identifier
    source_identity_hash: Sha256
    source_artifact_sha256: Sha256
    source_provenance_record_hash: Sha256


class UnseenSourceRegistryV56(StrictModel):
    schema_version: Literal["5.6-unseen-source-registry"] = (
        "5.6-unseen-source-registry"
    )
    registry_id: Identifier
    required_prior_campaign_ids: list[Identifier]
    exclusions: Annotated[list[PriorSourceExclusionV56], Field(min_length=1)]
    identity_hash_rule: Literal[
        "authority_api_country_indicator_period_canonical_json_sha256"
    ] = "authority_api_country_indicator_period_canonical_json_sha256"
    frozen_before_selection: Literal[True] = True
    fixture_only: bool
    frozen_at: datetime
    registry_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_registry(self) -> "UnseenSourceRegistryV56":
        campaigns = [item.campaign_id for item in self.exclusions]
        if campaigns != sorted(set(campaigns)):
            raise ValueError("source exclusions must be campaign-sorted and unique")
        if self.required_prior_campaign_ids != sorted(
            set(self.required_prior_campaign_ids)
        ):
            raise ValueError("required prior campaign IDs must be sorted and unique")
        if set(self.required_prior_campaign_ids) != set(campaigns):
            raise ValueError("every required prior campaign needs one exclusion")
        if self.frozen_at.utcoffset() is None:
            raise ValueError("source registry frozen_at must be timezone-aware")
        if self.registry_hash and self.registry_hash != self.content_hash():
            raise ValueError("source registry hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "registry_hash")

    def assert_sealed(self) -> None:
        if not self.registry_hash or self.registry_hash != self.content_hash():
            raise ValueError("source registry is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "UnseenSourceRegistryV56":
        data.setdefault("frozen_at", _utc_now())
        draft = cls(**data)
        payload = draft.model_dump(exclude={"registry_hash"})
        payload["registry_hash"] = draft.content_hash()
        return cls(**payload)

    def exclusion_hashes(self) -> list[str]:
        return sorted(
            {
                value
                for item in self.exclusions
                for value in (
                    item.source_identity_hash,
                    item.source_artifact_sha256,
                    item.source_provenance_record_hash,
                )
            }
        )


class SourceProbeV56(StrictModel):
    sequence: Annotated[int, Field(ge=1)]
    source_identity_hash: Sha256
    response_artifact_sha256: Sha256 | None = None
    status: ProbeStatusV56

    @model_validator(mode="after")
    def validate_probe(self) -> "SourceProbeV56":
        if self.status in {
            "PRIOR_IDENTITY_EXCLUDED",
            "FETCH_FAILED",
        } and self.response_artifact_sha256 is not None:
            raise ValueError("unfetched source probe cannot bind response bytes")
        if self.status in {
            "PRIOR_ARTIFACT_EXCLUDED",
            "DATA_INELIGIBLE",
            "SELECTED",
        } and self.response_artifact_sha256 is None:
            raise ValueError("fetched source probe must bind response bytes")
        return self


class UnseenSourceSelectionReceiptV56(StrictModel):
    schema_version: Literal["5.6-unseen-source-selection-receipt"] = (
        "5.6-unseen-source-selection-receipt"
    )
    receipt_id: Identifier
    task_id: Identifier
    source_registry_hash: Sha256
    selection_spec_hash: Sha256
    selection_seed_commitment: Sha256
    public_manifest_hash: Sha256
    probes: Annotated[list[SourceProbeV56], Field(min_length=1)]
    selected_source_identity_hash: Sha256
    selected_source_artifact_sha256: Sha256
    required_prior_campaigns_covered: Literal[True] = True
    selected_identity_was_not_prior: Literal[True] = True
    selected_artifact_was_not_prior: Literal[True] = True
    source_identity_disclosed: Literal[False] = False
    same_host_logical_custody_only: Literal[True] = True
    external_host_established: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    custody_key_id: Identifier
    custody_signature_base64: str | None = None
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "UnseenSourceSelectionReceiptV56":
        if [item.sequence for item in self.probes] != list(
            range(1, len(self.probes) + 1)
        ):
            raise ValueError("source probes must be contiguous and ordered")
        selected = [
            item for item in self.probes if item.status == "SELECTED"
        ]
        if len(selected) != 1 or selected[0] != self.probes[-1]:
            raise ValueError("exactly the terminal source probe must be selected")
        if (
            selected[0].source_identity_hash
            != self.selected_source_identity_hash
            or selected[0].response_artifact_sha256
            != self.selected_source_artifact_sha256
        ):
            raise ValueError("selected source hashes differ from terminal probe")
        if self.receipt_hash and (
            not self.custody_signature_base64
            or self.receipt_hash != self.content_hash()
        ):
            raise ValueError("source-selection receipt envelope differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(
            self,
            "custody_signature_base64",
            "receipt_hash",
        )

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")


class UnseenCampaignArtifactV56(StrictModel):
    path: Annotated[str, Field(pattern=r"^[A-Za-z0-9_.\-/]+$")]
    sha256: Sha256
    size_bytes: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def validate_path(self) -> "UnseenCampaignArtifactV56":
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or "." in path.parts:
            raise ValueError("unseen campaign artifact path is unsafe")
        return self


class UnseenCampaignManifestV56(StrictModel):
    schema_version: Literal["5.6-unseen-campaign-manifest"] = (
        "5.6-unseen-campaign-manifest"
    )
    task_id: Identifier
    source_registry_hash: Sha256
    source_selection_receipt_hash: Sha256
    inner_public_manifest_hash: Sha256
    files: Annotated[list[UnseenCampaignArtifactV56], Field(min_length=17)]
    fixture_only: bool
    private_target_values_disclosed: Literal[False] = False
    source_identity_disclosed: Literal[False] = False
    external_host_established: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "UnseenCampaignManifestV56":
        paths = [item.path for item in self.files]
        if paths != sorted(set(paths)):
            raise ValueError("unseen campaign paths must be sorted and unique")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("unseen campaign manifest hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    @classmethod
    def seal(cls, **data: object) -> "UnseenCampaignManifestV56":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"manifest_hash"})
        payload["manifest_hash"] = draft.content_hash()
        return cls(**payload)


class VerifiedUnseenCampaignV56(StrictModel):
    registry: UnseenSourceRegistryV56
    receipt: UnseenSourceSelectionReceiptV56
    manifest: UnseenCampaignManifestV56
    inner_public_dir: Path


def _receipt_message(unsigned_hash: str) -> bytes:
    return b"fma-v56-unseen-source-selection:" + bytes.fromhex(unsigned_hash)


def _issue_receipt(
    *,
    custody_private_key_pem: bytes,
    **data: object,
) -> UnseenSourceSelectionReceiptV56:
    private_key = serialization.load_pem_private_key(
        custody_private_key_pem,
        password=None,
    )
    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("source-selection custody key must be Ed25519")
    draft = UnseenSourceSelectionReceiptV56(**data)
    payload = draft.model_dump(mode="json")
    payload["custody_signature_base64"] = base64.b64encode(
        private_key.sign(_receipt_message(draft.unsigned_hash()))
    ).decode("ascii")
    signed = UnseenSourceSelectionReceiptV56(**payload)
    final = signed.model_dump(mode="json")
    final["receipt_hash"] = signed.content_hash()
    return UnseenSourceSelectionReceiptV56(**final)


def verify_source_selection_receipt_v56(
    *,
    receipt: UnseenSourceSelectionReceiptV56,
    registry: UnseenSourceRegistryV56,
    selection_spec: WorldBankSelectionSpecV55,
    custody_public_key_pem: bytes,
) -> bool:
    try:
        registry.assert_sealed()
        selection_spec.assert_sealed()
        if (
            not receipt.receipt_hash
            or receipt.receipt_hash != receipt.content_hash()
            or receipt.source_registry_hash != registry.registry_hash
            or receipt.selection_spec_hash != selection_spec.selection_spec_hash
            or selection_spec.prior_campaign_exclusion_hashes
            != registry.exclusion_hashes()
        ):
            return False
        prior_identities = {
            item.source_identity_hash for item in registry.exclusions
        }
        prior_artifacts = {
            item.source_artifact_sha256 for item in registry.exclusions
        }
        if (
            receipt.selected_source_identity_hash in prior_identities
            or receipt.selected_source_artifact_sha256 in prior_artifacts
        ):
            return False
        public_key = serialization.load_pem_public_key(custody_public_key_pem)
        if not isinstance(public_key, Ed25519PublicKey):
            return False
        signature = base64.b64decode(
            receipt.custody_signature_base64 or "",
            validate=True,
        )
        public_key.verify(
            signature,
            _receipt_message(receipt.unsigned_hash()),
        )
        return True
    except (TypeError, ValueError, InvalidSignature):
        return False


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _json_bytes(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _select_unseen_source(
    *,
    spec: WorldBankSelectionSpecV55,
    registry: UnseenSourceRegistryV56,
    selection_seed: bytes,
    fetcher: FetcherV55,
) -> tuple[
    list[float],
    str,
    str,
    str,
    bytes,
    list[SourceProbeV56],
    dict[str, bytes | None],
]:
    prior_identities = {
        item.source_identity_hash for item in registry.exclusions
    }
    prior_artifacts = {
        item.source_artifact_sha256 for item in registry.exclusions
    }
    probes: list[SourceProbeV56] = []
    cache: dict[str, bytes | None] = {}
    for sequence, (country_code, indicator_code) in enumerate(
        _candidate_order(spec, selection_seed),
        start=1,
    ):
        identity_hash = world_bank_source_identity_hash_v56(
            api_base=spec.api_base,
            country_code=country_code,
            indicator_code=indicator_code,
            period_start=spec.public_start_year,
            period_end=spec.private_end_year,
        )
        url = _source_url(
            spec=spec,
            country_code=country_code,
            indicator_code=indicator_code,
        )
        if identity_hash in prior_identities:
            cache[url] = None
            probes.append(
                SourceProbeV56(
                    sequence=sequence,
                    source_identity_hash=identity_hash,
                    status="PRIOR_IDENTITY_EXCLUDED",
                )
            )
            continue
        try:
            raw_bytes = fetcher(url)
        except (OSError, TimeoutError, ValueError):
            cache[url] = None
            probes.append(
                SourceProbeV56(
                    sequence=sequence,
                    source_identity_hash=identity_hash,
                    status="FETCH_FAILED",
                )
            )
            continue
        cache[url] = raw_bytes
        artifact_hash = hashlib.sha256(raw_bytes).hexdigest()
        if artifact_hash in prior_artifacts:
            cache[url] = None
            probes.append(
                SourceProbeV56(
                    sequence=sequence,
                    source_identity_hash=identity_hash,
                    response_artifact_sha256=artifact_hash,
                    status="PRIOR_ARTIFACT_EXCLUDED",
                )
            )
            continue
        parsed = _parse_complete_series(
            raw_bytes=raw_bytes,
            spec=spec,
            country_code=country_code,
            indicator_code=indicator_code,
        )
        if parsed is None:
            probes.append(
                SourceProbeV56(
                    sequence=sequence,
                    source_identity_hash=identity_hash,
                    response_artifact_sha256=artifact_hash,
                    status="DATA_INELIGIBLE",
                )
            )
            continue
        values, country_name, indicator_name = parsed
        probes.append(
            SourceProbeV56(
                sequence=sequence,
                source_identity_hash=identity_hash,
                response_artifact_sha256=artifact_hash,
                status="SELECTED",
            )
        )
        return (
            values,
            country_name,
            indicator_name,
            url,
            raw_bytes,
            probes,
            cache,
        )
    raise ValueError("no non-prior source satisfied the frozen quality rule")


def materialize_unseen_world_bank_campaign_v56(
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
    """Materialize I35 only after code-enforced I34 source exclusion."""

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
            task_iteration_id="i35",
        )
        if (
            selected_url not in cache
            or cache[selected_url] != selected_raw_bytes
            or not summary.task_id.startswith("i35-")
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
            for path in sorted(temporary.rglob("*"))
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


def verify_unseen_world_bank_campaign_v56(
    output_dir: Path,
) -> VerifiedUnseenCampaignV56:
    root = output_dir.resolve()
    manifest = UnseenCampaignManifestV56.model_validate_json(
        (root / "unseen_campaign_manifest_v56.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest.manifest_hash != manifest.content_hash():
        raise ValueError("unseen campaign manifest is unsealed")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "unseen_campaign_manifest_v56.json"
    }
    declared = {item.path: item for item in manifest.files}
    if actual != set(declared):
        raise ValueError("unseen campaign exact file set differs")
    for name, entry in declared.items():
        payload = (root / PurePosixPath(name)).read_bytes()
        if (
            len(payload) != entry.size_bytes
            or hashlib.sha256(payload).hexdigest() != entry.sha256
        ):
            raise ValueError(f"unseen campaign artifact differs: {name}")

    registry = UnseenSourceRegistryV56.model_validate_json(
        (root / "source_exclusion_registry_v56.json").read_text(
            encoding="utf-8"
        )
    )
    receipt = UnseenSourceSelectionReceiptV56.model_validate_json(
        (root / "source_selection_receipt_v56.json").read_text(
            encoding="utf-8"
        )
    )
    inner = root / "campaign_public_v55"
    launch = verify_public_launch_v55(inner)
    if not verify_source_selection_receipt_v56(
        receipt=receipt,
        registry=registry,
        selection_spec=launch.selection_spec,
        custody_public_key_pem=(inner / "custody_public_key.pem").read_bytes(),
    ):
        raise ValueError("unseen source-selection receipt is invalid")
    if (
        len(
            {
                manifest.task_id,
                receipt.task_id,
                launch.task_packet.task_id,
            }
        )
        != 1
        or manifest.source_registry_hash != registry.registry_hash
        or manifest.source_selection_receipt_hash != receipt.receipt_hash
        or manifest.inner_public_manifest_hash != launch.manifest.manifest_hash
        or receipt.public_manifest_hash != launch.manifest.manifest_hash
        or receipt.selection_seed_commitment
        != launch.task_packet.selection_seed_commitment
        or receipt.custody_key_id
        != launch.split_attestation.custody_key_id
        or len(
            {
                manifest.fixture_only,
                registry.fixture_only,
                launch.manifest.fixture_only,
            }
        )
        != 1
    ):
        raise ValueError("unseen campaign cross-bindings differ")
    return VerifiedUnseenCampaignV56(
        registry=registry,
        receipt=receipt,
        manifest=manifest,
        inner_public_dir=inner,
    )


__all__ = [
    "PriorSourceExclusionV56",
    "SourceProbeV56",
    "UnseenCampaignManifestV56",
    "UnseenSourceRegistryV56",
    "UnseenSourceSelectionReceiptV56",
    "VerifiedUnseenCampaignV56",
    "materialize_unseen_world_bank_campaign_v56",
    "verify_source_selection_receipt_v56",
    "verify_unseen_world_bank_campaign_v56",
    "world_bank_source_identity_hash_v56",
]
