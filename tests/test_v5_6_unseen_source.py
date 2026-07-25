from __future__ import annotations

import hashlib
import json
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fma.v5_2.ode_system import ODEThresholdsV52
from fma.v5_5.campaign_protocol import (
    ProspectiveCampaignProtocolV55,
    PublicEligibilitySettingsV55,
)
from fma.v5_5.world_bank_custodian import (
    WorldBankSelectionSpecV55,
    _candidate_order,
)
from fma.v5_6.unseen_source import (
    PriorSourceExclusionV56,
    UnseenSourceRegistryV56,
    UnseenSourceSelectionReceiptV56,
    materialize_unseen_world_bank_campaign_v56,
    verify_source_selection_receipt_v56,
    verify_unseen_world_bank_campaign_v56,
    world_bank_source_identity_hash_v56,
)


NOW = datetime(2026, 7, 26, 7, 0, 0, tzinfo=timezone.utc)
SELECTION_SEED = bytes(range(64, 96))
TARGET_KEY = bytes(range(32))
PROVENANCE_KEY = bytes(range(32, 64))


def _key_pair() -> tuple[bytes, bytes]:
    private = Ed25519PrivateKey.generate()
    private_pem = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def _protocol() -> ProspectiveCampaignProtocolV55:
    return ProspectiveCampaignProtocolV55.seal(
        protocol_id="i35-unseen-source-protocol",
        baseline_id="persistence_last_value",
        candidate_families=["constant", "exponential", "gompertz", "logistic"],
        maximum_candidate_search_count=16,
        public_eligibility=PublicEligibilitySettingsV55(
            expected_horizons=[1, 2, 3, 4],
            minimum_origin_count=12,
            contiguous_time_block_count=3,
            recent_origin_count=4,
            bootstrap_replicates=8192,
            bootstrap_block_length=4,
            multiplicity_correction_count=16,
            bootstrap_seed=104729,
        ),
        frozen_at=NOW,
    )


def _response(country_code: str, indicator_code: str) -> bytes:
    rows = [
        {
            "indicator": {
                "id": indicator_code,
                "value": f"Indicator {indicator_code}",
            },
            "country": {
                "id": country_code,
                "value": f"Country {country_code}",
            },
            "countryiso3code": country_code,
            "date": str(year),
            "value": float(1000 + (year - 1990) ** 2),
        }
        for year in reversed(range(1990, 2022))
    ]
    return json.dumps([{"page": 1}, rows], separators=(",", ":")).encode()


def _pair(url: str) -> tuple[str, str]:
    parts = urllib.parse.urlsplit(url).path.strip("/").split("/")
    return parts[parts.index("country") + 1], parts[
        parts.index("indicator") + 1
    ]


def _setup() -> tuple[
    ProspectiveCampaignProtocolV55,
    ODEThresholdsV52,
    WorldBankSelectionSpecV55,
    UnseenSourceRegistryV56,
    list[tuple[str, str]],
]:
    protocol = _protocol()
    thresholds = ODEThresholdsV52.seal()
    bare_spec = WorldBankSelectionSpecV55.seal(
        selection_spec_id="i35-unseen-source-selection",
        protocol_hash=protocol.protocol_hash,
        country_codes=["AAA", "BBB", "CCC", "DDD"],
        indicator_codes=[
            "AG.LND.FRST.K2",
            "NY.GDP.MKTP.KD",
            "SP.URB.TOTL",
        ],
        public_start_year=1990,
        public_end_year=2017,
        private_end_year=2021,
        ode_threshold_hash=thresholds.threshold_hash,
        prior_campaign_exclusion_hashes=[],
        fixture_only=True,
        frozen_at=NOW,
    )
    order = _candidate_order(bare_spec, SELECTION_SEED)
    first, second = order[:2]
    registry = UnseenSourceRegistryV56.seal(
        registry_id="i35-prior-source-registry",
        required_prior_campaign_ids=["i34"],
        exclusions=[
            PriorSourceExclusionV56(
                campaign_id="i34",
                source_identity_hash=world_bank_source_identity_hash_v56(
                    api_base=bare_spec.api_base,
                    country_code=first[0],
                    indicator_code=first[1],
                    period_start=1990,
                    period_end=2021,
                ),
                source_artifact_sha256=hashlib.sha256(
                    _response(*second)
                ).hexdigest(),
                source_provenance_record_hash=hashlib.sha256(
                    b"i34-provenance"
                ).hexdigest(),
            )
        ],
        fixture_only=True,
        frozen_at=NOW,
    )
    spec_payload = bare_spec.model_dump(
        mode="json",
        exclude={"selection_spec_hash"},
    )
    spec_payload["prior_campaign_exclusion_hashes"] = (
        registry.exclusion_hashes()
    )
    spec = WorldBankSelectionSpecV55.seal(**spec_payload)
    return protocol, thresholds, spec, registry, order


def test_unseen_custodian_enforces_identity_and_artifact_exclusions(
    tmp_path: Path,
) -> None:
    protocol, thresholds, spec, registry, order = _setup()
    custody_private, _ = _key_pair()
    fetched: list[tuple[str, str]] = []

    def fetcher(url: str) -> bytes:
        pair = _pair(url)
        fetched.append(pair)
        return _response(*pair)

    output = tmp_path / "unseen-public"
    summary = materialize_unseen_world_bank_campaign_v56(
        protocol=protocol,
        selection_spec=spec,
        source_registry=registry,
        ode_thresholds=thresholds,
        selection_seed=SELECTION_SEED,
        private_target_key_id="i35-target-key",
        private_target_key=TARGET_KEY,
        source_provenance_key_id="i35-provenance-key",
        source_provenance_key=PROVENANCE_KEY,
        custodian_host_id="logical-custodian",
        coordinator_host_id="logical-coordinator",
        generator_host_id="logical-generator",
        custody_key_id="i35-custody-key",
        custody_private_key_pem=custody_private,
        output_dir=output,
        fetcher=fetcher,
        retrieved_at=NOW,
    )

    assert summary.task_id.startswith("i35-wb-")
    assert order[0] not in fetched
    assert fetched[:2] == order[1:3]
    verified = verify_unseen_world_bank_campaign_v56(output)
    assert [item.status for item in verified.receipt.probes] == [
        "PRIOR_IDENTITY_EXCLUDED",
        "PRIOR_ARTIFACT_EXCLUDED",
        "SELECTED",
    ]
    assert verified.receipt.selected_source_identity_hash not in {
        item.source_identity_hash for item in registry.exclusions
    }
    assert verified.receipt.selected_source_artifact_sha256 not in {
        item.source_artifact_sha256 for item in registry.exclusions
    }
    assert verified.manifest.fixture_only is True
    assert verified.manifest.external_host_established is False
    assert verified.manifest.scientific_qualification_granted is False
    assert verified.manifest.real_world_action_authorized is False


def test_unseen_selection_receipt_rejects_forgery_and_manifest_drift(
    tmp_path: Path,
) -> None:
    protocol, thresholds, spec, registry, _ = _setup()
    custody_private, custody_public = _key_pair()
    output = tmp_path / "unseen-public"
    materialize_unseen_world_bank_campaign_v56(
        protocol=protocol,
        selection_spec=spec,
        source_registry=registry,
        ode_thresholds=thresholds,
        selection_seed=SELECTION_SEED,
        private_target_key_id="i35-target-key",
        private_target_key=TARGET_KEY,
        source_provenance_key_id="i35-provenance-key",
        source_provenance_key=PROVENANCE_KEY,
        custodian_host_id="logical-custodian",
        coordinator_host_id="logical-coordinator",
        generator_host_id="logical-generator",
        custody_key_id="i35-custody-key",
        custody_private_key_pem=custody_private,
        output_dir=output,
        fetcher=lambda url: _response(*_pair(url)),
        retrieved_at=NOW,
    )
    verified = verify_unseen_world_bank_campaign_v56(output)
    forged = verified.receipt.model_copy(
        update={"selected_source_identity_hash": "0" * 64}
    )
    assert verify_source_selection_receipt_v56(
        receipt=forged,
        registry=registry,
        selection_spec=spec,
        custody_public_key_pem=custody_public,
    ) is False

    receipt_path = output / "source_selection_receipt_v56.json"
    original = receipt_path.read_text(encoding="utf-8")
    receipt_path.write_text(original + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="artifact differs"):
        verify_unseen_world_bank_campaign_v56(output)


def test_unseen_custodian_requires_exact_typed_exclusion_binding(
    tmp_path: Path,
) -> None:
    protocol, thresholds, spec, registry, _ = _setup()
    custody_private, _ = _key_pair()
    drifted_payload = spec.model_dump(
        mode="json",
        exclude={"selection_spec_hash"},
    )
    drifted_payload["prior_campaign_exclusion_hashes"] = []
    drifted = WorldBankSelectionSpecV55.seal(**drifted_payload)
    with pytest.raises(ValueError, match="typed source exclusions"):
        materialize_unseen_world_bank_campaign_v56(
            protocol=protocol,
            selection_spec=drifted,
            source_registry=registry,
            ode_thresholds=thresholds,
            selection_seed=SELECTION_SEED,
            private_target_key_id="i35-target-key",
            private_target_key=TARGET_KEY,
            source_provenance_key_id="i35-provenance-key",
            source_provenance_key=PROVENANCE_KEY,
            custodian_host_id="logical-custodian",
            coordinator_host_id="logical-coordinator",
            generator_host_id="logical-generator",
            custody_key_id="i35-custody-key",
            custody_private_key_pem=custody_private,
            output_dir=tmp_path / "bad-binding",
            fetcher=lambda url: _response(*_pair(url)),
            retrieved_at=NOW,
        )


def test_unseen_receipt_model_rejects_nonterminal_selection() -> None:
    with pytest.raises(ValueError, match="terminal source probe"):
        UnseenSourceSelectionReceiptV56(
            receipt_id="bad-receipt",
            task_id="i35-bad",
            source_registry_hash="a" * 64,
            selection_spec_hash="b" * 64,
            selection_seed_commitment="c" * 64,
            public_manifest_hash="d" * 64,
            probes=[
                {
                    "sequence": 1,
                    "source_identity_hash": "e" * 64,
                    "response_artifact_sha256": "f" * 64,
                    "status": "SELECTED",
                },
                {
                    "sequence": 2,
                    "source_identity_hash": "1" * 64,
                    "status": "FETCH_FAILED",
                },
            ],
            selected_source_identity_hash="e" * 64,
            selected_source_artifact_sha256="f" * 64,
            custody_key_id="custody-key",
        )
