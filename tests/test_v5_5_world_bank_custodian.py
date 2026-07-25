from __future__ import annotations

import hashlib
import json
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fma.v5_5.campaign_protocol import (
    ProspectiveCampaignProtocolV55,
    PublicEligibilitySettingsV55,
)
from fma.v5_2.ode_system import ODEThresholdsV52
from fma.v5_5.split_custody import (
    EncryptedCustodyEnvelopeV55,
    SplitCustodyAttestationV55,
    open_private_target_envelope_v55,
    release_source_provenance_v55,
    sign_campaign_closeout_authorization_v55,
)
from fma.v5_5.world_bank_custodian import (
    WorldBankPublicManifestV55,
    WorldBankPublicTaskPacketV55,
    WorldBankSelectionSpecV55,
    _candidate_order,
    materialize_world_bank_campaign_v55,
)


NOW = datetime(2026, 7, 26, 3, 4, 5, tzinfo=timezone.utc)
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
        protocol_id="i34-world-bank-protocol",
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
            bootstrap_seed=6547,
        ),
        frozen_at=NOW,
    )


def _thresholds() -> ODEThresholdsV52:
    return ODEThresholdsV52.seal()


def _spec(
    protocol: ProspectiveCampaignProtocolV55,
    thresholds: ODEThresholdsV52,
) -> WorldBankSelectionSpecV55:
    return WorldBankSelectionSpecV55.seal(
        selection_spec_id="i34-world-bank-selection",
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
        prior_campaign_exclusion_hashes=[hashlib.sha256(b"prior").hexdigest()],
        fixture_only=True,
        frozen_at=NOW,
    )


def _pair_from_url(url: str) -> tuple[str, str]:
    parts = urllib.parse.urlsplit(url).path.strip("/").split("/")
    return parts[parts.index("country") + 1], parts[parts.index("indicator") + 1]


def _response(
    *,
    country_code: str,
    indicator_code: str,
    complete: bool,
) -> bytes:
    years = list(range(1990, 2022))
    if not complete:
        years = years[:-1]
    rows = [
        {
            "indicator": {
                "id": indicator_code,
                "value": f"Secret Indicator {indicator_code}",
            },
            "country": {
                "id": country_code,
                "value": f"Secret Country {country_code}",
            },
            "countryiso3code": country_code,
            "date": str(year),
            "value": float(1_000 + (year - 1990) * 11),
        }
        for year in reversed(years)
    ]
    return json.dumps([{"page": 1}, rows], separators=(",", ":")).encode()


def _fixture_fetcher(
    spec: WorldBankSelectionSpecV55,
) -> tuple[object, tuple[str, str]]:
    order = _candidate_order(spec, SELECTION_SEED)
    first = order[0]
    selected = order[1]

    def fetcher(url: str) -> bytes:
        pair = _pair_from_url(url)
        if pair == first:
            return _response(
                country_code=pair[0],
                indicator_code=pair[1],
                complete=False,
            )
        if pair == selected:
            return _response(
                country_code=pair[0],
                indicator_code=pair[1],
                complete=True,
            )
        raise OSError("fixture candidate unavailable")

    return fetcher, selected


def test_world_bank_custodian_selects_by_quality_and_withholds_source(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    thresholds = _thresholds()
    spec = _spec(protocol, thresholds)
    fetcher, selected = _fixture_fetcher(spec)
    custody_private, custody_public = _key_pair()
    output_dir = tmp_path / "public"
    summary = materialize_world_bank_campaign_v55(
        protocol=protocol,
        selection_spec=spec,
        ode_thresholds=thresholds,
        selection_seed=SELECTION_SEED,
        private_target_key_id="i34-target-key",
        private_target_key=TARGET_KEY,
        source_provenance_key_id="i34-provenance-key",
        source_provenance_key=PROVENANCE_KEY,
        custodian_host_id="logical-custodian",
        coordinator_host_id="logical-coordinator",
        generator_host_id="logical-generator",
        custody_key_id="i34-custody-key",
        custody_private_key_pem=custody_private,
        output_dir=output_dir,
        fetcher=fetcher,
        retrieved_at=NOW,
    )
    assert summary.fixture_only is True
    assert summary.external_host_established is False
    assert summary.scientific_qualification_granted is False

    packet = WorldBankPublicTaskPacketV55.model_validate_json(
        (output_dir / "public_task_packet_v55.json").read_text(encoding="utf-8")
    )
    packet.assert_sealed()
    assert packet.fixture_only is True
    assert len(packet.public_observations) == 28
    assert len(packet.targets) == 4
    assert all(item.value > 0 for item in packet.public_observations)
    assert packet.targets[0].time == packet.public_observations[-1].time + 1

    manifest = WorldBankPublicManifestV55.model_validate_json(
        (output_dir / "public_manifest_v55.json").read_text(encoding="utf-8")
    )
    for entry in manifest.files:
        payload = (output_dir / entry.path).read_bytes()
        assert len(payload) == entry.size_bytes
        assert hashlib.sha256(payload).hexdigest() == entry.sha256

    public_bytes = b"".join(
        path.read_bytes() for path in sorted(output_dir.iterdir())
    )
    selected_path = (
        f"/country/{selected[0]}/indicator/{selected[1]}".encode("ascii")
    )
    assert selected_path not in public_bytes
    assert f"Secret Country {selected[0]}".encode() not in public_bytes
    assert f"Secret Indicator {selected[1]}".encode() not in public_bytes

    target_envelope = EncryptedCustodyEnvelopeV55.model_validate_json(
        (output_dir / "private_target_envelope_v55.json").read_text(
            encoding="utf-8"
        )
    )
    capsule = open_private_target_envelope_v55(
        private_target_envelope=target_envelope,
        private_target_key=TARGET_KEY,
    )
    assert [item.target_id for item in capsule.holdout] == [
        "target-h1",
        "target-h2",
        "target-h3",
        "target-h4",
    ]
    assert capsule.secrecy_canary.encode() not in public_bytes

    provenance_envelope = EncryptedCustodyEnvelopeV55.model_validate_json(
        (output_dir / "source_provenance_envelope_v55.json").read_text(
            encoding="utf-8"
        )
    )
    attestation = SplitCustodyAttestationV55.model_validate_json(
        (output_dir / "split_custody_attestation_v55.json").read_text(
            encoding="utf-8"
        )
    )
    closeout_private, closeout_public = _key_pair()
    authorization = sign_campaign_closeout_authorization_v55(
        protocol=protocol,
        attestation=attestation,
        terminal_status="ABSTAIN",
        terminal_evidence_hash="e" * 64,
        authorization_id="i34-fixture-closeout",
        closeout_authority_key_id="i34-closeout-key",
        closeout_authority_private_key_pem=closeout_private,
        authorized_at=NOW,
    )
    record, receipt = release_source_provenance_v55(
        protocol=protocol,
        source_provenance_envelope=provenance_envelope,
        attestation=attestation,
        authorization=authorization,
        terminal_evidence_hash="e" * 64,
        source_provenance_key=PROVENANCE_KEY,
        custody_public_key_pem=custody_public,
        closeout_public_keys={"i34-closeout-key": closeout_public},
        disclosed_at=NOW,
    )
    assert record.source_title == (
        f"Secret Indicator {selected[1]} - Secret Country {selected[0]}"
    )
    assert record.table_or_series_id == (
        f"Secret Country {selected[0]} | Secret Indicator {selected[1]}"
    )
    assert f"/country/{selected[0]}/indicator/{selected[1]}" in (
        record.source_locator
    )
    assert receipt.private_target_envelope_accessed is False
    assert receipt.private_target_key_accessed is False


def test_world_bank_custodian_rejects_bad_seed_and_existing_output(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    thresholds = _thresholds()
    spec = _spec(protocol, thresholds)
    custody_private, _ = _key_pair()
    common = {
        "protocol": protocol,
        "selection_spec": spec,
        "ode_thresholds": thresholds,
        "private_target_key_id": "i34-target-key",
        "private_target_key": TARGET_KEY,
        "source_provenance_key_id": "i34-provenance-key",
        "source_provenance_key": PROVENANCE_KEY,
        "custodian_host_id": "logical-custodian",
        "coordinator_host_id": "logical-coordinator",
        "generator_host_id": "logical-generator",
        "custody_key_id": "i34-custody-key",
        "custody_private_key_pem": custody_private,
        "retrieved_at": NOW,
    }
    with pytest.raises(ValueError, match="exactly 32 bytes"):
        materialize_world_bank_campaign_v55(
            **common,
            selection_seed=b"short",
            output_dir=tmp_path / "bad-seed",
            fetcher=lambda _: b"",
        )

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        materialize_world_bank_campaign_v55(
            **common,
            selection_seed=SELECTION_SEED,
            output_dir=existing,
            fetcher=lambda _: pytest.fail("fetcher should not be called"),
        )


def test_world_bank_selection_spec_rejects_window_drift() -> None:
    protocol = _protocol()
    thresholds = _thresholds()
    with pytest.raises(ValueError, match="exactly 28 years"):
        WorldBankSelectionSpecV55.seal(
            selection_spec_id="bad-selection",
            protocol_hash=protocol.protocol_hash,
            country_codes=["AAA", "BBB", "CCC", "DDD"],
            indicator_codes=[
                "AG.LND.FRST.K2",
                "NY.GDP.MKTP.KD",
                "SP.URB.TOTL",
            ],
            public_start_year=1990,
            public_end_year=2016,
            private_end_year=2020,
            ode_threshold_hash=thresholds.threshold_hash,
            prior_campaign_exclusion_hashes=[],
            fixture_only=True,
            frozen_at=NOW,
        )
