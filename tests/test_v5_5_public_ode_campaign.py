from __future__ import annotations

import hashlib
import json
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fma.hashing import sha256_value
from fma.v5_2.ode_system import ODEThresholdsV52
from fma.v5_5.campaign_protocol import (
    ProspectiveCampaignProtocolV55,
    PublicEligibilitySettingsV55,
)
from fma.v5_5.public_ode_campaign import (
    PublicODECampaignResultV55,
    run_public_ode_campaign_v55,
)
from fma.v5_5.world_bank_custodian import (
    WorldBankSelectionSpecV55,
    materialize_world_bank_campaign_v55,
)


NOW = datetime(2026, 7, 26, 6, 7, 8, tzinfo=timezone.utc)


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


def _fixture_response(url: str) -> bytes:
    parts = urllib.parse.urlsplit(url).path.strip("/").split("/")
    country_code = parts[parts.index("country") + 1]
    indicator_code = parts[parts.index("indicator") + 1]
    rows = [
        {
            "indicator": {
                "id": indicator_code,
                "value": f"Fixture Indicator {indicator_code}",
            },
            "country": {
                "id": country_code,
                "value": f"Fixture Country {country_code}",
            },
            "countryiso3code": country_code,
            "date": str(year),
            "value": float(1000.0 + 4.0 * (year - 1990)),
        }
        for year in reversed(range(1990, 2022))
    ]
    return json.dumps([{"page": 1}, rows], separators=(",", ":")).encode()


def _materialize_public_fixture(tmp_path: Path) -> Path:
    protocol = ProspectiveCampaignProtocolV55.seal(
        protocol_id="public-runner-fixture",
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
            bootstrap_seed=1777,
        ),
        frozen_at=NOW,
    )
    thresholds = ODEThresholdsV52.seal(bootstrap_replicates=20)
    spec = WorldBankSelectionSpecV55.seal(
        selection_spec_id="public-runner-selection-fixture",
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
        prior_campaign_exclusion_hashes=[
            hashlib.sha256(b"prior-public-runner-fixture").hexdigest()
        ],
        fixture_only=True,
        frozen_at=NOW,
    )
    custody_private, _ = _key_pair()
    public_dir = tmp_path / "public"
    materialize_world_bank_campaign_v55(
        protocol=protocol,
        selection_spec=spec,
        ode_thresholds=thresholds,
        selection_seed=bytes(range(32)),
        private_target_key_id="fixture-target-key",
        private_target_key=bytes(range(32, 64)),
        source_provenance_key_id="fixture-provenance-key",
        source_provenance_key=bytes(range(64, 96)),
        custodian_host_id="fixture-custodian",
        coordinator_host_id="fixture-coordinator",
        generator_host_id="fixture-generator",
        custody_key_id="fixture-custody-signing-key",
        custody_private_key_pem=custody_private,
        output_dir=public_dir,
        fetcher=_fixture_response,
        retrieved_at=NOW,
    )
    return public_dir


def test_public_runner_is_terminal_public_only_and_fixture_abstains(
    tmp_path: Path,
) -> None:
    public_dir = _materialize_public_fixture(tmp_path)
    replay_secret = tmp_path / "replay-secret.bin"
    replay_secret.write_bytes(bytes(range(96, 128)))
    eligibility_private, _ = _key_pair()
    eligibility_key = tmp_path / "eligibility-private.pem"
    eligibility_key.write_bytes(eligibility_private)
    output_dir = tmp_path / "result"

    result = run_public_ode_campaign_v55(
        public_dir=public_dir,
        replay_secret_path=replay_secret,
        eligibility_private_key_path=eligibility_key,
        replay_key_id="fixture-replay-authority",
        eligibility_key_id="fixture-public-gate-authority",
        output_dir=output_dir,
    )

    assert result.public_gate_decision == "ABSTAIN"
    assert result.fixture_only is True
    assert result.private_evaluation_status == "NOT_AUTHORIZED_NOT_RUN"
    assert result.private_evaluations_consumed == 0
    assert result.private_target_plaintext_accessed is False
    assert result.private_target_key_accessed is False
    assert result.source_provenance_plaintext_accessed is False
    assert result.scientific_qualification_granted is False
    assert result.real_world_action_authorized is False
    assert result.rolling_selected_family in {
        "constant",
        "exponential",
        "gompertz",
        "logistic",
    }
    persisted = PublicODECampaignResultV55.model_validate_json(
        (output_dir / "public_ode_campaign_result_v55.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted == result
    assert persisted.result_hash == persisted.content_hash()

    candidate = json.loads(
        (output_dir / "candidate_evidence_v55.json").read_text(encoding="utf-8")
    )
    assert candidate["origin_sizes"] == list(range(12, 25))
    assert candidate["horizons"] == [1, 2, 3, 4]
    assert candidate["candidate_search_count"] == 4
    assert set(candidate["candidates"]) == {
        "constant",
        "exponential",
        "gompertz",
        "logistic",
    }
    assert all(
        item["training_window_rule"] == "all_available_public_prefix"
        for item in candidate["candidates"].values()
    )
    assert candidate["graph"]["unregistered_family_introduced"] is False
    assert candidate["graph"]["trailing_window_variant_introduced"] is False

    manifest = json.loads(
        (output_dir / "result_manifest_v55.json").read_text(encoding="utf-8")
    )
    expected_hash = sha256_value(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )
    assert manifest["manifest_hash"] == expected_hash
    assert manifest["private_evaluation_performed"] is False


def test_public_integrity_failure_precedes_local_authority_key_read(
    tmp_path: Path,
) -> None:
    public_dir = _materialize_public_fixture(tmp_path)
    snapshot_path = public_dir / "public_snapshot_v52.json"
    snapshot_path.write_bytes(snapshot_path.read_bytes() + b" ")
    output_dir = tmp_path / "result"

    with pytest.raises(ValueError, match="public artifact differs"):
        run_public_ode_campaign_v55(
            public_dir=public_dir,
            replay_secret_path=tmp_path / "missing-replay-secret",
            eligibility_private_key_path=tmp_path / "missing-gate-key",
            replay_key_id="fixture-replay-authority",
            eligibility_key_id="fixture-public-gate-authority",
            output_dir=output_dir,
        )
    assert not output_dir.exists()


def test_public_result_rejects_false_private_status() -> None:
    payload = {
        "task_id": "fixture-task",
        "public_manifest_hash": "a" * 64,
        "candidate_evidence_hash": "b" * 64,
        "forecast_bundle_hash": "c" * 64,
        "eligibility_input_hash": "d" * 64,
        "eligibility_assessment_hash": "e" * 64,
        "eligibility_receipt_hash": "f" * 64,
        "rolling_selected_family": "logistic",
        "v5_3_selected_family": "logistic",
        "selected_family_alignment": True,
        "all_registered_candidate_grids_complete": True,
        "public_scientific_acceptance_verified": True,
        "public_gate_decision": "ELIGIBLE",
        "graph_recovery_triggered": False,
        "private_evaluation_status": "NOT_AUTHORIZED_NOT_RUN",
        "fixture_only": False,
    }
    with pytest.raises(ValueError, match="private status differs"):
        PublicODECampaignResultV55(**payload)
