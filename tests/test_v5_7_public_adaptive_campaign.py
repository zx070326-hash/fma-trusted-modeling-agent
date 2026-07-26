from __future__ import annotations

import hashlib
import json
import shutil
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fma.hashing import canonical_json
from fma.v5_2.ode_system import ODEThresholdsV52
from fma.v5_5.campaign_protocol import (
    ProspectiveCampaignProtocolV55,
    PublicEligibilitySettingsV55,
)
from fma.v5_5.world_bank_custodian import WorldBankSelectionSpecV55
from fma.v5_6.public_hybrid_campaign import load_hybrid_thresholds_v56
from fma.v5_6.unseen_source import (
    PriorSourceExclusionV56,
    UnseenSourceRegistryV56,
    verify_unseen_world_bank_campaign_v56,
    world_bank_source_identity_hash_v56,
)
from fma.v5_7.public_adaptive_campaign import (
    AdaptiveCampaignProtocolV57,
    AdaptivePublicCampaignResultV57,
    load_adaptive_thresholds_v57,
    materialize_adaptive_forecast_plan_v57,
    run_public_adaptive_campaign_v57,
    verify_public_adaptive_campaign_v57,
)
from fma.v5_7.unseen_source import (
    materialize_unseen_world_bank_campaign_v57,
)


NOW = datetime(2026, 7, 26, 8, 0, 0, tzinfo=timezone.utc)
SEED = bytes(range(96, 128))


def _json(path: Path, value: object) -> None:
    path.write_text(
        canonical_json(value) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _private_key() -> bytes:
    key = Ed25519PrivateKey.generate()
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _pair(url: str) -> tuple[str, str]:
    parts = urllib.parse.urlsplit(url).path.strip("/").split("/")
    return parts[parts.index("country") + 1], parts[
        parts.index("indicator") + 1
    ]


def _response(country_code: str, indicator_code: str) -> bytes:
    rows = []
    for year in reversed(range(1990, 2022)):
        step = year - 1990
        value = 140.0 * pow(1.025, step)
        value *= 1.0 + 0.002 * ((step * 17) % 7 - 3)
        rows.append(
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
                "value": value,
            }
        )
    return json.dumps([{"page": 1}, rows], separators=(",", ":")).encode()


def _source_campaign(
    tmp_path: Path,
) -> tuple[
    Path,
    ProspectiveCampaignProtocolV55,
    UnseenSourceRegistryV56,
]:
    protocol = ProspectiveCampaignProtocolV55.seal(
        protocol_id="i36-public-adaptive-fixture",
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
    v52 = ODEThresholdsV52.seal()
    exclusions = [
        PriorSourceExclusionV56(
            campaign_id=campaign,
            source_identity_hash=world_bank_source_identity_hash_v56(
                api_base="https://api.worldbank.org/v2",
                country_code=country,
                indicator_code=indicator,
                period_start=1990,
                period_end=2021,
            ),
            source_artifact_sha256=hashlib.sha256(
                f"{campaign}-artifact".encode()
            ).hexdigest(),
            source_provenance_record_hash=hashlib.sha256(
                f"{campaign}-provenance".encode()
            ).hexdigest(),
        )
        for campaign, country, indicator in (
            ("i34", "ZZZ", "SP.URB.TOTL"),
            ("i35", "YYY", "NY.GDP.MKTP.KD"),
        )
    ]
    registry = UnseenSourceRegistryV56.seal(
        registry_id="i36-public-adaptive-registry",
        required_prior_campaign_ids=["i34", "i35"],
        exclusions=exclusions,
        fixture_only=True,
        frozen_at=NOW,
    )
    spec = WorldBankSelectionSpecV55.seal(
        selection_spec_id="i36-public-adaptive-selection",
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
        ode_threshold_hash=v52.threshold_hash,
        prior_campaign_exclusion_hashes=registry.exclusion_hashes(),
        fixture_only=True,
        frozen_at=NOW,
    )
    output = tmp_path / "source-campaign"
    summary = materialize_unseen_world_bank_campaign_v57(
        protocol=protocol,
        selection_spec=spec,
        source_registry=registry,
        ode_thresholds=v52,
        selection_seed=SEED,
        private_target_key_id="i36-target",
        private_target_key=b"t" * 32,
        source_provenance_key_id="i36-provenance",
        source_provenance_key=b"p" * 32,
        custodian_host_id="logical-custodian",
        coordinator_host_id="logical-coordinator",
        generator_host_id="logical-generator",
        custody_key_id="i36-custody",
        custody_private_key_pem=_private_key(),
        output_dir=output,
        fetcher=lambda url: _response(*_pair(url)),
        retrieved_at=NOW,
    )
    assert summary.task_id.startswith("i36-")
    assert verify_unseen_world_bank_campaign_v56(output).manifest.task_id == (
        summary.task_id
    )
    return output, protocol, registry


def _adaptive_protocol(
    *,
    v55_protocol: ProspectiveCampaignProtocolV55,
    registry: UnseenSourceRegistryV56,
    primary_threshold_hash: str,
    adaptive_threshold_hash: str,
) -> AdaptiveCampaignProtocolV57:
    root = Path(__file__).resolve().parents[1]
    return AdaptiveCampaignProtocolV57.seal(
        protocol_id="i36-public-adaptive-protocol",
        v55_protocol_hash=v55_protocol.protocol_hash,
        source_registry_hash=registry.registry_hash,
        primary_threshold_hash=primary_threshold_hash,
        adaptive_threshold_hash=adaptive_threshold_hash,
        primary_adapter_source_sha256=hashlib.sha256(
            (root / "fma" / "v5_6" / "hybrid_ode.py").read_bytes()
        ).hexdigest(),
        adaptive_adapter_source_sha256=hashlib.sha256(
            (
                root
                / "fma"
                / "v5_7"
                / "adaptive_positive_series.py"
            ).read_bytes()
        ).hexdigest(),
        unseen_source_adapter_source_sha256=hashlib.sha256(
            (root / "fma" / "v5_7" / "unseen_source.py").read_bytes()
        ).hexdigest(),
        unseen_source_core_source_sha256=hashlib.sha256(
            (root / "fma" / "v5_6" / "unseen_source.py").read_bytes()
        ).hexdigest(),
        world_bank_custodian_source_sha256=hashlib.sha256(
            (
                root
                / "fma"
                / "v5_5"
                / "world_bank_custodian.py"
            ).read_bytes()
        ).hexdigest(),
        public_runner_source_sha256=hashlib.sha256(
            (
                root
                / "fma"
                / "v5_7"
                / "public_adaptive_campaign.py"
            ).read_bytes()
        ).hexdigest(),
        primary_candidate_families=[
            "constant",
            "exponential",
            "gompertz",
            "logistic",
        ],
        primary_residual_modes=["ar1_residual", "trend_only"],
        recovery_growth_modes=[
            "log_growth_ar1",
            "log_random_walk_drift",
        ],
        required_public_levels=["L0", "L1", "L2", "L3", "L4"],
        frozen_at=NOW,
    )


def test_public_adaptive_runner_abstains_for_fixture_and_verifies(
    tmp_path: Path,
) -> None:
    source_dir, v55_protocol, registry = _source_campaign(tmp_path)
    root = Path(__file__).resolve().parents[1]
    primary_path = root / "V5_6_HYBRID_THRESHOLDS.json"
    adaptive_path = root / "V5_7_ADAPTIVE_THRESHOLDS.json"
    primary = load_hybrid_thresholds_v56(primary_path)
    adaptive = load_adaptive_thresholds_v57(adaptive_path)
    protocol = _adaptive_protocol(
        v55_protocol=v55_protocol,
        registry=registry,
        primary_threshold_hash=primary.threshold_hash,
        adaptive_threshold_hash=adaptive.threshold_hash,
    )
    plan = materialize_adaptive_forecast_plan_v57(
        unseen_campaign_dir=source_dir,
        protocol=protocol,
        primary_thresholds=primary,
        adaptive_thresholds=adaptive,
        frozen_at=NOW,
    )
    protocol_path = tmp_path / "protocol.json"
    plan_path = tmp_path / "plan.json"
    replay_secret = tmp_path / "replay.bin"
    _json(protocol_path, protocol)
    _json(plan_path, plan)
    replay_secret.write_bytes(b"r" * 32)

    output = tmp_path / "public-result"
    result = run_public_adaptive_campaign_v57(
        unseen_campaign_dir=source_dir,
        protocol_path=protocol_path,
        primary_threshold_path=primary_path,
        adaptive_threshold_path=adaptive_path,
        forecast_plan_path=plan_path,
        replay_secret_path=replay_secret,
        output_dir=output,
        created_at=NOW,
    )
    assert result.fixture_only is True
    assert result.public_gate_decision == "ABSTAIN"
    assert result.prediction_status == "PROVISIONAL_ONLY"
    assert result.private_evaluation_status == "NOT_AUTHORIZED_NOT_RUN"
    assert result.private_evaluations_consumed == 0
    assert result.scientific_qualification_granted is False
    assert result.real_world_action_authorized is False
    assert verify_public_adaptive_campaign_v57(
        unseen_campaign_dir=source_dir,
        output_dir=output,
    ) == result

    prediction = json.loads(
        (output / "adaptive_predictions_v57.json").read_text(
            encoding="utf-8"
        )
    )
    assert [item["horizon_steps"] for item in prediction["predictions"]] == [
        1,
        2,
        3,
        4,
    ]
    assert all(item["value"] > 0 for item in prediction["predictions"])
    assert prediction["private_holdout_accessed_before_artifact"] is False
    assert prediction["source_provenance_plaintext_accessed"] is False

    tampered = tmp_path / "tampered-result"
    shutil.copytree(output, tampered)
    (tampered / "REPORT.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact differs"):
        verify_public_adaptive_campaign_v57(
            unseen_campaign_dir=source_dir,
            output_dir=tampered,
        )


def test_public_result_requires_nonfixture_and_all_levels_for_eligibility() -> None:
    common = {
        "task_id": "i36-gate-test",
        "campaign_protocol_hash": "a" * 64,
        "source_campaign_manifest_hash": "b" * 64,
        "source_selection_receipt_hash": "c" * 64,
        "forecast_plan_hash": "d" * 64,
        "scientific_bundle_hash": "e" * 64,
        "prediction_hash": "f" * 64,
        "selected_branch": "log_growth",
        "selected_model_id": "log_growth_ar1",
        "recovery_triggered": True,
    }
    eligible = AdaptivePublicCampaignResultV57.seal(
        **common,
        fixture_only=False,
        public_level_statuses={
            "L0": "PASS",
            "L1": "PASS",
            "L2": "PASS",
            "L3": "PASS",
            "L4": "PASS",
        },
        public_scientific_acceptance=True,
        public_gate_decision="ELIGIBLE",
        prediction_status="REGISTERED_FOR_PRIVATE_EVALUATION",
        private_evaluation_status="BLOCKED_EXTERNAL_HOST_NOT_RUN",
    )
    assert eligible.public_gate_decision == "ELIGIBLE"
    fixture = AdaptivePublicCampaignResultV57.seal(
        **common,
        fixture_only=True,
        public_level_statuses=eligible.public_level_statuses,
        public_scientific_acceptance=False,
        public_gate_decision="ABSTAIN",
        prediction_status="PROVISIONAL_ONLY",
        private_evaluation_status="NOT_AUTHORIZED_NOT_RUN",
    )
    assert fixture.public_gate_decision == "ABSTAIN"
    failing = dict(eligible.public_level_statuses)
    failing["L3"] = "FAIL"
    rejected = AdaptivePublicCampaignResultV57.seal(
        **common,
        fixture_only=False,
        public_level_statuses=failing,
        public_scientific_acceptance=False,
        public_gate_decision="ABSTAIN",
        prediction_status="PROVISIONAL_ONLY",
        private_evaluation_status="NOT_AUTHORIZED_NOT_RUN",
    )
    assert rejected.public_gate_decision == "ABSTAIN"
