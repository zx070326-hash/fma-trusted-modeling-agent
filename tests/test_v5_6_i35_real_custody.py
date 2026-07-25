from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from fma.hashing import sha256_value
from fma.v5_5.public_ode_campaign import verify_public_launch_v55
from fma.v5_6.public_hybrid_campaign import (
    HybridCampaignProtocolV56,
    HybridForecastPlanV56,
    load_hybrid_thresholds_v56,
    materialize_hybrid_forecast_plan_v56,
)
from fma.v5_6.unseen_source import verify_unseen_world_bank_campaign_v56


ROOT = Path(__file__).resolve().parents[1]
I35 = ROOT / "experiments" / "iteration_35"
CAMPAIGN = I35 / "campaign_unseen_v56"
EXECUTION = I35 / "custodian_execution_v56.json"


def test_i35_real_source_is_frozen_new_and_still_private() -> None:
    verified = verify_unseen_world_bank_campaign_v56(CAMPAIGN)
    launch = verify_public_launch_v55(verified.inner_public_dir)
    assert verified.manifest.task_id == "i35-wb-e69813adc3410a607685"
    assert verified.manifest.task_id == launch.snapshot.task_id
    assert verified.manifest.fixture_only is False
    assert launch.snapshot.fixture_only is False
    assert verified.registry.required_prior_campaign_ids == ["i34"]
    assert verified.receipt.selected_identity_was_not_prior is True
    assert verified.receipt.selected_artifact_was_not_prior is True
    assert verified.receipt.selected_source_identity_hash not in {
        item.source_identity_hash for item in verified.registry.exclusions
    }
    assert verified.receipt.selected_source_artifact_sha256 not in {
        item.source_artifact_sha256 for item in verified.registry.exclusions
    }
    assert verified.receipt.source_identity_disclosed is False
    assert verified.receipt.same_host_logical_custody_only is True
    assert verified.receipt.external_host_established is False
    assert verified.receipt.scientific_qualification_granted is False
    assert verified.receipt.real_world_action_authorized is False
    assert launch.task_packet.source_identity_status == (
        "encrypted_until_closeout"
    )
    assert launch.manifest.private_target_values_disclosed is False
    assert launch.manifest.source_provenance_disclosed is False


def test_i35_custodian_execution_receipt_binds_frozen_source_commit() -> None:
    verified = verify_unseen_world_bank_campaign_v56(CAMPAIGN)
    receipt = json.loads(EXECUTION.read_text(encoding="utf-8"))
    assert receipt["receipt_hash"] == sha256_value(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_hash"
        }
    )
    assert receipt["task_id"] == verified.manifest.task_id
    assert receipt["source_commit"] == (
        "ce1673eb0a9ac6b8a199a7f94911b06d567757ef"
    )
    assert receipt["runner_sha256"] == hashlib.sha256(
        (I35 / "run_i35_custodian.py").read_bytes()
    ).hexdigest()
    assert receipt["unseen_source_adapter_sha256"] == hashlib.sha256(
        (ROOT / "fma" / "v5_6" / "unseen_source.py").read_bytes()
    ).hexdigest()
    assert receipt["source_registry_hash"] == verified.registry.registry_hash
    assert receipt["selection_spec_hash"] == (
        verified.receipt.selection_spec_hash
    )
    assert receipt["selection_seed_commitment"] == (
        verified.receipt.selection_seed_commitment
    )
    assert receipt["public_manifest_hash"] == (
        verified.receipt.public_manifest_hash
    )
    assert receipt["source_identity_disclosed"] is False
    assert receipt["private_target_values_disclosed"] is False
    assert receipt["external_host_established"] is False
    assert receipt["scientific_qualification_granted"] is False
    assert receipt["real_world_action_authorized"] is False


def test_i35_forecast_plan_was_frozen_before_public_model_run() -> None:
    science = I35 / "scientific"
    protocol = HybridCampaignProtocolV56.model_validate_json(
        (science / "HYBRID_CAMPAIGN_PROTOCOL_V56.json").read_text(
            encoding="utf-8"
        )
    )
    thresholds = load_hybrid_thresholds_v56(
        science / "HYBRID_THRESHOLDS_V56.json"
    )
    plan = HybridForecastPlanV56.model_validate_json(
        (science / "HYBRID_FORECAST_PLAN_V56.json").read_text(
            encoding="utf-8"
        )
    )
    plan.assert_sealed()
    replayed = materialize_hybrid_forecast_plan_v56(
        unseen_campaign_dir=CAMPAIGN,
        protocol=protocol,
        thresholds=thresholds,
        frozen_at=datetime.fromisoformat("2026-07-26T07:36:00+08:00"),
    )
    assert plan == replayed
    assert plan.private_target_values_accessed is False
    assert [item.target_id for item in plan.targets] == [
        "target-h1",
        "target-h2",
        "target-h3",
        "target-h4",
    ]
