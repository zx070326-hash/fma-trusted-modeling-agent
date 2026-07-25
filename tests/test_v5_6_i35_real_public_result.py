from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fma.v5_6.hybrid_ode import (
    HybridReplayReceiptV56,
    HybridScientificBundleV56,
)
from fma.v5_6.public_hybrid_campaign import (
    HybridPredictionArtifactV56,
    HybridResultManifestV56,
    verify_public_hybrid_campaign_v56,
)


ROOT = Path(__file__).resolve().parents[1]
I35 = ROOT / "experiments" / "iteration_35"
CAMPAIGN = I35 / "campaign_unseen_v56"
RESULT = I35 / "public_hybrid_run_v56"


def test_i35_real_public_result_is_replayable_negative_evidence() -> None:
    result = verify_public_hybrid_campaign_v56(
        unseen_campaign_dir=CAMPAIGN,
        output_dir=RESULT,
    )
    assert result.task_id == "i35-wb-e69813adc3410a607685"
    assert result.selected_candidate_id == "exponential.trend_only"
    assert result.recovery_triggered is True
    assert result.public_level_statuses == {
        "L0": "PASS",
        "L1": "PASS",
        "L2": "PASS",
        "L3": "FAIL",
        "L4": "PASS",
    }
    assert result.public_scientific_acceptance is False
    assert result.public_gate_decision == "ABSTAIN"
    assert result.prediction_status == "PROVISIONAL_ONLY"
    assert result.private_evaluation_status == "NOT_AUTHORIZED_NOT_RUN"
    assert result.private_evaluations_consumed == 0
    assert result.private_target_plaintext_accessed is False
    assert result.private_target_key_accessed is False
    assert result.external_host_established is False
    assert result.scientific_qualification_granted is False
    assert result.real_world_action_authorized is False


def test_i35_real_public_result_binds_processes_code_and_predictions() -> None:
    manifest = HybridResultManifestV56.model_validate_json(
        (RESULT / "result_manifest_v56.json").read_text(encoding="utf-8")
    )
    assert manifest.manifest_hash == manifest.content_hash()
    assert manifest.source_commit == (
        "8c4ccfffb60d5b2e8f08e0ecaf83b060fddec9b5"
    )
    assert manifest.runner_source_sha256 == hashlib.sha256(
        (
            ROOT / "fma" / "v5_6" / "public_hybrid_campaign.py"
        ).read_bytes()
    ).hexdigest()
    assert manifest.hybrid_adapter_source_sha256 == hashlib.sha256(
        (ROOT / "fma" / "v5_6" / "hybrid_ode.py").read_bytes()
    ).hexdigest()

    bundle = HybridScientificBundleV56.model_validate_json(
        (RESULT / "hybrid_ode_bundle_v56.json").read_text(encoding="utf-8")
    )
    receipts = [
        HybridReplayReceiptV56.model_validate(item)
        for item in json.loads(
            (RESULT / "hybrid_replay_receipts_v56.json").read_text(
                encoding="utf-8"
            )
        )
    ]
    assert bundle.bundle_hash == bundle.content_hash()
    assert len({item.process_id for item in receipts}) == 2
    assert len({item.deterministic_output_hash for item in receipts}) == 1
    assert all(item.fresh_process is True for item in receipts)
    assert all(item.receipt_hash == item.content_hash() for item in receipts)
    assert [item.receipt_hash for item in receipts] == (
        bundle.replay_receipt_hashes
    )

    prediction = HybridPredictionArtifactV56.model_validate_json(
        (RESULT / "hybrid_predictions_v56.json").read_text(encoding="utf-8")
    )
    assert prediction.prediction_hash == prediction.content_hash()
    assert prediction.status == "PROVISIONAL_ONLY"
    assert prediction.registered_by_code_owned_harness is False
    assert prediction.private_holdout_accessed_before_artifact is False
    assert [item.horizon_steps for item in prediction.predictions] == [
        1,
        2,
        3,
        4,
    ]
    assert all(item.value > 0 for item in prediction.predictions)
