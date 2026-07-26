from __future__ import annotations

import json
import subprocess
from pathlib import Path

from fma.v5_6.unseen_source import verify_unseen_world_bank_campaign_v56
from fma.v5_7.public_adaptive_campaign import (
    verify_public_adaptive_campaign_v57,
)


ROOT = Path(__file__).resolve().parents[1]
I36 = ROOT / "experiments" / "iteration_36"
SOURCE = I36 / "campaign_unseen_v57"
RESULT = I36 / "public_adaptive_run_v57"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_ancestor(older: str, newer: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", older, newer],
            cwd=ROOT,
            check=False,
            capture_output=True,
            timeout=30,
        ).returncode
        == 0
    )


def test_i36_is_one_frozen_real_double_unseen_public_attempt() -> None:
    unseen = verify_unseen_world_bank_campaign_v56(SOURCE)
    result = verify_public_adaptive_campaign_v57(
        unseen_campaign_dir=SOURCE,
        output_dir=RESULT,
    )
    execution = _json(I36 / "custodian_execution_v57.json")
    protocol = _json(
        I36 / "scientific" / "ADAPTIVE_CAMPAIGN_PROTOCOL_V57.json"
    )
    plan = _json(
        I36 / "scientific" / "ADAPTIVE_FORECAST_PLAN_V57.json"
    )
    manifest = _json(RESULT / "result_manifest_v57.json")

    assert unseen.manifest.fixture_only is False
    assert unseen.registry.required_prior_campaign_ids == ["i34", "i35"]
    assert [item.campaign_id for item in unseen.registry.exclusions] == [
        "i34",
        "i35",
    ]
    assert unseen.receipt.selected_identity_was_not_prior is True
    assert unseen.receipt.selected_artifact_was_not_prior is True
    assert unseen.receipt.source_identity_disclosed is False
    assert unseen.receipt.external_host_established is False
    assert execution["task_id"] == unseen.manifest.task_id == result.task_id
    assert execution["selection_seed_commitment"] == (
        unseen.receipt.selection_seed_commitment
    )
    assert execution["source_identity_disclosed"] is False
    assert execution["private_target_values_disclosed"] is False

    implementation_commit = str(protocol["implementation_source_commit"])
    custody_source_commit = str(execution["source_commit"])
    result_source_commit = str(manifest["source_commit"])
    assert _is_ancestor(implementation_commit, custody_source_commit)
    assert _is_ancestor(custody_source_commit, result_source_commit)
    assert plan["frozen_before_public_model_run"] is True
    assert plan["private_target_values_accessed"] is False
    assert result.forecast_plan_hash == plan["plan_hash"]

    assert result.fixture_only is False
    assert result.public_level_statuses == {
        "L0": "PASS",
        "L1": "PASS",
        "L2": "PASS",
        "L3": "PASS",
        "L4": "PASS",
    }
    assert result.public_scientific_acceptance is True
    assert result.public_gate_decision == "ELIGIBLE"
    assert result.recovery_triggered is True
    assert result.selected_branch == "log_growth"
    assert result.selected_model_id == "log_random_walk_drift"
    assert result.prediction_status == "REGISTERED_FOR_PRIVATE_EVALUATION"
    assert result.private_evaluation_status == (
        "BLOCKED_EXTERNAL_HOST_NOT_RUN"
    )
    assert result.private_evaluations_consumed == 0
    assert result.private_target_plaintext_accessed is False
    assert result.private_target_key_accessed is False
    assert result.source_provenance_plaintext_accessed is False
    assert result.external_host_established is False
    assert result.scientific_qualification_granted is False
    assert result.real_world_action_authorized is False


def test_i36_recovery_evidence_rejects_unstable_ar_and_registers_drift() -> None:
    bundle = _json(RESULT / "adaptive_positive_series_bundle_v57.json")
    prediction = _json(RESULT / "adaptive_predictions_v57.json")
    primary = bundle["primary_bundle"]
    graph = bundle["graph"]
    candidates = {
        item["candidate_id"]: item
        for item in bundle["growth_candidates"]
    }

    assert {
        item["level"]: item["status"] for item in primary["levels"]
    }["L3"] == "FAIL"
    assert graph["recovery_reason_codes"] == ["primary_l3_fail"]
    assert graph["selected_branch"] == "log_growth"
    assert graph["selected_model_id"] == "log_random_walk_drift"
    assert candidates["log_growth_ar1"]["scientifically_admissible"] is False
    assert (
        candidates["log_growth_ar1"]["admissibility_checks"][
            "growth_phi_window_stable"
        ]
        is False
    )
    assert (
        candidates["log_random_walk_drift"]["scientifically_admissible"]
        is True
    )
    assert all(
        candidates["log_random_walk_drift"]["admissibility_checks"].values()
    )
    assert prediction["status"] == "REGISTERED_FOR_PRIVATE_EVALUATION"
    assert prediction["registered_by_code_owned_harness"] is True
    assert prediction["diagnostic_fallback_used"] is False
    assert prediction["private_holdout_accessed_before_artifact"] is False
    assert prediction["source_provenance_plaintext_accessed"] is False
    assert len(prediction["predictions"]) == 4
    assert all(item["value"] > 0 for item in prediction["predictions"])
