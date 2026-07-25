from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest

from fma.schemas import ArtifactRef
from fma.storage import RunStore
from fma.v3.validator_input_contract_v39 import (
    ValidatorBundleV39,
    ValidatorInputBugEvidenceV39,
    ValidatorRecoveryEvolutionReportV391,
    ValidatorRecoveryManifestV391,
    ValidatorRecoveryWorldPackSpecV391,
    build_validator_input_bug_evidence_v39,
    verify_validator_recovery_run_v39,
    verify_validator_recovery_run_v391,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_V381 = (
    ROOT / "experiments" / "iteration_16"
    / "v381_target_discriminating_acquisition"
)
FAILED_V39 = (
    ROOT / "experiments" / "iteration_17"
    / "v39_validator_input_contract_recovery"
)
VALID_V391 = (
    ROOT / "experiments" / "iteration_17"
    / "v391_evaluator_partition_recovery"
)


@pytest.fixture(scope="module")
def v391_artifacts():
    store = RunStore.open_existing(VALID_V391)
    events = [
        json.loads(line)
        for line in store.event_path.read_text(encoding="utf-8").splitlines()
    ]
    refs = [
        ArtifactRef.model_validate(event["payload"])
        for event in events if event["event_type"] == "artifact_committed"
    ]

    def load(kind: str, model):
        ref = next(item for item in refs if item.kind == kind)
        return model.model_validate(store.load_artifact(ref))

    return {
        "store": store,
        "evidence": load("validator_input_bug_evidence_v391", ValidatorInputBugEvidenceV39),
        "spec": load("validator_recovery_spec_v391", ValidatorRecoveryWorldPackSpecV391),
        "legacy": load("legacy_validator_bundle_v391", ValidatorBundleV39),
        "recovered": load("recovered_validator_bundle_v391", ValidatorBundleV39),
        "report": load(
            "validator_recovery_evolution_report_v391",
            ValidatorRecoveryEvolutionReportV391,
        ),
        "manifest": load("validator_recovery_manifest_v391", ValidatorRecoveryManifestV391),
    }


def test_v39_bug_evidence_recomputes_and_preserves_skeleton_gap(v391_artifacts) -> None:
    rebuilt = build_validator_input_bug_evidence_v39(SOURCE_V381)
    stored = v391_artifacts["evidence"]
    assert rebuilt.evidence_hash == stored.evidence_hash
    assert stored.legacy_resolved_count == 8
    assert stored.recovered_resolved_count == 22
    assert stored.recovered_duffing_mean_private_target_loss > 1.0
    assert stored.historical_numeric_conclusions_superseded
    assert not stored.model_qualification_permitted


def test_v39_failed_partition_run_remains_replayable() -> None:
    assert verify_validator_recovery_run_v39(FAILED_V39)


def test_v391_input_contracts_and_public_quality_partition_are_complete(v391_artifacts) -> None:
    report = v391_artifacts["report"]
    assert report.performance_case_count == 52
    assert report.quality_case_count == 12
    assert report.legacy_quality_abstention_count == 12
    assert report.recovered_quality_abstention_count == 12
    assert report.legacy_action_fold_misuse_count == 384
    assert report.recovered_action_fold_contract_count == 384
    assert report.evaluator_case_partition_rule == "public_observation_quality_flags_only"
    assert not report.private_performance_eligible_used_for_partition
    recovered = v391_artifacts["recovered"]
    assert all(
        binding.contract_valid and binding.simulator_input_value_count == 6
        for case in recovered.case_receipts
        for challenge in case.challenges
        for binding in challenge.fold_input_bindings
    )


def test_v391_recovers_validator_but_does_not_promote_models(v391_artifacts) -> None:
    report = v391_artifacts["report"]
    assert math.isclose(report.legacy_coverage, 21 / 52, abs_tol=1e-12)
    assert math.isclose(report.recovered_coverage, 1.0, abs_tol=1e-12)
    assert report.paired_improvement_ci_lower > 4.0
    assert report.recovered_mean_target_loss_by_mechanism["duffing_oscillator"] > 1.0
    assert report.ready_for_skeleton_factorial == all(report.gates.values())
    assert report.status == (
        "validator_input_contract_recovered_ready_for_skeleton_factorial_v391"
    )
    assert not report.task_router_permitted
    assert not report.qualification_permitted
    assert not report.confirmation_permitted
    assert not report.real_world_authorization_permitted


def test_v391_run_replays_and_tampering_fails_closed(v391_artifacts, tmp_path) -> None:
    assert len(v391_artifacts["manifest"].artifact_refs) == 8
    assert verify_validator_recovery_run_v391(v391_artifacts["store"].run_directory)
    copied = tmp_path / "tampered_v391"
    shutil.copytree(v391_artifacts["store"].run_directory, copied)
    events = [
        json.loads(line)
        for line in (copied / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    target = next(
        event["payload"]
        for event in events
        if event["event_type"] == "artifact_committed"
        and event["payload"]["kind"] == "recovered_validator_bundle_v391"
    )
    path = copied / target["relative_path"]
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["bundle_id"] = "tampered_bundle"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    assert not verify_validator_recovery_run_v391(copied)
