from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest

from fma.schemas import ArtifactRef
from fma.storage import RunStore
from fma.v3.target_discriminating_acquisition_v381 import (
    AcquisitionBundleV381,
    AcquisitionTrainingEvidenceV381,
    TargetDiscriminatingAcquisitionPolicyV381,
    TargetDiscriminatingEvolutionReportV381,
    TargetDiscriminatingManifestV381,
    TargetDiscriminatingWorldPackSpecV381,
    build_acquisition_training_evidence_v381,
    verify_target_discriminating_run_v381,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_V38 = ROOT / "experiments" / "iteration_15" / "v38_target_clarification"
VALID_V381 = (
    ROOT / "experiments" / "iteration_16"
    / "v381_target_discriminating_acquisition"
)


@pytest.fixture(scope="module")
def v381_artifacts():
    store = RunStore.open_existing(VALID_V381)
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
        "training": load(
            "acquisition_training_evidence_v381", AcquisitionTrainingEvidenceV381
        ),
        "baseline_policy": load(
            "baseline_acquisition_policy_v381",
            TargetDiscriminatingAcquisitionPolicyV381,
        ),
        "candidate_policy": load(
            "candidate_acquisition_policy_v381",
            TargetDiscriminatingAcquisitionPolicyV381,
        ),
        "spec": load(
            "target_discriminating_spec_v381",
            TargetDiscriminatingWorldPackSpecV381,
        ),
        "baseline": load("baseline_acquisition_bundle_v381", AcquisitionBundleV381),
        "candidate": load("candidate_acquisition_bundle_v381", AcquisitionBundleV381),
        "report": load(
            "target_discriminating_evolution_report_v381",
            TargetDiscriminatingEvolutionReportV381,
        ),
        "manifest": load(
            "target_discriminating_manifest_v381", TargetDiscriminatingManifestV381
        ),
    }


def test_v381_training_failure_evidence_recomputes(v381_artifacts) -> None:
    rebuilt = build_acquisition_training_evidence_v381(SOURCE_V38)
    stored = v381_artifacts["training"]
    assert rebuilt.evidence_hash == stored.evidence_hash
    assert stored.source_acquisition_case_count == 21
    assert stored.random_one_action_resolved_count == 0
    assert stored.disagreement_one_action_resolved_count == 0
    assert not stored.protocol_effect_guaranteed


def test_v381_policies_are_equal_budget_private_blind_and_fresh(v381_artifacts) -> None:
    baseline = v381_artifacts["baseline_policy"]
    candidate = v381_artifacts["candidate_policy"]
    spec = v381_artifacts["spec"]
    assert baseline.action_budget == candidate.action_budget == 1
    assert baseline.action_cost == candidate.action_cost == 1
    assert not baseline.private_observation_visible_before_permission
    assert not candidate.private_observation_visible_before_permission
    assert not baseline.real_world_execution_permitted
    assert not candidate.real_world_execution_permitted
    assert set(spec.seeds).isdisjoint(
        {21001, 21059, 21107, 21157, 21211, 21269, 21317, 21377,
         21433, 21487, 21529, 21587, 21649, 21683, 21737, 21799}
    )


def test_v381_executes_exact_targets_without_private_proposal_leak(v381_artifacts) -> None:
    for bundle in (v381_artifacts["baseline"], v381_artifacts["candidate"]):
        expected = [item for item in bundle.case_receipts if item.acquisition_expected]
        assert len(expected) == 22
        assert all(item.execution is not None for item in expected)
        assert all(item.proposal.budget_after == 0 for item in expected)
        assert all(not item.proposal.private_mechanism_seen for item in expected)
        assert all(not item.proposal.private_observation_seen for item in expected)
        assert all(not item.proposal.private_probe_seen for item in expected)
        assert all(not item.proposal.private_target_loss_seen for item in expected)
        assert all(not item.execution.real_world_execution for item in expected)
    candidate = v381_artifacts["candidate"]
    expected_candidate = [
        item for item in candidate.case_receipts if item.acquisition_expected
    ]
    assert all(len(item.proposal.proposal_model_hashes) == 3 for item in expected_candidate)
    assert all(len(item.proposal.action_scores) == 6 for item in expected_candidate)


def test_v381_refutes_repeat_acquisition_and_emits_stop_recovery(v381_artifacts) -> None:
    report = v381_artifacts["report"]
    assert report.acquisition_case_count == 22
    assert report.baseline_executed_count == report.candidate_executed_count == 22
    assert report.baseline_resolved_count == report.candidate_resolved_count == 0
    assert math.isclose(report.baseline_mean_adjudicated_target_loss, 10.0)
    assert math.isclose(report.candidate_mean_adjudicated_target_loss, 10.0)
    assert math.isclose(report.paired_mean_target_loss_improvement, 0.0)
    assert report.paired_improvement_ci_lower == report.paired_improvement_ci_upper == 0.0
    assert not report.ready_for_confirmation
    assert report.status == "target_discriminating_acquisition_refuted_v381"
    assert (
        report.recovery_action
        == "stop_repeat_acquisition_reclassify_estimator_or_family"
    )
    assert not report.task_router_permitted
    assert not report.qualification_permitted
    assert not report.confirmation_permitted
    assert not report.real_world_authorization_permitted


def test_v381_run_replays_and_tampering_fails_closed(v381_artifacts, tmp_path) -> None:
    assert len(v381_artifacts["manifest"].artifact_refs) == 16
    assert verify_target_discriminating_run_v381(
        v381_artifacts["store"].run_directory
    )
    copied = tmp_path / "tampered_v381"
    shutil.copytree(v381_artifacts["store"].run_directory, copied)
    events = [
        json.loads(line)
        for line in (copied / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    target = next(
        event["payload"]
        for event in events
        if event["event_type"] == "artifact_committed"
        and event["payload"]["kind"] == "candidate_acquisition_bundle_v381"
    )
    path = copied / target["relative_path"]
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["bundle_id"] = "tampered_bundle"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    assert not verify_target_discriminating_run_v381(copied)
