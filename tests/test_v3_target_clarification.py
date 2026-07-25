from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest

from fma.schemas import ArtifactRef
from fma.storage import RunStore
from fma.v3.target_clarification_v38 import (
    TargetClarificationBundleV38,
    TargetClarificationEvolutionReportV38,
    TargetClarificationManifestV38,
    TargetClarificationPolicyV38,
    TargetClarificationWorldPackSpecV38,
    build_target_clarification_policy_v38,
    verify_target_clarification_run_v38,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_V371 = (
    ROOT / "experiments" / "iteration_14"
    / "v371_target_aware_challenge_disposition"
)
VALID_V38 = ROOT / "experiments" / "iteration_15" / "v38_target_clarification"


@pytest.fixture(scope="module")
def v38_artifacts():
    store = RunStore.open_existing(VALID_V38)
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
        "policy": load("target_clarification_policy_v38", TargetClarificationPolicyV38),
        "spec": load("target_clarification_spec_v38", TargetClarificationWorldPackSpecV38),
        "bundle": load("target_clarification_bundle_v38", TargetClarificationBundleV38),
        "report": load(
            "target_clarification_evolution_report_v38",
            TargetClarificationEvolutionReportV38,
        ),
        "manifest": load(
            "target_clarification_manifest_v38", TargetClarificationManifestV38
        ),
    }


def test_v38_policy_is_bound_to_independently_verified_v371(v38_artifacts) -> None:
    rebuilt = build_target_clarification_policy_v38(SOURCE_V371)
    stored = v38_artifacts["policy"]
    assert rebuilt.policy_hash == stored.policy_hash
    assert rebuilt.source_clarification_count == 39
    assert not rebuilt.real_world_execution_permitted
    assert not rebuilt.task_router_permitted


def test_v38_clarification_updates_authoritative_target_with_lineage(v38_artifacts) -> None:
    receipts = v38_artifacts["bundle"].case_receipts
    executed = [item for item in receipts if item.action.status == "executed_synthetic"]
    assert len(executed) == 39
    assert all(item.action.budget_after == 0 for item in executed)
    assert all(item.authoritative_target_state.target_status == "authoritative" for item in executed)
    assert all(not item.authoritative_target_state.unresolved_fields for item in executed)
    assert all(
        item.action.after_contract.parent_contract_hash
        == item.action.before_contract_hash
        for item in executed
    )
    assert all(
        item.action.after_contract.triggering_evidence_hash
        == item.action.clarification_evidence.evidence_hash
        for item in executed
    )
    assert all(
        challenge.relevant_fold_indices
        == ([0] if challenge.decision_target == "free_run_prediction" else [1, 2])
        for item in executed for challenge in item.conditioned_challenges
    )


def test_v38_report_is_action_scoped_and_preserves_unresolved_cases(v38_artifacts) -> None:
    report = v38_artifacts["report"]
    assert report.expected_action_count == report.executed_action_count == 39
    assert math.isclose(report.action_precision, 1.0, abs_tol=1e-12)
    assert math.isclose(report.action_recall, 1.0, abs_tol=1e-12)
    assert math.isclose(report.target_accuracy_before, 1 / 3, abs_tol=1e-12)
    assert math.isclose(report.target_accuracy_after, 1.0, abs_tol=1e-12)
    assert report.next_action_counts == {
        "acquire_target_discriminating_evidence": 21,
        "expand_non_nested_family": 9,
        "proceed_private_validation": 9,
    }
    assert report.ready_for_composed_synthetic_loop == all(report.gates.values())
    assert not report.task_router_permitted
    assert not report.overall_qualification_permitted
    assert not report.confirmation_permitted
    assert not report.real_world_authorization_permitted


def test_v38_run_reloads_and_independently_replays(v38_artifacts) -> None:
    assert len(v38_artifacts["manifest"].artifact_refs) == 13
    assert verify_target_clarification_run_v38(v38_artifacts["store"].run_directory)


def test_v38_tampering_fails_closed(v38_artifacts, tmp_path) -> None:
    copied = tmp_path / "tampered_v38"
    shutil.copytree(v38_artifacts["store"].run_directory, copied)
    events = [
        json.loads(line)
        for line in (copied / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    target = next(
        event["payload"]
        for event in events
        if event["event_type"] == "artifact_committed"
        and event["payload"]["kind"] == "target_clarification_bundle_v38"
    )
    path = copied / target["relative_path"]
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["bundle_id"] = "tampered_bundle"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    assert not verify_target_clarification_run_v38(copied)
