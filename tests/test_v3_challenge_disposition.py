from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest

from fma.schemas import ArtifactRef
from fma.storage import RunStore
from fma.v3.challenge_disposition_v371 import (
    ChallengeDispositionBundleV371,
    ChallengeDispositionEvolutionReportV371,
    ChallengeDispositionManifestV371,
    ChallengeDispositionPolicyV371,
    ModelChallengeWorldPackSpecV371,
    build_challenge_disposition_policy_v371,
    verify_challenge_disposition_run_v371,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_V37 = ROOT / "experiments" / "iteration_14" / "v37_applicability_model_challenge"
VALID_V371 = (
    ROOT / "experiments" / "iteration_14" / "v371_target_aware_challenge_disposition"
)
INVALID_V371 = ROOT / "experiments" / "iteration_14" / "v371_challenge_disposition"


@pytest.fixture(scope="module")
def v371_artifacts():
    store = RunStore.open_existing(VALID_V371)
    events = [json.loads(line) for line in store.event_path.read_text().splitlines()]
    refs = [
        ArtifactRef.model_validate(event["payload"])
        for event in events if event["event_type"] == "artifact_committed"
    ]

    def load(kind: str, model):
        ref = next(item for item in refs if item.kind == kind)
        return model.model_validate(store.load_artifact(ref))

    return {
        "store": store,
        "policy": load("challenge_disposition_policy_v371", ChallengeDispositionPolicyV371),
        "spec": load("model_challenge_spec_v371", ModelChallengeWorldPackSpecV371),
        "bundle": load("challenge_disposition_bundle_v371", ChallengeDispositionBundleV371),
        "report": load(
            "challenge_disposition_evolution_report_v371",
            ChallengeDispositionEvolutionReportV371,
        ),
        "manifest": load("challenge_disposition_manifest_v371", ChallengeDispositionManifestV371),
    }


def test_v371_policy_is_bound_to_verified_v37_failure(v371_artifacts) -> None:
    rebuilt = build_challenge_disposition_policy_v371(SOURCE_V37)
    stored = v371_artifacts["policy"]
    assert rebuilt.policy_hash == stored.policy_hash
    gains = stored.training_nonlinear_residual_gain_by_mechanism
    assert gains["duffing_oscillator"] >= stored.nonlinear_residual_gain_trigger
    assert all(
        value < stored.nonlinear_residual_gain_trigger
        for mechanism, value in gains.items()
        if mechanism != "duffing_oscillator"
    )
    assert not stored.execution_permitted
    assert not stored.task_router_permitted


def test_v371_target_authority_is_bound_and_all_actions_are_exercised(v371_artifacts) -> None:
    bundle = v371_artifacts["bundle"]
    clarifications = [
        item for item in bundle.dispositions
        if item.proposed_action == "clarify_decision_target"
    ]
    assert clarifications
    assert all(item.observed_target_status == "default_unverified" for item in clarifications)
    assert all(item.observed_unresolved_fields == ["decision_target"] for item in clarifications)
    assert len({item.target_state_hash for item in bundle.dispositions}) == 64
    assert {item.proposed_action for item in bundle.dispositions} == {
        "repair_data_quality",
        "clarify_decision_target",
        "acquire_target_discriminating_evidence",
        "expand_non_nested_family",
        "proceed_private_validation",
    }
    assert all(not item.private_mechanism_seen for item in bundle.dispositions)
    assert all(not item.private_probe_seen for item in bundle.dispositions)
    assert all(not item.private_target_loss_seen for item in bundle.dispositions)


def test_v371_report_recomputes_accuracy_and_stays_action_scoped(v371_artifacts) -> None:
    report = v371_artifacts["report"]
    expected_accuracy = sum(item.route_correct for item in report.case_results) / len(
        report.case_results
    )
    assert math.isclose(report.route_accuracy, expected_accuracy, abs_tol=1e-12)
    assert report.action_counts["clarify_decision_target"] == 39
    assert report.false_private_validation_count == 0
    assert report.ready_for_synthetic_action_experiment == all(report.gates.values())
    assert not report.task_router_permitted
    assert not report.overall_qualification_permitted
    assert not report.confirmation_permitted
    assert not report.real_world_authorization_permitted


def test_v371_shared_blind_spot_run_is_now_invalid(v371_artifacts) -> None:
    assert not verify_challenge_disposition_run_v371(INVALID_V371)
    assert verify_challenge_disposition_run_v371(v371_artifacts["store"].run_directory)


def test_v371_tampering_fails_closed(v371_artifacts, tmp_path) -> None:
    copied = tmp_path / "tampered_v371"
    shutil.copytree(v371_artifacts["store"].run_directory, copied)
    events = [json.loads(line) for line in (copied / "events.jsonl").read_text().splitlines()]
    target = next(
        event["payload"]
        for event in events
        if event["event_type"] == "artifact_committed"
        and event["payload"]["kind"] == "challenge_disposition_bundle_v371"
    )
    path = copied / target["relative_path"]
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["bundle_id"] = "tampered_bundle"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    assert not verify_challenge_disposition_run_v371(copied)
