from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from fma.schemas import ArtifactRef
from fma.storage import RunStore
from fma.v3.evidence_compiled_growth_v313 import (
    CompiledConceptLibraryV313,
    EvidenceCompiledGrowthManifestV313,
    EvidenceCompiledGrowthReportV313,
    GrowthDiscoveryBundleV313,
    PrivateConceptAdjudicationV313,
    verify_evidence_compiled_growth_run_v313,
)
from fma.v3.evidence_concept_compiler_v313 import ConceptExperienceStoreV313
from fma.v4.atomic_admission import (
    reconcile_atomic_concept_admission_v40,
    verify_atomic_concept_admission_v40,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_V312 = ROOT / "experiments/iteration_20/v312_open_set_concept_confirmation"
DEVELOPMENT_V313 = (
    ROOT / "experiments/iteration_21/v313_evidence_concept_development"
)
CONFIRMATION_V313 = (
    ROOT / "experiments/iteration_21/v313_evidence_concept_confirmation"
)
ITERATION_STATUS = ROOT / "experiments/iteration_21/STATUS.json"


def _load_artifacts(run_directory: Path) -> dict[str, object]:
    store = RunStore.open_existing(run_directory)
    events = [
        json.loads(line)
        for line in store.event_path.read_text(encoding="utf-8").splitlines()
    ]
    refs = [
        ArtifactRef.model_validate(event["payload"])
        for event in events
        if event["event_type"] == "artifact_committed"
    ]

    def load(kind: str, model):
        ref = next(item for item in refs if item.kind == kind)
        return model.model_validate(store.load_artifact(ref))

    return {
        "store": store,
        "refs": refs,
        "bundle": load("growth_discovery_bundle_v313", GrowthDiscoveryBundleV313),
        "adjudication": load(
            "private_concept_adjudication_v313", PrivateConceptAdjudicationV313
        ),
        "experience": load(
            "concept_experience_store_v313", ConceptExperienceStoreV313
        ),
        "library": load(
            "compiled_concept_library_v313", CompiledConceptLibraryV313
        ),
        "report": load(
            "evidence_compiled_growth_report_v313",
            EvidenceCompiledGrowthReportV313,
        ),
        "manifest": load(
            "evidence_compiled_growth_manifest_v313",
            EvidenceCompiledGrowthManifestV313,
        ),
    }


@pytest.fixture(scope="module")
def development_artifacts() -> dict[str, object]:
    return _load_artifacts(DEVELOPMENT_V313)


@pytest.fixture(scope="module")
def confirmation_artifacts() -> dict[str, object]:
    return _load_artifacts(CONFIRMATION_V313)


def test_v313_development_remains_diagnostic_only(
    development_artifacts: dict[str, object],
) -> None:
    report = development_artifacts["report"]
    assert report.status == "evidence_concept_development_diagnostic_v313"
    assert report.phase == "development"
    assert not report.ready_for_concept_admission
    assert all(report.gates.values())
    assert report.performance_case_count == 30
    assert report.quality_case_count == 6
    assert report.concept_recovery_accuracy == 1.0
    assert report.pair_concept_consistency == 1.0
    assert report.maximum_pair_loss_difference == pytest.approx(
        0.04990929936816163
    )


def test_v313_confirmation_preserves_the_frozen_refutation(
    confirmation_artifacts: dict[str, object],
) -> None:
    report = confirmation_artifacts["report"]
    manifest = confirmation_artifacts["manifest"]
    assert report.status == "evidence_compiled_concepts_refuted_v313"
    assert report.phase == "confirmation"
    assert not report.ready_for_concept_admission
    assert [key for key, passed in report.gates.items() if not passed] == [
        "paired_prediction_invariance"
    ]
    assert report.performance_case_count == 42
    assert report.quality_case_count == 6
    assert report.baseline_expression_evaluation_count == 168
    assert report.candidate_expression_evaluation_count == 168
    assert report.concept_recovery_accuracy == 1.0
    assert report.pair_concept_consistency == 1.0
    assert report.maximum_pair_loss_difference == pytest.approx(
        0.054602806823069186
    )
    assert report.material_negative_transfer_count == 0
    assert report.material_negative_transfer_upper_95 <= 0.1
    assert not report.model_qualification_permitted
    assert not report.task_router_permitted
    assert not report.real_world_execution_permitted
    assert manifest.terminal_status == report.status


def test_v313_attempts_are_complete_equal_budget_and_decoy_free(
    confirmation_artifacts: dict[str, object],
) -> None:
    report = confirmation_artifacts["report"]
    bundle = confirmation_artifacts["bundle"]
    performance = [item for item in bundle.case_receipts if not item.quality_flags]
    quality = [item for item in bundle.case_receipts if item.quality_flags]
    attempts = [
        attempt
        for receipt in performance
        for attempt in receipt.baseline_attempts + receipt.candidate_attempts
    ]
    assert len(performance) == 42
    assert len(quality) == 6
    assert len(attempts) == 336
    assert len({attempt.attempt_hash for attempt in attempts}) == 336
    assert all(len(item.baseline_attempts) == 4 for item in performance)
    assert all(len(item.candidate_attempts) == 4 for item in performance)
    assert report.candidate_selection_counts == {
        "affine_rate_decoy": 0,
        "generalized_capacity_growth": 14,
        "hyperbolic_net_growth": 14,
        "log_capacity_growth": 14,
    }


def test_v313_refuted_store_is_explicitly_quarantined(
    confirmation_artifacts: dict[str, object],
) -> None:
    report = confirmation_artifacts["report"]
    experience = confirmation_artifacts["experience"]
    status = json.loads(ITERATION_STATUS.read_text(encoding="utf-8"))
    assert status["formal_status"] == report.status
    assert status["report_hash"] == report.report_hash
    assert status["experience_store_hash"] == experience.store_hash
    assert status["experience_store_status"] == "quarantined"
    assert not status["eligible_for_active_concept_retrieval"]
    assert not status["eligible_for_model_qualification"]
    assert status["repair_requires_new_version"]
    assert not status["confirmation_seeds_reusable"]


def test_v4_atomic_adapter_revokes_v313_partial_admission(
    confirmation_artifacts: dict[str, object],
) -> None:
    report = confirmation_artifacts["report"]
    experience = confirmation_artifacts["experience"]
    outcome = reconcile_atomic_concept_admission_v40(
        report,
        confirmation_artifacts["adjudication"],
        confirmation_artifacts["library"],
        experience,
        receipt_id="frozen_v313_atomic_rejection",
        created_at=report.created_at,
    )
    assert experience.active_concept_versions  # Frozen defect remains auditable.
    assert outcome.receipt.decision == "rejected"
    assert outcome.receipt.failed_gates == ["paired_prediction_invariance"]
    assert outcome.receipt.compensating_revocation_count == 3
    assert not outcome.experience_store.active_concept_versions
    assert verify_atomic_concept_admission_v40(
        outcome,
        report,
        confirmation_artifacts["adjudication"],
        confirmation_artifacts["library"],
        experience,
    )


def test_v4_atomic_adapter_commits_complete_passing_stage(
    confirmation_artifacts: dict[str, object],
) -> None:
    frozen = confirmation_artifacts["report"]
    gates = dict(frozen.gates)
    gates["paired_prediction_invariance"] = True
    passing = EvidenceCompiledGrowthReportV313.seal(
        **frozen.model_dump(
            exclude={"report_hash", "ready_for_concept_admission", "status", "gates"}
        ),
        gates=gates,
        ready_for_concept_admission=True,
        status="evidence_compiled_concepts_admitted_v313",
    )
    outcome = reconcile_atomic_concept_admission_v40(
        passing,
        confirmation_artifacts["adjudication"],
        confirmation_artifacts["library"],
        confirmation_artifacts["experience"],
        receipt_id="synthetic_atomic_commit_control",
        created_at=passing.created_at,
    )
    assert outcome.receipt.decision == "committed"
    assert outcome.receipt.committed_concept_versions == {
        "generalized_capacity_growth": 1,
        "hyperbolic_net_growth": 1,
        "log_capacity_growth": 1,
    }


def test_v313_replays_and_tampering_fails_closed(
    confirmation_artifacts: dict[str, object], tmp_path: Path,
) -> None:
    assert verify_evidence_compiled_growth_run_v313(
        CONFIRMATION_V313,
        source_v312_run_directory=SOURCE_V312,
        development_run_directory=DEVELOPMENT_V313,
    )
    copied = tmp_path / "tampered_v313"
    shutil.copytree(confirmation_artifacts["store"].run_directory, copied)
    events = [
        json.loads(line)
        for line in (copied / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    target = next(
        event["payload"]
        for event in events
        if event["event_type"] == "artifact_committed"
        and event["payload"]["kind"] == "evidence_compiled_growth_report_v313"
    )
    path = copied / target["relative_path"]
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["status"] = "tampered_status"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    assert not verify_evidence_compiled_growth_run_v313(
        copied,
        source_v312_run_directory=SOURCE_V312,
        development_run_directory=DEVELOPMENT_V313,
    )
