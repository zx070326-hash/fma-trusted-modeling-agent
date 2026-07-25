from __future__ import annotations

import inspect
import json
import shutil
from pathlib import Path

import pytest

from fma.schemas import ArtifactRef
from fma.storage import RunStore
from fma.v3.open_set_concept_evolution_v312 import (
    ConceptEvolutionBundleV312,
    ConceptEvolutionManifestV312,
    ConceptEvolutionReportV312,
    PrivateConceptWorldPackSpecV312,
    PrivateConceptWorldPackV312,
    PublicConceptProtocolV312,
    PublicConceptWorldPackV312,
    execute_concept_evolution_v312,
    verify_concept_evolution_run_v312,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_V311 = (
    ROOT / "experiments/iteration_19/v311_representation_topology_confirmation"
)
DEVELOPMENT_V312 = (
    ROOT / "experiments/iteration_20/v312_open_set_concept_development"
)
CONFIRMATION_V312 = (
    ROOT / "experiments/iteration_20/v312_open_set_concept_confirmation"
)


def _load_artifacts(run_directory: Path) -> dict[str, object]:
    store = RunStore.open_existing(run_directory)
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
        "refs": refs,
        "protocol": load(
            "public_concept_protocol_v312", PublicConceptProtocolV312
        ),
        "spec": load(
            "private_concept_worldpack_spec_v312",
            PrivateConceptWorldPackSpecV312,
        ),
        "public_pack": load(
            "public_concept_worldpack_v312", PublicConceptWorldPackV312
        ),
        "private_pack": load(
            "private_concept_worldpack_v312", PrivateConceptWorldPackV312
        ),
        "bundle": load(
            "concept_evolution_bundle_v312", ConceptEvolutionBundleV312
        ),
        "report": load(
            "concept_evolution_report_v312", ConceptEvolutionReportV312
        ),
        "manifest": load(
            "concept_evolution_manifest_v312", ConceptEvolutionManifestV312
        ),
    }


@pytest.fixture(scope="module")
def development_artifacts() -> dict[str, object]:
    return _load_artifacts(DEVELOPMENT_V312)


@pytest.fixture(scope="module")
def confirmation_artifacts() -> dict[str, object]:
    return _load_artifacts(CONFIRMATION_V312)


def test_v312_public_executor_is_private_blind_and_typed(
    confirmation_artifacts: dict[str, object],
) -> None:
    parameters = inspect.signature(execute_concept_evolution_v312).parameters
    assert list(parameters)[:3] == ["public_protocol", "public_pack", "policy"]
    assert all("private" not in name for name in parameters)
    public_pack = confirmation_artifacts["public_pack"]
    public_payload = public_pack.model_dump(mode="json")
    assert not any(
        key in json.dumps(public_payload)
        for key in (
            "gompertz_open_set",
            "pendulum_open_set",
            "anonymous_scaled_permuted",
            "private_probe",
            "expected_concept",
        )
    )
    assert all(
        case.state_names
        == [f"z{index}" for index in range(len(case.state_names))]
        and not case.semantic_state_labels_available
        and not case.representation_metadata_available
        for case in public_pack.cases
    )


def test_v312_control_arm_is_not_erased_by_admission_threshold(
    development_artifacts: dict[str, object],
) -> None:
    bundle = development_artifacts["bundle"]
    threshold_misses = [
        receipt
        for receipt in bundle.case_receipts
        if not receipt.quality_flags
        and not any(attempt.valid for attempt in receipt.baseline_attempts)
    ]
    assert threshold_misses
    assert all(
        receipt.baseline_decision.selected_concept is not None
        and receipt.candidate_decision.selected_concept is not None
        for receipt in threshold_misses
    )


def test_v312_development_remains_diagnostic_only(
    development_artifacts: dict[str, object],
) -> None:
    report = development_artifacts["report"]
    assert report.phase == "development"
    assert report.status == "concept_evolution_development_diagnostic_v312"
    assert not report.ready_for_concept_admission
    assert report.performance_case_count == 20
    assert report.quality_case_count == 4
    assert report.material_negative_transfer_count == 0
    assert [key for key, passed in report.gates.items() if not passed] == [
        "material_negative_transfer_controlled"
    ]


def test_v312_confirmation_admits_only_privately_supported_concepts(
    confirmation_artifacts: dict[str, object],
) -> None:
    report = confirmation_artifacts["report"]
    manifest = confirmation_artifacts["manifest"]
    spec = confirmation_artifacts["spec"]
    development_report = _load_artifacts(DEVELOPMENT_V312)["report"]
    assert report.phase == "confirmation"
    assert report.status == "open_set_concepts_admitted_v312"
    assert report.ready_for_concept_admission
    assert all(report.gates.values())
    assert report.performance_case_count == 36
    assert report.quality_case_count == 4
    assert report.concept_recovery_accuracy == 1.0
    assert report.pair_concept_consistency == 1.0
    assert report.material_negative_transfer_count == 0
    assert report.material_negative_transfer_upper_95 <= 0.1
    assert report.candidate_selection_counts["logarithmic_rate"] == 18
    assert report.candidate_selection_counts["periodic_restoring_force"] == 18
    assert all(
        report.candidate_selection_counts[concept] == 0
        for concept in (
            "saturating_rate_decoy",
            "scalar_affine_decoy",
            "kinematic_cubic_decoy",
            "uncoupled_linear_decoy",
        )
    )
    assert [entry.status for entry in report.concept_ledger[:2]] == [
        "admitted_private_confirmation",
        "admitted_private_confirmation",
    ]
    assert all(
        entry.status == "rejected_private_confirmation"
        for entry in report.concept_ledger[2:]
    )
    assert spec.development_report_hash == development_report.report_hash
    assert manifest.terminal_status == report.status
    assert len(manifest.artifact_refs) == 9


def test_v312_attempt_buffer_is_complete_equal_budget_and_private_free(
    confirmation_artifacts: dict[str, object],
) -> None:
    bundle = confirmation_artifacts["bundle"]
    report = confirmation_artifacts["report"]
    performance = [item for item in bundle.case_receipts if not item.quality_flags]
    quality = [item for item in bundle.case_receipts if item.quality_flags]
    attempts = [
        attempt
        for receipt in performance
        for attempt in receipt.baseline_attempts + receipt.candidate_attempts
    ]
    assert len(performance) == 36
    assert len(quality) == 4
    assert len(attempts) == 288
    assert len({item.attempt_hash for item in attempts}) == 288
    assert all(receipt.all_attempts_persisted for receipt in performance)
    assert all(
        not attempt.private_values_used
        and not attempt.proposal.private_values_used
        and not attempt.proposal.arbitrary_code_used
        for attempt in attempts
    )
    assert report.baseline_expression_evaluation_count == 144
    assert report.candidate_expression_evaluation_count == 144
    assert report.equal_evaluation_budget
    assert report.all_attempts_persisted


def test_v312_replays_and_tampering_fails_closed(
    confirmation_artifacts: dict[str, object], tmp_path: Path,
) -> None:
    assert verify_concept_evolution_run_v312(
        CONFIRMATION_V312,
        source_run_directory=SOURCE_V311,
        development_run_directory=DEVELOPMENT_V312,
    )
    copied = tmp_path / "tampered_v312"
    shutil.copytree(confirmation_artifacts["store"].run_directory, copied)
    events = [
        json.loads(line)
        for line in (copied / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    target = next(
        event["payload"]
        for event in events
        if event["event_type"] == "artifact_committed"
        and event["payload"]["kind"] == "concept_evolution_bundle_v312"
    )
    path = copied / target["relative_path"]
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["bundle_id"] = "tampered_bundle"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    assert not verify_concept_evolution_run_v312(
        copied,
        source_run_directory=SOURCE_V311,
        development_run_directory=DEVELOPMENT_V312,
    )
