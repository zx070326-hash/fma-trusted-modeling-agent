from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from fma.hashing import sha256_value
from fma.v5.workspace_schemas import StageId
from fma.v5_2.ode_system import (
    ODEScientificBundleV52,
    ODEThresholdsV52,
    ODETimeSeriesSnapshotV52,
    build_ode_bundle_v52,
)
from fma.v6.decision_value import DecisionValueEvidenceV62
from fma.v6.executable_candidate import (
    ADAPTIVE_POSITIVE_SERIES_ADAPTER_ID,
    EXECUTABLE_CANDIDATE_INTENT_PATH,
    EXECUTABLE_CANDIDATE_IR_PATH,
    EXECUTABLE_CANDIDATE_RECEIPT_PATH,
    EXECUTABLE_CANDIDATE_RESOLUTION_PATH,
    SCALAR_ODE_ADAPTER_ID,
    ExecutableCandidateReceiptV62,
    ExecutableCandidateResolutionV62,
    RegisteredFamilySearchIRV62,
    RegisteredFamilySearchIntentV62,
    allowed_family_registry_hash_v62,
    build_executable_candidate_receipt_v62,
    registered_families_for_adapter_v62,
)
from fma.v6.provenance import DataProvenanceBindingV62
from fma.v6.scientific_closure import (
    LocalExternalQualificationBindingV62,
    StageEvidenceAdmissionV62,
    build_stage_evidence_admission_v62,
    evaluate_scientific_closure_v62,
    scientific_closure_summary_v62,
    verify_scientific_closure_v62,
)
from fma.v6.scientific_success import (
    ClaimKindV61,
    RollingConfirmationV61,
    ScientificSuccessDimensionV61,
    ScientificSuccessReportV61,
)


WORKSPACE_HASH = "a" * 64
CONTRACT_HASH = "b" * 64
ADAPTER_ID = "scalar_autonomous_ode_v52"
STAGES: tuple[StageId, ...] = ("S0", "S1", "S2", "S3", "S4", "S5", "S6")
DIMENSION_IDS = (
    "workflow_integrity",
    "data_provenance",
    "local_adapter_checks",
    "leakage_safe_confirmation",
    "decision_value",
    "mechanism_identification",
    "external_generalization",
    "scientific_qualification",
)


@lru_cache(maxsize=2)
def _execution_bundle(fixture_only: bool) -> ODEScientificBundleV52:
    times = np.arange(36, dtype=float)
    observations = (
        180.0 / (1.0 + 8.0 * np.exp(-0.16 * times))
    ).tolist()
    snapshot = ODETimeSeriesSnapshotV52.seal(
        task_id="closure-task",
        time_unit="year",
        state_unit="positive_index",
        times=times.tolist(),
        observations=observations,
        source_id="closure-test-series",
        fixture_only=fixture_only,
    )
    return build_ode_bundle_v52(
        snapshot=snapshot,
        thresholds=ODEThresholdsV52.seal(),
    )


SNAPSHOT_HASH = _execution_bundle(False).snapshot_hash


def test_summary_does_not_surface_unadmitted_provenance(
    tmp_path: Path,
) -> None:
    provenance_path = tmp_path / "data" / "source_provenance_v62" / "binding.json"
    provenance_path.parent.mkdir(parents=True)
    provenance_path.write_text("{}\n", encoding="utf-8", newline="\n")
    workspace = SimpleNamespace(
        root=tmp_path,
        _certificate_for_current_node=lambda stage: None,
        current_gate=lambda stage: None,
        verify_certificate=lambda certificate: False,
    )

    summary = scientific_closure_summary_v62(workspace)

    assert summary["evaluated"] is False
    assert summary["source_integrity_status"] == "NOT_RUN"
    assert summary["scientific_provenance_status"] == "NOT_RUN"
    assert summary["reason_codes"] == ["current_s2_provenance_not_admitted"]


def _gates() -> dict[StageId, str]:
    return {stage: sha256_value({"stage": stage}) for stage in STAGES}


def _v61_report(
    *,
    claim_kind: ClaimKindV61 = "predictive",
    fixture_only: bool = False,
) -> ScientificSuccessReportV61:
    bundle = _execution_bundle(fixture_only)
    actual_values_hash = sha256_value([10.0, 11.0, 12.0])
    confirmation = RollingConfirmationV61.seal(
        adapter_id=ADAPTER_ID,
        status="PASS",
        observation_count=36,
        requested_fold_count=6,
        completed_fold_count=6,
        admissible_fold_count=6,
        selected_model_ids=["logistic"] * 6,
        checks={"confirmation_passed": True},
        metrics={"confirmation_relative_rmse": 0.05},
        thresholds={"maximum_confirmation_relative_rmse": 0.20},
        reason_codes=[],
        actual_values_hash=actual_values_hash,
        prediction_values_hash=sha256_value([10.1, 11.1, 12.1]),
        persistence_values_hash=sha256_value([9.8, 10.8, 11.8]),
    )
    statuses = {
        "workflow_integrity": "PASS",
        "data_provenance": "NOT_RUN" if fixture_only else "HUMAN",
        "local_adapter_checks": "PASS",
        "leakage_safe_confirmation": "PASS",
        "decision_value": "NOT_RUN",
        "mechanism_identification": "NOT_RUN",
        "external_generalization": "NOT_RUN",
        "scientific_qualification": "NOT_RUN",
    }
    dimensions = [
        ScientificSuccessDimensionV61(
            dimension_id=dimension_id,
            status=statuses[dimension_id],
            required_for_claim=dimension_id
            in {
                "workflow_integrity",
                "data_provenance",
                "local_adapter_checks",
                "leakage_safe_confirmation",
            },
            reason_codes=[],
            evidence_refs=[],
        )
        for dimension_id in sorted(DIMENSION_IDS)
    ]
    return ScientificSuccessReportV61.seal(
        workspace_spec_hash=WORKSPACE_HASH,
        contract_hash=CONTRACT_HASH,
        adapter_id=ADAPTER_ID,
        claim_kind=claim_kind,
        current_gate_hashes=_gates(),
        adapter_binding_hash="d" * 64,
        scientific_bundle_hash=bundle.bundle_hash,
        fixture_only=fixture_only,
        dimensions=dimensions,
        rolling_confirmation=confirmation,
        local_predictive_gate_status="PASS",
        scientific_success_status="NOT_RUN" if fixture_only else "HUMAN",
        claim_ceiling=(
            "fixture_protocol_only"
            if fixture_only
            else "local_leakage_safe_predictive_evidence"
        ),
    )


def _provenance(
    *,
    snapshot_hash: str | None = None,
    fixture_only: bool = False,
) -> DataProvenanceBindingV62:
    if snapshot_hash is None:
        snapshot_hash = _execution_bundle(fixture_only).snapshot_hash
    return DataProvenanceBindingV62.seal(
        workspace_spec_hash=WORKSPACE_HASH,
        s1_gate_hash=_gates()["S1"],
        s2_attempt=1,
        raw_baseline_hash="0" * 64,
        raw_tree_hash="1" * 64,
        ledger_hash="2" * 64,
        processed_snapshot_hash=snapshot_hash,
        processed_file_hash="3" * 64,
        transform_script_hash="4" * 64,
        transform_receipt_hash="b" * 64,
        transform_params_hashes=["5" * 64],
        source_contract_hash="6" * 64,
        source_receipt_hash="7" * 64,
        source_verification_hash="8" * 64,
        source_acquisition_authority_receipt_hash="c" * 64,
        s2_source_reverification_receipt_hash="d" * 64,
        source_acquisition_authority_key_id="fixture-source-acquisition",
        source_reverification_authority_key_id="fixture-source-reverify",
        source_acquisition_authority_mode="external_hmac",
        source_reverification_authority_mode="v5_workspace_hmac",
        source_transport_mode=(
            "fixture_injected"
            if fixture_only
            else "live_https_no_redirect"
        ),
        official_live_transport_authenticated=not fixture_only,
        source_raw_hash="9" * 64,
        measurement_schema_hash="a" * 64,
        fixture_only=fixture_only,
        status="PASS",
        checks={
            "authenticated_raw_baseline_current": True,
            "official_source_binding_exact": True,
            "processed_snapshot_exact": True,
            "source_acquisition_authority_authenticated": True,
            "current_s2_source_reverification_authenticated": True,
        },
        scientific_provenance_status=(
            "NOT_RUN" if fixture_only else "HUMAN"
        ),
        reason_codes=[],
    )


def _decision(
    report: ScientificSuccessReportV61,
) -> DecisionValueEvidenceV62:
    return DecisionValueEvidenceV62.seal(
        contract_hash="3" * 64,
        success_contract_hash=report.contract_hash,
        snapshot_hash=_execution_bundle(report.fixture_only).snapshot_hash,
        adapter_id=report.adapter_id,
        fixture_only=report.fixture_only,
        status="PASS",
        scientific_decision_status=(
            "NOT_RUN" if report.fixture_only else "HUMAN"
        ),
        requested_fold_count=6,
        completed_fold_count=6,
        admissible_fold_count=6,
        completed_origin_indices=[30, 31, 32, 33, 34, 35],
        training_snapshot_hashes=[
            sha256_value({"fold": index}) for index in range(6)
        ],
        model_action_hashes=[
            sha256_value({"action": index}) for index in range(6)
        ],
        checks={
            "all_requested_folds_completed": True,
            "all_inner_selections_admissible": True,
            "persistence_decision_loss_improved": True,
            "mean_normalized_regret_bounded": True,
        },
        metrics={
            "relative_loss_improvement": 0.25,
            "mean_normalized_regret": 0.04,
        },
        thresholds={
            "minimum_relative_loss_improvement": 0.05,
            "maximum_mean_normalized_regret": 0.20,
        },
        reason_codes=[],
        actual_values_hash=report.rolling_confirmation.actual_values_hash,
        model_actions_hash=sha256_value([10.1, 11.1, 12.1]),
        baseline_actions_hash=sha256_value([9.8, 10.8, 11.8]),
    )


def _workspace_and_admission(
    *,
    root: Path,
    report: ScientificSuccessReportV61,
    provenance: DataProvenanceBindingV62,
    decision: DecisionValueEvidenceV62 | None = None,
) -> tuple[SimpleNamespace, StageEvidenceAdmissionV62]:
    bundle = _execution_bundle(report.fixture_only)
    intent = RegisteredFamilySearchIntentV62(
        candidate_id="candidate.branch_a",
        allowed_adapter_ids=[
            SCALAR_ODE_ADAPTER_ID,
            ADAPTIVE_POSITIVE_SERIES_ADAPTER_ID,
        ],
    )
    candidate_structural_hash = sha256_value(
        {"candidate": intent.candidate_id}
    )
    execution_ir = RegisteredFamilySearchIRV62.seal(
        candidate_id=intent.candidate_id,
        candidate_structural_hash=candidate_structural_hash,
        model_intent_hash=intent.content_hash(),
        allowed_adapter_ids=intent.allowed_adapter_ids,
    )
    gates = _gates()
    certificates: dict[str, SimpleNamespace] = {}
    workspace = SimpleNamespace(
        root=root,
        spec=SimpleNamespace(
            spec_hash=WORKSPACE_HASH,
            workspace_id="closure-task",
        ),
        gate_hashes=gates,
        certificates=certificates,
        current_gate=lambda stage: gates.get(stage),
        _certificate_for_current_node=lambda stage: certificates.get(stage),
        _latest_attempt=lambda stage: 1,
        verify_certificate=lambda certificate: certificate is not None,
        verify=lambda: True,
    )
    resolution = ExecutableCandidateResolutionV62.seal(
        workspace_spec_hash=WORKSPACE_HASH,
        s1_gate_hash=gates["S1"],
        s2_attempt=1,
        execution_ir_hash=execution_ir.ir_hash,
        model_spec_hash=sha256_value({"model": intent.candidate_id}),
        selected_candidate_id=intent.candidate_id,
        selected_candidate_structural_hash=candidate_structural_hash,
        adapter_id=SCALAR_ODE_ADAPTER_ID,
        allowed_families=list(
            registered_families_for_adapter_v62(SCALAR_ODE_ADAPTER_ID)
        ),
        allowed_family_registry_hash=allowed_family_registry_hash_v62(
            SCALAR_ODE_ADAPTER_ID
        ),
    )
    receipt = build_executable_candidate_receipt_v62(
        workspace=workspace,
        resolution=resolution,
        bundle=bundle,
    )

    execution_intent_path = root / EXECUTABLE_CANDIDATE_INTENT_PATH
    execution_ir_path = root / EXECUTABLE_CANDIDATE_IR_PATH
    execution_resolution_path = (
        root / EXECUTABLE_CANDIDATE_RESOLUTION_PATH
    )
    execution_receipt_path = root / EXECUTABLE_CANDIDATE_RECEIPT_PATH
    bundle_path = root / "results" / "ode_scientific_bundle.json"
    provenance_path = (
        root / "data" / "source_provenance_v62" / "binding.json"
    )
    rolling_path = root / "results" / "rolling_confirmation_v61.json"
    artifacts = {
        execution_intent_path: intent,
        execution_ir_path: execution_ir,
        execution_resolution_path: resolution,
        execution_receipt_path: receipt,
        bundle_path: bundle,
        provenance_path: provenance,
        rolling_path: report.rolling_confirmation,
    }
    for path, artifact in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(artifact.model_dump_json(), encoding="utf-8")

    def file_binding(relative_path: str, path: Path) -> SimpleNamespace:
        return SimpleNamespace(
            relative_path=relative_path,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            size_bytes=path.stat().st_size,
        )

    certificates.update(
        {
        "S1": SimpleNamespace(
            certificate_hash=gates["S1"],
            attempt=1,
            manifest=SimpleNamespace(
                manifest_hash=sha256_value({"manifest": "S1"}),
                files=[
                    file_binding(
                        EXECUTABLE_CANDIDATE_INTENT_PATH,
                        execution_intent_path,
                    ),
                    file_binding(
                        EXECUTABLE_CANDIDATE_IR_PATH,
                        execution_ir_path,
                    ),
                ],
            ),
        ),
        "S2": SimpleNamespace(
            certificate_hash=gates["S2"],
            attempt=1,
            manifest=SimpleNamespace(
                manifest_hash=sha256_value({"manifest": "S2"}),
                files=[
                    file_binding(
                        EXECUTABLE_CANDIDATE_RESOLUTION_PATH,
                        execution_resolution_path,
                    ),
                    file_binding(
                        "data/source_provenance_v62/binding.json",
                        provenance_path,
                    )
                ],
            ),
        ),
        "S3": SimpleNamespace(
            certificate_hash=gates["S3"],
            attempt=1,
            manifest=SimpleNamespace(
                manifest_hash=sha256_value({"manifest": "S3"}),
                files=[
                    file_binding(
                        EXECUTABLE_CANDIDATE_RECEIPT_PATH,
                        execution_receipt_path,
                    ),
                    file_binding(
                        "results/ode_scientific_bundle.json",
                        bundle_path,
                    ),
                ],
            ),
        ),
        "S4": SimpleNamespace(
            certificate_hash=gates["S4"],
            attempt=1,
            manifest=SimpleNamespace(
                manifest_hash=sha256_value({"manifest": "S4"}),
                files=[
                    file_binding(
                        "results/rolling_confirmation_v61.json",
                        rolling_path,
                    )
                ],
            ),
        ),
        }
    )
    if decision is not None:
        decision_path = root / "results" / "decision_value_evidence_v62.json"
        decision_path.write_text(
            decision.model_dump_json(),
            encoding="utf-8",
        )
        certificates["S5"] = SimpleNamespace(
            certificate_hash=gates["S5"],
            attempt=1,
            manifest=SimpleNamespace(
                manifest_hash=sha256_value({"manifest": "S5"}),
                files=[
                    file_binding(
                        "results/decision_value_evidence_v62.json",
                        decision_path,
                    )
                ],
            ),
        )
    admission = build_stage_evidence_admission_v62(
        workspace=workspace,  # type: ignore[arg-type]
        v61_report=report,
        provenance=provenance,
        decision_evidence=decision,
    )
    return workspace, admission


def _external_binding(
    *,
    report: ScientificSuccessReportV61,
    provenance: DataProvenanceBindingV62,
    admission: StageEvidenceAdmissionV62,
    decision: DecisionValueEvidenceV62 | None = None,
) -> LocalExternalQualificationBindingV62:
    return LocalExternalQualificationBindingV62.seal(
        workspace_spec_hash=WORKSPACE_HASH,
        v61_report_hash=report.report_hash,
        claim_kind=report.claim_kind,
        current_gate_hashes=_gates(),
        s2_attempt=1,
        source_snapshot_hash=provenance.processed_snapshot_hash,
        source_verification_hash=provenance.source_verification_hash,
        provenance_binding_hash=provenance.binding_hash,
        stage_admission_hash=admission.admission_hash,
        decision_evidence_hash=decision.evidence_hash if decision else None,
        external_artifact_hash="4" * 64,
    )


def _dimension_statuses(report: object) -> dict[str, str]:
    return {
        item.dimension_id: item.status  # type: ignore[attr-defined]
        for item in report.dimensions  # type: ignore[attr-defined]
    }


def test_predictive_mechanical_evidence_does_not_close_scientific_claim(
    tmp_path: Path,
) -> None:
    v61 = _v61_report()
    provenance = _provenance()
    workspace, admission = _workspace_and_admission(
        root=tmp_path,
        report=v61,
        provenance=provenance,
    )

    report = evaluate_scientific_closure_v62(
        workspace=workspace,  # type: ignore[arg-type]
        v61_report=v61,
        provenance=provenance,
        stage_admission=admission,
    )

    statuses = _dimension_statuses(report)
    assert statuses["data_provenance"] == "HUMAN"
    assert statuses["external_generalization"] == "NOT_RUN"
    assert statuses["scientific_qualification"] == "NOT_RUN"
    assert report.local_evidence_status == "HUMAN"
    assert report.scientific_closure_status == "NOT_RUN"
    assert report.claim_ceiling == "workflow_integrity_only"
    assert report.scientific_qualification_granted is False
    assert report.real_world_action_authorized is False
    assert "external_generalization" in report.closure_required_dimension_ids
    assert "scientific_qualification" in report.closure_required_dimension_ids

    verification = verify_scientific_closure_v62(
        workspace=workspace,  # type: ignore[arg-type]
        report=report,
        v61_report=v61,
        provenance=provenance,
        stage_admission=admission,
    )
    assert verification.status == "PASS"
    assert all(verification.checks.values())

    unsealed = type(report)(
        **report.model_dump(exclude={"report_hash"})
    )
    unsealed_verification = verify_scientific_closure_v62(
        workspace=workspace,  # type: ignore[arg-type]
        report=unsealed,
        v61_report=v61,
        provenance=provenance,
        stage_admission=admission,
    )
    assert unsealed_verification.status == "FAIL"
    assert unsealed_verification.checks["report_self_hash"] is False


def test_fixture_cannot_be_promoted_even_with_external_artifact_binding(
    tmp_path: Path,
) -> None:
    v61 = _v61_report(fixture_only=True)
    provenance = _provenance(fixture_only=True)
    workspace, admission = _workspace_and_admission(
        root=tmp_path,
        report=v61,
        provenance=provenance,
    )
    binding = _external_binding(
        report=v61,
        provenance=provenance,
        admission=admission,
    )

    report = evaluate_scientific_closure_v62(
        workspace=workspace,  # type: ignore[arg-type]
        v61_report=v61,
        provenance=provenance,
        stage_admission=admission,
        external_binding=binding,
    )

    statuses = _dimension_statuses(report)
    assert statuses["data_provenance"] == "NOT_RUN"
    assert statuses["external_generalization"] == "NOT_RUN"
    assert statuses["scientific_qualification"] == "NOT_RUN"
    assert report.scientific_closure_status == "NOT_RUN"
    assert report.claim_ceiling == "fixture_protocol_only"
    assert report.scientific_qualification_granted is False


def test_prescriptive_claim_requires_bound_decision_evidence(
    tmp_path: Path,
) -> None:
    v61 = _v61_report(claim_kind="prescriptive")
    provenance = _provenance()
    absent_workspace, absent_admission = _workspace_and_admission(
        root=tmp_path / "absent",
        report=v61,
        provenance=provenance,
    )

    absent = evaluate_scientific_closure_v62(
        workspace=absent_workspace,  # type: ignore[arg-type]
        v61_report=v61,
        provenance=provenance,
        stage_admission=absent_admission,
    )
    assert absent.local_evidence_status == "NOT_RUN"
    assert _dimension_statuses(absent)["decision_value"] == "NOT_RUN"

    decision = _decision(v61)
    decision_workspace, decision_admission = _workspace_and_admission(
        root=tmp_path / "present",
        report=v61,
        provenance=provenance,
        decision=decision,
    )
    present = evaluate_scientific_closure_v62(
        workspace=decision_workspace,  # type: ignore[arg-type]
        v61_report=v61,
        provenance=provenance,
        stage_admission=decision_admission,
        decision_evidence=decision,
    )
    assert present.local_evidence_status == "HUMAN"
    assert _dimension_statuses(present)["decision_value"] == "HUMAN"
    assert present.scientific_closure_status == "NOT_RUN"
    assert present.decision_evidence_hash == decision.evidence_hash

    substituted = DecisionValueEvidenceV62.seal(
        **{
            **decision.model_dump(exclude={"evidence_hash"}),
            "snapshot_hash": "f" * 64,
        }
    )
    with pytest.raises(ValueError, match="manifests|another snapshot"):
        evaluate_scientific_closure_v62(
            workspace=decision_workspace,  # type: ignore[arg-type]
            v61_report=v61,
            provenance=provenance,
            stage_admission=decision_admission,
            decision_evidence=substituted,
        )


@pytest.mark.parametrize(
    ("claim_kind", "required_dimension", "local_status"),
    [
        ("descriptive", "leakage_safe_confirmation", "HUMAN"),
        ("mechanistic", "mechanism_identification", "NOT_RUN"),
        ("generalization", "external_generalization", "HUMAN"),
    ],
)
def test_claim_kind_controls_required_dimensions(
    tmp_path: Path,
    claim_kind: ClaimKindV61,
    required_dimension: str,
    local_status: str,
) -> None:
    v61 = _v61_report(claim_kind=claim_kind)
    provenance = _provenance()
    workspace, admission = _workspace_and_admission(
        root=tmp_path,
        report=v61,
        provenance=provenance,
    )
    report = evaluate_scientific_closure_v62(
        workspace=workspace,  # type: ignore[arg-type]
        v61_report=v61,
        provenance=provenance,
        stage_admission=admission,
    )

    if claim_kind == "descriptive":
        assert required_dimension not in report.local_required_dimension_ids
    elif claim_kind == "generalization":
        assert required_dimension in report.closure_required_dimension_ids
        assert required_dimension not in report.local_required_dimension_ids
    else:
        assert required_dimension in report.local_required_dimension_ids
    assert report.local_evidence_status == local_status
    assert report.scientific_closure_status == "NOT_RUN"


def test_stale_gate_and_substituted_snapshot_fail_closed(
    tmp_path: Path,
) -> None:
    v61 = _v61_report()
    provenance = _provenance()
    workspace, admission = _workspace_and_admission(
        root=tmp_path,
        report=v61,
        provenance=provenance,
    )
    report = evaluate_scientific_closure_v62(
        workspace=workspace,  # type: ignore[arg-type]
        v61_report=v61,
        provenance=provenance,
        stage_admission=admission,
    )
    workspace.gate_hashes["S6"] = "f" * 64

    verification = verify_scientific_closure_v62(
        workspace=workspace,  # type: ignore[arg-type]
        report=report,
        v61_report=v61,
        provenance=provenance,
        stage_admission=admission,
    )
    assert verification.status == "FAIL"
    assert verification.checks["gate_binding_current"] is False
    assert verification.checks["deterministic_recomputation"] is False

    with pytest.raises(ValueError, match="not current|stale"):
        evaluate_scientific_closure_v62(
            workspace=workspace,  # type: ignore[arg-type]
            v61_report=v61,
            provenance=provenance,
            stage_admission=admission,
        )
    workspace.gate_hashes["S6"] = _gates()["S6"]
    substituted_provenance = DataProvenanceBindingV62.seal(
        **{
            **provenance.model_dump(exclude={"binding_hash"}),
            "processed_snapshot_hash": "0" * 64,
        }
    )
    with pytest.raises(ValueError, match="manifests|another snapshot"):
        evaluate_scientific_closure_v62(
            workspace=workspace,  # type: ignore[arg-type]
            v61_report=v61,
            provenance=substituted_provenance,
            stage_admission=admission,
        )
    workspace.certificates["S2"].attempt = 2
    with pytest.raises(ValueError, match="current S2 attempt"):
        evaluate_scientific_closure_v62(
            workspace=workspace,  # type: ignore[arg-type]
            v61_report=v61,
            provenance=provenance,
            stage_admission=admission,
        )


def test_local_external_binding_cannot_encode_signature_authority(
    tmp_path: Path,
) -> None:
    v61 = _v61_report()
    provenance = _provenance()
    workspace, admission = _workspace_and_admission(
        root=tmp_path,
        report=v61,
        provenance=provenance,
    )

    with pytest.raises(ValidationError):
        LocalExternalQualificationBindingV62(
            workspace_spec_hash=WORKSPACE_HASH,
            v61_report_hash=v61.report_hash,
            claim_kind=v61.claim_kind,
            current_gate_hashes=_gates(),
            s2_attempt=1,
            source_snapshot_hash=SNAPSHOT_HASH,
            source_verification_hash=provenance.source_verification_hash,
            provenance_binding_hash=provenance.binding_hash,
            stage_admission_hash=admission.admission_hash,
            independent_signature_verified=True,
        )

    binding = _external_binding(
        report=v61,
        provenance=provenance,
        admission=admission,
    )
    report = evaluate_scientific_closure_v62(
        workspace=workspace,  # type: ignore[arg-type]
        v61_report=v61,
        provenance=provenance,
        stage_admission=admission,
        external_binding=binding,
    )
    statuses = _dimension_statuses(report)
    assert statuses["external_generalization"] == "NOT_RUN"
    assert statuses["scientific_qualification"] == "NOT_RUN"
    assert report.scientific_qualification_granted is False


def test_stage_admission_is_derived_from_current_manifest_files(
    tmp_path: Path,
) -> None:
    v61 = _v61_report()
    provenance = _provenance()
    workspace, admission = _workspace_and_admission(
        root=tmp_path,
        report=v61,
        provenance=provenance,
    )
    assert admission.status == "PASS"
    assert admission.stage_manifest_hashes.keys() == {
        "S1",
        "S2",
        "S3",
        "S4",
    }
    assert admission.executable_candidate_receipt_hash != "0" * 64
    assert admission.scientific_bundle_hash == v61.scientific_bundle_hash
    assert admission.checks["execution_receipt_replayed"] is True

    rolling_path = (
        tmp_path / "results" / "rolling_confirmation_v61.json"
    )
    rolling_path.write_text("{}\n", encoding="utf-8")
    stale = build_stage_evidence_admission_v62(
        workspace=workspace,  # type: ignore[arg-type]
        v61_report=v61,
        provenance=provenance,
    )
    assert stale.status == "FAIL"
    assert "rolling_confirmation_file_exact" in stale.reason_codes

    with pytest.raises(ValueError, match="not admitted"):
        evaluate_scientific_closure_v62(
            workspace=workspace,  # type: ignore[arg-type]
            v61_report=v61,
            provenance=provenance,
            stage_admission=stale,
        )


@pytest.mark.parametrize(
    ("relative_path", "reason_code"),
    [
        (
            EXECUTABLE_CANDIDATE_INTENT_PATH,
            "execution_intent_file_exact",
        ),
        (
            EXECUTABLE_CANDIDATE_IR_PATH,
            "execution_ir_file_exact",
        ),
        (
            EXECUTABLE_CANDIDATE_RESOLUTION_PATH,
            "execution_resolution_file_exact",
        ),
        (
            EXECUTABLE_CANDIDATE_RECEIPT_PATH,
            "execution_receipt_file_exact",
        ),
        (
            "results/ode_scientific_bundle.json",
            "scientific_bundle_file_exact",
        ),
    ],
)
def test_executable_admission_fails_closed_on_missing_or_forged_artifact(
    tmp_path: Path,
    relative_path: str,
    reason_code: str,
) -> None:
    v61 = _v61_report()
    provenance = _provenance()
    workspace, admission = _workspace_and_admission(
        root=tmp_path,
        report=v61,
        provenance=provenance,
    )
    assert admission.status == "PASS"

    target = tmp_path / relative_path
    target.write_text("{}\n", encoding="utf-8", newline="\n")
    failed = build_stage_evidence_admission_v62(
        workspace=workspace,  # type: ignore[arg-type]
        v61_report=v61,
        provenance=provenance,
    )

    assert failed.status == "FAIL"
    assert reason_code in failed.reason_codes
    assert failed.checks["execution_receipt_replayed"] is False


def test_executable_admission_replays_even_authenticated_receipt_bytes(
    tmp_path: Path,
) -> None:
    v61 = _v61_report()
    provenance = _provenance()
    workspace, admission = _workspace_and_admission(
        root=tmp_path,
        report=v61,
        provenance=provenance,
    )
    assert admission.status == "PASS"
    receipt_path = tmp_path / EXECUTABLE_CANDIDATE_RECEIPT_PATH
    receipt = ExecutableCandidateReceiptV62.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    )
    forged = ExecutableCandidateReceiptV62.seal(
        **{
            **receipt.model_dump(exclude={"receipt_hash"}),
            "s2_gate_hash": "f" * 64,
        }
    )
    receipt_path.write_text(forged.model_dump_json(), encoding="utf-8")
    binding = next(
        item
        for item in workspace.certificates["S3"].manifest.files
        if item.relative_path == EXECUTABLE_CANDIDATE_RECEIPT_PATH
    )
    binding.sha256 = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    binding.size_bytes = receipt_path.stat().st_size

    failed = build_stage_evidence_admission_v62(
        workspace=workspace,  # type: ignore[arg-type]
        v61_report=v61,
        provenance=provenance,
    )

    assert failed.status == "FAIL"
    assert failed.checks["execution_receipt_file_exact"] is True
    assert failed.checks["execution_receipt_replayed"] is False
