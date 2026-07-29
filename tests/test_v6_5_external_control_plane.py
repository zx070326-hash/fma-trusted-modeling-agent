"""Additive V6.5 single-writer external-qualification controls.

The installed model and keys are test controls.  These tests establish local
control-plane behavior only; they do not establish external independence or
scientific qualification.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

import pytest

from fma.hashing import sha256_value
from fma.v5.stage_workspace import StageWorkspaceV50
from fma.v5.workspace_schemas import PredictionSealV50
import fma.v6.external_control_plane as control_module
import fma.v6.external_qualification_cli as control_cli
import fma.v6.external_prediction_runtime as prediction_runtime
from fma.v6.external_qualification_coordinator import (
    ExternalQualificationCoordinatorError,
    ExternalQualificationCoordinatorV63,
)
from fma.v6.external_control_plane import (
    CONTROL_FAILURE_KIND_V65,
    CONTROL_RESOLUTION_KIND_V65,
    ExternalControlOperationFailedV65,
    ExternalControlPlaneErrorV65,
    ExternalControlRuntimeIdentityV65,
    ExternalQualificationControlPlaneV65,
    issue_external_control_principal_v65,
)
import fma.v6.external_qualification as qualification
import fma.v6 as v6_public

import test_v6_3_external_prediction_runtime as runtime_control


def _setup_input_frozen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    StageWorkspaceV50,
    dict[str, bytes],
    dict[str, bytes],
    Any,
    Any,
    Any,
]:
    workspace = runtime_control._new_workspace(tmp_path / "task")
    _snapshot, _bundle, _receipt, summary = (
        runtime_control._install_real_ode_model(workspace)
    )
    private_keys, public_keys = runtime_control._keys()
    monkeypatch.setattr(
        qualification,
        "scientific_closure_summary_v62",
        lambda _workspace: summary,
    )
    monkeypatch.setattr(
        StageWorkspaceV50,
        "current_gate",
        lambda _workspace, stage: runtime_control.GATES.get(stage),
    )
    contract = (
        qualification.freeze_predictive_external_qualification_contract_v63(
            workspace=workspace,
            qualification_id="qual.v65-control",
            task_id=workspace.spec.workspace_id,
            maximum_metric_value=0.20,
            minimum_external_observation_count=12,
            coordinator_host_id="coordinator-host",
            generator_host_id="generator-host",
            custody_key_id="custody-key",
            registry_key_id="registry-key",
            evaluator_key_id="evaluator-key",
            promotion_key_id="promotion-key",
            trusted_public_keys=public_keys,
            frozen_at=runtime_control.T0,
        )
    )
    target_ids = [f"target.{index:02d}" for index in range(12)]
    forecast_input = qualification.commit_external_forecast_input_v63(
        workspace=workspace,
        contract=contract,
        target_ids=target_ids,
        forecast_times=[36.0 + index for index in range(12)],
        frozen_at=runtime_control.T0 + timedelta(minutes=1),
    )
    custody = qualification.sign_external_evidence_custody_v63(
        private_key_pem=private_keys["custody-key"],
        qualification_id=contract.qualification_id,
        contract_hash=contract.contract_hash,
        local_context_hash=contract.local_context_hash,
        v62_report_hash=contract.v62_report_hash,
        external_snapshot_hash=forecast_input.input_hash,
        holdout_commitment_hash=sha256_value({"private": "holdout"}),
        normalization_scale_commitment_hash=sha256_value(
            {
                "holdout_observation_count": 12,
                "target_squared_value_sum": 1200.0,
            }
        ),
        target_order_hash=forecast_input.target_order_hash,
        holdout_observation_count=12,
        fixture_only=False,
        measurement_protocol_hash=sha256_value(
            {"measurement": "protocol"}
        ),
        measurement_review_hash=sha256_value(
            {"review": "independent"}
        ),
        external_environment_hash=sha256_value(
            {"environment": "external"}
        ),
        strict_unseen_verified=True,
        independent_measurement_review_passed=True,
        external_environment_verified=True,
        holdout_frozen_before_prediction=True,
        custodian_host_id="custodian-host",
        coordinator_host_id=contract.coordinator_host_id,
        generator_host_id=contract.generator_host_id,
        attested_at=runtime_control.T0 + timedelta(minutes=2),
        custody_key_id="custody-key",
    )
    return (
        workspace,
        public_keys,
        private_keys,
        contract,
        forecast_input,
        custody,
    )


def _control(
    workspace: StageWorkspaceV50,
    public_keys: dict[str, bytes],
    *,
    activate: bool = True,
) -> ExternalQualificationControlPlaneV65:
    issued_at = datetime.now(timezone.utc)
    principal = issue_external_control_principal_v65(
        workspace=workspace,
        principal_id="test-control-principal",
        actor_type="operator",
        qualification_id="qual.v65-control",
        issued_at=issued_at,
        expires_at=issued_at + timedelta(hours=1),
    )
    control = ExternalQualificationControlPlaneV65(
        workspace.root,
        authority_key=runtime_control.AUTHORITY_KEY,
        authority_key_id=runtime_control.AUTHORITY_KEY_ID,
        trusted_public_keys=public_keys,
        principal=principal,
    )
    if activate:
        legacy = control.state(qualification_id="qual.v65-control")
        if legacy.control_status == "LEGACY_UNMANAGED":
            control.activate(
                qualification_id="qual.v65-control",
                expected_v63_state_hash=legacy.v63_state_hash,
                expected_v63_phase=legacy.v63_phase,
            )
    return control


def _different_custody(
    custody: Any,
    private_keys: dict[str, bytes],
) -> Any:
    payload = custody.model_dump(
        mode="python",
        exclude={"signature_base64", "custody_hash"},
    )
    payload["measurement_review_hash"] = sha256_value(
        {"review": "different"}
    )
    return qualification.sign_external_evidence_custody_v63(
        private_key_pem=private_keys["custody-key"],
        **payload,
    )


def test_all_typed_operations_use_controlled_ingress_protocol_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        workspace,
        public_keys,
        private_keys,
        contract,
        _forecast_input,
        custody,
    ) = _setup_input_frozen(tmp_path, monkeypatch)
    control = _control(workspace, public_keys)
    initial = control.state(qualification_id=contract.qualification_id)
    assert initial.control_status == "ACTIVE"
    assert initial.v63_phase == "INPUT_FROZEN"

    admitted = control.ingest_custody(
        qualification_id=contract.qualification_id,
        expected_v63_state_hash=initial.v63_state_hash,
        expected_v63_phase=initial.v63_phase,
        custody=custody,
    )
    assert admitted.state.control_status == "ACTIVE"
    assert admitted.state.v63_phase == "CUSTODY_VERIFIED"

    predicted = control.run_prediction(
        qualification_id=contract.qualification_id,
        expected_v63_state_hash=admitted.state.v63_state_hash,
        expected_v63_phase=admitted.state.v63_phase,
    )
    assert predicted.state.v63_phase == "PREDICTION_BOUND"
    binding = workspace._artifacts_of_kind(
        "current_model_prediction_binding_v63",
        qualification.CurrentModelPredictionBindingV63,
    )[0][1]
    registration = qualification.sign_external_prediction_registration_v63(
        private_key_pem=private_keys["registry-key"],
        qualification_id=contract.qualification_id,
        task_id=contract.task_id,
        contract_hash=contract.contract_hash,
        local_context_hash=contract.local_context_hash,
        custody_hash=custody.custody_hash,
        current_model_prediction_binding_hash=binding.binding_hash,
        generator_execution_receipt_hash=(
            binding.generator_execution_receipt_hash
        ),
        s4_gate_hash=contract.s4_gate_hash,
        training_snapshot_hash=contract.processed_snapshot_hash,
        candidate_hash=contract.selected_model_identity_hash,
        prediction_artifact_hash=binding.prediction_artifact_hash,
        external_snapshot_hash=custody.external_snapshot_hash,
        holdout_commitment_hash=custody.holdout_commitment_hash,
        normalization_scale_commitment_hash=(
            custody.normalization_scale_commitment_hash
        ),
        target_order_hash=custody.target_order_hash,
        holdout_observation_count=custody.holdout_observation_count,
        registry_host_id="registry-host",
        coordinator_host_id=contract.coordinator_host_id,
        generator_host_id=contract.generator_host_id,
        registry_key_id="registry-key",
    )
    registered = control.ingest_registration(
        qualification_id=contract.qualification_id,
        expected_v63_state_hash=predicted.state.v63_state_hash,
        expected_v63_phase=predicted.state.v63_phase,
        registration=registration,
    )
    assert registered.state.control_status == "ACTIVE"
    assert registered.state.v63_phase == "PREDICTION_REGISTERED"
    assert registered.scientific_qualification_granted is False
    assert registered.real_world_action_authorized is False

    reserved = control.reserve_evaluation(
        qualification_id=contract.qualification_id,
        expected_v63_state_hash=registered.state.v63_state_hash,
        expected_v63_phase=registered.state.v63_phase,
        evaluator_key_id="evaluator-key",
        evaluator_host_id="evaluator-host",
    )
    assert reserved.state.v63_phase == "EVALUATION_RESERVED"
    prediction_seal = workspace._artifacts_of_kind(
        "prediction_seal_v50",
        PredictionSealV50,
    )[0][1]
    reservation = workspace._artifacts_of_kind(
        "external_evaluation_reservation_v63",
        qualification.ExternalEvaluationReservationV63,
    )[0][1]
    evaluation = qualification.sign_external_aggregate_evaluation_v63(
        private_key_pem=private_keys["evaluator-key"],
        evaluation_id="evaluation.v65-control",
        qualification_id=contract.qualification_id,
        contract_hash=contract.contract_hash,
        local_context_hash=contract.local_context_hash,
        custody_hash=custody.custody_hash,
        registration_hash=registration.registration_hash,
        prediction_seal_hash=prediction_seal.seal_hash,
        reservation_hash=reservation.reservation_hash,
        prediction_artifact_hash=registration.prediction_artifact_hash,
        external_snapshot_hash=custody.external_snapshot_hash,
        holdout_commitment_hash=custody.holdout_commitment_hash,
        normalization_scale_commitment_hash=(
            custody.normalization_scale_commitment_hash
        ),
        target_order_hash=custody.target_order_hash,
        holdout_observation_count=custody.holdout_observation_count,
        squared_error_sum=12.0,
        target_squared_value_sum=1200.0,
        aggregate_metric_value=0.10,
        evaluator_host_id="evaluator-host",
        coordinator_host_id=contract.coordinator_host_id,
        generator_host_id=contract.generator_host_id,
        evaluated_at=reservation.reserved_at + timedelta(seconds=1),
        evaluator_key_id="evaluator-key",
    )
    evaluated = control.ingest_evaluation(
        qualification_id=contract.qualification_id,
        expected_v63_state_hash=reserved.state.v63_state_hash,
        expected_v63_phase=reserved.state.v63_phase,
        evaluation=evaluation,
    )
    assert evaluated.state.v63_phase == "AWAITING_PROMOTION"
    promotion = qualification.sign_external_predictive_promotion_v63(
        contract=contract,
        custody=custody,
        registration=registration,
        prediction_seal=prediction_seal,
        evaluation=evaluation,
        integrity_incident_free=True,
        promotion_host_id="promotion-host",
        promotion_key_id="promotion-key",
        private_key_pem=private_keys["promotion-key"],
        decided_at=evaluation.evaluated_at + timedelta(seconds=1),
    )
    original_commit = StageWorkspaceV50.commit_evidence
    completion_fault = {"armed": True}

    def fail_promotion_completion_once(
        target: StageWorkspaceV50,
        kind: str,
        payload: object,
    ) -> Any:
        if (
            completion_fault["armed"]
            and kind == control_module.CONTROL_COMPLETION_KIND_V65
            and isinstance(payload, dict)
            and payload.get("operation_type") == "ingest_promotion"
        ):
            completion_fault["armed"] = False
            raise RuntimeError("simulated completion commit failure")
        return original_commit(target, kind, payload)

    monkeypatch.setattr(
        StageWorkspaceV50,
        "commit_evidence",
        fail_promotion_completion_once,
    )
    with pytest.raises(ExternalControlOperationFailedV65) as failure_info:
        control.ingest_promotion(
            qualification_id=contract.qualification_id,
            expected_v63_state_hash=evaluated.state.v63_state_hash,
            expected_v63_phase=evaluated.state.v63_phase,
            promotion=promotion,
        )
    pending = control.state(qualification_id=contract.qualification_id)
    assert pending.v63_phase == "EXTERNALLY_QUALIFIED"
    assert pending.control_status == "PENDING_FAILURE"
    with pytest.raises(
        ExternalControlPlaneErrorV65,
        match="progressed V6.3 operation cannot abort",
    ):
        control.abort_attempt(
            qualification_id=contract.qualification_id,
            operation_id=failure_info.value.operation_id,
        )
    promoted = control.ingest_promotion(
        qualification_id=contract.qualification_id,
        expected_v63_state_hash=evaluated.state.v63_state_hash,
        expected_v63_phase=evaluated.state.v63_phase,
        promotion=promotion,
    )
    assert promoted.resumed is True
    assert promoted.state.v63_phase == "EXTERNALLY_QUALIFIED"
    assert promoted.v63_protocol_qualification_granted is True
    assert promoted.scientific_qualification_granted is False
    assert promoted.real_world_action_authorized is False
    latest = control.latest_qualification(
        qualification_id=contract.qualification_id
    )
    assert latest.projection_status == "WORKFLOW_VERIFIED"
    assert latest.v63_protocol_qualification_granted is True
    assert latest.v64_deployment_anchor_verified is False
    assert latest.v65_predictive_quality_authority_verified is False
    assert latest.scientific_qualification_granted is False
    assert latest.real_world_action_authorized is False


def test_failure_receipt_exact_retry_conflict_and_second_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        workspace,
        public_keys,
        private_keys,
        contract,
        _forecast_input,
        custody,
    ) = _setup_input_frozen(tmp_path, monkeypatch)
    control = _control(workspace, public_keys)
    initial = control.state(qualification_id=contract.qualification_id)
    original_executor = control_module._execute_operation_v65

    def timeout_executor(*_args: Any, **_kwargs: Any) -> Any:
        raise TimeoutError("controlled transient failure")

    monkeypatch.setattr(
        control_module, "_execute_operation_v65", timeout_executor
    )
    with pytest.raises(ExternalControlOperationFailedV65) as exc_info:
        control.ingest_custody(
            qualification_id=contract.qualification_id,
            expected_v63_state_hash=initial.v63_state_hash,
            expected_v63_phase=initial.v63_phase,
            custody=custody,
        )
    assert exc_info.value.failure_class == "RETRYABLE"
    failed = control.state(qualification_id=contract.qualification_id)
    assert failed.control_status == "PENDING_FAILURE"
    assert failed.pending_failure_class == "RETRYABLE"

    with pytest.raises(
        ExternalControlPlaneErrorV65,
        match="divergent request",
    ):
        control.ingest_custody(
            qualification_id=contract.qualification_id,
            expected_v63_state_hash=initial.v63_state_hash,
            expected_v63_phase=initial.v63_phase,
            custody=_different_custody(custody, private_keys),
        )
    with pytest.raises(
        ExternalControlPlaneErrorV65,
        match="second unresolved",
    ):
        control.run_prediction(
            qualification_id=contract.qualification_id,
            expected_v63_state_hash=failed.v63_state_hash,
            expected_v63_phase=failed.v63_phase,
        )

    original_runtime = control._runtime_identity
    changed_runtime_payload = original_runtime.model_dump(
        mode="python", exclude={"runtime_hash"}
    )
    changed_runtime_payload["dependency_versions"] = {
        **original_runtime.dependency_versions,
        "pydantic": "different-runtime",
    }
    control._runtime_identity = ExternalControlRuntimeIdentityV65.seal(
        **changed_runtime_payload
    )
    with pytest.raises(
        ExternalControlPlaneErrorV65,
        match="divergent request",
    ):
        control.ingest_custody(
            qualification_id=contract.qualification_id,
            expected_v63_state_hash=initial.v63_state_hash,
            expected_v63_phase=initial.v63_phase,
            custody=custody,
        )
    control._runtime_identity = original_runtime
    monkeypatch.setattr(
        control_module, "_execute_operation_v65", original_executor
    )
    resumed = control.ingest_custody(
        qualification_id=contract.qualification_id,
        expected_v63_state_hash=initial.v63_state_hash,
        expected_v63_phase=initial.v63_phase,
        custody=custody,
    )
    assert resumed.resumed is True
    assert resumed.state.control_status == "ACTIVE"
    assert resumed.state.v63_phase == "CUSTODY_VERIFIED"
    reopened = StageWorkspaceV50.open_existing(
        workspace.root,
        authority_key=runtime_control.AUTHORITY_KEY,
        authority_key_id=runtime_control.AUTHORITY_KEY_ID,
    )
    assert len(
        reopened._artifacts_of_kind(CONTROL_FAILURE_KIND_V65)
    ) == 1
    assert len(
        reopened._artifacts_of_kind(CONTROL_RESOLUTION_KIND_V65)
    ) == 1


def test_exact_retry_recovers_v63_partial_phase_without_new_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        workspace,
        public_keys,
        _private_keys,
        contract,
        _forecast_input,
        custody,
    ) = _setup_input_frozen(tmp_path, monkeypatch)
    control = _control(workspace, public_keys)
    initial = control.state(qualification_id=contract.qualification_id)
    original_executor = control_module._execute_operation_v65

    def partial_executor(
        _control: Any,
        action_workspace: StageWorkspaceV50,
        operation_type: str,
        payload: object,
        _state: Any,
    ) -> Any:
        assert operation_type == "ingest_custody"
        assert payload == custody
        action_workspace.commit_evidence(
            "external_evidence_custody_v63",
            custody.model_dump(mode="json"),
        )
        raise TimeoutError("failure after the V6.3 custody commit")

    monkeypatch.setattr(
        control_module, "_execute_operation_v65", partial_executor
    )
    with pytest.raises(ExternalControlOperationFailedV65):
        control.ingest_custody(
            qualification_id=contract.qualification_id,
            expected_v63_state_hash=initial.v63_state_hash,
            expected_v63_phase=initial.v63_phase,
            custody=custody,
        )
    partial = control.state(qualification_id=contract.qualification_id)
    assert partial.control_status == "PENDING_FAILURE"
    assert partial.v63_phase == "CUSTODY_COMMITTED"

    monkeypatch.setattr(
        control_module, "_execute_operation_v65", original_executor
    )
    resumed = control.ingest_custody(
        qualification_id=contract.qualification_id,
        expected_v63_state_hash=initial.v63_state_hash,
        expected_v63_phase=initial.v63_phase,
        custody=custody,
    )
    assert resumed.resumed is True
    assert resumed.state.control_status == "ACTIVE"
    assert resumed.state.v63_phase == "CUSTODY_VERIFIED"


def test_completion_commit_then_raise_is_success_not_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        workspace,
        public_keys,
        _private_keys,
        contract,
        _forecast_input,
        custody,
    ) = _setup_input_frozen(tmp_path, monkeypatch)
    control = _control(workspace, public_keys)
    initial = control.state(qualification_id=contract.qualification_id)
    original_commit = StageWorkspaceV50.commit_evidence
    fault = {"armed": True}

    def commit_completion_then_raise(
        target: StageWorkspaceV50,
        kind: str,
        payload: object,
    ) -> Any:
        reference = original_commit(target, kind, payload)
        if (
            fault["armed"]
            and kind == control_module.CONTROL_COMPLETION_KIND_V65
            and isinstance(payload, dict)
            and payload.get("operation_type") == "ingest_custody"
        ):
            fault["armed"] = False
            raise OSError("simulated ambiguous completion acknowledgement")
        return reference

    monkeypatch.setattr(
        StageWorkspaceV50,
        "commit_evidence",
        commit_completion_then_raise,
    )
    completed = control.ingest_custody(
        qualification_id=contract.qualification_id,
        expected_v63_state_hash=initial.v63_state_hash,
        expected_v63_phase=initial.v63_phase,
        custody=custody,
    )
    assert completed.state.control_status == "ACTIVE"
    assert completed.state.v63_phase == "CUSTODY_VERIFIED"
    reopened = StageWorkspaceV50.open_existing(
        workspace.root,
        authority_key=runtime_control.AUTHORITY_KEY,
        authority_key_id=runtime_control.AUTHORITY_KEY_ID,
    )
    assert len(
        reopened._artifacts_of_kind(
            control_module.CONTROL_COMPLETION_KIND_V65
        )
    ) == 1
    assert not reopened._artifacts_of_kind(CONTROL_FAILURE_KIND_V65)


def test_process_death_and_direct_v63_write_cannot_be_adopted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        workspace,
        public_keys,
        _private_keys,
        contract,
        forecast_input,
        custody,
    ) = _setup_input_frozen(tmp_path, monkeypatch)
    control = _control(workspace, public_keys)
    initial = control.state(qualification_id=contract.qualification_id)
    original_executor = control_module._execute_operation_v65

    def die_after_action(*args: Any, **kwargs: Any) -> Any:
        original_executor(*args, **kwargs)
        raise SystemExit("simulated process death")

    monkeypatch.setattr(
        control_module, "_execute_operation_v65", die_after_action
    )
    with pytest.raises(SystemExit, match="simulated process death"):
        control.ingest_custody(
            qualification_id=contract.qualification_id,
            expected_v63_state_hash=initial.v63_state_hash,
            expected_v63_phase=initial.v63_phase,
            custody=custody,
        )
    monkeypatch.setattr(
        control_module, "_execute_operation_v65", original_executor
    )
    interrupted = control.state(
        qualification_id=contract.qualification_id
    )
    assert interrupted.control_status == "INCONSISTENT"
    prediction_runtime.run_current_model_external_prediction_v63(
        workspace=workspace,
        contract=contract,
        forecast_input=forecast_input,
        custody=custody,
    )
    attacked = control.state(qualification_id=contract.qualification_id)
    assert attacked.control_status == "INCONSISTENT"
    with pytest.raises(
        ExternalControlPlaneErrorV65,
        match="unreceipted V6.3 artifacts cannot be adopted",
    ):
        control.ingest_custody(
            qualification_id=contract.qualification_id,
            expected_v63_state_hash=initial.v63_state_hash,
            expected_v63_phase=initial.v63_phase,
            custody=custody,
        )
    reopened = StageWorkspaceV50.open_existing(
        workspace.root,
        authority_key=runtime_control.AUTHORITY_KEY,
        authority_key_id=runtime_control.AUTHORITY_KEY_ID,
    )
    assert not reopened._artifacts_of_kind(
        control_module.CONTROL_COMPLETION_KIND_V65
    )
    assert not reopened._artifacts_of_kind(CONTROL_FAILURE_KIND_V65)


def test_human_required_failure_can_abort_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        workspace,
        public_keys,
        _private_keys,
        contract,
        _forecast_input,
        custody,
    ) = _setup_input_frozen(tmp_path, monkeypatch)
    control = _control(workspace, public_keys)
    initial = control.state(qualification_id=contract.qualification_id)

    def invalid_executor(*_args: Any, **_kwargs: Any) -> Any:
        raise qualification.ExternalQualificationError(
            "human reconciliation required"
        )

    monkeypatch.setattr(
        control_module, "_execute_operation_v65", invalid_executor
    )
    with pytest.raises(ExternalControlOperationFailedV65) as exc_info:
        control.ingest_custody(
            qualification_id=contract.qualification_id,
            expected_v63_state_hash=initial.v63_state_hash,
            expected_v63_phase=initial.v63_phase,
            custody=custody,
        )
    assert exc_info.value.failure_class == "HUMAN_REQUIRED"

    aborted = control.abort_attempt(
        qualification_id=contract.qualification_id,
        operation_id=exc_info.value.operation_id,
    )
    assert aborted.control_status == "ABORTED"
    assert aborted.pending_operation_id is None
    assert aborted.scientific_qualification_granted is False
    assert aborted.real_world_action_authorized is False


def test_direct_v63_artifact_is_inconsistent_in_v65_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        workspace,
        public_keys,
        _private_keys,
        contract,
        _forecast_input,
        custody,
    ) = _setup_input_frozen(tmp_path, monkeypatch)
    control = _control(workspace, public_keys)
    admission = qualification.admit_external_evidence_custody_v63(
        workspace=workspace,
        contract=contract,
        custody=custody,
        trusted_public_keys=public_keys,
    )
    assert admission.status == "VERIFIED"

    projected = control.state(
        qualification_id=contract.qualification_id
    )
    assert projected.v63_phase == "CUSTODY_VERIFIED"
    assert projected.control_status == "INCONSISTENT"
    assert "uncontrolled_v63_artifact" in projected.reason_codes
    assert projected.v63_protocol_qualification_granted is False
    assert projected.scientific_qualification_granted is False
    assert projected.real_world_action_authorized is False


def test_legacy_v63_is_not_retroactively_misclassified_or_adopted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        workspace,
        public_keys,
        _private_keys,
        contract,
        _forecast_input,
        custody,
    ) = _setup_input_frozen(tmp_path, monkeypatch)
    admission = qualification.admit_external_evidence_custody_v63(
        workspace=workspace,
        contract=contract,
        custody=custody,
        trusted_public_keys=public_keys,
    )
    assert admission.status == "VERIFIED"
    control = _control(workspace, public_keys, activate=False)
    legacy = control.state(qualification_id=contract.qualification_id)
    assert legacy.control_status == "LEGACY_UNMANAGED"
    assert legacy.reason_codes == ["v65_control_not_activated"]
    with pytest.raises(
        ExternalControlPlaneErrorV65,
        match="V6.5 activation requires INPUT_FROZEN",
    ):
        control.activate(
            qualification_id=contract.qualification_id,
            expected_v63_state_hash=legacy.v63_state_hash,
            expected_v63_phase=legacy.v63_phase,
        )


def test_contract_frozen_cannot_activate_into_a_dead_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = runtime_control._new_workspace(tmp_path / "task")
    _snapshot, _bundle, _receipt, summary = (
        runtime_control._install_real_ode_model(workspace)
    )
    _private_keys, public_keys = runtime_control._keys()
    monkeypatch.setattr(
        qualification,
        "scientific_closure_summary_v62",
        lambda _workspace: summary,
    )
    monkeypatch.setattr(
        StageWorkspaceV50,
        "current_gate",
        lambda _workspace, stage: runtime_control.GATES.get(stage),
    )
    contract = (
        qualification.freeze_predictive_external_qualification_contract_v63(
            workspace=workspace,
            qualification_id="qual.v65-control",
            task_id=workspace.spec.workspace_id,
            maximum_metric_value=0.20,
            minimum_external_observation_count=12,
            coordinator_host_id="coordinator-host",
            generator_host_id="generator-host",
            custody_key_id="custody-key",
            registry_key_id="registry-key",
            evaluator_key_id="evaluator-key",
            promotion_key_id="promotion-key",
            trusted_public_keys=public_keys,
            frozen_at=runtime_control.T0,
        )
    )
    issued_at = datetime.now(timezone.utc)
    principal = issue_external_control_principal_v65(
        workspace=workspace,
        principal_id="contract-frozen-principal",
        actor_type="operator",
        qualification_id=contract.qualification_id,
        issued_at=issued_at,
        expires_at=issued_at + timedelta(hours=1),
    )
    control = ExternalQualificationControlPlaneV65(
        workspace.root,
        authority_key=runtime_control.AUTHORITY_KEY,
        authority_key_id=runtime_control.AUTHORITY_KEY_ID,
        trusted_public_keys=public_keys,
        principal=principal,
    )
    legacy = control.state(qualification_id=contract.qualification_id)
    assert legacy.v63_phase == "CONTRACT_FROZEN"
    assert legacy.control_status == "LEGACY_UNMANAGED"
    with pytest.raises(
        ExternalControlPlaneErrorV65,
        match="CONTRACT_FROZEN has no controlled forecast-input transition",
    ):
        control.activate(
            qualification_id=contract.qualification_id,
            expected_v63_state_hash=legacy.v63_state_hash,
            expected_v63_phase=legacy.v63_phase,
        )


def test_inconsistent_status_dominates_pending_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        workspace,
        public_keys,
        _private_keys,
        contract,
        _forecast_input,
        custody,
    ) = _setup_input_frozen(tmp_path, monkeypatch)
    control = _control(workspace, public_keys)
    initial = control.state(qualification_id=contract.qualification_id)

    def timeout_executor(*_args: Any, **_kwargs: Any) -> Any:
        raise TimeoutError("controlled transient failure")

    monkeypatch.setattr(
        control_module, "_execute_operation_v65", timeout_executor
    )
    with pytest.raises(ExternalControlOperationFailedV65):
        control.ingest_custody(
            qualification_id=contract.qualification_id,
            expected_v63_state_hash=initial.v63_state_hash,
            expected_v63_phase=initial.v63_phase,
            custody=custody,
        )
    qualification.admit_external_evidence_custody_v63(
        workspace=workspace,
        contract=contract,
        custody=custody,
        trusted_public_keys=public_keys,
    )
    projected = control.state(qualification_id=contract.qualification_id)
    assert projected.control_status == "INCONSISTENT"
    assert projected.pending_failure_class is None
    assert projected.pending_operation_id is None
    assert "uncontrolled_v63_artifact" in projected.reason_codes


def test_public_package_does_not_export_legacy_v63_mutators() -> None:
    forbidden = {
        "admit_external_evidence_custody_v63",
        "assess_external_predictive_qualification_v63",
        "commit_external_forecast_input_v63",
        "freeze_predictive_external_qualification_contract_v63",
        "issue_current_model_prediction_binding_v63",
        "register_external_prediction_v63",
        "reserve_external_evaluation_v63",
        "run_current_model_external_prediction_v63",
    }
    assert not (forbidden & set(v6_public.__all__))
    assert all(not hasattr(v6_public, name) for name in forbidden)


def test_mutation_requires_exact_authenticated_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        workspace,
        public_keys,
        _private_keys,
        contract,
        _forecast_input,
        custody,
    ) = _setup_input_frozen(tmp_path, monkeypatch)
    issued_at = datetime.now(timezone.utc)
    principal = issue_external_control_principal_v65(
        workspace=workspace,
        principal_id="activation-only-principal",
        actor_type="operator",
        qualification_id=contract.qualification_id,
        allowed_operations=["activate"],
        issued_at=issued_at,
        expires_at=issued_at + timedelta(hours=1),
    )
    control = ExternalQualificationControlPlaneV65(
        workspace.root,
        authority_key=runtime_control.AUTHORITY_KEY,
        authority_key_id=runtime_control.AUTHORITY_KEY_ID,
        trusted_public_keys=public_keys,
        principal=principal,
    )
    legacy = control.state(qualification_id=contract.qualification_id)
    active = control.activate(
        qualification_id=contract.qualification_id,
        expected_v63_state_hash=legacy.v63_state_hash,
        expected_v63_phase=legacy.v63_phase,
    )
    with pytest.raises(
        PermissionError, match="lacks capability ingest_custody"
    ):
        control.ingest_custody(
            qualification_id=contract.qualification_id,
            expected_v63_state_hash=active.v63_state_hash,
            expected_v63_phase=active.v63_phase,
            custody=custody,
        )


def test_principal_is_bound_to_workspace_and_qualification_audience(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_one, public_keys, *_rest = _setup_input_frozen(
        tmp_path / "one", monkeypatch
    )
    workspace_two, public_keys_two, *_rest_two = _setup_input_frozen(
        tmp_path / "two", monkeypatch
    )
    issued_at = datetime.now(timezone.utc)
    principal = issue_external_control_principal_v65(
        workspace=workspace_one,
        principal_id="audience-bound-principal",
        actor_type="operator",
        qualification_id="qual.v65-control",
        issued_at=issued_at,
        expires_at=issued_at + timedelta(hours=1),
    )
    wrong_workspace = ExternalQualificationControlPlaneV65(
        workspace_two.root,
        authority_key=runtime_control.AUTHORITY_KEY,
        authority_key_id=runtime_control.AUTHORITY_KEY_ID,
        trusted_public_keys=public_keys_two,
        principal=principal,
    )
    state_two = wrong_workspace.state(
        qualification_id="qual.v65-control"
    )
    with pytest.raises(
        PermissionError, match="workspace genesis differs"
    ):
        wrong_workspace.activate(
            qualification_id="qual.v65-control",
            expected_v63_state_hash=state_two.v63_state_hash,
            expected_v63_phase=state_two.v63_phase,
        )

    wrong_qualification = ExternalQualificationControlPlaneV65(
        workspace_one.root,
        authority_key=runtime_control.AUTHORITY_KEY,
        authority_key_id=runtime_control.AUTHORITY_KEY_ID,
        trusted_public_keys=public_keys,
        principal=principal,
    )
    with pytest.raises(PermissionError, match="audience differs"):
        wrong_qualification.activate(
            qualification_id="qual.different",
            expected_v63_state_hash=sha256_value({"state": "different"}),
            expected_v63_phase="INPUT_FROZEN",
        )


def test_activated_workspace_rejects_direct_v63_coordinator_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        workspace,
        public_keys,
        _private_keys,
        contract,
        _forecast_input,
        custody,
    ) = _setup_input_frozen(tmp_path, monkeypatch)
    control = _control(workspace, public_keys)
    initial = control.state(qualification_id=contract.qualification_id)
    admitted = control.ingest_custody(
        qualification_id=contract.qualification_id,
        expected_v63_state_hash=initial.v63_state_hash,
        expected_v63_phase=initial.v63_phase,
        custody=custody,
    )
    legacy_coordinator = ExternalQualificationCoordinatorV63(
        workspace.root,
        authority_key=runtime_control.AUTHORITY_KEY,
        authority_key_id=runtime_control.AUTHORITY_KEY_ID,
        trusted_public_keys=public_keys,
    )
    with pytest.raises(
        ExternalQualificationCoordinatorError,
        match="internal to the V6.5 control plane",
    ):
        legacy_coordinator.run_prediction(
            qualification_id=contract.qualification_id,
            expected_state_hash=admitted.state.v63_state_hash,
        )
    unchanged = control.state(qualification_id=contract.qualification_id)
    assert unchanged.control_status == "ACTIVE"
    assert unchanged.v63_phase == "CUSTODY_VERIFIED"


def test_cli_mutations_route_v65_and_require_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (
        workspace,
        public_keys,
        _private_keys,
        contract,
        _forecast_input,
        custody,
    ) = _setup_input_frozen(tmp_path, monkeypatch)
    issued_at = datetime.now(timezone.utc)
    principal = issue_external_control_principal_v65(
        workspace=workspace,
        principal_id="cli-control-principal",
        actor_type="operator",
        qualification_id=contract.qualification_id,
        issued_at=issued_at,
        expires_at=issued_at + timedelta(hours=1),
    )
    authority_path = tmp_path / "authority.key"
    authority_path.write_bytes(runtime_control.AUTHORITY_KEY)
    keys_path = tmp_path / "public_keys.json"
    keys_path.write_text(
        json.dumps(
            {
                "keys": {
                    key_id: value.decode("utf-8")
                    for key_id, value in public_keys.items()
                }
            }
        ),
        encoding="utf-8",
    )
    principal_path = tmp_path / "principal.json"
    principal_path.write_text(
        principal.model_dump_json(), encoding="utf-8"
    )
    custody_path = tmp_path / "custody.json"
    custody_path.write_text(custody.model_dump_json(), encoding="utf-8")
    read_only = ExternalQualificationControlPlaneV65(
        workspace.root,
        authority_key=runtime_control.AUTHORITY_KEY,
        authority_key_id=runtime_control.AUTHORITY_KEY_ID,
        trusted_public_keys=public_keys,
    )
    legacy = read_only.state(qualification_id=contract.qualification_id)
    common = [
        "--workspace",
        str(workspace.root),
        "--authority-key-file",
        str(authority_path),
        "--authority-key-id",
        runtime_control.AUTHORITY_KEY_ID,
        "--public-key-manifest",
        str(keys_path),
        "--qualification-id",
        contract.qualification_id,
        "--principal-capability-file",
        str(principal_path),
    ]
    assert control_cli.main(
        [
            "activate",
            *common,
            "--expected-state-hash",
            legacy.v63_state_hash,
            "--expected-phase",
            legacy.v63_phase,
        ]
    ) == 0
    capsys.readouterr()
    active = read_only.state(qualification_id=contract.qualification_id)
    assert control_cli.main(
        [
            "ingest-custody",
            *common,
            "--expected-state-hash",
            active.v63_state_hash,
            "--expected-phase",
            active.v63_phase,
            "--artifact-file",
            str(custody_path),
        ]
    ) == 0
    capsys.readouterr()
    final = read_only.state(qualification_id=contract.qualification_id)
    assert final.control_status == "ACTIVE"
    assert final.v63_phase == "CUSTODY_VERIFIED"
    parser = control_cli._parser()
    option_strings = {
        option
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
        for command in action.choices.values()
        for command_action in command._actions
        for option in command_action.option_strings
    }
    assert "--actor" not in option_strings
