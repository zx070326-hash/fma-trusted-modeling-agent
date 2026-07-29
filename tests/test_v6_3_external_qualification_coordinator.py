"""Coordinator protocol controls.

These tests exercise locking, replay, and idempotency on a real
content-addressed ``StageWorkspaceV50``.  The installed V6.2 closure summary
is a test control and is not scientific qualification evidence.
"""

from __future__ import annotations

import argparse
import multiprocessing
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from fma.hashing import sha256_value
from fma.v5.stage_workspace import StageWorkspaceV50
import fma.v6.external_qualification as qualification
import fma.v6.external_qualification_coordinator as coordinator_module
import fma.v6.external_control_plane as v65_control_module
from fma.v6.external_qualification_cli import _parser
from fma.v6.external_qualification_coordinator import (
    ExternalQualificationCoordinatorError,
    ExternalQualificationCoordinatorV63,
    OPERATION_INTENT_KIND_V63,
    project_external_qualification_state_v63,
)

import test_v6_3_external_prediction_runtime as runtime_control


def _control_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    StageWorkspaceV50,
    dict[str, bytes],
    dict[str, bytes],
    Any,
    Any,
    Any,
    dict[str, Any],
]:
    workspace = runtime_control._new_workspace(tmp_path / "task")
    _snapshot, _bundle, _receipt, summary = (
        runtime_control._install_real_ode_model(workspace)
    )
    private_keys, public_keys = runtime_control._keys()
    contract, forecast_input, custody = runtime_control._freeze_and_admit(
        workspace=workspace,
        summary=summary,
        private_keys=private_keys,
        public_keys=public_keys,
        monkeypatch=monkeypatch,
    )
    return (
        workspace,
        public_keys,
        private_keys,
        contract,
        forecast_input,
        custody,
        summary,
    )


def _coordinator(
    root: Path,
    public_keys: dict[str, bytes],
) -> ExternalQualificationCoordinatorV63:
    control = v65_control_module.ExternalQualificationControlPlaneV65(
        root,
        authority_key=runtime_control.AUTHORITY_KEY,
        authority_key_id=runtime_control.AUTHORITY_KEY_ID,
        trusted_public_keys=public_keys,
    )
    return control._v63_coordinator()


def _register_current_prediction(
    *,
    workspace: StageWorkspaceV50,
    contract: Any,
    custody: Any,
    private_keys: dict[str, bytes],
    public_keys: dict[str, bytes],
) -> tuple[Any, Any]:
    bindings = workspace._artifacts_of_kind(
        "current_model_prediction_binding_v63",
        qualification.CurrentModelPredictionBindingV63,
    )
    assert len(bindings) == 1
    binding = bindings[0][1]
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
    seal = qualification.register_external_prediction_v63(
        workspace=workspace,
        contract=contract,
        custody=custody,
        prediction_binding=binding,
        registration=registration,
        trusted_public_keys=public_keys,
    )
    return registration, seal


def _install_spawn_control(summary: dict[str, Any]) -> None:
    qualification.scientific_closure_summary_v62 = (
        lambda _workspace: summary
    )
    StageWorkspaceV50.current_gate = (  # type: ignore[method-assign]
        lambda _workspace, stage: runtime_control.GATES.get(stage)
    )


def _spawn_prediction_worker(
    root: str,
    public_keys: dict[str, bytes],
    summary: dict[str, Any],
    qualification_id: str,
    expected_state_hash: str,
    queue: Any,
) -> None:
    try:
        _install_spawn_control(summary)
        result = _coordinator(Path(root), public_keys).run_prediction(
            qualification_id=qualification_id,
            expected_state_hash=expected_state_hash,
            actor="server",
        )
        queue.put(
            {
                "ok": True,
                "resumed": result.resumed,
                "operation_id": result.operation_id,
                "state": result.state.phase,
            }
        )
    except Exception as exc:  # pragma: no cover - returned to parent process.
        queue.put(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )


def test_projector_and_prediction_crash_retry_are_append_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        workspace,
        public_keys,
        _private_keys,
        contract,
        _forecast_input,
        _custody,
        _summary,
    ) = _control_setup(tmp_path, monkeypatch)
    coordinator = _coordinator(workspace.root, public_keys)

    before = coordinator.state(qualification_id=contract.qualification_id)
    assert before.phase == "CUSTODY_VERIFIED"
    assert before.scientific_qualification_granted is False
    original = coordinator_module._seal_operation_receipt

    def crash_before_receipt(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated crash after prediction action")

    monkeypatch.setattr(
        coordinator_module,
        "_seal_operation_receipt",
        crash_before_receipt,
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        coordinator.run_prediction(
            qualification_id=contract.qualification_id,
            expected_state_hash=before.state_hash,
        )

    interrupted = coordinator.state(
        qualification_id=contract.qualification_id
    )
    assert interrupted.phase == "PREDICTION_BOUND"
    assert interrupted.pending_operation_id is not None
    assert interrupted.scientific_qualification_granted is False

    monkeypatch.setattr(
        coordinator_module,
        "_seal_operation_receipt",
        original,
    )
    resumed = coordinator.run_prediction(
        qualification_id=contract.qualification_id,
        expected_state_hash=before.state_hash,
    )
    assert resumed.resumed is True
    assert resumed.state.phase == "PREDICTION_BOUND"
    assert resumed.state.pending_operation_id is None
    assert resumed.scientific_qualification_granted is False

    reopened = StageWorkspaceV50.open_existing(
        workspace.root,
        authority_key=runtime_control.AUTHORITY_KEY,
        authority_key_id=runtime_control.AUTHORITY_KEY_ID,
    )
    assert len(
        reopened._artifacts_of_kind(OPERATION_INTENT_KIND_V63)
    ) == 1


def test_state_projection_fails_closed_when_numeric_replay_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        workspace,
        public_keys,
        _private_keys,
        contract,
        _forecast_input,
        _custody,
        _summary,
    ) = _control_setup(tmp_path, monkeypatch)
    coordinator = _coordinator(workspace.root, public_keys)
    before = coordinator.state(qualification_id=contract.qualification_id)
    completed = coordinator.run_prediction(
        qualification_id=contract.qualification_id,
        expected_state_hash=before.state_hash,
    )
    assert completed.state.phase == "PREDICTION_BOUND"

    def reject_numeric_replay(*_args: Any, **_kwargs: Any) -> Any:
        raise coordinator_module.ExternalPredictionRuntimeError(
            "numeric replay rejected the signed prediction"
        )

    monkeypatch.setattr(
        coordinator_module,
        "verify_current_model_external_prediction_v63",
        reject_numeric_replay,
    )
    projected = coordinator.state(
        qualification_id=contract.qualification_id
    )
    assert projected.phase == "INCONSISTENT"
    assert projected.scientific_qualification_granted is False
    assert "authority_replay_failed" in projected.reason_codes


def test_direct_prediction_mutator_cannot_bypass_coordinator_intent(
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
        _summary,
    ) = _control_setup(tmp_path, monkeypatch)
    coordinator_module.run_current_model_external_prediction_v63(
        workspace=workspace,
        contract=contract,
        forecast_input=forecast_input,
        custody=custody,
    )

    projected = _coordinator(workspace.root, public_keys).state(
        qualification_id=contract.qualification_id
    )
    assert projected.phase == "INCONSISTENT"
    assert projected.v63_protocol_qualification_granted is False
    assert projected.scientific_qualification_granted is False
    assert "authority_replay_failed" in projected.reason_codes


def test_pending_intent_for_other_qualification_does_not_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        workspace,
        public_keys,
        _private_keys,
        contract,
        _forecast_input,
        _custody,
        _summary,
    ) = _control_setup(tmp_path, monkeypatch)
    unrelated_request_hash = sha256_value(
        {"qualification_id": "qual.unrelated", "operation": "prediction"}
    )
    unrelated = coordinator_module._seal_intent(
        workspace,
        operation_id=coordinator_module._operation_id(
            "run_prediction", unrelated_request_hash
        ),
        qualification_id="qual.unrelated",
        operation_type="run_prediction",
        request_hash=unrelated_request_hash,
        expected_state_hash=sha256_value({"state": "unrelated"}),
        expected_phase="CUSTODY_VERIFIED",
        actor="server",
    )
    workspace.commit_evidence(
        OPERATION_INTENT_KIND_V63,
        unrelated.model_dump(mode="json"),
    )

    coordinator = _coordinator(workspace.root, public_keys)
    before = coordinator.state(qualification_id=contract.qualification_id)
    completed = coordinator.run_prediction(
        qualification_id=contract.qualification_id,
        expected_state_hash=before.state_hash,
        actor="server",
    )
    assert completed.state.phase == "PREDICTION_BOUND"
    assert completed.state.pending_operation_id is None


def test_reservation_crash_before_packet_resumes_exactly_once(
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
        _summary,
    ) = _control_setup(tmp_path, monkeypatch)
    coordinator = _coordinator(workspace.root, public_keys)
    initial = coordinator.state(qualification_id=contract.qualification_id)
    predicted = coordinator.run_prediction(
        qualification_id=contract.qualification_id,
        expected_state_hash=initial.state_hash,
    )
    assert predicted.state.phase == "PREDICTION_BOUND"
    registration, prediction_seal = _register_current_prediction(
        workspace=workspace,
        contract=contract,
        custody=custody,
        private_keys=private_keys,
        public_keys=public_keys,
    )
    registered = coordinator.state(
        qualification_id=contract.qualification_id
    )
    assert registered.phase == "PREDICTION_REGISTERED"

    original = coordinator_module._commit_or_recover_dispatch_packet

    def crash_before_packet(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("simulated crash after reservation")

    monkeypatch.setattr(
        coordinator_module,
        "_commit_or_recover_dispatch_packet",
        crash_before_packet,
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        coordinator.reserve_evaluation(
            qualification_id=contract.qualification_id,
            expected_state_hash=registered.state_hash,
            evaluator_key_id="evaluator-key",
            evaluator_host_id="evaluator-host",
        )

    interrupted = coordinator.state(
        qualification_id=contract.qualification_id
    )
    assert interrupted.phase == "EVALUATION_RESERVED"
    assert interrupted.pending_operation_id is not None

    monkeypatch.setattr(
        coordinator_module,
        "_commit_or_recover_dispatch_packet",
        original,
    )
    resumed = coordinator.reserve_evaluation(
        qualification_id=contract.qualification_id,
        expected_state_hash=registered.state_hash,
        evaluator_key_id="evaluator-key",
        evaluator_host_id="evaluator-host",
    )
    assert resumed.resumed is True
    assert resumed.state.phase == "EVALUATION_RESERVED"
    assert resumed.state.pending_operation_id is None
    assert resumed.dispatch_packet is not None
    assert resumed.scientific_qualification_granted is False
    reopened = StageWorkspaceV50.open_existing(
        workspace.root,
        authority_key=runtime_control.AUTHORITY_KEY,
        authority_key_id=runtime_control.AUTHORITY_KEY_ID,
    )
    assert len(
        reopened._artifacts_of_kind(
            coordinator_module.RESERVATION_KIND_V63
        )
    ) == 1
    assert len(
        reopened._artifacts_of_kind(
            coordinator_module.DISPATCH_PACKET_KIND_V63
        )
    ) == 1
    reservations = reopened._artifacts_of_kind(
        coordinator_module.RESERVATION_KIND_V63,
        qualification.ExternalEvaluationReservationV63,
    )
    assert len(reservations) == 1
    reservation = reservations[0][1]
    evaluation = qualification.sign_external_aggregate_evaluation_v63(
        private_key_pem=private_keys["evaluator-key"],
        evaluation_id="evaluation.partial-ingress",
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
    reopened.commit_evidence(
        coordinator_module.EVALUATION_KIND_V63,
        evaluation.model_dump(mode="json"),
    )
    partial = _coordinator(workspace.root, public_keys).state(
        qualification_id=contract.qualification_id
    )
    assert partial.phase == "EVALUATION_COMMITTED"
    assert partial.next_valid_actions == ["resume_evaluation_ingest"]
    assert partial.scientific_qualification_granted is False


def test_stale_expected_state_hash_commits_no_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        workspace,
        public_keys,
        _private_keys,
        contract,
        _forecast_input,
        _custody,
        _summary,
    ) = _control_setup(tmp_path, monkeypatch)
    coordinator = _coordinator(workspace.root, public_keys)
    before = coordinator.state(qualification_id=contract.qualification_id)

    workspace.commit_evidence(
        "coordinator_state_probe",
        {
            "schema_version": "test-control",
            "probe_hash": sha256_value({"probe": "state-tip"}),
        },
    )
    after = coordinator.state(qualification_id=contract.qualification_id)
    assert after.phase == before.phase
    assert after.graph_event_tip != before.graph_event_tip
    assert after.state_hash != before.state_hash

    with pytest.raises(
        ExternalQualificationCoordinatorError,
        match="expected_state_hash is stale",
    ):
        coordinator.run_prediction(
            qualification_id=contract.qualification_id,
            expected_state_hash=before.state_hash,
        )
    reopened = StageWorkspaceV50.open_existing(
        workspace.root,
        authority_key=runtime_control.AUTHORITY_KEY,
        authority_key_id=runtime_control.AUTHORITY_KEY_ID,
    )
    assert reopened._artifacts_of_kind(OPERATION_INTENT_KIND_V63) == []


def test_spawned_processes_serialize_one_prediction_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        workspace,
        public_keys,
        _private_keys,
        contract,
        _forecast_input,
        _custody,
        summary,
    ) = _control_setup(tmp_path, monkeypatch)
    before = _coordinator(workspace.root, public_keys).state(
        qualification_id=contract.qualification_id
    )

    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(
            target=_spawn_prediction_worker,
            args=(
                str(workspace.root),
                public_keys,
                summary,
                contract.qualification_id,
                before.state_hash,
                queue,
            ),
        )
        for _index in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0

    results = [queue.get(timeout=5) for _process in processes]
    assert all(item["ok"] for item in results), results
    assert {item["state"] for item in results} == {"PREDICTION_BOUND"}
    assert len({item["operation_id"] for item in results}) == 1
    assert sorted(item["resumed"] for item in results) == [False, True]

    reopened = StageWorkspaceV50.open_existing(
        workspace.root,
        authority_key=runtime_control.AUTHORITY_KEY,
        authority_key_id=runtime_control.AUTHORITY_KEY_ID,
    )
    projected = project_external_qualification_state_v63(
        reopened,
        qualification_id=contract.qualification_id,
        trusted_public_keys=public_keys,
    )
    assert projected.phase == "PREDICTION_BOUND"
    assert len(
        reopened._artifacts_of_kind(
            "current_model_prediction_binding_v63"
        )
    ) == 1
    assert len(
        reopened._artifacts_of_kind(OPERATION_INTENT_KIND_V63)
    ) == 1


def test_cli_exposes_no_signer_or_external_role_secret_option() -> None:
    parser = _parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    command_names = set(subparsers.choices)
    assert not {
        "sign",
        "sign-custody",
        "sign-registration",
        "sign-evaluation",
        "sign-promotion",
        "generate-key",
    } & command_names

    option_strings = {
        option
        for command in subparsers.choices.values()
        for action in command._actions
        for option in action.option_strings
    }
    assert "--private-key" not in option_strings
    assert "--role-key-file" not in option_strings
    assert "--signing-key-file" not in option_strings
    assert "--authority-key-file" in option_strings
    assert "--public-key-manifest" in option_strings
