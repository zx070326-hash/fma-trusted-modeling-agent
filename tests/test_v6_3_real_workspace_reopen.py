from __future__ import annotations

import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

from fma.hashing import sha256_value
from fma.v5.scaffold import scaffold_task_workspace
from fma.v5.stage_workspace import StageWorkspaceV50
from fma.v5.workspace_schemas import (
    PredictionSealV50,
    RoleExecutionReceiptV50,
    TaskWorkspaceSpecV50,
    WorkflowProfileV50,
)
import fma.v6.external_qualification as qualification


UTC = timezone.utc
T0 = datetime(2026, 7, 1, tzinfo=UTC)
AUTHORITY_KEY = b"v63-real-workspace-authority-key-0001"
AUTHORITY_KEY_ID = "v63-real-workspace-authority"
GATES = {
    "S4": sha256_value({"real-workspace-test-gate": "S4"}),
    "S6": sha256_value({"real-workspace-test-gate": "S6"}),
}
CONSUMPTION_KIND = "external_evaluation_consumption_v63"


def _key_pair() -> tuple[bytes, bytes]:
    private = Ed25519PrivateKey.generate()
    return (
        private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ),
        private.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ),
    )


def _keys() -> tuple[dict[str, bytes], dict[str, bytes]]:
    private_keys: dict[str, bytes] = {}
    public_keys: dict[str, bytes] = {}
    for role in ("custody", "registry", "evaluator", "promotion"):
        private, public = _key_pair()
        private_keys[f"{role}-key"] = private
        public_keys[f"{role}-key"] = public
    return private_keys, public_keys


def _summary() -> dict[str, Any]:
    scientific_bundle_hash = sha256_value({"artifact": "scientific-bundle"})
    executable_receipt_hash = sha256_value(
        {"artifact": "executable-receipt"}
    )
    selected_model_identity_hash = sha256_value(
        {
            "selected_model_id": "logistic",
            "executable_candidate_receipt_hash": executable_receipt_hash,
            "scientific_bundle_hash": scientific_bundle_hash,
        }
    )
    return {
        "schema_version": "6.2",
        "evaluated": True,
        "source_integrity_status": "PASS",
        "scientific_provenance_status": "HUMAN",
        "stage_admission_status": "PASS",
        "closure_verification_status": "PASS",
        "local_evidence_status": "HUMAN",
        "scientific_closure_status": "NOT_RUN",
        "claim_ceiling": "workflow_integrity_only",
        "claim_kind": "predictive",
        "fixture_only": False,
        "dimensions": {
            "workflow_integrity": {"status": "PASS"},
            "local_adapter_checks": {"status": "PASS"},
            "leakage_safe_confirmation": {"status": "PASS"},
        },
        "report_hash": sha256_value({"artifact": "v62-report"}),
        "admission_hash": sha256_value({"artifact": "v62-admission"}),
        "verification_hash": sha256_value(
            {"artifact": "v62-verification"}
        ),
        "scientific_bundle_hash": scientific_bundle_hash,
        "processed_snapshot_hash": sha256_value(
            {"artifact": "processed-snapshot"}
        ),
        "executable_candidate_receipt_hash": executable_receipt_hash,
        "selected_model_id": "logistic",
        "selected_model_identity_hash": selected_model_identity_hash,
        "scientific_qualification_granted": False,
        "real_world_action_authorized": False,
    }


@pytest.fixture(autouse=True)
def _fixed_local_closure(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = _summary()
    monkeypatch.setattr(
        qualification,
        "scientific_closure_summary_v62",
        lambda workspace: summary,
    )
    monkeypatch.setattr(
        StageWorkspaceV50,
        "current_gate",
        lambda workspace, stage: GATES.get(stage),
    )


def _new_workspace(root: Path) -> StageWorkspaceV50:
    scaffold_task_workspace(
        root,
        "v63-real-ledger",
        "Verify V6.3 authority replay on a real content-addressed workspace",
    )
    spec = TaskWorkspaceSpecV50.seal(
        workspace_id="v63-real-ledger",
        graph_id="v5-v63-real-ledger",
        objective=(
            "Verify V6.3 authority replay on a real content-addressed workspace"
        ),
        mission_hash=sha256_value({"mission": "v63-real-ledger"}),
        evidence_snapshot_hash=sha256_value(
            {"evidence": "v63-real-ledger"}
        ),
        evaluator_epoch="v63-real-ledger-v1",
        profile=WorkflowProfileV50.seal(),
        evidence_scope="development",
        created_at=T0,
    )
    return StageWorkspaceV50.create(
        root,
        spec,
        authority_key=AUTHORITY_KEY,
        authority_key_id=AUTHORITY_KEY_ID,
    )


def _build_registered_chain(
    workspace: StageWorkspaceV50,
    private_keys: dict[str, bytes],
    public_keys: dict[str, bytes],
) -> dict[str, Any]:
    contract = (
        qualification.freeze_predictive_external_qualification_contract_v63(
            workspace=workspace,
            qualification_id="qual.real-ledger",
            task_id="task.real-ledger",
            maximum_metric_value=0.20,
            minimum_external_observation_count=12,
            coordinator_host_id="coordinator-host",
            generator_host_id="generator-host",
            custody_key_id="custody-key",
            registry_key_id="registry-key",
            evaluator_key_id="evaluator-key",
            promotion_key_id="promotion-key",
            trusted_public_keys=public_keys,
            frozen_at=T0,
        )
    )
    target_ids = [f"target.{index:02d}" for index in range(12)]
    forecast_input = qualification.commit_external_forecast_input_v63(
        workspace=workspace,
        contract=contract,
        target_ids=target_ids,
        forecast_times=[1000.0 + index for index in range(12)],
        frozen_at=T0 + timedelta(seconds=30),
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
        target_order_hash=sha256_value(target_ids),
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
        attested_at=T0 + timedelta(minutes=1),
        custody_key_id="custody-key",
    )
    custody_admission = qualification.admit_external_evidence_custody_v63(
        workspace=workspace,
        contract=contract,
        custody=custody,
        trusted_public_keys=public_keys,
    )
    predictions = [100.0 + index for index in range(12)]
    vector = qualification.ExternalPredictionVectorV63.seal(
        qualification_id=contract.qualification_id,
        local_context_hash=contract.local_context_hash,
        selected_model_identity_hash=contract.selected_model_identity_hash,
        external_snapshot_hash=custody.external_snapshot_hash,
        target_ids=target_ids,
        target_order_hash=custody.target_order_hash,
        predictions=predictions,
        prediction_values_hash=sha256_value(predictions),
    )
    prediction_ref = workspace.commit_evidence(
        "external_prediction_vector_v63",
        vector.model_dump(mode="json"),
    )
    generator_input_hash = sha256_value(
        {
            "contract_hash": contract.contract_hash,
            "local_context_hash": contract.local_context_hash,
            "scientific_bundle_hash": contract.scientific_bundle_hash,
            "processed_snapshot_hash": contract.processed_snapshot_hash,
            "executable_candidate_receipt_hash": (
                contract.executable_candidate_receipt_hash
            ),
            "selected_model_identity_hash": (
                contract.selected_model_identity_hash
            ),
            "external_snapshot_hash": custody.external_snapshot_hash,
            "holdout_observation_count": custody.holdout_observation_count,
        }
    )
    trace_ref = workspace.commit_evidence(
        "v63_prediction_transport_trace",
        {
            "role": "modeler",
            "subject_id": contract.task_id,
            "input_authority_hash": generator_input_hash,
            "run_id": "prediction-generator-run",
            "context_id": "prediction-generator-context",
        },
    )
    generator_receipt = workspace.issue_role_execution(
        stage="S4",
        execution_id="external-prediction-generation",
        role="modeler",
        subject_id=contract.task_id,
        input_authority_hash=generator_input_hash,
        run_id="prediction-generator-run",
        context_id="prediction-generator-context",
        provider="test-harness",
        model="deterministic-test-adapter",
        prompt_hash=sha256_value({"prompt": "external prediction"}),
        output_schema_hash=contract.prediction_output_schema_hash,
        transport_trace_hash=trace_ref.sha256,
        output_artifact_hash=prediction_ref.sha256,
    )

    binding = qualification.issue_current_model_prediction_binding_v63(
        workspace=workspace,
        contract=contract,
        custody=custody,
        prediction_vector=vector,
        generator_execution_receipt=generator_receipt,
    )
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
    prediction_seal = qualification.register_external_prediction_v63(
        workspace=workspace,
        contract=contract,
        custody=custody,
        prediction_binding=binding,
        registration=registration,
        trusted_public_keys=public_keys,
    )
    reservation = qualification.reserve_external_evaluation_v63(
        workspace=workspace,
        contract=contract,
        custody=custody,
        prediction_binding=binding,
        registration=registration,
        prediction_seal=prediction_seal,
        evaluator_key_id="evaluator-key",
        evaluator_host_id="evaluator-host",
        reserved_at=prediction_seal.registered_at + timedelta(seconds=1),
    )
    evaluation = qualification.sign_external_aggregate_evaluation_v63(
        private_key_pem=private_keys["evaluator-key"],
        evaluation_id="evaluation.one",
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
        decided_at=reservation.reserved_at + timedelta(seconds=2),
    )
    return {
        "contract": contract,
        "forecast_input": forecast_input,
        "custody": custody,
        "custody_admission": custody_admission,
        "prediction_vector": vector,
        "generator_receipt": generator_receipt,
        "binding": binding,
        "registration": registration,
        "prediction_seal": prediction_seal,
        "reservation": reservation,
        "evaluation": evaluation,
        "promotion": promotion,
    }


def _assess(
    workspace: StageWorkspaceV50,
    chain: dict[str, Any],
    public_keys: dict[str, bytes],
) -> Any:
    return qualification.assess_external_predictive_qualification_v63(
        workspace=workspace,
        contract=chain["contract"],
        custody=chain["custody"],
        prediction_binding=chain["binding"],
        registration=chain["registration"],
        prediction_seal=chain["prediction_seal"],
        reservation=chain["reservation"],
        evaluation=chain["evaluation"],
        promotion=chain["promotion"],
        trusted_public_keys=public_keys,
    )


def _reopen(root: Path) -> StageWorkspaceV50:
    return StageWorkspaceV50.open_existing(
        root,
        authority_key=AUTHORITY_KEY,
        authority_key_id=AUTHORITY_KEY_ID,
    )


def _ledger_projection(
    workspace: StageWorkspaceV50,
) -> tuple[bytes, tuple[str, ...]]:
    return (
        workspace.graph.store.event_path.read_bytes(),
        tuple(
            path.name
            for path in sorted(
                workspace.graph.store.artifact_directory.glob("*.json")
            )
        ),
    )


def test_real_workspace_reopens_and_replays_v63_authority_chain(
    tmp_path: Path,
) -> None:
    private_keys, public_keys = _keys()
    root = tmp_path / "task"
    workspace = _new_workspace(root)
    chain = _build_registered_chain(workspace, private_keys, public_keys)
    receipt = _assess(workspace, chain, public_keys)

    assert receipt.status == "EXTERNALLY_QUALIFIED"
    assert workspace.graph.store.verify_event_chain()
    assert workspace.graph.verify()
    assert workspace.verify()

    reopened = _reopen(root)
    assert reopened.graph.store.verify_event_chain()
    assert reopened.graph.verify()
    assert reopened.verify()
    with reopened.graph.store.writer_transaction():
        assert reopened.graph.project_state().snapshot.snapshot_hash
    for artifact_hash in receipt.authority_artifact_hashes.values():
        assert reopened._artifact_payload_by_hash(artifact_hash)
    role_receipts = reopened._artifacts_of_kind(
        "role_execution_receipt_v50",
        RoleExecutionReceiptV50,
    )
    assert len(role_receipts) == 1
    assert reopened.verify_role_execution(role_receipts[0][1])
    prediction_seals = reopened._artifacts_of_kind(
        "prediction_seal_v50",
        PredictionSealV50,
    )
    assert len(prediction_seals) == 1
    assert reopened.verify_prediction_seal(prediction_seals[0][1])

    before = _ledger_projection(reopened)
    replay = qualification.verify_external_predictive_qualification_v63(
        workspace=reopened,
        receipt=receipt,
        trusted_public_keys=public_keys,
    )
    assert replay.status == "PASS"
    assert all(replay.checks.values())
    assert _ledger_projection(reopened) == before


def test_read_only_replay_does_not_repair_missing_consumption(
    tmp_path: Path,
) -> None:
    private_keys, public_keys = _keys()
    root = tmp_path / "task"
    workspace = _new_workspace(root)
    chain = _build_registered_chain(workspace, private_keys, public_keys)
    incomplete_root = tmp_path / "missing-consumption"
    shutil.copytree(root, incomplete_root)
    receipt = _assess(workspace, chain, public_keys)

    incomplete = _reopen(incomplete_root)
    evaluation_ref = incomplete.commit_evidence(
        "external_aggregate_evaluation_v63",
        chain["evaluation"].model_dump(mode="json"),
    )
    promotion_ref = incomplete.commit_evidence(
        "external_predictive_promotion_v63",
        chain["promotion"].model_dump(mode="json"),
    )
    assert (
        evaluation_ref.sha256
        == receipt.authority_artifact_hashes["evaluation"]
    )
    assert (
        promotion_ref.sha256
        == receipt.authority_artifact_hashes["promotion"]
    )
    incomplete.commit_evidence(
        "external_predictive_qualification_v63",
        receipt.model_dump(mode="json"),
    )
    incomplete = _reopen(incomplete_root)
    assert incomplete.verify()
    assert incomplete._artifacts_of_kind(CONSUMPTION_KIND) == []

    before = _ledger_projection(incomplete)
    replay = qualification.verify_external_predictive_qualification_v63(
        workspace=incomplete,
        receipt=receipt,
        trusted_public_keys=public_keys,
    )
    assert replay.status == "FAIL"
    assert _ledger_projection(incomplete) == before
    assert incomplete._artifacts_of_kind(CONSUMPTION_KIND) == []
