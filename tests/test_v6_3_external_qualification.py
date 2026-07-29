from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import hmac
from types import SimpleNamespace
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

import fma.v6.external_qualification as qualification_module
from fma.hashing import sha256_value
from fma.v5.workspace_schemas import PredictionSealV50, RoleExecutionReceiptV50
from fma.v6.external_qualification import (
    ExternalForecastInputV63,
    ExternalPredictionVectorV63,
    ExternalQualificationError,
    admit_external_evidence_custody_v63,
    assess_external_predictive_qualification_v63,
    commit_external_forecast_input_v63,
    freeze_predictive_external_qualification_contract_v63,
    issue_current_model_prediction_binding_v63,
    register_external_prediction_v63,
    reserve_external_evaluation_v63,
    sign_external_aggregate_evaluation_v63,
    sign_external_evidence_custody_v63,
    sign_external_prediction_registration_v63,
    sign_external_predictive_promotion_v63,
    verify_external_predictive_qualification_v63,
)


UTC = timezone.utc
T0 = datetime(2026, 1, 1, tzinfo=UTC)
WORKSPACE_HASH = sha256_value({"workspace": "v63-test"})


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


def _keys(
    *,
    reuse_physical_key: bool = False,
) -> tuple[dict[str, bytes], dict[str, bytes]]:
    private_keys: dict[str, bytes] = {}
    public_keys: dict[str, bytes] = {}
    shared = _key_pair() if reuse_physical_key else None
    for role in ("custody", "registry", "evaluator", "promotion"):
        private, public = shared or _key_pair()
        key_id = f"{role}-key"
        private_keys[key_id] = private
        public_keys[key_id] = public
    return private_keys, public_keys


def _summary(*, fixture_only: bool = False) -> dict[str, Any]:
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
        "scientific_provenance_status": (
            "NOT_RUN" if fixture_only else "HUMAN"
        ),
        "stage_admission_status": "PASS",
        "closure_verification_status": "PASS",
        "local_evidence_status": (
            "NOT_RUN" if fixture_only else "HUMAN"
        ),
        "scientific_closure_status": "NOT_RUN",
        "claim_ceiling": (
            "fixture_protocol_only"
            if fixture_only
            else "workflow_integrity_only"
        ),
        "claim_kind": "predictive",
        "fixture_only": fixture_only,
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


class _Workspace:
    def __init__(self, *, fixture_only: bool = False) -> None:
        self.spec = SimpleNamespace(spec_hash=WORKSPACE_HASH)
        self.summary = _summary(fixture_only=fixture_only)
        self.gates = {
            "S4": sha256_value({"gate": "S4"}),
            "S6": sha256_value({"gate": "S6"}),
        }
        self.evidence: dict[str, list[dict[str, Any]]] = {}
        self.committed_hashes: set[str] = set()
        self.payload_by_hash: dict[str, dict[str, Any]] = {}
        self.seals: list[PredictionSealV50] = []
        self.authority_key_id = "workspace-authority"

    def verify(self) -> bool:
        return True

    def current_gate(self, stage: str) -> str | None:
        return self.gates.get(stage)

    def commit_evidence(self, kind: str, payload: object) -> SimpleNamespace:
        assert isinstance(payload, dict)
        self.evidence.setdefault(kind, []).append(payload)
        artifact_hash = sha256_value({"kind": kind, "payload": payload})
        self.committed_hashes.add(artifact_hash)
        self.payload_by_hash[artifact_hash] = payload
        return SimpleNamespace(sha256=artifact_hash)

    def _artifacts_of_kind(
        self,
        kind: str,
        model_type: type[Any],
    ) -> list[tuple[None, Any]]:
        return [
            (
                SimpleNamespace(
                    sha256=sha256_value({"kind": kind, "payload": payload})
                ),
                model_type.model_validate(payload),
            )
            for payload in self.evidence.get(kind, [])
        ]

    def issue_prediction_seal(self, **data: object) -> PredictionSealV50:
        if any(seal.task_id == data["task_id"] for seal in self.seals):
            raise RuntimeError("prediction registration is immutable")
        unsigned = PredictionSealV50(
            workspace_spec_hash=WORKSPACE_HASH,
            s4_gate_hash=self.gates["S4"],
            **data,
            registered_at=T0 + timedelta(minutes=2),
            authority_key_id="workspace-authority",
        )
        payload = unsigned.model_dump(mode="json")
        payload["authority_auth_tag"] = sha256_value({"auth": data})
        payload["seal_hash"] = sha256_value(
            {key: value for key, value in payload.items() if key != "seal_hash"}
        )
        seal = PredictionSealV50.model_validate(payload)
        self.seals.append(seal)
        self.commit_evidence(
            "prediction_seal_v50",
            seal.model_dump(mode="json"),
        )
        return seal

    def verify_prediction_seal(self, seal: PredictionSealV50) -> bool:
        return seal in self.seals

    def _mac(self, kind: str, unsigned_hash: str) -> str:
        return sha256_value(
            {
                "fake_workspace_secret": "v63-test-only",
                "kind": kind,
                "unsigned_hash": unsigned_hash,
            }
        )

    def _verify_mac(
        self,
        kind: str,
        unsigned_hash: str,
        tag: str | None,
    ) -> bool:
        return bool(
            tag
            and hmac.compare_digest(tag, self._mac(kind, unsigned_hash))
        )

    def _committed_artifact_hashes(self) -> set[str]:
        return set(self.committed_hashes)

    def _artifact_payload_by_hash(self, artifact_hash: str) -> object:
        return self.payload_by_hash[artifact_hash]

    def verify_role_execution(self, receipt: RoleExecutionReceiptV50) -> bool:
        return any(
            receipt == candidate
            for _, candidate in self._artifacts_of_kind(
                "role_execution_receipt_v50",
                RoleExecutionReceiptV50,
            )
        )


@pytest.fixture(autouse=True)
def _replay_real_summary_function(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        qualification_module,
        "scientific_closure_summary_v62",
        lambda workspace: workspace.summary,
    )


def _build_chain(
    *,
    workspace: _Workspace,
    private_keys: dict[str, bytes],
    public_keys: dict[str, bytes],
    custody_fixture_only: bool = False,
) -> dict[str, Any]:
    contract = freeze_predictive_external_qualification_contract_v63(
        workspace=workspace,  # type: ignore[arg-type]
        qualification_id="qual.v63",
        task_id="task.v63",
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
    target_ids = [f"target.{index:02d}" for index in range(24)]
    forecast_times = [1000.0 + index for index in range(24)]
    forecast_input = commit_external_forecast_input_v63(
        workspace=workspace,  # type: ignore[arg-type]
        contract=contract,
        target_ids=target_ids,
        forecast_times=forecast_times,
        frozen_at=T0 + timedelta(seconds=30),
    )
    custody = sign_external_evidence_custody_v63(
        private_key_pem=private_keys["custody-key"],
        qualification_id=contract.qualification_id,
        contract_hash=contract.contract_hash,
        local_context_hash=contract.local_context_hash,
        v62_report_hash=contract.v62_report_hash,
        external_snapshot_hash=forecast_input.input_hash,
        holdout_commitment_hash=sha256_value({"private": "holdout"}),
        normalization_scale_commitment_hash=sha256_value(
            {
                "holdout_observation_count": 24,
                "target_squared_value_sum": 2400.0,
            }
        ),
        target_order_hash=sha256_value(target_ids),
        holdout_observation_count=24,
        fixture_only=custody_fixture_only,
        measurement_protocol_hash=sha256_value(
            {"measurement": "protocol"}
        ),
        measurement_review_hash=sha256_value({"review": "independent"}),
        external_environment_hash=sha256_value({"environment": "external"}),
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
    custody_admission = admit_external_evidence_custody_v63(
        workspace=workspace,  # type: ignore[arg-type]
        contract=contract,
        custody=custody,
        trusted_public_keys=public_keys,
    )
    prediction_vector = ExternalPredictionVectorV63.seal(
        qualification_id=contract.qualification_id,
        local_context_hash=contract.local_context_hash,
        selected_model_identity_hash=contract.selected_model_identity_hash,
        external_snapshot_hash=custody.external_snapshot_hash,
        target_ids=target_ids,
        target_order_hash=custody.target_order_hash,
        predictions=[100.0 + index for index in range(24)],
        prediction_values_hash=sha256_value(
            [100.0 + index for index in range(24)]
        ),
    )
    prediction_output = workspace.commit_evidence(
        "external_prediction_vector_v63",
        prediction_vector.model_dump(mode="json"),
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
    unsigned_role = RoleExecutionReceiptV50(
        execution_id="external-prediction-generation",
        stage="S4",
        role="modeler",
        subject_id=contract.task_id,
        input_authority_hash=generator_input_hash,
        run_id="prediction-generator-run",
        context_id="prediction-generator-context",
        provider="test-harness",
        model="deterministic-test-adapter",
        prompt_hash=sha256_value({"prompt": "external prediction"}),
        output_schema_hash=contract.prediction_output_schema_hash,
        transport_trace_hash=sha256_value({"trace": "prediction"}),
        output_artifact_hash=prediction_output.sha256,
        issued_by="harness",
        issued_at=T0 + timedelta(minutes=1, seconds=30),
        authority_key_id=workspace.authority_key_id,
    )
    role_payload = unsigned_role.model_dump(mode="json")
    role_payload["authority_auth_tag"] = workspace._mac(
        "role_execution_receipt_v50",
        unsigned_role.unsigned_hash(),
    )
    role_payload["receipt_hash"] = sha256_value(
        {
            key: value
            for key, value in role_payload.items()
            if key != "receipt_hash"
        }
    )
    generator_receipt = RoleExecutionReceiptV50.model_validate(role_payload)
    workspace.commit_evidence(
        "role_execution_receipt_v50",
        generator_receipt.model_dump(mode="json"),
    )
    prediction_binding = issue_current_model_prediction_binding_v63(
        workspace=workspace,  # type: ignore[arg-type]
        contract=contract,
        custody=custody,
        prediction_vector=prediction_vector,
        generator_execution_receipt=generator_receipt,
    )
    registration = sign_external_prediction_registration_v63(
        private_key_pem=private_keys["registry-key"],
        qualification_id=contract.qualification_id,
        task_id=contract.task_id,
        contract_hash=contract.contract_hash,
        local_context_hash=contract.local_context_hash,
        custody_hash=custody.custody_hash,
        current_model_prediction_binding_hash=(
            prediction_binding.binding_hash
        ),
        generator_execution_receipt_hash=(
            prediction_binding.generator_execution_receipt_hash
        ),
        s4_gate_hash=contract.s4_gate_hash,
        training_snapshot_hash=contract.processed_snapshot_hash,
        candidate_hash=contract.selected_model_identity_hash,
        prediction_artifact_hash=prediction_binding.prediction_artifact_hash,
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
        registered_at=T0 + timedelta(minutes=2),
        registry_key_id="registry-key",
    )
    prediction_seal = register_external_prediction_v63(
        workspace=workspace,  # type: ignore[arg-type]
        contract=contract,
        custody=custody,
        prediction_binding=prediction_binding,
        registration=registration,
        trusted_public_keys=public_keys,
    )
    reservation = reserve_external_evaluation_v63(
        workspace=workspace,  # type: ignore[arg-type]
        contract=contract,
        custody=custody,
        prediction_binding=prediction_binding,
        registration=registration,
        prediction_seal=prediction_seal,
        evaluator_key_id="evaluator-key",
        evaluator_host_id="evaluator-host",
        reserved_at=T0 + timedelta(minutes=2, seconds=30),
    )
    evaluation = sign_external_aggregate_evaluation_v63(
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
        squared_error_sum=24.0,
        target_squared_value_sum=2400.0,
        aggregate_metric_value=0.10,
        evaluator_host_id="evaluator-host",
        coordinator_host_id=contract.coordinator_host_id,
        generator_host_id=contract.generator_host_id,
        evaluated_at=T0 + timedelta(minutes=3),
        evaluator_key_id="evaluator-key",
    )
    promotion = sign_external_predictive_promotion_v63(
        contract=contract,
        custody=custody,
        registration=registration,
        prediction_seal=prediction_seal,
        evaluation=evaluation,
        integrity_incident_free=True,
        promotion_host_id="promotion-host",
        promotion_key_id="promotion-key",
        private_key_pem=private_keys["promotion-key"],
        decided_at=T0 + timedelta(minutes=4),
    )
    return {
        "contract": contract,
        "forecast_input": forecast_input,
        "custody": custody,
        "custody_admission": custody_admission,
        "prediction_vector": prediction_vector,
        "generator_receipt": generator_receipt,
        "prediction_binding": prediction_binding,
        "registration": registration,
        "prediction_seal": prediction_seal,
        "reservation": reservation,
        "evaluation": evaluation,
        "promotion": promotion,
    }


def _assess(
    workspace: _Workspace,
    chain: dict[str, Any],
    public_keys: dict[str, bytes],
    *,
    promotion: object = ...,
) -> Any:
    selected_promotion = (
        chain["promotion"] if promotion is ... else promotion
    )
    return assess_external_predictive_qualification_v63(
        workspace=workspace,  # type: ignore[arg-type]
        contract=chain["contract"],
        custody=chain["custody"],
        prediction_binding=chain["prediction_binding"],
        registration=chain["registration"],
        prediction_seal=chain["prediction_seal"],
        reservation=chain["reservation"],
        evaluation=chain["evaluation"],
        promotion=selected_promotion,
        trusted_public_keys=public_keys,
    )


def test_valid_external_chain_raises_only_predictive_ceiling() -> None:
    private_keys, public_keys = _keys()
    workspace = _Workspace()
    chain = _build_chain(
        workspace=workspace,
        private_keys=private_keys,
        public_keys=public_keys,
    )

    receipt = _assess(workspace, chain, public_keys)

    assert receipt.status == "EXTERNALLY_QUALIFIED"
    assert (
        receipt.claim_ceiling
        == "externally_qualified_predictive_evidence"
    )
    assert receipt.predictive_qualification_granted is True
    assert receipt.scientific_qualification_granted is True
    assert receipt.mechanistic_qualification_granted is False
    assert receipt.prescriptive_qualification_granted is False
    assert receipt.real_world_action_authorized is False
    assert all(receipt.checks.values())

    restarted = _Workspace()
    restarted.summary = copy.deepcopy(workspace.summary)
    restarted.gates = dict(workspace.gates)
    restarted.evidence = copy.deepcopy(workspace.evidence)
    restarted.committed_hashes = set(workspace.committed_hashes)
    restarted.payload_by_hash = copy.deepcopy(workspace.payload_by_hash)
    restarted.seals = list(workspace.seals)
    replay = verify_external_predictive_qualification_v63(
        workspace=restarted,  # type: ignore[arg-type]
        receipt=receipt,
        trusted_public_keys=public_keys,
    )
    assert replay.status == "PASS"
    assert all(replay.checks.values())


def test_evaluation_reservation_is_idempotent_and_immutable() -> None:
    private_keys, public_keys = _keys()
    workspace = _Workspace()
    chain = _build_chain(
        workspace=workspace,
        private_keys=private_keys,
        public_keys=public_keys,
    )

    repeated = reserve_external_evaluation_v63(
        workspace=workspace,  # type: ignore[arg-type]
        contract=chain["contract"],
        custody=chain["custody"],
        prediction_binding=chain["prediction_binding"],
        registration=chain["registration"],
        prediction_seal=chain["prediction_seal"],
        evaluator_key_id="evaluator-key",
        evaluator_host_id="evaluator-host",
        reserved_at=chain["reservation"].reserved_at,
    )

    assert repeated == chain["reservation"]
    assert len(workspace.evidence["external_evaluation_reservation_v63"]) == 1
    with pytest.raises(
        ExternalQualificationError,
        match="authority differs",
    ):
        reserve_external_evaluation_v63(
            workspace=workspace,  # type: ignore[arg-type]
            contract=chain["contract"],
            custody=chain["custody"],
            prediction_binding=chain["prediction_binding"],
            registration=chain["registration"],
            prediction_seal=chain["prediction_seal"],
            evaluator_key_id="different-evaluator-key",
            evaluator_host_id="evaluator-host",
        )
    with pytest.raises(
        ExternalQualificationError,
        match="immutable",
    ):
        reserve_external_evaluation_v63(
            workspace=workspace,  # type: ignore[arg-type]
            contract=chain["contract"],
            custody=chain["custody"],
            prediction_binding=chain["prediction_binding"],
            registration=chain["registration"],
            prediction_seal=chain["prediction_seal"],
            evaluator_key_id="evaluator-key",
            evaluator_host_id="different-evaluator-host",
        )
    substituted_binding = chain["prediction_binding"].model_copy(
        update={"binding_hash": sha256_value({"substituted": "binding"})}
    )
    with pytest.raises(
        ExternalQualificationError,
        match="binding envelope rejected",
    ):
        reserve_external_evaluation_v63(
            workspace=workspace,  # type: ignore[arg-type]
            contract=chain["contract"],
            custody=chain["custody"],
            prediction_binding=substituted_binding,
            registration=chain["registration"],
            prediction_seal=chain["prediction_seal"],
            evaluator_key_id="evaluator-key",
            evaluator_host_id="evaluator-host",
        )


def test_evaluation_requires_the_exact_committed_reservation() -> None:
    private_keys, public_keys = _keys()
    workspace = _Workspace()
    chain = _build_chain(
        workspace=workspace,
        private_keys=private_keys,
        public_keys=public_keys,
    )
    evaluation_data = chain["evaluation"].model_dump(
        exclude={
            "reservation_hash",
            "signature_base64",
            "evaluation_hash",
        }
    )
    with pytest.raises(ValueError, match="reservation_hash"):
        sign_external_aggregate_evaluation_v63(
            private_key_pem=private_keys["evaluator-key"],
            **evaluation_data,
        )

    wrong_evaluation = sign_external_aggregate_evaluation_v63(
        private_key_pem=private_keys["evaluator-key"],
        **{
            **evaluation_data,
            "reservation_hash": sha256_value({"reservation": "wrong"}),
        },
    )
    wrong_promotion = sign_external_predictive_promotion_v63(
        contract=chain["contract"],
        custody=chain["custody"],
        registration=chain["registration"],
        prediction_seal=chain["prediction_seal"],
        evaluation=wrong_evaluation,
        integrity_incident_free=True,
        promotion_host_id="promotion-host",
        promotion_key_id="promotion-key",
        private_key_pem=private_keys["promotion-key"],
        decided_at=T0 + timedelta(minutes=4),
    )
    with pytest.raises(
        ExternalQualificationError,
        match="reservation binding rejected",
    ):
        _assess(
            workspace,
            {**chain, "evaluation": wrong_evaluation},
            public_keys,
            promotion=wrong_promotion,
        )
    with pytest.raises(
        ExternalQualificationError,
        match="requires an external evaluation reservation",
    ):
        assess_external_predictive_qualification_v63(
            workspace=workspace,  # type: ignore[arg-type]
            contract=chain["contract"],
            custody=chain["custody"],
            prediction_binding=chain["prediction_binding"],
            registration=chain["registration"],
            prediction_seal=chain["prediction_seal"],
            reservation=None,  # type: ignore[arg-type]
            evaluation=chain["evaluation"],
            promotion=chain["promotion"],
            trusted_public_keys=public_keys,
        )


def test_replay_is_read_only_when_consumption_is_missing() -> None:
    private_keys, public_keys = _keys()
    workspace = _Workspace()
    chain = _build_chain(
        workspace=workspace,
        private_keys=private_keys,
        public_keys=public_keys,
    )
    receipt = _assess(workspace, chain, public_keys)
    workspace.evidence.pop("external_evaluation_consumption_v63")
    before_evidence = copy.deepcopy(workspace.evidence)
    before_hashes = set(workspace.committed_hashes)
    before_payloads = copy.deepcopy(workspace.payload_by_hash)

    replay = verify_external_predictive_qualification_v63(
        workspace=workspace,  # type: ignore[arg-type]
        receipt=receipt,
        trusted_public_keys=public_keys,
    )

    assert replay.status == "FAIL"
    assert "external_evaluation_consumption_v63" not in workspace.evidence
    assert workspace.evidence == before_evidence
    assert workspace.committed_hashes == before_hashes
    assert workspace.payload_by_hash == before_payloads


def test_prediction_vector_must_be_committed_under_its_exact_kind() -> None:
    private_keys, public_keys = _keys()
    workspace = _Workspace()
    chain = _build_chain(
        workspace=workspace,
        private_keys=private_keys,
        public_keys=public_keys,
    )
    payload = workspace.evidence.pop("external_prediction_vector_v63")[0]
    workspace.commit_evidence("substituted_prediction_payload_v63", payload)

    with pytest.raises(
        ExternalQualificationError,
        match="exact committed envelope",
    ):
        _assess(workspace, chain, public_keys)


def test_prediction_binding_issuer_is_idempotent() -> None:
    private_keys, public_keys = _keys()
    workspace = _Workspace()
    chain = _build_chain(
        workspace=workspace,
        private_keys=private_keys,
        public_keys=public_keys,
    )

    repeated = issue_current_model_prediction_binding_v63(
        workspace=workspace,  # type: ignore[arg-type]
        contract=chain["contract"],
        custody=chain["custody"],
        prediction_vector=chain["prediction_vector"],
        generator_execution_receipt=chain["generator_receipt"],
    )

    assert repeated == chain["prediction_binding"]
    assert (
        len(workspace.evidence["current_model_prediction_binding_v63"])
        == 1
    )


def test_prediction_binding_issuer_rejects_generation_before_custody() -> None:
    private_keys, public_keys = _keys()
    workspace = _Workspace()
    chain = _build_chain(
        workspace=workspace,
        private_keys=private_keys,
        public_keys=public_keys,
    )
    unsigned = chain["generator_receipt"].model_copy(
        update={
            "issued_at": T0 + timedelta(seconds=45),
            "authority_auth_tag": None,
            "receipt_hash": None,
        }
    )
    payload = unsigned.model_dump(mode="json")
    payload["authority_auth_tag"] = workspace._mac(
        "role_execution_receipt_v50",
        unsigned.unsigned_hash(),
    )
    payload["receipt_hash"] = sha256_value(
        {
            key: value
            for key, value in payload.items()
            if key != "receipt_hash"
        }
    )
    early_receipt = RoleExecutionReceiptV50.model_validate(payload)
    workspace.commit_evidence(
        "role_execution_receipt_v50",
        early_receipt.model_dump(mode="json"),
    )

    with pytest.raises(
        ExternalQualificationError,
        match="execution receipt binding rejected",
    ):
        issue_current_model_prediction_binding_v63(
            workspace=workspace,  # type: ignore[arg-type]
            contract=chain["contract"],
            custody=chain["custody"],
            prediction_vector=chain["prediction_vector"],
            generator_execution_receipt=early_receipt,
        )
    assert (
        len(workspace.evidence["current_model_prediction_binding_v63"])
        == 1
    )


def test_prediction_binding_issuer_rejects_target_order_substitution() -> None:
    private_keys, public_keys = _keys()
    workspace = _Workspace()
    chain = _build_chain(
        workspace=workspace,
        private_keys=private_keys,
        public_keys=public_keys,
    )
    substituted_target_ids = list(
        reversed(chain["forecast_input"].target_ids)
    )
    substituted_predictions = list(
        reversed(chain["prediction_vector"].predictions)
    )
    substituted_vector = ExternalPredictionVectorV63.seal(
        qualification_id=chain["contract"].qualification_id,
        local_context_hash=chain["contract"].local_context_hash,
        selected_model_identity_hash=(
            chain["contract"].selected_model_identity_hash
        ),
        external_snapshot_hash=chain["forecast_input"].input_hash,
        target_ids=substituted_target_ids,
        target_order_hash=sha256_value(substituted_target_ids),
        predictions=substituted_predictions,
        prediction_values_hash=sha256_value(substituted_predictions),
    )
    workspace.commit_evidence(
        "external_prediction_vector_v63",
        substituted_vector.model_dump(mode="json"),
    )

    with pytest.raises(
        ExternalQualificationError,
        match="differs from frozen forecast input",
    ):
        issue_current_model_prediction_binding_v63(
            workspace=workspace,  # type: ignore[arg-type]
            contract=chain["contract"],
            custody=chain["custody"],
            prediction_vector=substituted_vector,
            generator_execution_receipt=chain["generator_receipt"],
        )
    assert (
        len(workspace.evidence["current_model_prediction_binding_v63"])
        == 1
    )


def test_forecast_input_is_idempotent_unique_and_exact_kind() -> None:
    private_keys, public_keys = _keys()
    workspace = _Workspace()
    chain = _build_chain(
        workspace=workspace,
        private_keys=private_keys,
        public_keys=public_keys,
    )
    forecast_input = chain["forecast_input"]

    assert isinstance(forecast_input, ExternalForecastInputV63)
    repeated = commit_external_forecast_input_v63(
        workspace=workspace,  # type: ignore[arg-type]
        contract=chain["contract"],
        target_ids=forecast_input.target_ids,
        forecast_times=forecast_input.forecast_times,
        frozen_at=forecast_input.frozen_at,
    )
    assert repeated == forecast_input
    assert len(workspace.evidence["external_forecast_input_v63"]) == 1

    with pytest.raises(
        ExternalQualificationError,
        match="external_forecast_input_v63 is immutable",
    ):
        commit_external_forecast_input_v63(
            workspace=workspace,  # type: ignore[arg-type]
            contract=chain["contract"],
            target_ids=list(reversed(forecast_input.target_ids)),
            forecast_times=forecast_input.forecast_times,
            frozen_at=forecast_input.frozen_at,
        )

    with pytest.raises(
        ExternalQualificationError,
        match="predictive_external_qualification_contract_v63 is immutable",
    ):
        freeze_predictive_external_qualification_contract_v63(
            workspace=workspace,  # type: ignore[arg-type]
            qualification_id="qual.second-shot",
            task_id=chain["contract"].task_id,
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

    payload = workspace.evidence.pop("external_forecast_input_v63")[0]
    workspace.commit_evidence("substituted_external_forecast_input_v63", payload)
    with pytest.raises(
        ExternalQualificationError,
        match="one exact external forecast input is required",
    ):
        _assess(workspace, chain, public_keys)


def test_fixture_workspace_is_rejected_before_protocol_freeze() -> None:
    _, public_keys = _keys()
    workspace = _Workspace(fixture_only=True)

    with pytest.raises(ExternalQualificationError, match="non_fixture"):
        freeze_predictive_external_qualification_contract_v63(
            workspace=workspace,  # type: ignore[arg-type]
            qualification_id="qual.fixture",
            task_id="task.fixture",
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


def test_non_predictive_v62_claim_is_rejected() -> None:
    _, public_keys = _keys()
    workspace = _Workspace()
    workspace.summary["claim_kind"] = "mechanistic"

    with pytest.raises(
        ExternalQualificationError,
        match="predictive_claim_scope",
    ):
        freeze_predictive_external_qualification_contract_v63(
            workspace=workspace,  # type: ignore[arg-type]
            qualification_id="qual.mechanistic",
            task_id="task.mechanistic",
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


def test_failed_custody_is_durably_rejected_without_prediction_seal() -> None:
    private_keys, public_keys = _keys()
    workspace = _Workspace()
    contract = freeze_predictive_external_qualification_contract_v63(
        workspace=workspace,  # type: ignore[arg-type]
        qualification_id="qual.failed-custody",
        task_id="task.failed-custody",
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
    target_ids = [f"target.{index:02d}" for index in range(24)]
    forecast_input = commit_external_forecast_input_v63(
        workspace=workspace,  # type: ignore[arg-type]
        contract=contract,
        target_ids=target_ids,
        forecast_times=[1000.0 + index for index in range(24)],
        frozen_at=T0 + timedelta(seconds=30),
    )
    custody = sign_external_evidence_custody_v63(
        private_key_pem=private_keys["custody-key"],
        qualification_id=contract.qualification_id,
        contract_hash=contract.contract_hash,
        local_context_hash=contract.local_context_hash,
        v62_report_hash=contract.v62_report_hash,
        external_snapshot_hash=forecast_input.input_hash,
        holdout_commitment_hash=sha256_value({"private": "failed"}),
        normalization_scale_commitment_hash=sha256_value(
            {
                "holdout_observation_count": 24,
                "target_squared_value_sum": 2400.0,
            }
        ),
        target_order_hash=sha256_value(target_ids),
        holdout_observation_count=24,
        fixture_only=False,
        measurement_protocol_hash=sha256_value({"measurement": "failed"}),
        measurement_review_hash=sha256_value({"review": "failed"}),
        external_environment_hash=sha256_value(
            {"environment": "failed"}
        ),
        strict_unseen_verified=False,
        independent_measurement_review_passed=False,
        external_environment_verified=True,
        holdout_frozen_before_prediction=True,
        custodian_host_id="custodian-host",
        coordinator_host_id=contract.coordinator_host_id,
        generator_host_id=contract.generator_host_id,
        attested_at=T0 + timedelta(minutes=1),
        custody_key_id="custody-key",
    )

    admission = admit_external_evidence_custody_v63(
        workspace=workspace,  # type: ignore[arg-type]
        contract=contract,
        custody=custody,
        trusted_public_keys=public_keys,
    )

    assert admission.status == "REJECTED"
    assert admission.reason_codes == [
        "independent_measurement_review_failed",
        "strict_unseen_not_verified",
    ]
    assert "external_evidence_custody_v63" in workspace.evidence
    assert "external_custody_admission_v63" in workspace.evidence
    assert workspace.seals == []


def test_external_fixture_custody_cannot_receive_a_prediction_seal() -> None:
    private_keys, public_keys = _keys()
    workspace = _Workspace()

    with pytest.raises(ExternalQualificationError, match="external_fixture_only"):
        _build_chain(
            workspace=workspace,
            private_keys=private_keys,
            public_keys=public_keys,
            custody_fixture_only=True,
        )

    admission = workspace.evidence["external_custody_admission_v63"][0]
    assert admission["status"] == "REJECTED"
    assert admission["reason_codes"] == ["external_fixture_only"]
    assert workspace.seals == []


def test_stale_s6_or_closure_is_rejected() -> None:
    private_keys, public_keys = _keys()
    workspace = _Workspace()
    chain = _build_chain(
        workspace=workspace,
        private_keys=private_keys,
        public_keys=public_keys,
    )
    workspace.gates["S6"] = sha256_value({"gate": "S6-new-attempt"})

    with pytest.raises(ExternalQualificationError, match="stale"):
        _assess(workspace, chain, public_keys)


def test_tampered_aggregate_evaluation_is_rejected() -> None:
    private_keys, public_keys = _keys()
    workspace = _Workspace()
    chain = _build_chain(
        workspace=workspace,
        private_keys=private_keys,
        public_keys=public_keys,
    )
    chain["evaluation"] = chain["evaluation"].model_copy(
        update={"aggregate_metric_value": 0.01}
    )

    with pytest.raises(ExternalQualificationError, match="envelope"):
        _assess(workspace, chain, public_keys)


def test_registration_cannot_substitute_an_unqualified_candidate() -> None:
    private_keys, public_keys = _keys()
    workspace = _Workspace()
    chain = _build_chain(
        workspace=workspace,
        private_keys=private_keys,
        public_keys=public_keys,
    )
    substituted = sign_external_prediction_registration_v63(
        private_key_pem=private_keys["registry-key"],
        **{
            **chain["registration"].model_dump(
                exclude={
                    "candidate_hash",
                    "signature_base64",
                    "registration_hash",
                }
            ),
            "candidate_hash": sha256_value(
                {"candidate": "not-current-qualified-model"}
            ),
        },
    )

    with pytest.raises(
        ExternalQualificationError,
        match="registration.*(?:binding|immutable)",
    ):
        register_external_prediction_v63(
            workspace=workspace,  # type: ignore[arg-type]
            contract=chain["contract"],
            custody=chain["custody"],
            prediction_binding=chain["prediction_binding"],
            registration=substituted,
            trusted_public_keys=public_keys,
        )


def test_invalid_evaluator_host_does_not_poison_the_one_evaluation_slot() -> None:
    private_keys, public_keys = _keys()
    workspace = _Workspace()
    chain = _build_chain(
        workspace=workspace,
        private_keys=private_keys,
        public_keys=public_keys,
    )
    colliding = sign_external_aggregate_evaluation_v63(
        private_key_pem=private_keys["evaluator-key"],
        **{
            **chain["evaluation"].model_dump(
                exclude={
                    "evaluator_host_id",
                    "signature_base64",
                    "evaluation_hash",
                }
            ),
            "evaluator_host_id": chain["custody"].custodian_host_id,
        },
    )
    colliding_promotion = sign_external_predictive_promotion_v63(
        contract=chain["contract"],
        custody=chain["custody"],
        registration=chain["registration"],
        prediction_seal=chain["prediction_seal"],
        evaluation=colliding,
        integrity_incident_free=True,
        promotion_host_id="promotion-host",
        promotion_key_id="promotion-key",
        private_key_pem=private_keys["promotion-key"],
        decided_at=T0 + timedelta(minutes=4),
    )

    with pytest.raises(ExternalQualificationError, match="hosts must be distinct"):
        _assess(
            workspace,
            {**chain, "evaluation": colliding},
            public_keys,
            promotion=colliding_promotion,
        )

    assert "external_aggregate_evaluation_v63" not in workspace.evidence
    assert _assess(workspace, chain, public_keys).status == (
        "EXTERNALLY_QUALIFIED"
    )


def test_invalid_promotion_host_does_not_poison_final_promotion() -> None:
    private_keys, public_keys = _keys()
    workspace = _Workspace()
    chain = _build_chain(
        workspace=workspace,
        private_keys=private_keys,
        public_keys=public_keys,
    )
    colliding_promotion = sign_external_predictive_promotion_v63(
        contract=chain["contract"],
        custody=chain["custody"],
        registration=chain["registration"],
        prediction_seal=chain["prediction_seal"],
        evaluation=chain["evaluation"],
        integrity_incident_free=True,
        promotion_host_id=chain["evaluation"].evaluator_host_id,
        promotion_key_id="promotion-key",
        private_key_pem=private_keys["promotion-key"],
        decided_at=T0 + timedelta(minutes=4),
    )

    with pytest.raises(
        ExternalQualificationError,
        match="promotion host is not independent",
    ):
        _assess(
            workspace,
            chain,
            public_keys,
            promotion=colliding_promotion,
        )

    assert "external_predictive_promotion_v63" not in workspace.evidence
    assert _assess(workspace, chain, public_keys).status == (
        "EXTERNALLY_QUALIFIED"
    )


def test_protocol_rejects_key_id_and_physical_key_collisions() -> None:
    _, repeated_public_keys = _keys(reuse_physical_key=True)
    workspace = _Workspace()

    with pytest.raises(ValueError, match="distinct physical keys"):
        freeze_predictive_external_qualification_contract_v63(
            workspace=workspace,  # type: ignore[arg-type]
            qualification_id="qual.collision",
            task_id="task.collision",
            maximum_metric_value=0.20,
            minimum_external_observation_count=12,
            coordinator_host_id="coordinator-host",
            generator_host_id="generator-host",
            custody_key_id="custody-key",
            registry_key_id="registry-key",
            evaluator_key_id="evaluator-key",
            promotion_key_id="promotion-key",
            trusted_public_keys=repeated_public_keys,
            frozen_at=T0,
        )


def test_missing_promotion_is_not_run_and_never_authorizes_action() -> None:
    private_keys, public_keys = _keys()
    workspace = _Workspace()
    chain = _build_chain(
        workspace=workspace,
        private_keys=private_keys,
        public_keys=public_keys,
    )

    receipt = _assess(workspace, chain, public_keys, promotion=None)

    assert receipt.status == "NOT_RUN"
    assert receipt.reason_codes == ["external_promotion_missing"]
    assert receipt.predictive_qualification_granted is False
    assert receipt.scientific_qualification_granted is False
    assert receipt.real_world_action_authorized is False
    assert (
        receipt.claim_ceiling
        == "workflow_integrity_only"
    )


def test_second_distinct_external_evaluation_is_rejected() -> None:
    private_keys, public_keys = _keys()
    workspace = _Workspace()
    chain = _build_chain(
        workspace=workspace,
        private_keys=private_keys,
        public_keys=public_keys,
    )
    assert _assess(workspace, chain, public_keys).status == (
        "EXTERNALLY_QUALIFIED"
    )

    second = sign_external_aggregate_evaluation_v63(
        private_key_pem=private_keys["evaluator-key"],
        **{
            **chain["evaluation"].model_dump(
                exclude={
                    "evaluation_id",
                    "evaluated_at",
                    "signature_base64",
                    "evaluation_hash",
                }
            ),
            "evaluation_id": "evaluation.two",
            "evaluated_at": T0 + timedelta(minutes=5),
        },
    )
    second_promotion = sign_external_predictive_promotion_v63(
        contract=chain["contract"],
        custody=chain["custody"],
        registration=chain["registration"],
        prediction_seal=chain["prediction_seal"],
        evaluation=second,
        integrity_incident_free=True,
        promotion_host_id="promotion-host",
        promotion_key_id="promotion-key",
        private_key_pem=private_keys["promotion-key"],
        decided_at=T0 + timedelta(minutes=6),
    )
    chain["evaluation"] = second
    chain["promotion"] = second_promotion

    with pytest.raises(
        ExternalQualificationError,
        match="(?:duplicate external evaluation|evaluation_v63 is immutable)",
    ):
        _assess(workspace, chain, public_keys)


def test_valid_signed_but_threshold_failing_result_is_rejected() -> None:
    private_keys, public_keys = _keys()
    workspace = _Workspace()
    chain = _build_chain(
        workspace=workspace,
        private_keys=private_keys,
        public_keys=public_keys,
    )
    failing = sign_external_aggregate_evaluation_v63(
        private_key_pem=private_keys["evaluator-key"],
        **{
            **chain["evaluation"].model_dump(
                exclude={
                    "aggregate_metric_value",
                    "squared_error_sum",
                    "signature_base64",
                    "evaluation_hash",
                }
            ),
            "squared_error_sum": 384.0,
            "aggregate_metric_value": 0.40,
        },
    )
    rejection = sign_external_predictive_promotion_v63(
        contract=chain["contract"],
        custody=chain["custody"],
        registration=chain["registration"],
        prediction_seal=chain["prediction_seal"],
        evaluation=failing,
        integrity_incident_free=True,
        promotion_host_id="promotion-host",
        promotion_key_id="promotion-key",
        private_key_pem=private_keys["promotion-key"],
        decided_at=T0 + timedelta(minutes=4),
    )
    chain["evaluation"] = failing
    chain["promotion"] = rejection

    receipt = _assess(workspace, chain, public_keys)

    assert receipt.status == "REJECTED"
    assert "external_metric_threshold_failed" in receipt.reason_codes
    assert receipt.predictive_qualification_granted is False
    assert receipt.real_world_action_authorized is False


def test_signed_rejection_is_final_for_the_consumed_evaluation() -> None:
    private_keys, public_keys = _keys()
    workspace = _Workspace()
    chain = _build_chain(
        workspace=workspace,
        private_keys=private_keys,
        public_keys=public_keys,
    )
    rejection = sign_external_predictive_promotion_v63(
        contract=chain["contract"],
        custody=chain["custody"],
        registration=chain["registration"],
        prediction_seal=chain["prediction_seal"],
        evaluation=chain["evaluation"],
        integrity_incident_free=False,
        promotion_host_id="promotion-host",
        promotion_key_id="promotion-key",
        private_key_pem=private_keys["promotion-key"],
        decided_at=T0 + timedelta(minutes=4),
    )
    assert (
        _assess(workspace, chain, public_keys, promotion=rejection).status
        == "REJECTED"
    )

    with pytest.raises(
        ExternalQualificationError,
        match="(?:already final|promotion_v63 is immutable)",
    ):
        _assess(workspace, chain, public_keys)
