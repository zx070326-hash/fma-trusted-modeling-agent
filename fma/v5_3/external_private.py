"""Actually callable separate-host private evaluation for FMA V5.3.

The worker and host-management authorities sign with distinct Ed25519 keys.
The coordinator verifies pinned public keys and public commitments.  No API in
this module grants scientific qualification or authorizes real-world action.
"""

from __future__ import annotations

import base64
import math
import os
import platform
import sys
from datetime import datetime, timezone
from typing import Annotated, Literal, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import Field, model_validator

from fma.hashing import canonical_json, sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5.external_harness import (
    PredictionDocumentV50,
    PrivateCaseCapsuleV50,
)

from .campaign import I32GraphBindingV53, PredictionRegistrationV53
from .custody import (
    CustodyVerificationReceiptV53,
    ExternalCustodyAttestationV53,
    PrivateScoreContractV53,
    public_key_fingerprint_v53,
)
from .ode_forecast import ODEForecastBundleV53


FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def runtime_environment_fingerprint_v53() -> str:
    return sha256_value(
        {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "byteorder": sys.byteorder,
        }
    )


def _load_private_key(private_key_pem: bytes) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(private_key_pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("private evaluation key must be Ed25519")
    return key


def _load_public_key(public_key_pem: bytes) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(public_key_pem)
    if not isinstance(key, Ed25519PublicKey):
        raise TypeError("private evaluation public key must be Ed25519")
    return key


def _verify_signature(
    *,
    public_key_pem: bytes,
    signature_base64: str,
    payload: bytes,
) -> bool:
    try:
        _load_public_key(public_key_pem).verify(
            base64.b64decode(signature_base64.encode("ascii"), validate=True),
            payload,
        )
        return True
    except (InvalidSignature, TypeError, ValueError):
        return False


def _private_event_hashes(
    *,
    request_hash: str,
    prediction_snapshot_hash: str,
    capsule_commitment: str,
    quality_score: float,
    integrity_valid: bool,
    domain_valid: bool,
) -> tuple[list[str], str]:
    inputs = sha256_value(
        {
            "event": "external_private_inputs_validated",
            "request_hash": request_hash,
            "prediction_snapshot_hash": prediction_snapshot_hash,
            "capsule_commitment": capsule_commitment,
        }
    )
    score = sha256_value(
        {
            "event": "external_private_aggregate_scored",
            "previous_event_hash": inputs,
            "quality_score": quality_score,
            "integrity_valid": integrity_valid,
            "prediction_domain_valid": domain_valid,
        }
    )
    events = [inputs, score]
    return events, sha256_value(events)


class PrivateEvaluationRequestV53(StrictModel):
    schema_version: Literal["5.3"] = "5.3"
    request_id: Identifier
    case_id: Identifier
    evaluator_epoch: Identifier
    score_contract_hash: Sha256
    forecast_plan_hash: Sha256
    forecast_bundle_hash: Sha256
    custody_attestation_hash: Sha256
    prediction_registration_hash: Sha256
    graph_binding_hash: Sha256
    prediction_snapshot_hash: Sha256
    prediction_semantic_hash: Sha256
    private_capsule_commitment: Sha256
    metric: Literal["mean_absolute_error"] = "mean_absolute_error"
    minimum_quality_score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    maximum_private_evaluations: Literal[1] = 1
    created_at: datetime
    request_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_request(self) -> "PrivateEvaluationRequestV53":
        if self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        if self.request_hash and self.request_hash != self.content_hash():
            raise ValueError("private request hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "request_hash")

    def assert_sealed(self) -> None:
        if not self.request_hash or self.request_hash != self.content_hash():
            raise ValueError("private evaluation request is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PrivateEvaluationRequestV53":
        data.setdefault("created_at", _utc_now())
        draft = cls(**data)
        payload = draft.model_dump(exclude={"request_hash"})
        payload["request_hash"] = draft.content_hash()
        return cls(**payload)


class ExternalPrivateWorkerReceiptV53(StrictModel):
    schema_version: Literal["5.3-external-worker"] = "5.3-external-worker"
    request_hash: Sha256
    case_id: Identifier
    score_contract_hash: Sha256
    custody_attestation_hash: Sha256
    prediction_registration_hash: Sha256
    graph_binding_hash: Sha256
    worker_id: Identifier
    worker_host_id: Identifier
    worker_process_id: Annotated[int, Field(ge=1)]
    worker_executable_hash: Sha256
    runner_source_hash: Sha256
    environment_fingerprint: Sha256
    prediction_snapshot_hash: Sha256
    prediction_semantic_hash: Sha256
    private_capsule_commitment: Sha256
    metric: Literal["mean_absolute_error"] = "mean_absolute_error"
    quality_score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    integrity_valid: bool
    prediction_domain_valid: bool
    threshold_passed: bool
    private_evaluation_count: Literal[1] = 1
    event_hashes: Annotated[list[Sha256], Field(min_length=2)]
    event_chain_hash: Sha256
    reason_codes: list[Identifier] = Field(default_factory=list)
    private_values_disclosed: Literal[False] = False
    per_target_feedback_disclosed: Literal[False] = False
    secrecy_canary_disclosed: Literal[False] = False
    evaluated_at: datetime
    worker_key_id: Identifier
    signature_base64: Annotated[str, Field(min_length=40)] | None = None
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "ExternalPrivateWorkerReceiptV53":
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("worker reason codes must be sorted and unique")
        if self.event_hashes != list(dict.fromkeys(self.event_hashes)):
            raise ValueError("private worker event hashes must be unique")
        if self.event_chain_hash != sha256_value(self.event_hashes):
            raise ValueError("private worker event chain differs")
        if self.threshold_passed and (
            not self.integrity_valid or not self.prediction_domain_valid
        ):
            raise ValueError("invalid private evaluation cannot pass")
        if (not self.integrity_valid or not self.prediction_domain_valid) and (
            self.quality_score != 0
        ):
            raise ValueError("invalid evaluation requires zero quality score")
        if self.evaluated_at.utcoffset() is None:
            raise ValueError("evaluated_at must be timezone-aware")
        if self.receipt_hash and (
            not self.signature_base64 or self.receipt_hash != self.content_hash()
        ):
            raise ValueError("external worker receipt envelope differs")
        return self

    def unsigned_bytes(self) -> bytes:
        return canonical_json(
            self.model_dump(
                mode="json",
                exclude={"signature_base64", "receipt_hash"},
            )
        ).encode("utf-8")

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if (
            not self.signature_base64
            or not self.receipt_hash
            or self.receipt_hash != self.content_hash()
        ):
            raise ValueError("external worker receipt is not sealed")


class ExternalWorkerHostAttestationV53(StrictModel):
    schema_version: Literal["5.3-worker-host"] = "5.3-worker-host"
    attestation_id: Identifier
    request_hash: Sha256
    worker_receipt_hash: Sha256
    worker_host_id: Identifier
    coordinator_host_id: Identifier
    generator_host_id: Identifier
    worker_key_id: Identifier
    worker_public_key_fingerprint: Sha256
    worker_executable_hash: Sha256
    runner_source_hash: Sha256
    environment_fingerprint: Sha256
    separately_administered_host: Literal[True] = True
    independent_management_key_control: Literal[True] = True
    private_execution_after_registration: Literal[True] = True
    attested_at: datetime
    host_attester_key_id: Identifier
    signature_base64: Annotated[str, Field(min_length=40)] | None = None
    attestation_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_attestation(self) -> "ExternalWorkerHostAttestationV53":
        if self.worker_host_id in {
            self.coordinator_host_id,
            self.generator_host_id,
        }:
            raise ValueError("worker host must differ from generator/coordinator")
        if self.attested_at.utcoffset() is None:
            raise ValueError("attested_at must be timezone-aware")
        if self.attestation_hash and (
            not self.signature_base64 or self.attestation_hash != self.content_hash()
        ):
            raise ValueError("worker host attestation envelope differs")
        return self

    def unsigned_bytes(self) -> bytes:
        return canonical_json(
            self.model_dump(
                mode="json",
                exclude={"signature_base64", "attestation_hash"},
            )
        ).encode("utf-8")

    def content_hash(self) -> str:
        return _hash_without(self, "attestation_hash")

    def assert_sealed(self) -> None:
        if (
            not self.signature_base64
            or not self.attestation_hash
            or self.attestation_hash != self.content_hash()
        ):
            raise ValueError("worker host attestation is not sealed")


class ExternalPrivateRunVerificationV53(StrictModel):
    schema_version: Literal["5.3"] = "5.3"
    request_hash: Sha256
    worker_receipt_hash: Sha256
    host_attestation_hash: Sha256
    custody_verification_receipt_hash: Sha256
    trusted_key_set_hash: Sha256
    status: Literal["VERIFIED", "REJECTED"]
    reason_codes: list[Identifier]
    worker_signature_valid: bool
    host_signature_valid: bool
    public_bindings_valid: bool
    separate_host_runtime_bound: bool
    private_threshold_passed: bool
    quality_score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    private_values_disclosed: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    verified_at: datetime
    verification_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_verification(self) -> "ExternalPrivateRunVerificationV53":
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("verification reasons must be sorted and unique")
        expected = (
            self.worker_signature_valid
            and self.host_signature_valid
            and self.public_bindings_valid
            and self.separate_host_runtime_bound
            and not self.reason_codes
        )
        if (self.status == "VERIFIED") != expected:
            raise ValueError("private run verification status differs")
        if self.verified_at.utcoffset() is None:
            raise ValueError("verified_at must be timezone-aware")
        if self.verification_hash and (self.verification_hash != self.content_hash()):
            raise ValueError("private run verification hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "verification_hash")

    @classmethod
    def seal(cls, **data: object) -> "ExternalPrivateRunVerificationV53":
        data.setdefault("verified_at", _utc_now())
        draft = cls(**data)
        payload = draft.model_dump(exclude={"verification_hash"})
        payload["verification_hash"] = draft.content_hash()
        return cls(**payload)


def build_private_evaluation_request_v53(
    *,
    request_id: str,
    evaluator_epoch: str,
    score_contract: PrivateScoreContractV53,
    forecast_bundle: ODEForecastBundleV53,
    custody_attestation: ExternalCustodyAttestationV53,
    custody_verification: CustodyVerificationReceiptV53,
    prediction_registration: PredictionRegistrationV53,
    graph_binding: I32GraphBindingV53,
    created_at: datetime | None = None,
) -> PrivateEvaluationRequestV53:
    """Build the single public request only after graph-bound registration."""

    score_contract.assert_forecast_plan(forecast_bundle.forecast_plan)
    custody_attestation.assert_sealed()
    custody_verification_hash = custody_verification.receipt_hash
    prediction_registration.assert_sealed()
    if not forecast_bundle.scientific_acceptance:
        raise ValueError("private request requires accepted public evidence")
    if custody_verification.status != "VERIFIED":
        raise ValueError("private request requires verified custody")
    if graph_binding.private_evaluation_status != "NOT_RUN":
        raise ValueError("private evaluation was already attempted")
    bindings = {
        "score_contract": (
            graph_binding.score_contract_hash,
            prediction_registration.score_contract_hash,
            custody_attestation.score_contract_hash,
            custody_verification.score_contract_hash,
        ),
        "forecast_bundle": (
            graph_binding.forecast_bundle_hash,
            prediction_registration.forecast_bundle_hash,
        ),
        "custody_attestation": (
            graph_binding.custody_attestation_hash,
            prediction_registration.custody_attestation_hash,
            custody_verification.attestation_hash,
        ),
        "prediction_registration": (graph_binding.prediction_registration_hash,),
    }
    expected = {
        "score_contract": score_contract.contract_hash,
        "forecast_bundle": forecast_bundle.bundle_hash,
        "custody_attestation": custody_attestation.attestation_hash,
        "prediction_registration": prediction_registration.registration_hash,
    }
    for label, values in bindings.items():
        if any(value != expected[label] for value in values):
            raise ValueError(f"{label} binding mismatch")
    if (
        prediction_registration.custody_verification_receipt_hash
        != custody_verification_hash
    ):
        raise ValueError("prediction registration binds another custody receipt")
    return PrivateEvaluationRequestV53.seal(
        request_id=request_id,
        case_id=score_contract.case_id,
        evaluator_epoch=evaluator_epoch,
        score_contract_hash=score_contract.contract_hash,
        forecast_plan_hash=forecast_bundle.forecast_plan.plan_hash,
        forecast_bundle_hash=forecast_bundle.bundle_hash,
        custody_attestation_hash=custody_attestation.attestation_hash,
        prediction_registration_hash=prediction_registration.registration_hash,
        graph_binding_hash=graph_binding.binding_hash,
        prediction_snapshot_hash=(prediction_registration.prediction_snapshot_hash),
        prediction_semantic_hash=(prediction_registration.prediction_semantic_hash),
        private_capsule_commitment=custody_attestation.capsule_commitment,
        minimum_quality_score=score_contract.minimum_quality_score,
        created_at=created_at or _utc_now(),
    )


def evaluate_external_private_inputs_v53(
    *,
    request: PrivateEvaluationRequestV53,
    score_contract: PrivateScoreContractV53,
    prediction: PredictionDocumentV50,
    prediction_bytes_hash: str,
    capsule: PrivateCaseCapsuleV50,
    worker_id: str,
    worker_host_id: str,
    worker_executable_hash: str,
    runner_source_hash: str,
    worker_key_id: str,
    worker_private_key_pem: bytes,
    evaluated_at: datetime | None = None,
) -> ExternalPrivateWorkerReceiptV53:
    """External worker computation; emits one aggregate score and commitments."""

    request.assert_sealed()
    score_contract.assert_sealed()
    capsule.assert_sealed()
    reasons: list[str] = []
    if request.case_id != score_contract.case_id:
        reasons.append("request_case_mismatch")
    if request.score_contract_hash != score_contract.contract_hash:
        reasons.append("score_contract_binding_mismatch")
    if capsule.case_id != request.case_id:
        reasons.append("capsule_case_mismatch")
    if capsule.public_case_hash != score_contract.public_case_hash:
        reasons.append("public_case_binding_mismatch")
    if capsule.capsule_hash != request.private_capsule_commitment:
        reasons.append("capsule_commitment_mismatch")
    if not math.isclose(
        capsule.quality_scale,
        score_contract.quality_scale,
        rel_tol=0,
        abs_tol=0,
    ):
        reasons.append("quality_scale_mismatch")
    if prediction_bytes_hash != request.prediction_snapshot_hash:
        reasons.append("prediction_snapshot_hash_mismatch")
    semantic_hash = sha256_value(prediction)
    if semantic_hash != request.prediction_semantic_hash:
        reasons.append("prediction_semantic_hash_mismatch")
    if capsule.secrecy_canary.encode("utf-8") in canonical_json(prediction).encode(
        "utf-8"
    ):
        reasons.append("secrecy_canary_in_prediction")

    prediction_by_id = {
        point.target_id: point.value for point in prediction.predictions
    }
    target_by_id = {target.target_id: target.value for target in capsule.holdout}
    domain_valid = (
        prediction.case_id == request.case_id
        and sorted(prediction_by_id) == score_contract.target_ids
        and set(prediction_by_id) == set(target_by_id)
    )
    if not domain_valid:
        reasons.append("prediction_domain_invalid")
    raw_mae: float | None = None
    if domain_valid:
        raw_mae = sum(
            abs(prediction_by_id[key] - target_by_id[key])
            for key in sorted(target_by_id)
        ) / len(target_by_id)
    integrity_valid = not reasons
    quality_score = (
        max(0.0, min(1.0, 1.0 - raw_mae / score_contract.quality_scale))
        if integrity_valid and raw_mae is not None
        else 0.0
    )
    threshold_passed = (
        integrity_valid
        and domain_valid
        and quality_score >= request.minimum_quality_score
    )
    event_hashes, event_chain_hash = _private_event_hashes(
        request_hash=str(request.request_hash),
        prediction_snapshot_hash=prediction_bytes_hash,
        capsule_commitment=str(capsule.capsule_hash),
        quality_score=quality_score,
        integrity_valid=integrity_valid,
        domain_valid=domain_valid,
    )
    unsigned = ExternalPrivateWorkerReceiptV53(
        request_hash=request.request_hash,
        case_id=request.case_id,
        score_contract_hash=score_contract.contract_hash,
        custody_attestation_hash=request.custody_attestation_hash,
        prediction_registration_hash=request.prediction_registration_hash,
        graph_binding_hash=request.graph_binding_hash,
        worker_id=worker_id,
        worker_host_id=worker_host_id,
        worker_process_id=os.getpid(),
        worker_executable_hash=worker_executable_hash,
        runner_source_hash=runner_source_hash,
        environment_fingerprint=runtime_environment_fingerprint_v53(),
        prediction_snapshot_hash=prediction_bytes_hash,
        prediction_semantic_hash=semantic_hash,
        private_capsule_commitment=capsule.capsule_hash,
        quality_score=quality_score,
        integrity_valid=integrity_valid,
        prediction_domain_valid=domain_valid,
        threshold_passed=threshold_passed,
        event_hashes=event_hashes,
        event_chain_hash=event_chain_hash,
        reason_codes=sorted(set(reasons)),
        evaluated_at=evaluated_at or _utc_now(),
        worker_key_id=worker_key_id,
    )
    signature = _load_private_key(worker_private_key_pem).sign(
        unsigned.unsigned_bytes()
    )
    tagged_payload = unsigned.model_dump(mode="json")
    tagged_payload["signature_base64"] = base64.b64encode(signature).decode("ascii")
    tagged = ExternalPrivateWorkerReceiptV53(**tagged_payload)
    final_payload = tagged.model_dump(mode="json")
    final_payload["receipt_hash"] = tagged.content_hash()
    return ExternalPrivateWorkerReceiptV53(**final_payload)


def sign_worker_host_attestation_v53(
    *,
    worker_receipt: ExternalPrivateWorkerReceiptV53,
    worker_public_key_pem: bytes,
    coordinator_host_id: str,
    generator_host_id: str,
    attestation_id: str,
    host_attester_key_id: str,
    host_attester_private_key_pem: bytes,
    attested_at: datetime | None = None,
) -> ExternalWorkerHostAttestationV53:
    worker_receipt.assert_sealed()
    unsigned = ExternalWorkerHostAttestationV53(
        attestation_id=attestation_id,
        request_hash=worker_receipt.request_hash,
        worker_receipt_hash=worker_receipt.receipt_hash,
        worker_host_id=worker_receipt.worker_host_id,
        coordinator_host_id=coordinator_host_id,
        generator_host_id=generator_host_id,
        worker_key_id=worker_receipt.worker_key_id,
        worker_public_key_fingerprint=public_key_fingerprint_v53(worker_public_key_pem),
        worker_executable_hash=worker_receipt.worker_executable_hash,
        runner_source_hash=worker_receipt.runner_source_hash,
        environment_fingerprint=worker_receipt.environment_fingerprint,
        attested_at=attested_at or _utc_now(),
        host_attester_key_id=host_attester_key_id,
    )
    signature = _load_private_key(host_attester_private_key_pem).sign(
        unsigned.unsigned_bytes()
    )
    tagged_payload = unsigned.model_dump(mode="json")
    tagged_payload["signature_base64"] = base64.b64encode(signature).decode("ascii")
    tagged = ExternalWorkerHostAttestationV53(**tagged_payload)
    final_payload = tagged.model_dump(mode="json")
    final_payload["attestation_hash"] = tagged.content_hash()
    return ExternalWorkerHostAttestationV53(**final_payload)


def verify_external_private_run_v53(
    *,
    request: PrivateEvaluationRequestV53,
    score_contract: PrivateScoreContractV53,
    custody_attestation: ExternalCustodyAttestationV53,
    custody_verification: CustodyVerificationReceiptV53,
    worker_receipt: ExternalPrivateWorkerReceiptV53,
    host_attestation: ExternalWorkerHostAttestationV53,
    trusted_worker_public_keys: Mapping[str, bytes],
    trusted_host_public_keys: Mapping[str, bytes],
    expected_coordinator_host_id: str,
    expected_generator_host_id: str,
) -> ExternalPrivateRunVerificationV53:
    """Verify external signatures, runtime identity, and exact campaign bindings."""

    request.assert_sealed()
    score_contract.assert_sealed()
    custody_attestation.assert_sealed()
    reasons: list[str] = []
    worker_signature_valid = False
    host_signature_valid = False
    try:
        worker_receipt.assert_sealed()
    except ValueError:
        reasons.append("worker_receipt_envelope_invalid")
    worker_key = trusted_worker_public_keys.get(worker_receipt.worker_key_id)
    if worker_key is None:
        reasons.append("worker_key_not_pinned")
    elif worker_receipt.signature_base64:
        worker_signature_valid = _verify_signature(
            public_key_pem=worker_key,
            signature_base64=worker_receipt.signature_base64,
            payload=worker_receipt.unsigned_bytes(),
        )
        if not worker_signature_valid:
            reasons.append("worker_signature_invalid")
    else:
        reasons.append("worker_signature_missing")

    try:
        host_attestation.assert_sealed()
    except ValueError:
        reasons.append("host_attestation_envelope_invalid")
    host_key = trusted_host_public_keys.get(host_attestation.host_attester_key_id)
    if host_key is None:
        reasons.append("host_attester_key_not_pinned")
    elif host_attestation.signature_base64:
        host_signature_valid = _verify_signature(
            public_key_pem=host_key,
            signature_base64=host_attestation.signature_base64,
            payload=host_attestation.unsigned_bytes(),
        )
        if not host_signature_valid:
            reasons.append("host_attestation_signature_invalid")
    else:
        reasons.append("host_attestation_signature_missing")

    public_bindings_valid = (
        custody_verification.status == "VERIFIED"
        and request.score_contract_hash == score_contract.contract_hash
        and request.custody_attestation_hash == custody_attestation.attestation_hash
        and worker_receipt.request_hash == request.request_hash
        and worker_receipt.case_id == request.case_id
        and worker_receipt.score_contract_hash == request.score_contract_hash
        and worker_receipt.custody_attestation_hash == request.custody_attestation_hash
        and worker_receipt.prediction_registration_hash
        == request.prediction_registration_hash
        and worker_receipt.graph_binding_hash == request.graph_binding_hash
        and worker_receipt.prediction_snapshot_hash == request.prediction_snapshot_hash
        and worker_receipt.prediction_semantic_hash == request.prediction_semantic_hash
        and worker_receipt.private_capsule_commitment
        == request.private_capsule_commitment
    )
    if not public_bindings_valid:
        reasons.append("public_binding_mismatch")

    separate_host_runtime_bound = (
        host_attestation.request_hash == request.request_hash
        and host_attestation.worker_receipt_hash == worker_receipt.receipt_hash
        and host_attestation.worker_host_id == worker_receipt.worker_host_id
        and host_attestation.worker_host_id == custody_attestation.custodian_host_id
        and host_attestation.coordinator_host_id == expected_coordinator_host_id
        and host_attestation.generator_host_id == expected_generator_host_id
        and host_attestation.worker_host_id
        not in {expected_coordinator_host_id, expected_generator_host_id}
        and host_attestation.worker_key_id == worker_receipt.worker_key_id
        and worker_key is not None
        and host_attestation.worker_public_key_fingerprint
        == public_key_fingerprint_v53(worker_key)
        and host_attestation.worker_executable_hash
        == worker_receipt.worker_executable_hash
        and host_attestation.runner_source_hash == worker_receipt.runner_source_hash
        and host_attestation.environment_fingerprint
        == worker_receipt.environment_fingerprint
        and worker_receipt.worker_key_id
        not in {
            custody_attestation.attester_key_id,
            host_attestation.host_attester_key_id,
        }
        and custody_attestation.attester_key_id != host_attestation.host_attester_key_id
    )
    if not separate_host_runtime_bound:
        reasons.append("separate_host_runtime_invalid")
    if not worker_receipt.integrity_valid:
        reasons.append("worker_integrity_failed")
    if not worker_receipt.prediction_domain_valid:
        reasons.append("prediction_domain_invalid")
    reasons = sorted(set(reasons))
    trusted_key_set_hash = sha256_value(
        {
            "worker": {
                key_id: public_key_fingerprint_v53(public_key)
                for key_id, public_key in sorted(trusted_worker_public_keys.items())
            },
            "host": {
                key_id: public_key_fingerprint_v53(public_key)
                for key_id, public_key in sorted(trusted_host_public_keys.items())
            },
        }
    )
    return ExternalPrivateRunVerificationV53.seal(
        request_hash=request.request_hash,
        worker_receipt_hash=(
            worker_receipt.receipt_hash if worker_receipt.receipt_hash else "0" * 64
        ),
        host_attestation_hash=(
            host_attestation.attestation_hash
            if host_attestation.attestation_hash
            else "0" * 64
        ),
        custody_verification_receipt_hash=custody_verification.receipt_hash,
        trusted_key_set_hash=trusted_key_set_hash,
        status="VERIFIED" if not reasons else "REJECTED",
        reason_codes=reasons,
        worker_signature_valid=worker_signature_valid,
        host_signature_valid=host_signature_valid,
        public_bindings_valid=public_bindings_valid,
        separate_host_runtime_bound=separate_host_runtime_bound,
        private_threshold_passed=worker_receipt.threshold_passed,
        quality_score=worker_receipt.quality_score,
    )


__all__ = [
    "ExternalPrivateRunVerificationV53",
    "ExternalPrivateWorkerReceiptV53",
    "ExternalWorkerHostAttestationV53",
    "PrivateEvaluationRequestV53",
    "build_private_evaluation_request_v53",
    "evaluate_external_private_inputs_v53",
    "runtime_environment_fingerprint_v53",
    "sign_worker_host_attestation_v53",
    "verify_external_private_run_v53",
]
