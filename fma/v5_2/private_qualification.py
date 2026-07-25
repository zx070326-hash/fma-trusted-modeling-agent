"""Private, process-separated qualification protocol for V5.2.

The local runner establishes a fresh-process information-flow boundary.  It
deliberately cannot grant scientific qualification: that transition requires
an independently administered, separately hosted worker and host attestation.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from fma.hashing import canonical_json, sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5.external_harness import (
    PredictionDocumentV50,
    PrivateCaseCapsuleV50,
)


IsolationLevelV52 = Literal["same_host_process", "separate_host_attested"]
QualificationStatusV52 = Literal[
    "REJECTED",
    "LOCAL_PROTOCOL_VALIDATED",
    "SCIENTIFICALLY_QUALIFIED",
]
FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_environment_fingerprint_v52() -> str:
    return sha256_value(
        {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "byteorder": sys.byteorder,
        }
    )


def _private_event_chain(
    *,
    request_hash: str,
    prediction_snapshot_hash: str,
    capsule_commitment: str,
    quality_score: float,
    integrity_valid: bool,
    domain_valid: bool,
) -> tuple[list[str], str]:
    input_event = sha256_value(
        {
            "event": "private_inputs_validated",
            "request_hash": request_hash,
            "prediction_snapshot_hash": prediction_snapshot_hash,
            "capsule_commitment": capsule_commitment,
        }
    )
    score_event = sha256_value(
        {
            "event": "private_score_computed",
            "previous_event_hash": input_event,
            "quality_score": quality_score,
            "integrity_valid": integrity_valid,
            "prediction_domain_valid": domain_valid,
        }
    )
    events = [input_event, score_event]
    return events, sha256_value(events)


class PrivateEvaluationRequestV52(StrictModel):
    schema_version: Literal["5.2"] = "5.2"
    request_id: Identifier
    case_id: Identifier
    public_case_hash: Sha256
    registration_hash: Sha256
    prediction_snapshot_hash: Sha256
    prediction_semantic_hash: Sha256
    private_capsule_commitment: Sha256
    evaluator_epoch: Identifier
    metric: Literal["mean_absolute_error"] = "mean_absolute_error"
    minimum_quality_score: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ]
    created_at: datetime
    request_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_request(self) -> "PrivateEvaluationRequestV52":
        if self.request_hash and self.request_hash != self.content_hash():
            raise ValueError("request_hash does not match request")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "request_hash")

    def assert_sealed(self) -> None:
        if not self.request_hash or self.request_hash != self.content_hash():
            raise ValueError("private evaluation request is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PrivateEvaluationRequestV52":
        data.setdefault("created_at", _utc_now())
        draft = cls(**data)
        payload = draft.model_dump(exclude={"request_hash"})
        payload["request_hash"] = draft.content_hash()
        return cls(**payload)


class PrivateWorkerReceiptV52(StrictModel):
    schema_version: Literal["5.2-private-worker"] = "5.2-private-worker"
    request_hash: Sha256
    case_id: Identifier
    worker_id: Identifier
    worker_host_id: Identifier
    worker_process_id: Annotated[int, Field(ge=1)]
    isolation_level: IsolationLevelV52
    worker_executable_hash: Sha256
    runner_source_hash: Sha256
    environment_fingerprint: Sha256
    prediction_snapshot_hash: Sha256
    prediction_semantic_hash: Sha256
    private_capsule_commitment: Sha256
    metric: Literal["mean_absolute_error"] = "mean_absolute_error"
    raw_mae: FiniteNumber | None = None
    quality_score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    integrity_valid: bool
    prediction_domain_valid: bool
    threshold_passed: bool
    event_hashes: Annotated[list[Sha256], Field(min_length=2)]
    event_chain_hash: Sha256
    reason_codes: list[Identifier] = Field(default_factory=list)
    private_values_disclosed: Literal[False] = False
    secrecy_canary_disclosed: Literal[False] = False
    evaluated_at: datetime
    worker_key_id: Identifier
    worker_auth_tag: Sha256 | None = None
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "PrivateWorkerReceiptV52":
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("reason_codes must be sorted and unique")
        if len(self.event_hashes) != len(set(self.event_hashes)):
            raise ValueError("private event hashes must be unique")
        if self.event_chain_hash != sha256_value(self.event_hashes):
            raise ValueError("private event chain hash differs")
        if self.threshold_passed and (
            not self.integrity_valid
            or not self.prediction_domain_valid
            or self.reason_codes
        ):
            raise ValueError("threshold cannot pass invalid private evaluation")
        if (not self.integrity_valid or not self.prediction_domain_valid) and (
            self.quality_score != 0
        ):
            raise ValueError("invalid evaluation requires quality_score=0")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match worker receipt")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "worker_auth_tag", "receipt_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")


class PrivateWorkerAuthorityV52:
    def __init__(self, *, key_id: str, secret: bytes) -> None:
        if len(secret) < 32:
            raise ValueError("worker authority secret needs at least 32 bytes")
        self.key_id = key_id
        self._secret = bytes(secret)

    def _mac(self, unsigned_hash: str) -> str:
        return hmac.new(
            self._secret,
            f"private_worker_v52:{unsigned_hash}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def issue(self, **data: object) -> PrivateWorkerReceiptV52:
        data["worker_key_id"] = self.key_id
        data.setdefault("evaluated_at", _utc_now())
        unsigned = PrivateWorkerReceiptV52(**data)
        tagged_payload = unsigned.model_dump(mode="json")
        tagged_payload["worker_auth_tag"] = self._mac(unsigned.unsigned_hash())
        tagged = PrivateWorkerReceiptV52(**tagged_payload)
        final_payload = tagged.model_dump(mode="json")
        final_payload["receipt_hash"] = tagged.content_hash()
        return PrivateWorkerReceiptV52(**final_payload)

    def verify(self, receipt: PrivateWorkerReceiptV52) -> bool:
        try:
            return bool(
                receipt.receipt_hash
                and receipt.receipt_hash == receipt.content_hash()
                and receipt.worker_key_id == self.key_id
                and receipt.worker_auth_tag
                and hmac.compare_digest(
                    receipt.worker_auth_tag,
                    self._mac(receipt.unsigned_hash()),
                )
            )
        except (TypeError, ValueError):
            return False


class ExternalHostAttestationV52(StrictModel):
    schema_version: Literal["5.2-host-attestation"] = "5.2-host-attestation"
    attestation_id: Identifier
    worker_host_id: Identifier
    coordinator_host_id: Identifier
    isolation_level: Literal["separate_host_attested"] = (
        "separate_host_attested"
    )
    worker_executable_hash: Sha256
    runner_source_hash: Sha256
    environment_fingerprint: Sha256
    external_control_plane_verified: Literal[True] = True
    attested_at: datetime
    attester_key_id: Identifier
    attester_auth_tag: Sha256 | None = None
    attestation_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_attestation(self) -> "ExternalHostAttestationV52":
        if self.worker_host_id == self.coordinator_host_id:
            raise ValueError("external worker and coordinator hosts must differ")
        if (
            self.attestation_hash
            and self.attestation_hash != self.content_hash()
        ):
            raise ValueError("attestation_hash does not match attestation")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "attester_auth_tag", "attestation_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "attestation_hash")


class HostAttestationAuthorityV52:
    def __init__(self, *, key_id: str, secret: bytes) -> None:
        if len(secret) < 32:
            raise ValueError("attestation secret needs at least 32 bytes")
        self.key_id = key_id
        self._secret = bytes(secret)

    def _mac(self, unsigned_hash: str) -> str:
        return hmac.new(
            self._secret,
            f"host_attestation_v52:{unsigned_hash}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def issue(self, **data: object) -> ExternalHostAttestationV52:
        data["attester_key_id"] = self.key_id
        data.setdefault("attested_at", _utc_now())
        unsigned = ExternalHostAttestationV52(**data)
        tagged_payload = unsigned.model_dump(mode="json")
        tagged_payload["attester_auth_tag"] = self._mac(
            unsigned.unsigned_hash()
        )
        tagged = ExternalHostAttestationV52(**tagged_payload)
        final_payload = tagged.model_dump(mode="json")
        final_payload["attestation_hash"] = tagged.content_hash()
        return ExternalHostAttestationV52(**final_payload)

    def verify(self, attestation: ExternalHostAttestationV52) -> bool:
        try:
            return bool(
                attestation.attestation_hash
                and attestation.attestation_hash == attestation.content_hash()
                and attestation.attester_key_id == self.key_id
                and attestation.attester_auth_tag
                and hmac.compare_digest(
                    attestation.attester_auth_tag,
                    self._mac(attestation.unsigned_hash()),
                )
            )
        except (TypeError, ValueError):
            return False


class PrivateQualificationReceiptV52(StrictModel):
    schema_version: Literal["5.2"] = "5.2"
    request_hash: Sha256
    worker_receipt_hash: Sha256
    host_attestation_hash: Sha256 | None = None
    status: QualificationStatusV52
    qualification_granted: bool
    quality_score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    reason_codes: list[Identifier] = Field(default_factory=list)
    private_acceptance_data_exposed: Literal[False] = False
    consequential_action_permitted: Literal[False] = False
    decided_at: datetime
    promotion_key_id: Identifier
    promotion_auth_tag: Sha256 | None = None
    qualification_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_qualification(self) -> "PrivateQualificationReceiptV52":
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("reason_codes must be sorted and unique")
        if self.qualification_granted != (
            self.status == "SCIENTIFICALLY_QUALIFIED"
        ):
            raise ValueError("qualification flag and status disagree")
        if self.qualification_granted and not self.host_attestation_hash:
            raise ValueError("scientific qualification requires host attestation")
        if (
            self.qualification_hash
            and self.qualification_hash != self.content_hash()
        ):
            raise ValueError("qualification_hash does not match receipt")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(
            self, "promotion_auth_tag", "qualification_hash"
        )

    def content_hash(self) -> str:
        return _hash_without(self, "qualification_hash")


class PrivatePromotionAuthorityV52:
    """Code-owned promotion gate; it never consumes raw private values."""

    def __init__(
        self,
        *,
        key_id: str,
        secret: bytes,
        coordinator_host_id: str,
    ) -> None:
        if len(secret) < 32:
            raise ValueError("promotion authority secret needs at least 32 bytes")
        self.key_id = key_id
        self.coordinator_host_id = coordinator_host_id
        self._secret = bytes(secret)

    def _mac(self, unsigned_hash: str) -> str:
        return hmac.new(
            self._secret,
            f"private_promotion_v52:{unsigned_hash}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def assess(
        self,
        *,
        request: PrivateEvaluationRequestV52,
        worker_receipt: PrivateWorkerReceiptV52,
        worker_authority: PrivateWorkerAuthorityV52,
        host_attestation: ExternalHostAttestationV52 | None = None,
        attestation_authority: HostAttestationAuthorityV52 | None = None,
    ) -> PrivateQualificationReceiptV52:
        request.assert_sealed()
        reasons: list[str] = []
        if not worker_authority.verify(worker_receipt):
            reasons.append("invalid_worker_signature")
        if worker_receipt.request_hash != request.request_hash:
            reasons.append("request_binding_mismatch")
        if worker_receipt.case_id != request.case_id:
            reasons.append("case_binding_mismatch")
        if (
            worker_receipt.prediction_snapshot_hash
            != request.prediction_snapshot_hash
        ):
            reasons.append("prediction_binding_mismatch")
        if (
            worker_receipt.prediction_semantic_hash
            != request.prediction_semantic_hash
        ):
            reasons.append("prediction_semantic_binding_mismatch")
        if (
            worker_receipt.private_capsule_commitment
            != request.private_capsule_commitment
        ):
            reasons.append("capsule_binding_mismatch")
        if not worker_receipt.integrity_valid:
            reasons.append("worker_integrity_failed")
        if not worker_receipt.prediction_domain_valid:
            reasons.append("prediction_domain_invalid")
        if not worker_receipt.threshold_passed:
            reasons.append("private_threshold_failed")

        external_valid = False
        if worker_receipt.isolation_level == "same_host_process":
            reasons.append("same_host_process_not_scientific_qualification")
        elif host_attestation is None or attestation_authority is None:
            reasons.append("external_host_attestation_missing")
        else:
            external_valid = (
                attestation_authority.verify(host_attestation)
                and host_attestation.worker_host_id
                == worker_receipt.worker_host_id
                and host_attestation.coordinator_host_id
                == self.coordinator_host_id
                and host_attestation.worker_executable_hash
                == worker_receipt.worker_executable_hash
                and host_attestation.runner_source_hash
                == worker_receipt.runner_source_hash
                and host_attestation.environment_fingerprint
                == worker_receipt.environment_fingerprint
                and host_attestation.worker_host_id
                != self.coordinator_host_id
            )
            if not external_valid:
                reasons.append("external_host_attestation_invalid")

        reasons = sorted(set(reasons))
        if not reasons and external_valid:
            status: QualificationStatusV52 = "SCIENTIFICALLY_QUALIFIED"
        elif (
            reasons == ["same_host_process_not_scientific_qualification"]
            and worker_receipt.integrity_valid
            and worker_receipt.prediction_domain_valid
            and worker_receipt.threshold_passed
        ):
            status = "LOCAL_PROTOCOL_VALIDATED"
        else:
            status = "REJECTED"
        unsigned = PrivateQualificationReceiptV52(
            request_hash=request.request_hash,
            worker_receipt_hash=worker_receipt.receipt_hash,
            host_attestation_hash=(
                host_attestation.attestation_hash
                if host_attestation and external_valid
                else None
            ),
            status=status,
            qualification_granted=status == "SCIENTIFICALLY_QUALIFIED",
            quality_score=worker_receipt.quality_score,
            reason_codes=reasons,
            decided_at=_utc_now(),
            promotion_key_id=self.key_id,
        )
        tagged_payload = unsigned.model_dump(mode="json")
        tagged_payload["promotion_auth_tag"] = self._mac(
            unsigned.unsigned_hash()
        )
        tagged = PrivateQualificationReceiptV52(**tagged_payload)
        final_payload = tagged.model_dump(mode="json")
        final_payload["qualification_hash"] = tagged.content_hash()
        return PrivateQualificationReceiptV52(**final_payload)

    def verify(self, receipt: PrivateQualificationReceiptV52) -> bool:
        try:
            return bool(
                receipt.qualification_hash
                and receipt.qualification_hash == receipt.content_hash()
                and receipt.promotion_key_id == self.key_id
                and receipt.promotion_auth_tag
                and hmac.compare_digest(
                    receipt.promotion_auth_tag,
                    self._mac(receipt.unsigned_hash()),
                )
            )
        except (TypeError, ValueError):
            return False


class LocalPrivateRunReceiptV52(StrictModel):
    schema_version: Literal["5.2"] = "5.2"
    process_id: Annotated[int, Field(ge=1)]
    command_hash: Sha256
    exit_code: int
    stdout_hash: Sha256
    stderr_hash: Sha256
    request_bytes_hash: Sha256
    prediction_bytes_hash: Sha256
    capsule_bytes_hash: Sha256
    worker_output_hash: Sha256
    verified_worker_executable_hash: Sha256
    verified_runner_source_hash: Sha256
    verified_environment_fingerprint: Sha256
    verified_event_chain_hash: Sha256
    fresh_process: Literal[True] = True
    isolation_level: Literal["same_host_process"] = "same_host_process"
    host_secrecy_attested: Literal[False] = False
    scientific_qualification_permitted: Literal[False] = False
    run_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_run(self) -> "LocalPrivateRunReceiptV52":
        if self.run_hash and self.run_hash != self.content_hash():
            raise ValueError("run_hash does not match local process receipt")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "run_hash")

    def assert_sealed(self) -> None:
        if not self.run_hash or self.run_hash != self.content_hash():
            raise ValueError("local process receipt is not sealed")


def evaluate_private_inputs_v52(
    *,
    request: PrivateEvaluationRequestV52,
    prediction: PredictionDocumentV50,
    prediction_bytes_hash: str,
    capsule: PrivateCaseCapsuleV50,
    worker_authority: PrivateWorkerAuthorityV52,
    worker_id: str,
    worker_host_id: str,
    worker_executable_hash: str,
    runner_source_hash: str,
) -> PrivateWorkerReceiptV52:
    """Private-worker implementation; returns commitments and scores only."""

    request.assert_sealed()
    capsule.assert_sealed()
    reasons: list[str] = []
    if capsule.case_id != request.case_id:
        reasons.append("capsule_case_mismatch")
    if capsule.public_case_hash != request.public_case_hash:
        reasons.append("public_case_binding_mismatch")
    if capsule.capsule_hash != request.private_capsule_commitment:
        reasons.append("capsule_commitment_mismatch")
    if prediction_bytes_hash != request.prediction_snapshot_hash:
        reasons.append("prediction_snapshot_hash_mismatch")
    semantic_hash = sha256_value(prediction)
    if semantic_hash != request.prediction_semantic_hash:
        reasons.append("prediction_semantic_hash_mismatch")
    if capsule.secrecy_canary.encode("utf-8") in canonical_json(
        prediction
    ).encode("utf-8"):
        reasons.append("secrecy_canary_in_prediction")

    prediction_by_id = {
        point.target_id: point.value for point in prediction.predictions
    }
    target_by_id = {
        target.target_id: target.value for target in capsule.holdout
    }
    domain_valid = (
        prediction.case_id == request.case_id
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
        max(0.0, min(1.0, 1.0 - raw_mae / capsule.quality_scale))
        if integrity_valid and raw_mae is not None
        else 0.0
    )
    threshold_passed = (
        integrity_valid
        and domain_valid
        and quality_score >= request.minimum_quality_score
    )
    if integrity_valid and domain_valid and not threshold_passed:
        reasons.append("private_threshold_failed")
    event_hashes, event_chain_hash = _private_event_chain(
        request_hash=request.request_hash,
        prediction_snapshot_hash=prediction_bytes_hash,
        capsule_commitment=str(capsule.capsule_hash),
        quality_score=quality_score,
        integrity_valid=integrity_valid,
        domain_valid=domain_valid,
    )
    return worker_authority.issue(
        request_hash=request.request_hash,
        case_id=request.case_id,
        worker_id=worker_id,
        worker_host_id=worker_host_id,
        worker_process_id=os.getpid(),
        isolation_level="same_host_process",
        worker_executable_hash=worker_executable_hash,
        runner_source_hash=runner_source_hash,
        environment_fingerprint=runtime_environment_fingerprint_v52(),
        prediction_snapshot_hash=prediction_bytes_hash,
        prediction_semantic_hash=semantic_hash,
        private_capsule_commitment=capsule.capsule_hash,
        raw_mae=raw_mae,
        quality_score=quality_score,
        integrity_valid=integrity_valid,
        prediction_domain_valid=domain_valid,
        threshold_passed=threshold_passed,
        event_hashes=event_hashes,
        event_chain_hash=event_chain_hash,
        reason_codes=sorted(set(reasons)),
    )


def run_local_private_worker_v52(
    *,
    request: PrivateEvaluationRequestV52,
    prediction_path: str | Path,
    private_capsule_path: str | Path,
    worker_secret: bytes,
    worker_key_id: str,
    worker_id: str,
    worker_host_id: str,
    output_directory: str | Path,
) -> tuple[PrivateWorkerReceiptV52, LocalPrivateRunReceiptV52]:
    """Run the private evaluator in a fresh local process."""

    request.assert_sealed()
    if len(worker_secret) < 32:
        raise ValueError("worker secret needs at least 32 bytes")
    prediction = Path(prediction_path).resolve()
    capsule = Path(private_capsule_path).resolve()
    output_dir = Path(output_directory).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / f"{request.request_hash}.request.json"
    output_path = output_dir / f"{request.request_hash}.worker.json"
    request_path.write_text(
        canonical_json(request) + "\n", encoding="utf-8", newline="\n"
    )
    module_root = Path(__file__).resolve().parents[2]
    runner_path = Path(__file__).with_name("private_worker.py").resolve()
    with tempfile.TemporaryDirectory(
        prefix="fma-v52-private-", dir=output_dir
    ) as temporary:
        secret_path = Path(temporary) / "worker.key"
        secret_path.write_bytes(worker_secret)
        try:
            os.chmod(secret_path, 0o600)
        except OSError:
            pass
        command = [
            sys.executable,
            str(runner_path),
            "--request",
            str(request_path),
            "--prediction",
            str(prediction),
            "--capsule",
            str(capsule),
            "--secret-file",
            str(secret_path),
            "--worker-key-id",
            worker_key_id,
            "--worker-id",
            worker_id,
            "--worker-host-id",
            worker_host_id,
            "--output",
            str(output_path),
        ]
        environment = {
            key: value
            for key, value in os.environ.items()
            if key.upper() in {"PATH", "SYSTEMROOT", "WINDIR", "PATHEXT"}
        }
        environment["PYTHONPATH"] = str(module_root)
        completed = subprocess.run(
            command,
            cwd=module_root,
            env=environment,
            capture_output=True,
            check=False,
            timeout=120,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            "private worker failed with exit code "
            f"{completed.returncode}; stderr hash="
            f"{hashlib.sha256(completed.stderr).hexdigest()}"
        )
    worker_output = output_path.read_bytes()
    receipt = PrivateWorkerReceiptV52.model_validate_json(worker_output)
    expected_executable_hash = _file_hash(Path(sys.executable))
    expected_runner_hash = _file_hash(runner_path)
    expected_environment = runtime_environment_fingerprint_v52()
    worker_authority = PrivateWorkerAuthorityV52(
        key_id=worker_key_id, secret=worker_secret
    )
    if not worker_authority.verify(receipt):
        raise RuntimeError("private worker signature verification failed")
    if receipt.request_hash != request.request_hash:
        raise RuntimeError("private worker returned another request")
    if receipt.prediction_snapshot_hash != _file_hash(prediction):
        raise RuntimeError("private worker returned another prediction")
    if receipt.private_capsule_commitment != request.private_capsule_commitment:
        raise RuntimeError("private worker returned another capsule commitment")
    if receipt.worker_executable_hash != expected_executable_hash:
        raise RuntimeError("private worker executable identity differs")
    if receipt.runner_source_hash != expected_runner_hash:
        raise RuntimeError("private worker source identity differs")
    if receipt.environment_fingerprint != expected_environment:
        raise RuntimeError("private worker environment identity differs")
    local_draft = LocalPrivateRunReceiptV52(
        process_id=receipt.worker_process_id,
        command_hash=sha256_value(
            [Path(command[0]).name, Path(command[1]).name, *command[2::2]]
        ),
        exit_code=completed.returncode,
        stdout_hash=hashlib.sha256(completed.stdout).hexdigest(),
        stderr_hash=hashlib.sha256(completed.stderr).hexdigest(),
        request_bytes_hash=_file_hash(request_path),
        prediction_bytes_hash=_file_hash(prediction),
        capsule_bytes_hash=_file_hash(capsule),
        worker_output_hash=hashlib.sha256(worker_output).hexdigest(),
        verified_worker_executable_hash=expected_executable_hash,
        verified_runner_source_hash=expected_runner_hash,
        verified_environment_fingerprint=expected_environment,
        verified_event_chain_hash=receipt.event_chain_hash,
    )
    local_payload = local_draft.model_dump(exclude={"run_hash"})
    local_payload["run_hash"] = local_draft.content_hash()
    return receipt, LocalPrivateRunReceiptV52(**local_payload)
