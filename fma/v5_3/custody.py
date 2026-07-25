"""Verification-only public contracts for an external private-data custodian.

The coordinator stores trusted Ed25519 public keys, never custodian private
keys or private target values.  A valid receipt proves that a pinned key signed
the stated commitment and isolation claims.  It does not by itself prove the
claims true, validate an external timestamp anchor, or grant qualification.
"""

from __future__ import annotations

import base64
import hashlib
import math
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
    PrivateCaseCapsuleV50,
    PrivateTargetV50,
)

from .ode_forecast import ODEForecastPlanV53


FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _signed_bytes(model: StrictModel) -> bytes:
    return canonical_json(
        model.model_dump(
            mode="json",
            exclude={"signature_base64", "attestation_hash"},
        )
    ).encode("utf-8")


class PrivateScoreContractV53(StrictModel):
    """Public scoring contract frozen before a custodian sees source values."""

    schema_version: Literal["5.3"] = "5.3"
    contract_id: Identifier
    case_id: Identifier
    protocol_hash: Sha256
    public_case_hash: Sha256
    forecast_plan_hash: Sha256
    target_ids: Annotated[list[Identifier], Field(min_length=1)]
    metric: Literal["mean_absolute_error"] = "mean_absolute_error"
    quality_scale: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    minimum_quality_score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    maximum_private_evaluations: Literal[1] = 1
    frozen_at: datetime
    contract_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "PrivateScoreContractV53":
        if self.target_ids != sorted(set(self.target_ids)):
            raise ValueError("private target IDs must be sorted and unique")
        if self.frozen_at.utcoffset() is None:
            raise ValueError("frozen_at must be timezone-aware")
        if self.contract_hash and self.contract_hash != self.content_hash():
            raise ValueError("contract_hash does not match score contract")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "contract_hash")

    def assert_sealed(self) -> None:
        if not self.contract_hash or self.contract_hash != self.content_hash():
            raise ValueError("private score contract is not sealed")

    def assert_forecast_plan(self, plan: ODEForecastPlanV53) -> None:
        self.assert_sealed()
        plan.assert_sealed()
        if self.forecast_plan_hash != plan.plan_hash:
            raise ValueError("score contract is bound to another forecast plan")
        if self.target_ids != [item.target_id for item in plan.targets]:
            raise ValueError("score contract targets differ from forecast plan")

    @classmethod
    def seal(cls, **data: object) -> "PrivateScoreContractV53":
        data.setdefault("frozen_at", _utc_now())
        draft = cls(**data)
        payload = draft.model_dump(exclude={"contract_hash"})
        payload["contract_hash"] = draft.content_hash()
        return cls(**payload)


class ExternalCustodyAttestationV53(StrictModel):
    """Public statement signed on the separately administered custodian."""

    schema_version: Literal["5.3-external-custody"] = "5.3-external-custody"
    attestation_id: Identifier
    score_contract_hash: Sha256
    protocol_hash: Sha256
    public_case_hash: Sha256
    forecast_plan_hash: Sha256
    capsule_commitment: Sha256
    capsule_bytes_hash: Sha256
    private_source_manifest_hash: Sha256
    target_id_commitment: Sha256
    custodian_host_id: Identifier
    coordinator_host_id: Identifier
    generator_host_id: Identifier
    capsule_created_before_generator_release: Literal[True] = True
    generator_egress_denied_until_registration: Literal[True] = True
    independent_management_key_control: Literal[True] = True
    private_values_disclosed: Literal[False] = False
    external_anchor_receipt_hash: Sha256
    attested_at: datetime
    attester_key_id: Identifier
    signature_base64: Annotated[str, Field(min_length=40)] | None = None
    attestation_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_attestation(self) -> "ExternalCustodyAttestationV53":
        if self.custodian_host_id in {
            self.coordinator_host_id,
            self.generator_host_id,
        }:
            raise ValueError("custodian host must differ from generator/coordinator")
        if self.attested_at.utcoffset() is None:
            raise ValueError("attested_at must be timezone-aware")
        if self.attestation_hash and (
            not self.signature_base64 or self.attestation_hash != self.content_hash()
        ):
            raise ValueError("attestation hash/signature envelope differs")
        return self

    def unsigned_bytes(self) -> bytes:
        return _signed_bytes(self)

    def content_hash(self) -> str:
        return _hash_without(self, "attestation_hash")

    def assert_sealed(self) -> None:
        if (
            not self.signature_base64
            or not self.attestation_hash
            or self.attestation_hash != self.content_hash()
        ):
            raise ValueError("external custody attestation is not sealed")


class CustodyVerificationReceiptV53(StrictModel):
    """Coordinator-side verification; explicitly not scientific promotion."""

    schema_version: Literal["5.3"] = "5.3"
    attestation_hash: Sha256
    score_contract_hash: Sha256
    trusted_key_set_hash: Sha256
    status: Literal["VERIFIED", "REJECTED"]
    reason_codes: list[Identifier]
    signature_valid: bool
    public_bindings_valid: bool
    separate_host_claim_bound: bool
    external_anchor_content_verified: Literal[False] = False
    qualification_granted: Literal[False] = False
    private_acceptance_data_exposed: Literal[False] = False
    verified_at: datetime
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "CustodyVerificationReceiptV53":
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("reason codes must be sorted and unique")
        expected = (
            self.signature_valid
            and self.public_bindings_valid
            and self.separate_host_claim_bound
            and not self.reason_codes
        )
        if (self.status == "VERIFIED") != expected:
            raise ValueError("custody verification status differs from checks")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("custody verification receipt hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    @classmethod
    def seal(cls, **data: object) -> "CustodyVerificationReceiptV53":
        data.setdefault("verified_at", _utc_now())
        draft = cls(**data)
        payload = draft.model_dump(exclude={"receipt_hash"})
        payload["receipt_hash"] = draft.content_hash()
        return cls(**payload)


def public_key_fingerprint_v53(public_key_pem: bytes) -> str:
    key = serialization.load_pem_public_key(public_key_pem)
    if not isinstance(key, Ed25519PublicKey):
        raise TypeError("custody key must be Ed25519")
    raw = key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


def sign_external_custody_attestation_v53(
    *,
    private_key_pem: bytes,
    **data: object,
) -> ExternalCustodyAttestationV53:
    """External-only signer; callers must keep private key material off coordinator."""

    key = serialization.load_pem_private_key(private_key_pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("custody key must be Ed25519")
    data.setdefault("attested_at", _utc_now())
    unsigned = ExternalCustodyAttestationV53(**data)
    signature = key.sign(unsigned.unsigned_bytes())
    tagged_payload = unsigned.model_dump(mode="json")
    tagged_payload["signature_base64"] = base64.b64encode(signature).decode("ascii")
    tagged = ExternalCustodyAttestationV53(**tagged_payload)
    final_payload = tagged.model_dump(mode="json")
    final_payload["attestation_hash"] = tagged.content_hash()
    return ExternalCustodyAttestationV53(**final_payload)


def verify_external_custody_attestation_v53(
    *,
    attestation: ExternalCustodyAttestationV53,
    score_contract: PrivateScoreContractV53,
    forecast_plan: ODEForecastPlanV53,
    trusted_public_keys: Mapping[str, bytes],
    expected_coordinator_host_id: str,
    expected_generator_host_id: str,
) -> CustodyVerificationReceiptV53:
    """Verify only public bindings and a pinned signature; fail closed."""

    score_contract.assert_forecast_plan(forecast_plan)
    reasons: list[str] = []
    signature_valid = False
    public_bindings_valid = (
        attestation.score_contract_hash == score_contract.contract_hash
        and attestation.protocol_hash == score_contract.protocol_hash
        and attestation.public_case_hash == score_contract.public_case_hash
        and attestation.forecast_plan_hash == forecast_plan.plan_hash
        and attestation.target_id_commitment == sha256_value(score_contract.target_ids)
    )
    if not public_bindings_valid:
        reasons.append("public_binding_mismatch")
    separate_host = (
        attestation.coordinator_host_id == expected_coordinator_host_id
        and attestation.generator_host_id == expected_generator_host_id
        and attestation.custodian_host_id
        not in {expected_coordinator_host_id, expected_generator_host_id}
    )
    if not separate_host:
        reasons.append("separate_host_claim_invalid")
    try:
        attestation.assert_sealed()
    except ValueError:
        reasons.append("attestation_envelope_invalid")
    public_key_pem = trusted_public_keys.get(attestation.attester_key_id)
    if public_key_pem is None:
        reasons.append("attester_key_not_pinned")
    elif attestation.signature_base64:
        try:
            public_key = serialization.load_pem_public_key(public_key_pem)
            if not isinstance(public_key, Ed25519PublicKey):
                raise TypeError("pinned key is not Ed25519")
            public_key.verify(
                base64.b64decode(
                    attestation.signature_base64.encode("ascii"),
                    validate=True,
                ),
                attestation.unsigned_bytes(),
            )
            signature_valid = True
        except (InvalidSignature, TypeError, ValueError):
            reasons.append("attestation_signature_invalid")
    else:
        reasons.append("attestation_signature_missing")
    reasons = sorted(set(reasons))
    trusted_key_set_hash = sha256_value(
        {
            key_id: public_key_fingerprint_v53(public_key)
            for key_id, public_key in sorted(trusted_public_keys.items())
        }
    )
    return CustodyVerificationReceiptV53.seal(
        attestation_hash=(
            attestation.attestation_hash if attestation.attestation_hash else "0" * 64
        ),
        score_contract_hash=score_contract.contract_hash,
        trusted_key_set_hash=trusted_key_set_hash,
        status="VERIFIED" if not reasons else "REJECTED",
        reason_codes=reasons,
        signature_valid=signature_valid,
        public_bindings_valid=public_bindings_valid,
        separate_host_claim_bound=separate_host,
    )


def create_external_capsule_and_attestation_v53(
    *,
    score_contract: PrivateScoreContractV53,
    private_targets: list[PrivateTargetV50],
    secrecy_canary: str,
    private_source_manifest_hash: str,
    external_anchor_receipt_hash: str,
    custodian_host_id: str,
    coordinator_host_id: str,
    generator_host_id: str,
    attestation_id: str,
    attester_key_id: str,
    private_key_pem: bytes,
    attested_at: datetime | None = None,
) -> tuple[
    PrivateCaseCapsuleV50,
    bytes,
    ExternalCustodyAttestationV53,
]:
    """Construct private bytes on the custodian and return their public commitment."""

    score_contract.assert_sealed()
    if [item.target_id for item in private_targets] != score_contract.target_ids:
        raise ValueError("private targets differ from the frozen score contract")
    if any(not math.isfinite(item.value) for item in private_targets):
        raise ValueError("private target values must be finite")
    capsule = PrivateCaseCapsuleV50.seal(
        case_id=score_contract.case_id,
        public_case_hash=score_contract.public_case_hash,
        holdout=private_targets,
        quality_scale=score_contract.quality_scale,
        secrecy_canary=secrecy_canary,
    )
    capsule_bytes = (canonical_json(capsule) + "\n").encode("utf-8")
    attestation = sign_external_custody_attestation_v53(
        private_key_pem=private_key_pem,
        attestation_id=attestation_id,
        score_contract_hash=score_contract.contract_hash,
        protocol_hash=score_contract.protocol_hash,
        public_case_hash=score_contract.public_case_hash,
        forecast_plan_hash=score_contract.forecast_plan_hash,
        capsule_commitment=capsule.capsule_hash,
        capsule_bytes_hash=hashlib.sha256(capsule_bytes).hexdigest(),
        private_source_manifest_hash=private_source_manifest_hash,
        target_id_commitment=sha256_value(score_contract.target_ids),
        custodian_host_id=custodian_host_id,
        coordinator_host_id=coordinator_host_id,
        generator_host_id=generator_host_id,
        external_anchor_receipt_hash=external_anchor_receipt_hash,
        attested_at=attested_at or _utc_now(),
        attester_key_id=attester_key_id,
    )
    return capsule, capsule_bytes, attestation


__all__ = [
    "CustodyVerificationReceiptV53",
    "ExternalCustodyAttestationV53",
    "PrivateScoreContractV53",
    "create_external_capsule_and_attestation_v53",
    "public_key_fingerprint_v53",
    "sign_external_custody_attestation_v53",
    "verify_external_custody_attestation_v53",
]
