"""Independent signed promotion for a graph-bound V5.3 campaign."""

from __future__ import annotations

import base64
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
from fma.v5.stage_workspace import StageWorkspaceV50

from .campaign import (
    I32GraphBindingResultV53,
    PredictionRegistrationV53,
    PublicPredictionRegistryV53,
    verify_i32_graph_binding_v53,
)
from .custody import (
    CustodyVerificationReceiptV53,
    ExternalCustodyAttestationV53,
)
from .external_private import (
    ExternalPrivateRunVerificationV53,
    ExternalPrivateWorkerReceiptV53,
    ExternalWorkerHostAttestationV53,
)
from .ode_forecast import ODEForecastBundleV53


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ExternalPromotionDecisionV53(StrictModel):
    """Decision signed by an authority outside generation and evaluation."""

    schema_version: Literal["5.3-external-promotion"] = "5.3-external-promotion"
    campaign_id: Identifier
    graph_binding_hash: Sha256
    prediction_registration_hash: Sha256
    custody_verification_receipt_hash: Sha256
    private_run_verification_hash: Sha256
    public_scientific_acceptance: bool
    non_fixture_evidence: bool
    graph_binding_current_verified: bool
    prediction_registration_verified: bool
    private_run_verified: bool
    private_threshold_passed: bool
    external_anchor_content_verified: bool
    integrity_incident_free: bool
    decision: Literal["QUALIFY", "REJECT"]
    reason_codes: list[Identifier]
    qualification_granted: bool
    private_acceptance_data_exposed: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    decided_at: datetime
    promotion_key_id: Identifier
    signature_base64: Annotated[str, Field(min_length=40)] | None = None
    decision_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> "ExternalPromotionDecisionV53":
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("promotion reasons must be sorted and unique")
        required = (
            self.public_scientific_acceptance
            and self.non_fixture_evidence
            and self.graph_binding_current_verified
            and self.prediction_registration_verified
            and self.private_run_verified
            and self.private_threshold_passed
            and self.external_anchor_content_verified
            and self.integrity_incident_free
            and not self.reason_codes
        )
        if self.qualification_granted != (self.decision == "QUALIFY"):
            raise ValueError("promotion decision and qualification flag differ")
        if self.qualification_granted != required:
            raise ValueError("qualification differs from mandatory evidence")
        if self.decided_at.utcoffset() is None:
            raise ValueError("decided_at must be timezone-aware")
        if self.decision_hash and (
            not self.signature_base64 or self.decision_hash != self.content_hash()
        ):
            raise ValueError("promotion decision envelope differs")
        return self

    def unsigned_bytes(self) -> bytes:
        return canonical_json(
            self.model_dump(
                mode="json",
                exclude={"signature_base64", "decision_hash"},
            )
        ).encode("utf-8")

    def content_hash(self) -> str:
        return _hash_without(self, "decision_hash")

    def assert_sealed(self) -> None:
        if (
            not self.signature_base64
            or not self.decision_hash
            or self.decision_hash != self.content_hash()
        ):
            raise ValueError("promotion decision is not sealed")


class ScientificQualificationReceiptV53(StrictModel):
    schema_version: Literal["5.3"] = "5.3"
    campaign_id: Identifier
    graph_binding_hash: Sha256
    prediction_registration_hash: Sha256
    private_run_verification_hash: Sha256
    promotion_decision_hash: Sha256 | None = None
    status: Literal["NOT_RUN", "REJECTED", "SCIENTIFICALLY_QUALIFIED"]
    qualification_granted: bool
    reason_codes: list[Identifier]
    graph_binding_valid: bool
    prediction_registration_valid: bool
    promotion_signature_valid: bool
    public_scientific_acceptance: bool
    non_fixture_evidence: bool
    private_run_verified: bool
    private_threshold_passed: bool
    external_anchor_content_verified: bool
    integrity_incident_free: bool
    private_acceptance_data_exposed: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    assessed_at: datetime
    qualification_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "ScientificQualificationReceiptV53":
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("qualification reasons must be sorted and unique")
        if self.qualification_granted != (self.status == "SCIENTIFICALLY_QUALIFIED"):
            raise ValueError("qualification status and flag differ")
        if self.qualification_granted and (
            not self.promotion_decision_hash
            or not self.graph_binding_valid
            or not self.prediction_registration_valid
            or not self.promotion_signature_valid
            or not self.public_scientific_acceptance
            or not self.non_fixture_evidence
            or not self.private_run_verified
            or not self.private_threshold_passed
            or not self.external_anchor_content_verified
            or not self.integrity_incident_free
            or self.reason_codes
        ):
            raise ValueError("qualified receipt lacks mandatory evidence")
        if self.assessed_at.utcoffset() is None:
            raise ValueError("assessed_at must be timezone-aware")
        if self.qualification_hash and (self.qualification_hash != self.content_hash()):
            raise ValueError("qualification receipt hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "qualification_hash")

    @classmethod
    def seal(cls, **data: object) -> "ScientificQualificationReceiptV53":
        data.setdefault("assessed_at", _utc_now())
        draft = cls(**data)
        payload = draft.model_dump(exclude={"qualification_hash"})
        payload["qualification_hash"] = draft.content_hash()
        return cls(**payload)


def sign_external_promotion_decision_v53(
    *,
    campaign_id: str,
    graph_binding: I32GraphBindingResultV53,
    prediction_registration: PredictionRegistrationV53,
    custody_verification: CustodyVerificationReceiptV53,
    private_run_verification: ExternalPrivateRunVerificationV53,
    forecast_bundle: ODEForecastBundleV53,
    graph_binding_current_verified: bool,
    prediction_registration_verified: bool,
    external_anchor_content_verified: bool,
    integrity_incident_free: bool,
    promotion_key_id: str,
    promotion_private_key_pem: bytes,
    decided_at: datetime | None = None,
) -> ExternalPromotionDecisionV53:
    """External signer derives its decision; callers cannot request QUALIFY."""

    reasons: list[str] = []
    conditions = {
        "public_scientific_evidence_failed": (forecast_bundle.scientific_acceptance),
        "fixture_only_evidence": not forecast_bundle.fixture_only,
        "graph_binding_invalid": graph_binding_current_verified,
        "prediction_registration_invalid": (prediction_registration_verified),
        "private_run_invalid": (private_run_verification.status == "VERIFIED"),
        "private_threshold_failed": (private_run_verification.private_threshold_passed),
        "external_anchor_unverified": external_anchor_content_verified,
        "integrity_incident_present": integrity_incident_free,
    }
    for reason, passed in conditions.items():
        if not passed:
            reasons.append(reason)
    qualify = not reasons
    unsigned = ExternalPromotionDecisionV53(
        campaign_id=campaign_id,
        graph_binding_hash=graph_binding.binding.binding_hash,
        prediction_registration_hash=(prediction_registration.registration_hash),
        custody_verification_receipt_hash=custody_verification.receipt_hash,
        private_run_verification_hash=(private_run_verification.verification_hash),
        public_scientific_acceptance=(forecast_bundle.scientific_acceptance),
        non_fixture_evidence=not forecast_bundle.fixture_only,
        graph_binding_current_verified=graph_binding_current_verified,
        prediction_registration_verified=(prediction_registration_verified),
        private_run_verified=(private_run_verification.status == "VERIFIED"),
        private_threshold_passed=(private_run_verification.private_threshold_passed),
        external_anchor_content_verified=(external_anchor_content_verified),
        integrity_incident_free=integrity_incident_free,
        decision="QUALIFY" if qualify else "REJECT",
        reason_codes=sorted(reasons),
        qualification_granted=qualify,
        decided_at=decided_at or _utc_now(),
        promotion_key_id=promotion_key_id,
    )
    key = serialization.load_pem_private_key(promotion_private_key_pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("promotion key must be Ed25519")
    signature = key.sign(unsigned.unsigned_bytes())
    tagged_payload = unsigned.model_dump(mode="json")
    tagged_payload["signature_base64"] = base64.b64encode(signature).decode("ascii")
    tagged = ExternalPromotionDecisionV53(**tagged_payload)
    final_payload = tagged.model_dump(mode="json")
    final_payload["decision_hash"] = tagged.content_hash()
    return ExternalPromotionDecisionV53(**final_payload)


def _verify_promotion_signature(
    decision: ExternalPromotionDecisionV53,
    trusted_public_keys: Mapping[str, bytes],
) -> bool:
    try:
        decision.assert_sealed()
        key_bytes = trusted_public_keys[decision.promotion_key_id]
        key = serialization.load_pem_public_key(key_bytes)
        if not isinstance(key, Ed25519PublicKey):
            return False
        key.verify(
            base64.b64decode(decision.signature_base64.encode("ascii"), validate=True),
            decision.unsigned_bytes(),
        )
        return True
    except (
        InvalidSignature,
        KeyError,
        TypeError,
        ValueError,
    ):
        return False


def assess_scientific_qualification_v53(
    *,
    campaign_id: str,
    workspace: StageWorkspaceV50,
    graph_binding: I32GraphBindingResultV53,
    registry: PublicPredictionRegistryV53,
    prediction_registration: PredictionRegistrationV53,
    forecast_bundle: ODEForecastBundleV53,
    custody_attestation: ExternalCustodyAttestationV53,
    custody_verification: CustodyVerificationReceiptV53,
    worker_receipt: ExternalPrivateWorkerReceiptV53,
    worker_host_attestation: ExternalWorkerHostAttestationV53,
    private_run_verification: ExternalPrivateRunVerificationV53,
    promotion_decision: ExternalPromotionDecisionV53 | None,
    trusted_promotion_public_keys: Mapping[str, bytes],
) -> ScientificQualificationReceiptV53:
    """Final code-owned assessment; missing external promotion is NOT_RUN."""

    graph_valid = verify_i32_graph_binding_v53(
        workspace=workspace, result=graph_binding
    )
    registration_valid = registry.verify_registration(prediction_registration)
    promotion_signature_valid = bool(
        promotion_decision
        and _verify_promotion_signature(
            promotion_decision, trusted_promotion_public_keys
        )
    )
    reasons: list[str] = []
    if promotion_decision is None:
        reasons.append("external_promotion_decision_missing")
    else:
        bindings = {
            "graph_binding": (
                promotion_decision.graph_binding_hash,
                graph_binding.binding.binding_hash,
            ),
            "prediction_registration": (
                promotion_decision.prediction_registration_hash,
                prediction_registration.registration_hash,
            ),
            "custody_verification": (
                promotion_decision.custody_verification_receipt_hash,
                custody_verification.receipt_hash,
            ),
            "private_run_verification": (
                promotion_decision.private_run_verification_hash,
                private_run_verification.verification_hash,
            ),
        }
        if any(first != second for first, second in bindings.values()):
            reasons.append("promotion_public_binding_mismatch")
        if not promotion_signature_valid:
            reasons.append("promotion_signature_invalid")
        authority_ids = {
            custody_attestation.attester_key_id,
            worker_receipt.worker_key_id,
            worker_host_attestation.host_attester_key_id,
            promotion_decision.promotion_key_id,
        }
        if len(authority_ids) != 4:
            reasons.append("promotion_authority_not_independent")
        if promotion_decision.decision != "QUALIFY":
            reasons.extend(promotion_decision.reason_codes)
    if not graph_valid:
        reasons.append("graph_binding_invalid")
    if not registration_valid:
        reasons.append("prediction_registration_invalid")
    if not forecast_bundle.scientific_acceptance:
        reasons.append("public_scientific_evidence_failed")
    if forecast_bundle.fixture_only:
        reasons.append("fixture_only_evidence")
    if private_run_verification.status != "VERIFIED":
        reasons.append("private_run_invalid")
    if not private_run_verification.private_threshold_passed:
        reasons.append("private_threshold_failed")
    reasons = sorted(set(reasons))

    qualified = bool(
        promotion_decision
        and promotion_decision.qualification_granted
        and promotion_signature_valid
        and not reasons
    )
    status: Literal["NOT_RUN", "REJECTED", "SCIENTIFICALLY_QUALIFIED"]
    if promotion_decision is None:
        status = "NOT_RUN"
    elif qualified:
        status = "SCIENTIFICALLY_QUALIFIED"
    else:
        status = "REJECTED"
    return ScientificQualificationReceiptV53.seal(
        campaign_id=campaign_id,
        graph_binding_hash=graph_binding.binding.binding_hash,
        prediction_registration_hash=(prediction_registration.registration_hash),
        private_run_verification_hash=(private_run_verification.verification_hash),
        promotion_decision_hash=(
            promotion_decision.decision_hash if promotion_decision else None
        ),
        status=status,
        qualification_granted=qualified,
        reason_codes=reasons,
        graph_binding_valid=graph_valid,
        prediction_registration_valid=registration_valid,
        promotion_signature_valid=promotion_signature_valid,
        public_scientific_acceptance=(forecast_bundle.scientific_acceptance),
        non_fixture_evidence=not forecast_bundle.fixture_only,
        private_run_verified=(private_run_verification.status == "VERIFIED"),
        private_threshold_passed=(private_run_verification.private_threshold_passed),
        external_anchor_content_verified=bool(
            promotion_decision and promotion_decision.external_anchor_content_verified
        ),
        integrity_incident_free=bool(
            promotion_decision and promotion_decision.integrity_incident_free
        ),
    )


__all__ = [
    "ExternalPromotionDecisionV53",
    "ScientificQualificationReceiptV53",
    "assess_scientific_qualification_v53",
    "sign_external_promotion_decision_v53",
]
