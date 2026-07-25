"""Fail-closed V5.5 path from public eligibility to encrypted private scoring.

The public chain is verified before a caller needs the private-target AES key.
Historical V5.3 receipts remain unchanged and are wrapped by an additive V5.5
receipt that binds authorization, split custody, and a create-once budget claim.
"""

from __future__ import annotations

import base64
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Mapping, Literal

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
from fma.v5_3.custody import (
    ExternalCustodyAttestationV53,
    PrivateScoreContractV53,
    sign_external_custody_attestation_v53,
)
from fma.v5_3.external_private import (
    ExternalPrivateWorkerReceiptV53,
    PrivateEvaluationRequestV53,
    evaluate_external_private_inputs_v53,
)
from fma.v5_4.public_eligibility import (
    PrivateEvaluationAuthorizationV54,
    PublicEligibilityAssessmentV54,
    PublicEligibilityContractV54,
    PublicEligibilityInputV54,
    PublicEligibilityReceiptV54,
    assess_public_eligibility_v54,
    verify_private_evaluation_authorization_v54,
)

from .campaign_protocol import (
    CandidateSelectionPolicyV55,
    ProspectiveCampaignProtocolV55,
    PublicLaunchBindingV55,
    verify_public_launch_binding_v55,
)
from .split_custody import (
    EncryptedCustodyEnvelopeV55,
    SplitCustodyAttestationV55,
    open_private_target_envelope_v55,
    signing_key_fingerprint_v55,
    verify_split_custody_attestation_signature_v55,
    verify_split_custody_bindings_v55,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _load_private_key(private_key_pem: bytes) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(private_key_pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("signing key must be Ed25519")
    return key


def _load_public_key(public_key_pem: bytes) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(public_key_pem)
    if not isinstance(key, Ed25519PublicKey):
        raise TypeError("verification key must be Ed25519")
    return key


def _public_key_pem(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _verify_signature(
    *,
    public_key_pem: bytes,
    signature_base64: str | None,
    payload: bytes,
) -> bool:
    if not signature_base64:
        return False
    try:
        _load_public_key(public_key_pem).verify(
            base64.b64decode(signature_base64, validate=True),
            payload,
        )
        return True
    except (InvalidSignature, TypeError, ValueError):
        return False


class LegacyCustodyBridgeV55(StrictModel):
    """Code-owned exact binding between V5.3 and V5.5 custody artifacts."""

    schema_version: Literal["5.5-legacy-custody-bridge"] = (
        "5.5-legacy-custody-bridge"
    )
    case_id: Identifier
    protocol_hash: Sha256
    score_contract_hash: Sha256
    forecast_plan_hash: Sha256
    v53_custody_attestation_hash: Sha256
    split_custody_attestation_hash: Sha256
    private_capsule_commitment: Sha256
    capsule_bytes_hash: Sha256
    private_target_envelope_hash: Sha256
    source_provenance_commitment: Sha256
    custody_key_id: Identifier
    custodian_host_id: Identifier
    coordinator_host_id: Identifier
    generator_host_id: Identifier
    bridge_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bridge(self) -> "LegacyCustodyBridgeV55":
        if self.custodian_host_id in {
            self.coordinator_host_id,
            self.generator_host_id,
        }:
            raise ValueError("custodian host must differ from generator/coordinator")
        if self.bridge_hash and self.bridge_hash != self.content_hash():
            raise ValueError("legacy custody bridge hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "bridge_hash")

    def assert_sealed(self) -> None:
        if not self.bridge_hash or self.bridge_hash != self.content_hash():
            raise ValueError("legacy custody bridge is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "LegacyCustodyBridgeV55":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"bridge_hash"})
        payload["bridge_hash"] = draft.content_hash()
        return cls(**payload)


class PrivateEvaluationBudgetClaimV55(StrictModel):
    """Create-once claim written before private-target key access."""

    schema_version: Literal["5.5-private-evaluation-budget-claim"] = (
        "5.5-private-evaluation-budget-claim"
    )
    request_hash: Sha256
    budget_ledger_id: Identifier
    authorization_hash: Sha256
    private_target_envelope_hash: Sha256
    split_custody_attestation_hash: Sha256
    maximum_private_evaluations: Literal[1] = 1
    claimed_private_evaluation_count: Literal[1] = 1
    target_accessed_at_claim: Literal[False] = False
    private_evaluation_performed_at_claim: Literal[False] = False
    fixture_only: bool
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    claimed_at: datetime
    claim_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_claim(self) -> "PrivateEvaluationBudgetClaimV55":
        if self.claimed_at.utcoffset() is None:
            raise ValueError("budget claim time must be timezone-aware")
        if self.claim_hash and self.claim_hash != self.content_hash():
            raise ValueError("private evaluation budget claim hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "claim_hash")

    def assert_sealed(self) -> None:
        if not self.claim_hash or self.claim_hash != self.content_hash():
            raise ValueError("private evaluation budget claim is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PrivateEvaluationBudgetClaimV55":
        data.setdefault("claimed_at", _utc_now())
        draft = cls(**data)
        payload = draft.model_dump(exclude={"claim_hash"})
        payload["claim_hash"] = draft.content_hash()
        return cls(**payload)


class AuthorizedEncryptedPrivateWorkerReceiptV55(StrictModel):
    """Worker-signed wrapper around one historical V5.3 aggregate receipt."""

    schema_version: Literal["5.5-authorized-encrypted-private-worker"] = (
        "5.5-authorized-encrypted-private-worker"
    )
    protocol_hash: Sha256
    public_launch_binding_hash: Sha256
    public_eligibility_contract_hash: Sha256
    public_eligibility_input_hash: Sha256
    public_eligibility_assessment_hash: Sha256
    public_eligibility_receipt_hash: Sha256
    private_evaluation_authorization_hash: Sha256
    request_hash: Sha256
    legacy_custody_bridge_hash: Sha256
    split_custody_attestation_hash: Sha256
    private_target_envelope_hash: Sha256
    private_target_commitment: Sha256
    budget_ledger_id: Identifier
    budget_claim_hash: Sha256
    budget_claim_file_sha256: Sha256
    v53_worker_receipt_hash: Sha256
    worker_id: Identifier
    worker_host_id: Identifier
    worker_key_id: Identifier
    worker_public_key_fingerprint: Sha256
    public_authorization_verified_before_target_access: Literal[True] = True
    custody_verified_before_target_access: Literal[True] = True
    budget_claimed_before_target_access: Literal[True] = True
    private_target_envelope_accessed: Literal[True] = True
    private_target_key_accessed: Literal[True] = True
    private_evaluation_count: Literal[1] = 1
    private_values_disclosed: Literal[False] = False
    per_target_feedback_disclosed: Literal[False] = False
    secrecy_canary_disclosed: Literal[False] = False
    fixture_only: bool
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    evaluated_at: datetime
    signature_base64: Annotated[str, Field(min_length=40)] | None = None
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "AuthorizedEncryptedPrivateWorkerReceiptV55":
        if self.evaluated_at.utcoffset() is None:
            raise ValueError("private evaluation time must be timezone-aware")
        if self.receipt_hash and (
            not self.signature_base64 or self.receipt_hash != self.content_hash()
        ):
            raise ValueError("authorized private worker receipt envelope differs")
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
            raise ValueError("authorized private worker receipt is not sealed")


class AuthorizedEncryptedPrivateOutputV55(StrictModel):
    """Single create-once output containing the claim and both worker receipts."""

    schema_version: Literal["5.5-authorized-encrypted-private-output"] = (
        "5.5-authorized-encrypted-private-output"
    )
    budget_claim: PrivateEvaluationBudgetClaimV55
    worker_receipt_v53: ExternalPrivateWorkerReceiptV53
    worker_receipt_v55: AuthorizedEncryptedPrivateWorkerReceiptV55
    output_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_output(self) -> "AuthorizedEncryptedPrivateOutputV55":
        self.budget_claim.assert_sealed()
        self.worker_receipt_v53.assert_sealed()
        self.worker_receipt_v55.assert_sealed()
        if (
            self.worker_receipt_v55.budget_claim_hash
            != self.budget_claim.claim_hash
            or self.worker_receipt_v55.v53_worker_receipt_hash
            != self.worker_receipt_v53.receipt_hash
        ):
            raise ValueError("authorized private output bindings differ")
        if self.output_hash and self.output_hash != self.content_hash():
            raise ValueError("authorized private output hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "output_hash")

    def assert_sealed(self) -> None:
        if not self.output_hash or self.output_hash != self.content_hash():
            raise ValueError("authorized private output is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "AuthorizedEncryptedPrivateOutputV55":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"output_hash"})
        payload["output_hash"] = draft.content_hash()
        return cls(**payload)


def create_v53_custody_compatibility_v55(
    *,
    protocol: ProspectiveCampaignProtocolV55,
    score_contract: PrivateScoreContractV53,
    capsule: PrivateCaseCapsuleV50,
    private_target_envelope: EncryptedCustodyEnvelopeV55,
    split_custody_attestation: SplitCustodyAttestationV55,
    custody_private_key_pem: bytes,
    v53_attestation_id: str,
    external_anchor_receipt_hash: str,
    attested_at: datetime | None = None,
) -> tuple[ExternalCustodyAttestationV53, LegacyCustodyBridgeV55]:
    """Create V5.3 compatibility artifacts for the exact V5.5 ciphertext."""

    protocol.assert_sealed()
    score_contract.assert_sealed()
    capsule.assert_sealed()
    private_target_envelope.assert_sealed()
    split_custody_attestation.assert_sealed()
    key = _load_private_key(custody_private_key_pem)
    public_key_pem = _public_key_pem(key)
    if not verify_split_custody_attestation_signature_v55(
        attestation=split_custody_attestation,
        custody_public_key_pem=public_key_pem,
    ):
        raise ValueError("split custody attestation is not authenticated")
    if not (
        protocol.protocol_hash
        == score_contract.protocol_hash
        == split_custody_attestation.protocol_hash
        and score_contract.contract_hash
        == split_custody_attestation.score_contract_hash
        and score_contract.case_id
        == capsule.case_id
        == private_target_envelope.case_id
        == split_custody_attestation.case_id
        and capsule.capsule_hash
        == private_target_envelope.plaintext_commitment
        == split_custody_attestation.private_target_commitment
        and private_target_envelope.envelope_hash
        == split_custody_attestation.private_target_envelope_hash
    ):
        raise ValueError("V5.3 compatibility inputs differ")
    capsule_bytes = (canonical_json(capsule) + "\n").encode("utf-8")
    v53_attestation = sign_external_custody_attestation_v53(
        private_key_pem=custody_private_key_pem,
        attestation_id=v53_attestation_id,
        score_contract_hash=score_contract.contract_hash,
        protocol_hash=protocol.protocol_hash,
        public_case_hash=score_contract.public_case_hash,
        forecast_plan_hash=score_contract.forecast_plan_hash,
        capsule_commitment=capsule.capsule_hash,
        capsule_bytes_hash=hashlib.sha256(capsule_bytes).hexdigest(),
        private_source_manifest_hash=(
            split_custody_attestation.source_provenance_commitment
        ),
        target_id_commitment=sha256_value(score_contract.target_ids),
        custodian_host_id=split_custody_attestation.custodian_host_id,
        coordinator_host_id=split_custody_attestation.coordinator_host_id,
        generator_host_id=split_custody_attestation.generator_host_id,
        external_anchor_receipt_hash=external_anchor_receipt_hash,
        attested_at=attested_at or _utc_now(),
        attester_key_id=split_custody_attestation.custody_key_id,
    )
    bridge = LegacyCustodyBridgeV55.seal(
        case_id=score_contract.case_id,
        protocol_hash=protocol.protocol_hash,
        score_contract_hash=score_contract.contract_hash,
        forecast_plan_hash=score_contract.forecast_plan_hash,
        v53_custody_attestation_hash=v53_attestation.attestation_hash,
        split_custody_attestation_hash=(
            split_custody_attestation.attestation_hash
        ),
        private_capsule_commitment=capsule.capsule_hash,
        capsule_bytes_hash=v53_attestation.capsule_bytes_hash,
        private_target_envelope_hash=private_target_envelope.envelope_hash,
        source_provenance_commitment=(
            split_custody_attestation.source_provenance_commitment
        ),
        custody_key_id=split_custody_attestation.custody_key_id,
        custodian_host_id=split_custody_attestation.custodian_host_id,
        coordinator_host_id=split_custody_attestation.coordinator_host_id,
        generator_host_id=split_custody_attestation.generator_host_id,
    )
    if not verify_legacy_custody_bridge_v55(
        score_contract=score_contract,
        v53_custody_attestation=v53_attestation,
        private_target_envelope=private_target_envelope,
        split_custody_attestation=split_custody_attestation,
        bridge=bridge,
        custody_public_key_pem=public_key_pem,
    ):
        raise ValueError("V5.3 compatibility bridge did not verify")
    return v53_attestation, bridge


def verify_legacy_custody_bridge_v55(
    *,
    score_contract: PrivateScoreContractV53,
    v53_custody_attestation: ExternalCustodyAttestationV53,
    private_target_envelope: EncryptedCustodyEnvelopeV55,
    split_custody_attestation: SplitCustodyAttestationV55,
    bridge: LegacyCustodyBridgeV55,
    custody_public_key_pem: bytes,
) -> bool:
    """Verify both custody signatures and their exact common commitment."""

    try:
        score_contract.assert_sealed()
        v53_custody_attestation.assert_sealed()
        private_target_envelope.assert_sealed()
        split_custody_attestation.assert_sealed()
        bridge.assert_sealed()
        v53_signature_valid = _verify_signature(
            public_key_pem=custody_public_key_pem,
            signature_base64=v53_custody_attestation.signature_base64,
            payload=v53_custody_attestation.unsigned_bytes(),
        )
        return bool(
            v53_signature_valid
            and verify_split_custody_attestation_signature_v55(
                attestation=split_custody_attestation,
                custody_public_key_pem=custody_public_key_pem,
            )
            and bridge.case_id
            == score_contract.case_id
            == split_custody_attestation.case_id
            == private_target_envelope.case_id
            and bridge.protocol_hash
            == score_contract.protocol_hash
            == v53_custody_attestation.protocol_hash
            == split_custody_attestation.protocol_hash
            and bridge.score_contract_hash
            == score_contract.contract_hash
            == v53_custody_attestation.score_contract_hash
            == split_custody_attestation.score_contract_hash
            and bridge.forecast_plan_hash
            == score_contract.forecast_plan_hash
            == v53_custody_attestation.forecast_plan_hash
            and bridge.v53_custody_attestation_hash
            == v53_custody_attestation.attestation_hash
            and bridge.split_custody_attestation_hash
            == split_custody_attestation.attestation_hash
            and bridge.private_capsule_commitment
            == v53_custody_attestation.capsule_commitment
            == split_custody_attestation.private_target_commitment
            == private_target_envelope.plaintext_commitment
            and bridge.capsule_bytes_hash
            == v53_custody_attestation.capsule_bytes_hash
            and bridge.private_target_envelope_hash
            == split_custody_attestation.private_target_envelope_hash
            == private_target_envelope.envelope_hash
            and bridge.source_provenance_commitment
            == v53_custody_attestation.private_source_manifest_hash
            == split_custody_attestation.source_provenance_commitment
            and bridge.custody_key_id
            == v53_custody_attestation.attester_key_id
            == split_custody_attestation.custody_key_id
            and bridge.custodian_host_id
            == v53_custody_attestation.custodian_host_id
            == split_custody_attestation.custodian_host_id
            and bridge.coordinator_host_id
            == v53_custody_attestation.coordinator_host_id
            == split_custody_attestation.coordinator_host_id
            and bridge.generator_host_id
            == v53_custody_attestation.generator_host_id
            == split_custody_attestation.generator_host_id
        )
    except (TypeError, ValueError):
        return False


def assert_authorized_encrypted_private_preconditions_v55(
    *,
    protocol: ProspectiveCampaignProtocolV55,
    candidate_policy: CandidateSelectionPolicyV55,
    public_launch_binding: PublicLaunchBindingV55,
    eligibility_contract: PublicEligibilityContractV54,
    eligibility_input: PublicEligibilityInputV54,
    eligibility_assessment: PublicEligibilityAssessmentV54,
    eligibility_receipt: PublicEligibilityReceiptV54,
    private_authorization: PrivateEvaluationAuthorizationV54,
    eligibility_authority_public_key_pem: bytes,
    request: PrivateEvaluationRequestV53,
    score_contract: PrivateScoreContractV53,
    prediction: PredictionDocumentV50,
    prediction_bytes_hash: str,
    v53_custody_attestation: ExternalCustodyAttestationV53,
    private_target_envelope: EncryptedCustodyEnvelopeV55,
    source_provenance_envelope: EncryptedCustodyEnvelopeV55,
    split_custody_attestation: SplitCustodyAttestationV55,
    legacy_custody_bridge: LegacyCustodyBridgeV55,
    custody_public_key_pem: bytes,
    expected_coordinator_host_id: str,
    expected_generator_host_id: str,
) -> None:
    """Verify every public condition needed before private-target key access."""

    if not verify_public_launch_binding_v55(
        protocol=protocol,
        policy=candidate_policy,
        contract=eligibility_contract,
        binding=public_launch_binding,
    ):
        raise ValueError("public launch binding is not verified")
    recomputed = assess_public_eligibility_v54(
        contract=eligibility_contract,
        evidence=eligibility_input,
    )
    if recomputed.assessment_hash != eligibility_assessment.assessment_hash:
        raise ValueError("public eligibility assessment is not a deterministic replay")
    if not verify_private_evaluation_authorization_v54(
        authorization=private_authorization,
        request=request,
        contract=eligibility_contract,
        evidence=eligibility_input,
        assessment=eligibility_assessment,
        receipt=eligibility_receipt,
        authority_public_key_pem=eligibility_authority_public_key_pem,
    ):
        raise ValueError("private evaluation is not publicly authorized")
    request.assert_sealed()
    score_contract.assert_sealed()
    if not (
        request.case_id == score_contract.case_id == eligibility_contract.task_id
        and request.score_contract_hash == score_contract.contract_hash
        and request.forecast_plan_hash == score_contract.forecast_plan_hash
        and score_contract.protocol_hash == protocol.protocol_hash
        and request.custody_attestation_hash
        == v53_custody_attestation.attestation_hash
        and request.private_capsule_commitment
        == v53_custody_attestation.capsule_commitment
        == split_custody_attestation.private_target_commitment
        == private_target_envelope.plaintext_commitment
    ):
        raise ValueError("private request custody or campaign binding differs")
    if prediction_bytes_hash != request.prediction_snapshot_hash:
        raise ValueError("prediction snapshot hash differs before target access")
    if sha256_value(prediction) != request.prediction_semantic_hash:
        raise ValueError("prediction semantic hash differs before target access")
    if not verify_split_custody_bindings_v55(
        protocol=protocol,
        score_contract=score_contract,
        private_target_envelope=private_target_envelope,
        source_provenance_envelope=source_provenance_envelope,
        attestation=split_custody_attestation,
        custody_public_key_pem=custody_public_key_pem,
        expected_coordinator_host_id=expected_coordinator_host_id,
        expected_generator_host_id=expected_generator_host_id,
    ):
        raise ValueError("split custody bindings are not verified")
    if not verify_legacy_custody_bridge_v55(
        score_contract=score_contract,
        v53_custody_attestation=v53_custody_attestation,
        private_target_envelope=private_target_envelope,
        split_custody_attestation=split_custody_attestation,
        bridge=legacy_custody_bridge,
        custody_public_key_pem=custody_public_key_pem,
    ):
        raise ValueError("V5.3/V5.5 custody bridge is not verified")


def claim_private_evaluation_budget_v55(
    *,
    ledger_root: Path,
    budget_ledger_id: str,
    request: PrivateEvaluationRequestV53,
    private_authorization: PrivateEvaluationAuthorizationV54,
    private_target_envelope: EncryptedCustodyEnvelopeV55,
    split_custody_attestation: SplitCustodyAttestationV55,
    fixture_only: bool,
    claimed_at: datetime | None = None,
) -> tuple[PrivateEvaluationBudgetClaimV55, Path]:
    """Atomically consume the request's only private-evaluation budget."""

    request.assert_sealed()
    private_authorization.assert_sealed()
    private_target_envelope.assert_sealed()
    split_custody_attestation.assert_sealed()
    if request.maximum_private_evaluations != 1:
        raise ValueError("V5.5 supports exactly one private evaluation")
    claim = PrivateEvaluationBudgetClaimV55.seal(
        request_hash=request.request_hash,
        budget_ledger_id=budget_ledger_id,
        authorization_hash=private_authorization.authorization_hash,
        private_target_envelope_hash=private_target_envelope.envelope_hash,
        split_custody_attestation_hash=split_custody_attestation.attestation_hash,
        fixture_only=fixture_only,
        claimed_at=claimed_at or _utc_now(),
    )
    root = ledger_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    claim_path = (root / f"{request.request_hash}.json").resolve()
    if claim_path.parent != root:
        raise ValueError("private evaluation claim path escapes ledger root")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(claim_path, flags, 0o600)
    except FileExistsError as exc:
        raise ValueError("private evaluation budget was already claimed") from exc
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            handle.write(canonical_json(claim) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # A partial claim intentionally remains and still consumes the budget.
        raise
    return claim, claim_path


def assert_durable_private_evaluation_budget_claim_v55(
    *,
    claim: PrivateEvaluationBudgetClaimV55,
    claim_path: Path,
) -> str:
    """Require the exact create-once claim file before target decryption."""

    claim.assert_sealed()
    resolved = claim_path.resolve()
    if resolved.name != f"{claim.request_hash}.json":
        raise ValueError("private evaluation budget claim filename differs")
    expected = (canonical_json(claim) + "\n").encode("utf-8")
    try:
        observed = resolved.read_bytes()
    except OSError as exc:
        raise ValueError("private evaluation budget claim file is unavailable") from exc
    if observed != expected:
        raise ValueError("private evaluation budget claim file differs")
    return hashlib.sha256(observed).hexdigest()


def verify_private_evaluation_budget_claim_v55(
    *,
    claim: PrivateEvaluationBudgetClaimV55,
    request: PrivateEvaluationRequestV53,
    private_authorization: PrivateEvaluationAuthorizationV54,
    private_target_envelope: EncryptedCustodyEnvelopeV55,
    split_custody_attestation: SplitCustodyAttestationV55,
) -> bool:
    try:
        claim.assert_sealed()
        return bool(
            claim.request_hash == request.request_hash
            and claim.authorization_hash == private_authorization.authorization_hash
            and claim.private_target_envelope_hash
            == private_target_envelope.envelope_hash
            and claim.split_custody_attestation_hash
            == split_custody_attestation.attestation_hash
        )
    except (TypeError, ValueError):
        return False


def evaluate_authorized_encrypted_private_inputs_v55(
    *,
    protocol: ProspectiveCampaignProtocolV55,
    candidate_policy: CandidateSelectionPolicyV55,
    public_launch_binding: PublicLaunchBindingV55,
    eligibility_contract: PublicEligibilityContractV54,
    eligibility_input: PublicEligibilityInputV54,
    eligibility_assessment: PublicEligibilityAssessmentV54,
    eligibility_receipt: PublicEligibilityReceiptV54,
    private_authorization: PrivateEvaluationAuthorizationV54,
    eligibility_authority_public_key_pem: bytes,
    request: PrivateEvaluationRequestV53,
    score_contract: PrivateScoreContractV53,
    prediction: PredictionDocumentV50,
    prediction_bytes_hash: str,
    v53_custody_attestation: ExternalCustodyAttestationV53,
    private_target_envelope: EncryptedCustodyEnvelopeV55,
    private_target_key_path: Path,
    source_provenance_envelope: EncryptedCustodyEnvelopeV55,
    split_custody_attestation: SplitCustodyAttestationV55,
    legacy_custody_bridge: LegacyCustodyBridgeV55,
    custody_public_key_pem: bytes,
    budget_claim: PrivateEvaluationBudgetClaimV55,
    budget_claim_path: Path,
    expected_coordinator_host_id: str,
    expected_generator_host_id: str,
    worker_id: str,
    worker_host_id: str,
    worker_executable_hash: str,
    runner_source_hash: str,
    worker_key_id: str,
    worker_private_key_pem: bytes,
    fixture_only: bool,
    evaluated_at: datetime | None = None,
) -> AuthorizedEncryptedPrivateOutputV55:
    """Decrypt and score only after the full public chain and budget verify."""

    assert_authorized_encrypted_private_preconditions_v55(
        protocol=protocol,
        candidate_policy=candidate_policy,
        public_launch_binding=public_launch_binding,
        eligibility_contract=eligibility_contract,
        eligibility_input=eligibility_input,
        eligibility_assessment=eligibility_assessment,
        eligibility_receipt=eligibility_receipt,
        private_authorization=private_authorization,
        eligibility_authority_public_key_pem=(
            eligibility_authority_public_key_pem
        ),
        request=request,
        score_contract=score_contract,
        prediction=prediction,
        prediction_bytes_hash=prediction_bytes_hash,
        v53_custody_attestation=v53_custody_attestation,
        private_target_envelope=private_target_envelope,
        source_provenance_envelope=source_provenance_envelope,
        split_custody_attestation=split_custody_attestation,
        legacy_custody_bridge=legacy_custody_bridge,
        custody_public_key_pem=custody_public_key_pem,
        expected_coordinator_host_id=expected_coordinator_host_id,
        expected_generator_host_id=expected_generator_host_id,
    )
    if not verify_private_evaluation_budget_claim_v55(
        claim=budget_claim,
        request=request,
        private_authorization=private_authorization,
        private_target_envelope=private_target_envelope,
        split_custody_attestation=split_custody_attestation,
    ):
        raise ValueError("private evaluation budget claim is not verified")
    if budget_claim.fixture_only != fixture_only:
        raise ValueError("fixture-only status differs from budget claim")
    budget_claim_file_sha256 = (
        assert_durable_private_evaluation_budget_claim_v55(
            claim=budget_claim,
            claim_path=budget_claim_path,
        )
    )
    try:
        private_target_key = private_target_key_path.resolve().read_bytes()
    except OSError as exc:
        raise ValueError("private target key file is unavailable") from exc
    capsule = open_private_target_envelope_v55(
        private_target_envelope=private_target_envelope,
        private_target_key=private_target_key,
    )
    actual_evaluated_at = evaluated_at or _utc_now()
    if actual_evaluated_at < budget_claim.claimed_at:
        raise ValueError("private evaluation predates its budget claim")
    worker_receipt_v53 = evaluate_external_private_inputs_v53(
        request=request,
        score_contract=score_contract,
        prediction=prediction,
        prediction_bytes_hash=prediction_bytes_hash,
        capsule=capsule,
        worker_id=worker_id,
        worker_host_id=worker_host_id,
        worker_executable_hash=worker_executable_hash,
        runner_source_hash=runner_source_hash,
        worker_key_id=worker_key_id,
        worker_private_key_pem=worker_private_key_pem,
        evaluated_at=actual_evaluated_at,
    )
    worker_key = _load_private_key(worker_private_key_pem)
    worker_public_pem = _public_key_pem(worker_key)
    unsigned = AuthorizedEncryptedPrivateWorkerReceiptV55(
        protocol_hash=protocol.protocol_hash,
        public_launch_binding_hash=public_launch_binding.binding_hash,
        public_eligibility_contract_hash=eligibility_contract.contract_hash,
        public_eligibility_input_hash=eligibility_input.input_hash,
        public_eligibility_assessment_hash=eligibility_assessment.assessment_hash,
        public_eligibility_receipt_hash=eligibility_receipt.receipt_hash,
        private_evaluation_authorization_hash=(
            private_authorization.authorization_hash
        ),
        request_hash=request.request_hash,
        legacy_custody_bridge_hash=legacy_custody_bridge.bridge_hash,
        split_custody_attestation_hash=(
            split_custody_attestation.attestation_hash
        ),
        private_target_envelope_hash=private_target_envelope.envelope_hash,
        private_target_commitment=private_target_envelope.plaintext_commitment,
        budget_ledger_id=budget_claim.budget_ledger_id,
        budget_claim_hash=budget_claim.claim_hash,
        budget_claim_file_sha256=budget_claim_file_sha256,
        v53_worker_receipt_hash=worker_receipt_v53.receipt_hash,
        worker_id=worker_id,
        worker_host_id=worker_host_id,
        worker_key_id=worker_key_id,
        worker_public_key_fingerprint=signing_key_fingerprint_v55(
            worker_public_pem
        ),
        fixture_only=fixture_only,
        evaluated_at=actual_evaluated_at,
    )
    payload = unsigned.model_dump(mode="json")
    payload["signature_base64"] = base64.b64encode(
        worker_key.sign(unsigned.unsigned_bytes())
    ).decode("ascii")
    tagged = AuthorizedEncryptedPrivateWorkerReceiptV55(**payload)
    final_payload = tagged.model_dump(mode="json")
    final_payload["receipt_hash"] = tagged.content_hash()
    worker_receipt_v55 = AuthorizedEncryptedPrivateWorkerReceiptV55(
        **final_payload
    )
    return AuthorizedEncryptedPrivateOutputV55.seal(
        budget_claim=budget_claim,
        worker_receipt_v53=worker_receipt_v53,
        worker_receipt_v55=worker_receipt_v55,
    )


def verify_authorized_encrypted_private_output_v55(
    *,
    output: AuthorizedEncryptedPrivateOutputV55,
    protocol: ProspectiveCampaignProtocolV55,
    candidate_policy: CandidateSelectionPolicyV55,
    public_launch_binding: PublicLaunchBindingV55,
    eligibility_contract: PublicEligibilityContractV54,
    eligibility_input: PublicEligibilityInputV54,
    eligibility_assessment: PublicEligibilityAssessmentV54,
    eligibility_receipt: PublicEligibilityReceiptV54,
    private_authorization: PrivateEvaluationAuthorizationV54,
    eligibility_authority_public_key_pem: bytes,
    request: PrivateEvaluationRequestV53,
    score_contract: PrivateScoreContractV53,
    prediction: PredictionDocumentV50,
    prediction_bytes_hash: str,
    v53_custody_attestation: ExternalCustodyAttestationV53,
    private_target_envelope: EncryptedCustodyEnvelopeV55,
    source_provenance_envelope: EncryptedCustodyEnvelopeV55,
    split_custody_attestation: SplitCustodyAttestationV55,
    legacy_custody_bridge: LegacyCustodyBridgeV55,
    custody_public_key_pem: bytes,
    trusted_worker_public_keys: Mapping[str, bytes],
    expected_coordinator_host_id: str,
    expected_generator_host_id: str,
) -> bool:
    """Publicly verify the additive receipt without a private-target key."""

    try:
        output.assert_sealed()
        assert_authorized_encrypted_private_preconditions_v55(
            protocol=protocol,
            candidate_policy=candidate_policy,
            public_launch_binding=public_launch_binding,
            eligibility_contract=eligibility_contract,
            eligibility_input=eligibility_input,
            eligibility_assessment=eligibility_assessment,
            eligibility_receipt=eligibility_receipt,
            private_authorization=private_authorization,
            eligibility_authority_public_key_pem=(
                eligibility_authority_public_key_pem
            ),
            request=request,
            score_contract=score_contract,
            prediction=prediction,
            prediction_bytes_hash=prediction_bytes_hash,
            v53_custody_attestation=v53_custody_attestation,
            private_target_envelope=private_target_envelope,
            source_provenance_envelope=source_provenance_envelope,
            split_custody_attestation=split_custody_attestation,
            legacy_custody_bridge=legacy_custody_bridge,
            custody_public_key_pem=custody_public_key_pem,
            expected_coordinator_host_id=expected_coordinator_host_id,
            expected_generator_host_id=expected_generator_host_id,
        )
        if not verify_private_evaluation_budget_claim_v55(
            claim=output.budget_claim,
            request=request,
            private_authorization=private_authorization,
            private_target_envelope=private_target_envelope,
            split_custody_attestation=split_custody_attestation,
        ):
            return False
        receipt_v53 = output.worker_receipt_v53
        receipt_v55 = output.worker_receipt_v55
        worker_public_key = trusted_worker_public_keys.get(
            receipt_v55.worker_key_id
        )
        if worker_public_key is None:
            return False
        return bool(
            receipt_v53.worker_key_id == receipt_v55.worker_key_id
            and receipt_v53.worker_id == receipt_v55.worker_id
            and receipt_v53.worker_host_id == receipt_v55.worker_host_id
            and receipt_v55.worker_public_key_fingerprint
            == signing_key_fingerprint_v55(worker_public_key)
            and _verify_signature(
                public_key_pem=worker_public_key,
                signature_base64=receipt_v53.signature_base64,
                payload=receipt_v53.unsigned_bytes(),
            )
            and _verify_signature(
                public_key_pem=worker_public_key,
                signature_base64=receipt_v55.signature_base64,
                payload=receipt_v55.unsigned_bytes(),
            )
            and receipt_v55.protocol_hash == protocol.protocol_hash
            and receipt_v55.public_launch_binding_hash
            == public_launch_binding.binding_hash
            and receipt_v55.public_eligibility_contract_hash
            == eligibility_contract.contract_hash
            and receipt_v55.public_eligibility_input_hash
            == eligibility_input.input_hash
            and receipt_v55.public_eligibility_assessment_hash
            == eligibility_assessment.assessment_hash
            and receipt_v55.public_eligibility_receipt_hash
            == eligibility_receipt.receipt_hash
            and receipt_v55.private_evaluation_authorization_hash
            == private_authorization.authorization_hash
            and receipt_v55.request_hash == request.request_hash
            and receipt_v55.legacy_custody_bridge_hash
            == legacy_custody_bridge.bridge_hash
            and receipt_v55.split_custody_attestation_hash
            == split_custody_attestation.attestation_hash
            and receipt_v55.private_target_envelope_hash
            == private_target_envelope.envelope_hash
            and receipt_v55.private_target_commitment
            == request.private_capsule_commitment
            and receipt_v55.budget_ledger_id
            == output.budget_claim.budget_ledger_id
            and receipt_v55.budget_claim_hash == output.budget_claim.claim_hash
            and receipt_v55.budget_claim_file_sha256
            == hashlib.sha256(
                (canonical_json(output.budget_claim) + "\n").encode("utf-8")
            ).hexdigest()
            and receipt_v55.v53_worker_receipt_hash == receipt_v53.receipt_hash
            and receipt_v55.fixture_only == output.budget_claim.fixture_only
            and receipt_v53.request_hash == request.request_hash
            and receipt_v53.case_id == request.case_id
            and receipt_v53.score_contract_hash == score_contract.contract_hash
            and receipt_v53.custody_attestation_hash
            == v53_custody_attestation.attestation_hash
            and receipt_v53.prediction_snapshot_hash
            == request.prediction_snapshot_hash
            and receipt_v53.prediction_semantic_hash
            == request.prediction_semantic_hash
            and receipt_v53.private_capsule_commitment
            == request.private_capsule_commitment
            and receipt_v55.evaluated_at >= output.budget_claim.claimed_at
        )
    except (TypeError, ValueError):
        return False


__all__ = [
    "AuthorizedEncryptedPrivateOutputV55",
    "AuthorizedEncryptedPrivateWorkerReceiptV55",
    "LegacyCustodyBridgeV55",
    "PrivateEvaluationBudgetClaimV55",
    "assert_durable_private_evaluation_budget_claim_v55",
    "assert_authorized_encrypted_private_preconditions_v55",
    "claim_private_evaluation_budget_v55",
    "create_v53_custody_compatibility_v55",
    "evaluate_authorized_encrypted_private_inputs_v55",
    "verify_authorized_encrypted_private_output_v55",
    "verify_legacy_custody_bridge_v55",
    "verify_private_evaluation_budget_claim_v55",
]
