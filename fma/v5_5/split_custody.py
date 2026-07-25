"""Split encryption and controlled provenance release for V5.5 campaigns.

Private targets and source provenance are encrypted under distinct AES-256-GCM
keys and associated-data domains.  The provenance release API intentionally
has no private-target envelope or key parameter.
"""

from __future__ import annotations

import base64
import hashlib
import math
import os
import secrets
from datetime import datetime, timezone
from typing import Annotated, Literal, Mapping

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import Field, model_validator

from fma.hashing import canonical_json, sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5.external_harness import PrivateCaseCapsuleV50, PrivateTargetV50
from fma.v5_3.custody import PrivateScoreContractV53

from .campaign_protocol import ProspectiveCampaignProtocolV55


TerminalCampaignStatusV55 = Literal[
    "ABSTAIN",
    "PRIVATE_EVALUATION_COMPLETE",
    "TERMINATED",
]


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


def signing_key_fingerprint_v55(public_key_pem: bytes) -> str:
    key = _load_public_key(public_key_pem)
    der = key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return hashlib.sha256(der).hexdigest()


def encryption_key_fingerprint_v55(key: bytes) -> str:
    if len(key) != 32:
        raise ValueError("AES-256-GCM keys must contain exactly 32 bytes")
    return hashlib.sha256(key).hexdigest()


def _custody_associated_data_v55(
    *,
    envelope_id: str,
    case_id: str,
    domain: str,
    key_id: str,
    key_fingerprint: str,
    plaintext_commitment: str,
) -> bytes:
    return canonical_json(
        {
            "schema_version": "5.5-encrypted-custody-envelope",
            "envelope_id": envelope_id,
            "case_id": case_id,
            "domain": domain,
            "key_id": key_id,
            "key_fingerprint": key_fingerprint,
            "plaintext_commitment": plaintext_commitment,
        }
    ).encode("utf-8")


class SourceProvenanceDraftV55(StrictModel):
    """Custodian input that contains source metadata but no private targets."""

    case_id: Identifier
    source_authority: Annotated[str, Field(min_length=3)]
    source_title: Annotated[str, Field(min_length=3)]
    source_locator: Annotated[str, Field(min_length=8)]
    table_or_series_id: Annotated[str, Field(min_length=1)]
    public_period_start: Annotated[str, Field(min_length=1)]
    public_period_end: Annotated[str, Field(min_length=1)]
    private_period_start: Annotated[str, Field(min_length=1)]
    private_period_end: Annotated[str, Field(min_length=1)]
    source_artifact_sha256: Sha256
    prior_campaign_exclusion_hashes: list[Sha256]
    nonreuse_assertion: Literal[True] = True
    retrieved_at: datetime

    @model_validator(mode="after")
    def validate_draft(self) -> "SourceProvenanceDraftV55":
        if self.prior_campaign_exclusion_hashes != sorted(
            set(self.prior_campaign_exclusion_hashes)
        ):
            raise ValueError(
                "prior campaign exclusion hashes must be sorted and unique"
            )
        if self.retrieved_at.utcoffset() is None:
            raise ValueError("retrieved_at must be timezone-aware")
        return self


class SourceProvenanceRecordV55(SourceProvenanceDraftV55):
    """Sealed metadata-only record released after authorized closeout."""

    schema_version: Literal["5.5-source-provenance-record"] = (
        "5.5-source-provenance-record"
    )
    protocol_hash: Sha256
    secrecy_canary: Annotated[str, Field(min_length=32)]
    record_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_record(self) -> "SourceProvenanceRecordV55":
        if self.record_hash and self.record_hash != self.content_hash():
            raise ValueError("source provenance record hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "record_hash")

    def assert_sealed(self) -> None:
        if not self.record_hash or self.record_hash != self.content_hash():
            raise ValueError("source provenance record is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "SourceProvenanceRecordV55":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"record_hash"})
        payload["record_hash"] = draft.content_hash()
        return cls(**payload)


class EncryptedCustodyEnvelopeV55(StrictModel):
    """Public ciphertext and commitment for exactly one custody domain."""

    schema_version: Literal["5.5-encrypted-custody-envelope"] = (
        "5.5-encrypted-custody-envelope"
    )
    envelope_id: Identifier
    case_id: Identifier
    domain: Literal["private_targets", "source_provenance"]
    key_id: Identifier
    key_fingerprint: Sha256
    plaintext_commitment: Sha256
    nonce_base64: Annotated[str, Field(min_length=16)]
    ciphertext_base64: Annotated[str, Field(min_length=24)]
    ciphertext_sha256: Sha256
    associated_data_sha256: Sha256
    envelope_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_envelope(self) -> "EncryptedCustodyEnvelopeV55":
        try:
            nonce = base64.b64decode(self.nonce_base64, validate=True)
            ciphertext = base64.b64decode(self.ciphertext_base64, validate=True)
        except ValueError as exc:
            raise ValueError("custody envelope base64 is invalid") from exc
        if len(nonce) != 12:
            raise ValueError("AES-GCM nonce must contain exactly 12 bytes")
        if len(ciphertext) < 17:
            raise ValueError("AES-GCM ciphertext is too short")
        if hashlib.sha256(ciphertext).hexdigest() != self.ciphertext_sha256:
            raise ValueError("custody ciphertext hash differs")
        if self.envelope_hash and self.envelope_hash != self.content_hash():
            raise ValueError("custody envelope hash differs")
        return self

    def associated_data(self) -> bytes:
        return _custody_associated_data_v55(
            envelope_id=self.envelope_id,
            case_id=self.case_id,
            domain=self.domain,
            key_id=self.key_id,
            key_fingerprint=self.key_fingerprint,
            plaintext_commitment=self.plaintext_commitment,
        )

    def content_hash(self) -> str:
        return _hash_without(self, "envelope_hash")

    def assert_sealed(self) -> None:
        if not self.envelope_hash or self.envelope_hash != self.content_hash():
            raise ValueError("encrypted custody envelope is not sealed")
        if hashlib.sha256(self.associated_data()).hexdigest() != (
            self.associated_data_sha256
        ):
            raise ValueError("custody associated-data hash differs")


class SplitCustodyAttestationV55(StrictModel):
    """Custodian-signed binding for two independently keyed ciphertexts."""

    schema_version: Literal["5.5-split-custody-attestation"] = (
        "5.5-split-custody-attestation"
    )
    attestation_id: Identifier
    case_id: Identifier
    protocol_hash: Sha256
    score_contract_hash: Sha256
    private_target_envelope_hash: Sha256
    private_target_commitment: Sha256
    private_target_key_id: Identifier
    private_target_key_fingerprint: Sha256
    source_provenance_envelope_hash: Sha256
    source_provenance_commitment: Sha256
    source_provenance_key_id: Identifier
    source_provenance_key_fingerprint: Sha256
    distinct_encryption_key_domains: Literal[True] = True
    source_release_policy: Literal["after_signed_campaign_closeout_only"] = (
        "after_signed_campaign_closeout_only"
    )
    envelopes_created_before_generator_release: Literal[True] = True
    private_target_values_disclosed: Literal[False] = False
    source_provenance_disclosed: Literal[False] = False
    custodian_host_id: Identifier
    coordinator_host_id: Identifier
    generator_host_id: Identifier
    attested_at: datetime
    custody_key_id: Identifier
    custody_public_key_fingerprint: Sha256
    signature_base64: Annotated[str, Field(min_length=40)] | None = None
    attestation_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_attestation(self) -> "SplitCustodyAttestationV55":
        if self.private_target_key_id == self.source_provenance_key_id:
            raise ValueError("custody encryption key IDs must differ")
        if (
            self.private_target_key_fingerprint
            == self.source_provenance_key_fingerprint
        ):
            raise ValueError("custody encryption key fingerprints must differ")
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
            raise ValueError("split custody attestation envelope differs")
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
            raise ValueError("split custody attestation is not sealed")


class CampaignCloseoutAuthorizationV55(StrictModel):
    """Independent authority approval for metadata-only disclosure."""

    schema_version: Literal["5.5-campaign-closeout-authorization"] = (
        "5.5-campaign-closeout-authorization"
    )
    authorization_id: Identifier
    case_id: Identifier
    protocol_hash: Sha256
    split_custody_attestation_hash: Sha256
    terminal_status: TerminalCampaignStatusV55
    terminal_evidence_hash: Sha256
    release_source_provenance: Literal[True] = True
    private_target_release_authorized: Literal[False] = False
    authorized_at: datetime
    closeout_authority_key_id: Identifier
    closeout_authority_public_key_fingerprint: Sha256
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    signature_base64: Annotated[str, Field(min_length=40)] | None = None
    authorization_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_authorization(self) -> "CampaignCloseoutAuthorizationV55":
        if self.authorized_at.utcoffset() is None:
            raise ValueError("authorized_at must be timezone-aware")
        if self.authorization_hash and (
            not self.signature_base64 or self.authorization_hash != self.content_hash()
        ):
            raise ValueError("closeout authorization envelope differs")
        return self

    def unsigned_bytes(self) -> bytes:
        return canonical_json(
            self.model_dump(
                mode="json",
                exclude={"signature_base64", "authorization_hash"},
            )
        ).encode("utf-8")

    def content_hash(self) -> str:
        return _hash_without(self, "authorization_hash")

    def assert_sealed(self) -> None:
        if (
            not self.signature_base64
            or not self.authorization_hash
            or self.authorization_hash != self.content_hash()
        ):
            raise ValueError("campaign closeout authorization is not sealed")


class SourceProvenanceDisclosureReceiptV55(StrictModel):
    """Replayable closeout projection; never a private-evaluation receipt."""

    schema_version: Literal["5.5-source-provenance-disclosure-receipt"] = (
        "5.5-source-provenance-disclosure-receipt"
    )
    case_id: Identifier
    protocol_hash: Sha256
    source_record_hash: Sha256
    source_provenance_envelope_hash: Sha256
    split_custody_attestation_hash: Sha256
    closeout_authorization_hash: Sha256
    source_provenance_status: Literal["RELEASED"] = "RELEASED"
    private_target_envelope_accessed: Literal[False] = False
    private_target_key_accessed: Literal[False] = False
    private_evaluation_performed: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    disclosed_at: datetime
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "SourceProvenanceDisclosureReceiptV55":
        if self.disclosed_at.utcoffset() is None:
            raise ValueError("disclosed_at must be timezone-aware")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("source disclosure receipt hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("source provenance disclosure receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "SourceProvenanceDisclosureReceiptV55":
        data.setdefault("disclosed_at", _utc_now())
        draft = cls(**data)
        payload = draft.model_dump(exclude={"receipt_hash"})
        payload["receipt_hash"] = draft.content_hash()
        return cls(**payload)


def _encrypt_model_v55(
    *,
    model: StrictModel,
    plaintext_commitment: str,
    envelope_id: str,
    case_id: str,
    domain: Literal["private_targets", "source_provenance"],
    key_id: str,
    key: bytes,
    nonce: bytes | None,
) -> EncryptedCustodyEnvelopeV55:
    key_fingerprint = encryption_key_fingerprint_v55(key)
    actual_nonce = nonce if nonce is not None else os.urandom(12)
    if len(actual_nonce) != 12:
        raise ValueError("AES-GCM nonce must contain exactly 12 bytes")
    associated_data = _custody_associated_data_v55(
        envelope_id=envelope_id,
        case_id=case_id,
        domain=domain,
        key_id=key_id,
        key_fingerprint=key_fingerprint,
        plaintext_commitment=plaintext_commitment,
    )
    plaintext = (canonical_json(model) + "\n").encode("utf-8")
    ciphertext = AESGCM(key).encrypt(actual_nonce, plaintext, associated_data)
    draft = EncryptedCustodyEnvelopeV55(
        envelope_id=envelope_id,
        case_id=case_id,
        domain=domain,
        key_id=key_id,
        key_fingerprint=key_fingerprint,
        plaintext_commitment=plaintext_commitment,
        nonce_base64=base64.b64encode(actual_nonce).decode("ascii"),
        ciphertext_base64=base64.b64encode(ciphertext).decode("ascii"),
        ciphertext_sha256=hashlib.sha256(ciphertext).hexdigest(),
        associated_data_sha256=hashlib.sha256(associated_data).hexdigest(),
    )
    payload = draft.model_dump(mode="json")
    payload["envelope_hash"] = draft.content_hash()
    return EncryptedCustodyEnvelopeV55(**payload)


def _decrypt_envelope_v55(
    *,
    envelope: EncryptedCustodyEnvelopeV55,
    key: bytes,
) -> bytes:
    envelope.assert_sealed()
    if encryption_key_fingerprint_v55(key) != envelope.key_fingerprint:
        raise ValueError("custody decryption key fingerprint differs")
    try:
        return AESGCM(key).decrypt(
            base64.b64decode(envelope.nonce_base64, validate=True),
            base64.b64decode(envelope.ciphertext_base64, validate=True),
            envelope.associated_data(),
        )
    except (InvalidTag, ValueError) as exc:
        raise ValueError("custody envelope authentication failed") from exc


def _sign_attestation_v55(
    *,
    private_key_pem: bytes,
    **data: object,
) -> SplitCustodyAttestationV55:
    key = _load_private_key(private_key_pem)
    public_key_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    data["custody_public_key_fingerprint"] = signing_key_fingerprint_v55(public_key_pem)
    unsigned = SplitCustodyAttestationV55(**data)
    payload = unsigned.model_dump(mode="json")
    payload["signature_base64"] = base64.b64encode(
        key.sign(unsigned.unsigned_bytes())
    ).decode("ascii")
    signed = SplitCustodyAttestationV55(**payload)
    final_payload = signed.model_dump(mode="json")
    final_payload["attestation_hash"] = signed.content_hash()
    return SplitCustodyAttestationV55(**final_payload)


def verify_split_custody_attestation_signature_v55(
    *,
    attestation: SplitCustodyAttestationV55,
    custody_public_key_pem: bytes,
) -> bool:
    try:
        attestation.assert_sealed()
        if attestation.custody_public_key_fingerprint != (
            signing_key_fingerprint_v55(custody_public_key_pem)
        ):
            return False
        _load_public_key(custody_public_key_pem).verify(
            base64.b64decode(attestation.signature_base64, validate=True),
            attestation.unsigned_bytes(),
        )
        return True
    except (InvalidSignature, TypeError, ValueError):
        return False


def create_split_custody_envelopes_v55(
    *,
    protocol: ProspectiveCampaignProtocolV55,
    score_contract: PrivateScoreContractV53,
    private_targets: list[PrivateTargetV50],
    source_provenance: SourceProvenanceDraftV55,
    private_target_envelope_id: str,
    source_provenance_envelope_id: str,
    private_target_key_id: str,
    private_target_key: bytes,
    source_provenance_key_id: str,
    source_provenance_key: bytes,
    custodian_host_id: str,
    coordinator_host_id: str,
    generator_host_id: str,
    attestation_id: str,
    custody_key_id: str,
    custody_private_key_pem: bytes,
    private_target_nonce: bytes | None = None,
    source_provenance_nonce: bytes | None = None,
    private_target_canary: str | None = None,
    source_provenance_canary: str | None = None,
    attested_at: datetime | None = None,
) -> tuple[
    PrivateCaseCapsuleV50,
    SourceProvenanceRecordV55,
    EncryptedCustodyEnvelopeV55,
    EncryptedCustodyEnvelopeV55,
    SplitCustodyAttestationV55,
]:
    """Create separately decryptable target and provenance ciphertexts."""

    protocol.assert_sealed()
    score_contract.assert_sealed()
    if score_contract.protocol_hash != protocol.protocol_hash:
        raise ValueError("score contract is bound to another prospective protocol")
    if source_provenance.case_id != score_contract.case_id:
        raise ValueError("source provenance belongs to another case")
    if [item.target_id for item in private_targets] != score_contract.target_ids:
        raise ValueError("private targets differ from the frozen score contract")
    if any(not math.isfinite(item.value) for item in private_targets):
        raise ValueError("private target values must be finite")
    if private_target_key_id == source_provenance_key_id:
        raise ValueError("private target and provenance key IDs must differ")
    if private_target_key == source_provenance_key:
        raise ValueError("private target and provenance encryption keys must differ")
    if private_target_envelope_id == source_provenance_envelope_id:
        raise ValueError("private target and provenance envelope IDs must differ")

    capsule = PrivateCaseCapsuleV50.seal(
        case_id=score_contract.case_id,
        public_case_hash=score_contract.public_case_hash,
        holdout=private_targets,
        quality_scale=score_contract.quality_scale,
        secrecy_canary=private_target_canary
        or "fma-v55-target-" + secrets.token_hex(32),
    )
    provenance = SourceProvenanceRecordV55.seal(
        **source_provenance.model_dump(mode="json"),
        protocol_hash=protocol.protocol_hash,
        secrecy_canary=source_provenance_canary
        or "fma-v55-provenance-" + secrets.token_hex(32),
    )
    target_envelope = _encrypt_model_v55(
        model=capsule,
        plaintext_commitment=capsule.capsule_hash,
        envelope_id=private_target_envelope_id,
        case_id=score_contract.case_id,
        domain="private_targets",
        key_id=private_target_key_id,
        key=private_target_key,
        nonce=private_target_nonce,
    )
    provenance_envelope = _encrypt_model_v55(
        model=provenance,
        plaintext_commitment=provenance.record_hash,
        envelope_id=source_provenance_envelope_id,
        case_id=score_contract.case_id,
        domain="source_provenance",
        key_id=source_provenance_key_id,
        key=source_provenance_key,
        nonce=source_provenance_nonce,
    )
    attestation = _sign_attestation_v55(
        private_key_pem=custody_private_key_pem,
        attestation_id=attestation_id,
        case_id=score_contract.case_id,
        protocol_hash=protocol.protocol_hash,
        score_contract_hash=score_contract.contract_hash,
        private_target_envelope_hash=target_envelope.envelope_hash,
        private_target_commitment=capsule.capsule_hash,
        private_target_key_id=private_target_key_id,
        private_target_key_fingerprint=target_envelope.key_fingerprint,
        source_provenance_envelope_hash=provenance_envelope.envelope_hash,
        source_provenance_commitment=provenance.record_hash,
        source_provenance_key_id=source_provenance_key_id,
        source_provenance_key_fingerprint=provenance_envelope.key_fingerprint,
        custodian_host_id=custodian_host_id,
        coordinator_host_id=coordinator_host_id,
        generator_host_id=generator_host_id,
        attested_at=attested_at or _utc_now(),
        custody_key_id=custody_key_id,
    )
    return (
        capsule,
        provenance,
        target_envelope,
        provenance_envelope,
        attestation,
    )


def verify_split_custody_bindings_v55(
    *,
    protocol: ProspectiveCampaignProtocolV55,
    score_contract: PrivateScoreContractV53,
    private_target_envelope: EncryptedCustodyEnvelopeV55,
    source_provenance_envelope: EncryptedCustodyEnvelopeV55,
    attestation: SplitCustodyAttestationV55,
    custody_public_key_pem: bytes,
    expected_coordinator_host_id: str,
    expected_generator_host_id: str,
) -> bool:
    try:
        protocol.assert_sealed()
        score_contract.assert_sealed()
        private_target_envelope.assert_sealed()
        source_provenance_envelope.assert_sealed()
        return bool(
            verify_split_custody_attestation_signature_v55(
                attestation=attestation,
                custody_public_key_pem=custody_public_key_pem,
            )
            and score_contract.protocol_hash
            == protocol.protocol_hash
            == attestation.protocol_hash
            and score_contract.case_id
            == private_target_envelope.case_id
            == source_provenance_envelope.case_id
            == attestation.case_id
            and score_contract.contract_hash == attestation.score_contract_hash
            and private_target_envelope.domain == "private_targets"
            and source_provenance_envelope.domain == "source_provenance"
            and private_target_envelope.envelope_hash
            == attestation.private_target_envelope_hash
            and source_provenance_envelope.envelope_hash
            == attestation.source_provenance_envelope_hash
            and private_target_envelope.plaintext_commitment
            == attestation.private_target_commitment
            and source_provenance_envelope.plaintext_commitment
            == attestation.source_provenance_commitment
            and private_target_envelope.key_id == attestation.private_target_key_id
            and source_provenance_envelope.key_id
            == attestation.source_provenance_key_id
            and private_target_envelope.key_fingerprint
            == attestation.private_target_key_fingerprint
            and source_provenance_envelope.key_fingerprint
            == attestation.source_provenance_key_fingerprint
            and private_target_envelope.key_fingerprint
            != source_provenance_envelope.key_fingerprint
            and attestation.coordinator_host_id == expected_coordinator_host_id
            and attestation.generator_host_id == expected_generator_host_id
            and attestation.custodian_host_id
            not in {expected_coordinator_host_id, expected_generator_host_id}
        )
    except (TypeError, ValueError):
        return False


def sign_campaign_closeout_authorization_v55(
    *,
    protocol: ProspectiveCampaignProtocolV55,
    attestation: SplitCustodyAttestationV55,
    terminal_status: TerminalCampaignStatusV55,
    terminal_evidence_hash: str,
    authorization_id: str,
    closeout_authority_key_id: str,
    closeout_authority_private_key_pem: bytes,
    authorized_at: datetime | None = None,
) -> CampaignCloseoutAuthorizationV55:
    """External closeout authority signs metadata release without custody keys."""

    protocol.assert_sealed()
    attestation.assert_sealed()
    if attestation.protocol_hash != protocol.protocol_hash:
        raise ValueError("closeout attestation is bound to another protocol")
    key = _load_private_key(closeout_authority_private_key_pem)
    public_key_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    fingerprint = signing_key_fingerprint_v55(public_key_pem)
    if fingerprint == attestation.custody_public_key_fingerprint:
        raise ValueError("closeout authority key must differ from custody signing key")
    unsigned = CampaignCloseoutAuthorizationV55(
        authorization_id=authorization_id,
        case_id=attestation.case_id,
        protocol_hash=protocol.protocol_hash,
        split_custody_attestation_hash=attestation.attestation_hash,
        terminal_status=terminal_status,
        terminal_evidence_hash=terminal_evidence_hash,
        authorized_at=authorized_at or _utc_now(),
        closeout_authority_key_id=closeout_authority_key_id,
        closeout_authority_public_key_fingerprint=fingerprint,
    )
    payload = unsigned.model_dump(mode="json")
    payload["signature_base64"] = base64.b64encode(
        key.sign(unsigned.unsigned_bytes())
    ).decode("ascii")
    signed = CampaignCloseoutAuthorizationV55(**payload)
    final_payload = signed.model_dump(mode="json")
    final_payload["authorization_hash"] = signed.content_hash()
    return CampaignCloseoutAuthorizationV55(**final_payload)


def verify_campaign_closeout_authorization_v55(
    *,
    authorization: CampaignCloseoutAuthorizationV55,
    protocol: ProspectiveCampaignProtocolV55,
    attestation: SplitCustodyAttestationV55,
    closeout_public_keys: Mapping[str, bytes],
) -> bool:
    try:
        protocol.assert_sealed()
        attestation.assert_sealed()
        authorization.assert_sealed()
        public_key_pem = closeout_public_keys.get(
            authorization.closeout_authority_key_id
        )
        if public_key_pem is None:
            return False
        if signing_key_fingerprint_v55(public_key_pem) != (
            authorization.closeout_authority_public_key_fingerprint
        ):
            return False
        _load_public_key(public_key_pem).verify(
            base64.b64decode(authorization.signature_base64, validate=True),
            authorization.unsigned_bytes(),
        )
        return bool(
            authorization.case_id == attestation.case_id
            and authorization.protocol_hash
            == protocol.protocol_hash
            == attestation.protocol_hash
            and authorization.split_custody_attestation_hash
            == attestation.attestation_hash
            and authorization.closeout_authority_public_key_fingerprint
            != attestation.custody_public_key_fingerprint
        )
    except (InvalidSignature, TypeError, ValueError):
        return False


def release_source_provenance_v55(
    *,
    protocol: ProspectiveCampaignProtocolV55,
    source_provenance_envelope: EncryptedCustodyEnvelopeV55,
    attestation: SplitCustodyAttestationV55,
    authorization: CampaignCloseoutAuthorizationV55,
    terminal_evidence_hash: str,
    source_provenance_key: bytes,
    custody_public_key_pem: bytes,
    closeout_public_keys: Mapping[str, bytes],
    disclosed_at: datetime | None = None,
) -> tuple[SourceProvenanceRecordV55, SourceProvenanceDisclosureReceiptV55]:
    """Release metadata only; this API cannot receive target ciphertext or key."""

    protocol.assert_sealed()
    source_provenance_envelope.assert_sealed()
    if source_provenance_envelope.domain != "source_provenance":
        raise ValueError("source release requires a provenance envelope")
    if not verify_split_custody_attestation_signature_v55(
        attestation=attestation,
        custody_public_key_pem=custody_public_key_pem,
    ):
        raise ValueError("split custody attestation is not authenticated")
    if not verify_campaign_closeout_authorization_v55(
        authorization=authorization,
        protocol=protocol,
        attestation=attestation,
        closeout_public_keys=closeout_public_keys,
    ):
        raise ValueError("source provenance release is not authorized")
    if terminal_evidence_hash != authorization.terminal_evidence_hash:
        raise ValueError("terminal campaign evidence hash differs")
    if not (
        source_provenance_envelope.case_id == attestation.case_id
        and source_provenance_envelope.envelope_hash
        == attestation.source_provenance_envelope_hash
        and source_provenance_envelope.plaintext_commitment
        == attestation.source_provenance_commitment
        and source_provenance_envelope.key_id == attestation.source_provenance_key_id
        and source_provenance_envelope.key_fingerprint
        == attestation.source_provenance_key_fingerprint
    ):
        raise ValueError("source provenance envelope binding differs")
    plaintext = _decrypt_envelope_v55(
        envelope=source_provenance_envelope,
        key=source_provenance_key,
    )
    record = SourceProvenanceRecordV55.model_validate_json(plaintext)
    record.assert_sealed()
    if not (
        record.case_id == attestation.case_id
        and record.protocol_hash == protocol.protocol_hash
        and record.record_hash == attestation.source_provenance_commitment
    ):
        raise ValueError("released source provenance record binding differs")
    receipt = SourceProvenanceDisclosureReceiptV55.seal(
        case_id=record.case_id,
        protocol_hash=record.protocol_hash,
        source_record_hash=record.record_hash,
        source_provenance_envelope_hash=source_provenance_envelope.envelope_hash,
        split_custody_attestation_hash=attestation.attestation_hash,
        closeout_authorization_hash=authorization.authorization_hash,
        disclosed_at=disclosed_at or _utc_now(),
    )
    return record, receipt


def open_private_target_envelope_v55(
    *,
    private_target_envelope: EncryptedCustodyEnvelopeV55,
    private_target_key: bytes,
) -> PrivateCaseCapsuleV50:
    """Private-worker-only target decryption; never used by provenance release."""

    if private_target_envelope.domain != "private_targets":
        raise ValueError("private target worker requires a target envelope")
    plaintext = _decrypt_envelope_v55(
        envelope=private_target_envelope,
        key=private_target_key,
    )
    capsule = PrivateCaseCapsuleV50.model_validate_json(plaintext)
    capsule.assert_sealed()
    if not (
        capsule.case_id == private_target_envelope.case_id
        and capsule.capsule_hash == private_target_envelope.plaintext_commitment
    ):
        raise ValueError("private target capsule binding differs")
    return capsule


def verify_source_provenance_disclosure_v55(
    *,
    protocol: ProspectiveCampaignProtocolV55,
    source_record: SourceProvenanceRecordV55,
    source_provenance_envelope: EncryptedCustodyEnvelopeV55,
    attestation: SplitCustodyAttestationV55,
    authorization: CampaignCloseoutAuthorizationV55,
    receipt: SourceProvenanceDisclosureReceiptV55,
    terminal_evidence_hash: str,
    custody_public_key_pem: bytes,
    closeout_public_keys: Mapping[str, bytes],
) -> bool:
    """Verify a released source record without any decryption or target access."""

    try:
        protocol.assert_sealed()
        source_record.assert_sealed()
        source_provenance_envelope.assert_sealed()
        attestation.assert_sealed()
        authorization.assert_sealed()
        receipt.assert_sealed()
        return bool(
            source_provenance_envelope.domain == "source_provenance"
            and verify_split_custody_attestation_signature_v55(
                attestation=attestation,
                custody_public_key_pem=custody_public_key_pem,
            )
            and verify_campaign_closeout_authorization_v55(
                authorization=authorization,
                protocol=protocol,
                attestation=attestation,
                closeout_public_keys=closeout_public_keys,
            )
            and terminal_evidence_hash == authorization.terminal_evidence_hash
            and source_record.case_id
            == source_provenance_envelope.case_id
            == attestation.case_id
            == authorization.case_id
            == receipt.case_id
            and source_record.protocol_hash
            == protocol.protocol_hash
            == attestation.protocol_hash
            == authorization.protocol_hash
            == receipt.protocol_hash
            and source_record.record_hash
            == source_provenance_envelope.plaintext_commitment
            == attestation.source_provenance_commitment
            == receipt.source_record_hash
            and source_provenance_envelope.envelope_hash
            == attestation.source_provenance_envelope_hash
            == receipt.source_provenance_envelope_hash
            and attestation.attestation_hash
            == authorization.split_custody_attestation_hash
            == receipt.split_custody_attestation_hash
            and authorization.authorization_hash == receipt.closeout_authorization_hash
        )
    except (TypeError, ValueError):
        return False


__all__ = [
    "CampaignCloseoutAuthorizationV55",
    "EncryptedCustodyEnvelopeV55",
    "SourceProvenanceDisclosureReceiptV55",
    "SourceProvenanceDraftV55",
    "SourceProvenanceRecordV55",
    "SplitCustodyAttestationV55",
    "create_split_custody_envelopes_v55",
    "encryption_key_fingerprint_v55",
    "open_private_target_envelope_v55",
    "release_source_provenance_v55",
    "sign_campaign_closeout_authorization_v55",
    "signing_key_fingerprint_v55",
    "verify_campaign_closeout_authorization_v55",
    "verify_source_provenance_disclosure_v55",
    "verify_split_custody_attestation_signature_v55",
    "verify_split_custody_bindings_v55",
]
