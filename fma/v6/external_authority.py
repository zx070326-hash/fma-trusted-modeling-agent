"""V6.4 deployment-anchored authority for V6.3 predictive qualification.

V6.3 deliberately accepts caller-supplied public keys and host labels.  That
is sufficient to exercise the protocol, but it cannot prove that the four
external roles were independently administered or separately hosted.  This
additive module raises the claim ceiling only when a deployment-injected trust
provider verifies:

* an Ed25519 root-signed manifest for the exact V6.3 contract;
* an Ed25519 root-signed, current revocation snapshot;
* four root-approved-attester signatures over role-specific host evidence;
* a read-only PASS replay of the exact V6.3 qualification receipt.

The provider object is a capability boundary, not a cryptographic proof of its
own deployment.  Production callers must inject it from a trusted process
boundary, keep its root and epoch-floor configuration outside task-controlled
inputs, and preserve the epoch floor across process restarts.  Constructing a
new provider from task-supplied keys only proves another local rehearsal.

No V6.4 result authorizes a real-world action.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Literal

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

from .external_qualification import (
    CurrentModelPredictionBindingV63,
    ExternalEvidenceCustodyV63,
    ExternalForecastInputV63,
    ExternalPredictionVectorV63,
    ExternalPredictiveQualificationReceiptV63,
    PredictiveExternalQualificationContractV63,
    external_qualification_key_fingerprint_v63,
    verify_external_predictive_qualification_v63,
)
from .external_prediction_runtime import (
    verify_current_model_external_prediction_v63,
)


ExternalAuthorityRoleV64 = Literal[
    "custody",
    "registry",
    "evaluator",
    "promotion",
]
AuthorityAssessmentModeV64 = Literal["rehearsal", "anchored"]
AuthorityAnchorStatusV64 = Literal[
    "ANCHORED_CURRENT",
    "UNANCHORED_REHEARSAL",
]
AuthorityQualificationStatusV64 = Literal[
    "NOT_RUN",
    "REJECTED",
    "EXTERNALLY_QUALIFIED",
]
AuthorityClaimCeilingV64 = Literal[
    "workflow_integrity_only",
    "externally_qualified_predictive_evidence",
]
AuthorityProtocolStatusV64 = Literal["NOT_RUN", "PASS", "FAIL"]

_AUTHORITY_ROLES = ("custody", "registry", "evaluator", "promotion")
_ROLE_ARTIFACTS: dict[str, tuple[str, str]] = {
    "custody": ("custody", "custodian_host_id"),
    "registry": ("registration", "registry_host_id"),
    "evaluator": ("evaluation", "evaluator_host_id"),
    "promotion": ("promotion", "promotion_host_id"),
}
_DEPLOYMENT_ASSUMPTION = (
    "provider_identity_and_epoch_floor_are_injected_by_a_trusted_deployment"
)
_MANIFEST_KIND = "external_authority_manifest_v64"
_REVOCATION_KIND = "external_authority_revocations_v64"
_HOST_ATTESTATION_KIND = "external_role_host_attestation_v64"
_AUTHORITY_RECEIPT_KIND = "external_authority_qualification_v64"


class ExternalAuthorityError(RuntimeError):
    """A fail-closed V6.4 authority validation error."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _assert_aware(value: datetime, field_name: str) -> None:
    if value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _unsigned_bytes(model: StrictModel, hash_field: str) -> bytes:
    return canonical_json(
        model.model_dump(
            mode="json",
            exclude={"signature_base64", hash_field},
        )
    ).encode("utf-8")


def _load_private_key(private_key_pem: bytes) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(private_key_pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("V6.4 signing key must be Ed25519")
    return key


def _load_public_key(public_key_pem: bytes) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(public_key_pem)
    if not isinstance(key, Ed25519PublicKey):
        raise TypeError("V6.4 trusted key must be Ed25519")
    return key


def _fingerprint_key(public_key_pem: bytes) -> str:
    return external_qualification_key_fingerprint_v63(public_key_pem)


def _sign_model(
    *,
    model_type: type[StrictModel],
    data: dict[str, object],
    private_key_pem: bytes,
    hash_field: str,
) -> StrictModel:
    unsigned = model_type(**data)
    signature = _load_private_key(private_key_pem).sign(
        _unsigned_bytes(unsigned, hash_field)
    )
    signed_payload = unsigned.model_dump(mode="json")
    signed_payload["signature_base64"] = base64.b64encode(signature).decode(
        "ascii"
    )
    signed = model_type(**signed_payload)
    final_payload = signed.model_dump(mode="json")
    final_payload[hash_field] = _hash_without(signed, hash_field)
    return model_type(**final_payload)


def _verify_signature(
    *,
    model: StrictModel,
    public_key: Ed25519PublicKey | None,
    signature_base64: str | None,
    hash_field: str,
) -> bool:
    if public_key is None or not signature_base64:
        return False
    try:
        signature = base64.b64decode(
            signature_base64.encode("ascii"),
            validate=True,
        )
        public_key.verify(signature, _unsigned_bytes(model, hash_field))
    except (InvalidSignature, TypeError, ValueError):
        return False
    return True


class DeploymentIdentityV64(StrictModel):
    """Root-approved deployment identity for a non-attested local role."""

    v63_host_id: Identifier
    host_identity_commitment: Sha256
    execution_boundary: Identifier
    management_domain: Identifier


class ExternalAuthorityManifestV64(StrictModel):
    """Root-signed authorization for one exact V6.3 contract."""

    schema_version: Literal["6.4-external-authority-manifest"] = (
        "6.4-external-authority-manifest"
    )
    manifest_id: Identifier
    trust_domain_id: Identifier
    qualification_id: Identifier
    task_id: Identifier
    v63_contract_hash: Sha256
    authority_key_fingerprints: dict[Identifier, Sha256]
    host_attester_key_id: Identifier
    host_attester_key_fingerprint: Sha256
    coordinator_identity: DeploymentIdentityV64
    generator_identity: DeploymentIdentityV64
    issued_at: datetime
    valid_from: datetime
    valid_until: datetime
    root_key_id: Identifier
    signature_base64: str | None = None
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "ExternalAuthorityManifestV64":
        if set(self.authority_key_fingerprints) != set(_AUTHORITY_ROLES):
            raise ValueError("V6.4 manifest must approve all four roles")
        if len(set(self.authority_key_fingerprints.values())) != 4:
            raise ValueError("V6.4 authority fingerprints must be distinct")
        _assert_aware(self.issued_at, "issued_at")
        _assert_aware(self.valid_from, "valid_from")
        _assert_aware(self.valid_until, "valid_until")
        if self.valid_until <= self.valid_from:
            raise ValueError("V6.4 manifest validity window is empty")
        if not self.valid_from <= self.issued_at <= self.valid_until:
            raise ValueError("V6.4 manifest issuance is outside validity")
        for field_name in (
            "host_identity_commitment",
            "execution_boundary",
            "management_domain",
        ):
            if getattr(self.coordinator_identity, field_name) == getattr(
                self.generator_identity,
                field_name,
            ):
                raise ValueError(
                    "V6.4 coordinator and generator identities must differ"
                )
        if self.signature_base64 is None and self.manifest_hash is not None:
            raise ValueError("V6.4 manifest hash requires a signature")
        if (
            self.manifest_hash is not None
            and self.manifest_hash != self.content_hash()
        ):
            raise ValueError("V6.4 manifest hash differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "signature_base64", "manifest_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if (
            not self.signature_base64
            or not self.manifest_hash
            or self.manifest_hash != self.content_hash()
        ):
            raise ValueError("V6.4 authority manifest is not sealed")


def sign_external_authority_manifest_v64(
    *,
    private_key_pem: bytes,
    **data: object,
) -> ExternalAuthorityManifestV64:
    """Sign a manifest outside the assessment path."""

    return _sign_model(
        model_type=ExternalAuthorityManifestV64,
        data=data,
        private_key_pem=private_key_pem,
        hash_field="manifest_hash",
    )  # type: ignore[return-value]


class ExternalAuthorityRevocationSnapshotV64(StrictModel):
    """Root-signed current revocation state for one trust domain."""

    schema_version: Literal["6.4-external-authority-revocations"] = (
        "6.4-external-authority-revocations"
    )
    snapshot_id: Identifier
    trust_domain_id: Identifier
    epoch: int = Field(ge=0)
    previous_snapshot_hash: Sha256 | None = None
    revoked_manifest_hashes: list[Sha256] = Field(default_factory=list)
    revoked_key_fingerprints: list[Sha256] = Field(default_factory=list)
    revoked_host_identity_commitments: list[Sha256] = Field(
        default_factory=list
    )
    effective_at: datetime
    valid_until: datetime
    root_key_id: Identifier
    signature_base64: str | None = None
    snapshot_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_snapshot(self) -> "ExternalAuthorityRevocationSnapshotV64":
        for values, label in (
            (self.revoked_manifest_hashes, "manifest"),
            (self.revoked_key_fingerprints, "key"),
            (self.revoked_host_identity_commitments, "host"),
        ):
            if values != sorted(set(values)):
                raise ValueError(
                    f"V6.4 revoked {label} entries must be sorted and unique"
                )
        _assert_aware(self.effective_at, "effective_at")
        _assert_aware(self.valid_until, "valid_until")
        if self.valid_until <= self.effective_at:
            raise ValueError("V6.4 revocation validity window is empty")
        if self.epoch > 0 and self.previous_snapshot_hash is None:
            raise ValueError(
                "V6.4 non-genesis revocation snapshot needs a predecessor"
            )
        if self.signature_base64 is None and self.snapshot_hash is not None:
            raise ValueError("V6.4 revocation hash requires a signature")
        if (
            self.snapshot_hash is not None
            and self.snapshot_hash != self.content_hash()
        ):
            raise ValueError("V6.4 revocation snapshot hash differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "signature_base64", "snapshot_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "snapshot_hash")

    def assert_sealed(self) -> None:
        if (
            not self.signature_base64
            or not self.snapshot_hash
            or self.snapshot_hash != self.content_hash()
        ):
            raise ValueError("V6.4 revocation snapshot is not sealed")


def sign_external_authority_revocations_v64(
    *,
    private_key_pem: bytes,
    **data: object,
) -> ExternalAuthorityRevocationSnapshotV64:
    """Sign a revocation snapshot outside the assessment path."""

    return _sign_model(
        model_type=ExternalAuthorityRevocationSnapshotV64,
        data=data,
        private_key_pem=private_key_pem,
        hash_field="snapshot_hash",
    )  # type: ignore[return-value]


class ExternalRoleHostAttestationV64(StrictModel):
    """Attester-signed host identity for one V6.3 authority artifact."""

    schema_version: Literal["6.4-external-role-host-attestation"] = (
        "6.4-external-role-host-attestation"
    )
    attestation_id: Identifier
    trust_domain_id: Identifier
    manifest_hash: Sha256
    qualification_id: Identifier
    task_id: Identifier
    v63_contract_hash: Sha256
    role: ExternalAuthorityRoleV64
    role_key_fingerprint: Sha256
    v63_role_evidence_hash: Sha256
    v63_declared_host_id: Identifier
    host_identity_commitment: Sha256
    execution_boundary: Identifier
    management_domain: Identifier
    external_control_plane_verified: Literal[True] = True
    real_world_action_authorized: Literal[False] = False
    attested_at: datetime
    valid_until: datetime
    attester_key_id: Identifier
    signature_base64: str | None = None
    attestation_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_attestation(self) -> "ExternalRoleHostAttestationV64":
        _assert_aware(self.attested_at, "attested_at")
        _assert_aware(self.valid_until, "valid_until")
        if self.valid_until <= self.attested_at:
            raise ValueError("V6.4 host-attestation validity window is empty")
        if self.signature_base64 is None and self.attestation_hash is not None:
            raise ValueError("V6.4 attestation hash requires a signature")
        if (
            self.attestation_hash is not None
            and self.attestation_hash != self.content_hash()
        ):
            raise ValueError("V6.4 host-attestation hash differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "signature_base64", "attestation_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "attestation_hash")

    def assert_sealed(self) -> None:
        if (
            not self.signature_base64
            or not self.attestation_hash
            or self.attestation_hash != self.content_hash()
        ):
            raise ValueError("V6.4 role host attestation is not sealed")


def sign_external_role_host_attestation_v64(
    *,
    private_key_pem: bytes,
    **data: object,
) -> ExternalRoleHostAttestationV64:
    """Sign a role-host attestation outside the assessment path."""

    return _sign_model(
        model_type=ExternalRoleHostAttestationV64,
        data=data,
        private_key_pem=private_key_pem,
        hash_field="attestation_hash",
    )  # type: ignore[return-value]


class LocalStaticTrustProviderV64:
    """Read-only local verifier that can never grant an external anchor.

    Caller-created keys can validate protocol mechanics but cannot establish
    deployment provenance.  Consequently every assessment using this class
    remains ``NOT_RUN / UNANCHORED_REHEARSAL``.  A future external provider
    plugin must supply a separately pinned deployment capability; this
    repository intentionally contains no test backdoor for that transition.
    """

    def __init__(
        self,
        *,
        provider_id: str,
        trust_domain_id: str,
        root_public_keys: Mapping[str, bytes],
        authority_public_keys: Mapping[str, bytes],
        host_attester_public_keys: Mapping[str, bytes],
        pinned_revocation_epoch: int,
        pinned_revocation_snapshot_hash: str,
        pinned_previous_snapshot_hash: str | None,
        pinned_revoked_manifest_hashes: Sequence[str] = (),
        pinned_revoked_key_fingerprints: Sequence[str] = (),
        pinned_revoked_host_identity_commitments: Sequence[str] = (),
    ) -> None:
        if pinned_revocation_epoch < 0:
            raise ValueError("pinned revocation epoch cannot be negative")
        if (
            len(pinned_revocation_snapshot_hash) != 64
            or pinned_revocation_epoch > 0
            and pinned_previous_snapshot_hash is None
        ):
            raise ValueError("V6.4 pinned revocation state is incomplete")
        if not root_public_keys:
            raise ValueError("V6.4 provider needs at least one root key")
        if not authority_public_keys:
            raise ValueError("V6.4 provider needs authority keys")
        if not host_attester_public_keys:
            raise ValueError("V6.4 provider needs a host-attester key")

        roots = {
            key_id: _load_public_key(bytes(public_key_pem))
            for key_id, public_key_pem in root_public_keys.items()
        }
        authorities = {
            key_id: _load_public_key(bytes(public_key_pem))
            for key_id, public_key_pem in authority_public_keys.items()
        }
        attesters = {
            key_id: _load_public_key(bytes(public_key_pem))
            for key_id, public_key_pem in host_attester_public_keys.items()
        }
        root_fingerprints = {
            key_id: _public_key_fingerprint(key)
            for key_id, key in roots.items()
        }
        authority_fingerprints = {
            key_id: _public_key_fingerprint(key)
            for key_id, key in authorities.items()
        }
        attester_fingerprints = {
            key_id: _public_key_fingerprint(key)
            for key_id, key in attesters.items()
        }
        all_fingerprints = [
            *root_fingerprints.values(),
            *authority_fingerprints.values(),
            *attester_fingerprints.values(),
        ]
        if len(all_fingerprints) != len(set(all_fingerprints)):
            raise ValueError(
                "V6.4 root, authority, and attester keys must be distinct"
            )

        self.provider_id = provider_id
        self.trust_domain_id = trust_domain_id
        self.anchor_provenance: Literal["LOCAL_REHEARSAL"] = (
            "LOCAL_REHEARSAL"
        )
        self._root_keys = MappingProxyType(roots)
        self._authority_keys = MappingProxyType(authorities)
        self._attester_keys = MappingProxyType(attesters)
        self._root_fingerprints = MappingProxyType(root_fingerprints)
        self._authority_fingerprints = MappingProxyType(
            authority_fingerprints
        )
        self._attester_fingerprints = MappingProxyType(attester_fingerprints)
        self._pinned_revocation_epoch = pinned_revocation_epoch
        self._pinned_revocation_snapshot_hash = (
            pinned_revocation_snapshot_hash
        )
        self._pinned_previous_snapshot_hash = pinned_previous_snapshot_hash
        self._pinned_revoked_manifest_hashes = frozenset(
            pinned_revoked_manifest_hashes
        )
        self._pinned_revoked_key_fingerprints = frozenset(
            pinned_revoked_key_fingerprints
        )
        self._pinned_revoked_host_identity_commitments = frozenset(
            pinned_revoked_host_identity_commitments
        )
        self._highest_revocation_epoch = pinned_revocation_epoch - 1
        self._accepted_snapshot_hash_by_epoch: dict[int, str] = {}
        self._epoch_lock = threading.Lock()
        self.configuration_hash = sha256_value(
            {
                "provider_id": provider_id,
                "trust_domain_id": trust_domain_id,
                "root_key_fingerprints": root_fingerprints,
                "authority_key_fingerprints": authority_fingerprints,
                "host_attester_key_fingerprints": attester_fingerprints,
                "anchor_provenance": self.anchor_provenance,
                "pinned_revocation_epoch": pinned_revocation_epoch,
                "pinned_revocation_snapshot_hash": (
                    pinned_revocation_snapshot_hash
                ),
                "pinned_previous_snapshot_hash": (
                    pinned_previous_snapshot_hash
                ),
                "pinned_revoked_manifest_hashes": sorted(
                    self._pinned_revoked_manifest_hashes
                ),
                "pinned_revoked_key_fingerprints": sorted(
                    self._pinned_revoked_key_fingerprints
                ),
                "pinned_revoked_host_identity_commitments": sorted(
                    self._pinned_revoked_host_identity_commitments
                ),
                "deployment_assumption": _DEPLOYMENT_ASSUMPTION,
            }
        )

    def trusted_assessment_time(self) -> datetime:
        """Return the provider clock; task input cannot choose this value."""

        return _utc_now()

    def replay_v63(
        self,
        *,
        workspace: StageWorkspaceV50,
        receipt: ExternalPredictiveQualificationReceiptV63,
    ) -> object:
        """Replay V6.3 with provider-held authority keys."""

        public_keys = {
            key_id: key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            for key_id, key in self._authority_keys.items()
        }
        return verify_external_predictive_qualification_v63(
            workspace=workspace,
            receipt=receipt,
            trusted_public_keys=public_keys,
        )

    def verify_manifest_signature(
        self,
        manifest: ExternalAuthorityManifestV64,
    ) -> bool:
        try:
            manifest.assert_sealed()
            return bool(
                manifest.trust_domain_id == self.trust_domain_id
                and _verify_signature(
                    model=manifest,
                    public_key=self._root_keys.get(manifest.root_key_id),
                    signature_base64=manifest.signature_base64,
                    hash_field="manifest_hash",
                )
            )
        except (AttributeError, TypeError, ValueError):
            return False

    def authority_key_matches(
        self,
        *,
        key_id: str,
        fingerprint: str,
    ) -> bool:
        return self._authority_fingerprints.get(key_id) == fingerprint

    def attester_key_matches(
        self,
        *,
        key_id: str,
        fingerprint: str,
    ) -> bool:
        return self._attester_fingerprints.get(key_id) == fingerprint

    def verify_host_attestation_signature(
        self,
        attestation: ExternalRoleHostAttestationV64,
    ) -> bool:
        try:
            attestation.assert_sealed()
            return bool(
                attestation.trust_domain_id == self.trust_domain_id
                and _verify_signature(
                    model=attestation,
                    public_key=self._attester_keys.get(
                        attestation.attester_key_id
                    ),
                    signature_base64=attestation.signature_base64,
                    hash_field="attestation_hash",
                )
            )
        except (AttributeError, TypeError, ValueError):
            return False

    def accept_current_revocation_snapshot(
        self,
        *,
        snapshot: ExternalAuthorityRevocationSnapshotV64,
        assessed_at: datetime,
    ) -> bool:
        """Verify and monotonically accept one root-signed snapshot."""

        try:
            snapshot.assert_sealed()
            _assert_aware(assessed_at, "assessed_at")
            root_key = self._root_keys.get(snapshot.root_key_id)
            signature_valid = _verify_signature(
                model=snapshot,
                public_key=root_key,
                signature_base64=snapshot.signature_base64,
                hash_field="snapshot_hash",
            )
            signer_fingerprint = self._root_fingerprints.get(
                snapshot.root_key_id
            )
            static_valid = bool(
                signature_valid
                and snapshot.trust_domain_id == self.trust_domain_id
                and snapshot.effective_at <= assessed_at
                <= snapshot.valid_until
                and snapshot.epoch == self._pinned_revocation_epoch
                and snapshot.snapshot_hash
                == self._pinned_revocation_snapshot_hash
                and snapshot.previous_snapshot_hash
                == self._pinned_previous_snapshot_hash
                and self._pinned_revoked_manifest_hashes.issubset(
                    snapshot.revoked_manifest_hashes
                )
                and self._pinned_revoked_key_fingerprints.issubset(
                    snapshot.revoked_key_fingerprints
                )
                and (
                    self._pinned_revoked_host_identity_commitments.issubset(
                        snapshot.revoked_host_identity_commitments
                    )
                )
                and signer_fingerprint
                and signer_fingerprint
                not in snapshot.revoked_key_fingerprints
            )
            if not static_valid or not snapshot.snapshot_hash:
                return False
            with self._epoch_lock:
                if snapshot.epoch < self._highest_revocation_epoch:
                    return False
                prior_hash = self._accepted_snapshot_hash_by_epoch.get(
                    snapshot.epoch
                )
                if prior_hash is not None and prior_hash != snapshot.snapshot_hash:
                    return False
                self._accepted_snapshot_hash_by_epoch[
                    snapshot.epoch
                ] = snapshot.snapshot_hash
                self._highest_revocation_epoch = max(
                    self._highest_revocation_epoch,
                    snapshot.epoch,
                )
            return True
        except (AttributeError, TypeError, ValueError):
            return False

    def root_key_fingerprint(self, key_id: str) -> str | None:
        return self._root_fingerprints.get(key_id)


# Backward-compatible name within this additive, unreleased module.  It is a
# local verifier, not an externally anchored deployment capability.
DeploymentTrustProviderV64 = LocalStaticTrustProviderV64


def _public_key_fingerprint(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


def _commit_or_replay_authority_artifact(
    *,
    workspace: StageWorkspaceV50,
    kind: str,
    model: StrictModel,
    model_type: type[StrictModel],
    same_identity: Callable[[StrictModel], bool],
    persist: bool,
) -> str:
    """Commit one immutable envelope, or recover its exact prior artifact."""

    transaction_factory = getattr(
        getattr(getattr(workspace, "graph", None), "store", None),
        "writer_transaction",
        None,
    )
    transaction = (
        transaction_factory()
        if callable(transaction_factory)
        else nullcontext()
    )
    with transaction:
        try:
            matches = [
                (reference, item)
                for reference, item in workspace._artifacts_of_kind(
                    kind,
                    model_type,
                )
                if same_identity(item)
            ]
        except (
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            raise ExternalAuthorityError(
                f"{kind} ledger could not be replayed"
            ) from exc
        exact = [
            reference.sha256
            for reference, item in matches
            if item == model
        ]
        if exact:
            if len(matches) != 1 or len(exact) != 1:
                raise ExternalAuthorityError(
                    f"{kind} ledger contains duplicates"
                )
            return exact[0]
        if matches:
            raise ExternalAuthorityError(
                f"{kind} is immutable for this authority identity"
            )
        if not persist:
            raise ExternalAuthorityError(f"{kind} is not committed")
        reference = workspace.commit_evidence(
            kind,
            model.model_dump(mode="json"),
        )
        return reference.sha256


def register_external_authority_manifest_v64(
    *,
    workspace: StageWorkspaceV50,
    manifest: ExternalAuthorityManifestV64,
) -> str:
    """Precommit a root-signed manifest before forecast input or custody."""

    manifest.assert_sealed()
    return _commit_or_replay_authority_artifact(
        workspace=workspace,
        kind=_MANIFEST_KIND,
        model=manifest,
        model_type=ExternalAuthorityManifestV64,
        same_identity=lambda item: (
            getattr(item, "qualification_id", None)
            == manifest.qualification_id
            and getattr(item, "task_id", None) == manifest.task_id
        ),
        persist=True,
    )


def _artifact_commit_sequence(
    workspace: StageWorkspaceV50,
    artifact_hash: str | None,
) -> int | None:
    if artifact_hash is None:
        return None
    try:
        records = [
            json.loads(line)
            for line in workspace.graph.store.event_path.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
    except (AttributeError, json.JSONDecodeError, OSError, TypeError):
        return None
    sequences = [
        item.get("sequence")
        for item in records
        if item.get("event_type") == "artifact_committed"
        and isinstance(item.get("payload"), dict)
        and item["payload"].get("sha256") == artifact_hash
    ]
    return sequences[0] if len(sequences) == 1 else None


def _exact_v63_receipt_artifact_hash(
    *,
    workspace: StageWorkspaceV50,
    receipt: ExternalPredictiveQualificationReceiptV63,
) -> str | None:
    try:
        matches = [
            reference.sha256
            for reference, item in workspace._artifacts_of_kind(
                "external_predictive_qualification_v63",
                ExternalPredictiveQualificationReceiptV63,
            )
            if item == receipt
        ]
    except (
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return None
    return matches[0] if len(matches) == 1 else None


class ExternalAuthorityQualificationReceiptV64(StrictModel):
    """Code-owned V6.4 claim receipt; action authority is always false."""

    schema_version: Literal["6.4-external-authority-qualification"] = (
        "6.4-external-authority-qualification"
    )
    qualification_id: Identifier
    task_id: Identifier
    assessment_request_hash: Sha256
    v63_contract_hash: Sha256
    v63_receipt_hash: Sha256
    v63_replay_hash: Sha256 | None = None
    trust_provider_id: Identifier | None = None
    trust_provider_configuration_hash: Sha256 | None = None
    manifest_hash: Sha256 | None = None
    revocation_snapshot_hash: Sha256 | None = None
    host_attestation_hashes: dict[Identifier, Sha256] = Field(
        default_factory=dict
    )
    authority_artifact_hashes: dict[Identifier, Sha256] = Field(
        default_factory=dict
    )
    assessment_mode: AuthorityAssessmentModeV64
    anchor_protocol_status: AuthorityProtocolStatusV64
    anchor_status: AuthorityAnchorStatusV64
    status: AuthorityQualificationStatusV64
    reason_codes: list[Identifier]
    checks: dict[Identifier, bool]
    anchored_predictive_qualification_granted: bool
    anchored_scientific_qualification_granted: bool
    claim_ceiling: AuthorityClaimCeilingV64
    deployment_integrity_assumption: Literal[
        "provider_identity_and_epoch_floor_are_injected_by_a_trusted_deployment"
    ] = _DEPLOYMENT_ASSUMPTION
    real_world_action_authorized: Literal[False] = False
    assessed_at: datetime
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "ExternalAuthorityQualificationReceiptV64":
        _assert_aware(self.assessed_at, "assessed_at")
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("V6.4 authority reasons must be sorted and unique")
        qualified = self.status == "EXTERNALLY_QUALIFIED"
        if (
            self.anchored_predictive_qualification_granted != qualified
            or self.anchored_scientific_qualification_granted != qualified
        ):
            raise ValueError("V6.4 qualification flags differ from status")
        if qualified and (
            self.assessment_mode != "anchored"
            or self.anchor_status != "ANCHORED_CURRENT"
            or set(self.host_attestation_hashes) != set(_AUTHORITY_ROLES)
            or set(self.authority_artifact_hashes)
            != {
                "v63_receipt",
                "manifest",
                "revocation_snapshot",
                *(
                    f"host_attestation_{role}"
                    for role in _AUTHORITY_ROLES
                ),
            }
            or not self.checks
            or not all(self.checks.values())
            or self.reason_codes
            or self.claim_ceiling
            != "externally_qualified_predictive_evidence"
        ):
            raise ValueError("V6.4 qualified receipt lacks mandatory evidence")
        if not qualified and (
            self.anchor_status == "ANCHORED_CURRENT"
            or self.claim_ceiling
            == "externally_qualified_predictive_evidence"
        ):
            raise ValueError("V6.4 unqualified receipt exceeds its authority")
        if self.assessment_mode == "rehearsal" and self.status != "NOT_RUN":
            raise ValueError("V6.4 rehearsal must remain NOT_RUN")
        if (
            self.anchor_protocol_status == "PASS"
            and any(
                not passed
                for check_id, passed in self.checks.items()
                if check_id != "deployment_anchor_current"
            )
        ):
            raise ValueError("V6.4 protocol PASS differs from checks")
        if self.receipt_hash is not None and self.authority_auth_tag is None:
            raise ValueError("V6.4 receipt hash requires workspace authority")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("V6.4 authority receipt hash differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(
            self,
            "authority_auth_tag",
            "receipt_hash",
        )

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("V6.4 authority receipt is not sealed")


def _issue_authority_receipt(
    *,
    workspace: StageWorkspaceV50,
    data: dict[str, object],
) -> ExternalAuthorityQualificationReceiptV64:
    data["authority_key_id"] = workspace.authority_key_id
    unsigned = ExternalAuthorityQualificationReceiptV64(**data)
    tagged_payload = unsigned.model_dump(mode="json")
    tagged_payload["authority_auth_tag"] = workspace._mac(
        _AUTHORITY_RECEIPT_KIND,
        unsigned.unsigned_hash(),
    )
    tagged = ExternalAuthorityQualificationReceiptV64(**tagged_payload)
    final_payload = tagged.model_dump(mode="json")
    final_payload["receipt_hash"] = tagged.content_hash()
    return ExternalAuthorityQualificationReceiptV64(**final_payload)


def _commit_authority_receipt(
    *,
    workspace: StageWorkspaceV50,
    receipt: ExternalAuthorityQualificationReceiptV64,
    persist: bool,
) -> ExternalAuthorityQualificationReceiptV64:
    _commit_or_replay_authority_artifact(
        workspace=workspace,
        kind=_AUTHORITY_RECEIPT_KIND,
        model=receipt,
        model_type=ExternalAuthorityQualificationReceiptV64,
        same_identity=lambda item: (
            getattr(item, "assessment_request_hash", None)
            == receipt.assessment_request_hash
        ),
        persist=persist,
    )
    return receipt


def _safe_v63_hash(
    model: object,
    field_name: str,
    fallback_label: str,
) -> str:
    value = getattr(model, field_name, None)
    if isinstance(value, str) and len(value) == 64:
        return value
    return sha256_value({"invalid_v63_binding": fallback_label})


def _base_checks() -> dict[str, bool]:
    return {
        "v63_contract_sealed": False,
        "v63_receipt_sealed": False,
        "v63_exact_binding": False,
        "v63_action_forbidden": False,
        "v63_replay_pass": False,
        "v63_status_qualified": False,
        "current_model_prediction_recomputed": False,
        "trust_approval_frozen_before_external_evidence": False,
        "manifest_transparency_precedes_forecast_input": False,
        "revocation_assessed_after_v63_finality": False,
        "deployment_anchor_current": False,
        "manifest_signature_current": False,
        "manifest_exact_contract": False,
        "manifest_role_keys_current": False,
        "revocation_snapshot_current": False,
        "no_revoked_authority_material": False,
        "host_attestation_set_complete": False,
        "host_attestation_signatures_current": False,
        "host_attestation_bindings_exact": False,
        "external_role_isolation": False,
        "coordinator_generator_isolation": False,
    }


def _make_receipt(
    *,
    workspace: StageWorkspaceV50,
    contract: PredictiveExternalQualificationContractV63,
    v63_receipt: ExternalPredictiveQualificationReceiptV63,
    assessed_at: datetime,
    assessment_mode: AuthorityAssessmentModeV64,
    provider: DeploymentTrustProviderV64 | None,
    manifest: ExternalAuthorityManifestV64 | None,
    revocation_snapshot: ExternalAuthorityRevocationSnapshotV64 | None,
    host_attestations: Sequence[ExternalRoleHostAttestationV64],
    checks: dict[str, bool],
    reasons: Sequence[str],
    status: AuthorityQualificationStatusV64,
    replay_hash: str | None,
    authority_artifact_hashes: Mapping[str, str],
    persist: bool,
) -> ExternalAuthorityQualificationReceiptV64:
    qualified = status == "EXTERNALLY_QUALIFIED"
    assessment_request_hash = sha256_value(
        {
            "qualification_id": getattr(contract, "qualification_id", None),
            "task_id": getattr(contract, "task_id", None),
            "v63_contract_hash": getattr(contract, "contract_hash", None),
            "v63_receipt_hash": getattr(v63_receipt, "receipt_hash", None),
            "provider_configuration_hash": (
                provider.configuration_hash if provider else None
            ),
            "manifest_hash": (
                manifest.manifest_hash if manifest else None
            ),
            "revocation_snapshot_hash": (
                revocation_snapshot.snapshot_hash
                if revocation_snapshot
                else None
            ),
            "host_attestation_hashes": sorted(
                item.attestation_hash or "" for item in host_attestations
            ),
            "assessment_mode": assessment_mode,
            "assessed_at": assessed_at.isoformat(),
        }
    )
    receipt = _issue_authority_receipt(
        workspace=workspace,
        data={
            "qualification_id": getattr(
                contract,
                "qualification_id",
                getattr(
                    v63_receipt,
                    "qualification_id",
                    "invalid.qualification",
                ),
            ),
            "task_id": getattr(
                contract,
                "task_id",
                getattr(v63_receipt, "task_id", "invalid.task"),
            ),
            "assessment_request_hash": assessment_request_hash,
            "v63_contract_hash": _safe_v63_hash(
                contract,
                "contract_hash",
                "contract",
            ),
            "v63_receipt_hash": _safe_v63_hash(
                v63_receipt,
                "receipt_hash",
                "receipt",
            ),
            "v63_replay_hash": replay_hash,
            "trust_provider_id": provider.provider_id if provider else None,
            "trust_provider_configuration_hash": (
                provider.configuration_hash if provider else None
            ),
            "manifest_hash": (
                manifest.manifest_hash
                if manifest and manifest.manifest_hash
                else None
            ),
            "revocation_snapshot_hash": (
                revocation_snapshot.snapshot_hash
                if revocation_snapshot and revocation_snapshot.snapshot_hash
                else None
            ),
            "host_attestation_hashes": {
                item.role: item.attestation_hash
                for item in host_attestations
                if item.attestation_hash
            },
            "authority_artifact_hashes": dict(
                authority_artifact_hashes
            ),
            "assessment_mode": assessment_mode,
            "anchor_protocol_status": (
                "NOT_RUN"
                if assessment_mode == "rehearsal"
                else (
                    "PASS"
                    if all(
                        passed
                        for check_id, passed in checks.items()
                        if check_id != "deployment_anchor_current"
                    )
                    else "FAIL"
                )
            ),
            "anchor_status": (
                "ANCHORED_CURRENT"
                if qualified
                else "UNANCHORED_REHEARSAL"
            ),
            "status": status,
            "reason_codes": sorted(set(reasons)),
            "checks": checks,
            "anchored_predictive_qualification_granted": qualified,
            "anchored_scientific_qualification_granted": qualified,
            "claim_ceiling": (
                "externally_qualified_predictive_evidence"
                if qualified
                else "workflow_integrity_only"
            ),
            "assessed_at": assessed_at,
        },
    )
    return _commit_authority_receipt(
        workspace=workspace,
        receipt=receipt,
        persist=persist,
    )


def assess_external_authority_v64(
    *,
    workspace: StageWorkspaceV50,
    contract: PredictiveExternalQualificationContractV63,
    v63_receipt: ExternalPredictiveQualificationReceiptV63,
    provider: DeploymentTrustProviderV64 | None = None,
    manifest: ExternalAuthorityManifestV64 | None = None,
    revocation_snapshot: ExternalAuthorityRevocationSnapshotV64 | None = None,
    host_attestations: Sequence[ExternalRoleHostAttestationV64] = (),
    assessment_mode: AuthorityAssessmentModeV64 = "rehearsal",
    assessed_at: datetime | None = None,
    _persist: bool = True,
) -> ExternalAuthorityQualificationReceiptV64:
    """Assess external authority without accepting any key mapping.

    ``assessment_mode`` defaults to ``rehearsal``.  Rehearsal can never grant
    qualification even when a caller supplies internally consistent keys.
    """

    if _persist and provider is not None:
        assessed_at = provider.trusted_assessment_time()
    else:
        assessed_at = assessed_at or _utc_now()
    _assert_aware(assessed_at, "assessed_at")
    checks = _base_checks()
    reasons: list[str] = []
    authority_artifact_hashes: dict[str, str] = {}
    v63_artifact_hash = _exact_v63_receipt_artifact_hash(
        workspace=workspace,
        receipt=v63_receipt,
    )
    if v63_artifact_hash:
        authority_artifact_hashes["v63_receipt"] = v63_artifact_hash
    try:
        if manifest is not None:
            authority_artifact_hashes["manifest"] = (
                _commit_or_replay_authority_artifact(
                    workspace=workspace,
                    kind=_MANIFEST_KIND,
                    model=manifest,
                    model_type=ExternalAuthorityManifestV64,
                    same_identity=lambda item: (
                        getattr(item, "qualification_id", None)
                        == manifest.qualification_id
                        and getattr(item, "task_id", None) == manifest.task_id
                    ),
                    persist=_persist and assessment_mode == "rehearsal",
                )
            )
        if revocation_snapshot is not None:
            authority_artifact_hashes["revocation_snapshot"] = (
                _commit_or_replay_authority_artifact(
                    workspace=workspace,
                    kind=_REVOCATION_KIND,
                    model=revocation_snapshot,
                    model_type=ExternalAuthorityRevocationSnapshotV64,
                    same_identity=lambda item: (
                        getattr(item, "trust_domain_id", None)
                        == revocation_snapshot.trust_domain_id
                        and getattr(item, "epoch", None)
                        == revocation_snapshot.epoch
                    ),
                    persist=_persist,
                )
            )
        for attestation in host_attestations:
            authority_artifact_hashes[
                f"host_attestation_{attestation.role}"
            ] = _commit_or_replay_authority_artifact(
                workspace=workspace,
                kind=_HOST_ATTESTATION_KIND,
                model=attestation,
                model_type=ExternalRoleHostAttestationV64,
                same_identity=lambda item, current=attestation: (
                    getattr(item, "qualification_id", None)
                    == current.qualification_id
                    and getattr(item, "task_id", None) == current.task_id
                    and getattr(item, "role", None) == current.role
                ),
                persist=_persist,
            )
    except ExternalAuthorityError:
        reasons.append("authority_input_ledger_invalid")

    if assessment_mode == "rehearsal":
        reasons.append("unanchored_rehearsal")
        return _make_receipt(
            workspace=workspace,
            contract=contract,
            v63_receipt=v63_receipt,
            assessed_at=assessed_at,
            assessment_mode=assessment_mode,
            provider=provider,
            manifest=manifest,
            revocation_snapshot=revocation_snapshot,
            host_attestations=host_attestations,
            checks=checks,
            reasons=reasons,
            status="NOT_RUN",
            replay_hash=None,
            authority_artifact_hashes=authority_artifact_hashes,
            persist=_persist,
        )

    if provider is None or manifest is None or revocation_snapshot is None:
        if provider is None:
            reasons.append("deployment_trust_provider_missing")
        if manifest is None:
            reasons.append("root_signed_manifest_missing")
        if revocation_snapshot is None:
            reasons.append("current_revocation_snapshot_missing")
        return _make_receipt(
            workspace=workspace,
            contract=contract,
            v63_receipt=v63_receipt,
            assessed_at=assessed_at,
            assessment_mode=assessment_mode,
            provider=provider,
            manifest=manifest,
            revocation_snapshot=revocation_snapshot,
            host_attestations=host_attestations,
            checks=checks,
            reasons=reasons,
            status="NOT_RUN",
            replay_hash=None,
            authority_artifact_hashes=authority_artifact_hashes,
            persist=_persist,
        )

    # V6.3 replay is deliberately the first trust-bearing operation.
    replay_hash: str | None = None
    try:
        contract.assert_sealed()
        checks["v63_contract_sealed"] = True
    except (AttributeError, TypeError, ValueError):
        reasons.append("v63_contract_invalid")
    try:
        v63_receipt.assert_sealed()
        checks["v63_receipt_sealed"] = True
    except (AttributeError, TypeError, ValueError):
        reasons.append("v63_receipt_invalid")
    checks["v63_exact_binding"] = bool(
        contract.contract_hash
        and v63_receipt.contract_hash == contract.contract_hash
        and v63_receipt.qualification_id == contract.qualification_id
        and v63_receipt.task_id == contract.task_id
        and v63_receipt.workspace_spec_hash == contract.workspace_spec_hash
        and v63_receipt.local_context_hash == contract.local_context_hash
    )
    if not checks["v63_exact_binding"]:
        reasons.append("v63_contract_receipt_binding_invalid")
    checks["v63_action_forbidden"] = (
        v63_receipt.real_world_action_authorized is False
    )
    if not checks["v63_action_forbidden"]:
        reasons.append("v63_action_authority_forbidden")
    checks["v63_status_qualified"] = bool(
        v63_receipt.status == "EXTERNALLY_QUALIFIED"
        and v63_receipt.predictive_qualification_granted
        and v63_receipt.scientific_qualification_granted
    )
    if not checks["v63_status_qualified"]:
        reasons.append("v63_external_qualification_absent")
    try:
        replay = provider.replay_v63(
            workspace=workspace,
            receipt=v63_receipt,
        )
        replay_hash_value = getattr(replay, "replay_hash", None)
        replay_hash = (
            replay_hash_value
            if isinstance(replay_hash_value, str)
            and len(replay_hash_value) == 64
            else None
        )
        checks["v63_replay_pass"] = bool(
            getattr(replay, "status", None) == "PASS"
            and getattr(replay, "checks", None)
            and all(getattr(replay, "checks").values())
            and getattr(replay, "receipt_hash", None)
            == v63_receipt.receipt_hash
        )
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        checks["v63_replay_pass"] = False
    if not checks["v63_replay_pass"]:
        reasons.append("v63_read_only_replay_failed")

    forecast_input: ExternalForecastInputV63 | None = None
    custody: ExternalEvidenceCustodyV63 | None = None
    try:
        forecast_input = ExternalForecastInputV63.model_validate(
            workspace._artifact_payload_by_hash(
                v63_receipt.authority_artifact_hashes["forecast_input"]
            )
        )
        custody = ExternalEvidenceCustodyV63.model_validate(
            workspace._artifact_payload_by_hash(
                v63_receipt.authority_artifact_hashes["custody"]
            )
        )
        committed_binding = CurrentModelPredictionBindingV63.model_validate(
            workspace._artifact_payload_by_hash(
                v63_receipt.authority_artifact_hashes["prediction_binding"]
            )
        )
        committed_vector = ExternalPredictionVectorV63.model_validate(
            workspace._artifact_payload_by_hash(
                v63_receipt.authority_artifact_hashes["prediction_vector"]
            )
        )
        current_model_replay = (
            verify_current_model_external_prediction_v63(
                workspace=workspace,
                contract=contract,
                forecast_input=forecast_input,
                custody=custody,
            )
        )
        checks["current_model_prediction_recomputed"] = bool(
            current_model_replay.forecast_input == forecast_input
            and current_model_replay.prediction_vector == committed_vector
            and current_model_replay.binding == committed_binding
            and current_model_replay.prediction_vector.vector_hash
            == committed_vector.vector_hash
            and current_model_replay.binding.binding_hash
            == committed_binding.binding_hash
        )
    except (
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        checks["current_model_prediction_recomputed"] = False
    if not checks["current_model_prediction_recomputed"]:
        reasons.append("current_model_prediction_recomputation_failed")

    checks["trust_approval_frozen_before_external_evidence"] = bool(
        forecast_input is not None
        and custody is not None
        and contract.frozen_at
        <= manifest.issued_at
        <= forecast_input.frozen_at
        <= custody.attested_at
    )
    if not checks["trust_approval_frozen_before_external_evidence"]:
        reasons.append("trust_approval_was_not_frozen_before_evidence")
    manifest_sequence = _artifact_commit_sequence(
        workspace,
        authority_artifact_hashes.get("manifest"),
    )
    forecast_sequence = _artifact_commit_sequence(
        workspace,
        v63_receipt.authority_artifact_hashes.get("forecast_input"),
    )
    custody_sequence = _artifact_commit_sequence(
        workspace,
        v63_receipt.authority_artifact_hashes.get("custody"),
    )
    checks["manifest_transparency_precedes_forecast_input"] = bool(
        manifest_sequence is not None
        and forecast_sequence is not None
        and custody_sequence is not None
        and manifest_sequence < forecast_sequence < custody_sequence
    )
    if not checks["manifest_transparency_precedes_forecast_input"]:
        reasons.append("manifest_was_not_precommitted_before_forecast")

    checks["manifest_signature_current"] = (
        provider.verify_manifest_signature(manifest)
        and manifest.valid_from <= assessed_at <= manifest.valid_until
    )
    if not checks["manifest_signature_current"]:
        reasons.append("root_signed_manifest_invalid_or_expired")
    checks["manifest_exact_contract"] = bool(
        manifest.v63_contract_hash == contract.contract_hash
        and manifest.qualification_id == contract.qualification_id
        and manifest.task_id == contract.task_id
        and manifest.coordinator_identity.v63_host_id
        == contract.coordinator_host_id
        and manifest.generator_identity.v63_host_id
        == contract.generator_host_id
    )
    if not checks["manifest_exact_contract"]:
        reasons.append("manifest_contract_binding_invalid")

    role_keys_current = True
    for role in _AUTHORITY_ROLES:
        key_id = contract.trusted_authority_key_ids.get(role)
        contract_fingerprint = (
            contract.trusted_authority_key_fingerprints.get(role)
        )
        manifest_fingerprint = manifest.authority_key_fingerprints.get(role)
        role_keys_current = bool(
            role_keys_current
            and key_id
            and contract_fingerprint
            and manifest_fingerprint == contract_fingerprint
            and provider.authority_key_matches(
                key_id=key_id,
                fingerprint=manifest_fingerprint,
            )
        )
    role_keys_current = bool(
        role_keys_current
        and provider.attester_key_matches(
            key_id=manifest.host_attester_key_id,
            fingerprint=manifest.host_attester_key_fingerprint,
        )
    )
    checks["manifest_role_keys_current"] = role_keys_current
    if not role_keys_current:
        reasons.append("manifest_key_approval_invalid")

    checks["revocation_snapshot_current"] = (
        provider.accept_current_revocation_snapshot(
            snapshot=revocation_snapshot,
            assessed_at=assessed_at,
        )
    )
    if not checks["revocation_snapshot_current"]:
        reasons.append("revocation_snapshot_invalid_or_rolled_back")

    by_role: dict[str, ExternalRoleHostAttestationV64] = {}
    duplicate_roles = False
    for item in host_attestations:
        if item.role in by_role:
            duplicate_roles = True
        by_role[item.role] = item
    checks["host_attestation_set_complete"] = bool(
        not duplicate_roles
        and len(host_attestations) == 4
        and set(by_role) == set(_AUTHORITY_ROLES)
    )
    if not checks["host_attestation_set_complete"]:
        reasons.append("host_attestation_set_incomplete")

    all_attestation_signatures = checks["host_attestation_set_complete"]
    all_attestation_bindings = checks["host_attestation_set_complete"]
    role_payloads: dict[str, Mapping[str, object]] = {}
    for role, item in by_role.items():
        signature_valid = bool(
            provider.verify_host_attestation_signature(item)
            and item.attester_key_id == manifest.host_attester_key_id
            and item.attested_at >= manifest.valid_from
            and item.attested_at <= assessed_at <= item.valid_until
            and item.valid_until <= manifest.valid_until
        )
        all_attestation_signatures = bool(
            all_attestation_signatures and signature_valid
        )

        artifact_role, host_field = _ROLE_ARTIFACTS[role]
        expected_evidence_hash = v63_receipt.authority_artifact_hashes.get(
            artifact_role
        )
        try:
            payload = workspace._artifact_payload_by_hash(
                expected_evidence_hash
            )
        except (AttributeError, KeyError, OSError, TypeError, ValueError):
            payload = None
        if isinstance(payload, Mapping):
            role_payloads[role] = payload
        expected_host_id = (
            payload.get(host_field) if isinstance(payload, Mapping) else None
        )
        binding_valid = bool(
            item.manifest_hash == manifest.manifest_hash
            and item.trust_domain_id == manifest.trust_domain_id
            and item.v63_contract_hash == contract.contract_hash
            and item.qualification_id == contract.qualification_id
            and item.task_id == contract.task_id
            and item.role_key_fingerprint
            == manifest.authority_key_fingerprints.get(role)
            and item.v63_role_evidence_hash == expected_evidence_hash
            and item.v63_declared_host_id == expected_host_id
            and item.real_world_action_authorized is False
        )
        all_attestation_bindings = bool(
            all_attestation_bindings and binding_valid
        )
    checks["host_attestation_signatures_current"] = (
        all_attestation_signatures
    )
    checks["host_attestation_bindings_exact"] = all_attestation_bindings
    if not all_attestation_signatures:
        reasons.append("host_attestation_signature_or_time_invalid")
    if not all_attestation_bindings:
        reasons.append("host_attestation_binding_invalid")

    promotion_payload = role_payloads.get("promotion")
    promotion_decided_at: datetime | None = None
    if promotion_payload is not None:
        raw_decided_at = promotion_payload.get("decided_at")
        if isinstance(raw_decided_at, datetime):
            promotion_decided_at = raw_decided_at
        elif isinstance(raw_decided_at, str):
            try:
                promotion_decided_at = datetime.fromisoformat(raw_decided_at)
            except ValueError:
                promotion_decided_at = None
    checks["revocation_assessed_after_v63_finality"] = bool(
        promotion_decided_at is not None
        and promotion_decided_at.utcoffset() is not None
        and promotion_decided_at
        <= revocation_snapshot.effective_at
        <= assessed_at
    )
    if not checks["revocation_assessed_after_v63_finality"]:
        reasons.append("revocation_was_not_assessed_after_v63_finality")

    role_identities = [
        (
            item.host_identity_commitment,
            item.execution_boundary,
            item.management_domain,
        )
        for item in by_role.values()
    ]
    checks["external_role_isolation"] = bool(
        len(role_identities) == 4
        and all(
            len({identity[index] for identity in role_identities}) == 4
            for index in range(3)
        )
    )
    if not checks["external_role_isolation"]:
        reasons.append("external_authority_roles_not_isolated")

    local_identities = [
        (
            manifest.coordinator_identity.host_identity_commitment,
            manifest.coordinator_identity.execution_boundary,
            manifest.coordinator_identity.management_domain,
        ),
        (
            manifest.generator_identity.host_identity_commitment,
            manifest.generator_identity.execution_boundary,
            manifest.generator_identity.management_domain,
        ),
    ]
    all_identities = [*local_identities, *role_identities]
    checks["coordinator_generator_isolation"] = bool(
        len(all_identities) == 6
        and all(
            len({identity[index] for identity in all_identities}) == 6
            for index in range(3)
        )
    )
    if not checks["coordinator_generator_isolation"]:
        reasons.append("coordinator_generator_authority_isolation_invalid")

    revoked_keys = set(revocation_snapshot.revoked_key_fingerprints)
    revoked_hosts = set(
        revocation_snapshot.revoked_host_identity_commitments
    )
    material_fingerprints = {
        *manifest.authority_key_fingerprints.values(),
        manifest.host_attester_key_fingerprint,
    }
    manifest_root_fingerprint = provider.root_key_fingerprint(
        manifest.root_key_id
    )
    if manifest_root_fingerprint:
        material_fingerprints.add(manifest_root_fingerprint)
    material_hosts = {
        manifest.coordinator_identity.host_identity_commitment,
        manifest.generator_identity.host_identity_commitment,
        *(
            item.host_identity_commitment
            for item in host_attestations
        ),
    }
    checks["no_revoked_authority_material"] = bool(
        manifest.manifest_hash
        and manifest.manifest_hash
        not in revocation_snapshot.revoked_manifest_hashes
        and not (material_fingerprints & revoked_keys)
        and not (material_hosts & revoked_hosts)
    )
    if not checks["no_revoked_authority_material"]:
        reasons.append("authority_material_revoked")

    # This repository has no immutable OS/KMS/HSM deployment-root pin.
    # Caller-created providers are therefore protocol rehearsals only.
    checks["deployment_anchor_current"] = False
    protocol_ready = all(
        passed
        for check_id, passed in checks.items()
        if check_id != "deployment_anchor_current"
    )
    reasons.append("external_deployment_anchor_not_installed")
    return _make_receipt(
        workspace=workspace,
        contract=contract,
        v63_receipt=v63_receipt,
        assessed_at=assessed_at,
        assessment_mode=assessment_mode,
        provider=provider,
        manifest=manifest,
        revocation_snapshot=revocation_snapshot,
        host_attestations=host_attestations,
        checks=checks,
        reasons=reasons,
        status="NOT_RUN" if protocol_ready else "REJECTED",
        replay_hash=replay_hash,
        authority_artifact_hashes=authority_artifact_hashes,
        persist=_persist,
    )


class ExternalAuthorityReplayV64(StrictModel):
    """Pure read-only replay result for one committed V6.4 receipt."""

    schema_version: Literal["6.4-external-authority-replay"] = (
        "6.4-external-authority-replay"
    )
    receipt_hash: Sha256 | None
    status: Literal["PASS", "FAIL"]
    checks: dict[Identifier, bool]
    reason_codes: list[Identifier]
    real_world_action_authorized: Literal[False] = False
    replay_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_replay(self) -> "ExternalAuthorityReplayV64":
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("V6.4 replay reasons must be sorted and unique")
        expected = bool(self.checks) and all(self.checks.values())
        if (self.status == "PASS") != expected:
            raise ValueError("V6.4 replay status differs from checks")
        if self.replay_hash and self.replay_hash != self.content_hash():
            raise ValueError("V6.4 replay hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "replay_hash")

    @classmethod
    def seal(cls, **data: object) -> "ExternalAuthorityReplayV64":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"replay_hash"})
        payload["replay_hash"] = draft.content_hash()
        return cls(**payload)


def _load_exact_artifact(
    *,
    workspace: StageWorkspaceV50,
    artifact_hash: str,
    kind: str,
    model_type: type[StrictModel],
) -> StrictModel:
    model = model_type.model_validate(
        workspace._artifact_payload_by_hash(artifact_hash)
    )
    exact = [
        reference.sha256
        for reference, item in workspace._artifacts_of_kind(
            kind,
            model_type,
        )
        if reference.sha256 == artifact_hash and item == model
    ]
    if exact != [artifact_hash]:
        raise ExternalAuthorityError(f"{kind} artifact binding differs")
    return model


def verify_external_authority_v64(
    *,
    workspace: StageWorkspaceV50,
    receipt: ExternalAuthorityQualificationReceiptV64,
    provider: DeploymentTrustProviderV64,
) -> ExternalAuthorityReplayV64:
    """Reload exact authority artifacts and recompute without repair."""

    checks: dict[str, bool] = {
        "workspace_verified_before": False,
        "receipt_self_hash": False,
        "receipt_workspace_authority": False,
        "receipt_committed": False,
        "authority_artifacts_reload": False,
        "assessment_recomputed": False,
        "workspace_verified_after": False,
    }
    reasons: list[str] = []
    try:
        checks["workspace_verified_before"] = bool(workspace.verify())
        receipt.assert_sealed()
        checks["receipt_self_hash"] = True
        checks["receipt_workspace_authority"] = bool(
            receipt.authority_key_id == workspace.authority_key_id
            and receipt.authority_auth_tag
            and workspace._verify_mac(
                _AUTHORITY_RECEIPT_KIND,
                receipt.unsigned_hash(),
                receipt.authority_auth_tag,
            )
        )
        exact_receipts = [
            reference.sha256
            for reference, item in workspace._artifacts_of_kind(
                _AUTHORITY_RECEIPT_KIND,
                ExternalAuthorityQualificationReceiptV64,
            )
            if item == receipt
        ]
        checks["receipt_committed"] = len(exact_receipts) == 1

        v63_receipt = _load_exact_artifact(
            workspace=workspace,
            artifact_hash=receipt.authority_artifact_hashes["v63_receipt"],
            kind="external_predictive_qualification_v63",
            model_type=ExternalPredictiveQualificationReceiptV63,
        )
        if not isinstance(
            v63_receipt,
            ExternalPredictiveQualificationReceiptV63,
        ):
            raise ExternalAuthorityError("V6.3 receipt type differs")
        contract = _load_exact_artifact(
            workspace=workspace,
            artifact_hash=v63_receipt.authority_artifact_hashes["contract"],
            kind="predictive_external_qualification_contract_v63",
            model_type=PredictiveExternalQualificationContractV63,
        )
        if not isinstance(
            contract,
            PredictiveExternalQualificationContractV63,
        ):
            raise ExternalAuthorityError("V6.3 contract type differs")

        manifest: ExternalAuthorityManifestV64 | None = None
        if "manifest" in receipt.authority_artifact_hashes:
            loaded_manifest = _load_exact_artifact(
                workspace=workspace,
                artifact_hash=receipt.authority_artifact_hashes["manifest"],
                kind=_MANIFEST_KIND,
                model_type=ExternalAuthorityManifestV64,
            )
            if not isinstance(loaded_manifest, ExternalAuthorityManifestV64):
                raise ExternalAuthorityError("manifest type differs")
            manifest = loaded_manifest
        revocations: ExternalAuthorityRevocationSnapshotV64 | None = None
        if "revocation_snapshot" in receipt.authority_artifact_hashes:
            loaded_revocations = _load_exact_artifact(
                workspace=workspace,
                artifact_hash=receipt.authority_artifact_hashes[
                    "revocation_snapshot"
                ],
                kind=_REVOCATION_KIND,
                model_type=ExternalAuthorityRevocationSnapshotV64,
            )
            if not isinstance(
                loaded_revocations,
                ExternalAuthorityRevocationSnapshotV64,
            ):
                raise ExternalAuthorityError("revocation type differs")
            revocations = loaded_revocations
        attestations: list[ExternalRoleHostAttestationV64] = []
        for role in _AUTHORITY_ROLES:
            mapping_role = f"host_attestation_{role}"
            if mapping_role not in receipt.authority_artifact_hashes:
                continue
            loaded_attestation = _load_exact_artifact(
                workspace=workspace,
                artifact_hash=receipt.authority_artifact_hashes[mapping_role],
                kind=_HOST_ATTESTATION_KIND,
                model_type=ExternalRoleHostAttestationV64,
            )
            if not isinstance(
                loaded_attestation,
                ExternalRoleHostAttestationV64,
            ):
                raise ExternalAuthorityError("host attestation type differs")
            attestations.append(loaded_attestation)
        checks["authority_artifacts_reload"] = True

        recomputed = assess_external_authority_v64(
            workspace=workspace,
            contract=contract,
            v63_receipt=v63_receipt,
            provider=provider,
            manifest=manifest,
            revocation_snapshot=revocations,
            host_attestations=attestations,
            assessment_mode=receipt.assessment_mode,
            assessed_at=receipt.assessed_at,
            _persist=False,
        )
        checks["assessment_recomputed"] = recomputed == receipt
        checks["workspace_verified_after"] = bool(workspace.verify())
        if recomputed != receipt:
            reasons.append("authority_assessment_recomputation_differs")
    except (
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        reasons.extend(
            ["external_authority_replay_failed", type(exc).__name__.lower()]
        )
    for check_id, passed in checks.items():
        if not passed:
            reasons.append(check_id)
    reasons = sorted(set(reasons))
    return ExternalAuthorityReplayV64.seal(
        receipt_hash=receipt.receipt_hash,
        status="PASS" if all(checks.values()) else "FAIL",
        checks=checks,
        reason_codes=reasons,
    )


__all__ = [
    "AuthorityAnchorStatusV64",
    "AuthorityAssessmentModeV64",
    "AuthorityQualificationStatusV64",
    "DeploymentIdentityV64",
    "DeploymentTrustProviderV64",
    "ExternalAuthorityError",
    "ExternalAuthorityManifestV64",
    "ExternalAuthorityQualificationReceiptV64",
    "ExternalAuthorityReplayV64",
    "ExternalAuthorityRevocationSnapshotV64",
    "ExternalRoleHostAttestationV64",
    "assess_external_authority_v64",
    "sign_external_authority_manifest_v64",
    "sign_external_authority_revocations_v64",
    "sign_external_role_host_attestation_v64",
    "verify_external_authority_v64",
]
