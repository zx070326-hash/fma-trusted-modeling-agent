"""Authenticated V6.7 write-ahead contracts for pre-data materialization.

The three human-readable pre-data documents are projections, not authority.
This additive module defines a deterministic intent that is sufficient to
recreate those exact projections and a completion that binds the intent to the
existing ``predata_preparation_v67`` evidence artifact.  It performs no file
writes and grants neither scientific qualification nor real-world authority.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Literal, Mapping, TypeAlias

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256

from .measurement_study_design import (
    MEASUREMENT_STUDY_DESIGN_PATH_V67,
    MeasurementStudyDesignContractV67,
)
from .predata_protocol import (
    PREDATA_EXECUTION_PROTOCOL_PATH_V67,
    PreDataExecutionProtocolV67,
)
from .public_source import SOURCE_CONTRACT_PATH, WorldBankSourceContractV62


PREDATA_PREPARATION_INTENT_KIND_V67 = "predata_preparation_intent_v67"
PREDATA_PREPARATION_COMPLETION_KIND_V67 = "predata_preparation_completion_v67"
PREDATA_PREPARATION_EVIDENCE_KIND_V67 = "predata_preparation_v67"
PREDATA_TRANSACTION_POLICY_KIND_V67 = "predata_transaction_policy_v67"

PREDATA_CONTRACT_PATHS_V67: tuple[str, str, str] = (
    SOURCE_CONTRACT_PATH,
    MEASUREMENT_STUDY_DESIGN_PATH_V67,
    PREDATA_EXECUTION_PROTOCOL_PATH_V67,
)

PreDataContractBundleV67: TypeAlias = tuple[
    WorldBankSourceContractV62,
    MeasurementStudyDesignContractV67,
    PreDataExecutionProtocolV67,
]


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _assert_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")


def _assert_bundle_v67(
    source_contract: WorldBankSourceContractV62,
    measurement_contract: MeasurementStudyDesignContractV67,
    predata_protocol: PreDataExecutionProtocolV67,
    *,
    workspace_spec_hash: str | None = None,
    s0_gate_hash: str | None = None,
) -> None:
    source_contract.assert_sealed()
    measurement_contract.assert_sealed()
    predata_protocol.assert_sealed()
    if (
        measurement_contract.source_contract_id != source_contract.contract_id
        or measurement_contract.source_contract_hash != source_contract.contract_hash
        or predata_protocol.source_contract_id != source_contract.contract_id
        or predata_protocol.source_contract_hash != source_contract.contract_hash
        or predata_protocol.measurement_contract_id != measurement_contract.contract_id
        or predata_protocol.measurement_contract_hash
        != measurement_contract.contract_hash
    ):
        raise ValueError("V6.7 pre-data transaction bundle bindings differ")
    if workspace_spec_hash is not None and (
        measurement_contract.workspace_spec_hash != workspace_spec_hash
        or predata_protocol.workspace_spec_hash != workspace_spec_hash
    ):
        raise ValueError("V6.7 pre-data transaction bundle uses another workspace")
    if s0_gate_hash is not None and (
        measurement_contract.s0_gate_hash != s0_gate_hash
        or predata_protocol.s0_gate_hash != s0_gate_hash
    ):
        raise ValueError("V6.7 pre-data transaction bundle uses another S0 gate")


def _json_projection_bytes_v67(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def predata_contract_file_bytes_v67(
    source_contract: WorldBankSourceContractV62,
    measurement_contract: MeasurementStudyDesignContractV67,
    predata_protocol: PreDataExecutionProtocolV67,
) -> dict[str, bytes]:
    """Return the exact UTF-8 projections used by the V6.7 Studio writer."""

    _assert_bundle_v67(
        source_contract,
        measurement_contract,
        predata_protocol,
    )
    models: PreDataContractBundleV67 = (
        source_contract,
        measurement_contract,
        predata_protocol,
    )
    return {
        relative_path: _json_projection_bytes_v67(model.model_dump(mode="json"))
        for relative_path, model in zip(
            PREDATA_CONTRACT_PATHS_V67,
            models,
            strict=True,
        )
    }


def predata_contract_file_hashes_v67(
    source_contract: WorldBankSourceContractV62,
    measurement_contract: MeasurementStudyDesignContractV67,
    predata_protocol: PreDataExecutionProtocolV67,
) -> dict[str, str]:
    """Hash all and only the three deterministic V6.7 projection files."""

    return {
        relative_path: hashlib.sha256(payload).hexdigest()
        for relative_path, payload in predata_contract_file_bytes_v67(
            source_contract,
            measurement_contract,
            predata_protocol,
        ).items()
    }


def _validated_file_hashes_v67(
    artifact_file_hashes: Mapping[str, str],
    *,
    source_contract: WorldBankSourceContractV62 | None = None,
    measurement_contract: MeasurementStudyDesignContractV67 | None = None,
    predata_protocol: PreDataExecutionProtocolV67 | None = None,
) -> dict[str, str]:
    if set(artifact_file_hashes) != set(PREDATA_CONTRACT_PATHS_V67):
        raise ValueError(
            "V6.7 pre-data transaction must bind exactly three contract paths"
        )
    normalized = {
        relative_path: str(artifact_file_hashes[relative_path])
        for relative_path in PREDATA_CONTRACT_PATHS_V67
    }
    if any(
        len(digest) != 64
        or digest.lower() != digest
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in normalized.values()
    ):
        raise ValueError("V6.7 pre-data transaction contains an invalid file hash")
    supplied_bundle = (
        source_contract,
        measurement_contract,
        predata_protocol,
    )
    if any(item is not None for item in supplied_bundle):
        if any(item is None for item in supplied_bundle):
            raise ValueError(
                "V6.7 file-hash replay requires the complete contract bundle"
            )
        assert source_contract is not None
        assert measurement_contract is not None
        assert predata_protocol is not None
        expected = predata_contract_file_hashes_v67(
            source_contract,
            measurement_contract,
            predata_protocol,
        )
        if normalized != expected:
            raise ValueError(
                "V6.7 pre-data transaction file hashes differ from its bundle"
            )
    return normalized


def predata_preparation_payload_v67(
    *,
    workspace_spec_hash: str,
    s0_gate_hash: str,
    source_contract: WorldBankSourceContractV62,
    measurement_contract: MeasurementStudyDesignContractV67,
    predata_protocol: PreDataExecutionProtocolV67,
    artifact_file_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build the unchanged V6.7 preparation-evidence payload."""

    _assert_bundle_v67(
        source_contract,
        measurement_contract,
        predata_protocol,
        workspace_spec_hash=workspace_spec_hash,
        s0_gate_hash=s0_gate_hash,
    )
    supplied_file_hashes = (
        predata_contract_file_hashes_v67(
            source_contract,
            measurement_contract,
            predata_protocol,
        )
        if artifact_file_hashes is None
        else artifact_file_hashes
    )
    file_hashes = _validated_file_hashes_v67(
        supplied_file_hashes,
        source_contract=source_contract,
        measurement_contract=measurement_contract,
        predata_protocol=predata_protocol,
    )
    return {
        "schema_version": "6.7-predata-preparation-evidence",
        "workspace_spec_hash": workspace_spec_hash,
        "s0_gate_hash": s0_gate_hash,
        "source_contract_id": source_contract.contract_id,
        "source_contract_hash": source_contract.contract_hash,
        "measurement_contract_id": measurement_contract.contract_id,
        "measurement_contract_hash": measurement_contract.contract_hash,
        "protocol_id": predata_protocol.protocol_id,
        "protocol_hash": predata_protocol.protocol_hash,
        "capability_pack_id": predata_protocol.adapter_binding.adapter_id,
        "capability_pack_hash": (predata_protocol.adapter_binding.capability_pack_hash),
        "artifact_file_hashes": file_hashes,
        "network_accessed": False,
        "observation_values_accessed": False,
        "observed_statistics_accessed": False,
        "private_acceptance_data_accessed": False,
        "scientific_qualification_granted": False,
        "real_world_action_authorized": False,
    }


def predata_preparation_payload_hash_v67(
    *,
    workspace_spec_hash: str,
    s0_gate_hash: str,
    source_contract: WorldBankSourceContractV62,
    measurement_contract: MeasurementStudyDesignContractV67,
    predata_protocol: PreDataExecutionProtocolV67,
    artifact_file_hashes: Mapping[str, str] | None = None,
) -> str:
    """Hash the exact existing preparation-evidence payload."""

    return sha256_value(
        predata_preparation_payload_v67(
            workspace_spec_hash=workspace_spec_hash,
            s0_gate_hash=s0_gate_hash,
            source_contract=source_contract,
            measurement_contract=measurement_contract,
            predata_protocol=predata_protocol,
            artifact_file_hashes=artifact_file_hashes,
        )
    )


class PreDataTransactionPolicyV67(StrictModel):
    """Harness-authenticated authority policy for one V6.7 workspace.

    A current policy requires the intent/evidence/completion protocol.  The
    explicit false legacy flag prevents a caller from silently treating an
    older evidence-only record as authoritative for a newly governed
    transaction.
    """

    schema_version: Literal["6.7-predata-transaction-policy"] = (
        "6.7-predata-transaction-policy"
    )
    workspace_spec_hash: Sha256
    workflow_mode: Literal["v67"] = "v67"
    workflow_mode_contract_hash: Sha256
    workflow_mode_artifact_hash: Sha256
    evidence_scope: Literal["development", "public_data"]
    transaction_protocol: Literal["intent_evidence_completion_required"] = (
        "intent_evidence_completion_required"
    )
    legacy_completion_permitted: Literal[False] = False
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    policy_hash: Sha256 | None = None
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_policy(self) -> "PreDataTransactionPolicyV67":
        if self.authority_auth_tag and not self.policy_hash:
            raise ValueError(
                "authenticated V6.7 pre-data transaction policy requires policy_hash"
            )
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("V6.7 pre-data transaction policy hash differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "authority_auth_tag", "policy_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    def authenticate(self, authority_auth_tag: str) -> "PreDataTransactionPolicyV67":
        """Bind a harness-computed, domain-separated HMAC tag and seal."""

        if self.authority_auth_tag is not None or self.policy_hash is not None:
            raise ValueError("V6.7 pre-data transaction policy is already authenticated")
        payload = self.model_dump(mode="json")
        payload["authority_auth_tag"] = authority_auth_tag
        payload["policy_hash"] = sha256_value(
            {key: value for key, value in payload.items() if key != "policy_hash"}
        )
        return type(self).model_validate(payload)

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if (
            not self.authority_auth_tag
            or not self.policy_hash
            or self.policy_hash != self.content_hash()
        ):
            raise ValueError(
                "V6.7 pre-data transaction policy is not authenticated and sealed"
            )


class PreDataPreparationIntentV67(StrictModel):
    """Authenticated, deterministic intent committed before any projection."""

    schema_version: Literal["6.7-predata-preparation-intent"] = (
        "6.7-predata-preparation-intent"
    )
    workspace_spec_hash: Sha256
    s0_gate_hash: Sha256
    workflow_mode: Literal["v67"] = "v67"
    workflow_mode_contract_hash: Sha256
    workflow_mode_artifact_hash: Sha256
    evidence_scope: Literal["development", "public_data"]
    source_contract: WorldBankSourceContractV62
    measurement_contract: MeasurementStudyDesignContractV67
    predata_protocol: PreDataExecutionProtocolV67
    artifact_file_hashes: dict[str, Sha256] = Field(min_length=3, max_length=3)
    preparation_evidence_payload_hash: Sha256
    network_accessed: Literal[False] = False
    observation_values_accessed: Literal[False] = False
    observed_statistics_accessed: Literal[False] = False
    private_acceptance_data_accessed: Literal[False] = False
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    intent_hash: Sha256 | None = None
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_intent(self) -> "PreDataPreparationIntentV67":
        _assert_bundle_v67(
            self.source_contract,
            self.measurement_contract,
            self.predata_protocol,
            workspace_spec_hash=self.workspace_spec_hash,
            s0_gate_hash=self.s0_gate_hash,
        )
        file_hashes = _validated_file_hashes_v67(
            self.artifact_file_hashes,
            source_contract=self.source_contract,
            measurement_contract=self.measurement_contract,
            predata_protocol=self.predata_protocol,
        )
        if self.artifact_file_hashes != file_hashes:
            raise ValueError("V6.7 pre-data intent file hashes are not normalized")
        expected_payload_hash = predata_preparation_payload_hash_v67(
            workspace_spec_hash=self.workspace_spec_hash,
            s0_gate_hash=self.s0_gate_hash,
            source_contract=self.source_contract,
            measurement_contract=self.measurement_contract,
            predata_protocol=self.predata_protocol,
            artifact_file_hashes=file_hashes,
        )
        if self.preparation_evidence_payload_hash != expected_payload_hash:
            raise ValueError("V6.7 pre-data intent preparation payload hash differs")
        expected_scope = (
            "development" if self.source_contract.fixture_only else "public_data"
        )
        if self.evidence_scope != expected_scope:
            raise ValueError(
                "V6.7 pre-data intent evidence scope differs from its source"
            )
        if self.authority_auth_tag and not self.intent_hash:
            raise ValueError("authenticated V6.7 pre-data intent requires intent_hash")
        if self.intent_hash and self.intent_hash != self.content_hash():
            raise ValueError("V6.7 pre-data intent hash differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "authority_auth_tag", "intent_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "intent_hash")

    def authenticate(self, authority_auth_tag: str) -> "PreDataPreparationIntentV67":
        """Bind a harness-computed HMAC tag and seal the resulting envelope."""

        if self.authority_auth_tag is not None or self.intent_hash is not None:
            raise ValueError("V6.7 pre-data intent is already authenticated")
        payload = self.model_dump(mode="json")
        payload["authority_auth_tag"] = authority_auth_tag
        payload["intent_hash"] = sha256_value(
            {key: value for key, value in payload.items() if key != "intent_hash"}
        )
        return type(self).model_validate(payload)

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if (
            not self.authority_auth_tag
            or not self.intent_hash
            or self.intent_hash != self.content_hash()
        ):
            raise ValueError(
                "V6.7 pre-data preparation intent is not authenticated and sealed"
            )


class PreDataPreparationCompletionV67(StrictModel):
    """Authenticated completion binding an intent to preparation evidence."""

    schema_version: Literal["6.7-predata-preparation-completion"] = (
        "6.7-predata-preparation-completion"
    )
    workspace_spec_hash: Sha256
    s0_gate_hash: Sha256
    workflow_mode: Literal["v67"] = "v67"
    workflow_mode_contract_hash: Sha256
    workflow_mode_artifact_hash: Sha256
    evidence_scope: Literal["development", "public_data"]
    intent_hash: Sha256
    intent_artifact_hash: Sha256
    preparation_evidence_kind: Literal["predata_preparation_v67"] = (
        PREDATA_PREPARATION_EVIDENCE_KIND_V67
    )
    preparation_evidence_artifact_hash: Sha256
    preparation_evidence_payload_hash: Sha256
    source_contract_hash: Sha256
    measurement_contract_hash: Sha256
    protocol_hash: Sha256
    artifact_file_hashes: dict[str, Sha256] = Field(min_length=3, max_length=3)
    completed_at: datetime
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    completion_hash: Sha256 | None = None
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_completion(self) -> "PreDataPreparationCompletionV67":
        _assert_aware(self.completed_at, "completed_at")
        file_hashes = _validated_file_hashes_v67(self.artifact_file_hashes)
        if self.artifact_file_hashes != file_hashes:
            raise ValueError("V6.7 pre-data completion file hashes are not normalized")
        if self.authority_auth_tag and not self.completion_hash:
            raise ValueError(
                "authenticated V6.7 pre-data completion requires completion_hash"
            )
        if self.completion_hash and self.completion_hash != self.content_hash():
            raise ValueError("V6.7 pre-data completion hash differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "authority_auth_tag", "completion_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "completion_hash")

    def authenticate(
        self, authority_auth_tag: str
    ) -> "PreDataPreparationCompletionV67":
        """Bind a harness-computed HMAC tag and seal the resulting envelope."""

        if self.authority_auth_tag is not None or self.completion_hash is not None:
            raise ValueError("V6.7 pre-data completion is already authenticated")
        payload = self.model_dump(mode="json")
        payload["authority_auth_tag"] = authority_auth_tag
        payload["completion_hash"] = sha256_value(
            {key: value for key, value in payload.items() if key != "completion_hash"}
        )
        return type(self).model_validate(payload)

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if (
            not self.authority_auth_tag
            or not self.completion_hash
            or self.completion_hash != self.content_hash()
        ):
            raise ValueError("V6.7 pre-data completion is not authenticated and sealed")


__all__ = [
    "PREDATA_CONTRACT_PATHS_V67",
    "PREDATA_PREPARATION_COMPLETION_KIND_V67",
    "PREDATA_PREPARATION_EVIDENCE_KIND_V67",
    "PREDATA_PREPARATION_INTENT_KIND_V67",
    "PREDATA_TRANSACTION_POLICY_KIND_V67",
    "PreDataContractBundleV67",
    "PreDataPreparationCompletionV67",
    "PreDataPreparationIntentV67",
    "PreDataTransactionPolicyV67",
    "predata_contract_file_bytes_v67",
    "predata_contract_file_hashes_v67",
    "predata_preparation_payload_hash_v67",
    "predata_preparation_payload_v67",
]
