from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from fma.hashing import sha256_value
from fma.studio.service import (
    StudioWorldBankDataRequestV62,
    build_world_bank_predata_bundle_v67,
)
from fma.v6.predata_transaction import (
    PREDATA_CONTRACT_PATHS_V67,
    PREDATA_PREPARATION_COMPLETION_KIND_V67,
    PREDATA_PREPARATION_EVIDENCE_KIND_V67,
    PREDATA_PREPARATION_INTENT_KIND_V67,
    PREDATA_TRANSACTION_POLICY_KIND_V67,
    PreDataPreparationCompletionV67,
    PreDataPreparationIntentV67,
    PreDataTransactionPolicyV67,
    predata_contract_file_bytes_v67,
    predata_contract_file_hashes_v67,
    predata_preparation_payload_hash_v67,
    predata_preparation_payload_v67,
)


AUTHORITY_KEY = b"v6-7-predata-transaction-test-authority"
AUTHORITY_KEY_ID = "predata-transaction-test-key"
WORKSPACE_HASH = "1" * 64
S0_GATE_HASH = "2" * 64
WORKFLOW_MODE_CONTRACT_HASH = "3" * 64
WORKFLOW_MODE_ARTIFACT_HASH = "4" * 64


def _authority_tag(kind: str, unsigned_hash: str) -> str:
    return hmac.new(
        AUTHORITY_KEY,
        f"fma-v5:{kind}:{unsigned_hash}".encode(),
        hashlib.sha256,
    ).hexdigest()


def _bundle():
    request = StudioWorldBankDataRequestV62(
        adapter_id="scalar_autonomous_ode_v52",
        contract_id="predata-transaction-source",
        country_code="BRA",
        indicator_id="NY.GDP.PCAP.CD",
        start_year=1990,
        end_year=2024,
        minimum_observations=23,
        state_unit="current_usd_per_person",
        attribution="World Bank World Development Indicators, CC BY 4.0.",
        semantic_name="annual GDP per capita",
        operational_definition=(
            "World Bank indicator NY.GDP.PCAP.CD for Brazil in current US dollars."
        ),
        observation_time_basis="calendar year",
        aggregation_level="national annual aggregate",
        fixture_only=False,
    )
    return build_world_bank_predata_bundle_v67(
        request=request,
        workspace_spec_hash=WORKSPACE_HASH,
        s0_gate_hash=S0_GATE_HASH,
    )


def _intent_draft() -> PreDataPreparationIntentV67:
    source, measurement, protocol = _bundle()
    file_hashes = predata_contract_file_hashes_v67(
        source,
        measurement,
        protocol,
    )
    return PreDataPreparationIntentV67(
        workspace_spec_hash=WORKSPACE_HASH,
        s0_gate_hash=S0_GATE_HASH,
        workflow_mode_contract_hash=WORKFLOW_MODE_CONTRACT_HASH,
        workflow_mode_artifact_hash=WORKFLOW_MODE_ARTIFACT_HASH,
        evidence_scope="public_data",
        source_contract=source,
        measurement_contract=measurement,
        predata_protocol=protocol,
        artifact_file_hashes=file_hashes,
        preparation_evidence_payload_hash=(
            predata_preparation_payload_hash_v67(
                workspace_spec_hash=WORKSPACE_HASH,
                s0_gate_hash=S0_GATE_HASH,
                source_contract=source,
                measurement_contract=measurement,
                predata_protocol=protocol,
                artifact_file_hashes=file_hashes,
            )
        ),
        authority_key_id=AUTHORITY_KEY_ID,
    )


def _policy_draft() -> PreDataTransactionPolicyV67:
    return PreDataTransactionPolicyV67(
        workspace_spec_hash=WORKSPACE_HASH,
        workflow_mode_contract_hash=WORKFLOW_MODE_CONTRACT_HASH,
        workflow_mode_artifact_hash=WORKFLOW_MODE_ARTIFACT_HASH,
        evidence_scope="public_data",
        authority_key_id=AUTHORITY_KEY_ID,
    )


def _sealed_intent() -> PreDataPreparationIntentV67:
    draft = _intent_draft()
    return draft.authenticate(
        _authority_tag(
            PREDATA_PREPARATION_INTENT_KIND_V67,
            draft.unsigned_hash(),
        )
    )


def _completion_draft(
    intent: PreDataPreparationIntentV67,
    *,
    completed_at: datetime | None = None,
) -> PreDataPreparationCompletionV67:
    assert intent.intent_hash is not None
    return PreDataPreparationCompletionV67(
        workspace_spec_hash=intent.workspace_spec_hash,
        s0_gate_hash=intent.s0_gate_hash,
        workflow_mode_contract_hash=intent.workflow_mode_contract_hash,
        workflow_mode_artifact_hash=intent.workflow_mode_artifact_hash,
        evidence_scope=intent.evidence_scope,
        intent_hash=intent.intent_hash,
        intent_artifact_hash=sha256_value(
            {
                "kind": PREDATA_PREPARATION_INTENT_KIND_V67,
                "intent_hash": intent.intent_hash,
            }
        ),
        preparation_evidence_artifact_hash=sha256_value(
            {
                "kind": PREDATA_PREPARATION_EVIDENCE_KIND_V67,
                "payload_hash": intent.preparation_evidence_payload_hash,
            }
        ),
        preparation_evidence_payload_hash=(intent.preparation_evidence_payload_hash),
        source_contract_hash=str(intent.source_contract.contract_hash),
        measurement_contract_hash=str(intent.measurement_contract.contract_hash),
        protocol_hash=str(intent.predata_protocol.protocol_hash),
        artifact_file_hashes=intent.artifact_file_hashes,
        completed_at=completed_at or datetime(2026, 7, 29, tzinfo=timezone.utc),
        authority_key_id=AUTHORITY_KEY_ID,
    )


def test_transaction_policy_is_authenticated_sealed_and_fail_closed() -> None:
    draft = _policy_draft()

    assert draft.schema_version == "6.7-predata-transaction-policy"
    assert draft.workflow_mode == "v67"
    assert draft.transaction_protocol == "intent_evidence_completion_required"
    assert draft.legacy_completion_permitted is False
    assert draft.scientific_qualification_granted is False
    assert draft.real_world_action_authorized is False
    with pytest.raises(ValueError, match="not authenticated and sealed"):
        draft.assert_sealed()

    sealed = draft.authenticate(
        _authority_tag(
            PREDATA_TRANSACTION_POLICY_KIND_V67,
            draft.unsigned_hash(),
        )
    )
    sealed.assert_sealed()
    assert sealed.unsigned_hash() == draft.unsigned_hash()
    assert sealed.policy_hash == sealed.content_hash()
    assert sealed.authority_auth_tag == _authority_tag(
        PREDATA_TRANSACTION_POLICY_KIND_V67,
        draft.unsigned_hash(),
    )
    with pytest.raises(ValueError, match="already authenticated"):
        sealed.authenticate("a" * 64)


def test_transaction_policy_rejects_tampering_and_authority_escalation() -> None:
    draft = _policy_draft()
    sealed = draft.authenticate(
        _authority_tag(
            PREDATA_TRANSACTION_POLICY_KIND_V67,
            draft.unsigned_hash(),
        )
    )

    for field, value in (
        ("workspace_spec_hash", "a" * 64),
        ("workflow_mode_contract_hash", "b" * 64),
        ("evidence_scope", "development"),
    ):
        tampered = sealed.model_dump(mode="json")
        tampered[field] = value
        with pytest.raises(ValidationError, match="policy hash differs"):
            PreDataTransactionPolicyV67.model_validate(tampered)

    for field, value in (
        ("legacy_completion_permitted", True),
        ("scientific_qualification_granted", True),
        ("real_world_action_authorized", True),
    ):
        escalated = draft.model_dump(mode="json")
        escalated[field] = value
        with pytest.raises(ValidationError):
            PreDataTransactionPolicyV67.model_validate(escalated)

    partial = draft.model_dump(mode="json")
    partial["authority_auth_tag"] = "a" * 64
    with pytest.raises(ValidationError, match="requires policy_hash"):
        PreDataTransactionPolicyV67.model_validate(partial)


def test_intent_deterministically_binds_exact_three_file_bundle() -> None:
    first = _intent_draft()
    second = _intent_draft()

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.unsigned_hash() == second.unsigned_hash()
    assert "created_at" not in type(first).model_fields
    assert tuple(first.artifact_file_hashes) == PREDATA_CONTRACT_PATHS_V67
    assert first.artifact_file_hashes == predata_contract_file_hashes_v67(
        first.source_contract,
        first.measurement_contract,
        first.predata_protocol,
    )
    with pytest.raises(ValueError, match="not authenticated and sealed"):
        first.assert_sealed()

    sealed = first.authenticate(
        _authority_tag(
            PREDATA_PREPARATION_INTENT_KIND_V67,
            first.unsigned_hash(),
        )
    )
    sealed.assert_sealed()
    assert sealed.unsigned_hash() == first.unsigned_hash()
    assert sealed.intent_hash == sealed.content_hash()
    assert sealed.scientific_qualification_granted is False
    assert sealed.real_world_action_authorized is False


def test_file_bytes_and_preparation_payload_match_existing_projection() -> None:
    source, measurement, protocol = _bundle()
    payloads = predata_contract_file_bytes_v67(
        source,
        measurement,
        protocol,
    )
    hashes = predata_contract_file_hashes_v67(
        source,
        measurement,
        protocol,
    )
    models = (source, measurement, protocol)
    assert tuple(payloads) == PREDATA_CONTRACT_PATHS_V67
    for relative_path, model in zip(
        PREDATA_CONTRACT_PATHS_V67,
        models,
        strict=True,
    ):
        serialized = payloads[relative_path]
        assert serialized.endswith(b"\n")
        assert json.loads(serialized) == model.model_dump(mode="json")
        assert hashlib.sha256(serialized).hexdigest() == hashes[relative_path]

    evidence = predata_preparation_payload_v67(
        workspace_spec_hash=WORKSPACE_HASH,
        s0_gate_hash=S0_GATE_HASH,
        source_contract=source,
        measurement_contract=measurement,
        predata_protocol=protocol,
        artifact_file_hashes=hashes,
    )
    assert evidence["schema_version"] == "6.7-predata-preparation-evidence"
    assert evidence["artifact_file_hashes"] == hashes
    assert evidence["network_accessed"] is False
    assert evidence["observation_values_accessed"] is False
    assert evidence["scientific_qualification_granted"] is False
    assert predata_preparation_payload_hash_v67(
        workspace_spec_hash=WORKSPACE_HASH,
        s0_gate_hash=S0_GATE_HASH,
        source_contract=source,
        measurement_contract=measurement,
        predata_protocol=protocol,
        artifact_file_hashes=hashes,
    ) == sha256_value(evidence)
    with pytest.raises(ValueError, match="exactly three contract paths"):
        predata_preparation_payload_v67(
            workspace_spec_hash=WORKSPACE_HASH,
            s0_gate_hash=S0_GATE_HASH,
            source_contract=source,
            measurement_contract=measurement,
            predata_protocol=protocol,
            artifact_file_hashes={},
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_path",
        "extra_path",
        "wrong_file_hash",
        "wrong_payload_hash",
    ],
)
def test_intent_rejects_inexact_projection_bindings(mutation: str) -> None:
    draft = _intent_draft()
    payload = draft.model_dump(mode="json")
    file_hashes = dict(payload["artifact_file_hashes"])
    if mutation == "missing_path":
        file_hashes.pop(PREDATA_CONTRACT_PATHS_V67[0])
    elif mutation == "extra_path":
        file_hashes["docs/undeclared.json"] = "a" * 64
    elif mutation == "wrong_file_hash":
        file_hashes[PREDATA_CONTRACT_PATHS_V67[0]] = "a" * 64
    else:
        payload["preparation_evidence_payload_hash"] = "a" * 64
    payload["artifact_file_hashes"] = file_hashes

    with pytest.raises(ValidationError):
        PreDataPreparationIntentV67.model_validate(payload)


def test_intent_rejects_unsealed_or_cross_bound_nested_contracts() -> None:
    draft = _intent_draft()
    payload = draft.model_dump(mode="json")
    source_payload = dict(payload["source_contract"])
    source_payload["contract_hash"] = "a" * 64
    payload["source_contract"] = source_payload

    with pytest.raises(ValidationError, match="source contract"):
        PreDataPreparationIntentV67.model_validate(payload)

    cross_bound = draft.model_dump(mode="json")
    measurement_payload = dict(cross_bound["measurement_contract"])
    measurement_payload["workspace_spec_hash"] = "b" * 64
    measurement_payload["contract_hash"] = None
    cross_bound["measurement_contract"] = measurement_payload
    with pytest.raises(ValidationError):
        PreDataPreparationIntentV67.model_validate(cross_bound)


def test_intent_rejects_scope_or_authority_escalation() -> None:
    draft = _intent_draft()
    wrong_scope = draft.model_dump(mode="json")
    wrong_scope["evidence_scope"] = "development"
    with pytest.raises(ValidationError, match="evidence scope"):
        PreDataPreparationIntentV67.model_validate(wrong_scope)

    escalated = draft.model_dump(mode="json")
    escalated["scientific_qualification_granted"] = True
    with pytest.raises(ValidationError):
        PreDataPreparationIntentV67.model_validate(escalated)


def test_completion_is_aware_authenticated_and_binds_both_artifacts() -> None:
    intent = _sealed_intent()
    draft = _completion_draft(intent)
    sealed = draft.authenticate(
        _authority_tag(
            PREDATA_PREPARATION_COMPLETION_KIND_V67,
            draft.unsigned_hash(),
        )
    )

    sealed.assert_sealed()
    assert sealed.intent_hash == intent.intent_hash
    assert len(sealed.intent_artifact_hash) == 64
    assert sealed.preparation_evidence_kind == (PREDATA_PREPARATION_EVIDENCE_KIND_V67)
    assert len(sealed.preparation_evidence_artifact_hash) == 64
    assert sealed.preparation_evidence_payload_hash == (
        intent.preparation_evidence_payload_hash
    )
    assert sealed.artifact_file_hashes == intent.artifact_file_hashes
    assert sealed.completion_hash == sealed.content_hash()
    assert sealed.scientific_qualification_granted is False
    assert sealed.real_world_action_authorized is False


def test_completion_rejects_naive_time_paths_and_hash_tampering() -> None:
    intent = _sealed_intent()
    with pytest.raises(ValidationError, match="timezone"):
        _completion_draft(
            intent,
            completed_at=datetime(2026, 7, 29),
        )

    draft = _completion_draft(intent)
    missing_path = draft.model_dump(mode="json")
    missing_hashes = dict(missing_path["artifact_file_hashes"])
    missing_hashes.pop(PREDATA_CONTRACT_PATHS_V67[0])
    missing_path["artifact_file_hashes"] = missing_hashes
    with pytest.raises(ValidationError):
        PreDataPreparationCompletionV67.model_validate(missing_path)

    sealed = draft.authenticate(
        _authority_tag(
            PREDATA_PREPARATION_COMPLETION_KIND_V67,
            draft.unsigned_hash(),
        )
    )
    tampered = sealed.model_dump(mode="json")
    tampered["preparation_evidence_artifact_hash"] = "a" * 64
    with pytest.raises(ValidationError, match="completion hash differs"):
        PreDataPreparationCompletionV67.model_validate(tampered)


def test_authentication_is_one_way_and_requires_hash_pair() -> None:
    intent = _intent_draft()
    sealed_intent = intent.authenticate(
        _authority_tag(
            PREDATA_PREPARATION_INTENT_KIND_V67,
            intent.unsigned_hash(),
        )
    )
    with pytest.raises(ValueError, match="already authenticated"):
        sealed_intent.authenticate("a" * 64)

    partial = intent.model_dump(mode="json")
    partial["authority_auth_tag"] = "a" * 64
    with pytest.raises(ValidationError, match="requires intent_hash"):
        PreDataPreparationIntentV67.model_validate(partial)

    completion = _completion_draft(sealed_intent)
    partial_completion = completion.model_dump(mode="json")
    partial_completion["authority_auth_tag"] = "a" * 64
    with pytest.raises(ValidationError, match="requires completion_hash"):
        PreDataPreparationCompletionV67.model_validate(partial_completion)
