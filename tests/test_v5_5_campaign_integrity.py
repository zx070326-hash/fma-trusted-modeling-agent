from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fma.hashing import canonical_json
from fma.v5.external_harness import PrivateTargetV50
from fma.v5_3.custody import PrivateScoreContractV53
from fma.v5_4.public_eligibility import PublicEligibilityContractV54
from fma.v5_5.campaign_protocol import (
    CandidateSelectionPolicyV55,
    ProspectiveCampaignProtocolV55,
    PublicEligibilitySettingsV55,
    materialize_public_launch_v55,
    verify_public_launch_binding_v55,
)
from fma.v5_5.closeout_authority import main as closeout_authority_main
from fma.v5_5.custodian_worker import main as custodian_worker_main
from fma.v5_5.protocol_materializer import main as protocol_materializer_main
from fma.v5_5.provenance_release_worker import main as provenance_release_main
from fma.v5_5.split_custody import (
    SourceProvenanceDraftV55,
    create_split_custody_envelopes_v55,
    open_private_target_envelope_v55,
    release_source_provenance_v55,
    sign_campaign_closeout_authorization_v55,
    verify_campaign_closeout_authorization_v55,
    verify_source_provenance_disclosure_v55,
    verify_split_custody_bindings_v55,
)


NOW = datetime(2026, 7, 26, 1, 2, 3, tzinfo=timezone.utc)
TARGET_KEY = bytes(range(32))
PROVENANCE_KEY = bytes(range(32, 64))


def _key_pair() -> tuple[bytes, bytes]:
    private = Ed25519PrivateKey.generate()
    private_pem = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def _protocol() -> ProspectiveCampaignProtocolV55:
    return ProspectiveCampaignProtocolV55.seal(
        protocol_id="i34-protocol",
        baseline_id="persistence_last_value",
        candidate_families=["constant", "exponential", "gompertz", "logistic"],
        maximum_candidate_search_count=16,
        public_eligibility=PublicEligibilitySettingsV55(
            expected_horizons=[1, 2, 3, 4],
            minimum_origin_count=12,
            contiguous_time_block_count=3,
            recent_origin_count=4,
            bootstrap_replicates=8192,
            bootstrap_block_length=4,
            multiplicity_correction_count=16,
            bootstrap_seed=5517,
        ),
        frozen_at=NOW,
    )


def _score_contract(
    protocol: ProspectiveCampaignProtocolV55,
) -> PrivateScoreContractV53:
    return PrivateScoreContractV53.seal(
        contract_id="i34-private-score",
        case_id="i34-case",
        protocol_hash=protocol.protocol_hash,
        public_case_hash="a" * 64,
        forecast_plan_hash="b" * 64,
        target_ids=["target-a", "target-b"],
        quality_scale=10.0,
        minimum_quality_score=0.5,
        frozen_at=NOW,
    )


def _source() -> SourceProvenanceDraftV55:
    return SourceProvenanceDraftV55(
        case_id="i34-case",
        source_authority="Example National Statistics Office",
        source_title="Official monthly series",
        source_locator="https://example.invalid/official-series",
        table_or_series_id="TABLE-7-SERIES-A",
        public_period_start="2010-01",
        public_period_end="2024-12",
        private_period_start="2025-01",
        private_period_end="2025-02",
        source_artifact_sha256="c" * 64,
        prior_campaign_exclusion_hashes=["d" * 64],
        retrieved_at=NOW,
    )


def _split():
    protocol = _protocol()
    score_contract = _score_contract(protocol)
    custody_private, custody_public = _key_pair()
    outputs = create_split_custody_envelopes_v55(
        protocol=protocol,
        score_contract=score_contract,
        private_targets=[
            PrivateTargetV50(target_id="target-a", value=12.5),
            PrivateTargetV50(target_id="target-b", value=13.5),
        ],
        source_provenance=_source(),
        private_target_envelope_id="i34-target-envelope",
        source_provenance_envelope_id="i34-provenance-envelope",
        private_target_key_id="i34-target-aes",
        private_target_key=TARGET_KEY,
        source_provenance_key_id="i34-provenance-aes",
        source_provenance_key=PROVENANCE_KEY,
        custodian_host_id="custodian-host",
        coordinator_host_id="coordinator-host",
        generator_host_id="generator-host",
        attestation_id="i34-split-custody",
        custody_key_id="i34-custody-signing",
        custody_private_key_pem=custody_private,
        private_target_nonce=b"\x01" * 12,
        source_provenance_nonce=b"\x02" * 12,
        private_target_canary="target-canary-" + "1" * 40,
        source_provenance_canary="provenance-canary-" + "2" * 40,
        attested_at=NOW,
    )
    return protocol, score_contract, custody_private, custody_public, outputs


def _write_json(path: Path, value: object) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8", newline="\n")


def test_protocol_materialization_has_one_exact_baseline_source() -> None:
    protocol = _protocol()
    policy, contract, binding = materialize_public_launch_v55(
        protocol=protocol,
        task_id="i34-case",
        eligibility_contract_id="i34-eligibility",
        materialized_at=NOW,
    )
    assert (
        protocol.baseline_id
        == policy.baseline_id
        == contract.baseline_id
        == binding.baseline_id
        == "persistence_last_value"
    )
    assert policy.policy_hash == contract.candidate_selection_rule_hash
    assert verify_public_launch_binding_v55(
        protocol=protocol,
        policy=policy,
        contract=contract,
        binding=binding,
    )
    assert "baseline" not in inspect.signature(materialize_public_launch_v55).parameters


def test_protocol_materialization_rejects_identifier_drift() -> None:
    protocol = _protocol()
    policy, contract, binding = materialize_public_launch_v55(
        protocol=protocol,
        task_id="i34-case",
        eligibility_contract_id="i34-eligibility",
        materialized_at=NOW,
    )
    policy_payload = policy.model_dump(exclude={"policy_hash"})
    policy_payload["baseline_id"] = "persistence-last-value"
    drifted_policy = CandidateSelectionPolicyV55.seal(**policy_payload)
    assert not verify_public_launch_binding_v55(
        protocol=protocol,
        policy=drifted_policy,
        contract=contract,
        binding=binding,
    )

    contract_payload = contract.model_dump(exclude={"contract_hash"})
    contract_payload["baseline_id"] = "persistence-last-value"
    contract_payload["candidate_selection_rule_hash"] = drifted_policy.policy_hash
    drifted_contract = PublicEligibilityContractV54.seal(**contract_payload)
    assert not verify_public_launch_binding_v55(
        protocol=protocol,
        policy=drifted_policy,
        contract=drifted_contract,
        binding=binding,
    )


def test_protocol_freeze_rejects_budget_correction_drift() -> None:
    with pytest.raises(ValueError, match="multiplicity correction"):
        ProspectiveCampaignProtocolV55.seal(
            protocol_id="bad-protocol",
            baseline_id="persistence_last_value",
            candidate_families=["constant", "exponential", "logistic"],
            maximum_candidate_search_count=16,
            public_eligibility=PublicEligibilitySettingsV55(
                expected_horizons=[1, 2],
                minimum_origin_count=12,
                contiguous_time_block_count=3,
                recent_origin_count=4,
                bootstrap_replicates=8192,
                bootstrap_block_length=4,
                multiplicity_correction_count=15,
            ),
            frozen_at=NOW,
        )


def test_split_custody_is_bound_and_public_artifacts_are_secret_free() -> None:
    protocol, score_contract, _, custody_public, outputs = _split()
    capsule, provenance, target_envelope, provenance_envelope, attestation = outputs
    assert target_envelope.key_id != provenance_envelope.key_id
    assert target_envelope.key_fingerprint != provenance_envelope.key_fingerprint
    assert verify_split_custody_bindings_v55(
        protocol=protocol,
        score_contract=score_contract,
        private_target_envelope=target_envelope,
        source_provenance_envelope=provenance_envelope,
        attestation=attestation,
        custody_public_key_pem=custody_public,
        expected_coordinator_host_id="coordinator-host",
        expected_generator_host_id="generator-host",
    )
    _, wrong_custody_public = _key_pair()
    assert not verify_split_custody_bindings_v55(
        protocol=protocol,
        score_contract=score_contract,
        private_target_envelope=target_envelope,
        source_provenance_envelope=provenance_envelope,
        attestation=attestation,
        custody_public_key_pem=wrong_custody_public,
        expected_coordinator_host_id="coordinator-host",
        expected_generator_host_id="generator-host",
    )

    public_bytes = canonical_json(
        {
            "target": target_envelope,
            "provenance": provenance_envelope,
            "attestation": attestation,
        }
    )
    for secret in (
        "Example National Statistics Office",
        "official-series",
        "TABLE-7-SERIES-A",
        "2010-01",
        "2025-02",
        "12.5",
        capsule.secrecy_canary,
        provenance.secrecy_canary,
    ):
        assert secret not in public_bytes


def test_split_custody_rejects_same_encryption_key_domain() -> None:
    protocol = _protocol()
    score_contract = _score_contract(protocol)
    custody_private, _ = _key_pair()
    with pytest.raises(ValueError, match="encryption keys must differ"):
        create_split_custody_envelopes_v55(
            protocol=protocol,
            score_contract=score_contract,
            private_targets=[
                PrivateTargetV50(target_id="target-a", value=12.5),
                PrivateTargetV50(target_id="target-b", value=13.5),
            ],
            source_provenance=_source(),
            private_target_envelope_id="i34-target-envelope",
            source_provenance_envelope_id="i34-provenance-envelope",
            private_target_key_id="i34-target-aes",
            private_target_key=TARGET_KEY,
            source_provenance_key_id="i34-provenance-aes",
            source_provenance_key=TARGET_KEY,
            custodian_host_id="custodian-host",
            coordinator_host_id="coordinator-host",
            generator_host_id="generator-host",
            attestation_id="i34-split-custody",
            custody_key_id="i34-custody-signing",
            custody_private_key_pem=custody_private,
        )


def test_provenance_release_is_closeout_authorized_and_target_blind() -> None:
    protocol, _, _, custody_public, outputs = _split()
    capsule, provenance, target_envelope, provenance_envelope, attestation = outputs
    closeout_private, closeout_public = _key_pair()
    authorization = sign_campaign_closeout_authorization_v55(
        protocol=protocol,
        attestation=attestation,
        terminal_status="ABSTAIN",
        terminal_evidence_hash="e" * 64,
        authorization_id="i34-closeout",
        closeout_authority_key_id="i34-closeout-key",
        closeout_authority_private_key_pem=closeout_private,
        authorized_at=NOW,
    )
    assert verify_campaign_closeout_authorization_v55(
        authorization=authorization,
        protocol=protocol,
        attestation=attestation,
        closeout_public_keys={"i34-closeout-key": closeout_public},
    )
    record, receipt = release_source_provenance_v55(
        protocol=protocol,
        source_provenance_envelope=provenance_envelope,
        attestation=attestation,
        authorization=authorization,
        terminal_evidence_hash="e" * 64,
        source_provenance_key=PROVENANCE_KEY,
        custody_public_key_pem=custody_public,
        closeout_public_keys={"i34-closeout-key": closeout_public},
        disclosed_at=NOW,
    )
    assert record == provenance
    assert receipt.source_provenance_status == "RELEASED"
    assert receipt.private_target_envelope_accessed is False
    assert receipt.private_target_key_accessed is False
    assert receipt.private_evaluation_performed is False
    assert verify_source_provenance_disclosure_v55(
        protocol=protocol,
        source_record=record,
        source_provenance_envelope=provenance_envelope,
        attestation=attestation,
        authorization=authorization,
        receipt=receipt,
        terminal_evidence_hash="e" * 64,
        custody_public_key_pem=custody_public,
        closeout_public_keys={"i34-closeout-key": closeout_public},
    )
    assert {"private_target_envelope", "private_target_key"}.isdisjoint(
        inspect.signature(release_source_provenance_v55).parameters
    )

    reopened = open_private_target_envelope_v55(
        private_target_envelope=target_envelope,
        private_target_key=TARGET_KEY,
    )
    assert reopened == capsule


def test_provenance_release_rejects_wrong_key_and_wrong_authority() -> None:
    protocol, _, custody_private, custody_public, outputs = _split()
    _, _, _, provenance_envelope, attestation = outputs
    closeout_private, closeout_public = _key_pair()
    authorization = sign_campaign_closeout_authorization_v55(
        protocol=protocol,
        attestation=attestation,
        terminal_status="TERMINATED",
        terminal_evidence_hash="e" * 64,
        authorization_id="i34-closeout",
        closeout_authority_key_id="i34-closeout-key",
        closeout_authority_private_key_pem=closeout_private,
        authorized_at=NOW,
    )
    with pytest.raises(ValueError, match="terminal campaign evidence hash differs"):
        release_source_provenance_v55(
            protocol=protocol,
            source_provenance_envelope=provenance_envelope,
            attestation=attestation,
            authorization=authorization,
            terminal_evidence_hash="f" * 64,
            source_provenance_key=PROVENANCE_KEY,
            custody_public_key_pem=custody_public,
            closeout_public_keys={"i34-closeout-key": closeout_public},
            disclosed_at=NOW,
        )
    _, wrong_closeout_public = _key_pair()
    with pytest.raises(ValueError, match="not authorized"):
        release_source_provenance_v55(
            protocol=protocol,
            source_provenance_envelope=provenance_envelope,
            attestation=attestation,
            authorization=authorization,
            terminal_evidence_hash="e" * 64,
            source_provenance_key=PROVENANCE_KEY,
            custody_public_key_pem=custody_public,
            closeout_public_keys={"i34-closeout-key": wrong_closeout_public},
            disclosed_at=NOW,
        )
    with pytest.raises(ValueError, match="fingerprint differs"):
        release_source_provenance_v55(
            protocol=protocol,
            source_provenance_envelope=provenance_envelope,
            attestation=attestation,
            authorization=authorization,
            terminal_evidence_hash="e" * 64,
            source_provenance_key=TARGET_KEY,
            custody_public_key_pem=custody_public,
            closeout_public_keys={"i34-closeout-key": closeout_public},
            disclosed_at=NOW,
        )
    with pytest.raises(ValueError, match="must differ from custody"):
        sign_campaign_closeout_authorization_v55(
            protocol=protocol,
            attestation=attestation,
            terminal_status="TERMINATED",
            terminal_evidence_hash="e" * 64,
            authorization_id="bad-closeout",
            closeout_authority_key_id="custody-key-reused",
            closeout_authority_private_key_pem=custody_private,
            authorized_at=NOW,
        )


def test_protocol_materializer_cli_is_create_once(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    _write_json(protocol_path, _protocol())
    output = tmp_path / "launch"
    args = [
        "--protocol",
        str(protocol_path),
        "--task-id",
        "i34-case",
        "--eligibility-contract-id",
        "i34-eligibility",
        "--output-dir",
        str(output),
    ]
    assert protocol_materializer_main(args) == 0
    policy = CandidateSelectionPolicyV55.model_validate_json(
        (output / "candidate_selection_policy_v55.json").read_text(encoding="utf-8")
    )
    contract = PublicEligibilityContractV54.model_validate_json(
        (output / "public_eligibility_contract_v54.json").read_text(encoding="utf-8")
    )
    assert policy.baseline_id == contract.baseline_id == "persistence_last_value"
    with pytest.raises(FileExistsError, match="already exists"):
        protocol_materializer_main(args)


def test_split_custody_and_provenance_release_clis(tmp_path: Path) -> None:
    protocol = _protocol()
    score_contract = _score_contract(protocol)
    custody_private, custody_public = _key_pair()
    closeout_private, closeout_public = _key_pair()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _write_json(inputs / "protocol.json", protocol)
    _write_json(inputs / "score_contract.json", score_contract)
    _write_json(
        inputs / "private_targets.json",
        {
            "targets": [
                {"target_id": "target-a", "value": 12.5},
                {"target_id": "target-b", "value": 13.5},
            ]
        },
    )
    _write_json(inputs / "source.json", _source())
    (inputs / "target.key").write_bytes(TARGET_KEY)
    (inputs / "provenance.key").write_bytes(PROVENANCE_KEY)
    (inputs / "custody_private.pem").write_bytes(custody_private)
    (inputs / "custody_public.pem").write_bytes(custody_public)
    (inputs / "closeout_private.pem").write_bytes(closeout_private)
    (inputs / "closeout_public.pem").write_bytes(closeout_public)
    (inputs / "terminal_evidence.json").write_text(
        '{"decision":"ABSTAIN"}\n',
        encoding="utf-8",
        newline="\n",
    )

    custody_output = tmp_path / "custody"
    assert (
        custodian_worker_main(
            [
                "--protocol",
                str(inputs / "protocol.json"),
                "--score-contract",
                str(inputs / "score_contract.json"),
                "--private-targets",
                str(inputs / "private_targets.json"),
                "--source-provenance",
                str(inputs / "source.json"),
                "--private-target-key-id",
                "i34-target-aes",
                "--private-target-key",
                str(inputs / "target.key"),
                "--source-provenance-key-id",
                "i34-provenance-aes",
                "--source-provenance-key",
                str(inputs / "provenance.key"),
                "--custodian-host-id",
                "custodian-host",
                "--coordinator-host-id",
                "coordinator-host",
                "--generator-host-id",
                "generator-host",
                "--attestation-id",
                "i34-split-custody",
                "--custody-key-id",
                "i34-custody-key",
                "--custody-private-key",
                str(inputs / "custody_private.pem"),
                "--output-dir",
                str(custody_output),
            ]
        )
        == 0
    )
    authorization_path = tmp_path / "closeout_authorization.json"
    assert (
        closeout_authority_main(
            [
                "--protocol",
                str(inputs / "protocol.json"),
                "--split-custody-attestation",
                str(custody_output / "split_custody_attestation_v55.json"),
                "--terminal-status",
                "ABSTAIN",
                "--terminal-evidence",
                str(inputs / "terminal_evidence.json"),
                "--authorization-id",
                "i34-closeout",
                "--closeout-authority-key-id",
                "i34-closeout-key",
                "--closeout-authority-private-key",
                str(inputs / "closeout_private.pem"),
                "--output",
                str(authorization_path),
            ]
        )
        == 0
    )
    release_output = tmp_path / "released"
    assert (
        provenance_release_main(
            [
                "--protocol",
                str(inputs / "protocol.json"),
                "--source-provenance-envelope",
                str(custody_output / "source_provenance_envelope_v55.json"),
                "--split-custody-attestation",
                str(custody_output / "split_custody_attestation_v55.json"),
                "--closeout-authorization",
                str(authorization_path),
                "--terminal-evidence",
                str(inputs / "terminal_evidence.json"),
                "--source-provenance-key",
                str(inputs / "provenance.key"),
                "--custody-public-key",
                str(inputs / "custody_public.pem"),
                "--closeout-authority-public-key",
                str(inputs / "closeout_public.pem"),
                "--output-dir",
                str(release_output),
            ]
        )
        == 0
    )
    released = json.loads(
        (release_output / "source_provenance_record_v55.json").read_text(
            encoding="utf-8"
        )
    )
    receipt = json.loads(
        (release_output / "source_provenance_disclosure_receipt_v55.json").read_text(
            encoding="utf-8"
        )
    )
    assert released["source_authority"] == "Example National Statistics Office"
    assert "holdout" not in released
    assert receipt["private_target_envelope_accessed"] is False
    assert receipt["private_target_key_accessed"] is False
