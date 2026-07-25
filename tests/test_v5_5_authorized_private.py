from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fma.hashing import canonical_json, sha256_value
from fma.v5.external_harness import (
    PredictionDocumentV50,
    PredictionPointV50,
    PrivateTargetV50,
)
from fma.v5_3.custody import PrivateScoreContractV53
from fma.v5_3.external_private import PrivateEvaluationRequestV53
from fma.v5_4.public_eligibility import (
    PairedForecastLossV54,
    PrivateEvaluationAuthorizationV54,
    PublicEligibilityAssessmentV54,
    PublicEligibilityAuthorityV54,
    PublicEligibilityInputV54,
    assess_public_eligibility_v54,
    authorize_private_evaluation_request_v54,
)
from fma.v5_5.authorized_private import (
    LegacyCustodyBridgeV55,
    assert_authorized_encrypted_private_preconditions_v55,
    claim_private_evaluation_budget_v55,
    create_v53_custody_compatibility_v55,
    evaluate_authorized_encrypted_private_inputs_v55,
    verify_authorized_encrypted_private_output_v55,
    verify_legacy_custody_bridge_v55,
)
from fma.v5_5.authorized_private_worker import main as private_worker_main
from fma.v5_5.campaign_protocol import (
    ProspectiveCampaignProtocolV55,
    PublicEligibilitySettingsV55,
    materialize_public_launch_v55,
)
from fma.v5_5.split_custody import (
    EncryptedCustodyEnvelopeV55,
    SourceProvenanceDraftV55,
    create_split_custody_envelopes_v55,
)


NOW = datetime(2026, 7, 26, 2, 3, 4, tzinfo=timezone.utc)
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


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8", newline="\n")


def _chain() -> dict[str, object]:
    protocol = ProspectiveCampaignProtocolV55.seal(
        protocol_id="i34-private-chain",
        baseline_id="persistence_last_value",
        candidate_families=["constant", "exponential", "gompertz", "logistic"],
        maximum_candidate_search_count=4,
        public_eligibility=PublicEligibilitySettingsV55(
            expected_horizons=[1, 2, 3, 4],
            minimum_origin_count=12,
            contiguous_time_block_count=3,
            recent_origin_count=4,
            bootstrap_replicates=1024,
            bootstrap_block_length=4,
            multiplicity_correction_count=4,
            bootstrap_seed=5517,
        ),
        frozen_at=NOW,
    )
    policy, eligibility_contract, launch_binding = materialize_public_launch_v55(
        protocol=protocol,
        task_id="i34-case",
        eligibility_contract_id="i34-public-eligibility",
        materialized_at=NOW,
    )
    score_contract = PrivateScoreContractV53.seal(
        contract_id="i34-private-score",
        case_id="i34-case",
        protocol_hash=protocol.protocol_hash,
        public_case_hash=_sha("public-case"),
        forecast_plan_hash=_sha("forecast-plan"),
        target_ids=["target-a", "target-b"],
        quality_scale=10.0,
        minimum_quality_score=0.5,
        frozen_at=NOW,
    )
    custody_private, custody_public = _key_pair()
    (
        capsule,
        _,
        target_envelope,
        provenance_envelope,
        split_attestation,
    ) = create_split_custody_envelopes_v55(
        protocol=protocol,
        score_contract=score_contract,
        private_targets=[
            PrivateTargetV50(target_id="target-a", value=12.5),
            PrivateTargetV50(target_id="target-b", value=13.5),
        ],
        source_provenance=SourceProvenanceDraftV55(
            case_id="i34-case",
            source_authority="Fixture statistics authority",
            source_title="Fixture monthly series",
            source_locator="https://example.invalid/i34-fixture",
            table_or_series_id="FIXTURE-SERIES",
            public_period_start="2010-01",
            public_period_end="2024-12",
            private_period_start="2025-01",
            private_period_end="2025-02",
            source_artifact_sha256=_sha("source-artifact"),
            prior_campaign_exclusion_hashes=[_sha("prior-campaign")],
            retrieved_at=NOW,
        ),
        private_target_envelope_id="i34-target-envelope",
        source_provenance_envelope_id="i34-provenance-envelope",
        private_target_key_id="i34-target-key",
        private_target_key=TARGET_KEY,
        source_provenance_key_id="i34-provenance-key",
        source_provenance_key=PROVENANCE_KEY,
        custodian_host_id="custodian-host",
        coordinator_host_id="coordinator-host",
        generator_host_id="generator-host",
        attestation_id="i34-split-attestation",
        custody_key_id="i34-custody-key",
        custody_private_key_pem=custody_private,
        private_target_nonce=b"\x10" * 12,
        source_provenance_nonce=b"\x20" * 12,
        private_target_canary="target-canary-" + "1" * 40,
        source_provenance_canary="provenance-canary-" + "2" * 40,
        attested_at=NOW,
    )
    v53_attestation, custody_bridge = create_v53_custody_compatibility_v55(
        protocol=protocol,
        score_contract=score_contract,
        capsule=capsule,
        private_target_envelope=target_envelope,
        split_custody_attestation=split_attestation,
        custody_private_key_pem=custody_private,
        v53_attestation_id="i34-v53-attestation",
        external_anchor_receipt_hash=_sha("external-anchor"),
        attested_at=NOW,
    )
    prediction = PredictionDocumentV50(
        case_id="i34-case",
        predictions=[
            PredictionPointV50(target_id="target-a", value=12.0),
            PredictionPointV50(target_id="target-b", value=14.0),
        ],
    )
    prediction_bytes = (canonical_json(prediction) + "\n").encode("utf-8")
    request = PrivateEvaluationRequestV53.seal(
        request_id="i34-private-request",
        case_id="i34-case",
        evaluator_epoch="i34-epoch",
        score_contract_hash=score_contract.contract_hash,
        forecast_plan_hash=score_contract.forecast_plan_hash,
        forecast_bundle_hash=_sha("forecast-bundle"),
        custody_attestation_hash=v53_attestation.attestation_hash,
        prediction_registration_hash=_sha("prediction-registration"),
        graph_binding_hash=_sha("graph-binding"),
        prediction_snapshot_hash=hashlib.sha256(prediction_bytes).hexdigest(),
        prediction_semantic_hash=sha256_value(prediction),
        private_capsule_commitment=capsule.capsule_hash,
        minimum_quality_score=score_contract.minimum_quality_score,
        created_at=NOW,
    )
    rows = [
        PairedForecastLossV54(
            origin=origin,
            horizon=horizon,
            candidate_loss=0.8,
            baseline_loss=1.0,
        )
        for origin in range(1, 13)
        for horizon in eligibility_contract.expected_horizons
    ]
    eligibility_input = PublicEligibilityInputV54.seal(
        task_id="i34-case",
        contract_hash=eligibility_contract.contract_hash,
        candidate_id="logistic",
        baseline_id=eligibility_contract.baseline_id,
        candidate_search_count=4,
        public_scientific_acceptance_verified=True,
        fixture_only=False,
        source_artifact_hashes=[_sha("public-evidence")],
        rows=rows,
    )
    eligibility_assessment = assess_public_eligibility_v54(
        contract=eligibility_contract,
        evidence=eligibility_input,
    )
    assert eligibility_assessment.decision == "ELIGIBLE"
    eligibility_private, eligibility_public = _key_pair()
    eligibility_authority = PublicEligibilityAuthorityV54(
        key_id="i34-eligibility-key",
        private_key_pem=eligibility_private,
    )
    eligibility_receipt = eligibility_authority.issue(
        receipt_id="i34-eligibility-receipt",
        assessment=eligibility_assessment,
    )
    private_authorization = authorize_private_evaluation_request_v54(
        authorization_id="i34-private-authorization",
        request=request,
        contract=eligibility_contract,
        evidence=eligibility_input,
        assessment=eligibility_assessment,
        receipt=eligibility_receipt,
        authority_public_key_pem=eligibility_public,
        issued_at=NOW,
    )
    worker_private, worker_public = _key_pair()
    return {
        "protocol": protocol,
        "candidate_policy": policy,
        "public_launch_binding": launch_binding,
        "eligibility_contract": eligibility_contract,
        "eligibility_input": eligibility_input,
        "eligibility_assessment": eligibility_assessment,
        "eligibility_receipt": eligibility_receipt,
        "private_authorization": private_authorization,
        "eligibility_authority": eligibility_authority,
        "eligibility_authority_public_key_pem": eligibility_public,
        "request": request,
        "score_contract": score_contract,
        "prediction": prediction,
        "prediction_bytes": prediction_bytes,
        "v53_custody_attestation": v53_attestation,
        "private_target_envelope": target_envelope,
        "source_provenance_envelope": provenance_envelope,
        "split_custody_attestation": split_attestation,
        "legacy_custody_bridge": custody_bridge,
        "custody_public_key_pem": custody_public,
        "worker_private_key_pem": worker_private,
        "worker_public_key_pem": worker_public,
        "capsule": capsule,
    }


def _precondition_kwargs(chain: dict[str, object]) -> dict[str, object]:
    names = {
        "protocol",
        "candidate_policy",
        "public_launch_binding",
        "eligibility_contract",
        "eligibility_input",
        "eligibility_assessment",
        "eligibility_receipt",
        "private_authorization",
        "eligibility_authority_public_key_pem",
        "request",
        "score_contract",
        "prediction",
        "v53_custody_attestation",
        "private_target_envelope",
        "source_provenance_envelope",
        "split_custody_attestation",
        "legacy_custody_bridge",
        "custody_public_key_pem",
    }
    values = {name: chain[name] for name in names}
    values["prediction_bytes_hash"] = hashlib.sha256(
        chain["prediction_bytes"]
    ).hexdigest()
    values["expected_coordinator_host_id"] = "coordinator-host"
    values["expected_generator_host_id"] = "generator-host"
    return values


def _evaluate(
    chain: dict[str, object],
    *,
    ledger_root: Path,
):
    assert_authorized_encrypted_private_preconditions_v55(
        **_precondition_kwargs(chain)
    )
    claim, claim_path = claim_private_evaluation_budget_v55(
        ledger_root=ledger_root,
        budget_ledger_id="i34-private-ledger",
        request=chain["request"],
        private_authorization=chain["private_authorization"],
        private_target_envelope=chain["private_target_envelope"],
        split_custody_attestation=chain["split_custody_attestation"],
        fixture_only=True,
        claimed_at=NOW + timedelta(seconds=1),
    )
    target_key_path = ledger_root.parent / "private-target.key"
    target_key_path.write_bytes(TARGET_KEY)
    return evaluate_authorized_encrypted_private_inputs_v55(
        **_precondition_kwargs(chain),
        private_target_key_path=target_key_path,
        budget_claim=claim,
        budget_claim_path=claim_path,
        worker_id="private-worker",
        worker_host_id="custodian-host",
        worker_executable_hash=_sha("python"),
        runner_source_hash=_sha("runner"),
        worker_key_id="i34-worker-key",
        worker_private_key_pem=chain["worker_private_key_pem"],
        fixture_only=True,
        evaluated_at=NOW + timedelta(seconds=2),
    )


def test_authorized_encrypted_private_path_binds_and_verifies(
    tmp_path: Path,
) -> None:
    chain = _chain()
    output = _evaluate(chain, ledger_root=tmp_path / "ledger")
    assert output.worker_receipt_v53.quality_score == pytest.approx(0.95)
    assert output.worker_receipt_v53.threshold_passed is True
    assert output.worker_receipt_v55.fixture_only is True
    assert output.worker_receipt_v55.scientific_qualification_granted is False
    assert output.worker_receipt_v55.real_world_action_authorized is False
    assert verify_authorized_encrypted_private_output_v55(
        output=output,
        **_precondition_kwargs(chain),
        trusted_worker_public_keys={
            "i34-worker-key": chain["worker_public_key_pem"]
        },
    )
    public_output = canonical_json(output)
    capsule = chain["capsule"]
    assert capsule.secrecy_canary not in public_output
    assert '"value":12.5' not in public_output
    assert '"value":13.5' not in public_output

    _, wrong_worker_public = _key_pair()
    assert not verify_authorized_encrypted_private_output_v55(
        output=output,
        **_precondition_kwargs(chain),
        trusted_worker_public_keys={"i34-worker-key": wrong_worker_public},
    )


def test_private_budget_is_create_once_across_output_paths(tmp_path: Path) -> None:
    chain = _chain()
    claim_private_evaluation_budget_v55(
        ledger_root=tmp_path / "ledger",
        budget_ledger_id="i34-private-ledger",
        request=chain["request"],
        private_authorization=chain["private_authorization"],
        private_target_envelope=chain["private_target_envelope"],
        split_custody_attestation=chain["split_custody_attestation"],
        fixture_only=True,
        claimed_at=NOW,
    )
    with pytest.raises(ValueError, match="already claimed"):
        claim_private_evaluation_budget_v55(
            ledger_root=tmp_path / "ledger",
            budget_ledger_id="i34-private-ledger",
            request=chain["request"],
            private_authorization=chain["private_authorization"],
            private_target_envelope=chain["private_target_envelope"],
            split_custody_attestation=chain["split_custody_attestation"],
            fixture_only=True,
            claimed_at=NOW + timedelta(seconds=1),
        )


def test_fixture_cannot_forge_eligible_assessment_replay() -> None:
    chain = _chain()
    original_input = chain["eligibility_input"]
    fixture_payload = original_input.model_dump(exclude={"input_hash"})
    fixture_payload["fixture_only"] = True
    fixture_input = PublicEligibilityInputV54.seal(**fixture_payload)
    actual_assessment = assess_public_eligibility_v54(
        contract=chain["eligibility_contract"],
        evidence=fixture_input,
    )
    assert actual_assessment.decision == "ABSTAIN"

    forged_payload = chain["eligibility_assessment"].model_dump(
        exclude={"assessment_hash"}
    )
    forged_payload["input_hash"] = fixture_input.input_hash
    forged_assessment = PublicEligibilityAssessmentV54.seal(**forged_payload)
    authority = chain["eligibility_authority"]
    forged_receipt = authority.issue(
        receipt_id="forged-fixture-receipt",
        assessment=forged_assessment,
    )
    forged_authorization = authorize_private_evaluation_request_v54(
        authorization_id="forged-fixture-authorization",
        request=chain["request"],
        contract=chain["eligibility_contract"],
        evidence=fixture_input,
        assessment=forged_assessment,
        receipt=forged_receipt,
        authority_public_key_pem=chain["eligibility_authority_public_key_pem"],
        issued_at=NOW,
    )
    kwargs = _precondition_kwargs(chain)
    kwargs.update(
        eligibility_input=fixture_input,
        eligibility_assessment=forged_assessment,
        eligibility_receipt=forged_receipt,
        private_authorization=forged_authorization,
    )
    with pytest.raises(ValueError, match="deterministic replay"):
        assert_authorized_encrypted_private_preconditions_v55(**kwargs)


def test_legacy_bridge_rejects_commitment_drift() -> None:
    chain = _chain()
    bridge = chain["legacy_custody_bridge"]
    payload = bridge.model_dump(exclude={"bridge_hash"})
    payload["private_capsule_commitment"] = _sha("other-capsule")
    drifted = LegacyCustodyBridgeV55.seal(**payload)
    assert not verify_legacy_custody_bridge_v55(
        score_contract=chain["score_contract"],
        v53_custody_attestation=chain["v53_custody_attestation"],
        private_target_envelope=chain["private_target_envelope"],
        split_custody_attestation=chain["split_custody_attestation"],
        bridge=drifted,
        custody_public_key_pem=chain["custody_public_key_pem"],
    )
    kwargs = _precondition_kwargs(chain)
    kwargs["legacy_custody_bridge"] = drifted
    with pytest.raises(ValueError, match="bridge is not verified"):
        assert_authorized_encrypted_private_preconditions_v55(**kwargs)


def test_swapped_envelopes_and_wrong_target_key_fail_closed(
    tmp_path: Path,
) -> None:
    chain = _chain()
    kwargs = _precondition_kwargs(chain)
    kwargs["private_target_envelope"] = chain["source_provenance_envelope"]
    kwargs["source_provenance_envelope"] = chain["private_target_envelope"]
    with pytest.raises(ValueError, match="custody or campaign binding differs"):
        assert_authorized_encrypted_private_preconditions_v55(**kwargs)

    assert_authorized_encrypted_private_preconditions_v55(
        **_precondition_kwargs(chain)
    )
    claim, claim_path = claim_private_evaluation_budget_v55(
        ledger_root=tmp_path / "ledger",
        budget_ledger_id="i34-private-ledger",
        request=chain["request"],
        private_authorization=chain["private_authorization"],
        private_target_envelope=chain["private_target_envelope"],
        split_custody_attestation=chain["split_custody_attestation"],
        fixture_only=True,
        claimed_at=NOW,
    )
    wrong_key_path = tmp_path / "wrong-target.key"
    wrong_key_path.write_bytes(b"\xff" * 32)
    with pytest.raises(ValueError, match="decryption key fingerprint differs"):
        evaluate_authorized_encrypted_private_inputs_v55(
            **_precondition_kwargs(chain),
            private_target_key_path=wrong_key_path,
            budget_claim=claim,
            budget_claim_path=claim_path,
            worker_id="private-worker",
            worker_host_id="custodian-host",
            worker_executable_hash=_sha("python"),
            runner_source_hash=_sha("runner"),
            worker_key_id="i34-worker-key",
            worker_private_key_pem=chain["worker_private_key_pem"],
            fixture_only=True,
            evaluated_at=NOW + timedelta(seconds=1),
        )


def test_tampered_budget_claim_file_blocks_target_decryption(
    tmp_path: Path,
) -> None:
    chain = _chain()
    claim, claim_path = claim_private_evaluation_budget_v55(
        ledger_root=tmp_path / "ledger",
        budget_ledger_id="i34-private-ledger",
        request=chain["request"],
        private_authorization=chain["private_authorization"],
        private_target_envelope=chain["private_target_envelope"],
        split_custody_attestation=chain["split_custody_attestation"],
        fixture_only=True,
        claimed_at=NOW,
    )
    claim_path.write_text("tampered\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="claim file differs"):
        evaluate_authorized_encrypted_private_inputs_v55(
            **_precondition_kwargs(chain),
            private_target_key_path=tmp_path / "private-target.key",
            budget_claim=claim,
            budget_claim_path=claim_path,
            worker_id="private-worker",
            worker_host_id="custodian-host",
            worker_executable_hash=_sha("python"),
            runner_source_hash=_sha("runner"),
            worker_key_id="i34-worker-key",
            worker_private_key_pem=chain["worker_private_key_pem"],
            fixture_only=True,
            evaluated_at=NOW + timedelta(seconds=1),
        )


def test_rehashed_tampered_ciphertext_fails_public_custody_binding() -> None:
    chain = _chain()
    envelope = chain["private_target_envelope"]
    payload = envelope.model_dump(exclude={"envelope_hash"})
    ciphertext = bytearray(base64.b64decode(payload["ciphertext_base64"]))
    ciphertext[0] ^= 1
    payload["ciphertext_base64"] = base64.b64encode(ciphertext).decode("ascii")
    payload["ciphertext_sha256"] = hashlib.sha256(ciphertext).hexdigest()
    draft = EncryptedCustodyEnvelopeV55(**payload)
    payload["envelope_hash"] = draft.content_hash()
    tampered = EncryptedCustodyEnvelopeV55(**payload)
    kwargs = _precondition_kwargs(chain)
    kwargs["private_target_envelope"] = tampered
    with pytest.raises(ValueError, match="split custody bindings"):
        assert_authorized_encrypted_private_preconditions_v55(**kwargs)


def test_prediction_drift_fails_before_target_access() -> None:
    chain = _chain()
    kwargs = _precondition_kwargs(chain)
    kwargs["prediction_bytes_hash"] = _sha("different-prediction-bytes")
    with pytest.raises(ValueError, match="snapshot hash differs"):
        assert_authorized_encrypted_private_preconditions_v55(**kwargs)


def _write_cli_artifacts(
    root: Path,
    chain: dict[str, object],
    *,
    abstain: bool,
) -> list[str]:
    values = dict(chain)
    if abstain:
        original_input = chain["eligibility_input"]
        payload = original_input.model_dump(exclude={"input_hash"})
        payload["fixture_only"] = True
        eligibility_input = PublicEligibilityInputV54.seal(**payload)
        assessment = assess_public_eligibility_v54(
            contract=chain["eligibility_contract"],
            evidence=eligibility_input,
        )
        authority = chain["eligibility_authority"]
        receipt = authority.issue(
            receipt_id="abstain-receipt",
            assessment=assessment,
        )
        authorization = PrivateEvaluationAuthorizationV54.seal(
            authorization_id="abstain-not-an-authorization",
            request_hash=chain["request"].request_hash,
            contract_hash=chain["eligibility_contract"].contract_hash,
            input_hash=eligibility_input.input_hash,
            assessment_hash=assessment.assessment_hash,
            eligibility_receipt_hash=receipt.receipt_hash,
            eligibility_authority_key_id=receipt.authority_key_id,
            eligibility_authority_public_key_fingerprint=(
                receipt.authority_public_key_fingerprint
            ),
            issued_at=NOW,
        )
        values.update(
            eligibility_input=eligibility_input,
            eligibility_assessment=assessment,
            eligibility_receipt=receipt,
            private_authorization=authorization,
        )
    public_files = {
        "protocol": "protocol",
        "candidate-policy": "candidate_policy",
        "public-launch-binding": "public_launch_binding",
        "eligibility-contract": "eligibility_contract",
        "eligibility-input": "eligibility_input",
        "eligibility-assessment": "eligibility_assessment",
        "eligibility-receipt": "eligibility_receipt",
        "private-authorization": "private_authorization",
        "request": "request",
        "score-contract": "score_contract",
        "v53-custody-attestation": "v53_custody_attestation",
        "private-target-envelope": "private_target_envelope",
        "source-provenance-envelope": "source_provenance_envelope",
        "split-custody-attestation": "split_custody_attestation",
        "legacy-custody-bridge": "legacy_custody_bridge",
    }
    arguments: list[str] = []
    for option, key in public_files.items():
        path = root / f"{option}.json"
        _write_json(path, values[key])
        arguments.extend([f"--{option}", str(path)])
    prediction_path = root / "prediction.json"
    prediction_path.write_bytes(chain["prediction_bytes"])
    eligibility_key_path = root / "eligibility-public.pem"
    eligibility_key_path.write_bytes(
        chain["eligibility_authority_public_key_pem"]
    )
    custody_key_path = root / "custody-public.pem"
    custody_key_path.write_bytes(chain["custody_public_key_pem"])
    arguments.extend(
        [
            "--prediction",
            str(prediction_path),
            "--eligibility-authority-public-key",
            str(eligibility_key_path),
            "--custody-public-key",
            str(custody_key_path),
            "--expected-coordinator-host-id",
            "coordinator-host",
            "--expected-generator-host-id",
            "generator-host",
            "--private-target-key",
            str(root / "private-target.key"),
            "--budget-ledger-root",
            str(root / "ledger"),
            "--budget-ledger-id",
            "i34-private-ledger",
            "--worker-id",
            "private-worker",
            "--worker-host-id",
            "custodian-host",
            "--worker-key-id",
            "i34-worker-key",
            "--worker-private-key",
            str(root / "worker-private.pem"),
            "--output",
            str(root / "output.json"),
            "--fixture-only",
        ]
    )
    return arguments


def test_cli_abstain_does_not_read_nonexistent_target_key(
    tmp_path: Path,
) -> None:
    chain = _chain()
    arguments = _write_cli_artifacts(tmp_path, chain, abstain=True)
    assert not (tmp_path / "private-target.key").exists()
    with pytest.raises(ValueError, match="not publicly authorized"):
        private_worker_main(arguments)
    assert not (tmp_path / "ledger").exists()
    assert not (tmp_path / "output.json").exists()


def test_cli_success_then_duplicate_request_is_rejected(tmp_path: Path) -> None:
    chain = _chain()
    arguments = _write_cli_artifacts(tmp_path, chain, abstain=False)
    (tmp_path / "private-target.key").write_bytes(TARGET_KEY)
    (tmp_path / "worker-private.pem").write_bytes(
        chain["worker_private_key_pem"]
    )
    assert private_worker_main(arguments) == 0
    assert (tmp_path / "output.json").exists()

    second_output = tmp_path / "output-2.json"
    second_arguments = list(arguments)
    output_index = second_arguments.index("--output") + 1
    second_arguments[output_index] = str(second_output)
    with pytest.raises(ValueError, match="already claimed"):
        private_worker_main(second_arguments)
    assert not second_output.exists()


def test_cli_existing_output_prevents_claim_and_key_read(tmp_path: Path) -> None:
    chain = _chain()
    arguments = _write_cli_artifacts(tmp_path, chain, abstain=False)
    (tmp_path / "output.json").write_text(
        "reserved\n",
        encoding="utf-8",
        newline="\n",
    )
    assert not (tmp_path / "private-target.key").exists()
    with pytest.raises(ValueError, match="output already exists"):
        private_worker_main(arguments)
    assert not (tmp_path / "ledger").exists()
