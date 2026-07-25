from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

from fma.v5_3.external_private import PrivateEvaluationRequestV53
from fma.v5_4.public_eligibility import (
    PairedForecastLossV54,
    PublicEligibilityAuthorityV54,
    PublicEligibilityContractV54,
    PublicEligibilityInputV54,
    assess_public_eligibility_v54,
    authorize_private_evaluation_request_v54,
    verify_private_evaluation_authorization_v54,
    verify_public_eligibility_receipt_v54,
)


ROOT = Path(__file__).resolve().parents[1]
I32_CANDIDATES = (
    ROOT
    / "experiments"
    / "iteration_32"
    / "campaigns"
    / "i32-shadow-177747afada8fc62a6ed"
    / "modeler"
    / "candidate_results.json"
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _contract(
    *,
    task_id: str = "task-v54",
    candidate_budget: int = 4,
    baseline_id: str = "persistence",
) -> PublicEligibilityContractV54:
    return PublicEligibilityContractV54.seal(
        contract_id=f"{task_id}-eligibility",
        task_id=task_id,
        baseline_id=baseline_id,
        candidate_selection_rule_hash=_sha("selection-rule"),
        expected_horizons=[1, 2, 3, 4],
        minimum_origin_count=12,
        contiguous_time_block_count=3,
        recent_origin_count=4,
        minimum_origin_win_fraction=0.6,
        bootstrap_confidence=0.95,
        bootstrap_replicates=4096,
        bootstrap_block_length=4,
        multiplicity_correction_count=candidate_budget,
        bootstrap_seed=1729,
        frozen_at=datetime(2026, 7, 25, tzinfo=timezone.utc),
    )


def _evidence(
    contract: PublicEligibilityContractV54,
    *,
    origin_advantages: list[float],
    public_scientific_acceptance: bool = True,
    fixture_only: bool = False,
) -> PublicEligibilityInputV54:
    rows = [
        PairedForecastLossV54(
            origin=origin,
            horizon=horizon,
            candidate_loss=1.0 - advantage,
            baseline_loss=1.0,
        )
        for origin, advantage in enumerate(origin_advantages, start=1)
        for horizon in contract.expected_horizons
    ]
    return PublicEligibilityInputV54.seal(
        task_id=contract.task_id,
        contract_hash=contract.contract_hash,
        candidate_id="candidate",
        baseline_id=contract.baseline_id,
        candidate_search_count=contract.multiplicity_correction_count,
        public_scientific_acceptance_verified=public_scientific_acceptance,
        fixture_only=fixture_only,
        source_artifact_hashes=[_sha("candidate-artifact")],
        rows=rows,
    )


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


def _private_request() -> PrivateEvaluationRequestV53:
    return PrivateEvaluationRequestV53.seal(
        request_id="private-request",
        case_id="private-case",
        evaluator_epoch="epoch-1",
        score_contract_hash=_sha("score-contract"),
        forecast_plan_hash=_sha("forecast-plan"),
        forecast_bundle_hash=_sha("forecast-bundle"),
        custody_attestation_hash=_sha("custody-attestation"),
        prediction_registration_hash=_sha("prediction-registration"),
        graph_binding_hash=_sha("graph-binding"),
        prediction_snapshot_hash=_sha("prediction-snapshot"),
        prediction_semantic_hash=_sha("prediction-semantic"),
        private_capsule_commitment=_sha("private-capsule"),
        minimum_quality_score=0.7,
        created_at=datetime(2026, 7, 25, tzinfo=timezone.utc),
    )


def test_stable_advantage_is_eligible_and_authenticated() -> None:
    contract = _contract()
    evidence = _evidence(
        contract,
        origin_advantages=[
            0.20,
            0.16,
            0.18,
            0.22,
            0.14,
            0.19,
            0.17,
            0.21,
            0.15,
            0.20,
            0.18,
            0.16,
        ],
    )

    assessment = assess_public_eligibility_v54(
        contract=contract,
        evidence=evidence,
    )
    assert assessment.decision == "ELIGIBLE"
    assert assessment.public_gate_eligible is True
    assert all(assessment.checks.values())
    assert assessment.scientific_qualification_granted is False

    private_pem, public_pem = _key_pair()
    authority = PublicEligibilityAuthorityV54(
        key_id="public-gate",
        private_key_pem=private_pem,
    )
    receipt = authority.issue(receipt_id="receipt-1", assessment=assessment)
    assert authority.verify(assessment=assessment, receipt=receipt) is True
    assert (
        verify_public_eligibility_receipt_v54(
            assessment=assessment,
            receipt=receipt,
            authority_public_key_pem=public_pem,
        )
        is True
    )
    _, wrong_public_pem = _key_pair()
    assert (
        verify_public_eligibility_receipt_v54(
            assessment=assessment,
            receipt=receipt,
            authority_public_key_pem=wrong_public_pem,
        )
        is False
    )

    request = _private_request()
    authorization = authorize_private_evaluation_request_v54(
        authorization_id="authorization-1",
        request=request,
        contract=contract,
        evidence=evidence,
        assessment=assessment,
        receipt=receipt,
        authority_public_key_pem=public_pem,
    )
    assert (
        verify_private_evaluation_authorization_v54(
            authorization=authorization,
            request=request,
            contract=contract,
            evidence=evidence,
            assessment=assessment,
            receipt=receipt,
            authority_public_key_pem=public_pem,
        )
        is True
    )

    wrong_private_pem, _ = _key_pair()
    wrong_authority = PublicEligibilityAuthorityV54(
        key_id="public-gate",
        private_key_pem=wrong_private_pem,
    )
    assert wrong_authority.verify(assessment=assessment, receipt=receipt) is False


def test_late_collapse_abstains_despite_positive_pooled_mean() -> None:
    contract = _contract()
    evidence = _evidence(
        contract,
        origin_advantages=[
            0.50,
            0.45,
            0.40,
            0.35,
            0.30,
            0.25,
            0.20,
            0.15,
            -0.04,
            -0.06,
            -0.08,
            -0.10,
        ],
    )

    assessment = assess_public_eligibility_v54(
        contract=contract,
        evidence=evidence,
    )
    assert assessment.metrics.mean_advantage > 0
    assert assessment.decision == "ABSTAIN"
    assert assessment.checks["recent_window_advantage"] is False
    assert assessment.checks["all_contiguous_time_blocks"] is False
    assert assessment.checks["all_recent_horizon_advantages"] is False


def test_fixture_or_failed_scientific_assessment_cannot_be_eligible() -> None:
    contract = _contract()
    stable = [0.2] * 12

    fixture_assessment = assess_public_eligibility_v54(
        contract=contract,
        evidence=_evidence(contract, origin_advantages=stable, fixture_only=True),
    )
    assert fixture_assessment.decision == "ABSTAIN"
    assert fixture_assessment.checks["nonfixture_public_evidence"] is False

    scientific_failure = assess_public_eligibility_v54(
        contract=contract,
        evidence=_evidence(
            contract,
            origin_advantages=stable,
            public_scientific_acceptance=False,
        ),
    )
    assert scientific_failure.decision == "ABSTAIN"
    assert scientific_failure.checks["public_scientific_acceptance"] is False


def test_selection_adjusted_bootstrap_can_force_abstention() -> None:
    contract = PublicEligibilityContractV54.seal(
        contract_id="bootstrap-contract",
        task_id="bootstrap-task",
        baseline_id="persistence",
        candidate_selection_rule_hash=_sha("selection-rule"),
        expected_horizons=[1, 2, 3, 4],
        minimum_origin_count=12,
        contiguous_time_block_count=3,
        recent_origin_count=4,
        minimum_origin_win_fraction=0.5,
        bootstrap_confidence=0.95,
        bootstrap_replicates=4096,
        bootstrap_block_length=4,
        multiplicity_correction_count=4,
        bootstrap_seed=1729,
        frozen_at=datetime(2026, 7, 25, tzinfo=timezone.utc),
    )
    evidence = _evidence(
        contract,
        origin_advantages=[
            1.0,
            1.0,
            -1.0,
            -0.9,
            -0.9,
            1.0,
            1.0,
            -1.0,
            -1.0,
            -0.9,
            1.0,
            1.0,
        ],
    )

    assessment = assess_public_eligibility_v54(
        contract=contract,
        evidence=evidence,
    )
    assert assessment.metrics.mean_advantage > 0
    assert all(value > 0 for value in assessment.metrics.contiguous_time_block_means)
    assert assessment.metrics.recent_mean_advantage > 0
    assert assessment.decision == "ABSTAIN"
    assert assessment.checks["selection_adjusted_bootstrap_lower_bound"] is False


def test_incomplete_or_duplicate_origin_horizon_grid_is_rejected() -> None:
    contract = _contract()
    evidence = _evidence(contract, origin_advantages=[0.2] * 12)
    payload = evidence.model_dump(mode="json", exclude={"input_hash"})
    payload["rows"] = payload["rows"][:-1]
    incomplete = PublicEligibilityInputV54.seal(**payload)
    with pytest.raises(ValueError, match="origin-horizon grid"):
        assess_public_eligibility_v54(contract=contract, evidence=incomplete)

    payload = evidence.model_dump(mode="json", exclude={"input_hash"})
    payload["rows"][1] = payload["rows"][0]
    with pytest.raises(ValidationError, match="sorted and unique"):
        PublicEligibilityInputV54.seal(**payload)


def test_candidate_search_cannot_exceed_frozen_correction_budget() -> None:
    contract = _contract(candidate_budget=4)
    evidence = _evidence(contract, origin_advantages=[0.2] * 12)
    payload = evidence.model_dump(mode="json", exclude={"input_hash"})
    payload["candidate_search_count"] = 5
    excessive = PublicEligibilityInputV54.seal(**payload)
    with pytest.raises(ValueError, match="exceeds the frozen correction budget"):
        assess_public_eligibility_v54(contract=contract, evidence=excessive)


def test_contract_requires_enough_bootstrap_tail_resolution() -> None:
    with pytest.raises(ValidationError, match="multiplicity-adjusted tail"):
        PublicEligibilityContractV54.seal(
            contract_id="bad-contract",
            task_id="task-v54",
            baseline_id="persistence",
            candidate_selection_rule_hash=_sha("selection-rule"),
            expected_horizons=[1, 2, 3, 4],
            minimum_origin_count=12,
            contiguous_time_block_count=3,
            recent_origin_count=4,
            bootstrap_replicates=1000,
            bootstrap_block_length=4,
            multiplicity_correction_count=32,
            frozen_at=datetime(2026, 7, 25, tzinfo=timezone.utc),
        )


def test_consumed_i32_shadow_would_have_abstained_before_private_evaluation() -> None:
    artifact_bytes = I32_CANDIDATES.read_bytes()
    artifact = json.loads(artifact_bytes)
    contract = _contract(
        task_id="i32-shadow-177747afada8fc62a6ed",
        candidate_budget=len(artifact["candidates"]),
        baseline_id="persistence_last_value",
    )
    selected = artifact["selection"]["selected_candidate_id"]
    baseline = "persistence_last_value"
    selected_rows = artifact["candidates"][selected]["rolling_validation"]["rows"]
    baseline_rows = artifact["candidates"][baseline]["rolling_validation"]["rows"]
    baseline_by_coordinate = {
        (row["train_size"], row["horizon"]): row for row in baseline_rows
    }
    scale = 1130.708449021917
    rows = [
        PairedForecastLossV54(
            origin=row["train_size"],
            horizon=row["horizon"],
            candidate_loss=row["absolute_error"] / scale,
            baseline_loss=baseline_by_coordinate[(row["train_size"], row["horizon"])][
                "absolute_error"
            ]
            / scale,
        )
        for row in selected_rows
    ]
    evidence = PublicEligibilityInputV54.seal(
        task_id=contract.task_id,
        contract_hash=contract.contract_hash,
        candidate_id=selected,
        baseline_id=baseline,
        candidate_search_count=len(artifact["candidates"]),
        public_scientific_acceptance_verified=artifact["claim_limits"][
            "public_scientific_acceptance"
        ],
        fixture_only=False,
        source_artifact_hashes=[hashlib.sha256(artifact_bytes).hexdigest()],
        rows=rows,
    )

    assessment = assess_public_eligibility_v54(
        contract=contract,
        evidence=evidence,
    )
    assert assessment.decision == "ABSTAIN"
    assert assessment.metrics.mean_advantage > 0
    assert assessment.metrics.recent_mean_advantage < 0
    assert assessment.checks["public_scientific_acceptance"] is False
    assert assessment.checks["origin_win_fraction"] is False
    assert assessment.checks["all_contiguous_time_blocks"] is False
    assert assessment.checks["recent_window_advantage"] is False
    assert assessment.checks["all_recent_horizon_advantages"] is False
    assert assessment.private_evaluation_performed is False
    assert assessment.scientific_qualification_granted is False

    private_pem, public_pem = _key_pair()
    authority = PublicEligibilityAuthorityV54(
        key_id="public-gate",
        private_key_pem=private_pem,
    )
    receipt = authority.issue(receipt_id="i32-receipt", assessment=assessment)
    with pytest.raises(ValueError, match="requires an eligible public assessment"):
        authorize_private_evaluation_request_v54(
            authorization_id="i32-authorization",
            request=_private_request(),
            contract=contract,
            evidence=evidence,
            assessment=assessment,
            receipt=receipt,
            authority_public_key_pem=public_pem,
        )
