from __future__ import annotations

import base64
import math
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

from fma.hashing import sha256_value
from fma.v6.external_qualification import (
    ExternalAggregateEvaluationV63,
    ExternalPredictionVectorV63,
    PredictiveExternalQualificationContractV63,
    external_qualification_key_fingerprint_v63,
    sign_external_aggregate_evaluation_v63,
)
from fma.v6.predictive_quality import (
    ExternalAggregateQualityEvaluationV65,
    IntervalImplementationManifestV65,
    PredictiveQualityContractV65,
    PredictiveQualityError,
    PublicPredictiveQualityPackV65,
    assess_predictive_quality_v65,
    freeze_predictive_quality_contract_v65,
    freeze_public_predictive_quality_pack_v65,
    seal_external_aggregate_quality_evaluation_v65,
)


T0 = datetime(2026, 7, 28, tzinfo=timezone.utc)
ROLE_PRIVATE_KEYS = {
    role: Ed25519PrivateKey.generate()
    for role in ("custody", "registry", "evaluator", "promotion")
}
ROLE_KEY_IDS = {
    role: f"{role}-key"
    for role in ROLE_PRIVATE_KEYS
}
TRUSTED_PUBLIC_KEYS = {
    ROLE_KEY_IDS[role]: private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    for role, private_key in ROLE_PRIVATE_KEYS.items()
}
EVALUATOR_PRIVATE_KEY = ROLE_PRIVATE_KEYS["evaluator"].private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
)


def _hash(label: str) -> str:
    return sha256_value({"label": label})


def _implementation_manifest(
    *,
    adapter_id: str,
    protocol_hash: str,
) -> IntervalImplementationManifestV65:
    return IntervalImplementationManifestV65.seal(
        interval_adapter_id=adapter_id,
        interval_adapter_protocol_hash=protocol_hash,
        module_name="tests.synthetic_quality_adapter",
        module_source_sha256=_hash("quality-adapter-source"),
        loaded_callable_code_hashes={
            "build_intervals": _hash("build-intervals-callable"),
        },
        python_implementation="CPython",
        python_version="test-version",
        numpy_version="test-version",
        scipy_version="test-version",
        optimizer_policy="test_optimizer_policy",
        model_selection_policy="test_model_selection_policy",
    )


def _v63_contract() -> PredictiveExternalQualificationContractV63:
    fingerprints = {
        role: external_qualification_key_fingerprint_v63(
            TRUSTED_PUBLIC_KEYS[key_id]
        )
        for role, key_id in ROLE_KEY_IDS.items()
    }
    return PredictiveExternalQualificationContractV63.seal(
        qualification_id="qualification.v65",
        task_id="task.v65",
        local_context_hash=_hash("local-context"),
        workspace_spec_hash=_hash("workspace"),
        v62_report_hash=_hash("v62-report"),
        s4_gate_hash=_hash("s4"),
        s6_gate_hash=_hash("s6"),
        scientific_bundle_hash=_hash("bundle"),
        processed_snapshot_hash=_hash("snapshot"),
        executable_candidate_receipt_hash=_hash("executable"),
        selected_model_id="model.selected",
        selected_model_identity_hash=_hash("selected-model"),
        maximum_metric_value=0.20,
        minimum_external_observation_count=4,
        trusted_authority_key_ids=ROLE_KEY_IDS,
        trusted_authority_key_fingerprints=fingerprints,
        trusted_authority_set_hash=sha256_value(
            {
                "key_ids": ROLE_KEY_IDS,
                "fingerprints": fingerprints,
            }
        ),
        coordinator_host_id="coordinator-host",
        generator_host_id="generator-host",
        frozen_at=T0,
    )


def _quality_chain() -> tuple[
    PredictiveExternalQualificationContractV63,
    PredictiveQualityContractV65,
    ExternalPredictionVectorV63,
    PublicPredictiveQualityPackV65,
]:
    v63_contract = _v63_contract()
    interval_adapter_id = "training-only-empirical-v65"
    interval_adapter_protocol_hash = _hash("interval-adapter")
    contract = freeze_predictive_quality_contract_v65(
        v63_contract=v63_contract,
        quality_overlay_id="quality.v65",
        interval_alpha=0.10,
        minimum_mse_skill=0.10,
        minimum_interval_score_skill=0.10,
        minimum_empirical_coverage=0.75,
        maximum_absolute_coverage_error=0.11,
        maximum_normalized_interval_score=0.20,
        minimum_external_observation_count=4,
        interval_adapter_id=interval_adapter_id,
        interval_adapter_protocol_hash=interval_adapter_protocol_hash,
        interval_implementation_manifest=_implementation_manifest(
            adapter_id=interval_adapter_id,
            protocol_hash=interval_adapter_protocol_hash,
        ),
        frozen_at=T0,
    )
    target_ids = [f"target.{index}" for index in range(4)]
    predictions = [10.0, 11.0, 12.0, 13.0]
    vector = ExternalPredictionVectorV63.seal(
        qualification_id=v63_contract.qualification_id,
        local_context_hash=v63_contract.local_context_hash,
        selected_model_identity_hash=(v63_contract.selected_model_identity_hash),
        external_snapshot_hash=_hash("external-snapshot"),
        target_ids=target_ids,
        target_order_hash=sha256_value(target_ids),
        predictions=predictions,
        prediction_values_hash=sha256_value(predictions),
    )
    pack = freeze_public_predictive_quality_pack_v65(
        contract=contract,
        v63_contract=v63_contract,
        prediction_vector=vector,
        persistence_baseline_point=9.5,
        lower_bounds=[9.0, 10.0, 11.0, 12.0],
        upper_bounds=[11.0, 12.0, 13.0, 14.0],
        baseline_lower_bounds=[8.5, 8.5, 8.5, 8.5],
        baseline_upper_bounds=[10.5, 10.5, 10.5, 10.5],
        interval_calibration_receipt_hash=_hash("calibration"),
        fixture_only=False,
        packed_at=T0,
    )
    return v63_contract, contract, vector, pack


def _evaluation(
    *,
    contract: PredictiveQualityContractV65,
    pack: PublicPredictiveQualityPackV65,
    model_sse: float = 1.0,
    baseline_sse: float = 4.0,
    target_ss: float = 400.0,
    model_lower_miss_count: int = 0,
    model_upper_miss_count: int = 0,
    model_lower_shortfall_sum: float = 0.0,
    model_upper_excess_sum: float = 0.0,
    baseline_lower_miss_count: int = 1,
    baseline_upper_miss_count: int = 0,
    baseline_lower_shortfall_sum: float = 1.0,
    baseline_upper_excess_sum: float = 0.0,
) -> tuple[
    ExternalAggregateEvaluationV63,
    ExternalAggregateQualityEvaluationV65,
]:
    v63_evaluation = sign_external_aggregate_evaluation_v63(
        private_key_pem=EVALUATOR_PRIVATE_KEY,
        evaluation_id="evaluation.v63",
        qualification_id=contract.qualification_id,
        contract_hash=contract.v63_contract_hash,
        local_context_hash=contract.v63_local_context_hash,
        custody_hash=_hash("custody"),
        registration_hash=_hash("registration"),
        prediction_seal_hash=_hash("prediction-seal"),
        reservation_hash=_hash("reservation"),
        prediction_artifact_hash=_hash("prediction-artifact"),
        external_snapshot_hash=pack.external_snapshot_hash,
        holdout_commitment_hash=_hash("holdout"),
        normalization_scale_commitment_hash=sha256_value(
            {
                "holdout_observation_count": len(pack.target_ids),
                "target_squared_value_sum": target_ss,
            }
        ),
        target_order_hash=pack.target_order_hash,
        holdout_observation_count=len(pack.target_ids),
        squared_error_sum=model_sse,
        target_squared_value_sum=target_ss,
        aggregate_metric_value=math.sqrt(model_sse / target_ss),
        evaluator_host_id="evaluator-host",
        coordinator_host_id="coordinator-host",
        generator_host_id="generator-host",
        evaluator_key_id="evaluator-key",
        evaluated_at=T0,
    )
    quality_evaluation = seal_external_aggregate_quality_evaluation_v65(
        contract=contract,
        prediction_pack=pack,
        v63_evaluation=v63_evaluation,
        evaluation_id="evaluation.v65",
        baseline_squared_error_sum=baseline_sse,
        model_lower_miss_count=model_lower_miss_count,
        model_upper_miss_count=model_upper_miss_count,
        model_lower_shortfall_sum=model_lower_shortfall_sum,
        model_upper_excess_sum=model_upper_excess_sum,
        baseline_lower_miss_count=baseline_lower_miss_count,
        baseline_upper_miss_count=baseline_upper_miss_count,
        baseline_lower_shortfall_sum=baseline_lower_shortfall_sum,
        baseline_upper_excess_sum=baseline_upper_excess_sum,
        evaluator_private_key_pem=EVALUATOR_PRIVATE_KEY,
        evaluator_key_id=ROLE_KEY_IDS["evaluator"],
        evaluated_at=T0,
    )
    return v63_evaluation, quality_evaluation


def _assess(
    *,
    v63_contract: PredictiveExternalQualificationContractV63,
    contract: PredictiveQualityContractV65,
    vector: ExternalPredictionVectorV63,
    pack: PublicPredictiveQualityPackV65,
    evaluation: tuple[
        ExternalAggregateEvaluationV63,
        ExternalAggregateQualityEvaluationV65,
    ],
):
    v63_evaluation, quality_evaluation = evaluation
    return assess_predictive_quality_v65(
        contract=contract,
        v63_contract=v63_contract,
        v63_evaluation=v63_evaluation,
        prediction_vector=vector,
        prediction_pack=pack,
        evaluation=quality_evaluation,
        trusted_public_keys=TRUSTED_PUBLIC_KEYS,
    )


def test_quality_overlay_passes_all_recomputed_thresholds() -> None:
    v63_contract, contract, vector, pack = _quality_chain()
    result = _assess(
        v63_contract=v63_contract,
        contract=contract,
        vector=vector,
        pack=pack,
        evaluation=_evaluation(contract=contract, pack=pack),
    )

    assert result.status == "PASS"
    assert result.normalized_rmse == pytest.approx(0.05)
    assert result.mse_skill_over_persistence == pytest.approx(0.75)
    assert result.empirical_interval_coverage == pytest.approx(1.0)
    assert result.model_interval_score == pytest.approx(8.0)
    assert result.baseline_interval_score == pytest.approx(28.0)
    assert result.normalized_interval_score == pytest.approx(0.20)
    assert result.interval_score_skill_over_persistence == pytest.approx(
        1.0 - 8.0 / 28.0
    )
    assert result.absolute_coverage_error == pytest.approx(0.10)
    assert result.quality_overlay_passed is True
    assert result.scientific_qualification_granted is False
    assert result.real_world_action_authorized is False
    result.assert_sealed()


def test_good_nrmse_but_worse_than_persistence_is_rejected() -> None:
    v63_contract, contract, vector, pack = _quality_chain()
    result = _assess(
        v63_contract=v63_contract,
        contract=contract,
        vector=vector,
        pack=pack,
        evaluation=_evaluation(
            contract=contract,
            pack=pack,
            model_sse=4.0,
            baseline_sse=2.0,
        ),
    )

    assert result.normalized_rmse == pytest.approx(0.10)
    assert result.mse_skill_over_persistence == pytest.approx(-1.0)
    assert result.status == "REJECT"
    assert "persistence_baseline_skill_threshold_failed" in (result.reason_codes)


def test_undercoverage_is_rejected() -> None:
    v63_contract, contract, vector, pack = _quality_chain()
    result = _assess(
        v63_contract=v63_contract,
        contract=contract,
        vector=vector,
        pack=pack,
        evaluation=_evaluation(
            contract=contract,
            pack=pack,
            model_lower_miss_count=2,
            model_lower_shortfall_sum=0.1,
        ),
    )

    assert result.empirical_interval_coverage == pytest.approx(0.50)
    assert result.status == "REJECT"
    assert "empirical_interval_coverage_threshold_failed" in (result.reason_codes)


def test_excessive_normalized_interval_score_is_rejected() -> None:
    v63_contract, contract, vector, _ = _quality_chain()
    wide_pack = freeze_public_predictive_quality_pack_v65(
        contract=contract,
        v63_contract=v63_contract,
        prediction_vector=vector,
        persistence_baseline_point=9.5,
        lower_bounds=[5.0, 6.0, 7.0, 8.0],
        upper_bounds=[15.0, 16.0, 17.0, 18.0],
        baseline_lower_bounds=[4.5, 4.5, 4.5, 4.5],
        baseline_upper_bounds=[14.5, 14.5, 14.5, 14.5],
        interval_calibration_receipt_hash=_hash("wide-calibration"),
        fixture_only=False,
        packed_at=T0,
    )
    result = _assess(
        v63_contract=v63_contract,
        contract=contract,
        vector=vector,
        pack=wide_pack,
        evaluation=_evaluation(contract=contract, pack=wide_pack),
    )

    assert result.normalized_interval_score == pytest.approx(1.0)
    assert result.status == "REJECT"
    assert "normalized_interval_score_threshold_failed" in (result.reason_codes)


def test_interval_score_not_better_than_persistence_is_rejected() -> None:
    v63_contract, contract, vector, pack = _quality_chain()
    result = _assess(
        v63_contract=v63_contract,
        contract=contract,
        vector=vector,
        pack=pack,
        evaluation=_evaluation(
            contract=contract,
            pack=pack,
            baseline_lower_miss_count=0,
            baseline_lower_shortfall_sum=0.0,
        ),
    )

    assert result.interval_score_skill_over_persistence == pytest.approx(0.0)
    assert result.status == "REJECT"
    assert "persistence_interval_score_skill_threshold_failed" in (result.reason_codes)


def test_tampered_binding_fails_closed() -> None:
    v63_contract, contract, vector, pack = _quality_chain()
    tampered_pack = pack.model_copy(update={"persistence_baseline_point": 99.0})

    with pytest.raises(
        PredictiveQualityError,
        match="public pack is unsealed",
    ):
        _assess(
            v63_contract=v63_contract,
            contract=contract,
            vector=vector,
            pack=tampered_pack,
            evaluation=_evaluation(contract=contract, pack=pack),
        )


def test_forged_v63_evaluator_signature_is_rejected() -> None:
    v63_contract, contract, vector, pack = _quality_chain()
    v63_evaluation, quality_evaluation = _evaluation(
        contract=contract,
        pack=pack,
    )
    payload = v63_evaluation.model_dump(mode="json")
    payload["signature_base64"] = base64.b64encode(b"x" * 64).decode("ascii")
    payload["evaluation_hash"] = None
    forged_unsigned = ExternalAggregateEvaluationV63.model_validate(payload)
    payload["evaluation_hash"] = forged_unsigned.content_hash()
    forged = ExternalAggregateEvaluationV63.model_validate(payload)

    with pytest.raises(
        PredictiveQualityError,
        match="external aggregate evaluation authority rejected",
    ):
        assess_predictive_quality_v65(
            contract=contract,
            v63_contract=v63_contract,
            v63_evaluation=forged,
            prediction_vector=vector,
            prediction_pack=pack,
            evaluation=quality_evaluation,
            trusted_public_keys=TRUSTED_PUBLIC_KEYS,
        )


def test_valid_v63_evaluation_cannot_authorize_forged_v65_statistics() -> None:
    v63_contract, contract, vector, pack = _quality_chain()
    v63_evaluation, quality_evaluation = _evaluation(
        contract=contract,
        pack=pack,
    )
    payload = quality_evaluation.model_dump(mode="json")
    payload["baseline_squared_error_sum"] = 2.0
    payload["evaluation_hash"] = None
    forged_unsigned = ExternalAggregateQualityEvaluationV65.model_validate(
        payload
    )
    payload["evaluation_hash"] = forged_unsigned.content_hash()
    forged = ExternalAggregateQualityEvaluationV65.model_validate(payload)

    with pytest.raises(
        PredictiveQualityError,
        match="external aggregate evaluation authority rejected",
    ):
        assess_predictive_quality_v65(
            contract=contract,
            v63_contract=v63_contract,
            v63_evaluation=v63_evaluation,
            prediction_vector=vector,
            prediction_pack=pack,
            evaluation=forged,
            trusted_public_keys=TRUSTED_PUBLIC_KEYS,
        )


def test_runtime_authority_substitution_is_rejected() -> None:
    v63_contract, contract, vector, pack = _quality_chain()
    evaluation = _evaluation(contract=contract, pack=pack)
    substituted_keys = dict(TRUSTED_PUBLIC_KEYS)
    substituted_keys[ROLE_KEY_IDS["evaluator"]] = (
        Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )

    with pytest.raises(
        PredictiveQualityError,
        match="external aggregate evaluation authority rejected",
    ):
        v63_evaluation, quality_evaluation = evaluation
        assess_predictive_quality_v65(
            contract=contract,
            v63_contract=v63_contract,
            v63_evaluation=v63_evaluation,
            prediction_vector=vector,
            prediction_pack=pack,
            evaluation=quality_evaluation,
            trusted_public_keys=substituted_keys,
        )


def test_fixture_pack_is_diagnostic_and_rejected() -> None:
    v63_contract, contract, vector, pack = _quality_chain()
    fixture_pack = freeze_public_predictive_quality_pack_v65(
        contract=contract,
        v63_contract=v63_contract,
        prediction_vector=vector,
        persistence_baseline_point=pack.persistence_baseline_point,
        lower_bounds=pack.lower_bounds,
        upper_bounds=pack.upper_bounds,
        baseline_lower_bounds=pack.baseline_lower_bounds,
        baseline_upper_bounds=pack.baseline_upper_bounds,
        interval_calibration_receipt_hash=_hash("fixture-calibration"),
        fixture_only=True,
        packed_at=T0,
    )
    result = _assess(
        v63_contract=v63_contract,
        contract=contract,
        vector=vector,
        pack=fixture_pack,
        evaluation=_evaluation(contract=contract, pack=fixture_pack),
    )

    assert result.fixture_only is True
    assert result.status == "REJECT"
    assert "fixture_or_control_evidence_rejected" in result.reason_codes
    assert result.scientific_qualification_granted is False
    assert result.standalone_scientific_claim_authorized is False
    assert result.real_world_action_authorized is False


def test_normalized_interval_score_avoids_scale_product_overflow() -> None:
    v63_contract, contract, vector, pack = _quality_chain()
    result = _assess(
        v63_contract=v63_contract,
        contract=contract,
        vector=vector,
        pack=pack,
        evaluation=_evaluation(
            contract=contract,
            pack=pack,
            model_sse=1e306,
            baseline_sse=2e306,
            target_ss=1e308,
        ),
    )

    assert math.isfinite(result.normalized_interval_score)
    assert result.normalized_interval_score >= 0.0


def test_zero_baseline_denominator_is_rejected() -> None:
    _, contract, _, pack = _quality_chain()

    with pytest.raises(ValidationError):
        _evaluation(
            contract=contract,
            pack=pack,
            baseline_sse=0.0,
        )


def test_numerically_empty_baseline_denominator_fails_closed() -> None:
    v63_contract, contract, vector, pack = _quality_chain()

    with pytest.raises(PredictiveQualityError, match="stable improvement"):
        _assess(
            v63_contract=v63_contract,
            contract=contract,
            vector=vector,
            pack=pack,
            evaluation=_evaluation(
                contract=contract,
                pack=pack,
                model_sse=0.0,
                baseline_sse=1e-20,
            ),
        )


def test_public_pack_cannot_contain_private_target_values() -> None:
    _, _, _, pack = _quality_chain()

    assert "target_values" not in pack.model_dump(mode="json")
    with pytest.raises(ValidationError):
        type(pack).model_validate(
            {
                **pack.model_dump(mode="json"),
                "target_values": [10.0, 11.0, 12.0, 13.0],
            }
        )


def test_quality_pack_cannot_be_frozen_after_external_evaluation() -> None:
    v63_contract, contract, vector, _ = _quality_chain()
    late_pack = freeze_public_predictive_quality_pack_v65(
        contract=contract,
        v63_contract=v63_contract,
        prediction_vector=vector,
        persistence_baseline_point=9.5,
        lower_bounds=[9.0, 10.0, 11.0, 12.0],
        upper_bounds=[11.0, 12.0, 13.0, 14.0],
        baseline_lower_bounds=[8.5, 8.5, 8.5, 8.5],
        baseline_upper_bounds=[10.5, 10.5, 10.5, 10.5],
        interval_calibration_receipt_hash=_hash("late-calibration"),
        fixture_only=False,
        packed_at=T0 + timedelta(seconds=1),
    )
    with pytest.raises(PredictiveQualityError, match="chronology"):
        _evaluation(contract=contract, pack=late_pack)
