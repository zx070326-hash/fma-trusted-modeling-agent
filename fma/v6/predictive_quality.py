"""Additive V6.5 predictive-quality overlay for V6.3 point forecasts.

The overlay compares the registered model with a frozen persistence baseline
and evaluates interval calibration and sharpness from aggregate sufficient
statistics.  It never consumes private target values, grants scientific
qualification, or authorizes real-world action.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256

from .external_qualification import (
    ExternalAggregateEvaluationV63,
    ExternalQualificationError,
    ExternalPredictionVectorV63,
    PredictiveExternalQualificationContractV63,
    _assert_trusted_authority_set,
    _require_signed,
    _sign_model,
)


FiniteNonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
FinitePositive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Probability = Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
OpenProbability = Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)]
NonEmptyText = Annotated[str, Field(min_length=1)]
QualityStatusV65 = Literal["PASS", "REJECT"]
_MINIMUM_BASELINE_RELATIVE_MSE = 1e-15


class PredictiveQualityError(RuntimeError):
    """A fail-closed V6.5 integrity or binding error."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _assert_aware(value: datetime, field_name: str) -> None:
    if value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


class IntervalImplementationManifestV65(StrictModel):
    """Frozen code/runtime identity for one typed interval implementation."""

    schema_version: Literal["6.5-interval-implementation-manifest"] = (
        "6.5-interval-implementation-manifest"
    )
    interval_adapter_id: Identifier
    interval_adapter_protocol_hash: Sha256
    module_name: NonEmptyText
    module_source_sha256: Sha256
    loaded_callable_code_hashes: Annotated[
        dict[Identifier, Sha256],
        Field(min_length=1),
    ]
    python_implementation: NonEmptyText
    python_version: NonEmptyText
    numpy_version: NonEmptyText
    scipy_version: NonEmptyText
    optimizer_policy: NonEmptyText
    model_selection_policy: NonEmptyText
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "IntervalImplementationManifestV65":
        if list(self.loaded_callable_code_hashes) != sorted(
            self.loaded_callable_code_hashes
        ):
            raise ValueError(
                "V6.5 implementation callable identities must be sorted"
            )
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("V6.5 interval implementation manifest hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError(
                "V6.5 interval implementation manifest is not sealed"
            )

    @classmethod
    def seal(cls, **data: object) -> "IntervalImplementationManifestV65":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"manifest_hash"})
        payload["manifest_hash"] = draft.content_hash()
        return cls(**payload)


class PredictiveQualityContractV65(StrictModel):
    """Quality thresholds frozen before external targets are evaluated."""

    schema_version: Literal["6.5-predictive-quality-contract"] = (
        "6.5-predictive-quality-contract"
    )
    quality_overlay_id: Identifier
    qualification_id: Identifier
    task_id: Identifier
    v63_contract_hash: Sha256
    v63_contract_envelope_hash: Sha256
    v63_local_context_hash: Sha256
    v63_maximum_normalized_rmse: FiniteNonNegative
    v63_minimum_external_observation_count: Annotated[int, Field(ge=3)]
    v63_evaluator_key_id: Identifier
    v63_evaluator_key_fingerprint: Sha256
    baseline_policy: Literal["persistence_last_public_observation"] = (
        "persistence_last_public_observation"
    )
    baseline_interval_policy: Literal[
        "training_only_persistence_rolling_origin_empirical_intervals"
    ] = "training_only_persistence_rolling_origin_empirical_intervals"
    interval_evidence_kind: Literal[
        "rolling_origin_empirical_diagnostic"
    ] = "rolling_origin_empirical_diagnostic"
    finite_sample_coverage_guaranteed: Literal[False] = False
    temporal_dependence_coverage_guaranteed: Literal[False] = False
    post_selection_coverage_guaranteed: Literal[False] = False
    interval_claim_ceiling: Literal[
        "diagnostic_interval_quality_only"
    ] = "diagnostic_interval_quality_only"
    interval_adapter_id: Identifier
    interval_adapter_protocol_hash: Sha256
    interval_implementation_manifest: IntervalImplementationManifestV65
    interval_implementation_manifest_hash: Sha256
    interval_alpha: OpenProbability
    minimum_mse_skill: Probability
    minimum_interval_score_skill: Probability
    minimum_empirical_coverage: Probability
    maximum_absolute_coverage_error: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ]
    maximum_normalized_interval_score: FinitePositive
    minimum_external_observation_count: Annotated[int, Field(ge=3)]
    normalized_rmse_formula: Literal[
        "sqrt(model_squared_error_sum/target_squared_value_sum)"
    ] = "sqrt(model_squared_error_sum/target_squared_value_sum)"
    mse_skill_formula: Literal[
        "1-model_squared_error_sum/baseline_squared_error_sum"
    ] = "1-model_squared_error_sum/baseline_squared_error_sum"
    empirical_coverage_formula: Literal["1-interval_miss_count/observation_count"] = (
        "1-interval_miss_count/observation_count"
    )
    normalized_interval_score_formula: Literal[
        "model_interval_score/sqrt(observation_count*target_squared_value_sum)"
    ] = "model_interval_score/sqrt(observation_count*target_squared_value_sum)"
    interval_score_policy: Literal["winkler_interval_score"] = "winkler_interval_score"
    interval_score_skill_formula: Literal[
        "1-model_interval_score/baseline_interval_score"
    ] = "1-model_interval_score/baseline_interval_score"
    interval_width_source: Literal["recomputed_from_public_prediction_pack"] = (
        "recomputed_from_public_prediction_pack"
    )
    aggregate_feedback_only: Literal[True] = True
    frozen_before_private_evaluation: Literal[True] = True
    frozen_at: datetime
    contract_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "PredictiveQualityContractV65":
        if (
            self.minimum_external_observation_count
            < self.v63_minimum_external_observation_count
        ):
            raise ValueError("V6.5 minimum observation count cannot weaken V6.3")
        self.interval_implementation_manifest.assert_sealed()
        if (
            self.interval_implementation_manifest.interval_adapter_id
            != self.interval_adapter_id
            or self.interval_implementation_manifest.interval_adapter_protocol_hash
            != self.interval_adapter_protocol_hash
            or self.interval_implementation_manifest.manifest_hash
            != self.interval_implementation_manifest_hash
        ):
            raise ValueError(
                "V6.5 interval implementation differs from the frozen adapter"
            )
        _assert_aware(self.frozen_at, "frozen_at")
        if self.contract_hash and self.contract_hash != self.content_hash():
            raise ValueError("V6.5 predictive-quality contract hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "contract_hash")

    def assert_sealed(self) -> None:
        if not self.contract_hash or self.contract_hash != self.content_hash():
            raise ValueError("V6.5 predictive-quality contract is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PredictiveQualityContractV65":
        data.setdefault("frozen_at", _utc_now())
        draft = cls(**data)
        payload = draft.model_dump(exclude={"contract_hash"})
        payload["contract_hash"] = draft.content_hash()
        return cls(**payload)


class PublicPredictiveQualityPackV65(StrictModel):
    """Public point, baseline, and interval forecasts without target values."""

    schema_version: Literal["6.5-public-predictive-quality-pack"] = (
        "6.5-public-predictive-quality-pack"
    )
    quality_overlay_id: Identifier
    qualification_id: Identifier
    task_id: Identifier
    quality_contract_hash: Sha256
    v63_contract_hash: Sha256
    v63_prediction_vector_hash: Sha256
    v63_prediction_vector_envelope_hash: Sha256
    processed_snapshot_hash: Sha256
    external_snapshot_hash: Sha256
    target_ids: Annotated[list[Identifier], Field(min_length=1)]
    target_order_hash: Sha256
    model_point_predictions: Annotated[list[FinitePositive], Field(min_length=1)]
    model_prediction_values_hash: Sha256
    persistence_baseline_point: FinitePositive
    persistence_baseline_binding_hash: Sha256
    interval_alpha: OpenProbability
    interval_adapter_id: Identifier
    interval_adapter_protocol_hash: Sha256
    interval_calibration_receipt_hash: Sha256
    lower_bounds: Annotated[list[FiniteNonNegative], Field(min_length=1)]
    upper_bounds: Annotated[list[FinitePositive], Field(min_length=1)]
    baseline_lower_bounds: Annotated[list[FiniteNonNegative], Field(min_length=1)]
    baseline_upper_bounds: Annotated[list[FinitePositive], Field(min_length=1)]
    interval_bounds_hash: Sha256
    private_target_values_included: Literal[False] = False
    fixture_only: bool
    packed_at: datetime
    pack_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_pack(self) -> "PublicPredictiveQualityPackV65":
        _assert_aware(self.packed_at, "packed_at")
        count = len(self.target_ids)
        if len(set(self.target_ids)) != count:
            raise ValueError("V6.5 target IDs must be unique")
        if not (
            len(self.model_point_predictions)
            == len(self.lower_bounds)
            == len(self.upper_bounds)
            == len(self.baseline_lower_bounds)
            == len(self.baseline_upper_bounds)
            == count
        ):
            raise ValueError("V6.5 public prediction lengths differ")
        if self.target_order_hash != sha256_value(self.target_ids):
            raise ValueError("V6.5 target-order hash differs")
        if self.model_prediction_values_hash != sha256_value(
            self.model_point_predictions
        ):
            raise ValueError("V6.5 model-prediction hash differs")
        if self.persistence_baseline_binding_hash != sha256_value(
            {
                "baseline_policy": ("persistence_last_public_observation"),
                "processed_snapshot_hash": self.processed_snapshot_hash,
                "persistence_baseline_point": (self.persistence_baseline_point),
            }
        ):
            raise ValueError("V6.5 persistence-baseline binding differs")
        if self.interval_bounds_hash != sha256_value(
            {
                "alpha": self.interval_alpha,
                "lower_bounds": self.lower_bounds,
                "upper_bounds": self.upper_bounds,
                "baseline_lower_bounds": self.baseline_lower_bounds,
                "baseline_upper_bounds": self.baseline_upper_bounds,
            }
        ):
            raise ValueError("V6.5 interval-bounds hash differs")
        if any(
            not lower <= point <= upper
            for lower, point, upper in zip(
                self.lower_bounds,
                self.model_point_predictions,
                self.upper_bounds,
            )
        ):
            raise ValueError("V6.5 intervals must contain their model point forecasts")
        if any(
            lower >= upper for lower, upper in zip(self.lower_bounds, self.upper_bounds)
        ):
            raise ValueError("V6.5 model intervals must have positive width")
        if any(
            not lower <= self.persistence_baseline_point <= upper
            for lower, upper in zip(
                self.baseline_lower_bounds,
                self.baseline_upper_bounds,
            )
        ):
            raise ValueError("V6.5 baseline intervals must contain persistence")
        if any(
            lower >= upper
            for lower, upper in zip(
                self.baseline_lower_bounds,
                self.baseline_upper_bounds,
            )
        ):
            raise ValueError("V6.5 baseline intervals must have positive width")
        if self.pack_hash and self.pack_hash != self.content_hash():
            raise ValueError("V6.5 public prediction pack hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "pack_hash")

    def assert_sealed(self) -> None:
        if not self.pack_hash or self.pack_hash != self.content_hash():
            raise ValueError("V6.5 public prediction pack is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PublicPredictiveQualityPackV65":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"pack_hash"})
        payload["pack_hash"] = draft.content_hash()
        return cls(**payload)


class ExternalAggregateQualityEvaluationV65(StrictModel):
    """Aggregate-only private evaluation sufficient statistics.

    The only outcome-dependent values are aggregate SSEs, target square sum,
    directional miss counts, and directional miss magnitudes.  Public interval
    widths are recomputed from the bound prediction pack.  No per-observation
    target or loss is accepted by this schema.
    """

    schema_version: Literal["6.5-external-aggregate-quality-evaluation"] = (
        "6.5-external-aggregate-quality-evaluation"
    )
    evaluation_id: Identifier
    quality_overlay_id: Identifier
    qualification_id: Identifier
    task_id: Identifier
    quality_contract_hash: Sha256
    public_prediction_pack_hash: Sha256
    v63_evaluation_hash: Sha256
    v63_contract_hash: Sha256
    v63_prediction_vector_hash: Sha256
    external_snapshot_hash: Sha256
    target_order_hash: Sha256
    observation_count: Annotated[int, Field(ge=1)]
    model_squared_error_sum: FiniteNonNegative
    baseline_squared_error_sum: FinitePositive
    target_squared_value_sum: FinitePositive
    model_lower_miss_count: Annotated[int, Field(ge=0)]
    model_upper_miss_count: Annotated[int, Field(ge=0)]
    model_lower_shortfall_sum: FiniteNonNegative
    model_upper_excess_sum: FiniteNonNegative
    baseline_lower_miss_count: Annotated[int, Field(ge=0)]
    baseline_upper_miss_count: Annotated[int, Field(ge=0)]
    baseline_lower_shortfall_sum: FiniteNonNegative
    baseline_upper_excess_sum: FiniteNonNegative
    aggregate_only: Literal[True] = True
    per_observation_feedback_released: Literal[False] = False
    per_observation_target_values_released: Literal[False] = False
    aggregate_statistics_release_scope: Literal[
        "trusted_evaluator_to_coordinator_only"
    ] = "trusted_evaluator_to_coordinator_only"
    privacy_guarantee: Literal["no_formal_privacy_guarantee"] = (
        "no_formal_privacy_guarantee"
    )
    aggregate_statistics_may_leak_information: Literal[True] = True
    evaluator_host_id: Identifier
    evaluator_key_id: Identifier
    evaluated_at: datetime
    signature_base64: Annotated[str, Field(min_length=40)] | None = None
    evaluation_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_evaluation(
        self,
    ) -> "ExternalAggregateQualityEvaluationV65":
        miss_pairs = (
            (
                self.model_lower_miss_count,
                self.model_upper_miss_count,
                self.model_lower_shortfall_sum,
                self.model_upper_excess_sum,
                "model",
            ),
            (
                self.baseline_lower_miss_count,
                self.baseline_upper_miss_count,
                self.baseline_lower_shortfall_sum,
                self.baseline_upper_excess_sum,
                "baseline",
            ),
        )
        for lower_count, upper_count, lower_sum, upper_sum, label in miss_pairs:
            if lower_count + upper_count > self.observation_count:
                raise ValueError(
                    f"V6.5 {label} interval misses exceed observation count"
                )
            if (lower_count == 0) != (lower_sum == 0):
                raise ValueError(f"V6.5 {label} lower misses and shortfall differ")
            if (upper_count == 0) != (upper_sum == 0):
                raise ValueError(f"V6.5 {label} upper misses and excess differ")
        _assert_aware(self.evaluated_at, "evaluated_at")
        if self.evaluation_hash and (
            not self.signature_base64
            or self.evaluation_hash != self.content_hash()
        ):
            raise ValueError("V6.5 aggregate evaluation signature envelope differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evaluation_hash")

    def assert_sealed(self) -> None:
        if (
            not self.signature_base64
            or not self.evaluation_hash
            or self.evaluation_hash != self.content_hash()
        ):
            raise ValueError("V6.5 aggregate evaluation is not sealed")


class PredictiveQualityAssessmentV65(StrictModel):
    """Code-derived quality result with no qualification or action authority."""

    schema_version: Literal["6.5-predictive-quality-assessment"] = (
        "6.5-predictive-quality-assessment"
    )
    quality_overlay_id: Identifier
    qualification_id: Identifier
    task_id: Identifier
    quality_contract_hash: Sha256
    public_prediction_pack_hash: Sha256
    external_evaluation_hash: Sha256
    v63_contract_hash: Sha256
    v63_prediction_vector_hash: Sha256
    observation_count: Annotated[int, Field(ge=1)]
    normalized_rmse: FiniteNonNegative
    mse_skill_over_persistence: Annotated[float, Field(allow_inf_nan=False)]
    empirical_interval_coverage: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ]
    baseline_empirical_interval_coverage: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ]
    absolute_coverage_error: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    model_interval_score: FinitePositive
    baseline_interval_score: FinitePositive
    normalized_interval_score: FiniteNonNegative
    interval_score_skill_over_persistence: Annotated[float, Field(allow_inf_nan=False)]
    checks: dict[Identifier, bool]
    status: QualityStatusV65
    reason_codes: list[Identifier]
    quality_overlay_passed: bool
    fixture_only: bool
    interval_evidence_kind: Literal[
        "rolling_origin_empirical_diagnostic"
    ] = "rolling_origin_empirical_diagnostic"
    interval_claim_ceiling: Literal[
        "diagnostic_interval_quality_only"
    ] = "diagnostic_interval_quality_only"
    finite_sample_coverage_guaranteed: Literal[False] = False
    temporal_dependence_coverage_guaranteed: Literal[False] = False
    post_selection_coverage_guaranteed: Literal[False] = False
    requires_separate_v63_qualification: Literal[True] = True
    scientific_qualification_granted: Literal[False] = False
    standalone_scientific_claim_authorized: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    assessment_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_assessment(self) -> "PredictiveQualityAssessmentV65":
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("V6.5 assessment reasons must be sorted and unique")
        passed = bool(self.checks) and all(self.checks.values())
        if self.quality_overlay_passed != passed:
            raise ValueError("V6.5 quality flag differs from checks")
        if self.status != ("PASS" if passed else "REJECT"):
            raise ValueError("V6.5 quality status differs from checks")
        if passed and self.reason_codes:
            raise ValueError("V6.5 passing assessment has rejection reasons")
        if not passed and not self.reason_codes:
            raise ValueError("V6.5 rejected assessment lacks reason codes")
        if self.assessment_hash and (self.assessment_hash != self.content_hash()):
            raise ValueError("V6.5 assessment hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "assessment_hash")

    def assert_sealed(self) -> None:
        if not self.assessment_hash or self.assessment_hash != self.content_hash():
            raise ValueError("V6.5 predictive-quality assessment is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PredictiveQualityAssessmentV65":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"assessment_hash"})
        payload["assessment_hash"] = draft.content_hash()
        return cls(**payload)


def _assert_v63_contract_binding(
    *,
    contract: PredictiveQualityContractV65,
    v63_contract: PredictiveExternalQualificationContractV63,
) -> None:
    try:
        contract.assert_sealed()
        v63_contract.assert_sealed()
    except ValueError as exc:
        raise PredictiveQualityError("V6.5 or V6.3 contract is unsealed") from exc
    expected = {
        "qualification_id": v63_contract.qualification_id,
        "task_id": v63_contract.task_id,
        "v63_contract_hash": v63_contract.contract_hash,
        "v63_contract_envelope_hash": sha256_value(v63_contract),
        "v63_local_context_hash": v63_contract.local_context_hash,
        "v63_maximum_normalized_rmse": (v63_contract.maximum_metric_value),
        "v63_minimum_external_observation_count": (
            v63_contract.minimum_external_observation_count
        ),
        "v63_evaluator_key_id": (
            v63_contract.trusted_authority_key_ids["evaluator"]
        ),
        "v63_evaluator_key_fingerprint": (
            v63_contract.trusted_authority_key_fingerprints["evaluator"]
        ),
    }
    actual = {key: getattr(contract, key) for key in expected}
    if actual != expected:
        raise PredictiveQualityError(
            "V6.5 contract does not bind the supplied V6.3 contract"
        )


def freeze_predictive_quality_contract_v65(
    *,
    v63_contract: PredictiveExternalQualificationContractV63,
    quality_overlay_id: str,
    interval_alpha: float,
    minimum_mse_skill: float,
    minimum_interval_score_skill: float,
    minimum_empirical_coverage: float,
    maximum_absolute_coverage_error: float,
    maximum_normalized_interval_score: float,
    minimum_external_observation_count: int,
    interval_adapter_id: str,
    interval_adapter_protocol_hash: str,
    interval_implementation_manifest: IntervalImplementationManifestV65,
    frozen_at: datetime | None = None,
) -> PredictiveQualityContractV65:
    """Freeze an additive quality policy around one sealed V6.3 contract."""

    try:
        v63_contract.assert_sealed()
    except ValueError as exc:
        raise PredictiveQualityError("V6.3 contract is unsealed") from exc
    return PredictiveQualityContractV65.seal(
        quality_overlay_id=quality_overlay_id,
        qualification_id=v63_contract.qualification_id,
        task_id=v63_contract.task_id,
        v63_contract_hash=v63_contract.contract_hash,
        v63_contract_envelope_hash=sha256_value(v63_contract),
        v63_local_context_hash=v63_contract.local_context_hash,
        v63_maximum_normalized_rmse=(v63_contract.maximum_metric_value),
        v63_minimum_external_observation_count=(
            v63_contract.minimum_external_observation_count
        ),
        v63_evaluator_key_id=(
            v63_contract.trusted_authority_key_ids["evaluator"]
        ),
        v63_evaluator_key_fingerprint=(
            v63_contract.trusted_authority_key_fingerprints["evaluator"]
        ),
        interval_adapter_id=interval_adapter_id,
        interval_adapter_protocol_hash=interval_adapter_protocol_hash,
        interval_implementation_manifest=interval_implementation_manifest,
        interval_implementation_manifest_hash=(
            interval_implementation_manifest.manifest_hash
        ),
        interval_alpha=interval_alpha,
        minimum_mse_skill=minimum_mse_skill,
        minimum_interval_score_skill=minimum_interval_score_skill,
        minimum_empirical_coverage=minimum_empirical_coverage,
        maximum_absolute_coverage_error=(maximum_absolute_coverage_error),
        maximum_normalized_interval_score=(maximum_normalized_interval_score),
        minimum_external_observation_count=(minimum_external_observation_count),
        frozen_at=frozen_at or _utc_now(),
    )


def _assert_prediction_pack_binding(
    *,
    contract: PredictiveQualityContractV65,
    v63_contract: PredictiveExternalQualificationContractV63,
    prediction_vector: ExternalPredictionVectorV63,
    prediction_pack: PublicPredictiveQualityPackV65,
) -> None:
    _assert_v63_contract_binding(
        contract=contract,
        v63_contract=v63_contract,
    )
    try:
        prediction_vector.assert_sealed()
        prediction_pack.assert_sealed()
    except ValueError as exc:
        raise PredictiveQualityError(
            "V6.3 prediction vector or V6.5 public pack is unsealed"
        ) from exc
    expected = {
        "quality_overlay_id": contract.quality_overlay_id,
        "qualification_id": contract.qualification_id,
        "task_id": contract.task_id,
        "quality_contract_hash": contract.contract_hash,
        "v63_contract_hash": contract.v63_contract_hash,
        "v63_prediction_vector_hash": prediction_vector.vector_hash,
        "v63_prediction_vector_envelope_hash": sha256_value(prediction_vector),
        "processed_snapshot_hash": v63_contract.processed_snapshot_hash,
        "external_snapshot_hash": prediction_vector.external_snapshot_hash,
        "target_ids": prediction_vector.target_ids,
        "target_order_hash": prediction_vector.target_order_hash,
        "model_point_predictions": prediction_vector.predictions,
        "model_prediction_values_hash": (prediction_vector.prediction_values_hash),
        "interval_alpha": contract.interval_alpha,
        "interval_adapter_id": contract.interval_adapter_id,
        "interval_adapter_protocol_hash": (contract.interval_adapter_protocol_hash),
    }
    actual = {key: getattr(prediction_pack, key) for key in expected}
    if actual != expected:
        raise PredictiveQualityError(
            "V6.5 public pack does not bind its contract and V6.3 vector"
        )
    if (
        prediction_vector.qualification_id != contract.qualification_id
        or prediction_vector.local_context_hash != contract.v63_local_context_hash
        or prediction_vector.selected_model_identity_hash
        != v63_contract.selected_model_identity_hash
    ):
        raise PredictiveQualityError(
            "V6.3 prediction vector is outside the frozen contract"
        )


def freeze_public_predictive_quality_pack_v65(
    *,
    contract: PredictiveQualityContractV65,
    v63_contract: PredictiveExternalQualificationContractV63,
    prediction_vector: ExternalPredictionVectorV63,
    persistence_baseline_point: float,
    lower_bounds: list[float],
    upper_bounds: list[float],
    baseline_lower_bounds: list[float],
    baseline_upper_bounds: list[float],
    interval_calibration_receipt_hash: str,
    fixture_only: bool,
    packed_at: datetime | None = None,
) -> PublicPredictiveQualityPackV65:
    """Freeze the target-free public predictions used by the evaluator."""

    _assert_v63_contract_binding(
        contract=contract,
        v63_contract=v63_contract,
    )
    try:
        prediction_vector.assert_sealed()
    except ValueError as exc:
        raise PredictiveQualityError("V6.3 prediction vector is unsealed") from exc
    pack = PublicPredictiveQualityPackV65.seal(
        quality_overlay_id=contract.quality_overlay_id,
        qualification_id=contract.qualification_id,
        task_id=contract.task_id,
        quality_contract_hash=contract.contract_hash,
        v63_contract_hash=contract.v63_contract_hash,
        v63_prediction_vector_hash=prediction_vector.vector_hash,
        v63_prediction_vector_envelope_hash=sha256_value(prediction_vector),
        processed_snapshot_hash=v63_contract.processed_snapshot_hash,
        external_snapshot_hash=prediction_vector.external_snapshot_hash,
        target_ids=prediction_vector.target_ids,
        target_order_hash=prediction_vector.target_order_hash,
        model_point_predictions=prediction_vector.predictions,
        model_prediction_values_hash=(prediction_vector.prediction_values_hash),
        persistence_baseline_point=persistence_baseline_point,
        persistence_baseline_binding_hash=sha256_value(
            {
                "baseline_policy": ("persistence_last_public_observation"),
                "processed_snapshot_hash": (v63_contract.processed_snapshot_hash),
                "persistence_baseline_point": (persistence_baseline_point),
            }
        ),
        interval_alpha=contract.interval_alpha,
        interval_adapter_id=contract.interval_adapter_id,
        interval_adapter_protocol_hash=(contract.interval_adapter_protocol_hash),
        interval_calibration_receipt_hash=(interval_calibration_receipt_hash),
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        baseline_lower_bounds=baseline_lower_bounds,
        baseline_upper_bounds=baseline_upper_bounds,
        interval_bounds_hash=sha256_value(
            {
                "alpha": contract.interval_alpha,
                "lower_bounds": lower_bounds,
                "upper_bounds": upper_bounds,
                "baseline_lower_bounds": baseline_lower_bounds,
                "baseline_upper_bounds": baseline_upper_bounds,
            }
        ),
        fixture_only=fixture_only,
        packed_at=packed_at or _utc_now(),
    )
    if pack.packed_at < contract.frozen_at:
        raise PredictiveQualityError(
            "V6.5 public prediction pack predates its frozen contract"
        )
    _assert_prediction_pack_binding(
        contract=contract,
        v63_contract=v63_contract,
        prediction_vector=prediction_vector,
        prediction_pack=pack,
    )
    return pack


def _assert_evaluation_binding(
    *,
    contract: PredictiveQualityContractV65,
    prediction_pack: PublicPredictiveQualityPackV65,
    v63_evaluation: ExternalAggregateEvaluationV63,
    evaluation: ExternalAggregateQualityEvaluationV65,
) -> None:
    try:
        v63_evaluation.assert_sealed()
        evaluation.assert_sealed()
    except ValueError as exc:
        raise PredictiveQualityError(
            "V6.3 or V6.5 aggregate evaluation is unsealed"
        ) from exc
    expected = {
        "quality_overlay_id": contract.quality_overlay_id,
        "qualification_id": contract.qualification_id,
        "task_id": contract.task_id,
        "quality_contract_hash": contract.contract_hash,
        "public_prediction_pack_hash": prediction_pack.pack_hash,
        "v63_evaluation_hash": v63_evaluation.evaluation_hash,
        "v63_contract_hash": contract.v63_contract_hash,
        "v63_prediction_vector_hash": (prediction_pack.v63_prediction_vector_hash),
        "external_snapshot_hash": prediction_pack.external_snapshot_hash,
        "target_order_hash": prediction_pack.target_order_hash,
        "observation_count": len(prediction_pack.target_ids),
        "evaluator_host_id": v63_evaluation.evaluator_host_id,
        "evaluator_key_id": v63_evaluation.evaluator_key_id,
    }
    actual = {key: getattr(evaluation, key) for key in expected}
    if actual != expected:
        raise PredictiveQualityError(
            "V6.5 aggregate evaluation does not bind the public pack"
        )
    if (
        v63_evaluation.qualification_id != contract.qualification_id
        or v63_evaluation.contract_hash != contract.v63_contract_hash
        or v63_evaluation.local_context_hash != contract.v63_local_context_hash
        or v63_evaluation.external_snapshot_hash
        != prediction_pack.external_snapshot_hash
        or v63_evaluation.target_order_hash != prediction_pack.target_order_hash
        or v63_evaluation.holdout_observation_count != len(prediction_pack.target_ids)
        or v63_evaluation.squared_error_sum != evaluation.model_squared_error_sum
        or v63_evaluation.target_squared_value_sum
        != evaluation.target_squared_value_sum
        or not v63_evaluation.aggregate_only
        or v63_evaluation.per_observation_feedback_released
        or v63_evaluation.private_values_disclosed
    ):
        raise PredictiveQualityError(
            "V6.5 statistics differ from the bound V6.3 evaluation"
        )
    if not (
        contract.frozen_at
        <= prediction_pack.packed_at
        <= v63_evaluation.evaluated_at
        <= evaluation.evaluated_at
    ):
        raise PredictiveQualityError(
            "V6.5 contract, pack, and evaluation chronology is invalid"
        )


def seal_external_aggregate_quality_evaluation_v65(
    *,
    contract: PredictiveQualityContractV65,
    prediction_pack: PublicPredictiveQualityPackV65,
    v63_evaluation: ExternalAggregateEvaluationV63,
    evaluation_id: str,
    baseline_squared_error_sum: float,
    model_lower_miss_count: int,
    model_upper_miss_count: int,
    model_lower_shortfall_sum: float,
    model_upper_excess_sum: float,
    baseline_lower_miss_count: int,
    baseline_upper_miss_count: int,
    baseline_lower_shortfall_sum: float,
    baseline_upper_excess_sum: float,
    evaluator_private_key_pem: bytes,
    evaluator_key_id: str,
    evaluated_at: datetime | None = None,
) -> ExternalAggregateQualityEvaluationV65:
    """Sign externally computed aggregate statistics without target values."""

    try:
        contract.assert_sealed()
        prediction_pack.assert_sealed()
        v63_evaluation.assert_sealed()
    except ValueError as exc:
        raise PredictiveQualityError(
            "V6.5 contract, public pack, or V6.3 evaluation is unsealed"
        ) from exc
    if (
        evaluator_key_id != contract.v63_evaluator_key_id
        or evaluator_key_id != v63_evaluation.evaluator_key_id
    ):
        raise PredictiveQualityError(
            "V6.5 aggregate evaluator differs from the frozen evaluator"
        )
    effective_evaluated_at = evaluated_at or _utc_now()
    signed = _sign_model(
        model_type=ExternalAggregateQualityEvaluationV65,
        data={
            "evaluation_id": evaluation_id,
            "quality_overlay_id": contract.quality_overlay_id,
            "qualification_id": contract.qualification_id,
            "task_id": contract.task_id,
            "quality_contract_hash": contract.contract_hash,
            "public_prediction_pack_hash": prediction_pack.pack_hash,
            "v63_evaluation_hash": v63_evaluation.evaluation_hash,
            "v63_contract_hash": contract.v63_contract_hash,
            "v63_prediction_vector_hash": (
                prediction_pack.v63_prediction_vector_hash
            ),
            "external_snapshot_hash": prediction_pack.external_snapshot_hash,
            "target_order_hash": prediction_pack.target_order_hash,
            "observation_count": len(prediction_pack.target_ids),
            "model_squared_error_sum": v63_evaluation.squared_error_sum,
            "baseline_squared_error_sum": baseline_squared_error_sum,
            "target_squared_value_sum": (
                v63_evaluation.target_squared_value_sum
            ),
            "model_lower_miss_count": model_lower_miss_count,
            "model_upper_miss_count": model_upper_miss_count,
            "model_lower_shortfall_sum": model_lower_shortfall_sum,
            "model_upper_excess_sum": model_upper_excess_sum,
            "baseline_lower_miss_count": baseline_lower_miss_count,
            "baseline_upper_miss_count": baseline_upper_miss_count,
            "baseline_lower_shortfall_sum": baseline_lower_shortfall_sum,
            "baseline_upper_excess_sum": baseline_upper_excess_sum,
            "evaluator_host_id": v63_evaluation.evaluator_host_id,
            "evaluator_key_id": evaluator_key_id,
            "evaluated_at": effective_evaluated_at,
        },
        private_key_pem=evaluator_private_key_pem,
        hash_field="evaluation_hash",
    )
    if not isinstance(signed, ExternalAggregateQualityEvaluationV65):
        raise PredictiveQualityError("V6.5 aggregate evaluation signing failed")
    evaluation = signed
    _assert_evaluation_binding(
        contract=contract,
        prediction_pack=prediction_pack,
        v63_evaluation=v63_evaluation,
        evaluation=evaluation,
    )
    return evaluation


def _recompute_metrics(
    *,
    contract: PredictiveQualityContractV65,
    prediction_pack: PublicPredictiveQualityPackV65,
    evaluation: ExternalAggregateQualityEvaluationV65,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    if (
        evaluation.baseline_squared_error_sum / evaluation.target_squared_value_sum
        <= _MINIMUM_BASELINE_RELATIVE_MSE
    ):
        raise PredictiveQualityError(
            "V6.5 persistence baseline leaves no stable improvement denominator"
        )
    normalized_rmse = math.sqrt(
        evaluation.model_squared_error_sum / evaluation.target_squared_value_sum
    )
    mse_skill = (
        1.0 - evaluation.model_squared_error_sum / evaluation.baseline_squared_error_sum
    )
    empirical_coverage = (
        1.0
        - (evaluation.model_lower_miss_count + evaluation.model_upper_miss_count)
        / evaluation.observation_count
    )
    baseline_empirical_coverage = (
        1.0
        - (evaluation.baseline_lower_miss_count + evaluation.baseline_upper_miss_count)
        / evaluation.observation_count
    )
    absolute_coverage_error = abs(empirical_coverage - (1.0 - contract.interval_alpha))
    model_width_sum = sum(
        upper - lower
        for lower, upper in zip(
            prediction_pack.lower_bounds,
            prediction_pack.upper_bounds,
        )
    )
    baseline_width_sum = sum(
        upper - lower
        for lower, upper in zip(
            prediction_pack.baseline_lower_bounds,
            prediction_pack.baseline_upper_bounds,
        )
    )
    penalty_scale = 2.0 / contract.interval_alpha
    model_interval_score = model_width_sum + penalty_scale * (
        evaluation.model_lower_shortfall_sum + evaluation.model_upper_excess_sum
    )
    baseline_interval_score = baseline_width_sum + penalty_scale * (
        evaluation.baseline_lower_shortfall_sum + evaluation.baseline_upper_excess_sum
    )
    if baseline_interval_score <= 0:
        raise PredictiveQualityError(
            "V6.5 baseline interval-score denominator is not positive"
        )
    # Divide in two stages so the mathematically equivalent normalizer does
    # not first overflow as ``observation_count * target_squared_value_sum``.
    normalized_interval_score = (
        model_interval_score / math.sqrt(evaluation.target_squared_value_sum)
    ) / math.sqrt(evaluation.observation_count)
    interval_score_skill = 1.0 - model_interval_score / baseline_interval_score
    return (
        normalized_rmse,
        mse_skill,
        empirical_coverage,
        baseline_empirical_coverage,
        absolute_coverage_error,
        model_interval_score,
        baseline_interval_score,
        normalized_interval_score,
        interval_score_skill,
    )


def assess_predictive_quality_v65(
    *,
    contract: PredictiveQualityContractV65,
    v63_contract: PredictiveExternalQualificationContractV63,
    v63_evaluation: ExternalAggregateEvaluationV63,
    prediction_vector: ExternalPredictionVectorV63,
    prediction_pack: PublicPredictiveQualityPackV65,
    evaluation: ExternalAggregateQualityEvaluationV65,
    trusted_public_keys: Mapping[str, bytes],
) -> PredictiveQualityAssessmentV65:
    """Recompute every threshold and issue a non-authoritative PASS/REJECT."""

    _assert_prediction_pack_binding(
        contract=contract,
        v63_contract=v63_contract,
        prediction_vector=prediction_vector,
        prediction_pack=prediction_pack,
    )
    try:
        _assert_trusted_authority_set(
            contract=v63_contract,
            trusted_public_keys=trusted_public_keys,
        )
        if (
            v63_evaluation.evaluator_key_id
            != v63_contract.trusted_authority_key_ids["evaluator"]
        ):
            raise ExternalQualificationError(
                "V6.3 evaluator key differs from the frozen evaluator"
            )
        _require_signed(
            model=v63_evaluation,
            key_id=v63_evaluation.evaluator_key_id,
            signature_base64=v63_evaluation.signature_base64,
            trusted_public_keys=trusted_public_keys,
            hash_field="evaluation_hash",
            label="V6.3 external aggregate evaluation",
        )
        if (
            evaluation.evaluator_key_id != contract.v63_evaluator_key_id
            or evaluation.evaluator_key_id != v63_evaluation.evaluator_key_id
            or evaluation.evaluator_host_id != v63_evaluation.evaluator_host_id
        ):
            raise ExternalQualificationError(
                "V6.5 evaluator differs from the frozen V6.3 evaluator"
            )
        _require_signed(
            model=evaluation,
            key_id=evaluation.evaluator_key_id,
            signature_base64=evaluation.signature_base64,
            trusted_public_keys=trusted_public_keys,
            hash_field="evaluation_hash",
            label="V6.5 external aggregate quality evaluation",
        )
    except ExternalQualificationError as exc:
        raise PredictiveQualityError(
            "external aggregate evaluation authority rejected"
        ) from exc
    _assert_evaluation_binding(
        contract=contract,
        prediction_pack=prediction_pack,
        v63_evaluation=v63_evaluation,
        evaluation=evaluation,
    )
    (
        normalized_rmse,
        mse_skill,
        empirical_coverage,
        baseline_empirical_coverage,
        absolute_coverage_error,
        model_interval_score,
        baseline_interval_score,
        normalized_interval_score,
        interval_score_skill,
    ) = _recompute_metrics(
        contract=contract,
        prediction_pack=prediction_pack,
        evaluation=evaluation,
    )
    checks = {
        "minimum_observation_count_met": (
            evaluation.observation_count >= contract.minimum_external_observation_count
        ),
        "v63_normalized_rmse_met": (
            normalized_rmse <= contract.v63_maximum_normalized_rmse
        ),
        "persistence_mse_skill_met": (mse_skill >= contract.minimum_mse_skill),
        "empirical_interval_coverage_met": (
            empirical_coverage >= contract.minimum_empirical_coverage
        ),
        "absolute_coverage_error_met": (
            absolute_coverage_error <= contract.maximum_absolute_coverage_error
        ),
        "normalized_interval_score_met": (
            normalized_interval_score <= contract.maximum_normalized_interval_score
        ),
        "persistence_interval_score_skill_met": (
            interval_score_skill >= contract.minimum_interval_score_skill
        ),
        "non_fixture_evidence": (not prediction_pack.fixture_only),
    }
    reason_by_check = {
        "minimum_observation_count_met": ("external_observation_count_below_threshold"),
        "v63_normalized_rmse_met": ("v63_normalized_rmse_threshold_failed"),
        "persistence_mse_skill_met": ("persistence_baseline_skill_threshold_failed"),
        "empirical_interval_coverage_met": (
            "empirical_interval_coverage_threshold_failed"
        ),
        "absolute_coverage_error_met": (
            "absolute_interval_coverage_error_threshold_failed"
        ),
        "normalized_interval_score_met": ("normalized_interval_score_threshold_failed"),
        "persistence_interval_score_skill_met": (
            "persistence_interval_score_skill_threshold_failed"
        ),
        "non_fixture_evidence": "fixture_or_control_evidence_rejected",
    }
    reasons = sorted(
        reason_by_check[name] for name, passed in checks.items() if not passed
    )
    passed = all(checks.values())
    return PredictiveQualityAssessmentV65.seal(
        quality_overlay_id=contract.quality_overlay_id,
        qualification_id=contract.qualification_id,
        task_id=contract.task_id,
        quality_contract_hash=contract.contract_hash,
        public_prediction_pack_hash=prediction_pack.pack_hash,
        external_evaluation_hash=evaluation.evaluation_hash,
        v63_contract_hash=contract.v63_contract_hash,
        v63_prediction_vector_hash=prediction_vector.vector_hash,
        observation_count=evaluation.observation_count,
        normalized_rmse=normalized_rmse,
        mse_skill_over_persistence=mse_skill,
        empirical_interval_coverage=empirical_coverage,
        baseline_empirical_interval_coverage=(baseline_empirical_coverage),
        absolute_coverage_error=absolute_coverage_error,
        model_interval_score=model_interval_score,
        baseline_interval_score=baseline_interval_score,
        normalized_interval_score=normalized_interval_score,
        interval_score_skill_over_persistence=interval_score_skill,
        checks=checks,
        status="PASS" if passed else "REJECT",
        reason_codes=reasons,
        quality_overlay_passed=passed,
        fixture_only=prediction_pack.fixture_only,
    )


__all__ = [
    "ExternalAggregateQualityEvaluationV65",
    "IntervalImplementationManifestV65",
    "PredictiveQualityAssessmentV65",
    "PredictiveQualityContractV65",
    "PredictiveQualityError",
    "PublicPredictiveQualityPackV65",
    "assess_predictive_quality_v65",
    "freeze_predictive_quality_contract_v65",
    "freeze_public_predictive_quality_pack_v65",
    "seal_external_aggregate_quality_evaluation_v65",
]
