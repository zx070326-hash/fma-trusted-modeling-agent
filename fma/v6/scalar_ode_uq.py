"""Training-only empirical intervals for the registered scalar ODE path.

This V6.5 adapter is intentionally narrow.  It supports regularly sampled,
positive scalar series whose current V6.2 executable receipt selected the
registered scalar ODE adapter.  Calibration uses expanding-origin forecasts
from public training observations only.  The output is a rolling-origin
diagnostic, not a conformal finite-sample coverage guarantee.  It never
accepts external target values.
"""

from __future__ import annotations

import hashlib
import math
import marshal
import platform
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Annotated, Literal, cast

import numpy as np
from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5_2.ode_system import (
    ODEFamilyV52,
    ODETimeSeriesSnapshotV52,
    _parameter_vector,
    _predict,
    fit_ode_v52,
)

from .executable_candidate import (
    SCALAR_ODE_ADAPTER_ID,
    ExecutableCandidateReceiptV62,
)
from .external_prediction_runtime import _scalar_ode_predictions
from .external_qualification import (
    ExternalAggregateEvaluationV63,
    ExternalForecastInputV63,
    ExternalPredictionVectorV63,
    PredictiveExternalQualificationContractV63,
)
from .predictive_quality import (
    ExternalAggregateQualityEvaluationV65,
    IntervalImplementationManifestV65,
    PredictiveQualityAssessmentV65,
    PredictiveQualityContractV65,
    PublicPredictiveQualityPackV65,
    assess_predictive_quality_v65,
    freeze_public_predictive_quality_pack_v65,
)


SCALAR_ODE_INTERVAL_ADAPTER_ID_V65 = (
    "scalar_ode_expanding_origin_log_empirical_v65"
)
MINIMUM_TRAINING_OBSERVATIONS_V65 = 12
MINIMUM_CALIBRATION_ORIGINS_V65 = 8
SCALAR_ODE_INTERVAL_PROTOCOL_HASH_V65 = sha256_value(
    {
        "schema_version": "6.5-scalar-ode-empirical-interval-protocol",
        "adapter_id": SCALAR_ODE_INTERVAL_ADAPTER_ID_V65,
        "input_domain": "positive_regular_scalar_series",
        "selected_family_policy": (
            "fixed_current_v62_selected_family_diagnostic_only"
        ),
        "model_selection_replayed_per_origin": False,
        "fit_policy": "expanding_origin_refit",
        "error_scale": "absolute_log_forecast_error",
        "quantile": "ceil((m+1)*(1-alpha))_order_statistic",
        "minimum_training_observations": (MINIMUM_TRAINING_OBSERVATIONS_V65),
        "minimum_calibration_origins": (MINIMUM_CALIBRATION_ORIGINS_V65),
        "baseline": "last_origin_observation_persistence",
        "private_targets_permitted": False,
        "finite_sample_coverage_guaranteed": False,
        "temporal_dependence_coverage_guaranteed": False,
        "post_selection_coverage_guaranteed": False,
        "claim_ceiling": "diagnostic_rolling_origin_empirical_interval_only",
        "runtime_binding": (
            "implementation_source_python_numpy_scipy_identity"
        ),
    }
)

FiniteNonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
FinitePositive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonEmptyText = Annotated[str, Field(min_length=1)]


class ScalarODEIntervalError(ValueError):
    """The narrow V6.5 interval adapter failed closed."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _assert_aware(value: datetime, field_name: str) -> None:
    if value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _finite_sample_quantile(errors: list[float], alpha: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise ScalarODEIntervalError("interval alpha must be strictly between zero and one")
    minimum_count = max(
        MINIMUM_CALIBRATION_ORIGINS_V65,
        math.ceil(1.0 / alpha) - 1,
    )
    if len(errors) < minimum_count:
        raise ScalarODEIntervalError(
            "too few public calibration origins for the requested order statistic"
        )
    if any(not math.isfinite(value) or value < 0 for value in errors):
        raise ScalarODEIntervalError(
            "public calibration errors must be finite and nonnegative"
        )
    ordered = sorted(errors)
    rank = math.ceil((len(ordered) + 1) * (1.0 - alpha))
    if rank > len(ordered):
        raise ScalarODEIntervalError(
            "requested empirical order statistic is unavailable"
        )
    return float(ordered[rank - 1])


def _implementation_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _runtime_identity() -> dict[str, str]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": package_version("scipy"),
    }


def _callable_semantic_sha256(callable_object: object) -> str:
    code = getattr(callable_object, "__code__", None)
    if code is None:
        raise ScalarODEIntervalError(
            "interval implementation contains an unhashable callable"
        )
    return sha256_value(
        {
            "module": getattr(callable_object, "__module__", None),
            "qualname": getattr(callable_object, "__qualname__", None),
            "marshalled_code_sha256": hashlib.sha256(
                marshal.dumps(code)
            ).hexdigest(),
            "defaults": repr(getattr(callable_object, "__defaults__", None)),
            "keyword_defaults": repr(
                getattr(callable_object, "__kwdefaults__", None)
            ),
        }
    )


def scalar_ode_interval_implementation_manifest_v65(
) -> IntervalImplementationManifestV65:
    """Return the currently loaded implementation identity for pre-freezing."""

    runtime_identity = _runtime_identity()
    return IntervalImplementationManifestV65.seal(
        interval_adapter_id=SCALAR_ODE_INTERVAL_ADAPTER_ID_V65,
        interval_adapter_protocol_hash=SCALAR_ODE_INTERVAL_PROTOCOL_HASH_V65,
        module_name="fma.v6.scalar_ode_uq",
        module_source_sha256=_implementation_source_sha256(),
        loaded_callable_code_hashes={
            "fit_ode_v52": _callable_semantic_sha256(fit_ode_v52),
            "parameter_vector": _callable_semantic_sha256(_parameter_vector),
            "predict_ode": _callable_semantic_sha256(_predict),
            "prefix_forecast": _callable_semantic_sha256(_prefix_forecast),
            "scalar_ode_predictions": _callable_semantic_sha256(
                _scalar_ode_predictions
            ),
        },
        python_implementation=runtime_identity["python_implementation"],
        python_version=runtime_identity["python_version"],
        numpy_version=runtime_identity["numpy_version"],
        scipy_version=runtime_identity["scipy_version"],
        optimizer_policy=(
            "fit_ode_v52_require_optimizer_converged_zero_skipped_failures"
        ),
        model_selection_policy=(
            "fixed_current_v62_selected_family_diagnostic_only"
        ),
    )


class ScalarODEIntervalCalibrationV65(StrictModel):
    """Content-addressed public-data calibration and interval receipt."""

    schema_version: Literal["6.5-scalar-ode-interval-calibration"] = (
        "6.5-scalar-ode-interval-calibration"
    )
    quality_overlay_id: Identifier
    qualification_id: Identifier
    task_id: Identifier
    quality_contract_hash: Sha256
    v63_contract_hash: Sha256
    processed_snapshot_hash: Sha256
    executable_candidate_receipt_hash: Sha256
    selected_model_id: Identifier
    selected_model_identity_hash: Sha256
    forecast_input_hash: Sha256
    prediction_vector_hash: Sha256
    target_order_hash: Sha256
    interval_adapter_id: Literal[
        "scalar_ode_expanding_origin_log_empirical_v65"
    ] = (
        SCALAR_ODE_INTERVAL_ADAPTER_ID_V65
    )
    interval_adapter_protocol_hash: Sha256 = SCALAR_ODE_INTERVAL_PROTOCOL_HASH_V65
    interval_alpha: Annotated[
        float,
        Field(gt=0, lt=1, allow_inf_nan=False),
    ]
    horizon_steps: Annotated[list[int], Field(min_length=1)]
    calibration_counts: Annotated[list[int], Field(min_length=1)]
    prefix_fit_attempt_counts: Annotated[list[int], Field(min_length=1)]
    prefix_fit_success_counts: Annotated[list[int], Field(min_length=1)]
    prefix_fit_failure_counts: Annotated[list[int], Field(min_length=1)]
    model_absolute_log_error_quantiles: Annotated[
        list[FiniteNonNegative],
        Field(min_length=1),
    ]
    baseline_absolute_log_error_quantiles: Annotated[
        list[FiniteNonNegative],
        Field(min_length=1),
    ]
    model_error_hashes: Annotated[list[Sha256], Field(min_length=1)]
    baseline_error_hashes: Annotated[list[Sha256], Field(min_length=1)]
    persistence_baseline_point: FinitePositive
    model_lower_bounds: Annotated[
        list[FinitePositive],
        Field(min_length=1),
    ]
    model_upper_bounds: Annotated[
        list[FinitePositive],
        Field(min_length=1),
    ]
    baseline_lower_bounds: Annotated[
        list[FinitePositive],
        Field(min_length=1),
    ]
    baseline_upper_bounds: Annotated[
        list[FinitePositive],
        Field(min_length=1),
    ]
    calibration_origin_set_hash: Sha256
    interval_implementation_manifest_hash: Sha256
    implementation_source_sha256: Sha256
    python_implementation: NonEmptyText
    python_version: NonEmptyText
    numpy_version: NonEmptyText
    scipy_version: NonEmptyText
    runtime_identity_hash: Sha256
    public_training_data_only: Literal[True] = True
    private_external_targets_accessed: Literal[False] = False
    interval_evidence_kind: Literal[
        "rolling_origin_empirical_diagnostic"
    ] = "rolling_origin_empirical_diagnostic"
    selected_family_policy: Literal[
        "fixed_current_v62_selected_family_diagnostic_only"
    ] = "fixed_current_v62_selected_family_diagnostic_only"
    model_selection_replayed_per_origin: Literal[False] = False
    finite_sample_coverage_guaranteed: Literal[False] = False
    temporal_dependence_coverage_guaranteed: Literal[False] = False
    post_selection_coverage_guaranteed: Literal[False] = False
    interval_claim_ceiling: Literal[
        "diagnostic_rolling_origin_empirical_interval_only"
    ] = "diagnostic_rolling_origin_empirical_interval_only"
    fixture_only: bool
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    generated_at: datetime
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "ScalarODEIntervalCalibrationV65":
        _assert_aware(self.generated_at, "generated_at")
        count = len(self.horizon_steps)
        arrays = (
            self.calibration_counts,
            self.prefix_fit_attempt_counts,
            self.prefix_fit_success_counts,
            self.prefix_fit_failure_counts,
            self.model_absolute_log_error_quantiles,
            self.baseline_absolute_log_error_quantiles,
            self.model_error_hashes,
            self.baseline_error_hashes,
            self.model_lower_bounds,
            self.model_upper_bounds,
            self.baseline_lower_bounds,
            self.baseline_upper_bounds,
        )
        if any(len(values) != count for values in arrays):
            raise ValueError("V6.5 interval calibration lengths differ")
        if self.horizon_steps != sorted(set(self.horizon_steps)) or any(
            step < 1 for step in self.horizon_steps
        ):
            raise ValueError("V6.5 interval horizons must be positive and increasing")
        minimum_count = max(
            MINIMUM_CALIBRATION_ORIGINS_V65,
            math.ceil(1.0 / self.interval_alpha) - 1,
        )
        if any(value < minimum_count for value in self.calibration_counts):
            raise ValueError("V6.5 interval calibration count is too small")
        if any(
            attempts != successes + failures
            or failures != 0
            or successes != calibration_count
            for attempts, successes, failures, calibration_count in zip(
                self.prefix_fit_attempt_counts,
                self.prefix_fit_success_counts,
                self.prefix_fit_failure_counts,
                self.calibration_counts,
            )
        ):
            raise ValueError(
                "V6.5 prefix fit accounting permits skipped calibration failures"
            )
        for lower, upper in zip(
            self.model_lower_bounds,
            self.model_upper_bounds,
        ):
            if lower >= upper:
                raise ValueError("V6.5 model interval is not ordered")
        for lower, upper in zip(
            self.baseline_lower_bounds,
            self.baseline_upper_bounds,
        ):
            if lower >= upper:
                raise ValueError("V6.5 baseline interval is not ordered")
        if self.interval_adapter_protocol_hash != SCALAR_ODE_INTERVAL_PROTOCOL_HASH_V65:
            raise ValueError("V6.5 scalar ODE interval protocol differs")
        runtime_identity = {
            "python_implementation": self.python_implementation,
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "scipy_version": self.scipy_version,
        }
        if self.runtime_identity_hash != sha256_value(runtime_identity):
            raise ValueError("V6.5 interval runtime identity hash differs")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("V6.5 scalar ODE interval receipt hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("V6.5 scalar ODE interval receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ScalarODEIntervalCalibrationV65":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"receipt_hash"})
        payload["receipt_hash"] = draft.content_hash()
        return cls(**payload)


def _regular_cadence(
    snapshot: ODETimeSeriesSnapshotV52,
) -> float:
    differences = np.diff(np.asarray(snapshot.times, dtype=float))
    cadence = float(np.median(differences))
    if (
        not math.isfinite(cadence)
        or cadence <= 0
        or not np.allclose(
            differences,
            cadence,
            rtol=1e-9,
            atol=1e-12,
        )
    ):
        raise ScalarODEIntervalError(
            "scalar ODE empirical interval adapter requires regular sampling"
        )
    return cadence


def _horizon_steps(
    *,
    snapshot: ODETimeSeriesSnapshotV52,
    forecast_input: ExternalForecastInputV63,
) -> list[int]:
    cadence = _regular_cadence(snapshot)
    final_time = float(snapshot.times[-1])
    result: list[int] = []
    for target_time in forecast_input.forecast_times:
        raw = (float(target_time) - final_time) / cadence
        step = int(round(raw))
        if step < 1 or not math.isclose(
            raw,
            step,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ScalarODEIntervalError(
                "forecast coordinate is off the public training cadence"
            )
        result.append(step)
    if result != sorted(set(result)):
        raise ScalarODEIntervalError(
            "forecast horizons must be positive and increasing"
        )
    return result


def _prefix_forecast(
    *,
    family: ODEFamilyV52,
    times: np.ndarray,
    values: np.ndarray,
    target_time: float,
) -> float:
    fit = fit_ode_v52(family, times, values)
    if not fit.optimizer_converged:
        raise ScalarODEIntervalError("public calibration refit did not converge")
    prediction = float(
        _predict(
            family,
            np.asarray([times[0], target_time], dtype=float),
            float(values[0]),
            _parameter_vector(fit),
        )[-1]
    )
    if not math.isfinite(prediction) or prediction <= 0:
        raise ScalarODEIntervalError(
            "public calibration prediction is not positive finite"
        )
    return prediction


def _assert_bindings(
    *,
    quality_contract: PredictiveQualityContractV65,
    v63_contract: PredictiveExternalQualificationContractV63,
    snapshot: ODETimeSeriesSnapshotV52,
    executable_receipt: ExecutableCandidateReceiptV62,
    forecast_input: ExternalForecastInputV63,
    prediction_vector: ExternalPredictionVectorV63,
    allow_fixture: bool,
) -> tuple[bool, IntervalImplementationManifestV65]:
    for item in (
        quality_contract,
        v63_contract,
        snapshot,
        executable_receipt,
        forecast_input,
        prediction_vector,
    ):
        try:
            item.assert_sealed()
        except ValueError as exc:
            raise ScalarODEIntervalError("V6.5 interval input is not sealed") from exc
    if (
        quality_contract.interval_adapter_id != SCALAR_ODE_INTERVAL_ADAPTER_ID_V65
        or quality_contract.interval_adapter_protocol_hash
        != SCALAR_ODE_INTERVAL_PROTOCOL_HASH_V65
    ):
        raise ScalarODEIntervalError(
            "quality contract selected another interval adapter"
        )
    current_implementation = (
        scalar_ode_interval_implementation_manifest_v65()
    )
    if (
        quality_contract.interval_implementation_manifest
        != current_implementation
        or quality_contract.interval_implementation_manifest_hash
        != current_implementation.manifest_hash
    ):
        raise ScalarODEIntervalError(
            "loaded interval implementation differs from the frozen manifest"
        )
    if (
        quality_contract.qualification_id != v63_contract.qualification_id
        or quality_contract.v63_contract_hash != v63_contract.contract_hash
        or quality_contract.task_id != v63_contract.task_id
        or snapshot.task_id != v63_contract.task_id
        or snapshot.snapshot_hash != v63_contract.processed_snapshot_hash
        or executable_receipt.receipt_hash
        != v63_contract.executable_candidate_receipt_hash
        or executable_receipt.adapter_id != SCALAR_ODE_ADAPTER_ID
        or executable_receipt.selected_model_id != v63_contract.selected_model_id
        or executable_receipt.bundle_hash != v63_contract.scientific_bundle_hash
        or not executable_receipt.bundle_scientific_acceptance
        or snapshot.fixture_only != executable_receipt.fixture_only
        or forecast_input.qualification_id != v63_contract.qualification_id
        or forecast_input.task_id != v63_contract.task_id
        or forecast_input.contract_hash != v63_contract.contract_hash
        or forecast_input.processed_snapshot_hash != snapshot.snapshot_hash
        or prediction_vector.qualification_id != v63_contract.qualification_id
        or prediction_vector.external_snapshot_hash != forecast_input.input_hash
        or prediction_vector.target_ids != forecast_input.target_ids
        or prediction_vector.target_order_hash != forecast_input.target_order_hash
        or prediction_vector.selected_model_identity_hash
        != v63_contract.selected_model_identity_hash
        or quality_contract.frozen_at > forecast_input.frozen_at
    ):
        raise ScalarODEIntervalError(
            "V6.5 interval inputs do not bind the current V6.3 model"
        )
    if snapshot.fixture_only and not allow_fixture:
        raise ScalarODEIntervalError(
            "fixture calibration requires explicit diagnostic opt-in"
        )
    expected = _scalar_ode_predictions(
        snapshot=snapshot,
        receipt=executable_receipt,
        forecast_input=forecast_input,
    )
    if len(expected) != len(prediction_vector.predictions) or any(
        not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
        for left, right in zip(expected, prediction_vector.predictions)
    ):
        raise ScalarODEIntervalError(
            "V6.3 point vector differs from deterministic scalar ODE replay"
        )
    return snapshot.fixture_only, current_implementation


def _multiplicative_interval(
    *,
    point: float,
    absolute_log_error: float,
) -> tuple[float, float]:
    try:
        lower = point * math.exp(-absolute_log_error)
        upper = point * math.exp(absolute_log_error)
    except OverflowError as exc:
        raise ScalarODEIntervalError(
            "empirical interval bound overflowed"
        ) from exc
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower <= 0
        or upper <= lower
    ):
        raise ScalarODEIntervalError(
            "empirical interval bounds are not positive finite and ordered"
        )
    return lower, upper


def calibrate_scalar_ode_intervals_v65(
    *,
    quality_contract: PredictiveQualityContractV65,
    v63_contract: PredictiveExternalQualificationContractV63,
    snapshot: ODETimeSeriesSnapshotV52,
    executable_receipt: ExecutableCandidateReceiptV62,
    forecast_input: ExternalForecastInputV63,
    prediction_vector: ExternalPredictionVectorV63,
    allow_fixture: bool = False,
    generated_at: datetime | None = None,
) -> tuple[
    ScalarODEIntervalCalibrationV65,
    PublicPredictiveQualityPackV65,
]:
    """Calibrate model and persistence intervals from public observations."""

    fixture_only, implementation_manifest = _assert_bindings(
        quality_contract=quality_contract,
        v63_contract=v63_contract,
        snapshot=snapshot,
        executable_receipt=executable_receipt,
        forecast_input=forecast_input,
        prediction_vector=prediction_vector,
        allow_fixture=allow_fixture,
    )
    effective_generated_at = generated_at or _utc_now()
    _assert_aware(effective_generated_at, "generated_at")
    if effective_generated_at < forecast_input.frozen_at:
        raise ScalarODEIntervalError(
            "interval calibration predates the frozen forecast input"
        )
    horizons = _horizon_steps(
        snapshot=snapshot,
        forecast_input=forecast_input,
    )
    times = np.asarray(snapshot.times, dtype=float)
    values = np.asarray(snapshot.observations, dtype=float)
    family = cast(ODEFamilyV52, executable_receipt.selected_family)

    calibration_counts: list[int] = []
    model_quantiles: list[float] = []
    baseline_quantiles: list[float] = []
    model_error_hashes: list[str] = []
    baseline_error_hashes: list[str] = []
    origin_sets: list[list[int]] = []
    prefix_fit_attempt_counts: list[int] = []
    prefix_fit_success_counts: list[int] = []
    prefix_fit_failure_counts: list[int] = []
    for horizon in horizons:
        model_errors: list[float] = []
        baseline_errors: list[float] = []
        origins: list[int] = []
        first_origin = MINIMUM_TRAINING_OBSERVATIONS_V65 - 1
        last_origin = len(values) - horizon - 1
        attempted = 0
        for origin in range(first_origin, last_origin + 1):
            attempted += 1
            actual = float(values[origin + horizon])
            baseline = float(values[origin])
            try:
                predicted = _prefix_forecast(
                    family=family,
                    times=times[: origin + 1],
                    values=values[: origin + 1],
                    target_time=float(times[origin + horizon]),
                )
            except (
                ArithmeticError,
                RuntimeError,
                ScalarODEIntervalError,
                ValueError,
            ) as exc:
                raise ScalarODEIntervalError(
                    "public calibration refit failed closed at "
                    f"horizon={horizon}, origin={origin}"
                ) from exc
            model_errors.append(abs(math.log(actual) - math.log(predicted)))
            baseline_errors.append(abs(math.log(actual) - math.log(baseline)))
            origins.append(origin)
        model_quantile = _finite_sample_quantile(
            model_errors,
            quality_contract.interval_alpha,
        )
        baseline_quantile = _finite_sample_quantile(
            baseline_errors,
            quality_contract.interval_alpha,
        )
        calibration_counts.append(len(origins))
        prefix_fit_attempt_counts.append(attempted)
        prefix_fit_success_counts.append(len(origins))
        prefix_fit_failure_counts.append(0)
        model_quantiles.append(model_quantile)
        baseline_quantiles.append(baseline_quantile)
        model_error_hashes.append(sha256_value(model_errors))
        baseline_error_hashes.append(sha256_value(baseline_errors))
        origin_sets.append(origins)

    model_intervals = [
        _multiplicative_interval(
            point=point,
            absolute_log_error=quantile,
        )
        for point, quantile in zip(
            prediction_vector.predictions,
            model_quantiles,
        )
    ]
    model_lower = [lower for lower, _ in model_intervals]
    model_upper = [upper for _, upper in model_intervals]
    persistence = float(values[-1])
    baseline_intervals = [
        _multiplicative_interval(
            point=persistence,
            absolute_log_error=quantile,
        )
        for quantile in baseline_quantiles
    ]
    baseline_lower = [lower for lower, _ in baseline_intervals]
    baseline_upper = [upper for _, upper in baseline_intervals]
    receipt = ScalarODEIntervalCalibrationV65.seal(
        quality_overlay_id=quality_contract.quality_overlay_id,
        qualification_id=v63_contract.qualification_id,
        task_id=v63_contract.task_id,
        quality_contract_hash=quality_contract.contract_hash,
        v63_contract_hash=v63_contract.contract_hash,
        processed_snapshot_hash=snapshot.snapshot_hash,
        executable_candidate_receipt_hash=executable_receipt.receipt_hash,
        selected_model_id=executable_receipt.selected_model_id,
        selected_model_identity_hash=(v63_contract.selected_model_identity_hash),
        forecast_input_hash=forecast_input.input_hash,
        prediction_vector_hash=prediction_vector.vector_hash,
        target_order_hash=forecast_input.target_order_hash,
        interval_alpha=quality_contract.interval_alpha,
        horizon_steps=horizons,
        calibration_counts=calibration_counts,
        prefix_fit_attempt_counts=prefix_fit_attempt_counts,
        prefix_fit_success_counts=prefix_fit_success_counts,
        prefix_fit_failure_counts=prefix_fit_failure_counts,
        model_absolute_log_error_quantiles=model_quantiles,
        baseline_absolute_log_error_quantiles=baseline_quantiles,
        model_error_hashes=model_error_hashes,
        baseline_error_hashes=baseline_error_hashes,
        persistence_baseline_point=persistence,
        model_lower_bounds=model_lower,
        model_upper_bounds=model_upper,
        baseline_lower_bounds=baseline_lower,
        baseline_upper_bounds=baseline_upper,
        calibration_origin_set_hash=sha256_value(origin_sets),
        interval_implementation_manifest_hash=(
            implementation_manifest.manifest_hash
        ),
        implementation_source_sha256=(
            implementation_manifest.module_source_sha256
        ),
        python_implementation=(
            implementation_manifest.python_implementation
        ),
        python_version=implementation_manifest.python_version,
        numpy_version=implementation_manifest.numpy_version,
        scipy_version=implementation_manifest.scipy_version,
        runtime_identity_hash=sha256_value(
            {
                "python_implementation": (
                    implementation_manifest.python_implementation
                ),
                "python_version": implementation_manifest.python_version,
                "numpy_version": implementation_manifest.numpy_version,
                "scipy_version": implementation_manifest.scipy_version,
            }
        ),
        fixture_only=fixture_only,
        generated_at=effective_generated_at,
    )
    pack = freeze_public_predictive_quality_pack_v65(
        contract=quality_contract,
        v63_contract=v63_contract,
        prediction_vector=prediction_vector,
        persistence_baseline_point=persistence,
        lower_bounds=model_lower,
        upper_bounds=model_upper,
        baseline_lower_bounds=baseline_lower,
        baseline_upper_bounds=baseline_upper,
        interval_calibration_receipt_hash=receipt.receipt_hash,
        fixture_only=fixture_only,
        packed_at=effective_generated_at,
    )
    return receipt, pack


def verify_scalar_ode_intervals_v65(
    *,
    receipt: ScalarODEIntervalCalibrationV65,
    quality_contract: PredictiveQualityContractV65,
    v63_contract: PredictiveExternalQualificationContractV63,
    snapshot: ODETimeSeriesSnapshotV52,
    executable_receipt: ExecutableCandidateReceiptV62,
    forecast_input: ExternalForecastInputV63,
    prediction_vector: ExternalPredictionVectorV63,
    prediction_pack: PublicPredictiveQualityPackV65,
) -> bool:
    """Deterministically replay the public calibration and exact pack."""

    try:
        receipt.assert_sealed()
        prediction_pack.assert_sealed()
        replayed_receipt, replayed_pack = calibrate_scalar_ode_intervals_v65(
            quality_contract=quality_contract,
            v63_contract=v63_contract,
            snapshot=snapshot,
            executable_receipt=executable_receipt,
            forecast_input=forecast_input,
            prediction_vector=prediction_vector,
            allow_fixture=receipt.fixture_only,
            generated_at=receipt.generated_at,
        )
    except (
        ArithmeticError,
        ScalarODEIntervalError,
        TypeError,
        ValueError,
    ):
        return False
    return replayed_receipt == receipt and replayed_pack == prediction_pack


def assess_scalar_ode_predictive_quality_v65(
    *,
    calibration_receipt: ScalarODEIntervalCalibrationV65,
    quality_contract: PredictiveQualityContractV65,
    v63_contract: PredictiveExternalQualificationContractV63,
    v63_evaluation: ExternalAggregateEvaluationV63,
    snapshot: ODETimeSeriesSnapshotV52,
    executable_receipt: ExecutableCandidateReceiptV62,
    forecast_input: ExternalForecastInputV63,
    prediction_vector: ExternalPredictionVectorV63,
    prediction_pack: PublicPredictiveQualityPackV65,
    quality_evaluation: ExternalAggregateQualityEvaluationV65,
    trusted_public_keys: Mapping[str, bytes],
) -> PredictiveQualityAssessmentV65:
    """Require semantic interval replay before assessing aggregate quality."""

    if not verify_scalar_ode_intervals_v65(
        receipt=calibration_receipt,
        quality_contract=quality_contract,
        v63_contract=v63_contract,
        snapshot=snapshot,
        executable_receipt=executable_receipt,
        forecast_input=forecast_input,
        prediction_vector=prediction_vector,
        prediction_pack=prediction_pack,
    ):
        raise ScalarODEIntervalError("scalar ODE interval calibration replay failed")
    return assess_predictive_quality_v65(
        contract=quality_contract,
        v63_contract=v63_contract,
        v63_evaluation=v63_evaluation,
        prediction_vector=prediction_vector,
        prediction_pack=prediction_pack,
        evaluation=quality_evaluation,
        trusted_public_keys=trusted_public_keys,
    )


__all__ = [
    "MINIMUM_CALIBRATION_ORIGINS_V65",
    "MINIMUM_TRAINING_OBSERVATIONS_V65",
    "SCALAR_ODE_INTERVAL_ADAPTER_ID_V65",
    "SCALAR_ODE_INTERVAL_PROTOCOL_HASH_V65",
    "ScalarODEIntervalCalibrationV65",
    "ScalarODEIntervalError",
    "assess_scalar_ode_predictive_quality_v65",
    "calibrate_scalar_ode_intervals_v65",
    "scalar_ode_interval_implementation_manifest_v65",
    "verify_scalar_ode_intervals_v65",
]
