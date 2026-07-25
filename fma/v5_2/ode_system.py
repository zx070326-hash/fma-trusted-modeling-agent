"""Production-scoped L0--L4 adapter for scalar autonomous ODE systems.

This is structurally different from the V5.1 event-process adapter: it fits
continuous state trajectories governed by differential equations rather than
event intensities or renewal distributions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import scipy
from pydantic import Field, model_validator
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5.check_registry import AdapterContextV50, AdapterOutcomeV50
from fma.v5.workspace_schemas import CodeManifestV50


ODEFamilyV52 = Literal["constant", "exponential", "gompertz", "logistic"]
LevelV52 = Literal["L0", "L1", "L2", "L3", "L4"]
LevelStatusV52 = Literal["PASS", "FAIL", "NOT_RUN"]
FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


class ODETimeSeriesSnapshotV52(StrictModel):
    schema_version: Literal["5.2"] = "5.2"
    task_id: Identifier
    time_unit: Identifier
    state_unit: Identifier
    times: Annotated[list[FiniteNumber], Field(min_length=12)]
    observations: Annotated[list[FiniteNumber], Field(min_length=12)]
    source_id: Annotated[str, Field(min_length=3)]
    fixture_only: bool
    snapshot_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_snapshot(self) -> "ODETimeSeriesSnapshotV52":
        if len(self.times) != len(self.observations):
            raise ValueError("times and observations must have equal length")
        if any(
            right <= left for left, right in zip(self.times, self.times[1:])
        ):
            raise ValueError("times must be strictly increasing")
        if any(value <= 0 for value in self.observations):
            raise ValueError("scalar ODE observations must be positive")
        if self.snapshot_hash and self.snapshot_hash != self.content_hash():
            raise ValueError("snapshot_hash does not match snapshot")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "snapshot_hash")

    def assert_sealed(self) -> None:
        if not self.snapshot_hash or self.snapshot_hash != self.content_hash():
            raise ValueError("ODE snapshot is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ODETimeSeriesSnapshotV52":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"snapshot_hash"})
        payload["snapshot_hash"] = draft.content_hash()
        return cls(**payload)


class ODEThresholdsV52(StrictModel):
    schema_version: Literal["5.2"] = "5.2"
    split_fraction: Annotated[float, Field(gt=0.5, lt=0.9)] = 0.7
    minimum_points_per_slice: Annotated[int, Field(ge=4, le=1000)] = 6
    maximum_validation_relative_rmse: Annotated[
        float, Field(gt=0, le=2, allow_inf_nan=False)
    ] = 0.15
    minimum_baseline_relative_improvement: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ] = 0.10
    maximum_absolute_residual_lag1_correlation: Annotated[
        float, Field(gt=0, le=1, allow_inf_nan=False)
    ] = 0.85
    minimum_validation_interval_coverage: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ] = 0.50
    maximum_parameter_condition_number: Annotated[
        float, Field(gt=1, allow_inf_nan=False)
    ] = 1e10
    bootstrap_replicates: Annotated[int, Field(ge=20, le=5000)] = 40
    bootstrap_seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 104729
    minimum_bootstrap_success_fraction: Annotated[
        float, Field(gt=0, le=1, allow_inf_nan=False)
    ] = 0.80
    maximum_forecast_interval_relative_width: Annotated[
        float, Field(gt=0, allow_inf_nan=False)
    ] = 2.0
    maximum_window_sensitivity_relative_range: Annotated[
        float, Field(gt=0, allow_inf_nan=False)
    ] = 1.0
    maximum_ensemble_forecast_coefficient_of_variation: Annotated[
        float, Field(gt=0, allow_inf_nan=False)
    ] = 2.0
    threshold_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_thresholds(self) -> "ODEThresholdsV52":
        if self.threshold_hash and self.threshold_hash != self.content_hash():
            raise ValueError("threshold_hash does not match thresholds")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "threshold_hash")

    def assert_sealed(self) -> None:
        if not self.threshold_hash or self.threshold_hash != self.content_hash():
            raise ValueError("ODE thresholds are not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ODEThresholdsV52":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"threshold_hash"})
        payload["threshold_hash"] = draft.content_hash()
        return cls(**payload)


class ODEFitV52(StrictModel):
    schema_version: Literal["5.2"] = "5.2"
    family: ODEFamilyV52
    parameter_names: list[Identifier]
    parameter_values: list[FiniteNumber]
    training_rmse: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    parameter_condition_number: Annotated[
        float, Field(ge=1, allow_inf_nan=False)
    ]
    optimizer_converged: bool
    optimizer_evaluations: Annotated[int, Field(ge=0)]
    fit_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_fit(self) -> "ODEFitV52":
        if len(self.parameter_names) != len(self.parameter_values):
            raise ValueError("parameter names and values differ in length")
        if self.parameter_names != sorted(set(self.parameter_names)):
            raise ValueError("parameter names must be sorted and unique")
        if self.fit_hash and self.fit_hash != self.content_hash():
            raise ValueError("fit_hash does not match fit")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "fit_hash")

    @classmethod
    def seal(cls, **data: object) -> "ODEFitV52":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"fit_hash"})
        payload["fit_hash"] = draft.content_hash()
        return cls(**payload)


class ODECandidateEvidenceV52(StrictModel):
    schema_version: Literal["5.2"] = "5.2"
    candidate_id: ODEFamilyV52
    equation: Annotated[str, Field(min_length=4)]
    fit: ODEFitV52
    validation_rmse: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    validation_relative_rmse: Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ]
    validation_score: FiniteNumber
    forecast_value: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> "ODECandidateEvidenceV52":
        if self.candidate_id != self.fit.family:
            raise ValueError("candidate and fit family differ")
        if self.evidence_hash and self.evidence_hash != self.content_hash():
            raise ValueError("evidence_hash does not match candidate evidence")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evidence_hash")

    @classmethod
    def seal(cls, **data: object) -> "ODECandidateEvidenceV52":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"evidence_hash"})
        payload["evidence_hash"] = draft.content_hash()
        return cls(**payload)


class ODELevelEvidenceV52(StrictModel):
    schema_version: Literal["5.2"] = "5.2"
    level: LevelV52
    status: LevelStatusV52
    checks: dict[Identifier, bool]
    metrics: dict[Identifier, FiniteNumber | int | None]
    thresholds: dict[Identifier, FiniteNumber | int]
    evidence: dict[str, Any]
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_level(self) -> "ODELevelEvidenceV52":
        if self.status == "PASS" and not self.checks:
            raise ValueError("passing level needs checks")
        if self.status == "PASS" and not all(self.checks.values()):
            raise ValueError("passing level contains a failed check")
        if self.evidence_hash and self.evidence_hash != self.content_hash():
            raise ValueError("evidence_hash does not match level evidence")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evidence_hash")

    @classmethod
    def seal(cls, **data: object) -> "ODELevelEvidenceV52":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"evidence_hash"})
        payload["evidence_hash"] = draft.content_hash()
        return cls(**payload)


class ODEScientificBundleV52(StrictModel):
    schema_version: Literal["5.2"] = "5.2"
    task_id: Identifier
    domain: Literal["scalar_autonomous_ode"] = "scalar_autonomous_ode"
    snapshot_hash: Sha256
    threshold_hash: Sha256
    candidate_registry_hash: Sha256
    selected_candidate_id: ODEFamilyV52
    selected_fit_hash: Sha256
    candidates: list[ODECandidateEvidenceV52]
    levels: list[ODELevelEvidenceV52]
    scientific_acceptance: bool
    fixture_only: bool
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    bundle_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bundle(self) -> "ODEScientificBundleV52":
        candidate_ids = [item.candidate_id for item in self.candidates]
        if candidate_ids != sorted(set(candidate_ids)):
            raise ValueError("candidate evidence must be sorted and unique")
        if self.selected_candidate_id not in candidate_ids:
            raise ValueError("selected candidate is absent")
        levels = [item.level for item in self.levels]
        if levels != ["L0", "L1", "L2", "L3", "L4"]:
            raise ValueError("bundle must contain ordered L0-L4 evidence")
        expected = all(item.status == "PASS" for item in self.levels)
        if self.scientific_acceptance != expected:
            raise ValueError("scientific acceptance differs from L0-L4")
        if self.bundle_hash and self.bundle_hash != self.content_hash():
            raise ValueError("bundle_hash does not match bundle")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "bundle_hash")

    @classmethod
    def seal(cls, **data: object) -> "ODEScientificBundleV52":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"bundle_hash"})
        payload["bundle_hash"] = draft.content_hash()
        return cls(**payload)


_EQUATIONS: dict[str, str] = {
    "constant": "dx/dt = 0",
    "exponential": "dx/dt = r*x",
    "gompertz": "dx/dt = r*x*log(K/x)",
    "logistic": "dx/dt = r*x*(1-x/K)",
}


def _predict(
    family: ODEFamilyV52,
    times: np.ndarray,
    x0: float,
    parameters: np.ndarray,
) -> np.ndarray:
    t = times - float(times[0])
    if family == "constant":
        return np.full_like(t, x0, dtype=float)
    r = float(parameters[0])
    if family == "exponential":
        return x0 * np.exp(np.clip(r * t, -60, 60))
    k = float(parameters[1])
    if family == "logistic":
        ratio = max(k / x0 - 1.0, -0.999999)
        return k / (1.0 + ratio * np.exp(np.clip(-r * t, -60, 60)))
    if family == "gompertz":
        return k * np.exp(
            np.log(x0 / k) * np.exp(np.clip(-r * t, -60, 60))
        )
    raise ValueError(f"unknown ODE family: {family}")


def _bounds(
    family: ODEFamilyV52, times: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    span = max(float(times[-1] - times[0]), 1e-9)
    rate = 1.0 / span
    if family == "constant":
        return np.array([]), np.array([]), np.array([])
    if family == "exponential":
        initial = np.array([max(math.log(values[-1] / values[0]) / span, 0)])
        return initial, np.array([-10 * rate]), np.array([10 * rate])
    lower_k = max(float(values.max()) * 1.0001, float(values[0]) * 1.001)
    upper_k = max(lower_k * 2, float(values.max()) * 100)
    initial = np.array([rate, max(float(values.max()) * 1.5, lower_k * 1.1)])
    return (
        initial,
        np.array([-10 * rate, lower_k]),
        np.array([10 * rate, upper_k]),
    )


def fit_ode_v52(
    family: ODEFamilyV52,
    times: np.ndarray,
    values: np.ndarray,
) -> ODEFitV52:
    x0 = float(values[0])
    scale = max(float(np.mean(values)), 1e-12)
    initial, lower, upper = _bounds(family, times, values)
    if family == "constant":
        residual = _predict(family, times, x0, initial) - values
        return ODEFitV52.seal(
            family=family,
            parameter_names=[],
            parameter_values=[],
            training_rmse=float(np.sqrt(np.mean(residual**2))),
            parameter_condition_number=1.0,
            optimizer_converged=True,
            optimizer_evaluations=0,
        )

    result = least_squares(
        lambda params: (
            _predict(family, times, x0, params) - values
        )
        / scale,
        initial,
        bounds=(lower, upper),
        method="trf",
        max_nfev=4000,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    prediction = _predict(family, times, x0, result.x)
    information = result.jac.T @ result.jac
    condition = (
        float(np.linalg.cond(information))
        if information.size > 1
        else 1.0
    )
    if not math.isfinite(condition):
        condition = 1e300
    names = ["r"] if family == "exponential" else ["K", "r"]
    values_by_name = (
        [float(result.x[0])]
        if family == "exponential"
        else [float(result.x[1]), float(result.x[0])]
    )
    return ODEFitV52.seal(
        family=family,
        parameter_names=names,
        parameter_values=values_by_name,
        training_rmse=float(np.sqrt(np.mean((prediction - values) ** 2))),
        parameter_condition_number=max(condition, 1.0),
        optimizer_converged=bool(result.success),
        optimizer_evaluations=int(result.nfev),
    )


def _parameter_vector(fit: ODEFitV52) -> np.ndarray:
    values = dict(zip(fit.parameter_names, fit.parameter_values))
    if fit.family == "constant":
        return np.array([])
    if fit.family == "exponential":
        return np.array([values["r"]])
    return np.array([values["r"], values["K"]])


def _split(
    snapshot: ODETimeSeriesSnapshotV52, fraction: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(snapshot.times)
    split = min(max(int(count * fraction), 2), count - 2)
    times = np.asarray(snapshot.times, dtype=float)
    values = np.asarray(snapshot.observations, dtype=float)
    return times[:split], values[:split], times[split:], values[split:]


def _candidate_evidence(
    family: ODEFamilyV52,
    train_t: np.ndarray,
    train_y: np.ndarray,
    validation_t: np.ndarray,
    validation_y: np.ndarray,
) -> ODECandidateEvidenceV52:
    fit = fit_ode_v52(family, train_t, train_y)
    parameters = _parameter_vector(fit)
    all_times = np.concatenate(([train_t[0]], validation_t))
    prediction = _predict(family, all_times, float(train_y[0]), parameters)[1:]
    rmse = float(np.sqrt(np.mean((prediction - validation_y) ** 2)))
    relative = rmse / max(float(np.mean(validation_y)), 1e-12)
    complexity_penalty = 0.005 * len(parameters)
    forecast_t = np.asarray(
        [train_t[0], validation_t[-1] + np.median(np.diff(validation_t))]
    )
    forecast = float(
        _predict(family, forecast_t, float(train_y[0]), parameters)[-1]
    )
    return ODECandidateEvidenceV52.seal(
        candidate_id=family,
        equation=_EQUATIONS[family],
        fit=fit,
        validation_rmse=rmse,
        validation_relative_rmse=relative,
        validation_score=-(relative + complexity_penalty),
        forecast_value=max(forecast, 1e-12),
    )


def _l2_evidence() -> ODELevelEvidenceV52:
    times = np.linspace(0, 4, 41)
    x0 = 2.0
    r = 0.25
    k = 12.0
    analytic = _predict("logistic", times, x0, np.array([r, k]))
    numeric = solve_ivp(
        lambda _t, state: r * state * (1 - state / k),
        (0, 4),
        [x0],
        t_eval=times,
        rtol=1e-11,
        atol=1e-12,
    ).y[0]
    solver_error = float(np.max(np.abs(analytic - numeric)))
    constant_limit = float(
        np.max(
            np.abs(
                _predict("exponential", times, x0, np.array([0.0])) - x0
            )
        )
    )
    huge_k = _predict("logistic", times, x0, np.array([r, 1e9]))
    exponential = _predict("exponential", times, x0, np.array([r]))
    large_capacity_error = float(
        np.max(np.abs(huge_k - exponential)) / np.max(exponential)
    )
    checks = {
        "analytic_numeric_agreement": solver_error <= 1e-7,
        "zero_rate_constant_limit": constant_limit <= 1e-12,
        "large_capacity_exponential_limit": large_capacity_error <= 1e-6,
    }
    return ODELevelEvidenceV52.seal(
        level="L2",
        status="PASS" if all(checks.values()) else "FAIL",
        checks=checks,
        metrics={
            "analytic_numeric_max_error": solver_error,
            "constant_limit_max_error": constant_limit,
            "large_capacity_relative_error": large_capacity_error,
        },
        thresholds={
            "analytic_numeric_max_error": 1e-7,
            "constant_limit_max_error": 1e-12,
            "large_capacity_relative_error": 1e-6,
        },
        evidence={
            "numeric_solver": "scipy.integrate.solve_ivp",
            "families_checked": ["exponential", "logistic"],
        },
    )


def _lag1(values: np.ndarray) -> float:
    if len(values) < 3 or np.std(values[:-1]) == 0 or np.std(values[1:]) == 0:
        return 0.0
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def _l4_evidence(
    *,
    selected: ODECandidateEvidenceV52,
    candidates: list[ODECandidateEvidenceV52],
    train_t: np.ndarray,
    train_y: np.ndarray,
    validation_t: np.ndarray,
    thresholds: ODEThresholdsV52,
) -> ODELevelEvidenceV52:
    family = selected.candidate_id
    fit = selected.fit
    fitted = _predict(
        family, train_t, float(train_y[0]), _parameter_vector(fit)
    )
    residuals = train_y - fitted
    rng = np.random.default_rng(thresholds.bootstrap_seed)
    horizon = float(
        validation_t[-1] + np.median(np.diff(validation_t))
    )
    forecast_times = np.array([train_t[0], horizon])
    forecasts: list[float] = []
    failures = 0
    for _ in range(thresholds.bootstrap_replicates):
        try:
            synthetic = np.maximum(
                fitted + rng.choice(residuals, size=len(residuals), replace=True),
                1e-9,
            )
            bootstrap_fit = fit_ode_v52(family, train_t, synthetic)
            if not bootstrap_fit.optimizer_converged:
                raise ValueError("optimizer failed")
            forecast = float(
                _predict(
                    family,
                    forecast_times,
                    float(synthetic[0]),
                    _parameter_vector(bootstrap_fit),
                )[-1]
            )
            if not math.isfinite(forecast) or forecast <= 0:
                raise ValueError("invalid forecast")
            forecasts.append(forecast)
        except (ArithmeticError, ValueError):
            failures += 1
    success = len(forecasts) / thresholds.bootstrap_replicates
    if forecasts:
        low, median, high = np.quantile(forecasts, [0.025, 0.5, 0.975])
        interval_width = float((high - low) / max(abs(median), 1e-12))
    else:
        low = median = high = math.nan
        interval_width = math.inf
    window_forecasts: list[float] = []
    for fraction in (0.65, 0.8, 1.0):
        count = max(4, int(len(train_t) * fraction))
        try:
            window_fit = fit_ode_v52(
                family, train_t[:count], train_y[:count]
            )
            prediction = _predict(
                family,
                np.array([train_t[0], horizon]),
                float(train_y[0]),
                _parameter_vector(window_fit),
            )[-1]
            if window_fit.optimizer_converged and math.isfinite(prediction):
                window_forecasts.append(float(prediction))
        except (ArithmeticError, ValueError):
            continue
    window_sensitivity = (
        (max(window_forecasts) - min(window_forecasts))
        / max(abs(float(np.median(window_forecasts))), 1e-12)
        if len(window_forecasts) == 3
        else math.inf
    )
    ensemble = [item.forecast_value for item in candidates]
    ensemble_cv = float(
        np.std(ensemble, ddof=1) / max(np.mean(ensemble), 1e-12)
    )
    checks = {
        "bootstrap_success_fraction": (
            success >= thresholds.minimum_bootstrap_success_fraction
        ),
        "forecast_interval_width_bounded": (
            interval_width
            <= thresholds.maximum_forecast_interval_relative_width
        ),
        "window_sensitivity_bounded": (
            window_sensitivity
            <= thresholds.maximum_window_sensitivity_relative_range
        ),
        "ensemble_disagreement_bounded": (
            ensemble_cv
            <= thresholds.maximum_ensemble_forecast_coefficient_of_variation
        ),
        "support_declared": True,
    }
    return ODELevelEvidenceV52.seal(
        level="L4",
        status="PASS" if all(checks.values()) else "FAIL",
        checks=checks,
        metrics={
            "bootstrap_success_fraction": success,
            "bootstrap_failures": failures,
            "forecast_interval_low": (
                float(low) if math.isfinite(float(low)) else None
            ),
            "forecast_interval_median": (
                float(median) if math.isfinite(float(median)) else None
            ),
            "forecast_interval_high": (
                float(high) if math.isfinite(float(high)) else None
            ),
            "forecast_interval_relative_width": (
                interval_width if math.isfinite(interval_width) else None
            ),
            "window_sensitivity_relative_range": (
                window_sensitivity
                if math.isfinite(window_sensitivity)
                else None
            ),
            "ensemble_forecast_coefficient_of_variation": ensemble_cv,
        },
        thresholds={
            "bootstrap_replicates": thresholds.bootstrap_replicates,
            "minimum_bootstrap_success_fraction": (
                thresholds.minimum_bootstrap_success_fraction
            ),
            "maximum_forecast_interval_relative_width": (
                thresholds.maximum_forecast_interval_relative_width
            ),
            "maximum_window_sensitivity_relative_range": (
                thresholds.maximum_window_sensitivity_relative_range
            ),
            "maximum_ensemble_forecast_coefficient_of_variation": (
                thresholds.maximum_ensemble_forecast_coefficient_of_variation
            ),
        },
        evidence={
            "bootstrap_seed": thresholds.bootstrap_seed,
            "forecast_hash": sha256_value(forecasts),
            "window_forecasts": window_forecasts,
            "ensemble_forecasts": ensemble,
            "support": {
                "state": "positive scalar state only",
                "time": "frozen observed interval plus one validation horizon",
                "dynamics": "autonomous smooth ODE candidate registry only",
            },
        },
    )


def build_ode_bundle_v52(
    *,
    snapshot: ODETimeSeriesSnapshotV52,
    thresholds: ODEThresholdsV52,
    replay_output_hashes: list[str] | None = None,
) -> ODEScientificBundleV52:
    snapshot.assert_sealed()
    thresholds.assert_sealed()
    train_t, train_y, validation_t, validation_y = _split(
        snapshot, thresholds.split_fraction
    )
    families: list[ODEFamilyV52] = [
        "constant",
        "exponential",
        "gompertz",
        "logistic",
    ]
    candidates = sorted(
        (
            _candidate_evidence(
                family, train_t, train_y, validation_t, validation_y
            )
            for family in families
        ),
        key=lambda item: item.candidate_id,
    )
    selected = sorted(
        candidates,
        key=lambda item: (
            -item.validation_score,
            len(item.fit.parameter_names),
            item.candidate_id,
        ),
    )[0]
    replay_hashes = list(replay_output_hashes or [])
    l0_checks = {
        "fresh_subprocess_replays_present": len(replay_hashes) == 2,
        "replay_output_hashes_identical": (
            len(replay_hashes) == 2 and len(set(replay_hashes)) == 1
        ),
        "source_and_environment_bound": len(replay_hashes) == 2,
    }
    l0 = ODELevelEvidenceV52.seal(
        level="L0",
        status=(
            "PASS"
            if all(l0_checks.values())
            else "NOT_RUN"
            if not replay_hashes
            else "FAIL"
        ),
        checks=l0_checks,
        metrics={"replay_count": len(replay_hashes)},
        thresholds={"fresh_subprocess_replays": 2},
        evidence={
            "replay_output_hashes": replay_hashes,
            "adapter_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "platform": platform.platform(),
        },
    )
    l1_checks = {
        "snapshot_sealed": snapshot.snapshot_hash == snapshot.content_hash(),
        "units_declared": bool(snapshot.time_unit and snapshot.state_unit),
        "strictly_increasing_time": all(
            right > left
            for left, right in zip(snapshot.times, snapshot.times[1:])
        ),
        "positive_finite_state": all(
            math.isfinite(value) and value > 0
            for value in snapshot.observations
        ),
        "slices_large_enough": (
            len(train_t) >= thresholds.minimum_points_per_slice
            and len(validation_t) >= thresholds.minimum_points_per_slice
        ),
    }
    l1 = ODELevelEvidenceV52.seal(
        level="L1",
        status="PASS" if all(l1_checks.values()) else "FAIL",
        checks=l1_checks,
        metrics={
            "observation_count": len(snapshot.times),
            "training_count": len(train_t),
            "validation_count": len(validation_t),
        },
        thresholds={
            "minimum_points_per_slice": thresholds.minimum_points_per_slice
        },
        evidence={
            "snapshot_hash": snapshot.snapshot_hash,
            "source_id": snapshot.source_id,
            "time_unit": snapshot.time_unit,
            "state_unit": snapshot.state_unit,
        },
    )
    l2 = _l2_evidence()
    selected_prediction = _predict(
        selected.candidate_id,
        np.concatenate(([train_t[0]], validation_t)),
        float(train_y[0]),
        _parameter_vector(selected.fit),
    )[1:]
    residual = validation_y - selected_prediction
    residual_lag = abs(_lag1(residual))
    sigma = max(selected.fit.training_rmse, 1e-12)
    coverage = float(np.mean(np.abs(residual) <= 1.96 * sigma))
    constant = next(
        item for item in candidates if item.candidate_id == "constant"
    )
    improvement = 1.0 - selected.validation_rmse / max(
        constant.validation_rmse, 1e-12
    )
    l3_checks = {
        "optimizer_converged": selected.fit.optimizer_converged,
        "validation_error_bounded": (
            selected.validation_relative_rmse
            <= thresholds.maximum_validation_relative_rmse
        ),
        "constant_baseline_improved": (
            improvement >= thresholds.minimum_baseline_relative_improvement
        ),
        "residual_lag_bounded": (
            residual_lag
            <= thresholds.maximum_absolute_residual_lag1_correlation
        ),
        "validation_interval_coverage": (
            coverage >= thresholds.minimum_validation_interval_coverage
        ),
        "identifiability_condition_bounded": (
            selected.fit.parameter_condition_number
            <= thresholds.maximum_parameter_condition_number
        ),
    }
    l3 = ODELevelEvidenceV52.seal(
        level="L3",
        status="PASS" if all(l3_checks.values()) else "FAIL",
        checks=l3_checks,
        metrics={
            "validation_rmse": selected.validation_rmse,
            "validation_relative_rmse": selected.validation_relative_rmse,
            "constant_baseline_relative_improvement": improvement,
            "absolute_residual_lag1_correlation": residual_lag,
            "validation_interval_coverage": coverage,
            "parameter_condition_number": selected.fit.parameter_condition_number,
        },
        thresholds={
            "maximum_validation_relative_rmse": (
                thresholds.maximum_validation_relative_rmse
            ),
            "minimum_baseline_relative_improvement": (
                thresholds.minimum_baseline_relative_improvement
            ),
            "maximum_absolute_residual_lag1_correlation": (
                thresholds.maximum_absolute_residual_lag1_correlation
            ),
            "minimum_validation_interval_coverage": (
                thresholds.minimum_validation_interval_coverage
            ),
            "maximum_parameter_condition_number": (
                thresholds.maximum_parameter_condition_number
            ),
        },
        evidence={
            "selected_candidate": selected.candidate_id,
            "selected_fit_hash": selected.fit.fit_hash,
            "constant_baseline_hash": constant.fit.fit_hash,
            "validation_residual_hash": sha256_value(residual.tolist()),
        },
    )
    l4 = _l4_evidence(
        selected=selected,
        candidates=candidates,
        train_t=train_t,
        train_y=train_y,
        validation_t=validation_t,
        thresholds=thresholds,
    )
    levels = [l0, l1, l2, l3, l4]
    return ODEScientificBundleV52.seal(
        task_id=snapshot.task_id,
        snapshot_hash=snapshot.snapshot_hash,
        threshold_hash=thresholds.threshold_hash,
        candidate_registry_hash=sha256_value(
            [{"candidate_id": item, "equation": _EQUATIONS[item]} for item in families]
        ),
        selected_candidate_id=selected.candidate_id,
        selected_fit_hash=selected.fit.fit_hash,
        candidates=candidates,
        levels=levels,
        scientific_acceptance=all(item.status == "PASS" for item in levels),
        fixture_only=snapshot.fixture_only,
    )


def deterministic_ode_replay_hash_v52(
    snapshot: ODETimeSeriesSnapshotV52,
    thresholds: ODEThresholdsV52,
) -> str:
    bundle = build_ode_bundle_v52(
        snapshot=snapshot, thresholds=thresholds, replay_output_hashes=None
    )
    return sha256_value(
        {
            "snapshot_hash": bundle.snapshot_hash,
            "threshold_hash": bundle.threshold_hash,
            "selected_candidate_id": bundle.selected_candidate_id,
            "selected_fit_hash": bundle.selected_fit_hash,
            "candidates": [
                {
                    "candidate_id": item.candidate_id,
                    "fit_hash": item.fit.fit_hash,
                    "validation_score": item.validation_score,
                }
                for item in bundle.candidates
            ],
            "levels_l1_l4": [
                item.evidence_hash for item in bundle.levels if item.level != "L0"
            ],
        }
    )


def run_ode_replays_v52(
    replay_input_path: str | Path,
    *,
    count: int = 2,
    timeout_seconds: int = 600,
) -> list[str]:
    input_path = Path(replay_input_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "TEMP": os.environ.get("TEMP", ""),
        "TMP": os.environ.get("TMP", ""),
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
    }
    hashes: list[str] = []
    for _ in range(count):
        with tempfile.TemporaryDirectory(prefix="fma-v52-ode-replay-") as temp:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "fma.v5_2.ode_system",
                    "replay",
                    str(input_path),
                ],
                cwd=temp,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                "ODE replay failed; stderr_sha256="
                + hashlib.sha256(completed.stderr.encode()).hexdigest()
            )
        hashes.append(str(json.loads(completed.stdout)["deterministic_output_hash"]))
    return hashes


def _read_manifest_file(context: AdapterContextV50, relative_path: str) -> bytes:
    path = (context.workspace_root / relative_path).resolve()
    root = context.workspace_root.resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError(relative_path)
    binding = next(
        (
            item
            for item in context.manifest.files
            if item.relative_path == relative_path
        ),
        None,
    )
    if binding is None:
        raise ValueError("ODE bundle is absent from frozen manifest")
    payload = path.read_bytes()
    if (
        len(payload) != binding.size_bytes
        or hashlib.sha256(payload).hexdigest() != binding.sha256
    ):
        raise ValueError("ODE bundle differs from frozen manifest")
    return payload


class ODELevelAdapterV52:
    adapter_id = "scalar_ode_scientific_adapter"
    adapter_version = "5.2"

    def __init__(self, level: LevelV52) -> None:
        self.level = level
        self.check_id = f"scalar_ode_{level.lower()}"

    def run(self, context: AdapterContextV50) -> AdapterOutcomeV50:
        bundle = ODEScientificBundleV52.model_validate_json(
            _read_manifest_file(
                context, "results/ode_scientific_bundle.json"
            )
        )
        evidence = next(item for item in bundle.levels if item.level == self.level)
        payload: dict[str, Any] = {
            "bundle_hash": bundle.bundle_hash,
            "level_evidence": evidence.model_dump(mode="json"),
            "fixture_only": bundle.fixture_only,
            "scientific_qualification_granted": False,
            "real_world_action_authorized": False,
        }
        if self.level == "L0":
            code_manifest = CodeManifestV50.model_validate_json(
                _read_manifest_file(context, "results/code_manifest.json")
            )
            payload["computation_artifact_sha256"] = (
                code_manifest.replay_receipt_hash
            )
        return AdapterOutcomeV50(
            status="PASS" if evidence.status == "PASS" else "FAIL",
            reason_code=(
                "scalar_ode_level_passed"
                if evidence.status == "PASS"
                else f"scalar_ode_level_{evidence.status.lower()}"
            ),
            metrics=evidence.metrics,
            thresholds=evidence.thresholds,
            evidence_payloads=[payload],
        )


def register_ode_adapters_v52(registry: Any) -> None:
    for level in ("L0", "L1", "L2", "L3", "L4"):
        registry.register(ODELevelAdapterV52(level))


def _main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] != "replay":
        raise SystemExit("usage: python -m fma.v5_2.ode_system replay INPUT")
    payload = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    snapshot = ODETimeSeriesSnapshotV52.model_validate(payload["snapshot"])
    thresholds = ODEThresholdsV52.model_validate(payload["thresholds"])
    print(
        json.dumps(
            {
                "deterministic_output_hash": deterministic_ode_replay_hash_v52(
                    snapshot, thresholds
                )
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
