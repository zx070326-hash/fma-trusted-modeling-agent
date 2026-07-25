"""V5.6 hybrid ODE trend plus explicit residual-process adapter.

V5.6 is additive.  It leaves the V5.2 autonomous-ODE and V5.3 forecast
artifacts unchanged, and adds a graph-governed recovery branch for serially
structured observation residuals.
"""

from __future__ import annotations

import hashlib
import hmac
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
from scipy.optimize import least_squares

from fma.hashing import canonical_json, sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5.check_registry import AdapterContextV50, AdapterOutcomeV50
from fma.v5.workspace_schemas import CodeManifestV50
from fma.v5_2.ode_system import ODEFamilyV52, ODETimeSeriesSnapshotV52, _predict


ResidualModeV56 = Literal["trend_only", "ar1_residual"]
LevelV56 = Literal["L0", "L1", "L2", "L3", "L4"]
LevelStatusV56 = Literal["PASS", "FAIL", "NOT_RUN"]
FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFinite = Annotated[float, Field(ge=0, allow_inf_nan=False)]

FAMILIES: tuple[ODEFamilyV52, ...] = (
    "constant",
    "exponential",
    "gompertz",
    "logistic",
)
RESIDUAL_MODES: tuple[ResidualModeV56, ...] = (
    "trend_only",
    "ar1_residual",
)


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _lag1(values: np.ndarray) -> float:
    if len(values) < 3:
        return 0.0
    left = values[:-1]
    right = values[1:]
    if np.std(left) <= 1e-15 or np.std(right) <= 1e-15:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


class HybridODEThresholdsV56(StrictModel):
    schema_version: Literal["5.6"] = "5.6"
    split_fraction: Annotated[float, Field(gt=0.5, lt=0.9)] = 0.7
    minimum_points_per_slice: Annotated[int, Field(ge=6)] = 8
    recovery_trigger_absolute_residual_lag1_correlation: Annotated[
        float,
        Field(gt=0, lt=1, allow_inf_nan=False),
    ] = 0.60
    maximum_innovation_absolute_lag1_correlation: Annotated[
        float,
        Field(gt=0, lt=1, allow_inf_nan=False),
    ] = 0.35
    maximum_absolute_ar1_phi: Annotated[
        float,
        Field(gt=0.5, lt=1, allow_inf_nan=False),
    ] = 0.95
    minimum_ar1_validation_relative_improvement: Annotated[
        float,
        Field(ge=0, le=1, allow_inf_nan=False),
    ] = 0.05
    maximum_phi_window_range: Annotated[
        float,
        Field(gt=0, le=1, allow_inf_nan=False),
    ] = 0.30
    maximum_innovation_mean_shift_standardized: Annotated[
        float,
        Field(gt=0, allow_inf_nan=False),
    ] = 1.50
    maximum_single_innovation_standardized: Annotated[
        float,
        Field(gt=0, allow_inf_nan=False),
    ] = 5.0
    maximum_validation_relative_rmse: Annotated[
        float,
        Field(gt=0, allow_inf_nan=False),
    ] = 0.15
    minimum_persistence_relative_improvement: Annotated[
        float,
        Field(ge=0, le=1, allow_inf_nan=False),
    ] = 0.10
    minimum_validation_interval_coverage: Annotated[
        float,
        Field(ge=0, le=1, allow_inf_nan=False),
    ] = 0.50
    maximum_dimensionless_parameter_condition_number: Annotated[
        float,
        Field(gt=1, allow_inf_nan=False),
    ] = 1e8
    selection_complexity_penalty_per_parameter: Annotated[
        float,
        Field(ge=0, allow_inf_nan=False),
    ] = 0.002
    bootstrap_replicates: Annotated[int, Field(ge=20, le=5000)] = 40
    bootstrap_seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 130363
    minimum_bootstrap_success_fraction: Annotated[
        float,
        Field(gt=0, le=1, allow_inf_nan=False),
    ] = 0.80
    maximum_forecast_interval_relative_width: Annotated[
        float,
        Field(gt=0, allow_inf_nan=False),
    ] = 2.0
    maximum_window_sensitivity_relative_range: Annotated[
        float,
        Field(gt=0, allow_inf_nan=False),
    ] = 1.0
    threshold_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_thresholds(self) -> "HybridODEThresholdsV56":
        if (
            self.recovery_trigger_absolute_residual_lag1_correlation
            <= self.maximum_innovation_absolute_lag1_correlation
        ):
            raise ValueError("recovery trigger must exceed innovation lag limit")
        if self.threshold_hash and self.threshold_hash != self.content_hash():
            raise ValueError("hybrid threshold hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "threshold_hash")

    def assert_sealed(self) -> None:
        if not self.threshold_hash or self.threshold_hash != self.content_hash():
            raise ValueError("hybrid thresholds are not sealed")

    @classmethod
    def seal(cls, **data: object) -> "HybridODEThresholdsV56":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"threshold_hash"})
        payload["threshold_hash"] = draft.content_hash()
        return cls(**payload)


class DimensionlessTrendFitV56(StrictModel):
    schema_version: Literal["5.6"] = "5.6"
    family: ODEFamilyV52
    time_origin: FiniteNumber
    time_span: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    state_scale: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    initial_state_value: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    dimensionless_parameter_names: list[Identifier]
    dimensionless_parameter_values: list[FiniteNumber]
    physical_parameter_names: list[Identifier]
    physical_parameter_values: list[FiniteNumber]
    training_rmse: NonNegativeFinite
    dimensionless_parameter_condition_number: Annotated[
        float,
        Field(ge=1, allow_inf_nan=False),
    ]
    optimizer_converged: bool
    optimizer_evaluations: Annotated[int, Field(ge=0)]
    fit_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_fit(self) -> "DimensionlessTrendFitV56":
        if len(self.dimensionless_parameter_names) != len(
            self.dimensionless_parameter_values
        ):
            raise ValueError("dimensionless trend names and values differ")
        if len(self.physical_parameter_names) != len(
            self.physical_parameter_values
        ):
            raise ValueError("physical trend names and values differ")
        if self.dimensionless_parameter_names != sorted(
            set(self.dimensionless_parameter_names)
        ):
            raise ValueError("dimensionless trend names must be sorted and unique")
        if self.physical_parameter_names != sorted(
            set(self.physical_parameter_names)
        ):
            raise ValueError("physical trend names must be sorted and unique")
        if self.fit_hash and self.fit_hash != self.content_hash():
            raise ValueError("dimensionless trend fit hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "fit_hash")

    @classmethod
    def seal(cls, **data: object) -> "DimensionlessTrendFitV56":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"fit_hash"})
        payload["fit_hash"] = draft.content_hash()
        return cls(**payload)


class ResidualProcessFitV56(StrictModel):
    schema_version: Literal["5.6"] = "5.6"
    mode: ResidualModeV56
    raw_phi: FiniteNumber
    effective_phi: FiniteNumber
    training_innovation_scale: NonNegativeFinite
    training_residual_hash: Sha256
    training_innovation_hash: Sha256
    fit_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_fit(self) -> "ResidualProcessFitV56":
        if self.mode == "trend_only" and (
            self.raw_phi != 0 or self.effective_phi != 0
        ):
            raise ValueError("trend-only residual process must have phi=0")
        if not -0.999 <= self.effective_phi <= 0.999:
            raise ValueError("effective AR coefficient is outside safe recursion")
        if self.fit_hash and self.fit_hash != self.content_hash():
            raise ValueError("residual-process fit hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "fit_hash")

    @classmethod
    def seal(cls, **data: object) -> "ResidualProcessFitV56":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"fit_hash"})
        payload["fit_hash"] = draft.content_hash()
        return cls(**payload)


class HybridCandidateEvidenceV56(StrictModel):
    schema_version: Literal["5.6"] = "5.6"
    candidate_id: Identifier
    family: ODEFamilyV52
    residual_mode: ResidualModeV56
    trend_fit: DimensionlessTrendFitV56
    residual_fit: ResidualProcessFitV56
    parameter_count: Annotated[int, Field(ge=0)]
    validation_rmse: NonNegativeFinite
    validation_relative_rmse: NonNegativeFinite
    validation_score: FiniteNumber
    persistence_relative_improvement: FiniteNumber
    same_family_ar1_relative_improvement: FiniteNumber | None
    absolute_validation_residual_lag1_correlation: NonNegativeFinite
    absolute_validation_innovation_lag1_correlation: NonNegativeFinite
    phi_window_range: NonNegativeFinite
    innovation_mean_shift_standardized: NonNegativeFinite
    maximum_single_innovation_standardized: NonNegativeFinite
    validation_interval_coverage: Annotated[
        float,
        Field(ge=0, le=1, allow_inf_nan=False),
    ]
    forecast_value: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    validation_residual_hash: Sha256
    validation_innovation_hash: Sha256
    admissibility_checks: dict[Identifier, bool]
    scientifically_admissible: bool
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_candidate(self) -> "HybridCandidateEvidenceV56":
        expected_id = f"{self.family}.{self.residual_mode}"
        if self.candidate_id != expected_id:
            raise ValueError("hybrid candidate ID differs from family and mode")
        if self.trend_fit.family != self.family:
            raise ValueError("hybrid candidate trend family differs")
        if self.residual_fit.mode != self.residual_mode:
            raise ValueError("hybrid candidate residual mode differs")
        expected = bool(self.admissibility_checks) and all(
            self.admissibility_checks.values()
        )
        if self.scientifically_admissible != expected:
            raise ValueError("hybrid candidate admissibility differs from checks")
        if self.evidence_hash and self.evidence_hash != self.content_hash():
            raise ValueError("hybrid candidate evidence hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evidence_hash")

    @classmethod
    def seal(cls, **data: object) -> "HybridCandidateEvidenceV56":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"evidence_hash"})
        payload["evidence_hash"] = draft.content_hash()
        return cls(**payload)


class HybridCandidateGraphV56(StrictModel):
    schema_version: Literal["5.6-candidate-graph"] = "5.6-candidate-graph"
    initial_candidate_ids: list[Identifier]
    recovery_candidate_ids: list[Identifier]
    evaluated_candidate_ids: list[Identifier]
    admissible_candidate_ids: list[Identifier]
    initial_selected_candidate_id: Identifier
    recovery_trigger_value: NonNegativeFinite
    recovery_trigger_threshold: NonNegativeFinite
    recovery_triggered: bool
    selected_candidate_id: Identifier
    graph_checks: dict[Identifier, bool]
    graph_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_graph(self) -> "HybridCandidateGraphV56":
        for values in (
            self.initial_candidate_ids,
            self.recovery_candidate_ids,
            self.evaluated_candidate_ids,
            self.admissible_candidate_ids,
        ):
            if values != sorted(set(values)):
                raise ValueError("hybrid graph candidate IDs must be sorted and unique")
        expected_trigger = (
            self.recovery_trigger_value > self.recovery_trigger_threshold
        )
        if self.recovery_triggered != expected_trigger:
            raise ValueError("hybrid graph recovery transition differs")
        if self.initial_selected_candidate_id not in self.initial_candidate_ids:
            raise ValueError("hybrid graph initial selection is absent")
        if self.selected_candidate_id not in self.evaluated_candidate_ids:
            raise ValueError("hybrid graph final selection is absent")
        if self.graph_hash and self.graph_hash != self.content_hash():
            raise ValueError("hybrid candidate graph hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "graph_hash")

    @classmethod
    def seal(cls, **data: object) -> "HybridCandidateGraphV56":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"graph_hash"})
        payload["graph_hash"] = draft.content_hash()
        return cls(**payload)


class HybridLevelEvidenceV56(StrictModel):
    schema_version: Literal["5.6"] = "5.6"
    level: LevelV56
    status: LevelStatusV56
    checks: dict[Identifier, bool]
    metrics: dict[Identifier, FiniteNumber | int | None]
    thresholds: dict[Identifier, FiniteNumber | int]
    evidence: dict[str, Any]
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_level(self) -> "HybridLevelEvidenceV56":
        if self.status == "PASS" and (
            not self.checks or not all(self.checks.values())
        ):
            raise ValueError("passing hybrid level contains failed checks")
        if self.evidence_hash and self.evidence_hash != self.content_hash():
            raise ValueError("hybrid level evidence hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evidence_hash")

    @classmethod
    def seal(cls, **data: object) -> "HybridLevelEvidenceV56":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"evidence_hash"})
        payload["evidence_hash"] = draft.content_hash()
        return cls(**payload)


class HybridScientificBundleV56(StrictModel):
    schema_version: Literal["5.6"] = "5.6"
    task_id: Identifier
    domain: Literal["scalar_ode_with_residual_process"] = (
        "scalar_ode_with_residual_process"
    )
    snapshot_hash: Sha256
    threshold_hash: Sha256
    candidate_registry_hash: Sha256
    graph: HybridCandidateGraphV56
    selected_candidate_id: Identifier
    candidates: list[HybridCandidateEvidenceV56]
    levels: list[HybridLevelEvidenceV56]
    replay_receipt_hashes: list[Sha256]
    scientific_acceptance: bool
    fixture_only: bool
    causal_mechanism_identified: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    bundle_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bundle(self) -> "HybridScientificBundleV56":
        candidate_ids = [item.candidate_id for item in self.candidates]
        if candidate_ids != sorted(set(candidate_ids)):
            raise ValueError("hybrid candidates must be sorted and unique")
        if self.selected_candidate_id != self.graph.selected_candidate_id:
            raise ValueError("hybrid bundle and graph selections differ")
        if self.selected_candidate_id not in candidate_ids:
            raise ValueError("hybrid selected candidate is absent")
        if [item.level for item in self.levels] != [
            "L0",
            "L1",
            "L2",
            "L3",
            "L4",
        ]:
            raise ValueError("hybrid bundle must contain ordered L0-L4 evidence")
        expected = all(item.status == "PASS" for item in self.levels)
        if self.scientific_acceptance != expected:
            raise ValueError("hybrid acceptance differs from L0-L4")
        if self.replay_receipt_hashes and len(self.replay_receipt_hashes) != 2:
            raise ValueError("hybrid replay receipts must be absent or a pair")
        if self.bundle_hash and self.bundle_hash != self.content_hash():
            raise ValueError("hybrid bundle hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "bundle_hash")

    @classmethod
    def seal(cls, **data: object) -> "HybridScientificBundleV56":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"bundle_hash"})
        payload["bundle_hash"] = draft.content_hash()
        return cls(**payload)


class HybridReplayReceiptV56(StrictModel):
    schema_version: Literal["5.6-replay-receipt"] = "5.6-replay-receipt"
    replay_id: Identifier
    replay_index: Annotated[int, Field(ge=1)]
    process_id: Annotated[int, Field(ge=1)]
    input_bytes_hash: Sha256
    input_semantic_hash: Sha256
    command_hash: Sha256
    exit_code: Literal[0] = 0
    stdout_hash: Sha256
    stderr_hash: Sha256
    deterministic_output_hash: Sha256
    source_hash: Sha256
    executable_hash: Sha256
    environment_fingerprint: Sha256
    fresh_process: Literal[True] = True
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "HybridReplayReceiptV56":
        if self.receipt_hash and (
            not self.authority_auth_tag
            or self.receipt_hash != self.content_hash()
        ):
            raise ValueError("hybrid replay receipt envelope differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "authority_auth_tag", "receipt_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")


class HybridReplayAuthorityV56:
    """Local HMAC authority; secret bytes stay outside model context."""

    def __init__(self, *, key_id: str, secret: bytes) -> None:
        if len(secret) < 32:
            raise ValueError("hybrid replay authority secret needs 32 bytes")
        self.key_id = key_id
        self._secret = bytes(secret)

    def _mac(self, unsigned_hash: str) -> str:
        return hmac.new(
            self._secret,
            f"hybrid_ode_replay_v56:{unsigned_hash}".encode(),
            hashlib.sha256,
        ).hexdigest()

    def issue(self, **data: object) -> HybridReplayReceiptV56:
        data["authority_key_id"] = self.key_id
        unsigned = HybridReplayReceiptV56(**data)
        payload = unsigned.model_dump(mode="json")
        payload["authority_auth_tag"] = self._mac(unsigned.unsigned_hash())
        tagged = HybridReplayReceiptV56(**payload)
        final = tagged.model_dump(mode="json")
        final["receipt_hash"] = tagged.content_hash()
        return HybridReplayReceiptV56(**final)

    def verify(self, receipt: HybridReplayReceiptV56) -> bool:
        try:
            return bool(
                receipt.receipt_hash
                and receipt.receipt_hash == receipt.content_hash()
                and receipt.authority_key_id == self.key_id
                and receipt.authority_auth_tag
                and hmac.compare_digest(
                    receipt.authority_auth_tag,
                    self._mac(receipt.unsigned_hash()),
                )
            )
        except (TypeError, ValueError):
            return False


def _candidate_id(family: ODEFamilyV52, mode: ResidualModeV56) -> str:
    return f"{family}.{mode}"


def _fit_trend(
    family: ODEFamilyV52,
    times: np.ndarray,
    values: np.ndarray,
) -> DimensionlessTrendFitV56:
    origin = float(times[0])
    span = max(float(times[-1] - times[0]), 1e-12)
    scale = max(float(np.mean(values)), 1e-12)
    dimensionless_t = (times - origin) / span
    dimensionless_y = values / scale
    x0 = float(dimensionless_y[0])

    if family == "constant":
        prediction = np.full_like(dimensionless_t, x0)
        return DimensionlessTrendFitV56.seal(
            family=family,
            time_origin=origin,
            time_span=span,
            state_scale=scale,
            initial_state_value=float(values[0]),
            dimensionless_parameter_names=[],
            dimensionless_parameter_values=[],
            physical_parameter_names=[],
            physical_parameter_values=[],
            training_rmse=float(
                np.sqrt(np.mean((prediction * scale - values) ** 2))
            ),
            dimensionless_parameter_condition_number=1.0,
            optimizer_converged=True,
            optimizer_evaluations=0,
        )

    if family == "exponential":
        initial = np.array(
            [float(np.clip(math.log(values[-1] / values[0]), -2, 2))]
        )
        lower = np.array([-10.0])
        upper = np.array([10.0])
    else:
        lower_k = max(
            float(dimensionless_y.max()) * 1.0001,
            x0 * 1.001,
        )
        upper_k = max(lower_k * 2, float(dimensionless_y.max()) * 100)
        initial = np.array(
            [
                1.0,
                max(float(dimensionless_y.max()) * 1.5, lower_k * 1.1),
            ]
        )
        lower = np.array([-10.0, lower_k])
        upper = np.array([10.0, upper_k])

    result = least_squares(
        lambda params: (
            _predict(family, dimensionless_t, x0, params) - dimensionless_y
        ),
        initial,
        bounds=(lower, upper),
        method="trf",
        max_nfev=4000,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    predicted = _predict(family, dimensionless_t, x0, result.x) * scale
    information = result.jac.T @ result.jac
    condition = (
        float(np.linalg.cond(information)) if information.size > 1 else 1.0
    )
    if not math.isfinite(condition):
        condition = 1e300
    if family == "exponential":
        dimensionless_names = ["r_times_span"]
        dimensionless_values = [float(result.x[0])]
        physical_names = ["r"]
        physical_values = [float(result.x[0] / span)]
    else:
        dimensionless_names = ["K_over_scale", "r_times_span"]
        dimensionless_values = [float(result.x[1]), float(result.x[0])]
        physical_names = ["K", "r"]
        physical_values = [
            float(result.x[1] * scale),
            float(result.x[0] / span),
        ]
    return DimensionlessTrendFitV56.seal(
        family=family,
        time_origin=origin,
        time_span=span,
        state_scale=scale,
        initial_state_value=float(values[0]),
        dimensionless_parameter_names=dimensionless_names,
        dimensionless_parameter_values=dimensionless_values,
        physical_parameter_names=physical_names,
        physical_parameter_values=physical_values,
        training_rmse=float(np.sqrt(np.mean((predicted - values) ** 2))),
        dimensionless_parameter_condition_number=max(condition, 1.0),
        optimizer_converged=bool(result.success),
        optimizer_evaluations=int(result.nfev),
    )


def _dimensionless_parameter_vector(
    fit: DimensionlessTrendFitV56,
) -> np.ndarray:
    values = dict(
        zip(
            fit.dimensionless_parameter_names,
            fit.dimensionless_parameter_values,
        )
    )
    if fit.family == "constant":
        return np.array([])
    if fit.family == "exponential":
        return np.array([values["r_times_span"]])
    return np.array([values["r_times_span"], values["K_over_scale"]])


def _trend_predict(
    fit: DimensionlessTrendFitV56,
    times: np.ndarray,
) -> np.ndarray:
    dimensionless_times = (times - fit.time_origin) / fit.time_span
    joined = np.concatenate(([0.0], dimensionless_times))
    values = _predict(
        fit.family,
        joined,
        fit.initial_state_value / fit.state_scale,
        _dimensionless_parameter_vector(fit),
    )[1:]
    return values * fit.state_scale


def _estimate_residual_process(
    mode: ResidualModeV56,
    residuals: np.ndarray,
) -> tuple[ResidualProcessFitV56, np.ndarray]:
    if mode == "trend_only":
        raw_phi = effective_phi = 0.0
        innovations = residuals.copy()
    else:
        denominator = float(np.dot(residuals[:-1], residuals[:-1]))
        raw_phi = (
            float(np.dot(residuals[:-1], residuals[1:]) / denominator)
            if denominator > 1e-18
            else 0.0
        )
        effective_phi = float(np.clip(raw_phi, -0.999, 0.999))
        innovations = residuals[1:] - effective_phi * residuals[:-1]
    scale = float(np.std(innovations, ddof=1)) if len(innovations) > 1 else 0.0
    return (
        ResidualProcessFitV56.seal(
            mode=mode,
            raw_phi=raw_phi,
            effective_phi=effective_phi,
            training_innovation_scale=max(scale, 0.0),
            training_residual_hash=sha256_value(residuals.tolist()),
            training_innovation_hash=sha256_value(innovations.tolist()),
        ),
        innovations,
    )


def _forecast_correction(
    *,
    last_residual: float,
    phi: float,
    horizon_steps: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [float((phi**int(step)) * last_residual) for step in horizon_steps],
        dtype=float,
    )


def _recursive_correction(
    *,
    last_residual: float,
    phi: float,
    horizon_count: int,
) -> np.ndarray:
    values: list[float] = []
    current = float(last_residual)
    for _ in range(horizon_count):
        current = phi * current
        values.append(current)
    return np.asarray(values, dtype=float)


def _phi_window_range(
    *,
    family: ODEFamilyV52,
    times: np.ndarray,
    values: np.ndarray,
) -> tuple[float, list[float]]:
    phis: list[float] = []
    for fraction in (0.70, 0.85, 1.0):
        count = max(8, int(len(times) * fraction))
        count = min(count, len(times))
        fit = _fit_trend(family, times[:count], values[:count])
        residuals = values[:count] - _trend_predict(fit, times[:count])
        residual_fit, _ = _estimate_residual_process(
            "ar1_residual",
            residuals,
        )
        phis.append(residual_fit.raw_phi)
    return float(max(phis) - min(phis)), phis


def _candidate_evidence(
    *,
    family: ODEFamilyV52,
    mode: ResidualModeV56,
    train_t: np.ndarray,
    train_y: np.ndarray,
    validation_t: np.ndarray,
    validation_y: np.ndarray,
    persistence_rmse: float,
    thresholds: HybridODEThresholdsV56,
    same_family_trend_rmse: float | None,
) -> HybridCandidateEvidenceV56:
    fit = _fit_trend(family, train_t, train_y)
    fitted_train = _trend_predict(fit, train_t)
    training_residuals = train_y - fitted_train
    residual_fit, training_innovations = _estimate_residual_process(
        mode,
        training_residuals,
    )
    trend_validation = _trend_predict(fit, validation_t)
    actual_trend_residuals = validation_y - trend_validation
    predictions: list[float] = []
    validation_innovations: list[float] = []
    previous = float(training_residuals[-1])
    for trend_value, residual in zip(
        trend_validation,
        actual_trend_residuals,
    ):
        correction = residual_fit.effective_phi * previous
        predictions.append(float(trend_value + correction))
        validation_innovations.append(float(residual - correction))
        previous = float(residual)
    prediction_array = np.asarray(predictions, dtype=float)
    validation_innovation_array = np.asarray(
        validation_innovations,
        dtype=float,
    )
    validation_residuals = validation_y - prediction_array
    rmse = float(np.sqrt(np.mean(validation_residuals**2)))
    relative_rmse = rmse / max(float(np.mean(validation_y)), 1e-12)
    parameter_count = len(fit.dimensionless_parameter_values) + (
        1 if mode == "ar1_residual" else 0
    )
    persistence_improvement = 1.0 - rmse / max(persistence_rmse, 1e-12)
    same_family_improvement = (
        1.0 - rmse / max(same_family_trend_rmse, 1e-12)
        if same_family_trend_rmse is not None
        else None
    )
    residual_lag = abs(_lag1(validation_residuals))
    innovation_lag = abs(_lag1(validation_innovation_array))
    innovation_scale = max(
        residual_fit.training_innovation_scale,
        1e-12,
    )
    training_innovation_mean = (
        float(np.mean(training_innovations))
        if len(training_innovations)
        else 0.0
    )
    validation_mean_shift = abs(
        float(np.mean(validation_innovation_array)) - training_innovation_mean
    ) / innovation_scale
    training_block_shifts = [
        abs(float(np.mean(block)) - training_innovation_mean)
        / innovation_scale
        for block in np.array_split(training_innovations, 3)
        if len(block)
    ]
    training_mean_shift = max(training_block_shifts, default=0.0)
    mean_shift = max(validation_mean_shift, training_mean_shift)
    validation_maximum_single = float(
        np.max(
            np.abs(validation_innovation_array - training_innovation_mean)
        )
        / innovation_scale
    )
    training_maximum_single = (
        float(
            np.max(np.abs(training_innovations - training_innovation_mean))
            / innovation_scale
        )
        if len(training_innovations)
        else 0.0
    )
    maximum_single = max(
        validation_maximum_single,
        training_maximum_single,
    )
    coverage_values = [
        abs(float(error)) <= 1.96 * innovation_scale
        for error in validation_residuals
    ]
    coverage = float(np.mean(coverage_values))
    phi_range, _ = (
        _phi_window_range(family=family, times=train_t, values=train_y)
        if mode == "ar1_residual"
        else (0.0, [0.0, 0.0, 0.0])
    )
    next_time = float(validation_t[-1] + np.median(np.diff(validation_t)))
    next_step = len(validation_t) + 1
    forecast = float(
        _trend_predict(fit, np.asarray([next_time]))[0]
        + _forecast_correction(
            last_residual=float(training_residuals[-1]),
            phi=residual_fit.effective_phi,
            horizon_steps=np.asarray([next_step]),
        )[0]
    )
    checks = {
        "optimizer_converged": fit.optimizer_converged,
        "positive_finite_predictions": bool(
            np.all(np.isfinite(prediction_array))
            and np.all(prediction_array > 0)
            and math.isfinite(forecast)
            and forecast > 0
        ),
        "validation_error_bounded": (
            relative_rmse <= thresholds.maximum_validation_relative_rmse
        ),
        "persistence_improved": (
            persistence_improvement
            >= thresholds.minimum_persistence_relative_improvement
        ),
        "innovation_lag_bounded": (
            innovation_lag
            <= thresholds.maximum_innovation_absolute_lag1_correlation
        ),
        "ar1_stationary_interior": (
            mode == "trend_only"
            or abs(residual_fit.raw_phi)
            <= thresholds.maximum_absolute_ar1_phi
        ),
        "ar1_improvement_material": (
            mode == "trend_only"
            or (
                same_family_improvement is not None
                and same_family_improvement
                >= thresholds.minimum_ar1_validation_relative_improvement
            )
        ),
        "phi_window_stable": (
            phi_range <= thresholds.maximum_phi_window_range
        ),
        "innovation_mean_shift_bounded": (
            mean_shift
            <= thresholds.maximum_innovation_mean_shift_standardized
        ),
        "single_innovation_bounded": (
            maximum_single
            <= thresholds.maximum_single_innovation_standardized
        ),
        "validation_interval_coverage": (
            coverage >= thresholds.minimum_validation_interval_coverage
        ),
        "dimensionless_identifiability_bounded": (
            fit.dimensionless_parameter_condition_number
            <= thresholds.maximum_dimensionless_parameter_condition_number
        ),
    }
    return HybridCandidateEvidenceV56.seal(
        candidate_id=_candidate_id(family, mode),
        family=family,
        residual_mode=mode,
        trend_fit=fit,
        residual_fit=residual_fit,
        parameter_count=parameter_count,
        validation_rmse=rmse,
        validation_relative_rmse=relative_rmse,
        validation_score=-(
            relative_rmse
            + thresholds.selection_complexity_penalty_per_parameter
            * parameter_count
        ),
        persistence_relative_improvement=persistence_improvement,
        same_family_ar1_relative_improvement=same_family_improvement,
        absolute_validation_residual_lag1_correlation=residual_lag,
        absolute_validation_innovation_lag1_correlation=innovation_lag,
        phi_window_range=phi_range,
        innovation_mean_shift_standardized=mean_shift,
        maximum_single_innovation_standardized=maximum_single,
        validation_interval_coverage=coverage,
        forecast_value=forecast,
        validation_residual_hash=sha256_value(validation_residuals.tolist()),
        validation_innovation_hash=sha256_value(
            validation_innovation_array.tolist()
        ),
        admissibility_checks=checks,
        scientifically_admissible=all(checks.values()),
    )


def _split(
    snapshot: ODETimeSeriesSnapshotV52,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(snapshot.times)
    split = min(max(int(count * fraction), 2), count - 2)
    times = np.asarray(snapshot.times, dtype=float)
    values = np.asarray(snapshot.observations, dtype=float)
    return times[:split], values[:split], times[split:], values[split:]


def _selection_key(
    candidate: HybridCandidateEvidenceV56,
) -> tuple[float, int, str]:
    return (
        -candidate.validation_score,
        candidate.parameter_count,
        candidate.candidate_id,
    )


def _l2_evidence() -> HybridLevelEvidenceV56:
    physical_t = np.linspace(10.0, 20.0, 41)
    span = physical_t[-1] - physical_t[0]
    scale = 7.0
    x0 = 2.0
    q = 0.8
    k_scaled = 12.0
    dimensionless_t = (physical_t - physical_t[0]) / span
    normalized = (
        _predict(
            "logistic",
            dimensionless_t,
            x0 / scale,
            np.array([q, k_scaled / scale]),
        )
        * scale
    )
    physical = _predict(
        "logistic",
        physical_t,
        x0,
        np.array([q / span, k_scaled]),
    )
    scaling_error = float(np.max(np.abs(normalized - physical)))
    phi_zero = _forecast_correction(
        last_residual=3.0,
        phi=0.0,
        horizon_steps=np.arange(1, 5),
    )
    closed = _forecast_correction(
        last_residual=3.0,
        phi=0.7,
        horizon_steps=np.arange(1, 6),
    )
    recursive = _recursive_correction(
        last_residual=3.0,
        phi=0.7,
        horizon_count=5,
    )
    recursion_error = float(np.max(np.abs(closed - recursive)))
    checks = {
        "dimensionless_physical_prediction_agreement": scaling_error <= 1e-10,
        "phi_zero_reduces_to_zero_correction": (
            float(np.max(np.abs(phi_zero))) <= 1e-12
        ),
        "closed_form_recursive_ar1_agreement": recursion_error <= 1e-12,
        "stationary_correction_decays": bool(
            all(
                abs(right) < abs(left)
                for left, right in zip(closed, closed[1:])
            )
        ),
    }
    return HybridLevelEvidenceV56.seal(
        level="L2",
        status="PASS" if all(checks.values()) else "FAIL",
        checks=checks,
        metrics={
            "dimensionless_physical_max_error": scaling_error,
            "phi_zero_max_correction": float(np.max(np.abs(phi_zero))),
            "recursive_closed_form_max_error": recursion_error,
        },
        thresholds={
            "dimensionless_physical_max_error": 1e-10,
            "phi_zero_max_correction": 1e-12,
            "recursive_closed_form_max_error": 1e-12,
        },
        evidence={
            "families_checked": ["logistic"],
            "residual_modes_checked": list(RESIDUAL_MODES),
        },
    )


def _l4_evidence(
    *,
    selected: HybridCandidateEvidenceV56,
    candidates: list[HybridCandidateEvidenceV56],
    train_t: np.ndarray,
    train_y: np.ndarray,
    validation_t: np.ndarray,
    thresholds: HybridODEThresholdsV56,
) -> HybridLevelEvidenceV56:
    fitted = _trend_predict(selected.trend_fit, train_t)
    residuals = train_y - fitted
    residual_fit, innovations = _estimate_residual_process(
        selected.residual_mode,
        residuals,
    )
    rng = np.random.default_rng(thresholds.bootstrap_seed)
    horizon_time = float(validation_t[-1] + np.median(np.diff(validation_t)))
    horizon_steps = len(validation_t) + 1
    forecasts: list[float] = []
    failures = 0
    for _ in range(thresholds.bootstrap_replicates):
        try:
            sampled = rng.choice(innovations, size=len(train_y), replace=True)
            synthetic_residual = np.empty(len(train_y), dtype=float)
            synthetic_residual[0] = float(residuals[0])
            for index in range(1, len(train_y)):
                synthetic_residual[index] = (
                    residual_fit.effective_phi * synthetic_residual[index - 1]
                    + sampled[index]
                )
            synthetic = np.maximum(fitted + synthetic_residual, 1e-9)
            bootstrap_trend = _fit_trend(
                selected.family,
                train_t,
                synthetic,
            )
            bootstrap_fitted = _trend_predict(bootstrap_trend, train_t)
            bootstrap_residual = synthetic - bootstrap_fitted
            bootstrap_process, _ = _estimate_residual_process(
                selected.residual_mode,
                bootstrap_residual,
            )
            forecast = float(
                _trend_predict(
                    bootstrap_trend,
                    np.asarray([horizon_time]),
                )[0]
                + _forecast_correction(
                    last_residual=float(bootstrap_residual[-1]),
                    phi=bootstrap_process.effective_phi,
                    horizon_steps=np.asarray([horizon_steps]),
                )[0]
            )
            if (
                not bootstrap_trend.optimizer_converged
                or not math.isfinite(forecast)
                or forecast <= 0
            ):
                raise ValueError("invalid hybrid bootstrap forecast")
            forecasts.append(forecast)
        except (ArithmeticError, ValueError):
            failures += 1
    success = len(forecasts) / thresholds.bootstrap_replicates
    if forecasts:
        low, median, high = np.quantile(forecasts, [0.025, 0.5, 0.975])
        interval_width = float(
            (high - low) / max(abs(float(median)), 1e-12)
        )
    else:
        low = median = high = math.nan
        interval_width = math.inf

    window_forecasts: list[float] = []
    for fraction in (0.70, 0.85, 1.0):
        count = max(8, int(len(train_t) * fraction))
        count = min(count, len(train_t))
        try:
            fit = _fit_trend(
                selected.family,
                train_t[:count],
                train_y[:count],
            )
            window_residual = train_y[:count] - _trend_predict(
                fit,
                train_t[:count],
            )
            process, _ = _estimate_residual_process(
                selected.residual_mode,
                window_residual,
            )
            step = max(
                1,
                int(
                    round(
                        (horizon_time - train_t[count - 1])
                        / np.median(np.diff(train_t[:count]))
                    )
                ),
            )
            forecast = float(
                _trend_predict(fit, np.asarray([horizon_time]))[0]
                + _forecast_correction(
                    last_residual=float(window_residual[-1]),
                    phi=process.effective_phi,
                    horizon_steps=np.asarray([step]),
                )[0]
            )
            if fit.optimizer_converged and math.isfinite(forecast):
                window_forecasts.append(forecast)
        except (ArithmeticError, ValueError):
            continue
    window_sensitivity = (
        (max(window_forecasts) - min(window_forecasts))
        / max(abs(float(np.median(window_forecasts))), 1e-12)
        if len(window_forecasts) == 3
        else math.inf
    )
    family_forecasts = [
        item.forecast_value
        for item in candidates
        if item.residual_mode == selected.residual_mode
    ]
    family_disagreement = (
        float(
            np.std(family_forecasts, ddof=1)
            / max(abs(float(np.mean(family_forecasts))), 1e-12)
        )
        if len(family_forecasts) > 1
        else 0.0
    )
    recovery_ablation = (
        selected.same_family_ar1_relative_improvement
        if selected.residual_mode == "ar1_residual"
        else None
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
        "registered_family_disagreement_finite": math.isfinite(
            family_disagreement
        ),
        "recovery_ablation_consistent": (
            selected.residual_mode == "trend_only"
            or (
                recovery_ablation is not None
                and recovery_ablation
                >= thresholds.minimum_ar1_validation_relative_improvement
            )
        ),
        "support_and_claim_limits_declared": True,
    }
    return HybridLevelEvidenceV56.seal(
        level="L4",
        status="PASS" if all(checks.values()) else "FAIL",
        checks=checks,
        metrics={
            "bootstrap_success_fraction": success,
            "bootstrap_failures": failures,
            "forecast_interval_low": _finite_or_none(float(low)),
            "forecast_interval_median": _finite_or_none(float(median)),
            "forecast_interval_high": _finite_or_none(float(high)),
            "forecast_interval_relative_width": _finite_or_none(interval_width),
            "window_sensitivity_relative_range": _finite_or_none(
                window_sensitivity
            ),
            "registered_family_forecast_coefficient_of_variation": (
                family_disagreement
            ),
            "selected_recovery_relative_improvement": recovery_ablation,
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
            "minimum_ar1_validation_relative_improvement": (
                thresholds.minimum_ar1_validation_relative_improvement
            ),
        },
        evidence={
            "bootstrap_seed": thresholds.bootstrap_seed,
            "bootstrap_forecast_hash": sha256_value(forecasts),
            "window_forecasts": window_forecasts,
            "registered_family_forecasts": family_forecasts,
            "support": {
                "state": "positive scalar observations",
                "trend": "registered autonomous ODE family only",
                "observation_process": "zero-intercept AR1 predictive residual only",
                "causal_identification": False,
            },
        },
    )


def _replay_environment_fingerprint() -> str:
    return sha256_value(
        {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "platform": platform.platform(),
        }
    )


def build_hybrid_ode_bundle_v56(
    *,
    snapshot: ODETimeSeriesSnapshotV52,
    thresholds: HybridODEThresholdsV56,
    replay_receipts: list[HybridReplayReceiptV56] | None = None,
    replay_authority: HybridReplayAuthorityV56 | None = None,
) -> HybridScientificBundleV56:
    snapshot.assert_sealed()
    thresholds.assert_sealed()
    train_t, train_y, validation_t, validation_y = _split(
        snapshot,
        thresholds.split_fraction,
    )
    persistence_prediction = np.repeat(train_y[-1], len(validation_y))
    persistence_rmse = float(
        np.sqrt(np.mean((persistence_prediction - validation_y) ** 2))
    )
    initial = [
        _candidate_evidence(
            family=family,
            mode="trend_only",
            train_t=train_t,
            train_y=train_y,
            validation_t=validation_t,
            validation_y=validation_y,
            persistence_rmse=persistence_rmse,
            thresholds=thresholds,
            same_family_trend_rmse=None,
        )
        for family in FAMILIES
    ]
    initial_selected = sorted(initial, key=_selection_key)[0]
    recovery_trigger_value = (
        initial_selected.absolute_validation_residual_lag1_correlation
    )
    recovery_triggered = (
        recovery_trigger_value
        > thresholds.recovery_trigger_absolute_residual_lag1_correlation
    )
    recovery: list[HybridCandidateEvidenceV56] = []
    if recovery_triggered:
        initial_by_family = {item.family: item for item in initial}
        recovery = [
            _candidate_evidence(
                family=family,
                mode="ar1_residual",
                train_t=train_t,
                train_y=train_y,
                validation_t=validation_t,
                validation_y=validation_y,
                persistence_rmse=persistence_rmse,
                thresholds=thresholds,
                same_family_trend_rmse=initial_by_family[
                    family
                ].validation_rmse,
            )
            for family in FAMILIES
        ]
    evaluated = sorted(initial + recovery, key=lambda item: item.candidate_id)
    admissible = [item for item in evaluated if item.scientifically_admissible]
    selected = (
        sorted(admissible, key=_selection_key)[0]
        if admissible
        else initial_selected
    )
    initial_ids = sorted(item.candidate_id for item in initial)
    recovery_ids = sorted(
        _candidate_id(family, "ar1_residual") for family in FAMILIES
    )
    graph_checks = {
        "all_initial_candidates_evaluated": (
            sorted(item.candidate_id for item in initial) == initial_ids
        ),
        "recovery_branch_matches_trigger": (
            (not recovery_triggered and not recovery)
            or (
                recovery_triggered
                and sorted(item.candidate_id for item in recovery)
                == recovery_ids
            )
        ),
        "no_unregistered_candidate": all(
            item.family in FAMILIES and item.residual_mode in RESIDUAL_MODES
            for item in evaluated
        ),
        "all_evaluated_candidates_guarded": all(
            bool(item.admissibility_checks) for item in evaluated
        ),
        "selected_candidate_is_evaluated": selected in evaluated,
    }
    graph = HybridCandidateGraphV56.seal(
        initial_candidate_ids=initial_ids,
        recovery_candidate_ids=recovery_ids,
        evaluated_candidate_ids=sorted(
            item.candidate_id for item in evaluated
        ),
        admissible_candidate_ids=sorted(
            item.candidate_id for item in admissible
        ),
        initial_selected_candidate_id=initial_selected.candidate_id,
        recovery_trigger_value=recovery_trigger_value,
        recovery_trigger_threshold=(
            thresholds.recovery_trigger_absolute_residual_lag1_correlation
        ),
        recovery_triggered=recovery_triggered,
        selected_candidate_id=selected.candidate_id,
        graph_checks=graph_checks,
    )

    supplied_receipts = list(replay_receipts or [])
    expected_semantic_hash = sha256_value(
        {
            "snapshot": snapshot.model_dump(mode="json"),
            "thresholds": thresholds.model_dump(mode="json"),
        }
    )
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    executable_hash = hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
    receipts_valid = bool(
        replay_authority
        and len(supplied_receipts) == 2
        and all(
            replay_authority.verify(item)
            and item.input_semantic_hash == expected_semantic_hash
            and item.source_hash == source_hash
            and item.executable_hash == executable_hash
            and item.environment_fingerprint
            == _replay_environment_fingerprint()
            for item in supplied_receipts
        )
        and [item.replay_index for item in supplied_receipts] == [1, 2]
        and len({item.process_id for item in supplied_receipts}) == 2
        and len(
            {item.deterministic_output_hash for item in supplied_receipts}
        )
        == 1
    )
    l0_checks = {
        "two_fresh_subprocess_replays_present": len(supplied_receipts) == 2,
        "replay_output_hashes_identical": (
            len(supplied_receipts) == 2
            and len(
                {
                    item.deterministic_output_hash
                    for item in supplied_receipts
                }
            )
            == 1
        ),
        "source_executable_environment_bound": receipts_valid,
        "authenticated_replay_receipts": receipts_valid,
    }
    l0 = HybridLevelEvidenceV56.seal(
        level="L0",
        status=(
            "PASS"
            if all(l0_checks.values())
            else "NOT_RUN"
            if not supplied_receipts
            else "FAIL"
        ),
        checks=l0_checks,
        metrics={"replay_count": len(supplied_receipts)},
        thresholds={"fresh_subprocess_replays": 2},
        evidence={
            "receipt_hashes": [
                item.receipt_hash for item in supplied_receipts
            ],
            "deterministic_output_hashes": [
                item.deterministic_output_hash
                for item in supplied_receipts
            ],
            "adapter_source_sha256": source_hash,
            "executable_sha256": executable_hash,
            "environment_fingerprint": _replay_environment_fingerprint(),
        },
    )

    cadence = np.diff(np.asarray(snapshot.times, dtype=float))
    median_cadence = float(np.median(cadence))
    cadence_relative_deviation = float(
        np.max(np.abs(cadence - median_cadence))
        / max(abs(median_cadence), 1e-12)
    )
    l1_checks = {
        "snapshot_sealed": snapshot.snapshot_hash == snapshot.content_hash(),
        "thresholds_sealed": thresholds.threshold_hash
        == thresholds.content_hash(),
        "units_declared": bool(snapshot.time_unit and snapshot.state_unit),
        "strictly_increasing_time": bool(np.all(cadence > 0)),
        "effectively_regular_cadence": cadence_relative_deviation <= 1e-9,
        "positive_finite_state": all(
            math.isfinite(value) and value > 0
            for value in snapshot.observations
        ),
        "slices_large_enough": (
            len(train_t) >= thresholds.minimum_points_per_slice
            and len(validation_t) >= thresholds.minimum_points_per_slice
        ),
        "candidate_graph_contract_satisfied": all(graph.graph_checks.values()),
    }
    l1 = HybridLevelEvidenceV56.seal(
        level="L1",
        status="PASS" if all(l1_checks.values()) else "FAIL",
        checks=l1_checks,
        metrics={
            "observation_count": len(snapshot.times),
            "training_count": len(train_t),
            "validation_count": len(validation_t),
            "cadence_relative_deviation": cadence_relative_deviation,
            "evaluated_candidate_count": len(evaluated),
        },
        thresholds={
            "minimum_points_per_slice": thresholds.minimum_points_per_slice,
            "maximum_cadence_relative_deviation": 1e-9,
        },
        evidence={
            "snapshot_hash": snapshot.snapshot_hash,
            "threshold_hash": thresholds.threshold_hash,
            "graph_hash": graph.graph_hash,
            "source_id": snapshot.source_id,
            "time_unit": snapshot.time_unit,
            "state_unit": snapshot.state_unit,
        },
    )
    l2 = _l2_evidence()
    l3_checks = {
        "selected_candidate_scientifically_admissible": (
            selected.scientifically_admissible
        ),
        "graph_checks_passed": all(graph.graph_checks.values()),
        "recovery_failure_fails_closed": (
            not recovery_triggered
            or any(item.scientifically_admissible for item in recovery)
        ),
    }
    l3 = HybridLevelEvidenceV56.seal(
        level="L3",
        status="PASS" if all(l3_checks.values()) else "FAIL",
        checks=l3_checks,
        metrics={
            "validation_relative_rmse": selected.validation_relative_rmse,
            "persistence_relative_improvement": (
                selected.persistence_relative_improvement
            ),
            "validation_residual_lag1_correlation": (
                selected.absolute_validation_residual_lag1_correlation
            ),
            "validation_innovation_lag1_correlation": (
                selected.absolute_validation_innovation_lag1_correlation
            ),
            "raw_ar1_phi": selected.residual_fit.raw_phi,
            "phi_window_range": selected.phi_window_range,
            "innovation_mean_shift_standardized": (
                selected.innovation_mean_shift_standardized
            ),
            "maximum_single_innovation_standardized": (
                selected.maximum_single_innovation_standardized
            ),
            "validation_interval_coverage": (
                selected.validation_interval_coverage
            ),
            "dimensionless_parameter_condition_number": (
                selected.trend_fit.dimensionless_parameter_condition_number
            ),
        },
        thresholds={
            "maximum_validation_relative_rmse": (
                thresholds.maximum_validation_relative_rmse
            ),
            "minimum_persistence_relative_improvement": (
                thresholds.minimum_persistence_relative_improvement
            ),
            "maximum_innovation_absolute_lag1_correlation": (
                thresholds.maximum_innovation_absolute_lag1_correlation
            ),
            "maximum_absolute_ar1_phi": thresholds.maximum_absolute_ar1_phi,
            "maximum_phi_window_range": thresholds.maximum_phi_window_range,
            "maximum_innovation_mean_shift_standardized": (
                thresholds.maximum_innovation_mean_shift_standardized
            ),
            "maximum_single_innovation_standardized": (
                thresholds.maximum_single_innovation_standardized
            ),
            "minimum_validation_interval_coverage": (
                thresholds.minimum_validation_interval_coverage
            ),
            "maximum_dimensionless_parameter_condition_number": (
                thresholds.maximum_dimensionless_parameter_condition_number
            ),
        },
        evidence={
            "selected_candidate_id": selected.candidate_id,
            "selected_candidate_evidence_hash": selected.evidence_hash,
            "selected_admissibility_checks": selected.admissibility_checks,
            "recovery_triggered": recovery_triggered,
            "admissible_candidate_ids": graph.admissible_candidate_ids,
        },
    )
    l4 = _l4_evidence(
        selected=selected,
        candidates=evaluated,
        train_t=train_t,
        train_y=train_y,
        validation_t=validation_t,
        thresholds=thresholds,
    )
    levels = [l0, l1, l2, l3, l4]
    return HybridScientificBundleV56.seal(
        task_id=snapshot.task_id,
        snapshot_hash=snapshot.snapshot_hash,
        threshold_hash=thresholds.threshold_hash,
        candidate_registry_hash=sha256_value(
            {
                "families": list(FAMILIES),
                "residual_modes": list(RESIDUAL_MODES),
                "graph": (
                    "trend_only_then_triggered_ar1_then_mechanism_guard"
                ),
            }
        ),
        graph=graph,
        selected_candidate_id=selected.candidate_id,
        candidates=evaluated,
        levels=levels,
        replay_receipt_hashes=[
            str(item.receipt_hash) for item in supplied_receipts
        ],
        scientific_acceptance=all(
            item.status == "PASS" for item in levels
        ),
        fixture_only=snapshot.fixture_only,
    )


def deterministic_hybrid_ode_hash_v56(
    *,
    snapshot: ODETimeSeriesSnapshotV52,
    thresholds: HybridODEThresholdsV56,
) -> str:
    bundle = build_hybrid_ode_bundle_v56(
        snapshot=snapshot,
        thresholds=thresholds,
    )
    return sha256_value(
        {
            "schema_version": "5.6-replay",
            "snapshot_hash": bundle.snapshot_hash,
            "threshold_hash": bundle.threshold_hash,
            "candidate_registry_hash": bundle.candidate_registry_hash,
            "graph_hash": bundle.graph.graph_hash,
            "selected_candidate_id": bundle.selected_candidate_id,
            "candidate_evidence_hashes": [
                item.evidence_hash for item in bundle.candidates
            ],
            "levels_l1_l4": [
                item.evidence_hash
                for item in bundle.levels
                if item.level != "L0"
            ],
            "source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "environment_fingerprint": _replay_environment_fingerprint(),
        }
    )


def run_authenticated_hybrid_replays_v56(
    replay_input_path: str | Path,
    *,
    authority: HybridReplayAuthorityV56,
    count: Literal[2] = 2,
    timeout_seconds: int = 600,
) -> list[HybridReplayReceiptV56]:
    input_path = Path(replay_input_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    input_bytes = input_path.read_bytes()
    parsed = json.loads(input_bytes)
    snapshot = ODETimeSeriesSnapshotV52.model_validate(parsed["snapshot"])
    thresholds = HybridODEThresholdsV56.model_validate(parsed["thresholds"])
    semantic_hash = sha256_value(
        {
            "snapshot": snapshot.model_dump(mode="json"),
            "thresholds": thresholds.model_dump(mode="json"),
        }
    )
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "TEMP": os.environ.get("TEMP", ""),
        "TMP": os.environ.get("TMP", ""),
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
    }
    command = [
        sys.executable,
        "-m",
        "fma.v5_6.hybrid_ode",
        "replay",
        str(input_path),
    ]
    receipts: list[HybridReplayReceiptV56] = []
    for index in range(1, count + 1):
        with tempfile.TemporaryDirectory(
            prefix="fma-v56-hybrid-replay-"
        ) as temporary:
            process = subprocess.Popen(
                command,
                cwd=temporary,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                process.kill()
                stdout, stderr = process.communicate()
                raise RuntimeError(
                    "V5.6 hybrid replay timed out; stderr_sha256="
                    + hashlib.sha256(stderr.encode()).hexdigest()
                ) from exc
        if process.returncode != 0:
            raise RuntimeError(
                "V5.6 hybrid replay failed; stderr_sha256="
                + hashlib.sha256(stderr.encode()).hexdigest()
            )
        deterministic_hash = str(
            json.loads(stdout)["deterministic_output_hash"]
        )
        receipts.append(
            authority.issue(
                replay_id=f"{snapshot.task_id}-hybrid-replay-{index}",
                replay_index=index,
                process_id=process.pid,
                input_bytes_hash=hashlib.sha256(input_bytes).hexdigest(),
                input_semantic_hash=semantic_hash,
                command_hash=sha256_value(command),
                exit_code=0,
                stdout_hash=hashlib.sha256(stdout.encode()).hexdigest(),
                stderr_hash=hashlib.sha256(stderr.encode()).hexdigest(),
                deterministic_output_hash=deterministic_hash,
                source_hash=hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                executable_hash=hashlib.sha256(
                    Path(sys.executable).read_bytes()
                ).hexdigest(),
                environment_fingerprint=_replay_environment_fingerprint(),
            )
        )
    return receipts


def _read_manifest_file(
    context: AdapterContextV50,
    relative_path: str,
) -> bytes:
    root = context.workspace_root.resolve()
    path = (root / relative_path).resolve()
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
        raise ValueError("hybrid ODE bundle is absent from frozen manifest")
    payload = path.read_bytes()
    if (
        len(payload) != binding.size_bytes
        or hashlib.sha256(payload).hexdigest() != binding.sha256
    ):
        raise ValueError("hybrid ODE bundle differs from frozen manifest")
    return payload


class HybridODELevelAdapterV56:
    adapter_id = "scalar_hybrid_ode_scientific_adapter"
    adapter_version = "5.6"

    def __init__(self, level: LevelV56) -> None:
        self.level = level
        self.check_id = f"scalar_hybrid_ode_{level.lower()}"

    def run(self, context: AdapterContextV50) -> AdapterOutcomeV50:
        bundle = HybridScientificBundleV56.model_validate_json(
            _read_manifest_file(
                context,
                "results/hybrid_ode_scientific_bundle.json",
            )
        )
        evidence = next(
            item for item in bundle.levels if item.level == self.level
        )
        payload: dict[str, Any] = {
            "bundle_hash": bundle.bundle_hash,
            "graph_hash": bundle.graph.graph_hash,
            "level_evidence": evidence.model_dump(mode="json"),
            "fixture_only": bundle.fixture_only,
            "causal_mechanism_identified": False,
            "scientific_qualification_granted": False,
            "real_world_action_authorized": False,
        }
        if self.level == "L0":
            code_manifest = CodeManifestV50.model_validate_json(
                _read_manifest_file(
                    context,
                    "results/code_manifest.json",
                )
            )
            payload["computation_artifact_sha256"] = (
                code_manifest.replay_receipt_hash
            )
        return AdapterOutcomeV50(
            status="PASS" if evidence.status == "PASS" else "FAIL",
            reason_code=(
                "scalar_hybrid_ode_level_passed"
                if evidence.status == "PASS"
                else f"scalar_hybrid_ode_level_{evidence.status.lower()}"
            ),
            metrics=evidence.metrics,
            thresholds=evidence.thresholds,
            evidence_payloads=[payload],
        )


def register_hybrid_ode_adapters_v56(registry: Any) -> None:
    for level in ("L0", "L1", "L2", "L3", "L4"):
        registry.register(HybridODELevelAdapterV56(level))


def _main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] != "replay":
        raise SystemExit(
            "usage: python -m fma.v5_6.hybrid_ode replay INPUT"
        )
    payload = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    snapshot = ODETimeSeriesSnapshotV52.model_validate(payload["snapshot"])
    thresholds = HybridODEThresholdsV56.model_validate(payload["thresholds"])
    print(
        canonical_json(
            {
                "deterministic_output_hash": (
                    deterministic_hybrid_ode_hash_v56(
                        snapshot=snapshot,
                        thresholds=thresholds,
                    )
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "DimensionlessTrendFitV56",
    "HybridCandidateEvidenceV56",
    "HybridCandidateGraphV56",
    "HybridODELevelAdapterV56",
    "HybridODEThresholdsV56",
    "HybridReplayAuthorityV56",
    "HybridReplayReceiptV56",
    "HybridScientificBundleV56",
    "build_hybrid_ode_bundle_v56",
    "deterministic_hybrid_ode_hash_v56",
    "register_hybrid_ode_adapters_v56",
    "run_authenticated_hybrid_replays_v56",
]
