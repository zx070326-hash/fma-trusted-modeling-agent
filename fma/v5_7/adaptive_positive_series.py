"""V5.7 graph recovery from autonomous ODEs to stochastic log growth."""

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

from fma.hashing import canonical_json, sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5.check_registry import AdapterContextV50, AdapterOutcomeV50
from fma.v5.workspace_schemas import CodeManifestV50
from fma.v5_2.ode_system import ODETimeSeriesSnapshotV52
from fma.v5_6.hybrid_ode import (
    HybridODEThresholdsV56,
    HybridScientificBundleV56,
    build_hybrid_ode_bundle_v56,
)


FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFinite = Annotated[float, Field(ge=0, allow_inf_nan=False)]
GrowthModeV57 = Literal["log_random_walk_drift", "log_growth_ar1"]
AdaptiveBranchV57 = Literal["hybrid_ode", "log_growth", "unresolved"]
LevelV57 = Literal["L0", "L1", "L2", "L3", "L4"]
LevelStatusV57 = Literal["PASS", "FAIL", "NOT_RUN", "HUMAN"]

GROWTH_MODES: tuple[GrowthModeV57, ...] = (
    "log_random_walk_drift",
    "log_growth_ar1",
)


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _lag1(values: np.ndarray) -> float:
    if len(values) < 3:
        return 0.0
    left = values[:-1]
    right = values[1:]
    if float(np.std(left)) <= 1e-15 or float(np.std(right)) <= 1e-15:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else 0.0


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


class AdaptiveThresholdsV57(StrictModel):
    schema_version: Literal["5.7"] = "5.7"
    split_fraction: Annotated[float, Field(gt=0.5, lt=0.9)]
    minimum_points_per_slice: Annotated[int, Field(ge=8)]
    maximum_validation_relative_rmse: NonNegativeFinite
    minimum_persistence_relative_improvement: FiniteNumber
    maximum_innovation_absolute_lag1_correlation: Annotated[
        float,
        Field(ge=0, le=1, allow_inf_nan=False),
    ]
    maximum_absolute_growth_ar1_phi: Annotated[
        float,
        Field(gt=0, lt=1, allow_inf_nan=False),
    ]
    minimum_growth_ar1_validation_relative_improvement: NonNegativeFinite
    maximum_growth_phi_window_range: NonNegativeFinite
    maximum_growth_drift_window_range_standardized: NonNegativeFinite
    maximum_innovation_mean_shift_standardized: NonNegativeFinite
    maximum_single_innovation_standardized: NonNegativeFinite
    minimum_validation_interval_coverage: Annotated[
        float,
        Field(ge=0, le=1, allow_inf_nan=False),
    ]
    maximum_absolute_mean_log_growth: NonNegativeFinite
    selection_complexity_penalty_per_parameter: NonNegativeFinite
    bootstrap_replicates: Annotated[int, Field(ge=20)]
    bootstrap_seed: Annotated[int, Field(ge=0, le=4_294_967_295)]
    minimum_bootstrap_success_fraction: Annotated[
        float,
        Field(ge=0, le=1, allow_inf_nan=False),
    ]
    maximum_forecast_interval_relative_width: NonNegativeFinite
    maximum_window_sensitivity_relative_range: NonNegativeFinite
    threshold_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_thresholds(self) -> "AdaptiveThresholdsV57":
        if self.threshold_hash and self.threshold_hash != self.content_hash():
            raise ValueError("V5.7 threshold hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "threshold_hash")

    def assert_sealed(self) -> None:
        if not self.threshold_hash or self.threshold_hash != self.content_hash():
            raise ValueError("V5.7 thresholds are not sealed")

    @classmethod
    def seal(cls, **data: object) -> "AdaptiveThresholdsV57":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"threshold_hash"})
        payload["threshold_hash"] = draft.content_hash()
        return cls(**payload)


class GrowthProcessFitV57(StrictModel):
    schema_version: Literal["5.7"] = "5.7"
    mode: GrowthModeV57
    mean_log_growth: FiniteNumber
    raw_phi: FiniteNumber
    effective_phi: FiniteNumber
    training_innovation_scale: NonNegativeFinite
    training_growth_hash: Sha256
    training_innovation_hash: Sha256
    fit_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_fit(self) -> "GrowthProcessFitV57":
        if self.mode == "log_random_walk_drift" and (
            self.raw_phi != 0 or self.effective_phi != 0
        ):
            raise ValueError("log drift candidate must have phi=0")
        if not -0.999 <= self.effective_phi <= 0.999:
            raise ValueError("effective growth phi is outside safe recursion")
        if self.fit_hash and self.fit_hash != self.content_hash():
            raise ValueError("growth-process fit hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "fit_hash")

    @classmethod
    def seal(cls, **data: object) -> "GrowthProcessFitV57":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"fit_hash"})
        payload["fit_hash"] = draft.content_hash()
        return cls(**payload)


class GrowthCandidateEvidenceV57(StrictModel):
    schema_version: Literal["5.7"] = "5.7"
    candidate_id: Identifier
    mode: GrowthModeV57
    process_fit: GrowthProcessFitV57
    parameter_count: Annotated[int, Field(ge=1, le=2)]
    validation_rmse: NonNegativeFinite
    validation_relative_rmse: NonNegativeFinite
    validation_score: FiniteNumber
    persistence_relative_improvement: FiniteNumber
    same_family_ar1_relative_improvement: FiniteNumber | None
    absolute_validation_innovation_lag1_correlation: NonNegativeFinite
    phi_window_range: NonNegativeFinite
    drift_window_range_standardized: NonNegativeFinite
    innovation_mean_shift_standardized: NonNegativeFinite
    maximum_single_innovation_standardized: NonNegativeFinite
    validation_interval_coverage: Annotated[
        float,
        Field(ge=0, le=1, allow_inf_nan=False),
    ]
    forecast_value: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    validation_prediction_hash: Sha256
    validation_innovation_hash: Sha256
    admissibility_checks: dict[Identifier, bool]
    scientifically_admissible: bool
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_candidate(self) -> "GrowthCandidateEvidenceV57":
        if self.candidate_id != self.mode:
            raise ValueError("growth candidate ID differs from mode")
        if self.process_fit.mode != self.mode:
            raise ValueError("growth candidate fit mode differs")
        expected = bool(self.admissibility_checks) and all(
            self.admissibility_checks.values()
        )
        if self.scientifically_admissible != expected:
            raise ValueError("growth admissibility differs from checks")
        if self.evidence_hash and self.evidence_hash != self.content_hash():
            raise ValueError("growth candidate evidence hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evidence_hash")

    @classmethod
    def seal(cls, **data: object) -> "GrowthCandidateEvidenceV57":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"evidence_hash"})
        payload["evidence_hash"] = draft.content_hash()
        return cls(**payload)


class AdaptiveCandidateGraphV57(StrictModel):
    schema_version: Literal["5.7-adaptive-candidate-graph"] = (
        "5.7-adaptive-candidate-graph"
    )
    primary_bundle_hash: Sha256
    primary_selected_candidate_id: Identifier
    primary_level_statuses: dict[
        Literal["L1", "L2", "L3", "L4"],
        LevelStatusV57,
    ]
    recovery_triggered: bool
    recovery_reason_codes: list[Identifier]
    recovery_candidate_ids: list[Identifier]
    admissible_recovery_candidate_ids: list[Identifier]
    selected_branch: AdaptiveBranchV57
    selected_model_id: Identifier
    graph_checks: dict[Identifier, bool]
    graph_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_graph(self) -> "AdaptiveCandidateGraphV57":
        for values in (
            self.recovery_reason_codes,
            self.recovery_candidate_ids,
            self.admissible_recovery_candidate_ids,
        ):
            if values != sorted(set(values)):
                raise ValueError("V5.7 graph lists must be sorted and unique")
        expected_trigger = not all(
            self.primary_level_statuses.get(level) == "PASS"
            for level in ("L1", "L2", "L3", "L4")
        )
        if self.recovery_triggered != expected_trigger:
            raise ValueError("V5.7 recovery trigger differs")
        if not self.recovery_triggered and self.recovery_candidate_ids:
            raise ValueError("V5.7 recovery candidates exist without trigger")
        if self.selected_branch == "hybrid_ode" and self.recovery_triggered:
            raise ValueError("failed primary branch cannot remain selected")
        if (
            self.selected_branch == "log_growth"
            and self.selected_model_id
            not in self.admissible_recovery_candidate_ids
        ):
            raise ValueError("selected growth model is not admissible")
        if (
            self.selected_branch == "unresolved"
            and self.admissible_recovery_candidate_ids
        ):
            raise ValueError("unresolved graph contains admissible recovery")
        if self.graph_hash and self.graph_hash != self.content_hash():
            raise ValueError("V5.7 graph hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "graph_hash")

    @classmethod
    def seal(cls, **data: object) -> "AdaptiveCandidateGraphV57":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"graph_hash"})
        payload["graph_hash"] = draft.content_hash()
        return cls(**payload)


class AdaptiveLevelEvidenceV57(StrictModel):
    schema_version: Literal["5.7"] = "5.7"
    level: LevelV57
    status: LevelStatusV57
    checks: dict[Identifier, bool]
    metrics: dict[Identifier, FiniteNumber | int | None]
    thresholds: dict[Identifier, FiniteNumber | int]
    evidence: dict[str, Any]
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_level(self) -> "AdaptiveLevelEvidenceV57":
        if self.status == "PASS" and (
            not self.checks or not all(self.checks.values())
        ):
            raise ValueError("passing V5.7 level contains failed checks")
        if self.evidence_hash and self.evidence_hash != self.content_hash():
            raise ValueError("V5.7 level evidence hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evidence_hash")

    @classmethod
    def seal(cls, **data: object) -> "AdaptiveLevelEvidenceV57":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"evidence_hash"})
        payload["evidence_hash"] = draft.content_hash()
        return cls(**payload)


class AdaptivePositiveSeriesBundleV57(StrictModel):
    schema_version: Literal["5.7"] = "5.7"
    task_id: Identifier
    domain: Literal["adaptive_positive_scalar_series"] = (
        "adaptive_positive_scalar_series"
    )
    snapshot_hash: Sha256
    primary_threshold_hash: Sha256
    adaptive_threshold_hash: Sha256
    primary_bundle: HybridScientificBundleV56
    growth_candidates: list[GrowthCandidateEvidenceV57]
    graph: AdaptiveCandidateGraphV57
    levels: list[AdaptiveLevelEvidenceV57]
    replay_receipt_hashes: list[Sha256]
    scientific_acceptance: bool
    fixture_only: bool
    causal_mechanism_identified: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    bundle_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bundle(self) -> "AdaptivePositiveSeriesBundleV57":
        ids = [item.candidate_id for item in self.growth_candidates]
        if ids != sorted(set(ids)):
            raise ValueError("V5.7 growth candidates must be sorted and unique")
        if [item.level for item in self.levels] != [
            "L0",
            "L1",
            "L2",
            "L3",
            "L4",
        ]:
            raise ValueError("V5.7 bundle must contain ordered L0-L4")
        expected = all(item.status == "PASS" for item in self.levels)
        if self.scientific_acceptance != expected:
            raise ValueError("V5.7 acceptance differs from levels")
        if self.replay_receipt_hashes and len(self.replay_receipt_hashes) != 2:
            raise ValueError("V5.7 replay receipts must be absent or a pair")
        if self.bundle_hash and self.bundle_hash != self.content_hash():
            raise ValueError("V5.7 bundle hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "bundle_hash")

    @classmethod
    def seal(cls, **data: object) -> "AdaptivePositiveSeriesBundleV57":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"bundle_hash"})
        payload["bundle_hash"] = draft.content_hash()
        return cls(**payload)


class AdaptiveReplayReceiptV57(StrictModel):
    schema_version: Literal["5.7-replay-receipt"] = "5.7-replay-receipt"
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
    def validate_receipt(self) -> "AdaptiveReplayReceiptV57":
        if self.receipt_hash and (
            not self.authority_auth_tag
            or self.receipt_hash != self.content_hash()
        ):
            raise ValueError("V5.7 replay receipt differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "authority_auth_tag", "receipt_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")


class AdaptiveReplayAuthorityV57:
    def __init__(self, *, key_id: str, secret: bytes) -> None:
        if len(secret) < 32:
            raise ValueError("V5.7 replay authority needs 32 bytes")
        self.key_id = key_id
        self._secret = bytes(secret)

    def _mac(self, unsigned_hash: str) -> str:
        return hmac.new(
            self._secret,
            f"adaptive_positive_series_v57:{unsigned_hash}".encode(),
            hashlib.sha256,
        ).hexdigest()

    def issue(self, **data: object) -> AdaptiveReplayReceiptV57:
        data["authority_key_id"] = self.key_id
        unsigned = AdaptiveReplayReceiptV57(**data)
        payload = unsigned.model_dump(mode="json")
        payload["authority_auth_tag"] = self._mac(unsigned.unsigned_hash())
        tagged = AdaptiveReplayReceiptV57(**payload)
        final = tagged.model_dump(mode="json")
        final["receipt_hash"] = tagged.content_hash()
        return AdaptiveReplayReceiptV57(**final)

    def verify(self, receipt: AdaptiveReplayReceiptV57) -> bool:
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


def _estimate_growth_process(
    mode: GrowthModeV57,
    growths: np.ndarray,
) -> tuple[GrowthProcessFitV57, np.ndarray]:
    if len(growths) < 2:
        raise ValueError("growth process needs at least two increments")
    mean = float(np.mean(growths))
    if mode == "log_random_walk_drift":
        raw_phi = effective_phi = 0.0
        innovations = growths - mean
    else:
        centered_left = growths[:-1] - mean
        centered_right = growths[1:] - mean
        denominator = float(np.dot(centered_left, centered_left))
        raw_phi = (
            float(np.dot(centered_left, centered_right) / denominator)
            if denominator > 1e-18
            else 0.0
        )
        effective_phi = float(np.clip(raw_phi, -0.999, 0.999))
        innovations = centered_right - effective_phi * centered_left
    scale = float(np.std(innovations, ddof=1)) if len(innovations) > 1 else 0.0
    return (
        GrowthProcessFitV57.seal(
            mode=mode,
            mean_log_growth=mean,
            raw_phi=raw_phi,
            effective_phi=effective_phi,
            training_innovation_scale=max(scale, 0.0),
            training_growth_hash=sha256_value(growths.tolist()),
            training_innovation_hash=sha256_value(innovations.tolist()),
        ),
        innovations,
    )


def _growth_window_stability(
    growths: np.ndarray,
) -> tuple[float, float, list[float], list[float]]:
    means: list[float] = []
    phis: list[float] = []
    for fraction in (0.70, 0.85, 1.0):
        count = max(7, int(len(growths) * fraction))
        count = min(count, len(growths))
        subset = growths[:count]
        means.append(float(np.mean(subset)))
        fit, _ = _estimate_growth_process("log_growth_ar1", subset)
        phis.append(fit.raw_phi)
    return (
        float(max(means) - min(means)),
        float(max(phis) - min(phis)),
        means,
        phis,
    )


def _growth_candidate(
    *,
    mode: GrowthModeV57,
    train_y: np.ndarray,
    validation_y: np.ndarray,
    persistence_rmse: float,
    thresholds: AdaptiveThresholdsV57,
    drift_rmse: float | None,
) -> GrowthCandidateEvidenceV57:
    train_growths = np.diff(np.log(train_y))
    fit, training_innovations = _estimate_growth_process(mode, train_growths)
    previous_level = float(train_y[-1])
    previous_growth = float(train_growths[-1])
    predictions: list[float] = []
    validation_innovations: list[float] = []
    coverages: list[bool] = []
    scale = max(fit.training_innovation_scale, 1e-12)
    for actual_level in validation_y:
        predicted_growth = float(
            fit.mean_log_growth
            + fit.effective_phi
            * (previous_growth - fit.mean_log_growth)
        )
        predicted_level = float(previous_level * math.exp(predicted_growth))
        actual_growth = float(math.log(actual_level / previous_level))
        innovation = actual_growth - predicted_growth
        predictions.append(predicted_level)
        validation_innovations.append(innovation)
        low = predicted_level * math.exp(-1.96 * scale)
        high = predicted_level * math.exp(1.96 * scale)
        coverages.append(low <= actual_level <= high)
        previous_growth = actual_growth
        previous_level = float(actual_level)
    prediction_array = np.asarray(predictions, dtype=float)
    innovation_array = np.asarray(validation_innovations, dtype=float)
    rmse = float(np.sqrt(np.mean((prediction_array - validation_y) ** 2)))
    relative_rmse = rmse / max(float(np.mean(validation_y)), 1e-12)
    persistence_improvement = 1.0 - rmse / max(persistence_rmse, 1e-12)
    same_family_improvement = (
        1.0 - rmse / max(drift_rmse, 1e-12)
        if drift_rmse is not None
        else None
    )
    innovation_lag = abs(_lag1(innovation_array))
    training_mean = (
        float(np.mean(training_innovations))
        if len(training_innovations)
        else 0.0
    )
    validation_mean_shift = abs(
        float(np.mean(innovation_array)) - training_mean
    ) / scale
    training_block_shift = max(
        (
            abs(float(np.mean(block)) - training_mean) / scale
            for block in np.array_split(training_innovations, 3)
            if len(block)
        ),
        default=0.0,
    )
    mean_shift = max(validation_mean_shift, training_block_shift)
    max_single = max(
        float(np.max(np.abs(innovation_array - training_mean)) / scale),
        (
            float(
                np.max(np.abs(training_innovations - training_mean))
                / scale
            )
            if len(training_innovations)
            else 0.0
        ),
    )
    drift_range, phi_range, _means, _phis = _growth_window_stability(
        train_growths
    )
    drift_range_standardized = drift_range / scale
    next_growth = float(
        fit.mean_log_growth
        + fit.effective_phi
        * (previous_growth - fit.mean_log_growth)
    )
    forecast = float(previous_level * math.exp(next_growth))
    parameter_count = 1 if mode == "log_random_walk_drift" else 2
    checks = {
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
        "growth_ar1_stationary_interior": (
            mode == "log_random_walk_drift"
            or abs(fit.raw_phi)
            <= thresholds.maximum_absolute_growth_ar1_phi
        ),
        "growth_ar1_improvement_material": (
            mode == "log_random_walk_drift"
            or (
                same_family_improvement is not None
                and same_family_improvement
                >= thresholds.minimum_growth_ar1_validation_relative_improvement
            )
        ),
        "growth_phi_window_stable": (
            mode == "log_random_walk_drift"
            or phi_range <= thresholds.maximum_growth_phi_window_range
        ),
        "growth_drift_window_stable": (
            drift_range_standardized
            <= thresholds.maximum_growth_drift_window_range_standardized
        ),
        "innovation_mean_shift_bounded": (
            mean_shift
            <= thresholds.maximum_innovation_mean_shift_standardized
        ),
        "single_innovation_bounded": (
            max_single <= thresholds.maximum_single_innovation_standardized
        ),
        "validation_interval_coverage": (
            float(np.mean(coverages))
            >= thresholds.minimum_validation_interval_coverage
        ),
        "mean_log_growth_plausible": (
            abs(fit.mean_log_growth)
            <= thresholds.maximum_absolute_mean_log_growth
        ),
    }
    return GrowthCandidateEvidenceV57.seal(
        candidate_id=mode,
        mode=mode,
        process_fit=fit,
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
        absolute_validation_innovation_lag1_correlation=innovation_lag,
        phi_window_range=0.0 if mode == "log_random_walk_drift" else phi_range,
        drift_window_range_standardized=drift_range_standardized,
        innovation_mean_shift_standardized=mean_shift,
        maximum_single_innovation_standardized=max_single,
        validation_interval_coverage=float(np.mean(coverages)),
        forecast_value=forecast,
        validation_prediction_hash=sha256_value(predictions),
        validation_innovation_hash=sha256_value(validation_innovations),
        admissibility_checks=checks,
        scientifically_admissible=all(checks.values()),
    )


def _selection_key(
    candidate: GrowthCandidateEvidenceV57,
) -> tuple[float, int, str]:
    return (
        -candidate.validation_score,
        candidate.parameter_count,
        candidate.candidate_id,
    )


def _l2_evidence() -> AdaptiveLevelEvidenceV57:
    values = np.asarray([4.0, 4.2, 4.1, 4.5], dtype=float)
    growth = np.diff(np.log(values))
    scaled_growth = np.diff(np.log(values * 17.0))
    scale_error = float(np.max(np.abs(growth - scaled_growth)))
    previous = 0.2
    phi = 0.8
    mean = 0.03
    recursive = []
    for _ in range(8):
        previous = mean + phi * (previous - mean)
        recursive.append(previous)
    closed = [
        mean + (phi**step) * (0.2 - mean)
        for step in range(1, 9)
    ]
    recursion_error = float(
        np.max(np.abs(np.asarray(recursive) - np.asarray(closed)))
    )
    phi_zero = mean + 0.0 * (0.2 - mean)
    roundtrip = float(
        np.max(np.abs(np.exp(np.log(values)) - values))
    )
    checks = {
        "positive_scale_invariant_log_growth": scale_error <= 1e-12,
        "recursive_closed_form_growth_agreement": recursion_error <= 1e-12,
        "phi_zero_reduces_to_drift": abs(phi_zero - mean) <= 1e-12,
        "stationary_growth_mean_reversion": abs(recursive[-1] - mean)
        < abs(0.2 - mean),
        "log_level_roundtrip": roundtrip <= 1e-12,
    }
    return AdaptiveLevelEvidenceV57.seal(
        level="L2",
        status="PASS" if all(checks.values()) else "FAIL",
        checks=checks,
        metrics={
            "scale_invariance_max_error": scale_error,
            "recursive_closed_form_max_error": recursion_error,
            "phi_zero_drift_error": abs(phi_zero - mean),
            "log_level_roundtrip_max_error": roundtrip,
        },
        thresholds={
            "identity_max_error": 1e-12,
        },
        evidence={
            "representation": "positive level with stochastic log increments",
            "causal_identification": False,
        },
    )


def _growth_l4(
    *,
    selected: GrowthCandidateEvidenceV57,
    train_y: np.ndarray,
    primary_selected_relative_rmse: float,
    thresholds: AdaptiveThresholdsV57,
) -> AdaptiveLevelEvidenceV57:
    growths = np.diff(np.log(train_y))
    fit, innovations = _estimate_growth_process(selected.mode, growths)
    rng = np.random.default_rng(thresholds.bootstrap_seed)
    forecasts: list[float] = []
    last_level = float(train_y[-1])
    last_growth = float(growths[-1])
    for _ in range(thresholds.bootstrap_replicates):
        try:
            sampled = float(rng.choice(innovations)) if len(innovations) else 0.0
            predicted_growth = float(
                fit.mean_log_growth
                + fit.effective_phi
                * (last_growth - fit.mean_log_growth)
                + sampled
            )
            forecast = float(last_level * math.exp(predicted_growth))
            if not math.isfinite(forecast) or forecast <= 0:
                raise ValueError("invalid V5.7 bootstrap forecast")
            forecasts.append(forecast)
        except (ArithmeticError, ValueError):
            continue
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
        count = max(9, int(len(train_y) * fraction))
        count = min(count, len(train_y))
        try:
            subset = train_y[:count]
            subset_growth = np.diff(np.log(subset))
            window_fit, _ = _estimate_growth_process(
                selected.mode,
                subset_growth,
            )
            current_level = float(subset[-1])
            current_growth = float(subset_growth[-1])
            horizon_steps = len(train_y) - count + 1
            for _ in range(horizon_steps):
                current_growth = float(
                    window_fit.mean_log_growth
                    + window_fit.effective_phi
                    * (
                        current_growth
                        - window_fit.mean_log_growth
                    )
                )
                current_level *= math.exp(current_growth)
            forecast = current_level
            if math.isfinite(forecast) and forecast > 0:
                window_forecasts.append(forecast)
        except (ArithmeticError, ValueError):
            continue
    sensitivity = (
        (max(window_forecasts) - min(window_forecasts))
        / max(abs(float(np.median(window_forecasts))), 1e-12)
        if len(window_forecasts) == 3
        else math.inf
    )
    recovery_improvement = (
        1.0
        - selected.validation_relative_rmse
        / max(primary_selected_relative_rmse, 1e-12)
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
            sensitivity
            <= thresholds.maximum_window_sensitivity_relative_range
        ),
        "representation_recovery_improves_primary": recovery_improvement > 0,
        "support_and_claim_limits_declared": True,
    }
    return AdaptiveLevelEvidenceV57.seal(
        level="L4",
        status="PASS" if all(checks.values()) else "FAIL",
        checks=checks,
        metrics={
            "bootstrap_success_fraction": success,
            "forecast_interval_low": _finite_or_none(float(low)),
            "forecast_interval_median": _finite_or_none(float(median)),
            "forecast_interval_high": _finite_or_none(float(high)),
            "forecast_interval_relative_width": _finite_or_none(interval_width),
            "window_sensitivity_relative_range": _finite_or_none(sensitivity),
            "representation_recovery_relative_improvement": (
                recovery_improvement
            ),
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
        },
        evidence={
            "bootstrap_seed": thresholds.bootstrap_seed,
            "bootstrap_forecast_hash": sha256_value(forecasts),
            "window_forecasts": window_forecasts,
            "selected_growth_candidate": selected.candidate_id,
            "causal_identification": False,
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


def build_adaptive_positive_series_bundle_v57(
    *,
    snapshot: ODETimeSeriesSnapshotV52,
    primary_thresholds: HybridODEThresholdsV56,
    adaptive_thresholds: AdaptiveThresholdsV57,
    replay_receipts: list[AdaptiveReplayReceiptV57] | None = None,
    replay_authority: AdaptiveReplayAuthorityV57 | None = None,
) -> AdaptivePositiveSeriesBundleV57:
    snapshot.assert_sealed()
    primary_thresholds.assert_sealed()
    adaptive_thresholds.assert_sealed()
    primary = build_hybrid_ode_bundle_v56(
        snapshot=snapshot,
        thresholds=primary_thresholds,
    )
    primary_statuses = {
        item.level: item.status
        for item in primary.levels
        if item.level != "L0"
    }
    recovery_triggered = not all(
        primary_statuses.get(level) == "PASS"
        for level in ("L1", "L2", "L3", "L4")
    )
    reason_codes = sorted(
        f"primary_{level.lower()}_{str(primary_statuses.get(level)).lower()}"
        for level in ("L1", "L2", "L3", "L4")
        if primary_statuses.get(level) != "PASS"
    )
    count = len(snapshot.times)
    split = min(
        max(int(count * adaptive_thresholds.split_fraction), 2),
        count - 2,
    )
    values = np.asarray(snapshot.observations, dtype=float)
    train_y = values[:split]
    validation_y = values[split:]
    persistence_predictions = np.concatenate(
        ([train_y[-1]], validation_y[:-1])
    )
    persistence_rmse = float(
        np.sqrt(
            np.mean((persistence_predictions - validation_y) ** 2)
        )
    )
    growth_candidates: list[GrowthCandidateEvidenceV57] = []
    if recovery_triggered:
        drift = _growth_candidate(
            mode="log_random_walk_drift",
            train_y=train_y,
            validation_y=validation_y,
            persistence_rmse=persistence_rmse,
            thresholds=adaptive_thresholds,
            drift_rmse=None,
        )
        growth_ar1 = _growth_candidate(
            mode="log_growth_ar1",
            train_y=train_y,
            validation_y=validation_y,
            persistence_rmse=persistence_rmse,
            thresholds=adaptive_thresholds,
            drift_rmse=drift.validation_rmse,
        )
        growth_candidates = sorted(
            [drift, growth_ar1],
            key=lambda item: item.candidate_id,
        )
    admissible = [
        item for item in growth_candidates if item.scientifically_admissible
    ]
    if not recovery_triggered:
        branch: AdaptiveBranchV57 = "hybrid_ode"
        selected_model_id = primary.selected_candidate_id
        selected_growth = None
    elif admissible:
        branch = "log_growth"
        selected_growth = sorted(admissible, key=_selection_key)[0]
        selected_model_id = selected_growth.candidate_id
    else:
        branch = "unresolved"
        selected_growth = (
            sorted(growth_candidates, key=_selection_key)[0]
            if growth_candidates
            else None
        )
        selected_model_id = (
            selected_growth.candidate_id
            if selected_growth
            else primary.selected_candidate_id
        )
    graph_checks = {
        "primary_branch_complete": [item.level for item in primary.levels]
        == ["L0", "L1", "L2", "L3", "L4"],
        "recovery_branch_matches_primary_status": (
            (not recovery_triggered and not growth_candidates)
            or (
                recovery_triggered
                and [item.candidate_id for item in growth_candidates]
                == sorted(GROWTH_MODES)
            )
        ),
        "no_unregistered_growth_candidate": all(
            item.mode in GROWTH_MODES for item in growth_candidates
        ),
        "all_recovery_candidates_guarded": all(
            bool(item.admissibility_checks) for item in growth_candidates
        ),
        "selection_matches_admissibility": (
            (branch == "hybrid_ode" and not recovery_triggered)
            or (branch == "log_growth" and bool(admissible))
            or (branch == "unresolved" and not admissible)
        ),
    }
    graph = AdaptiveCandidateGraphV57.seal(
        primary_bundle_hash=primary.bundle_hash,
        primary_selected_candidate_id=primary.selected_candidate_id,
        primary_level_statuses=primary_statuses,
        recovery_triggered=recovery_triggered,
        recovery_reason_codes=reason_codes,
        recovery_candidate_ids=[
            item.candidate_id for item in growth_candidates
        ],
        admissible_recovery_candidate_ids=sorted(
            item.candidate_id for item in admissible
        ),
        selected_branch=branch,
        selected_model_id=selected_model_id,
        graph_checks=graph_checks,
    )

    supplied = list(replay_receipts or [])
    semantic_hash = sha256_value(
        {
            "snapshot": snapshot.model_dump(mode="json"),
            "primary_thresholds": primary_thresholds.model_dump(mode="json"),
            "adaptive_thresholds": adaptive_thresholds.model_dump(mode="json"),
        }
    )
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    executable_hash = hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
    receipts_valid = bool(
        replay_authority
        and len(supplied) == 2
        and all(
            replay_authority.verify(item)
            and item.input_semantic_hash == semantic_hash
            and item.source_hash == source_hash
            and item.executable_hash == executable_hash
            and item.environment_fingerprint
            == _replay_environment_fingerprint()
            for item in supplied
        )
        and [item.replay_index for item in supplied] == [1, 2]
        and len({item.process_id for item in supplied}) == 2
        and len({item.deterministic_output_hash for item in supplied}) == 1
    )
    l0_checks = {
        "two_fresh_subprocess_replays_present": len(supplied) == 2,
        "replay_output_hashes_identical": (
            len(supplied) == 2
            and len({item.deterministic_output_hash for item in supplied}) == 1
        ),
        "source_executable_environment_bound": receipts_valid,
        "authenticated_replay_receipts": receipts_valid,
    }
    l0 = AdaptiveLevelEvidenceV57.seal(
        level="L0",
        status=(
            "PASS"
            if all(l0_checks.values())
            else "NOT_RUN"
            if not supplied
            else "FAIL"
        ),
        checks=l0_checks,
        metrics={"replay_count": len(supplied)},
        thresholds={"fresh_subprocess_replays": 2},
        evidence={
            "receipt_hashes": [item.receipt_hash for item in supplied],
            "deterministic_output_hashes": [
                item.deterministic_output_hash for item in supplied
            ],
            "adapter_source_sha256": source_hash,
            "executable_sha256": executable_hash,
            "environment_fingerprint": _replay_environment_fingerprint(),
        },
    )
    times = np.asarray(snapshot.times, dtype=float)
    cadence = np.diff(times)
    median_cadence = float(np.median(cadence))
    cadence_deviation = float(
        np.max(np.abs(cadence - median_cadence))
        / max(abs(median_cadence), 1e-12)
    )
    l1_checks = {
        "snapshot_sealed": snapshot.snapshot_hash == snapshot.content_hash(),
        "primary_thresholds_sealed": primary_thresholds.threshold_hash
        == primary_thresholds.content_hash(),
        "adaptive_thresholds_sealed": adaptive_thresholds.threshold_hash
        == adaptive_thresholds.content_hash(),
        "units_declared": bool(snapshot.time_unit and snapshot.state_unit),
        "strictly_increasing_time": bool(np.all(cadence > 0)),
        "effectively_regular_cadence": cadence_deviation <= 1e-9,
        "positive_finite_state": bool(
            np.all(np.isfinite(values)) and np.all(values > 0)
        ),
        "slices_large_enough": (
            len(train_y) >= adaptive_thresholds.minimum_points_per_slice
            and len(validation_y)
            >= adaptive_thresholds.minimum_points_per_slice
        ),
        "adaptive_graph_contract_satisfied": all(graph.graph_checks.values()),
    }
    l1 = AdaptiveLevelEvidenceV57.seal(
        level="L1",
        status="PASS" if all(l1_checks.values()) else "FAIL",
        checks=l1_checks,
        metrics={
            "observation_count": len(values),
            "training_count": len(train_y),
            "validation_count": len(validation_y),
            "cadence_relative_deviation": cadence_deviation,
            "growth_candidate_count": len(growth_candidates),
        },
        thresholds={
            "minimum_points_per_slice": (
                adaptive_thresholds.minimum_points_per_slice
            ),
            "maximum_cadence_relative_deviation": 1e-9,
        },
        evidence={
            "snapshot_hash": snapshot.snapshot_hash,
            "primary_threshold_hash": primary_thresholds.threshold_hash,
            "adaptive_threshold_hash": adaptive_thresholds.threshold_hash,
            "graph_hash": graph.graph_hash,
            "source_id": snapshot.source_id,
            "time_unit": snapshot.time_unit,
            "state_unit": snapshot.state_unit,
        },
    )
    l2 = _l2_evidence()
    if branch == "hybrid_ode":
        primary_l3 = next(
            item for item in primary.levels if item.level == "L3"
        )
        l3_checks = {
            "primary_l1_l4_pass": all(
                primary_statuses.get(level) == "PASS"
                for level in ("L1", "L2", "L3", "L4")
            ),
            "primary_selected_candidate_admissible": (
                primary_l3.status == "PASS"
            ),
            "adaptive_graph_checks_passed": all(graph.graph_checks.values()),
        }
        l3_metrics = {
            "selected_growth_validation_relative_rmse": None,
            "selected_growth_persistence_relative_improvement": None,
            "selected_growth_innovation_lag1_correlation": None,
        }
        l3_evidence = {
            "selected_branch": branch,
            "selected_model_id": selected_model_id,
            "primary_bundle_hash": primary.bundle_hash,
            "primary_l3_evidence_hash": primary_l3.evidence_hash,
        }
    else:
        l3_checks = {
            "representation_recovery_triggered": recovery_triggered,
            "selected_growth_candidate_scientifically_admissible": bool(
                selected_growth and selected_growth.scientifically_admissible
            ),
            "recovery_failure_fails_closed": (
                branch == "log_growth" or not admissible
            ),
            "adaptive_graph_checks_passed": all(graph.graph_checks.values()),
        }
        l3_metrics = {
            "selected_growth_validation_relative_rmse": (
                selected_growth.validation_relative_rmse
                if selected_growth
                else None
            ),
            "selected_growth_persistence_relative_improvement": (
                selected_growth.persistence_relative_improvement
                if selected_growth
                else None
            ),
            "selected_growth_innovation_lag1_correlation": (
                selected_growth.absolute_validation_innovation_lag1_correlation
                if selected_growth
                else None
            ),
        }
        l3_evidence = {
            "selected_branch": branch,
            "selected_model_id": selected_model_id,
            "selected_growth_evidence_hash": (
                selected_growth.evidence_hash if selected_growth else None
            ),
            "primary_bundle_hash": primary.bundle_hash,
        }
    l3 = AdaptiveLevelEvidenceV57.seal(
        level="L3",
        status="PASS" if all(l3_checks.values()) else "FAIL",
        checks=l3_checks,
        metrics=l3_metrics,
        thresholds={
            "maximum_validation_relative_rmse": (
                adaptive_thresholds.maximum_validation_relative_rmse
            ),
            "minimum_persistence_relative_improvement": (
                adaptive_thresholds.minimum_persistence_relative_improvement
            ),
            "maximum_innovation_absolute_lag1_correlation": (
                adaptive_thresholds.maximum_innovation_absolute_lag1_correlation
            ),
        },
        evidence=l3_evidence,
    )
    if branch == "hybrid_ode":
        primary_l4 = next(
            item for item in primary.levels if item.level == "L4"
        )
        l4_checks = {
            "primary_l4_pass": primary_l4.status == "PASS",
            "no_untriggered_growth_ablation": not growth_candidates,
            "support_and_claim_limits_declared": True,
        }
        l4 = AdaptiveLevelEvidenceV57.seal(
            level="L4",
            status="PASS" if all(l4_checks.values()) else "FAIL",
            checks=l4_checks,
            metrics={
                "bootstrap_success_fraction": primary_l4.metrics.get(
                    "bootstrap_success_fraction"
                ),
                "window_sensitivity_relative_range": primary_l4.metrics.get(
                    "window_sensitivity_relative_range"
                ),
            },
            thresholds={
                "minimum_bootstrap_success_fraction": (
                    primary_thresholds.minimum_bootstrap_success_fraction
                ),
                "maximum_window_sensitivity_relative_range": (
                    primary_thresholds.maximum_window_sensitivity_relative_range
                ),
            },
            evidence={
                "selected_branch": branch,
                "primary_l4_evidence_hash": primary_l4.evidence_hash,
                "causal_identification": False,
            },
        )
    elif branch == "log_growth" and selected_growth:
        primary_selected = next(
            item
            for item in primary.candidates
            if item.candidate_id == primary.selected_candidate_id
        )
        l4 = _growth_l4(
            selected=selected_growth,
            train_y=train_y,
            primary_selected_relative_rmse=(
                primary_selected.validation_relative_rmse
            ),
            thresholds=adaptive_thresholds,
        )
    else:
        l4 = AdaptiveLevelEvidenceV57.seal(
            level="L4",
            status="FAIL",
            checks={
                "selected_candidate_available": False,
                "support_and_claim_limits_declared": True,
            },
            metrics={
                "bootstrap_success_fraction": None,
                "window_sensitivity_relative_range": None,
            },
            thresholds={
                "minimum_bootstrap_success_fraction": (
                    adaptive_thresholds.minimum_bootstrap_success_fraction
                ),
            },
            evidence={
                "selected_branch": branch,
                "causal_identification": False,
            },
        )
    levels = [l0, l1, l2, l3, l4]
    return AdaptivePositiveSeriesBundleV57.seal(
        task_id=snapshot.task_id,
        snapshot_hash=snapshot.snapshot_hash,
        primary_threshold_hash=primary_thresholds.threshold_hash,
        adaptive_threshold_hash=adaptive_thresholds.threshold_hash,
        primary_bundle=primary,
        growth_candidates=growth_candidates,
        graph=graph,
        levels=levels,
        replay_receipt_hashes=[
            str(item.receipt_hash) for item in supplied
        ],
        scientific_acceptance=all(
            item.status == "PASS" for item in levels
        ),
        fixture_only=snapshot.fixture_only,
    )


def deterministic_adaptive_positive_series_hash_v57(
    *,
    snapshot: ODETimeSeriesSnapshotV52,
    primary_thresholds: HybridODEThresholdsV56,
    adaptive_thresholds: AdaptiveThresholdsV57,
) -> str:
    bundle = build_adaptive_positive_series_bundle_v57(
        snapshot=snapshot,
        primary_thresholds=primary_thresholds,
        adaptive_thresholds=adaptive_thresholds,
    )
    return sha256_value(
        {
            "schema_version": "5.7-replay",
            "snapshot_hash": bundle.snapshot_hash,
            "primary_threshold_hash": bundle.primary_threshold_hash,
            "adaptive_threshold_hash": bundle.adaptive_threshold_hash,
            "primary_bundle_hash": bundle.primary_bundle.bundle_hash,
            "growth_candidate_hashes": [
                item.evidence_hash for item in bundle.growth_candidates
            ],
            "graph_hash": bundle.graph.graph_hash,
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


def run_authenticated_adaptive_replays_v57(
    replay_input_path: str | Path,
    *,
    authority: AdaptiveReplayAuthorityV57,
    count: Literal[2] = 2,
    timeout_seconds: int = 600,
) -> list[AdaptiveReplayReceiptV57]:
    input_path = Path(replay_input_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    input_bytes = input_path.read_bytes()
    parsed = json.loads(input_bytes)
    snapshot = ODETimeSeriesSnapshotV52.model_validate(parsed["snapshot"])
    primary = HybridODEThresholdsV56.model_validate(
        parsed["primary_thresholds"]
    )
    adaptive = AdaptiveThresholdsV57.model_validate(
        parsed["adaptive_thresholds"]
    )
    semantic_hash = sha256_value(
        {
            "snapshot": snapshot.model_dump(mode="json"),
            "primary_thresholds": primary.model_dump(mode="json"),
            "adaptive_thresholds": adaptive.model_dump(mode="json"),
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
        "fma.v5_7.adaptive_positive_series",
        "replay",
        str(input_path),
    ]
    receipts: list[AdaptiveReplayReceiptV57] = []
    for index in range(1, count + 1):
        with tempfile.TemporaryDirectory(
            prefix="fma-v57-adaptive-replay-"
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
                _stdout, stderr = process.communicate()
                raise RuntimeError(
                    "V5.7 replay timed out; stderr_sha256="
                    + hashlib.sha256(stderr.encode()).hexdigest()
                ) from exc
        if process.returncode != 0:
            raise RuntimeError(
                "V5.7 replay failed; stderr_sha256="
                + hashlib.sha256(stderr.encode()).hexdigest()
            )
        deterministic_hash = str(
            json.loads(stdout)["deterministic_output_hash"]
        )
        receipts.append(
            authority.issue(
                replay_id=f"{snapshot.task_id}-adaptive-replay-{index}",
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
        raise ValueError("V5.7 bundle is absent from frozen manifest")
    payload = path.read_bytes()
    if (
        len(payload) != binding.size_bytes
        or hashlib.sha256(payload).hexdigest() != binding.sha256
    ):
        raise ValueError("V5.7 bundle differs from frozen manifest")
    return payload


class AdaptivePositiveSeriesLevelAdapterV57:
    adapter_id = "adaptive_positive_series_scientific_adapter"
    adapter_version = "5.7"

    def __init__(self, level: LevelV57) -> None:
        self.level = level
        self.check_id = f"adaptive_positive_series_{level.lower()}"

    def run(self, context: AdapterContextV50) -> AdapterOutcomeV50:
        bundle = AdaptivePositiveSeriesBundleV57.model_validate_json(
            _read_manifest_file(
                context,
                "results/adaptive_positive_series_bundle.json",
            )
        )
        evidence = next(
            item for item in bundle.levels if item.level == self.level
        )
        payload: dict[str, Any] = {
            "bundle_hash": bundle.bundle_hash,
            "graph_hash": bundle.graph.graph_hash,
            "selected_branch": bundle.graph.selected_branch,
            "selected_model_id": bundle.graph.selected_model_id,
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
                "adaptive_positive_series_level_passed"
                if evidence.status == "PASS"
                else f"adaptive_positive_series_level_{evidence.status.lower()}"
            ),
            metrics=evidence.metrics,
            thresholds=evidence.thresholds,
            evidence_payloads=[payload],
        )


def register_adaptive_positive_series_adapters_v57(registry: Any) -> None:
    for level in ("L0", "L1", "L2", "L3", "L4"):
        registry.register(AdaptivePositiveSeriesLevelAdapterV57(level))


def _main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] != "replay":
        raise SystemExit(
            "usage: python -m fma.v5_7.adaptive_positive_series replay INPUT"
        )
    payload = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    snapshot = ODETimeSeriesSnapshotV52.model_validate(payload["snapshot"])
    primary = HybridODEThresholdsV56.model_validate(
        payload["primary_thresholds"]
    )
    adaptive = AdaptiveThresholdsV57.model_validate(
        payload["adaptive_thresholds"]
    )
    print(
        canonical_json(
            {
                "deterministic_output_hash": (
                    deterministic_adaptive_positive_series_hash_v57(
                        snapshot=snapshot,
                        primary_thresholds=primary,
                        adaptive_thresholds=adaptive,
                    )
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "AdaptiveCandidateGraphV57",
    "AdaptivePositiveSeriesBundleV57",
    "AdaptivePositiveSeriesLevelAdapterV57",
    "AdaptiveReplayAuthorityV57",
    "AdaptiveReplayReceiptV57",
    "AdaptiveThresholdsV57",
    "GrowthCandidateEvidenceV57",
    "GrowthProcessFitV57",
    "build_adaptive_positive_series_bundle_v57",
    "deterministic_adaptive_positive_series_hash_v57",
    "register_adaptive_positive_series_adapters_v57",
    "run_authenticated_adaptive_replays_v57",
]
