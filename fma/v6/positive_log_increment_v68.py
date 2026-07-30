"""V6.8 pure statistical capability pack for positive log increments.

The pack is intentionally independent of the historical adaptive recovery
bundle.  It always evaluates exactly two registered stochastic candidates:

* a random walk in log level with drift; and
* an AR(1) process for log increments.

The implementation is predictive only.  Passing local L0--L4 evidence does
not identify a causal mechanism, grant scientific qualification, or
authorize a real-world action.
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

from fma.hashing import canonical_json, sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256


FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFinite = Annotated[float, Field(ge=0, allow_inf_nan=False)]
GrowthModeV68 = Literal["log_random_walk_drift", "log_growth_ar1"]
LevelV68 = Literal["L0", "L1", "L2", "L3", "L4"]
LevelStatusV68 = Literal["PASS", "FAIL", "NOT_RUN", "HUMAN"]
SelectionStatusV68 = Literal["SELECTED", "ABSTAIN"]

GROWTH_MODES_V68: tuple[GrowthModeV68, ...] = (
    "log_random_walk_drift",
    "log_growth_ar1",
)
MINIMUM_OBSERVATIONS_V68 = 34


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _lag1(values: np.ndarray) -> float:
    if len(values) < 3:
        return 0.0
    left = values[:-1]
    right = values[1:]
    if float(np.std(left)) <= 1e-15 or float(np.std(right)) <= 1e-15:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else 0.0


def _relative_improvement(candidate: float, baseline: float) -> float:
    """Return a finite improvement without rewarding a perfect baseline."""

    scale = 1e-12
    if baseline <= scale:
        return 0.0 if candidate <= scale else 1.0 - candidate / scale
    return 1.0 - candidate / baseline


class PositiveLogIncrementThresholdsV68(StrictModel):
    schema_version: Literal["6.8-positive-log-increment-thresholds"] = (
        "6.8-positive-log-increment-thresholds"
    )
    minimum_observations: Literal[34] = 34
    split_fraction: Annotated[float, Field(gt=0.5, lt=0.9)] = 0.7
    minimum_points_per_slice: Annotated[int, Field(ge=8)] = 8
    maximum_cadence_relative_deviation: NonNegativeFinite = 1e-9
    maximum_validation_relative_rmse: NonNegativeFinite = 0.15
    minimum_persistence_relative_improvement: NonNegativeFinite = 0.10
    maximum_innovation_absolute_lag1_correlation: Annotated[
        float,
        Field(ge=0, le=1, allow_inf_nan=False),
    ] = 0.35
    maximum_absolute_growth_ar1_phi: Annotated[
        float,
        Field(gt=0, lt=1, allow_inf_nan=False),
    ] = 0.95
    minimum_growth_ar1_validation_relative_improvement: NonNegativeFinite = 0.05
    maximum_growth_phi_window_range: NonNegativeFinite = 0.30
    maximum_growth_drift_window_range_standardized: NonNegativeFinite = 1.0
    maximum_innovation_mean_shift_standardized: NonNegativeFinite = 1.5
    maximum_single_innovation_standardized: NonNegativeFinite = 5.0
    minimum_validation_interval_coverage: Annotated[
        float,
        Field(ge=0, le=1, allow_inf_nan=False),
    ] = 0.50
    maximum_absolute_mean_log_growth: NonNegativeFinite = 0.50
    selection_complexity_penalty_per_parameter: NonNegativeFinite = 0.002
    bootstrap_replicates: Annotated[int, Field(ge=20)] = 40
    bootstrap_seed: Annotated[int, Field(ge=0, le=4_294_967_295)] = 155921
    minimum_bootstrap_success_fraction: Annotated[
        float,
        Field(ge=0, le=1, allow_inf_nan=False),
    ] = 0.80
    maximum_forecast_interval_relative_width: NonNegativeFinite = 2.0
    maximum_window_sensitivity_relative_range: NonNegativeFinite = 1.0
    threshold_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_thresholds(self) -> "PositiveLogIncrementThresholdsV68":
        if self.threshold_hash and self.threshold_hash != self.content_hash():
            raise ValueError("V6.8 threshold hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "threshold_hash")

    def assert_sealed(self) -> None:
        if not self.threshold_hash or self.threshold_hash != self.content_hash():
            raise ValueError("V6.8 thresholds are not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PositiveLogIncrementThresholdsV68":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"threshold_hash"})
        payload["threshold_hash"] = draft.content_hash()
        return cls(**payload)


class PositiveScalarSeriesSnapshotV68(StrictModel):
    schema_version: Literal["6.8-positive-scalar-series-snapshot"] = (
        "6.8-positive-scalar-series-snapshot"
    )
    task_id: Identifier
    time_unit: Identifier
    state_unit: Identifier
    times: Annotated[
        list[FiniteNumber],
        Field(min_length=MINIMUM_OBSERVATIONS_V68),
    ]
    observations: Annotated[
        list[FiniteNumber],
        Field(min_length=MINIMUM_OBSERVATIONS_V68),
    ]
    source_id: Annotated[str, Field(min_length=3)]
    fixture_only: bool
    snapshot_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_snapshot(self) -> "PositiveScalarSeriesSnapshotV68":
        if len(self.times) != len(self.observations):
            raise ValueError("times and observations must have equal length")
        if any(
            right <= left for left, right in zip(self.times, self.times[1:])
        ):
            raise ValueError("times must be strictly increasing")
        if any(value <= 0 for value in self.observations):
            raise ValueError("positive log-increment observations must be positive")
        if self.snapshot_hash and self.snapshot_hash != self.content_hash():
            raise ValueError("V6.8 snapshot hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "snapshot_hash")

    def assert_sealed(self) -> None:
        if not self.snapshot_hash or self.snapshot_hash != self.content_hash():
            raise ValueError("V6.8 snapshot is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PositiveScalarSeriesSnapshotV68":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"snapshot_hash"})
        payload["snapshot_hash"] = draft.content_hash()
        return cls(**payload)


class PositiveLogIncrementModelIRV68(StrictModel):
    """Observation-free executable intent for the fixed statistical skeleton."""

    schema_version: Literal["6.8-positive-log-increment-ir"] = (
        "6.8-positive-log-increment-ir"
    )
    capability_pack_id: Literal["positive_log_increment_v68"] = (
        "positive_log_increment_v68"
    )
    candidate_ids: list[GrowthModeV68]
    baseline_id: Literal["persistence"] = "persistence"
    forecast_horizon_steps: Literal[1] = 1
    forecast_estimand: Literal["plug_in_positive_level_point_forecast"] = (
        "plug_in_positive_level_point_forecast"
    )
    threshold_hash: Sha256
    model_text_executable: Literal[False] = False
    arbitrary_code_execution_permitted: Literal[False] = False
    causal_interpretation_permitted: Literal[False] = False
    ir_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_ir(self) -> "PositiveLogIncrementModelIRV68":
        if self.candidate_ids != sorted(GROWTH_MODES_V68):
            raise ValueError(
                "V6.8 log-increment IR must bind the exact candidate registry"
            )
        if self.ir_hash and self.ir_hash != self.content_hash():
            raise ValueError("V6.8 log-increment IR hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "ir_hash")

    def assert_sealed(self) -> None:
        if not self.ir_hash or self.ir_hash != self.content_hash():
            raise ValueError("V6.8 log-increment IR is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PositiveLogIncrementModelIRV68":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"ir_hash"})
        payload["ir_hash"] = draft.content_hash()
        return cls(**payload)


def compile_positive_log_increment_ir_v68(
    thresholds: PositiveLogIncrementThresholdsV68,
) -> PositiveLogIncrementModelIRV68:
    """Compile the fixed candidate grammar without accepting observations."""

    thresholds.assert_sealed()
    return PositiveLogIncrementModelIRV68.seal(
        candidate_ids=sorted(GROWTH_MODES_V68),
        threshold_hash=thresholds.threshold_hash,
    )


class LogIncrementProcessFitV68(StrictModel):
    schema_version: Literal["6.8-positive-log-increment-fit"] = (
        "6.8-positive-log-increment-fit"
    )
    mode: GrowthModeV68
    mean_log_growth: FiniteNumber
    raw_phi: FiniteNumber
    effective_phi: FiniteNumber
    training_innovation_scale: NonNegativeFinite
    training_growth_hash: Sha256
    training_innovation_hash: Sha256
    fit_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_fit(self) -> "LogIncrementProcessFitV68":
        if self.mode == "log_random_walk_drift" and (
            self.raw_phi != 0 or self.effective_phi != 0
        ):
            raise ValueError("log drift candidate must have phi=0")
        if not -0.999 <= self.effective_phi <= 0.999:
            raise ValueError("effective growth phi is outside safe recursion")
        if self.fit_hash and self.fit_hash != self.content_hash():
            raise ValueError("V6.8 fit hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "fit_hash")

    @classmethod
    def seal(cls, **data: object) -> "LogIncrementProcessFitV68":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"fit_hash"})
        payload["fit_hash"] = draft.content_hash()
        return cls(**payload)


class LogIncrementCandidateEvidenceV68(StrictModel):
    schema_version: Literal["6.8-positive-log-increment-candidate"] = (
        "6.8-positive-log-increment-candidate"
    )
    candidate_id: Identifier
    mode: GrowthModeV68
    process_fit: LogIncrementProcessFitV68
    parameter_count: Annotated[int, Field(ge=1, le=2)]
    validation_rmse: NonNegativeFinite
    validation_relative_rmse: NonNegativeFinite
    validation_score: FiniteNumber
    persistence_rmse: NonNegativeFinite
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
    local_development_admissible: bool
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_candidate(self) -> "LogIncrementCandidateEvidenceV68":
        if self.candidate_id != self.mode:
            raise ValueError("candidate ID differs from mode")
        if self.process_fit.mode != self.mode:
            raise ValueError("candidate fit mode differs")
        expected = bool(self.admissibility_checks) and all(
            self.admissibility_checks.values()
        )
        if self.local_development_admissible != expected:
            raise ValueError(
                "candidate local development admissibility differs from checks"
            )
        if self.evidence_hash and self.evidence_hash != self.content_hash():
            raise ValueError("candidate evidence hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evidence_hash")

    @classmethod
    def seal(cls, **data: object) -> "LogIncrementCandidateEvidenceV68":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"evidence_hash"})
        payload["evidence_hash"] = draft.content_hash()
        return cls(**payload)


class PositiveLogIncrementLevelEvidenceV68(StrictModel):
    schema_version: Literal["6.8-positive-log-increment-level"] = (
        "6.8-positive-log-increment-level"
    )
    level: LevelV68
    status: LevelStatusV68
    checks: dict[Identifier, bool]
    metrics: dict[Identifier, FiniteNumber | int | None]
    thresholds: dict[Identifier, FiniteNumber | int]
    evidence: dict[str, Any]
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_level(self) -> "PositiveLogIncrementLevelEvidenceV68":
        if self.status == "PASS" and (
            not self.checks or not all(self.checks.values())
        ):
            raise ValueError("passing V6.8 level contains failed checks")
        if self.evidence_hash and self.evidence_hash != self.content_hash():
            raise ValueError("V6.8 level evidence hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evidence_hash")

    def assert_sealed(self) -> None:
        if not self.evidence_hash or self.evidence_hash != self.content_hash():
            raise ValueError("V6.8 level evidence is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PositiveLogIncrementLevelEvidenceV68":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"evidence_hash"})
        payload["evidence_hash"] = draft.content_hash()
        return cls(**payload)


class PositiveLogIncrementReplayReceiptV68(StrictModel):
    schema_version: Literal["6.8-positive-log-increment-replay"] = (
        "6.8-positive-log-increment-replay"
    )
    replay_id: Identifier
    replay_index: Annotated[int, Field(ge=1, le=2)]
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
    def validate_receipt(self) -> "PositiveLogIncrementReplayReceiptV68":
        if self.receipt_hash and (
            not self.authority_auth_tag
            or self.receipt_hash != self.content_hash()
        ):
            raise ValueError("V6.8 replay receipt differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "authority_auth_tag", "receipt_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")


class PositiveLogIncrementReplayAuthorityV68:
    def __init__(self, *, key_id: str, secret: bytes) -> None:
        if len(secret) < 32:
            raise ValueError("V6.8 replay authority needs 32 bytes")
        self.key_id = key_id
        self._secret = bytes(secret)

    def _mac(self, unsigned_hash: str) -> str:
        return hmac.new(
            self._secret,
            f"positive_log_increment_v68:{unsigned_hash}".encode(),
            hashlib.sha256,
        ).hexdigest()

    def _issue_harness_receipt(
        self,
        **data: object,
    ) -> PositiveLogIncrementReplayReceiptV68:
        """Issue only from the code-owned subprocess launcher.

        This remains a local development witness.  The signer is deliberately
        not exposed as a public arbitrary-data issuance API, and its receipts
        are never external qualification evidence.
        """

        data["authority_key_id"] = self.key_id
        unsigned = PositiveLogIncrementReplayReceiptV68(**data)
        payload = unsigned.model_dump(mode="json")
        payload["authority_auth_tag"] = self._mac(unsigned.unsigned_hash())
        tagged = PositiveLogIncrementReplayReceiptV68(**payload)
        final = tagged.model_dump(mode="json")
        final["receipt_hash"] = tagged.content_hash()
        return PositiveLogIncrementReplayReceiptV68(**final)

    def verify(self, receipt: PositiveLogIncrementReplayReceiptV68) -> bool:
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


class PositiveLogIncrementBundleV68(StrictModel):
    schema_version: Literal["6.8-positive-log-increment-bundle"] = (
        "6.8-positive-log-increment-bundle"
    )
    task_id: Identifier
    domain: Literal["positive_regular_scalar_log_increment"] = (
        "positive_regular_scalar_log_increment"
    )
    claim_kind: Literal["predictive"] = "predictive"
    forecast_estimand: Literal["plug_in_positive_level_point_forecast"] = (
        "plug_in_positive_level_point_forecast"
    )
    snapshot_hash: Sha256
    threshold_hash: Sha256
    model_ir_hash: Sha256
    candidate_registry_hash: Sha256
    candidates: list[LogIncrementCandidateEvidenceV68]
    selection_status: SelectionStatusV68
    selected_model_id: Identifier | None
    diagnostic_model_id: Identifier
    levels: list[PositiveLogIncrementLevelEvidenceV68]
    replay_receipt_hashes: list[Sha256]
    local_l0_l4_complete: bool
    scientific_acceptance: Literal[False] = False
    fixture_only: bool
    interval_evidence_kind: Literal[
        "training_bootstrap_and_development_rolling_diagnostic"
    ] = "training_bootstrap_and_development_rolling_diagnostic"
    interval_claim_ceiling: Literal["diagnostic_interval_quality_only"] = (
        "diagnostic_interval_quality_only"
    )
    temporal_dependence_coverage_guaranteed: Literal[False] = False
    finite_sample_coverage_guaranteed: Literal[False] = False
    post_selection_coverage_guaranteed: Literal[False] = False
    replay_receipts_are_external_qualification_evidence: Literal[False] = False
    causal_mechanism_identified: Literal[False] = False
    real_world_capability_established: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    bundle_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bundle(self) -> "PositiveLogIncrementBundleV68":
        candidate_ids = [item.candidate_id for item in self.candidates]
        if candidate_ids != sorted(GROWTH_MODES_V68):
            raise ValueError("V6.8 bundle must evaluate exactly two candidates")
        if [item.level for item in self.levels] != [
            "L0",
            "L1",
            "L2",
            "L3",
            "L4",
        ]:
            raise ValueError("V6.8 bundle must contain ordered L0-L4")
        admissible = sorted(
            item.candidate_id
            for item in self.candidates
            if item.local_development_admissible
        )
        if self.selection_status == "SELECTED":
            if self.selected_model_id not in admissible:
                raise ValueError("selected V6.8 model is not admissible")
        elif self.selected_model_id is not None or admissible:
            raise ValueError("V6.8 abstention differs from candidate evidence")
        if self.diagnostic_model_id not in candidate_ids:
            raise ValueError("diagnostic V6.8 model is absent")
        expected_local_completion = all(
            item.status == "PASS" for item in self.levels
        )
        if self.local_l0_l4_complete != expected_local_completion:
            raise ValueError("V6.8 local L0-L4 completion differs from levels")
        if self.replay_receipt_hashes and len(self.replay_receipt_hashes) != 2:
            raise ValueError("V6.8 replay receipts must be absent or a pair")
        if self.bundle_hash and self.bundle_hash != self.content_hash():
            raise ValueError("V6.8 bundle hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "bundle_hash")

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if not self.bundle_hash or self.bundle_hash != self.content_hash():
            raise ValueError("V6.8 log-increment bundle is not sealed")
        for level in self.levels:
            level.assert_sealed()

    @classmethod
    def seal(cls, **data: object) -> "PositiveLogIncrementBundleV68":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"bundle_hash"})
        payload["bundle_hash"] = draft.content_hash()
        return cls(**payload)


def _candidate_registry_hash() -> str:
    return sha256_value(
        {
            "schema_version": "6.8-positive-log-increment-registry",
            "candidate_ids": list(GROWTH_MODES_V68),
            "baseline_ids": ["persistence_last_value"],
        }
    )


def _estimate_growth_process(
    mode: GrowthModeV68,
    growths: np.ndarray,
) -> tuple[LogIncrementProcessFitV68, np.ndarray]:
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
        LogIncrementProcessFitV68.seal(
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
) -> tuple[float, float]:
    means: list[float] = []
    phis: list[float] = []
    for fraction in (0.70, 0.85, 1.0):
        count = min(max(7, int(len(growths) * fraction)), len(growths))
        subset = growths[:count]
        means.append(float(np.mean(subset)))
        fit, _ = _estimate_growth_process("log_growth_ar1", subset)
        phis.append(fit.raw_phi)
    return float(max(means) - min(means)), float(max(phis) - min(phis))


def _candidate_evidence(
    *,
    mode: GrowthModeV68,
    train_y: np.ndarray,
    validation_y: np.ndarray,
    persistence_rmse: float,
    thresholds: PositiveLogIncrementThresholdsV68,
    drift_rmse: float | None,
) -> LogIncrementCandidateEvidenceV68:
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
            + fit.effective_phi * (previous_growth - fit.mean_log_growth)
        )
        try:
            predicted_level = float(previous_level * math.exp(predicted_growth))
        except OverflowError:
            predicted_level = math.inf
        actual_growth = float(math.log(actual_level / previous_level))
        innovation = actual_growth - predicted_growth
        predictions.append(predicted_level)
        validation_innovations.append(innovation)
        try:
            low = predicted_level * math.exp(-1.96 * scale)
            high = predicted_level * math.exp(1.96 * scale)
        except OverflowError:
            low, high = -math.inf, math.inf
        coverages.append(low <= actual_level <= high)
        previous_growth = actual_growth
        previous_level = float(actual_level)
    prediction_array = np.asarray(predictions, dtype=float)
    innovation_array = np.asarray(validation_innovations, dtype=float)
    if not bool(np.all(np.isfinite(prediction_array))):
        rmse = float(np.finfo(float).max)
    else:
        rmse = float(np.sqrt(np.mean((prediction_array - validation_y) ** 2)))
    relative_rmse = rmse / max(float(np.mean(validation_y)), 1e-12)
    persistence_improvement = _relative_improvement(rmse, persistence_rmse)
    same_family_improvement = (
        _relative_improvement(rmse, drift_rmse)
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
                np.max(np.abs(training_innovations - training_mean)) / scale
            )
            if len(training_innovations)
            else 0.0
        ),
    )
    drift_range, phi_range = _growth_window_stability(train_growths)
    drift_range_standardized = drift_range / scale
    next_growth = float(
        fit.mean_log_growth
        + fit.effective_phi * (previous_growth - fit.mean_log_growth)
    )
    try:
        forecast = float(previous_level * math.exp(next_growth))
    except OverflowError:
        forecast = math.inf
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
            or abs(fit.raw_phi) <= thresholds.maximum_absolute_growth_ar1_phi
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
            mean_shift <= thresholds.maximum_innovation_mean_shift_standardized
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
    safe_forecast = forecast if math.isfinite(forecast) and forecast > 0 else 1e-300
    safe_rmse = rmse if math.isfinite(rmse) else float(np.finfo(float).max)
    safe_relative_rmse = (
        relative_rmse
        if math.isfinite(relative_rmse)
        else float(np.finfo(float).max)
    )
    return LogIncrementCandidateEvidenceV68.seal(
        candidate_id=mode,
        mode=mode,
        process_fit=fit,
        parameter_count=parameter_count,
        validation_rmse=safe_rmse,
        validation_relative_rmse=safe_relative_rmse,
        validation_score=-(
            safe_relative_rmse
            + thresholds.selection_complexity_penalty_per_parameter
            * parameter_count
        ),
        persistence_rmse=persistence_rmse,
        persistence_relative_improvement=persistence_improvement,
        same_family_ar1_relative_improvement=same_family_improvement,
        absolute_validation_innovation_lag1_correlation=innovation_lag,
        phi_window_range=0.0 if mode == "log_random_walk_drift" else phi_range,
        drift_window_range_standardized=drift_range_standardized,
        innovation_mean_shift_standardized=mean_shift,
        maximum_single_innovation_standardized=max_single,
        validation_interval_coverage=float(np.mean(coverages)),
        forecast_value=safe_forecast,
        validation_prediction_hash=sha256_value(predictions),
        validation_innovation_hash=sha256_value(validation_innovations),
        admissibility_checks=checks,
        local_development_admissible=all(checks.values()),
    )


def _selection_key(
    candidate: LogIncrementCandidateEvidenceV68,
) -> tuple[float, int, str]:
    return (
        -candidate.validation_score,
        candidate.parameter_count,
        candidate.candidate_id,
    )


def _l2_evidence() -> PositiveLogIncrementLevelEvidenceV68:
    values = np.asarray([4.0, 4.2, 4.1, 4.5], dtype=float)
    growth = np.diff(np.log(values))
    scaled_growth = np.diff(np.log(values * 17.0))
    scale_error = float(np.max(np.abs(growth - scaled_growth)))
    previous = 0.2
    phi = 0.8
    mean = 0.03
    recursive: list[float] = []
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
    roundtrip = float(np.max(np.abs(np.exp(np.log(values)) - values)))
    oracle_growth = np.full(12, 0.04, dtype=float)
    oracle_fit, _ = _estimate_growth_process(
        "log_random_walk_drift",
        oracle_growth,
    )
    oracle_error = abs(oracle_fit.mean_log_growth - 0.04)
    checks = {
        "positive_scale_invariant_log_growth": scale_error <= 1e-12,
        "recursive_closed_form_growth_agreement": recursion_error <= 1e-12,
        "phi_zero_reduces_to_drift": abs(phi_zero - mean) <= 1e-12,
        "stationary_growth_mean_reversion": (
            abs(recursive[-1] - mean) < abs(0.2 - mean)
        ),
        "log_level_roundtrip": roundtrip <= 1e-12,
        "constant_growth_toy_oracle": oracle_error <= 1e-12,
    }
    return PositiveLogIncrementLevelEvidenceV68.seal(
        level="L2",
        status="PASS" if all(checks.values()) else "FAIL",
        checks=checks,
        metrics={
            "scale_invariance_max_error": scale_error,
            "recursive_closed_form_max_error": recursion_error,
            "phi_zero_drift_error": abs(phi_zero - mean),
            "log_level_roundtrip_max_error": roundtrip,
            "constant_growth_oracle_error": oracle_error,
        },
        thresholds={"identity_max_error": 1e-12},
        evidence={
            "representation": "positive level with stochastic log increments",
            "forecast_estimand": "plug_in_positive_level_point_forecast",
            "causal_identification": False,
        },
    )


def _l4_evidence(
    *,
    selected: LogIncrementCandidateEvidenceV68 | None,
    train_y: np.ndarray,
    observed_y: np.ndarray,
    thresholds: PositiveLogIncrementThresholdsV68,
) -> PositiveLogIncrementLevelEvidenceV68:
    if selected is None:
        return PositiveLogIncrementLevelEvidenceV68.seal(
            level="L4",
            status="FAIL",
            checks={
                "selected_candidate_available": False,
                "support_and_claim_limits_declared": True,
            },
            metrics={
                "bootstrap_success_fraction": None,
                "forecast_interval_relative_width": None,
                "window_sensitivity_relative_range": None,
            },
            thresholds={
                "minimum_bootstrap_success_fraction": (
                    thresholds.minimum_bootstrap_success_fraction
                )
            },
            evidence={
                "claim_ceiling": "no_admissible_predictive_model",
                "interval_claim_ceiling": "no_interval_claim",
                "temporal_dependence_coverage_guaranteed": False,
                "finite_sample_coverage_guaranteed": False,
                "post_selection_coverage_guaranteed": False,
                "causal_identification": False,
            },
        )
    # L4 is a post-selection development diagnostic.  Keep the exact
    # training-only fit used by candidate selection, but condition every
    # forecast on the final observed level/growth.  The bootstrap interval and
    # the stored point forecast therefore share one origin and horizon.
    growths = np.diff(np.log(train_y))
    fit, innovations = _estimate_growth_process(selected.mode, growths)
    rng = np.random.default_rng(thresholds.bootstrap_seed)
    forecasts: list[float] = []
    last_level = float(observed_y[-1])
    last_growth = float(np.diff(np.log(observed_y))[-1])
    point_growth = float(
        fit.mean_log_growth
        + fit.effective_phi * (last_growth - fit.mean_log_growth)
    )
    point_forecast = float(last_level * math.exp(point_growth))
    point_forecast_matches = math.isclose(
        point_forecast,
        selected.forecast_value,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    for _ in range(thresholds.bootstrap_replicates):
        try:
            sampled = float(rng.choice(innovations)) if len(innovations) else 0.0
            predicted_growth = float(
                fit.mean_log_growth
                + fit.effective_phi * (last_growth - fit.mean_log_growth)
                + sampled
            )
            forecast = float(last_level * math.exp(predicted_growth))
            if not math.isfinite(forecast) or forecast <= 0:
                raise ValueError("invalid V6.8 bootstrap forecast")
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
        count = min(max(9, int(len(train_y) * fraction)), len(train_y))
        try:
            subset = train_y[:count]
            subset_growth = np.diff(np.log(subset))
            window_fit, _ = _estimate_growth_process(
                selected.mode,
                subset_growth,
            )
            window_growth = float(
                window_fit.mean_log_growth
                + window_fit.effective_phi
                * (last_growth - window_fit.mean_log_growth)
            )
            window_forecast = float(last_level * math.exp(window_growth))
            if math.isfinite(window_forecast) and window_forecast > 0:
                window_forecasts.append(window_forecast)
        except (ArithmeticError, ValueError):
            continue
    sensitivity = (
        (max(window_forecasts) - min(window_forecasts))
        / max(abs(float(np.median(window_forecasts))), 1e-12)
        if len(window_forecasts) == 3
        else math.inf
    )
    checks = {
        "bootstrap_success_fraction": (
            success >= thresholds.minimum_bootstrap_success_fraction
        ),
        "forecast_interval_width_bounded": (
            interval_width <= thresholds.maximum_forecast_interval_relative_width
        ),
        "window_sensitivity_bounded": (
            sensitivity <= thresholds.maximum_window_sensitivity_relative_range
        ),
        "selected_candidate_beats_persistence": (
            selected.persistence_relative_improvement
            >= thresholds.minimum_persistence_relative_improvement
        ),
        "point_forecast_bound_to_selected_candidate": point_forecast_matches,
        "support_and_claim_limits_declared": True,
    }
    return PositiveLogIncrementLevelEvidenceV68.seal(
        level="L4",
        status="PASS" if all(checks.values()) else "FAIL",
        checks=checks,
        metrics={
            "bootstrap_success_fraction": success,
            "forecast_interval_low": _finite_or_none(float(low)),
            "forecast_interval_median": _finite_or_none(float(median)),
            "forecast_interval_high": _finite_or_none(float(high)),
            "forecast_interval_relative_width": _finite_or_none(interval_width),
            "selected_point_forecast": _finite_or_none(point_forecast),
            "window_sensitivity_relative_range": _finite_or_none(sensitivity),
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
            "forecast_origin_observation_count": len(observed_y),
            "point_forecast_hash": sha256_value(point_forecast),
            "selected_candidate": selected.candidate_id,
            "claim_ceiling": "local_predictive_evidence_only",
            "interval_claim_ceiling": "diagnostic_interval_quality_only",
            "temporal_dependence_coverage_guaranteed": False,
            "finite_sample_coverage_guaranteed": False,
            "post_selection_coverage_guaranteed": False,
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


def _semantic_input_hash(
    snapshot: PositiveScalarSeriesSnapshotV68,
    thresholds: PositiveLogIncrementThresholdsV68,
) -> str:
    return sha256_value(
        {
            "snapshot": snapshot.model_dump(mode="json"),
            "thresholds": thresholds.model_dump(mode="json"),
        }
    )


def build_positive_log_increment_bundle_v68(
    *,
    snapshot: PositiveScalarSeriesSnapshotV68,
    thresholds: PositiveLogIncrementThresholdsV68,
    replay_receipts: list[PositiveLogIncrementReplayReceiptV68] | None = None,
    replay_authority: PositiveLogIncrementReplayAuthorityV68 | None = None,
) -> PositiveLogIncrementBundleV68:
    snapshot.assert_sealed()
    thresholds.assert_sealed()
    model_ir = compile_positive_log_increment_ir_v68(thresholds)
    values = np.asarray(snapshot.observations, dtype=float)
    times = np.asarray(snapshot.times, dtype=float)
    count = len(values)
    split = min(
        max(int(count * thresholds.split_fraction), 2),
        count - 2,
    )
    train_y = values[:split]
    validation_y = values[split:]
    persistence_predictions = np.concatenate(
        ([train_y[-1]], validation_y[:-1])
    )
    persistence_rmse = float(
        np.sqrt(np.mean((persistence_predictions - validation_y) ** 2))
    )
    drift = _candidate_evidence(
        mode="log_random_walk_drift",
        train_y=train_y,
        validation_y=validation_y,
        persistence_rmse=persistence_rmse,
        thresholds=thresholds,
        drift_rmse=None,
    )
    growth_ar1 = _candidate_evidence(
        mode="log_growth_ar1",
        train_y=train_y,
        validation_y=validation_y,
        persistence_rmse=persistence_rmse,
        thresholds=thresholds,
        drift_rmse=drift.validation_rmse,
    )
    candidates = sorted(
        [drift, growth_ar1],
        key=lambda item: item.candidate_id,
    )
    admissible = [
        item for item in candidates if item.local_development_admissible
    ]
    selected = (
        sorted(admissible, key=_selection_key)[0] if admissible else None
    )
    diagnostic = sorted(candidates, key=_selection_key)[0]

    supplied = list(replay_receipts or [])
    semantic_hash = _semantic_input_hash(snapshot, thresholds)
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
            and item.environment_fingerprint == _replay_environment_fingerprint()
            for item in supplied
        )
        and [item.replay_index for item in supplied] == [1, 2]
        and len({item.process_id for item in supplied}) == 2
        and len({item.deterministic_output_hash for item in supplied}) == 1
    )
    l0_checks = {
        "two_fresh_subprocess_replays_present": len(supplied) == 2,
        "replay_processes_distinct": (
            len(supplied) == 2
            and len({item.process_id for item in supplied}) == 2
        ),
        "replay_output_hashes_identical": (
            len(supplied) == 2
            and len({item.deterministic_output_hash for item in supplied}) == 1
        ),
        "source_executable_environment_bound": receipts_valid,
        "local_harness_authenticated_replay_receipts": receipts_valid,
        "trusted_external_replay_authority_available": False,
    }
    l0 = PositiveLogIncrementLevelEvidenceV68.seal(
        level="L0",
        status=("FAIL" if supplied and not receipts_valid else "NOT_RUN"),
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
            "evidence_scope": "local_development_replay_only",
            "trusted_external_authority": False,
            "local_replay_cannot_close_l0": True,
        },
    )

    cadence = np.diff(times)
    median_cadence = float(np.median(cadence))
    cadence_deviation = float(
        np.max(np.abs(cadence - median_cadence))
        / max(abs(median_cadence), 1e-12)
    )
    l1_checks = {
        "snapshot_sealed": snapshot.snapshot_hash == snapshot.content_hash(),
        "thresholds_sealed": thresholds.threshold_hash
        == thresholds.content_hash(),
        "units_declared": bool(snapshot.time_unit and snapshot.state_unit),
        "strictly_increasing_time": bool(np.all(cadence > 0)),
        "effectively_regular_cadence": (
            cadence_deviation
            <= thresholds.maximum_cadence_relative_deviation
        ),
        "positive_finite_state": bool(
            np.all(np.isfinite(values)) and np.all(values > 0)
        ),
        "minimum_observation_budget": count >= thresholds.minimum_observations,
        "slices_large_enough": (
            len(train_y) >= thresholds.minimum_points_per_slice
            and len(validation_y) >= thresholds.minimum_points_per_slice
        ),
        "candidate_registry_exact": (
            [item.candidate_id for item in candidates]
            == sorted(GROWTH_MODES_V68)
        ),
    }
    l1 = PositiveLogIncrementLevelEvidenceV68.seal(
        level="L1",
        status="PASS" if all(l1_checks.values()) else "FAIL",
        checks=l1_checks,
        metrics={
            "observation_count": count,
            "training_count": len(train_y),
            "validation_count": len(validation_y),
            "cadence_relative_deviation": cadence_deviation,
            "candidate_count": len(candidates),
        },
        thresholds={
            "minimum_observation_count": thresholds.minimum_observations,
            "minimum_points_per_slice": thresholds.minimum_points_per_slice,
            "maximum_cadence_relative_deviation": (
                thresholds.maximum_cadence_relative_deviation
            ),
        },
        evidence={
            "snapshot_hash": snapshot.snapshot_hash,
            "threshold_hash": thresholds.threshold_hash,
            "candidate_registry_hash": _candidate_registry_hash(),
            "source_id": snapshot.source_id,
            "time_unit": snapshot.time_unit,
            "state_unit": snapshot.state_unit,
            "development_split_fraction": thresholds.split_fraction,
        },
    )
    l2 = _l2_evidence()
    l3_checks = {
        "both_registered_candidates_evaluated": (
            [item.candidate_id for item in candidates]
            == sorted(GROWTH_MODES_V68)
        ),
        "persistence_baseline_evaluated": math.isfinite(persistence_rmse),
        "all_candidates_guarded": all(
            bool(item.admissibility_checks) for item in candidates
        ),
        "selected_candidate_available": selected is not None,
        "selected_candidate_local_development_admissible": bool(
            selected and selected.local_development_admissible
        ),
        "selection_matches_frozen_rule": bool(
            selected
            and selected.candidate_id
            == sorted(admissible, key=_selection_key)[0].candidate_id
        ),
    }
    l3 = PositiveLogIncrementLevelEvidenceV68.seal(
        level="L3",
        status="PASS" if all(l3_checks.values()) else "FAIL",
        checks=l3_checks,
        metrics={
            "persistence_rmse": persistence_rmse,
            "selected_validation_relative_rmse": (
                selected.validation_relative_rmse if selected else None
            ),
            "selected_persistence_relative_improvement": (
                selected.persistence_relative_improvement if selected else None
            ),
            "selected_innovation_lag1_correlation": (
                selected.absolute_validation_innovation_lag1_correlation
                if selected
                else None
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
        },
        evidence={
            "selection_status": "SELECTED" if selected else "ABSTAIN",
            "selected_model_id": (
                selected.candidate_id if selected else None
            ),
            "diagnostic_model_id": diagnostic.candidate_id,
            "candidate_evidence_hashes": [
                item.evidence_hash for item in candidates
            ],
            "baseline_id": "persistence_last_value",
        },
    )
    l4 = _l4_evidence(
        selected=selected,
        train_y=train_y,
        observed_y=values,
        thresholds=thresholds,
    )
    levels = [l0, l1, l2, l3, l4]
    return PositiveLogIncrementBundleV68.seal(
        task_id=snapshot.task_id,
        snapshot_hash=str(snapshot.snapshot_hash),
        threshold_hash=str(thresholds.threshold_hash),
        model_ir_hash=str(model_ir.ir_hash),
        candidate_registry_hash=_candidate_registry_hash(),
        candidates=candidates,
        selection_status="SELECTED" if selected else "ABSTAIN",
        selected_model_id=selected.candidate_id if selected else None,
        diagnostic_model_id=diagnostic.candidate_id,
        levels=levels,
        replay_receipt_hashes=[
            str(item.receipt_hash) for item in supplied
        ],
        local_l0_l4_complete=all(
            item.status == "PASS" for item in levels
        ),
        fixture_only=snapshot.fixture_only,
    )


def execute_positive_log_increment_ir_v68(
    *,
    model_ir: PositiveLogIncrementModelIRV68,
    snapshot: PositiveScalarSeriesSnapshotV68,
    thresholds: PositiveLogIncrementThresholdsV68,
    replay_receipts: list[PositiveLogIncrementReplayReceiptV68] | None = None,
    replay_authority: PositiveLogIncrementReplayAuthorityV68 | None = None,
) -> PositiveLogIncrementBundleV68:
    """Execute one exact IR; the harness still owns replay and admission."""

    model_ir.assert_sealed()
    snapshot.assert_sealed()
    thresholds.assert_sealed()
    if model_ir.threshold_hash != thresholds.threshold_hash:
        raise ValueError("V6.8 log-increment IR threshold binding differs")
    expected = compile_positive_log_increment_ir_v68(thresholds)
    if model_ir != expected:
        raise ValueError("V6.8 log-increment IR differs from the compiler")
    bundle = build_positive_log_increment_bundle_v68(
        snapshot=snapshot,
        thresholds=thresholds,
        replay_receipts=replay_receipts,
        replay_authority=replay_authority,
    )
    if bundle.model_ir_hash != model_ir.ir_hash:
        raise ValueError("V6.8 log-increment bundle is bound to another IR")
    return bundle


def verify_positive_log_increment_level_v68(
    *,
    bundle: PositiveLogIncrementBundleV68,
    level: LevelV68,
    model_ir: PositiveLogIncrementModelIRV68,
    snapshot: PositiveScalarSeriesSnapshotV68,
    thresholds: PositiveLogIncrementThresholdsV68,
    replay_receipts: list[PositiveLogIncrementReplayReceiptV68] | None = None,
    replay_authority: PositiveLogIncrementReplayAuthorityV68 | None = None,
) -> PositiveLogIncrementLevelEvidenceV68:
    """Compatibility entrypoint for the input-bound verifier.

    The implementation import is delayed to keep the executor and verifier
    modules acyclic.  Unlike the original draft, frozen inputs are mandatory.
    """

    from .positive_log_increment_verifier_v68 import (
        recompute_positive_log_increment_level_v68,
    )

    return recompute_positive_log_increment_level_v68(
        bundle=bundle,
        level=level,
        model_ir=model_ir,
        snapshot=snapshot,
        thresholds=thresholds,
        replay_receipts=replay_receipts,
        replay_authority=replay_authority,
    )


def deterministic_positive_log_increment_hash_v68(
    *,
    snapshot: PositiveScalarSeriesSnapshotV68,
    thresholds: PositiveLogIncrementThresholdsV68,
) -> str:
    bundle = build_positive_log_increment_bundle_v68(
        snapshot=snapshot,
        thresholds=thresholds,
    )
    return sha256_value(
        {
            "schema_version": "6.8-positive-log-increment-deterministic",
            "snapshot_hash": bundle.snapshot_hash,
            "threshold_hash": bundle.threshold_hash,
            "model_ir_hash": bundle.model_ir_hash,
            "candidate_registry_hash": bundle.candidate_registry_hash,
            "candidate_hashes": [
                item.evidence_hash for item in bundle.candidates
            ],
            "selection_status": bundle.selection_status,
            "selected_model_id": bundle.selected_model_id,
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


def run_authenticated_positive_log_increment_replays_v68(
    replay_input_path: str | Path,
    *,
    authority: PositiveLogIncrementReplayAuthorityV68,
    count: Literal[2] = 2,
    timeout_seconds: int = 600,
) -> list[PositiveLogIncrementReplayReceiptV68]:
    if count != 2:
        raise ValueError("V6.8 requires exactly two replays")
    input_path = Path(replay_input_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    input_bytes = input_path.read_bytes()
    parsed = json.loads(input_bytes)
    snapshot = PositiveScalarSeriesSnapshotV68.model_validate(parsed["snapshot"])
    thresholds = PositiveLogIncrementThresholdsV68.model_validate(
        parsed["thresholds"]
    )
    snapshot.assert_sealed()
    thresholds.assert_sealed()
    semantic_hash = _semantic_input_hash(snapshot, thresholds)
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
        "fma.v6.positive_log_increment_v68",
        "replay",
        str(input_path),
    ]
    receipts: list[PositiveLogIncrementReplayReceiptV68] = []
    for index in range(1, count + 1):
        with tempfile.TemporaryDirectory(
            prefix="fma-v68-log-increment-replay-"
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
                    "V6.8 replay timed out; stderr_sha256="
                    + hashlib.sha256(stderr.encode()).hexdigest()
                ) from exc
        if process.returncode != 0:
            raise RuntimeError(
                "V6.8 replay failed; stderr_sha256="
                + hashlib.sha256(stderr.encode()).hexdigest()
            )
        worker_output = json.loads(stdout)
        if worker_output.get("input_semantic_hash") != semantic_hash:
            raise RuntimeError(
                "V6.8 replay input changed between harness freeze and worker"
            )
        deterministic_hash = str(
            worker_output["deterministic_output_hash"]
        )
        receipts.append(
            authority._issue_harness_receipt(
                replay_id=f"{snapshot.task_id}-log-increment-replay-{index}",
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


def _main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] != "replay":
        raise SystemExit(
            "usage: python -m fma.v6.positive_log_increment_v68 replay INPUT"
        )
    payload = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    snapshot = PositiveScalarSeriesSnapshotV68.model_validate(
        payload["snapshot"]
    )
    thresholds = PositiveLogIncrementThresholdsV68.model_validate(
        payload["thresholds"]
    )
    print(
        canonical_json(
            {
                "input_semantic_hash": _semantic_input_hash(
                    snapshot,
                    thresholds,
                ),
                "deterministic_output_hash": (
                    deterministic_positive_log_increment_hash_v68(
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
    "GROWTH_MODES_V68",
    "MINIMUM_OBSERVATIONS_V68",
    "LogIncrementCandidateEvidenceV68",
    "LogIncrementProcessFitV68",
    "PositiveLogIncrementBundleV68",
    "PositiveLogIncrementLevelEvidenceV68",
    "PositiveLogIncrementModelIRV68",
    "PositiveLogIncrementReplayAuthorityV68",
    "PositiveLogIncrementReplayReceiptV68",
    "PositiveLogIncrementThresholdsV68",
    "PositiveScalarSeriesSnapshotV68",
    "build_positive_log_increment_bundle_v68",
    "compile_positive_log_increment_ir_v68",
    "deterministic_positive_log_increment_hash_v68",
    "execute_positive_log_increment_ir_v68",
    "run_authenticated_positive_log_increment_replays_v68",
    "verify_positive_log_increment_level_v68",
]
