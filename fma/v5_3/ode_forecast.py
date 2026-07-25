"""Horizon-aligned scalar-ODE forecasting evidence for FMA V5.3.

V5.2 remains unchanged: its L4 evidence covers one validation horizon.  V5.3
adds a separate, sealed forecast plan and evaluates every registered target
horizon.  Candidate selection is still based only on the chronological public
development split; the selected family is then refit once on all public data
and the exact refit used for registration receives its own robustness checks.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import scipy
from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5.check_registry import AdapterContextV50, AdapterOutcomeV50
from fma.v5.external_harness import PredictionDocumentV50, PredictionPointV50
from fma.v5.workspace_schemas import CodeManifestV50
from fma.v5_2.ode_system import (
    ODECandidateEvidenceV52,
    ODEFamilyV52,
    ODEFitV52,
    ODEScientificBundleV52,
    ODEThresholdsV52,
    ODETimeSeriesSnapshotV52,
    _parameter_vector,
    _predict,
    build_ode_bundle_v52,
    fit_ode_v52,
)


FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]
EvidenceStatusV53 = Literal["PASS", "FAIL", "NOT_RUN"]


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ODEForecastTargetV53(StrictModel):
    target_id: Identifier
    time: FiniteNumber


class ODEForecastPlanV53(StrictModel):
    """Public, pre-result definition of the exact forecast support."""

    schema_version: Literal["5.3"] = "5.3"
    plan_id: Identifier
    task_id: Identifier
    public_snapshot_hash: Sha256
    threshold_hash: Sha256
    targets: Annotated[list[ODEForecastTargetV53], Field(min_length=1)]
    state_unit: Identifier
    time_unit: Identifier
    frozen_at: datetime
    plan_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_plan(self) -> "ODEForecastPlanV53":
        target_ids = [item.target_id for item in self.targets]
        target_times = [item.time for item in self.targets]
        if target_ids != sorted(set(target_ids)):
            raise ValueError("forecast target IDs must be sorted and unique")
        if any(right <= left for left, right in zip(target_times, target_times[1:])):
            raise ValueError("forecast target times must be strictly increasing")
        if self.frozen_at.utcoffset() is None:
            raise ValueError("frozen_at must be timezone-aware")
        if self.plan_hash and self.plan_hash != self.content_hash():
            raise ValueError("plan_hash does not match forecast plan")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "plan_hash")

    def assert_sealed(self) -> None:
        if not self.plan_hash or self.plan_hash != self.content_hash():
            raise ValueError("forecast plan is not sealed")

    def assert_compatible(
        self,
        snapshot: ODETimeSeriesSnapshotV52,
        thresholds: ODEThresholdsV52,
    ) -> None:
        self.assert_sealed()
        snapshot.assert_sealed()
        thresholds.assert_sealed()
        if self.task_id != snapshot.task_id:
            raise ValueError("forecast plan belongs to another task")
        if self.public_snapshot_hash != snapshot.snapshot_hash:
            raise ValueError("forecast plan is bound to another public snapshot")
        if self.threshold_hash != thresholds.threshold_hash:
            raise ValueError("forecast plan is bound to other thresholds")
        if self.state_unit != snapshot.state_unit:
            raise ValueError("forecast plan state unit differs from snapshot")
        if self.time_unit != snapshot.time_unit:
            raise ValueError("forecast plan time unit differs from snapshot")
        if any(item.time <= snapshot.times[-1] for item in self.targets):
            raise ValueError(
                "every forecast target must be beyond the public observation support"
            )

    @classmethod
    def seal(cls, **data: object) -> "ODEForecastPlanV53":
        data.setdefault("frozen_at", _utc_now())
        draft = cls(**data)
        payload = draft.model_dump(exclude={"plan_hash"})
        payload["plan_hash"] = draft.content_hash()
        return cls(**payload)


class ODEForecastReplayReceiptV53(StrictModel):
    """Harness-authenticated evidence for one fresh V5.3 replay process."""

    schema_version: Literal["5.3-replay-receipt"] = "5.3-replay-receipt"
    replay_id: Identifier
    replay_index: Annotated[int, Field(ge=1)]
    process_id: Annotated[int, Field(ge=1)]
    input_bytes_hash: Sha256
    input_semantic_hash: Sha256
    command_hash: Sha256
    exit_code: int
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
    def validate_receipt(self) -> "ODEForecastReplayReceiptV53":
        if self.exit_code != 0:
            raise ValueError("successful replay receipt requires exit_code=0")
        if self.receipt_hash and (
            not self.authority_auth_tag or self.receipt_hash != self.content_hash()
        ):
            raise ValueError("replay receipt envelope differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "authority_auth_tag", "receipt_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")


class ODEForecastReplayAuthorityV53:
    """Harness authority; private key material must not enter model context."""

    def __init__(self, *, key_id: str, secret: bytes) -> None:
        if len(secret) < 32:
            raise ValueError("replay authority secret needs at least 32 bytes")
        self.key_id = key_id
        self._secret = bytes(secret)

    def _mac(self, unsigned_hash: str) -> str:
        return hmac.new(
            self._secret,
            f"ode_forecast_replay_v53:{unsigned_hash}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def issue(self, **data: object) -> ODEForecastReplayReceiptV53:
        data["authority_key_id"] = self.key_id
        unsigned = ODEForecastReplayReceiptV53(**data)
        tagged_payload = unsigned.model_dump(mode="json")
        tagged_payload["authority_auth_tag"] = self._mac(unsigned.unsigned_hash())
        tagged = ODEForecastReplayReceiptV53(**tagged_payload)
        final_payload = tagged.model_dump(mode="json")
        final_payload["receipt_hash"] = tagged.content_hash()
        return ODEForecastReplayReceiptV53(**final_payload)

    def verify(self, receipt: ODEForecastReplayReceiptV53) -> bool:
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


class ODEDevelopmentAssessmentV53(StrictModel):
    """V5.3 development evidence without reinterpreting the V5.2 bundle."""

    schema_version: Literal["5.3"] = "5.3"
    source_bundle_hash: Sha256
    selected_fit_hash: Sha256
    checks: dict[Identifier, bool]
    raw_parameter_condition_number: Annotated[float, Field(ge=1, allow_inf_nan=False)]
    relative_sensitivity_condition_number: Annotated[
        float, Field(ge=1, allow_inf_nan=False)
    ]
    status: EvidenceStatusV53
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_assessment(self) -> "ODEDevelopmentAssessmentV53":
        if self.status == "PASS" and (not self.checks or not all(self.checks.values())):
            raise ValueError("passing development assessment has failed checks")
        if self.evidence_hash and self.evidence_hash != self.content_hash():
            raise ValueError("development evidence hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evidence_hash")

    @classmethod
    def seal(cls, **data: object) -> "ODEDevelopmentAssessmentV53":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"evidence_hash"})
        payload["evidence_hash"] = draft.content_hash()
        return cls(**payload)


class ODEHorizonEvidenceV53(StrictModel):
    schema_version: Literal["5.3"] = "5.3"
    target_id: Identifier
    target_time: FiniteNumber
    point_prediction: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    bootstrap_success_fraction: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    bootstrap_interval_low: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    bootstrap_interval_median: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    bootstrap_interval_high: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    bootstrap_interval_relative_width: Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ]
    window_forecasts: list[Annotated[float, Field(gt=0, allow_inf_nan=False)]]
    window_sensitivity_relative_range: Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ]
    ensemble_forecasts: list[Annotated[float, Field(gt=0, allow_inf_nan=False)]]
    ensemble_forecast_coefficient_of_variation: Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ]
    checks: dict[Identifier, bool]
    status: Literal["PASS", "FAIL"]
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_horizon(self) -> "ODEHorizonEvidenceV53":
        if not (
            self.bootstrap_interval_low
            <= self.bootstrap_interval_median
            <= self.bootstrap_interval_high
        ):
            raise ValueError("bootstrap interval is not ordered")
        if self.status == "PASS" and (not self.checks or not all(self.checks.values())):
            raise ValueError("passing horizon evidence has failed checks")
        if self.evidence_hash and self.evidence_hash != self.content_hash():
            raise ValueError("horizon evidence hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evidence_hash")

    @classmethod
    def seal(cls, **data: object) -> "ODEHorizonEvidenceV53":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"evidence_hash"})
        payload["evidence_hash"] = draft.content_hash()
        return cls(**payload)


class ODEFinalRefitEvidenceV53(StrictModel):
    """Evidence for the exact all-public-data refit used in registration."""

    schema_version: Literal["5.3"] = "5.3"
    source_bundle_hash: Sha256
    forecast_plan_hash: Sha256
    selected_candidate_id: ODEFamilyV52
    selected_family_locked_from_development: Literal[True] = True
    final_fit: ODEFitV52
    relative_sensitivity_condition_number: Annotated[
        float, Field(ge=1, allow_inf_nan=False)
    ]
    predictions: Annotated[list[PredictionPointV50], Field(min_length=1)]
    horizons: Annotated[list[ODEHorizonEvidenceV53], Field(min_length=1)]
    checks: dict[Identifier, bool]
    status: Literal["PASS", "FAIL"]
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_refit(self) -> "ODEFinalRefitEvidenceV53":
        if self.final_fit.family != self.selected_candidate_id:
            raise ValueError("final refit family differs from selected family")
        prediction_ids = [item.target_id for item in self.predictions]
        horizon_ids = [item.target_id for item in self.horizons]
        if prediction_ids != sorted(set(prediction_ids)):
            raise ValueError("final predictions must be sorted and unique")
        if horizon_ids != prediction_ids:
            raise ValueError("horizon evidence does not cover exact predictions")
        for prediction, horizon in zip(self.predictions, self.horizons):
            if not math.isclose(
                prediction.value,
                horizon.point_prediction,
                rel_tol=0,
                abs_tol=0,
            ):
                raise ValueError("horizon point differs from registered prediction")
        if self.status == "PASS" and (
            not self.checks
            or not all(self.checks.values())
            or any(item.status != "PASS" for item in self.horizons)
        ):
            raise ValueError("passing final refit has incomplete evidence")
        if self.evidence_hash and self.evidence_hash != self.content_hash():
            raise ValueError("final refit evidence hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evidence_hash")

    @classmethod
    def seal(cls, **data: object) -> "ODEFinalRefitEvidenceV53":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"evidence_hash"})
        payload["evidence_hash"] = draft.content_hash()
        return cls(**payload)

    def prediction_document(self, case_id: str) -> PredictionDocumentV50:
        return PredictionDocumentV50(
            case_id=case_id,
            predictions=self.predictions,
        )


class ODEForecastBundleV53(StrictModel):
    schema_version: Literal["5.3"] = "5.3"
    task_id: Identifier
    domain: Literal["scalar_autonomous_ode"] = "scalar_autonomous_ode"
    public_snapshot_hash: Sha256
    threshold_hash: Sha256
    forecast_plan: ODEForecastPlanV53
    development_bundle: ODEScientificBundleV52
    development_assessment: ODEDevelopmentAssessmentV53
    final_refit: ODEFinalRefitEvidenceV53
    replay_output_hashes: list[Sha256]
    replay_receipt_hashes: list[Sha256] = Field(default_factory=list)
    replay_authentication_required: bool
    l0_checks: dict[Identifier, bool]
    scientific_acceptance: bool
    fixture_only: bool
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    bundle_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bundle(self) -> "ODEForecastBundleV53":
        if self.task_id != self.forecast_plan.task_id:
            raise ValueError("bundle and forecast plan tasks differ")
        if self.public_snapshot_hash != self.forecast_plan.public_snapshot_hash:
            raise ValueError("bundle and forecast plan snapshots differ")
        if self.threshold_hash != self.forecast_plan.threshold_hash:
            raise ValueError("bundle and forecast plan thresholds differ")
        if self.development_bundle.bundle_hash != (
            self.development_assessment.source_bundle_hash
        ):
            raise ValueError("development assessment is bound to another bundle")
        if self.final_refit.source_bundle_hash != self.development_bundle.bundle_hash:
            raise ValueError("final refit is bound to another development bundle")
        if self.final_refit.forecast_plan_hash != self.forecast_plan.plan_hash:
            raise ValueError("final refit is bound to another forecast plan")
        expected = (
            bool(self.l0_checks)
            and all(self.l0_checks.values())
            and self.development_assessment.status == "PASS"
            and self.final_refit.status == "PASS"
        )
        if self.scientific_acceptance != expected:
            raise ValueError("scientific acceptance differs from V5.3 evidence")
        if self.fixture_only != self.development_bundle.fixture_only:
            raise ValueError("fixture status differs from development bundle")
        if self.replay_authentication_required != (not self.fixture_only):
            raise ValueError("replay authentication policy differs from fixture status")
        if self.replay_receipt_hashes and len(self.replay_receipt_hashes) != 2:
            raise ValueError("replay receipt hashes must be absent or a pair")
        if self.bundle_hash and self.bundle_hash != self.content_hash():
            raise ValueError("forecast bundle hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "bundle_hash")

    @classmethod
    def seal(cls, **data: object) -> "ODEForecastBundleV53":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"bundle_hash"})
        payload["bundle_hash"] = draft.content_hash()
        return cls(**payload)


def _relative_sensitivity_condition_number(
    *,
    family: ODEFamilyV52,
    fit: ODEFitV52,
    times: np.ndarray,
    values: np.ndarray,
) -> float:
    """Condition number for dimensionless relative parameter perturbations."""

    parameters = _parameter_vector(fit)
    if len(parameters) <= 1:
        return 1.0
    span = max(float(times[-1] - times[0]), 1e-12)
    state_scale = max(float(np.mean(values)), 1e-12)
    parameter_scales = np.asarray(
        [
            max(abs(float(parameters[0])), 1.0 / span),
            max(abs(float(parameters[1])), state_scale),
        ],
        dtype=float,
    )
    epsilon = 1e-6
    columns: list[np.ndarray] = []
    for index, scale in enumerate(parameter_scales):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += epsilon * scale
        minus[index] -= epsilon * scale
        if index == 1 and minus[index] <= 0:
            minus[index] = parameters[index]
            denominator = epsilon
        else:
            denominator = 2 * epsilon
        plus_prediction = _predict(family, times, float(values[0]), plus) / state_scale
        minus_prediction = (
            _predict(family, times, float(values[0]), minus) / state_scale
        )
        columns.append((plus_prediction - minus_prediction) / denominator)
    sensitivity = np.column_stack(columns)
    information = sensitivity.T @ sensitivity
    condition = float(np.linalg.cond(information))
    if not math.isfinite(condition):
        return 1e300
    return max(condition, 1.0)


def _selected_candidate(
    bundle: ODEScientificBundleV52,
) -> ODECandidateEvidenceV52:
    return next(
        item
        for item in bundle.candidates
        if item.candidate_id == bundle.selected_candidate_id
    )


def _development_assessment(
    *,
    bundle: ODEScientificBundleV52,
    snapshot: ODETimeSeriesSnapshotV52,
    thresholds: ODEThresholdsV52,
) -> ODEDevelopmentAssessmentV53:
    selected = _selected_candidate(bundle)
    split = min(
        max(int(len(snapshot.times) * thresholds.split_fraction), 2),
        len(snapshot.times) - 2,
    )
    train_t = np.asarray(snapshot.times[:split], dtype=float)
    train_y = np.asarray(snapshot.observations[:split], dtype=float)
    relative_condition = _relative_sensitivity_condition_number(
        family=selected.candidate_id,
        fit=selected.fit,
        times=train_t,
        values=train_y,
    )
    l1 = next(item for item in bundle.levels if item.level == "L1")
    l2 = next(item for item in bundle.levels if item.level == "L2")
    l3 = next(item for item in bundle.levels if item.level == "L3")
    checks = {f"l1_{key}": value for key, value in sorted(l1.checks.items())}
    checks.update({f"l2_{key}": value for key, value in sorted(l2.checks.items())})
    checks.update(
        {
            f"l3_{key}": value
            for key, value in sorted(l3.checks.items())
            if key != "identifiability_condition_bounded"
        }
    )
    checks["l3_scale_aware_identifiability_condition_bounded"] = (
        relative_condition <= thresholds.maximum_parameter_condition_number
    )
    return ODEDevelopmentAssessmentV53.seal(
        source_bundle_hash=bundle.bundle_hash,
        selected_fit_hash=selected.fit.fit_hash,
        checks=checks,
        raw_parameter_condition_number=selected.fit.parameter_condition_number,
        relative_sensitivity_condition_number=relative_condition,
        status="PASS" if all(checks.values()) else "FAIL",
    )


def _forecast_vector(
    *,
    family: ODEFamilyV52,
    fit: ODEFitV52,
    first_time: float,
    first_value: float,
    targets: list[ODEForecastTargetV53],
) -> np.ndarray:
    times = np.asarray([first_time, *[item.time for item in targets]])
    return _predict(
        family,
        times,
        first_value,
        _parameter_vector(fit),
    )[1:]


def _final_refit_evidence(
    *,
    development_bundle: ODEScientificBundleV52,
    plan: ODEForecastPlanV53,
    snapshot: ODETimeSeriesSnapshotV52,
    thresholds: ODEThresholdsV52,
) -> ODEFinalRefitEvidenceV53:
    family = development_bundle.selected_candidate_id
    public_t = np.asarray(snapshot.times, dtype=float)
    public_y = np.asarray(snapshot.observations, dtype=float)
    final_fit = fit_ode_v52(family, public_t, public_y)
    point_values = _forecast_vector(
        family=family,
        fit=final_fit,
        first_time=float(public_t[0]),
        first_value=float(public_y[0]),
        targets=plan.targets,
    )
    relative_condition = _relative_sensitivity_condition_number(
        family=family,
        fit=final_fit,
        times=public_t,
        values=public_y,
    )

    fitted = _predict(
        family,
        public_t,
        float(public_y[0]),
        _parameter_vector(final_fit),
    )
    residuals = public_y - fitted
    rng = np.random.default_rng(thresholds.bootstrap_seed)
    bootstrap_vectors: list[np.ndarray] = []
    for _ in range(thresholds.bootstrap_replicates):
        try:
            synthetic = np.maximum(
                fitted + rng.choice(residuals, size=len(residuals), replace=True),
                1e-9,
            )
            bootstrap_fit = fit_ode_v52(family, public_t, synthetic)
            if not bootstrap_fit.optimizer_converged:
                raise ValueError("optimizer failed")
            vector = _forecast_vector(
                family=family,
                fit=bootstrap_fit,
                first_time=float(public_t[0]),
                first_value=float(synthetic[0]),
                targets=plan.targets,
            )
            if np.any(~np.isfinite(vector)) or np.any(vector <= 0):
                raise ValueError("invalid forecast vector")
            bootstrap_vectors.append(vector)
        except (ArithmeticError, ValueError):
            continue
    success_fraction = len(bootstrap_vectors) / thresholds.bootstrap_replicates

    window_vectors: list[np.ndarray] = []
    for fraction in (0.65, 0.8, 1.0):
        count = max(4, int(len(public_t) * fraction))
        count = min(count, len(public_t))
        try:
            window_fit = fit_ode_v52(family, public_t[:count], public_y[:count])
            vector = _forecast_vector(
                family=family,
                fit=window_fit,
                first_time=float(public_t[0]),
                first_value=float(public_y[0]),
                targets=plan.targets,
            )
            if (
                not window_fit.optimizer_converged
                or np.any(~np.isfinite(vector))
                or np.any(vector <= 0)
            ):
                raise ValueError("invalid window forecast")
            window_vectors.append(vector)
        except (ArithmeticError, ValueError):
            continue

    ensemble_vectors: list[np.ndarray] = []
    for candidate in development_bundle.candidates:
        ensemble_fit = fit_ode_v52(candidate.candidate_id, public_t, public_y)
        vector = _forecast_vector(
            family=candidate.candidate_id,
            fit=ensemble_fit,
            first_time=float(public_t[0]),
            first_value=float(public_y[0]),
            targets=plan.targets,
        )
        if (
            ensemble_fit.optimizer_converged
            and np.all(np.isfinite(vector))
            and np.all(vector > 0)
        ):
            ensemble_vectors.append(vector)

    horizons: list[ODEHorizonEvidenceV53] = []
    for index, target in enumerate(plan.targets):
        bootstrap_values = [float(vector[index]) for vector in bootstrap_vectors]
        if bootstrap_values:
            low, median, high = (
                float(value)
                for value in np.quantile(bootstrap_values, [0.025, 0.5, 0.975])
            )
            interval_width = (high - low) / max(abs(median), 1e-12)
        else:
            # A finite sentinel preserves strict serialization while forcing
            # the corresponding checks to fail.
            low = median = high = max(float(point_values[index]), 1e-12)
            interval_width = 1e300
        window_values = [float(vector[index]) for vector in window_vectors]
        if len(window_values) == 3:
            window_sensitivity = (max(window_values) - min(window_values)) / max(
                abs(float(np.median(window_values))), 1e-12
            )
        else:
            window_sensitivity = 1e300
        ensemble_values = [float(vector[index]) for vector in ensemble_vectors]
        if len(ensemble_values) >= 2:
            ensemble_cv = float(
                np.std(ensemble_values, ddof=1)
                / max(abs(float(np.mean(ensemble_values))), 1e-12)
            )
        else:
            ensemble_cv = 1e300
        checks = {
            "bootstrap_success_fraction": (
                success_fraction >= thresholds.minimum_bootstrap_success_fraction
            ),
            "forecast_interval_width_bounded": (
                interval_width <= thresholds.maximum_forecast_interval_relative_width
            ),
            "window_sensitivity_bounded": (
                window_sensitivity
                <= thresholds.maximum_window_sensitivity_relative_range
            ),
            "ensemble_disagreement_bounded": (
                ensemble_cv
                <= (thresholds.maximum_ensemble_forecast_coefficient_of_variation)
            ),
            "target_beyond_public_support": target.time > public_t[-1],
        }
        horizons.append(
            ODEHorizonEvidenceV53.seal(
                target_id=target.target_id,
                target_time=target.time,
                point_prediction=float(point_values[index]),
                bootstrap_success_fraction=success_fraction,
                bootstrap_interval_low=low,
                bootstrap_interval_median=median,
                bootstrap_interval_high=high,
                bootstrap_interval_relative_width=interval_width,
                window_forecasts=window_values,
                window_sensitivity_relative_range=window_sensitivity,
                ensemble_forecasts=ensemble_values,
                ensemble_forecast_coefficient_of_variation=ensemble_cv,
                checks=checks,
                status="PASS" if all(checks.values()) else "FAIL",
            )
        )

    predictions = [
        PredictionPointV50(
            target_id=target.target_id,
            value=float(point_values[index]),
        )
        for index, target in enumerate(plan.targets)
    ]
    checks = {
        "selected_family_locked_from_development": True,
        "final_optimizer_converged": final_fit.optimizer_converged,
        "all_requested_horizons_evaluated": (
            [item.target_id for item in horizons]
            == [item.target_id for item in plan.targets]
        ),
        "all_predictions_positive_finite": bool(
            np.all(np.isfinite(point_values)) and np.all(point_values > 0)
        ),
        "scale_aware_identifiability_condition_bounded": (
            relative_condition <= thresholds.maximum_parameter_condition_number
        ),
        "all_horizon_robustness_checks_passed": all(
            item.status == "PASS" for item in horizons
        ),
    }
    return ODEFinalRefitEvidenceV53.seal(
        source_bundle_hash=development_bundle.bundle_hash,
        forecast_plan_hash=plan.plan_hash,
        selected_candidate_id=family,
        final_fit=final_fit,
        relative_sensitivity_condition_number=relative_condition,
        predictions=predictions,
        horizons=horizons,
        checks=checks,
        status="PASS" if all(checks.values()) else "FAIL",
    )


def build_ode_forecast_bundle_v53(
    *,
    snapshot: ODETimeSeriesSnapshotV52,
    thresholds: ODEThresholdsV52,
    forecast_plan: ODEForecastPlanV53,
    replay_output_hashes: list[str] | None = None,
    replay_receipts: list[ODEForecastReplayReceiptV53] | None = None,
    replay_authority: ODEForecastReplayAuthorityV53 | None = None,
) -> ODEForecastBundleV53:
    """Build public evidence without reading any private target value."""

    forecast_plan.assert_compatible(snapshot, thresholds)
    development = build_ode_bundle_v52(
        snapshot=snapshot,
        thresholds=thresholds,
        replay_output_hashes=None,
    )
    assessment = _development_assessment(
        bundle=development,
        snapshot=snapshot,
        thresholds=thresholds,
    )
    final_refit = _final_refit_evidence(
        development_bundle=development,
        plan=forecast_plan,
        snapshot=snapshot,
        thresholds=thresholds,
    )
    supplied_receipts = list(replay_receipts or [])
    expected_semantic_hash = sha256_value(
        {
            "snapshot": snapshot.model_dump(mode="json"),
            "thresholds": thresholds.model_dump(mode="json"),
            "forecast_plan": forecast_plan.model_dump(mode="json"),
        }
    )
    receipts_valid = bool(
        replay_authority
        and len(supplied_receipts) == 2
        and all(
            replay_authority.verify(item)
            and item.input_semantic_hash == expected_semantic_hash
            and item.source_hash
            == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            and item.executable_hash
            == hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
            and item.environment_fingerprint
            == sha256_value(
                {
                    "python_version": platform.python_version(),
                    "numpy_version": np.__version__,
                    "scipy_version": scipy.__version__,
                    "platform": platform.platform(),
                }
            )
            for item in supplied_receipts
        )
        and len({item.process_id for item in supplied_receipts}) == 2
        and [item.replay_index for item in supplied_receipts] == [1, 2]
    )
    replay_hashes = (
        [item.deterministic_output_hash for item in supplied_receipts]
        if supplied_receipts
        else list(replay_output_hashes or [])
    )
    fixture_control_replays = snapshot.fixture_only and not supplied_receipts
    l0_checks = {
        "two_fresh_subprocess_replays_present": len(replay_hashes) == 2,
        "replay_output_hashes_identical": (
            len(replay_hashes) == 2 and len(set(replay_hashes)) == 1
        ),
        "v53_source_and_environment_bound": len(replay_hashes) == 2,
        "authenticated_replay_receipts_or_fixture_control": (
            receipts_valid or fixture_control_replays
        ),
    }
    acceptance = (
        all(l0_checks.values())
        and assessment.status == "PASS"
        and final_refit.status == "PASS"
    )
    return ODEForecastBundleV53.seal(
        task_id=snapshot.task_id,
        public_snapshot_hash=snapshot.snapshot_hash,
        threshold_hash=thresholds.threshold_hash,
        forecast_plan=forecast_plan,
        development_bundle=development,
        development_assessment=assessment,
        final_refit=final_refit,
        replay_output_hashes=replay_hashes,
        replay_receipt_hashes=[str(item.receipt_hash) for item in supplied_receipts],
        replay_authentication_required=not snapshot.fixture_only,
        l0_checks=l0_checks,
        scientific_acceptance=acceptance,
        fixture_only=snapshot.fixture_only,
    )


def deterministic_ode_forecast_hash_v53(
    *,
    snapshot: ODETimeSeriesSnapshotV52,
    thresholds: ODEThresholdsV52,
    forecast_plan: ODEForecastPlanV53,
) -> str:
    bundle = build_ode_forecast_bundle_v53(
        snapshot=snapshot,
        thresholds=thresholds,
        forecast_plan=forecast_plan,
        replay_output_hashes=None,
    )
    return sha256_value(
        {
            "schema_version": "5.3-replay",
            "snapshot_hash": bundle.public_snapshot_hash,
            "threshold_hash": bundle.threshold_hash,
            "forecast_plan_hash": bundle.forecast_plan.plan_hash,
            "development_assessment_hash": (
                bundle.development_assessment.evidence_hash
            ),
            "final_refit_evidence_hash": bundle.final_refit.evidence_hash,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )


def run_ode_forecast_replays_v53(
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
        with tempfile.TemporaryDirectory(
            prefix="fma-v53-ode-forecast-replay-"
        ) as temporary:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "fma.v5_3.ode_forecast",
                    "replay",
                    str(input_path),
                ],
                cwd=temporary,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                "V5.3 ODE forecast replay failed; stderr_sha256="
                + hashlib.sha256(completed.stderr.encode()).hexdigest()
            )
        hashes.append(str(json.loads(completed.stdout)["deterministic_output_hash"]))
    return hashes


def run_authenticated_ode_forecast_replays_v53(
    replay_input_path: str | Path,
    *,
    authority: ODEForecastReplayAuthorityV53,
    count: Literal[2] = 2,
    timeout_seconds: int = 600,
) -> list[ODEForecastReplayReceiptV53]:
    """Run two fresh child processes and authenticate their public receipts."""

    input_path = Path(replay_input_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    input_bytes = input_path.read_bytes()
    parsed = json.loads(input_bytes)
    semantic_hash = sha256_value(
        {
            "snapshot": ODETimeSeriesSnapshotV52.model_validate(
                parsed["snapshot"]
            ).model_dump(mode="json"),
            "thresholds": ODEThresholdsV52.model_validate(
                parsed["thresholds"]
            ).model_dump(mode="json"),
            "forecast_plan": ODEForecastPlanV53.model_validate(
                parsed["forecast_plan"]
            ).model_dump(mode="json"),
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
        "fma.v5_3.ode_forecast",
        "replay",
        str(input_path),
    ]
    receipts: list[ODEForecastReplayReceiptV53] = []
    for index in range(1, count + 1):
        with tempfile.TemporaryDirectory(
            prefix="fma-v53-authenticated-replay-"
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
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                raise RuntimeError(
                    "authenticated V5.3 replay timed out; stderr_sha256="
                    + hashlib.sha256(stderr.encode()).hexdigest()
                )
        if process.returncode != 0:
            raise RuntimeError(
                "authenticated V5.3 replay failed; stderr_sha256="
                + hashlib.sha256(stderr.encode()).hexdigest()
            )
        output_hash = str(json.loads(stdout)["deterministic_output_hash"])
        receipts.append(
            authority.issue(
                replay_id=f"ode-forecast-replay-{index}",
                replay_index=index,
                process_id=process.pid,
                input_bytes_hash=hashlib.sha256(input_bytes).hexdigest(),
                input_semantic_hash=semantic_hash,
                command_hash=sha256_value([Path(command[0]).name, *command[1:4]]),
                exit_code=process.returncode,
                stdout_hash=hashlib.sha256(stdout.encode()).hexdigest(),
                stderr_hash=hashlib.sha256(stderr.encode()).hexdigest(),
                deterministic_output_hash=output_hash,
                source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                executable_hash=hashlib.sha256(
                    Path(sys.executable).read_bytes()
                ).hexdigest(),
                environment_fingerprint=sha256_value(
                    {
                        "python_version": platform.python_version(),
                        "numpy_version": np.__version__,
                        "scipy_version": scipy.__version__,
                        "platform": platform.platform(),
                    }
                ),
            )
        )
    return receipts


def _read_manifest_file_v53(context: AdapterContextV50, relative_path: str) -> bytes:
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
        raise ValueError("V5.3 forecast bundle is absent from frozen manifest")
    payload = path.read_bytes()
    if (
        len(payload) != binding.size_bytes
        or hashlib.sha256(payload).hexdigest() != binding.sha256
    ):
        raise ValueError("V5.3 forecast bundle differs from frozen manifest")
    return payload


class ODEForecastLevelAdapterV53:
    """Task-specific V5 adapter over the exact frozen V5.3 bundle."""

    adapter_id = "scalar_ode_forecast_v53_scientific_adapter"
    adapter_version = "5.3"

    def __init__(self, level: Literal["L0", "L1", "L2", "L3", "L4"]) -> None:
        self.level = level
        self.check_id = f"scalar_ode_forecast_v53_{level.lower()}"

    def run(self, context: AdapterContextV50) -> AdapterOutcomeV50:
        bundle = ODEForecastBundleV53.model_validate_json(
            _read_manifest_file_v53(context, "results/ode_forecast_bundle_v53.json")
        )
        if self.level == "L0":
            passed = bool(bundle.l0_checks) and all(bundle.l0_checks.values())
            metrics = {"replay_count": len(bundle.replay_output_hashes)}
            evidence_hash = sha256_value(bundle.replay_output_hashes)
            code_manifest = CodeManifestV50.model_validate_json(
                _read_manifest_file_v53(context, "results/code_manifest.json")
            )
            computation_artifact_sha256 = code_manifest.replay_receipt_hash
            if bundle.replay_authentication_required:
                replay_receipt_bytes = _read_manifest_file_v53(
                    context, code_manifest.replay_receipt_ref
                )
                replay_receipt_payload = json.loads(replay_receipt_bytes)
                passed = passed and (
                    len(bundle.replay_receipt_hashes) == 2
                    and code_manifest.replay_receipt_hash
                    == hashlib.sha256(replay_receipt_bytes).hexdigest()
                    and replay_receipt_payload.get("replay_receipt_hashes")
                    == bundle.replay_receipt_hashes
                )
        elif self.level in {"L1", "L2"}:
            source = next(
                item
                for item in bundle.development_bundle.levels
                if item.level == self.level
            )
            passed = source.status == "PASS"
            metrics = dict(source.metrics)
            evidence_hash = source.evidence_hash
        elif self.level == "L3":
            passed = bundle.development_assessment.status == "PASS"
            metrics = {
                "raw_parameter_condition_number": (
                    bundle.development_assessment.raw_parameter_condition_number
                ),
                "relative_sensitivity_condition_number": (
                    bundle.development_assessment.relative_sensitivity_condition_number
                ),
            }
            evidence_hash = bundle.development_assessment.evidence_hash
        else:
            passed = bundle.final_refit.status == "PASS"
            metrics = {
                "target_horizon_count": len(bundle.final_refit.horizons),
                "failed_target_horizon_count": sum(
                    item.status != "PASS" for item in bundle.final_refit.horizons
                ),
            }
            evidence_hash = bundle.final_refit.evidence_hash
        evidence_payload = {
            "schema_version": "5.3-adapter-evidence",
            "forecast_bundle_hash": bundle.bundle_hash,
            "forecast_plan_hash": bundle.forecast_plan.plan_hash,
            "level": self.level,
            "level_evidence_hash": evidence_hash,
            "fixture_only": bundle.fixture_only,
            "scientific_qualification_granted": False,
            "real_world_action_authorized": False,
        }
        if self.level == "L0":
            evidence_payload["computation_artifact_sha256"] = (
                computation_artifact_sha256
            )
        return AdapterOutcomeV50(
            status="PASS" if passed else "FAIL",
            reason_code=(
                "scalar_ode_forecast_v53_level_passed"
                if passed
                else "scalar_ode_forecast_v53_level_failed"
            ),
            metrics=metrics,
            evidence_payloads=[evidence_payload],
        )


def register_ode_forecast_adapters_v53(registry: object) -> None:
    register = getattr(registry, "register")
    for level in ("L0", "L1", "L2", "L3", "L4"):
        register(ODEForecastLevelAdapterV53(level))


def _replay_main(input_path: str) -> int:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    snapshot = ODETimeSeriesSnapshotV52.model_validate(payload["snapshot"])
    thresholds = ODEThresholdsV52.model_validate(payload["thresholds"])
    plan = ODEForecastPlanV53.model_validate(payload["forecast_plan"])
    output = {
        "schema_version": "5.3-replay-output",
        "deterministic_output_hash": deterministic_ode_forecast_hash_v53(
            snapshot=snapshot,
            thresholds=thresholds,
            forecast_plan=plan,
        ),
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "replay":
        raise SystemExit("usage: python -m fma.v5_3.ode_forecast replay INPUT.json")
    raise SystemExit(_replay_main(sys.argv[2]))


__all__ = [
    "ODEDevelopmentAssessmentV53",
    "ODEFinalRefitEvidenceV53",
    "ODEForecastBundleV53",
    "ODEForecastPlanV53",
    "ODEForecastReplayAuthorityV53",
    "ODEForecastReplayReceiptV53",
    "ODEForecastTargetV53",
    "ODEHorizonEvidenceV53",
    "ODEForecastLevelAdapterV53",
    "build_ode_forecast_bundle_v53",
    "deterministic_ode_forecast_hash_v53",
    "register_ode_forecast_adapters_v53",
    "run_authenticated_ode_forecast_replays_v53",
    "run_ode_forecast_replays_v53",
]
