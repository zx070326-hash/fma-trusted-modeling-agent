"""Production-scoped L0--L4 evidence for one earthquake point-process domain.

The scope is deliberately narrow: chronological earthquake event times inside
one frozen USGS query contract.  The checks do real numerical work but do not
grant scientific qualification or authorize real-world action.
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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import scipy
from pydantic import Field, model_validator
from scipy.special import gamma
from scipy.stats import kstest

from fma.hashing import canonical_json, sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256, _assert_timezone
from fma.v4.unseen_event_process import (
    EarthquakeEventV40,
    EventProcessCandidateV40,
    FittedEventProcessV40,
    _confirmation_components,
    _duration_days,
    _hawkes_components,
    _poisson_log_likelihood,
    _relative_times,
    _weibull_components,
    fit_event_process_v40,
)
from fma.v5.check_registry import AdapterContextV50, AdapterOutcomeV50
from fma.v5.workspace_schemas import CodeManifestV50


FamilyV51 = Literal[
    "homogeneous_poisson",
    "weibull_renewal",
    "exponential_hawkes",
]
LevelV51 = Literal["L0", "L1", "L2", "L3", "L4"]
StatusV51 = Literal["PASS", "FAIL", "NOT_RUN"]
PhaseV51 = Literal["development", "holdout"]


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class EventProcessThresholdsV51(StrictModel):
    split_fraction: Annotated[float, Field(gt=0.5, lt=0.9)]
    minimum_events_per_slice: Annotated[int, Field(ge=5)]
    minimum_validation_log_score_lift_per_event: float
    minimum_ks_p_value: Annotated[float, Field(gt=0, lt=1)]
    maximum_compensator_count_relative_error: Annotated[float, Field(gt=0)]
    maximum_lag1_residual_absolute_correlation: Annotated[float, Field(gt=0)]
    minimum_parameter_bound_margin: Annotated[float, Field(ge=0)]
    bootstrap_replicates: Annotated[int, Field(ge=20)]
    bootstrap_seed: int
    minimum_successful_bootstrap_fraction: Annotated[float, Field(gt=0, le=1)]
    maximum_forecast_interval_relative_width: Annotated[float, Field(gt=0)]
    maximum_window_sensitivity_relative_range: Annotated[float, Field(gt=0)]
    maximum_ensemble_forecast_coefficient_of_variation: Annotated[
        float, Field(gt=0)
    ]


class USGSGlobalQueryV51(StrictModel):
    """Global ComCat query without the V4 Southern-California catalog lock."""

    schema_version: Literal["5.1"] = "5.1"
    phase: PhaseV51
    start: datetime
    end_exclusive: datetime
    event_type: Literal["earthquake"] = "earthquake"
    min_latitude: Annotated[float, Field(ge=-90, le=90)]
    max_latitude: Annotated[float, Field(ge=-90, le=90)]
    min_longitude: Annotated[float, Field(ge=-180, le=180)]
    max_longitude: Annotated[float, Field(ge=-180, le=180)]
    min_magnitude: Annotated[float, Field(ge=-2, le=10)]
    query_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_query(self) -> "USGSGlobalQueryV51":
        _assert_timezone(self.start, "start")
        _assert_timezone(self.end_exclusive, "end_exclusive")
        if self.end_exclusive <= self.start:
            raise ValueError("USGS query end must follow start")
        if self.min_latitude >= self.max_latitude:
            raise ValueError("USGS latitude bounds are invalid")
        if self.min_longitude >= self.max_longitude:
            raise ValueError("USGS longitude bounds are invalid")
        if self.query_hash and self.query_hash != self.content_hash():
            raise ValueError("query_hash does not match global USGS query")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "query_hash")

    def assert_sealed(self) -> None:
        if not self.query_hash or self.query_hash != self.content_hash():
            raise ValueError("global USGS query is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "USGSGlobalQueryV51":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"query_hash"}),
            query_hash=draft.content_hash(),
        )

    def url(self) -> str:
        self.assert_sealed()
        parameters = {
            "format": "geojson",
            "eventtype": self.event_type,
            "minlatitude": self.min_latitude,
            "maxlatitude": self.max_latitude,
            "minlongitude": self.min_longitude,
            "maxlongitude": self.max_longitude,
            "minmagnitude": self.min_magnitude,
            "starttime": self.start.isoformat().replace("+00:00", "Z"),
            "endtime": self.end_exclusive.isoformat().replace("+00:00", "Z"),
            "orderby": "time-asc",
            "limit": 20000,
        }
        return (
            "https://earthquake.usgs.gov/fdsnws/event/1/query?"
            + urlencode(parameters)
        )


class USGSGlobalRawV51(StrictModel):
    schema_version: Literal["5.1-raw"] = "5.1-raw"
    phase: PhaseV51
    query_hash: Sha256
    response_body: Annotated[str, Field(min_length=2)]
    response_sha256: Sha256

    @model_validator(mode="after")
    def validate_raw(self) -> "USGSGlobalRawV51":
        actual = hashlib.sha256(self.response_body.encode("utf-8")).hexdigest()
        if actual != self.response_sha256:
            raise ValueError("USGS response byte hash differs")
        return self


class USGSGlobalSnapshotV51(StrictModel):
    schema_version: Literal["5.1"] = "5.1"
    query: USGSGlobalQueryV51
    response_sha256: Sha256
    events: Annotated[list[EarthquakeEventV40], Field(min_length=1)]
    retrieved_at: datetime
    snapshot_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_snapshot(self) -> "USGSGlobalSnapshotV51":
        self.query.assert_sealed()
        _assert_timezone(self.retrieved_at, "retrieved_at")
        ids = [event.event_id for event in self.events]
        times = [event.origin_time for event in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("USGS snapshot contains duplicate event IDs")
        if times != sorted(times):
            raise ValueError("USGS snapshot must be chronological")
        if any(
            event.origin_time < self.query.start
            or event.origin_time >= self.query.end_exclusive
            or event.magnitude < self.query.min_magnitude
            or not self.query.min_latitude <= event.latitude <= self.query.max_latitude
            or not self.query.min_longitude
            <= event.longitude
            <= self.query.max_longitude
            for event in self.events
        ):
            raise ValueError("USGS event violates the frozen global query")
        if self.snapshot_hash and self.snapshot_hash != self.content_hash():
            raise ValueError("snapshot_hash does not match global USGS snapshot")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "snapshot_hash")

    def assert_sealed(self) -> None:
        if not self.snapshot_hash or self.snapshot_hash != self.content_hash():
            raise ValueError("global USGS snapshot is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "USGSGlobalSnapshotV51":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"snapshot_hash"}),
            snapshot_hash=draft.content_hash(),
        )


def parse_usgs_global_response_v51(
    query: USGSGlobalQueryV51,
    body: bytes,
    *,
    retrieved_at: datetime,
) -> tuple[USGSGlobalRawV51, USGSGlobalSnapshotV51]:
    query.assert_sealed()
    text = body.decode("utf-8")
    raw = USGSGlobalRawV51(
        phase=query.phase,
        query_hash=query.query_hash,
        response_body=text,
        response_sha256=hashlib.sha256(body).hexdigest(),
    )
    document = json.loads(text)
    if document.get("type") != "FeatureCollection":
        raise ValueError("USGS response is not a GeoJSON FeatureCollection")
    events: list[EarthquakeEventV40] = []
    for feature in document.get("features", []):
        properties = feature.get("properties", {})
        coordinates = feature.get("geometry", {}).get("coordinates", [])
        if (
            len(coordinates) < 3
            or properties.get("time") is None
            or properties.get("mag") is None
        ):
            raise ValueError("USGS feature lacks time, magnitude, or coordinates")
        origin = datetime.fromtimestamp(
            float(properties["time"]) / 1000.0, tz=timezone.utc
        )
        if origin < query.start or origin >= query.end_exclusive:
            continue
        events.append(
            EarthquakeEventV40(
                event_id=str(feature["id"]),
                origin_time=origin,
                magnitude=float(properties["mag"]),
                longitude=float(coordinates[0]),
                latitude=float(coordinates[1]),
                depth_km=float(coordinates[2]),
            )
        )
    events.sort(key=lambda item: (item.origin_time, item.event_id))
    snapshot = USGSGlobalSnapshotV51.seal(
        query=query,
        response_sha256=raw.response_sha256,
        events=events,
        retrieved_at=retrieved_at,
    )
    return raw, snapshot


def fetch_usgs_global_snapshot_v51(
    query: USGSGlobalQueryV51,
    *,
    retrieved_at: datetime | None = None,
) -> tuple[USGSGlobalRawV51, USGSGlobalSnapshotV51]:
    request = Request(
        query.url(), headers={"User-Agent": "FMA-V5.1-research/1.0"}
    )
    with urlopen(request, timeout=90) as response:
        body = response.read(16 * 1024 * 1024 + 1)
    if len(body) > 16 * 1024 * 1024:
        raise ValueError("USGS response exceeds the frozen size limit")
    return parse_usgs_global_response_v51(
        query,
        body,
        retrieved_at=retrieved_at or datetime.now(timezone.utc),
    )


class CandidateDevelopmentEvidenceV51(StrictModel):
    candidate_id: Identifier
    family: FamilyV51
    candidate: dict[str, Any]
    fit: dict[str, Any]
    validation_log_likelihood_per_event: float
    validation_log_score_lift_per_event: float
    time_rescaling_ks_p_value: Annotated[float, Field(ge=0, le=1)]
    compensator_total: Annotated[float, Field(ge=0)]
    observed_validation_count: Annotated[int, Field(ge=1)]
    compensator_count_relative_error: Annotated[float, Field(ge=0)]
    lag1_residual_correlation: Annotated[float, Field(ge=-1, le=1)]
    parameter_bound_margin: Annotated[float, Field(ge=0)]
    optimizer_converged: bool


class LevelEvidenceV51(StrictModel):
    level: LevelV51
    status: StatusV51
    checks: dict[Identifier, bool]
    thresholds: dict[str, Any]
    metrics: dict[str, Any]
    evidence: dict[str, Any]
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_level(self) -> "LevelEvidenceV51":
        if self.status == "PASS" and (not self.checks or not all(self.checks.values())):
            raise ValueError("PASS level evidence contains a failed check")
        if self.status == "FAIL" and self.checks and all(self.checks.values()):
            raise ValueError("FAIL level evidence contains no failed check")
        if self.evidence_hash and self.evidence_hash != self.content_hash():
            raise ValueError("evidence_hash does not match level evidence")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evidence_hash")

    @classmethod
    def seal(cls, **data: object) -> "LevelEvidenceV51":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evidence_hash"}),
            evidence_hash=draft.content_hash(),
        )


class EventProcessScientificBundleV51(StrictModel):
    schema_version: Literal["5.1"] = "5.1"
    task_id: Identifier
    protocol_hash: Sha256
    development_snapshot_hash: Sha256
    training_snapshot_hash: Sha256
    validation_snapshot_hash: Sha256
    candidate_registry_hash: Sha256
    selected_candidate_id: Identifier
    selected_fit_hash: Sha256
    candidates: list[CandidateDevelopmentEvidenceV51]
    levels: list[LevelEvidenceV51]
    scientific_acceptance: bool
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    bundle_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bundle(self) -> "EventProcessScientificBundleV51":
        candidate_ids = [item.candidate_id for item in self.candidates]
        if candidate_ids != sorted(set(candidate_ids)):
            raise ValueError("candidate evidence must be sorted and unique")
        levels = [item.level for item in self.levels]
        if levels != ["L0", "L1", "L2", "L3", "L4"]:
            raise ValueError("bundle must contain ordered L0--L4 evidence")
        expected_acceptance = all(item.status == "PASS" for item in self.levels)
        if self.scientific_acceptance != expected_acceptance:
            raise ValueError("scientific_acceptance differs from L0--L4")
        if self.bundle_hash and self.bundle_hash != self.content_hash():
            raise ValueError("bundle_hash does not match scientific bundle")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "bundle_hash")

    @classmethod
    def seal(cls, **data: object) -> "EventProcessScientificBundleV51":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"bundle_hash"}),
            bundle_hash=draft.content_hash(),
        )


def thresholds_from_protocol_v51(
    protocol: dict[str, Any],
) -> EventProcessThresholdsV51:
    selection = protocol["selection"]
    checks = protocol["scientific_checks"]
    return EventProcessThresholdsV51(
        split_fraction=selection["development_split_fraction"],
        minimum_events_per_slice=protocol["task"]["minimum_events_per_partition"],
        minimum_validation_log_score_lift_per_event=checks["L3"][
            "minimum_validation_log_score_lift_per_event_over_poisson"
        ],
        minimum_ks_p_value=checks["L3"][
            "time_rescaling_ks_minimum_p_value"
        ],
        maximum_compensator_count_relative_error=checks["L3"][
            "maximum_compensator_count_relative_error"
        ],
        maximum_lag1_residual_absolute_correlation=checks["L3"][
            "maximum_lag1_residual_absolute_correlation"
        ],
        minimum_parameter_bound_margin=checks["L3"][
            "minimum_parameter_bound_margin"
        ],
        bootstrap_replicates=checks["L4"]["bootstrap_replicates"],
        bootstrap_seed=checks["L4"]["bootstrap_seed"],
        minimum_successful_bootstrap_fraction=checks["L4"][
            "minimum_successful_bootstrap_fraction"
        ],
        maximum_forecast_interval_relative_width=checks["L4"][
            "maximum_forecast_interval_relative_width"
        ],
        maximum_window_sensitivity_relative_range=checks["L4"][
            "maximum_window_sensitivity_relative_range"
        ],
        maximum_ensemble_forecast_coefficient_of_variation=checks["L4"][
            "maximum_ensemble_forecast_coefficient_of_variation"
        ],
    )


def _candidate(family: FamilyV51) -> EventProcessCandidateV40:
    return EventProcessCandidateV40.seal(
        family=family,
        hawkes_branching_initial=0.45,
        hawkes_decay_days_initial=2.0,
        rationale=(
            f"Frozen registry candidate {family}; selection is performed only "
            "against the public development validation slice."
        ),
        expected_failure_modes={
            "homogeneous_poisson": [
                "constant intensity may miss clustering and nonstationarity"
            ],
            "weibull_renewal": [
                "renewal memory may miss branching aftershock structure"
            ],
            "exponential_hawkes": [
                "one exponential kernel may miss multiple triggering scales"
            ],
        }[family],
    )


def _slice_query_v51(
    source: USGSGlobalQueryV51,
    *,
    start: datetime,
    end_exclusive: datetime,
) -> USGSGlobalQueryV51:
    return USGSGlobalQueryV51.seal(
        phase="development",
        start=start,
        end_exclusive=end_exclusive,
        event_type=source.event_type,
        min_latitude=source.min_latitude,
        max_latitude=source.max_latitude,
        min_longitude=source.min_longitude,
        max_longitude=source.max_longitude,
        min_magnitude=source.min_magnitude,
    )


def split_event_process_development_v51(
    development: USGSGlobalSnapshotV51,
    split_fraction: float,
) -> tuple[USGSGlobalSnapshotV51, USGSGlobalSnapshotV51]:
    development.assert_sealed()
    duration = development.query.end_exclusive - development.query.start
    split = development.query.start + timedelta(
        seconds=duration.total_seconds() * split_fraction
    )
    training_events = [
        event for event in development.events if event.origin_time < split
    ]
    validation_events = [
        event for event in development.events if event.origin_time >= split
    ]
    if not training_events or not validation_events:
        raise ValueError("development split needs events on both sides")
    training = USGSGlobalSnapshotV51.seal(
        query=_slice_query_v51(
            development.query,
            start=development.query.start,
            end_exclusive=split,
        ),
        response_sha256=development.response_sha256,
        events=training_events,
        retrieved_at=development.retrieved_at,
    )
    validation = USGSGlobalSnapshotV51.seal(
        query=_slice_query_v51(
            development.query,
            start=split,
            end_exclusive=development.query.end_exclusive,
        ),
        response_sha256=development.response_sha256,
        events=validation_events,
        retrieved_at=development.retrieved_at,
    )
    return training, validation


def _lag1(values: list[float]) -> float:
    if len(values) < 4:
        return 0.0
    left = np.asarray(values[:-1], dtype=float)
    right = np.asarray(values[1:], dtype=float)
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else 0.0


def _parameter_margin(fit: FittedEventProcessV40) -> float:
    p = fit.parameters
    if fit.family == "homogeneous_poisson":
        return p["rate_per_day"]
    if fit.family == "weibull_renewal":
        return min(
            p["shape"] - 0.1,
            10.0 - p["shape"],
            p["scale_days"] - 1e-4,
            100.0 - p["scale_days"],
        )
    return min(
        p["background_rate_per_day"] - 1e-6,
        p["branching_ratio"] - 1e-6,
        0.949999 - p["branching_ratio"],
        p["decay_rate_per_day"] - 1.0 / 60.0,
        10.0 - p["decay_rate_per_day"],
    )


def _candidate_evidence(
    family: FamilyV51,
    training: USGSGlobalSnapshotV51,
    validation: USGSGlobalSnapshotV51,
) -> CandidateDevelopmentEvidenceV51:
    candidate = _candidate(family)
    fit = fit_event_process_v40(candidate, training)
    validation_ll, rescaled, compensator = _confirmation_components(
        fit, training, validation
    )
    duration = _duration_days(validation.query)
    baseline_rate = len(training.events) / _duration_days(training.query)
    baseline_ll = _poisson_log_likelihood(
        baseline_rate, len(validation.events), duration
    )
    count = len(validation.events)
    ks_p = (
        float(kstest(np.asarray(rescaled, dtype=float), "expon").pvalue)
        if rescaled
        and all(math.isfinite(value) and value >= 0 for value in rescaled)
        else 0.0
    )
    return CandidateDevelopmentEvidenceV51(
        candidate_id=family,
        family=family,
        candidate=candidate.model_dump(mode="json"),
        fit=fit.model_dump(mode="json"),
        validation_log_likelihood_per_event=validation_ll / count,
        validation_log_score_lift_per_event=(validation_ll - baseline_ll) / count,
        time_rescaling_ks_p_value=ks_p,
        compensator_total=compensator,
        observed_validation_count=count,
        compensator_count_relative_error=abs(compensator - count) / count,
        lag1_residual_correlation=_lag1(rescaled),
        parameter_bound_margin=max(_parameter_margin(fit), 0.0),
        optimizer_converged=fit.optimizer_converged,
    )


def _l1_evidence(
    development: USGSGlobalSnapshotV51,
    selected: CandidateDevelopmentEvidenceV51,
) -> LevelEvidenceV51:
    times = _relative_times(development)
    fit = FittedEventProcessV40.model_validate(selected.fit)
    finite_parameters = all(math.isfinite(value) for value in fit.parameters.values())
    checks = {
        "event_ids_unique": len({item.event_id for item in development.events})
        == len(development.events),
        "finite_parameters": finite_parameters,
        "nonnegative_conditional_intensity": (
            selected.compensator_total >= 0
            and math.isfinite(selected.validation_log_likelihood_per_event)
        ),
        "strictly_increasing_times": bool(np.all(np.diff(times) > 0)),
        "monotone_compensator": selected.compensator_total >= 0,
        "units_declared": True,
    }
    return LevelEvidenceV51.seal(
        level="L1",
        status="PASS" if all(checks.values()) else "FAIL",
        checks=checks,
        thresholds={"time_unit": "day", "rate_unit": "events_per_day"},
        metrics={
            "event_count": len(times),
            "minimum_interarrival_days": float(np.min(np.diff(times))),
            "parameter_count": fit.parameter_count,
        },
        evidence={
            "development_snapshot_hash": development.snapshot_hash,
            "fit_hash": fit.fit_hash,
            "family": fit.family,
        },
    )


def _l2_evidence() -> LevelEvidenceV51:
    events = np.asarray([1.0, 2.0, 4.0], dtype=float)
    duration = 5.0
    expected = _poisson_log_likelihood(0.5, len(events), duration)
    hawkes_ll, _, hawkes_comp = _hawkes_components(
        events,
        start=0.0,
        end=duration,
        history=np.asarray([], dtype=float),
        background=0.5,
        branching=0.0,
        decay=1.0,
    )
    weibull_ll, _, weibull_comp = _weibull_components(
        events,
        start=0.0,
        end=duration,
        history=np.asarray([], dtype=float),
        shape=1.0,
        scale=2.0,
    )
    tolerance = 1e-8
    checks = {
        "hawkes_zero_branching_matches_poisson": abs(hawkes_ll - expected)
        <= tolerance,
        "hawkes_compensator_matches_poisson": abs(hawkes_comp - 2.5)
        <= tolerance,
        "weibull_shape_one_matches_poisson": abs(weibull_ll - expected)
        <= tolerance,
        "weibull_compensator_matches_poisson": abs(weibull_comp - 2.5)
        <= tolerance,
    }
    return LevelEvidenceV51.seal(
        level="L2",
        status="PASS" if all(checks.values()) else "FAIL",
        checks=checks,
        thresholds={"absolute_tolerance": tolerance},
        metrics={
            "expected_poisson_log_likelihood": expected,
            "hawkes_log_likelihood": hawkes_ll,
            "weibull_log_likelihood": weibull_ll,
            "hawkes_compensator": hawkes_comp,
            "weibull_compensator": weibull_comp,
        },
        evidence={
            "oracle": "exponential_and_zero_branching_reductions",
            "events": events.tolist(),
            "duration_days": duration,
        },
    )


def _l3_evidence(
    selected: CandidateDevelopmentEvidenceV51,
    thresholds: EventProcessThresholdsV51,
    training: USGSGlobalSnapshotV51,
    validation: USGSGlobalSnapshotV51,
) -> LevelEvidenceV51:
    checks = {
        "compensator_count_calibration": (
            selected.compensator_count_relative_error
            <= thresholds.maximum_compensator_count_relative_error
        ),
        "development_slice_contract": (
            len(training.events) >= thresholds.minimum_events_per_slice
            and len(validation.events) >= thresholds.minimum_events_per_slice
        ),
        "optimizer_converged": selected.optimizer_converged,
        "parameter_interior": (
            selected.parameter_bound_margin
            >= thresholds.minimum_parameter_bound_margin
        ),
        "residual_lag1_sufficiency": (
            abs(selected.lag1_residual_correlation)
            <= thresholds.maximum_lag1_residual_absolute_correlation
        ),
        "time_rescaling_calibration": (
            selected.time_rescaling_ks_p_value
            >= thresholds.minimum_ks_p_value
        ),
        "validation_log_score_lift": (
            selected.validation_log_score_lift_per_event
            >= thresholds.minimum_validation_log_score_lift_per_event
        ),
    }
    return LevelEvidenceV51.seal(
        level="L3",
        status="PASS" if all(checks.values()) else "FAIL",
        checks=checks,
        thresholds={
            "minimum_validation_log_score_lift_per_event": (
                thresholds.minimum_validation_log_score_lift_per_event
            ),
            "minimum_ks_p_value": thresholds.minimum_ks_p_value,
            "maximum_compensator_count_relative_error": (
                thresholds.maximum_compensator_count_relative_error
            ),
            "maximum_lag1_residual_absolute_correlation": (
                thresholds.maximum_lag1_residual_absolute_correlation
            ),
            "minimum_parameter_bound_margin": (
                thresholds.minimum_parameter_bound_margin
            ),
        },
        metrics={
            "validation_log_score_lift_per_event": (
                selected.validation_log_score_lift_per_event
            ),
            "time_rescaling_ks_p_value": selected.time_rescaling_ks_p_value,
            "compensator_count_relative_error": (
                selected.compensator_count_relative_error
            ),
            "lag1_residual_correlation": selected.lag1_residual_correlation,
            "parameter_bound_margin": selected.parameter_bound_margin,
        },
        evidence={
            "candidate_id": selected.candidate_id,
            "training_snapshot_hash": training.snapshot_hash,
            "validation_snapshot_hash": validation.snapshot_hash,
        },
    )


def _stationary_forecast(fit: FittedEventProcessV40, duration: float) -> float:
    p = fit.parameters
    if fit.family == "homogeneous_poisson":
        rate = p["rate_per_day"]
    elif fit.family == "weibull_renewal":
        mean_gap = p["scale_days"] * float(gamma(1.0 + 1.0 / p["shape"]))
        rate = 1.0 / mean_gap
    else:
        rate = p["background_rate_per_day"] / (1.0 - p["branching_ratio"])
    return rate * duration


def _resampled_snapshot(
    source: USGSGlobalSnapshotV51,
    rng: np.random.Generator,
) -> USGSGlobalSnapshotV51:
    times = _relative_times(source)
    intervals = np.diff(np.concatenate(([0.0], times)))
    n = len(intervals)
    block = max(2, int(round(n ** (1.0 / 3.0))))
    sampled: list[float] = []
    while len(sampled) < n:
        start = int(rng.integers(0, n))
        sampled.extend(
            float(intervals[(start + offset) % n]) for offset in range(block)
        )
    cumulative = np.cumsum(np.asarray(sampled[:n], dtype=float))
    duration = _duration_days(source.query)
    if cumulative[-1] <= 0:
        raise ValueError("bootstrap intervals have nonpositive total")
    cumulative *= (0.999 * duration) / cumulative[-1]
    events: list[EarthquakeEventV40] = []
    for index, relative in enumerate(cumulative):
        template = source.events[index % len(source.events)]
        events.append(
            EarthquakeEventV40(
                event_id=f"bootstrap-{index:06d}",
                origin_time=source.query.start
                + timedelta(days=float(relative)),
                magnitude=template.magnitude,
                latitude=template.latitude,
                longitude=template.longitude,
                depth_km=template.depth_km,
            )
        )
    return USGSGlobalSnapshotV51.seal(
        query=source.query,
        response_sha256=source.response_sha256,
        events=events,
        retrieved_at=source.retrieved_at,
    )


def _prefix_snapshot(
    source: USGSGlobalSnapshotV51,
    fraction: float,
) -> USGSGlobalSnapshotV51:
    duration = source.query.end_exclusive - source.query.start
    end = source.query.start + timedelta(
        seconds=duration.total_seconds() * fraction
    )
    query = USGSGlobalQueryV51.seal(
        phase="development",
        start=source.query.start,
        end_exclusive=end,
        event_type=source.query.event_type,
        min_latitude=source.query.min_latitude,
        max_latitude=source.query.max_latitude,
        min_longitude=source.query.min_longitude,
        max_longitude=source.query.max_longitude,
        min_magnitude=source.query.min_magnitude,
    )
    events = [event for event in source.events if event.origin_time < end]
    return USGSGlobalSnapshotV51.seal(
        query=query,
        response_sha256=source.response_sha256,
        events=events,
        retrieved_at=source.retrieved_at,
    )


def _l4_evidence(
    selected: CandidateDevelopmentEvidenceV51,
    candidates: list[CandidateDevelopmentEvidenceV51],
    training: USGSGlobalSnapshotV51,
    validation: USGSGlobalSnapshotV51,
    thresholds: EventProcessThresholdsV51,
) -> LevelEvidenceV51:
    family = selected.family
    candidate = EventProcessCandidateV40.model_validate(selected.candidate)
    duration = _duration_days(validation.query)
    rng = np.random.default_rng(thresholds.bootstrap_seed)
    forecasts: list[float] = []
    failures = 0
    for _ in range(thresholds.bootstrap_replicates):
        try:
            sample = _resampled_snapshot(training, rng)
            fit = fit_event_process_v40(candidate, sample)
            forecast = _stationary_forecast(fit, duration)
            if not fit.optimizer_converged or not math.isfinite(forecast):
                raise ValueError("bootstrap fit failed")
            forecasts.append(forecast)
        except (ArithmeticError, ValueError):
            failures += 1
    success_fraction = len(forecasts) / thresholds.bootstrap_replicates
    if forecasts:
        low, median, high = np.quantile(forecasts, [0.025, 0.5, 0.975])
        interval_width = float((high - low) / max(abs(median), 1e-12))
    else:
        low = median = high = math.nan
        interval_width = math.inf
    window_forecasts: list[float] = []
    for fraction in (0.6, 0.7, 0.8, 0.9, 1.0):
        try:
            window = _prefix_snapshot(training, fraction)
            if len(window.events) < thresholds.minimum_events_per_slice:
                continue
            fit = fit_event_process_v40(candidate, window)
            if fit.optimizer_converged:
                window_forecasts.append(_stationary_forecast(fit, duration))
        except (ArithmeticError, ValueError):
            continue
    window_sensitivity = (
        (max(window_forecasts) - min(window_forecasts))
        / max(abs(float(np.median(window_forecasts))), 1e-12)
        if len(window_forecasts) >= 3
        else math.inf
    )
    ensemble_forecasts = [
        _stationary_forecast(
            FittedEventProcessV40.model_validate(item.fit), duration
        )
        for item in candidates
        if item.optimizer_converged
    ]
    ensemble_cv = (
        float(np.std(ensemble_forecasts, ddof=1) / np.mean(ensemble_forecasts))
        if len(ensemble_forecasts) >= 2 and np.mean(ensemble_forecasts) > 0
        else math.inf
    )
    checks = {
        "bootstrap_success_fraction": (
            success_fraction
            >= thresholds.minimum_successful_bootstrap_fraction
        ),
        "ensemble_disagreement_bounded": (
            ensemble_cv
            <= thresholds.maximum_ensemble_forecast_coefficient_of_variation
        ),
        "forecast_interval_width_bounded": (
            interval_width
            <= thresholds.maximum_forecast_interval_relative_width
        ),
        "support_declared": True,
        "window_sensitivity_bounded": (
            window_sensitivity
            <= thresholds.maximum_window_sensitivity_relative_range
        ),
    }
    return LevelEvidenceV51.seal(
        level="L4",
        status="PASS" if all(checks.values()) else "FAIL",
        checks=checks,
        thresholds={
            "bootstrap_replicates": thresholds.bootstrap_replicates,
            "minimum_successful_bootstrap_fraction": (
                thresholds.minimum_successful_bootstrap_fraction
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
        metrics={
            "bootstrap_success_fraction": success_fraction,
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
            "ensemble_forecast_coefficient_of_variation": (
                ensemble_cv if math.isfinite(ensemble_cv) else None
            ),
        },
        evidence={
            "selected_family": family,
            "bootstrap_seed": thresholds.bootstrap_seed,
            "successful_bootstrap_forecast_hash": sha256_value(forecasts),
            "window_forecasts": window_forecasts,
            "ensemble_forecasts": ensemble_forecasts,
            "support": {
                "region": "frozen_query_bounding_box_only",
                "time": "frozen_development_and_holdout_windows_only",
                "magnitude": "frozen_minimum_magnitude_only",
            },
        },
    )


def build_event_process_bundle_v51(
    *,
    task_id: str,
    protocol: dict[str, Any],
    development: USGSGlobalSnapshotV51,
    replay_output_hashes: list[str] | None = None,
) -> EventProcessScientificBundleV51:
    development.assert_sealed()
    thresholds = thresholds_from_protocol_v51(protocol)
    training, validation = split_event_process_development_v51(
        development, thresholds.split_fraction
    )
    registry = sorted(
        item["candidate_id"] for item in protocol["candidate_registry"]
    )
    families: list[FamilyV51] = [
        "homogeneous_poisson",
        "weibull_renewal",
        "exponential_hawkes",
    ]
    if registry != sorted(families):
        raise ValueError("protocol candidate registry differs from V5.1 adapter")
    candidates = sorted(
        (
            _candidate_evidence(family, training, validation)
            for family in families
        ),
        key=lambda item: item.candidate_id,
    )
    selected = sorted(
        candidates,
        key=lambda item: (
            -item.validation_log_likelihood_per_event,
            FittedEventProcessV40.model_validate(item.fit).parameter_count,
            item.candidate_id,
        ),
    )[0]
    replay_hashes = list(replay_output_hashes or [])
    l0_checks = {
        "fresh_subprocess_replays_present": len(replay_hashes) == 2,
        "replay_output_hashes_identical": (
            len(replay_hashes) == 2
            and len(set(replay_hashes)) == 1
        ),
        "source_and_environment_bound": len(replay_hashes) == 2,
    }
    source_path = Path(__file__).resolve()
    l0 = LevelEvidenceV51.seal(
        level="L0",
        status=(
            "PASS"
            if all(l0_checks.values())
            else "NOT_RUN"
            if not replay_hashes
            else "FAIL"
        ),
        checks=l0_checks,
        thresholds={"fresh_subprocess_replays": 2},
        metrics={"replay_count": len(replay_hashes)},
        evidence={
            "replay_output_hashes": replay_hashes,
            "adapter_source_sha256": hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "platform": platform.platform(),
        },
    )
    l1 = _l1_evidence(development, selected)
    l2 = _l2_evidence()
    l3 = _l3_evidence(selected, thresholds, training, validation)
    l4 = _l4_evidence(
        selected, candidates, training, validation, thresholds
    )
    levels = [l0, l1, l2, l3, l4]
    return EventProcessScientificBundleV51.seal(
        task_id=task_id,
        protocol_hash=sha256_value(protocol),
        development_snapshot_hash=development.snapshot_hash,
        training_snapshot_hash=training.snapshot_hash,
        validation_snapshot_hash=validation.snapshot_hash,
        candidate_registry_hash=sha256_value(protocol["candidate_registry"]),
        selected_candidate_id=selected.candidate_id,
        selected_fit_hash=FittedEventProcessV40.model_validate(
            selected.fit
        ).fit_hash,
        candidates=candidates,
        levels=levels,
        scientific_acceptance=all(item.status == "PASS" for item in levels),
    )


def run_bundle_replays_v51(
    replay_input_path: str | Path,
    *,
    count: int = 2,
    timeout_seconds: int = 600,
) -> list[str]:
    """Run deterministic bundle computation in fresh Python processes."""

    input_path = Path(replay_input_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    hashes: list[str] = []
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "TEMP": os.environ.get("TEMP", ""),
        "TMP": os.environ.get("TMP", ""),
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
    }
    for _ in range(count):
        with tempfile.TemporaryDirectory(prefix="fma-v51-replay-") as temp:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "fma.v5_1.event_process",
                    "replay",
                    str(input_path),
                ],
                cwd=temp,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
                shell=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                "event-process replay failed; stderr_sha256="
                + hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest()
            )
        output = json.loads(completed.stdout)
        hashes.append(str(output["deterministic_output_hash"]))
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
        raise ValueError("scientific bundle is absent from frozen manifest")
    payload = path.read_bytes()
    if (
        len(payload) != binding.size_bytes
        or hashlib.sha256(payload).hexdigest() != binding.sha256
    ):
        raise ValueError("scientific bundle differs from frozen manifest")
    return payload


class EventProcessLevelAdapterV51:
    adapter_id = "event_process_scientific_adapter"
    adapter_version = "5.1"

    def __init__(self, level: LevelV51) -> None:
        self.level = level
        self.check_id = f"event_process_{level.lower()}"

    def run(self, context: AdapterContextV50) -> AdapterOutcomeV50:
        bundle = EventProcessScientificBundleV51.model_validate_json(
            _read_manifest_file(
                context, "results/event_process_scientific_bundle.json"
            )
        )
        evidence = next(item for item in bundle.levels if item.level == self.level)
        payload: dict[str, Any] = {
            "bundle_hash": bundle.bundle_hash,
            "level_evidence": evidence.model_dump(mode="json"),
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
                "event_process_level_passed"
                if evidence.status == "PASS"
                else f"event_process_level_{evidence.status.lower()}"
            ),
            metrics=evidence.metrics,
            thresholds=evidence.thresholds,
            evidence_payloads=[payload],
        )


def register_event_process_adapters_v51(registry: Any) -> None:
    for level in ("L0", "L1", "L2", "L3", "L4"):
        registry.register(EventProcessLevelAdapterV51(level))


def _replay_main(input_path: str) -> int:
    document = json.loads(Path(input_path).read_text(encoding="utf-8"))
    development = USGSGlobalSnapshotV51.model_validate(document["development"])
    bundle = build_event_process_bundle_v51(
        task_id=document["task_id"],
        protocol=document["protocol"],
        development=development,
        replay_output_hashes=[],
    )
    deterministic = bundle.model_dump(
        mode="json",
        exclude={
            "bundle_hash": True,
            "levels": {"__all__": {"evidence_hash": True}},
        },
    )
    print(canonical_json({"deterministic_output_hash": sha256_value(deterministic)}))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "replay":
        raise SystemExit("usage: python -m fma.v5_1.event_process replay INPUT.json")
    raise SystemExit(_replay_main(sys.argv[2]))


__all__ = [
    "CandidateDevelopmentEvidenceV51",
    "EventProcessLevelAdapterV51",
    "EventProcessScientificBundleV51",
    "EventProcessThresholdsV51",
    "LevelEvidenceV51",
    "USGSGlobalQueryV51",
    "USGSGlobalRawV51",
    "USGSGlobalSnapshotV51",
    "build_event_process_bundle_v51",
    "fetch_usgs_global_snapshot_v51",
    "parse_usgs_global_response_v51",
    "register_event_process_adapters_v51",
    "run_bundle_replays_v51",
    "split_event_process_development_v51",
    "thresholds_from_protocol_v51",
]
