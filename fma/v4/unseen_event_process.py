from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Callable, Literal
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from pydantic import Field, model_validator
from scipy.optimize import minimize
from scipy.stats import kstest

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore
from fma.v2.schemas import Identifier, Sha256, _assert_timezone

from .codex_frontier_driver import (
    CodexFrontierDriverV40,
    FrontierProposalTransportV40,
)
from .graph_loop import (
    GraphEdgeV40,
    GraphLoopContractV40,
    GraphLoopStoreV40,
    GraphNodeV40,
    PromotionReceiptV40,
)


PointProcessFamilyV40 = Literal[
    "homogeneous_poisson",
    "weibull_renewal",
    "exponential_hawkes",
]
PhaseV40 = Literal["development", "confirmation"]
FetchBytes = Callable[[str], bytes]


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class USGSCatalogQueryV40(StrictModel):
    phase: PhaseV40
    start: datetime
    end_exclusive: datetime
    catalog: Literal["ci"] = "ci"
    event_type: Literal["earthquake"] = "earthquake"
    min_latitude: Annotated[float, Field(ge=-90, le=90)] = 32.0
    max_latitude: Annotated[float, Field(ge=-90, le=90)] = 37.0
    min_longitude: Annotated[float, Field(ge=-180, le=180)] = -122.0
    max_longitude: Annotated[float, Field(ge=-180, le=180)] = -114.0
    min_magnitude: Annotated[float, Field(ge=-2, le=10)] = 2.5
    max_depth_km: Annotated[float, Field(ge=0, le=1000)] = 30.0
    query_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_query(self) -> "USGSCatalogQueryV40":
        _assert_timezone(self.start, "start")
        _assert_timezone(self.end_exclusive, "end_exclusive")
        if self.end_exclusive <= self.start:
            raise ValueError("USGS query end must follow start")
        if self.min_latitude >= self.max_latitude:
            raise ValueError("USGS query latitude bounds are invalid")
        if self.min_longitude >= self.max_longitude:
            raise ValueError("USGS query longitude bounds are invalid")
        if self.query_hash and self.query_hash != self.content_hash():
            raise ValueError("query_hash does not match USGS query")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "query_hash")

    def assert_sealed(self) -> None:
        if not self.query_hash or self.query_hash != self.content_hash():
            raise ValueError("USGS query is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "USGSCatalogQueryV40":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"query_hash"}),
            query_hash=draft.content_hash(),
        )

    def url(self) -> str:
        self.assert_sealed()
        parameters = {
            "format": "geojson",
            "catalog": self.catalog,
            "eventtype": self.event_type,
            "minlatitude": self.min_latitude,
            "maxlatitude": self.max_latitude,
            "minlongitude": self.min_longitude,
            "maxlongitude": self.max_longitude,
            "minmagnitude": self.min_magnitude,
            "maxdepth": self.max_depth_km,
            "starttime": self.start.isoformat().replace("+00:00", "Z"),
            "endtime": self.end_exclusive.isoformat().replace("+00:00", "Z"),
            "orderby": "time-asc",
            "limit": 20000,
        }
        return (
            "https://earthquake.usgs.gov/fdsnws/event/1/query?"
            + urlencode(parameters)
        )


class EarthquakeEventV40(StrictModel):
    event_id: Annotated[str, Field(min_length=1, max_length=100)]
    origin_time: datetime
    magnitude: Annotated[float, Field(ge=-2, le=10, allow_inf_nan=False)]
    latitude: Annotated[float, Field(ge=-90, le=90, allow_inf_nan=False)]
    longitude: Annotated[float, Field(ge=-180, le=180, allow_inf_nan=False)]
    depth_km: Annotated[float, Field(ge=-100, le=1000, allow_inf_nan=False)]

    @model_validator(mode="after")
    def validate_event(self) -> "EarthquakeEventV40":
        _assert_timezone(self.origin_time, "origin_time")
        return self


class USGSCatalogRawV40(StrictModel):
    phase: PhaseV40
    query_hash: Sha256
    response_body: Annotated[str, Field(min_length=2)]
    response_sha256: Sha256

    @model_validator(mode="after")
    def validate_raw(self) -> "USGSCatalogRawV40":
        actual = hashlib.sha256(self.response_body.encode("utf-8")).hexdigest()
        if actual != self.response_sha256:
            raise ValueError("USGS response byte hash differs")
        return self


class USGSCatalogSnapshotV40(StrictModel):
    schema_version: Literal["4.0"] = "4.0"
    query: USGSCatalogQueryV40
    response_sha256: Sha256
    events: Annotated[list[EarthquakeEventV40], Field(min_length=1)]
    retrieved_at: datetime
    snapshot_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_snapshot(self) -> "USGSCatalogSnapshotV40":
        self.query.assert_sealed()
        _assert_timezone(self.retrieved_at, "retrieved_at")
        ids = [event.event_id for event in self.events]
        times = [event.origin_time for event in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("USGS snapshot contains duplicate event IDs")
        if times != sorted(times):
            raise ValueError("USGS snapshot events must be chronological")
        if any(
            event.origin_time < self.query.start
            or event.origin_time >= self.query.end_exclusive
            or event.magnitude < self.query.min_magnitude
            or event.depth_km > self.query.max_depth_km
            or not self.query.min_latitude <= event.latitude <= self.query.max_latitude
            or not self.query.min_longitude <= event.longitude <= self.query.max_longitude
            for event in self.events
        ):
            raise ValueError("USGS snapshot event violates the frozen query")
        if self.snapshot_hash and self.snapshot_hash != self.content_hash():
            raise ValueError("snapshot_hash does not match USGS snapshot")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "snapshot_hash")

    def assert_sealed(self) -> None:
        if not self.snapshot_hash or self.snapshot_hash != self.content_hash():
            raise ValueError("USGS catalog snapshot is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "USGSCatalogSnapshotV40":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"snapshot_hash"}),
            snapshot_hash=draft.content_hash(),
        )


class EventProcessPublicSummaryV40(StrictModel):
    schema_version: Literal["4.0"] = "4.0"
    development_snapshot_hash: Sha256
    duration_days: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    event_count: Annotated[int, Field(ge=1)]
    rate_per_day: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    mean_interarrival_days: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    median_interarrival_days: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    interarrival_coefficient_of_variation: Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ]
    daily_count_fano_factor: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    daily_count_lag1_autocorrelation: Annotated[
        float, Field(ge=-1, le=1, allow_inf_nan=False)
    ]
    summary_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_summary(self) -> "EventProcessPublicSummaryV40":
        if self.summary_hash and self.summary_hash != self.content_hash():
            raise ValueError("summary_hash does not match public summary")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "summary_hash")

    def assert_sealed(self) -> None:
        if not self.summary_hash or self.summary_hash != self.content_hash():
            raise ValueError("public event-process summary is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EventProcessPublicSummaryV40":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"summary_hash"}),
            summary_hash=draft.content_hash(),
        )


class EventProcessCandidateV40(StrictModel):
    schema_version: Literal["4.0"] = "4.0"
    family: PointProcessFamilyV40
    hawkes_branching_initial: Annotated[
        float, Field(gt=0, lt=0.95, allow_inf_nan=False)
    ] = 0.5
    hawkes_decay_days_initial: Annotated[
        float, Field(ge=0.1, le=60, allow_inf_nan=False)
    ] = 7.0
    rationale: Annotated[str, Field(min_length=10, max_length=2000)]
    expected_failure_modes: Annotated[
        list[Annotated[str, Field(min_length=3, max_length=300)]],
        Field(min_length=1, max_length=8),
    ]
    candidate_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_candidate(self) -> "EventProcessCandidateV40":
        if self.candidate_hash and self.candidate_hash != self.content_hash():
            raise ValueError("candidate_hash does not match event-process candidate")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "candidate_hash")

    def assert_sealed(self) -> None:
        if not self.candidate_hash or self.candidate_hash != self.content_hash():
            raise ValueError("event-process candidate is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EventProcessCandidateV40":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"candidate_hash"}),
            candidate_hash=draft.content_hash(),
        )


class FittedEventProcessV40(StrictModel):
    schema_version: Literal["4.0"] = "4.0"
    candidate_hash: Sha256
    development_snapshot_hash: Sha256
    family: PointProcessFamilyV40
    parameters: dict[Identifier, Annotated[float, Field(allow_inf_nan=False)]]
    optimizer_converged: bool
    optimizer_message: Annotated[str, Field(min_length=1, max_length=1000)]
    development_log_likelihood: Annotated[float, Field(allow_inf_nan=False)]
    development_bic: Annotated[float, Field(allow_inf_nan=False)]
    parameter_count: Annotated[int, Field(ge=1, le=8)]
    fit_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_fit(self) -> "FittedEventProcessV40":
        expected = {
            "homogeneous_poisson": {"rate_per_day"},
            "weibull_renewal": {"shape", "scale_days"},
            "exponential_hawkes": {
                "background_rate_per_day",
                "branching_ratio",
                "decay_rate_per_day",
            },
        }[self.family]
        if set(self.parameters) != expected:
            raise ValueError("fitted parameters differ from the family contract")
        if self.parameter_count != len(expected):
            raise ValueError("parameter_count differs from fitted parameters")
        if self.fit_hash and self.fit_hash != self.content_hash():
            raise ValueError("fit_hash does not match fitted event process")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "fit_hash")

    def assert_sealed(self) -> None:
        if not self.fit_hash or self.fit_hash != self.content_hash():
            raise ValueError("fitted event process is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "FittedEventProcessV40":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"fit_hash"}),
            fit_hash=draft.content_hash(),
        )


class EventProcessPrivateEvaluationV40(StrictModel):
    schema_version: Literal["4.0"] = "4.0"
    evaluator_epoch: Literal["earthquake-point-process-v1"]
    candidate_hash: Sha256
    fit_hash: Sha256
    confirmation_snapshot_hash: Sha256
    baseline_development_bic: Annotated[float, Field(allow_inf_nan=False)]
    candidate_development_bic: Annotated[float, Field(allow_inf_nan=False)]
    baseline_confirmation_log_likelihood: Annotated[
        float, Field(allow_inf_nan=False)
    ]
    candidate_confirmation_log_likelihood: Annotated[
        float, Field(allow_inf_nan=False)
    ]
    confirmation_log_score_lift_nat_per_event: Annotated[
        float, Field(allow_inf_nan=False)
    ]
    time_rescaling_ks_pvalue: Annotated[float, Field(ge=0, le=1)]
    compensator_total: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    observed_confirmation_count: Annotated[int, Field(ge=1)]
    compensator_count_relative_error: Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ]
    gates: dict[Identifier, bool]
    decision: Literal["qualified", "rejected"]
    real_world_execution_permitted: Literal[False] = False
    evaluation_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_evaluation(self) -> "EventProcessPrivateEvaluationV40":
        if self.decision == "qualified" and not all(self.gates.values()):
            raise ValueError("qualified evaluation contains a failed gate")
        if self.decision == "rejected" and all(self.gates.values()):
            raise ValueError("rejected evaluation contains no failed gate")
        if self.evaluation_hash and self.evaluation_hash != self.content_hash():
            raise ValueError("evaluation_hash does not match private evaluation")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evaluation_hash")

    def assert_sealed(self) -> None:
        if not self.evaluation_hash or self.evaluation_hash != self.content_hash():
            raise ValueError("private event-process evaluation is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EventProcessPrivateEvaluationV40":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evaluation_hash"}),
            evaluation_hash=draft.content_hash(),
        )


class UnseenEventProcessSpecV40(StrictModel):
    schema_version: Literal["4.0"] = "4.0"
    experiment_id: Identifier = "usgs_earthquake_event_process_v40"
    evaluator_epoch: Literal["earthquake-point-process-v1"] = (
        "earthquake-point-process-v1"
    )
    development_query: USGSCatalogQueryV40
    confirmation_query: USGSCatalogQueryV40
    minimum_events_per_phase: Annotated[int, Field(ge=10)] = 300
    maximum_events_per_phase: Annotated[int, Field(ge=300, le=20000)] = 20000
    minimum_log_score_lift_nat_per_event: Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ] = 0.01
    minimum_time_rescaling_ks_pvalue: Annotated[
        float, Field(ge=0, le=1)
    ] = 0.01
    maximum_compensator_count_relative_error: Annotated[
        float, Field(gt=0, le=1)
    ] = 0.15
    maximum_hawkes_branching_ratio: Annotated[
        float, Field(gt=0, lt=1)
    ] = 0.95
    created_at: datetime
    spec_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "UnseenEventProcessSpecV40":
        _assert_timezone(self.created_at, "created_at")
        self.development_query.assert_sealed()
        self.confirmation_query.assert_sealed()
        if self.development_query.phase != "development":
            raise ValueError("development query phase differs")
        if self.confirmation_query.phase != "confirmation":
            raise ValueError("confirmation query phase differs")
        if self.development_query.end_exclusive > self.confirmation_query.start:
            raise ValueError("development and confirmation windows overlap")
        if self.maximum_events_per_phase < self.minimum_events_per_phase:
            raise ValueError("event-count bounds are invalid")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match unseen-task spec")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "spec_hash")

    def assert_sealed(self) -> None:
        if not self.spec_hash or self.spec_hash != self.content_hash():
            raise ValueError("unseen-task spec is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "UnseenEventProcessSpecV40":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"spec_hash"}),
            spec_hash=draft.content_hash(),
        )


class UnseenEventProcessReportV40(StrictModel):
    schema_version: Literal["4.0"] = "4.0"
    experiment_id: Identifier
    spec_hash: Sha256
    development_snapshot_hash: Sha256
    confirmation_snapshot_hash: Sha256
    candidate_hash: Sha256
    fit_hash: Sha256
    evaluation_hash: Sha256
    promotion_receipt_hash: Sha256
    graph_snapshot_hash: Sha256
    transport: Literal["fixture", "codex_cli"]
    decision: Literal["qualified", "rejected"]
    qualification_scope: Literal["qualified@usgs_catalog_shadow"] = (
        "qualified@usgs_catalog_shadow"
    )
    graph_replay_verified: bool
    real_world_execution_permitted: Literal[False] = False
    created_at: datetime
    report_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "UnseenEventProcessReportV40":
        _assert_timezone(self.created_at, "created_at")
        if self.decision == "qualified" and not self.graph_replay_verified:
            raise ValueError("qualification needs graph replay verification")
        if self.report_hash and self.report_hash != self.content_hash():
            raise ValueError("report_hash does not match unseen-task report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "report_hash")

    def assert_sealed(self) -> None:
        if not self.report_hash or self.report_hash != self.content_hash():
            raise ValueError("unseen-task report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "UnseenEventProcessReportV40":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"report_hash"}),
            report_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class UnseenEventProcessOutcomeV40:
    graph: GraphLoopStoreV40
    report: UnseenEventProcessReportV40
    candidate: EventProcessCandidateV40
    fit: FittedEventProcessV40
    evaluation: EventProcessPrivateEvaluationV40


def default_unseen_event_process_spec_v40(
    *, created_at: datetime | None = None
) -> UnseenEventProcessSpecV40:
    now = created_at or datetime.now(timezone.utc)
    development = USGSCatalogQueryV40.seal(
        phase="development",
        start=datetime(2023, 1, 1, tzinfo=timezone.utc),
        end_exclusive=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    confirmation = USGSCatalogQueryV40.seal(
        phase="confirmation",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_exclusive=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    return UnseenEventProcessSpecV40.seal(
        development_query=development,
        confirmation_query=confirmation,
        created_at=now,
    )


def _default_fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "FMA-V4-research-shadow/1.0"})
    with urlopen(request, timeout=90) as response:
        body = response.read(16 * 1024 * 1024 + 1)
    if len(body) > 16 * 1024 * 1024:
        raise ValueError("USGS response exceeds the frozen size limit")
    return body


def parse_usgs_catalog_response_v40(
    query: USGSCatalogQueryV40,
    body: bytes,
    *,
    retrieved_at: datetime,
) -> tuple[USGSCatalogRawV40, USGSCatalogSnapshotV40]:
    query.assert_sealed()
    text = body.decode("utf-8")
    raw = USGSCatalogRawV40(
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
        geometry = feature.get("geometry", {})
        coordinates = geometry.get("coordinates", [])
        if len(coordinates) < 3 or properties.get("time") is None:
            raise ValueError("USGS feature lacks time or coordinates")
        origin = datetime.fromtimestamp(
            float(properties["time"]) / 1000.0,
            tz=timezone.utc,
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
    events.sort(key=lambda event: (event.origin_time, event.event_id))
    snapshot = USGSCatalogSnapshotV40.seal(
        query=query,
        response_sha256=raw.response_sha256,
        events=events,
        retrieved_at=retrieved_at,
    )
    return raw, snapshot


def fetch_usgs_catalog_snapshot_v40(
    query: USGSCatalogQueryV40,
    *,
    fetch_bytes: FetchBytes | None = None,
    retrieved_at: datetime | None = None,
) -> tuple[USGSCatalogRawV40, USGSCatalogSnapshotV40]:
    fetcher = fetch_bytes or _default_fetch_bytes
    body = fetcher(query.url())
    return parse_usgs_catalog_response_v40(
        query,
        body,
        retrieved_at=retrieved_at or datetime.now(timezone.utc),
    )


def _duration_days(query: USGSCatalogQueryV40) -> float:
    return (query.end_exclusive - query.start).total_seconds() / 86400.0


def _relative_times(snapshot: USGSCatalogSnapshotV40) -> np.ndarray:
    return np.asarray(
        [
            (event.origin_time - snapshot.query.start).total_seconds() / 86400.0
            for event in snapshot.events
        ],
        dtype=float,
    )


def summarize_development_snapshot_v40(
    snapshot: USGSCatalogSnapshotV40,
) -> EventProcessPublicSummaryV40:
    snapshot.assert_sealed()
    times = _relative_times(snapshot)
    duration = _duration_days(snapshot.query)
    interarrivals = np.diff(times)
    if len(interarrivals) < 2 or np.any(interarrivals <= 0):
        raise ValueError("development snapshot needs distinct event times")
    daily, _ = np.histogram(times, bins=np.arange(0, math.ceil(duration) + 1))
    daily_mean = float(np.mean(daily))
    if len(daily) < 3 or float(np.std(daily[:-1])) == 0 or float(np.std(daily[1:])) == 0:
        lag1 = 0.0
    else:
        lag1 = float(np.corrcoef(daily[:-1], daily[1:])[0, 1])
    return EventProcessPublicSummaryV40.seal(
        development_snapshot_hash=snapshot.snapshot_hash,
        duration_days=duration,
        event_count=len(times),
        rate_per_day=len(times) / duration,
        mean_interarrival_days=float(np.mean(interarrivals)),
        median_interarrival_days=float(np.median(interarrivals)),
        interarrival_coefficient_of_variation=float(
            np.std(interarrivals, ddof=1) / np.mean(interarrivals)
        ),
        daily_count_fano_factor=(
            float(np.var(daily, ddof=1) / daily_mean) if daily_mean > 0 else 0.0
        ),
        daily_count_lag1_autocorrelation=lag1,
    )


def _poisson_log_likelihood(rate: float, count: int, duration: float) -> float:
    if rate <= 0 or duration <= 0:
        return -math.inf
    return count * math.log(rate) - rate * duration


def _hawkes_components(
    events: np.ndarray,
    *,
    start: float,
    end: float,
    history: np.ndarray,
    background: float,
    branching: float,
    decay: float,
) -> tuple[float, list[float], float]:
    if background <= 0 or not 0 <= branching < 1 or decay <= 0:
        return -math.inf, [], math.inf
    prior = [float(value) for value in history if value < start]
    log_terms: list[float] = []
    rescaled: list[float] = []
    left = start
    total_compensator = 0.0
    for event in events:
        event_time = float(event)
        intensity = background + branching * decay * sum(
            math.exp(-decay * (event_time - past))
            for past in prior
            if past < event_time
        )
        if not math.isfinite(intensity) or intensity <= 0:
            return -math.inf, [], math.inf
        interval_integral = background * (event_time - left)
        interval_integral += branching * sum(
            math.exp(-decay * max(left - past, 0.0))
            - math.exp(-decay * (event_time - past))
            for past in prior
            if past <= left
        )
        total_compensator += interval_integral
        rescaled.append(interval_integral)
        log_terms.append(math.log(intensity))
        prior.append(event_time)
        left = event_time
    tail = background * (end - left)
    tail += branching * sum(
        math.exp(-decay * max(left - past, 0.0))
        - math.exp(-decay * (end - past))
        for past in prior
        if past <= left
    )
    total_compensator += tail
    return sum(log_terms) - total_compensator, rescaled, total_compensator


def _weibull_components(
    events: np.ndarray,
    *,
    start: float,
    end: float,
    history: np.ndarray,
    shape: float,
    scale: float,
) -> tuple[float, list[float], float]:
    if shape <= 0 or scale <= 0:
        return -math.inf, [], math.inf
    prior_events = [float(value) for value in history if value < start]
    last = prior_events[-1] if prior_events else start
    age_at_start = max(start - last, 0.0)

    def cumulative_hazard(age: float) -> float:
        return (max(age, 0.0) / scale) ** shape

    log_likelihood = 0.0
    rescaled: list[float] = []
    total_compensator = 0.0
    age_left = age_at_start
    for event in events:
        age = float(event) - last
        increment = cumulative_hazard(age) - cumulative_hazard(age_left)
        hazard = (shape / scale) * (age / scale) ** (shape - 1.0)
        if not math.isfinite(hazard) or hazard <= 0 or increment < 0:
            return -math.inf, [], math.inf
        log_likelihood += math.log(hazard) - increment
        rescaled.append(increment)
        total_compensator += increment
        last = float(event)
        age_left = 0.0
    tail_age = end - last
    tail = cumulative_hazard(tail_age) - cumulative_hazard(age_left)
    total_compensator += tail
    log_likelihood -= tail
    return log_likelihood, rescaled, total_compensator


def fit_event_process_v40(
    candidate: EventProcessCandidateV40,
    development: USGSCatalogSnapshotV40,
) -> FittedEventProcessV40:
    candidate.assert_sealed()
    development.assert_sealed()
    events = _relative_times(development)
    duration = _duration_days(development.query)
    count = len(events)
    family = candidate.family
    if family == "homogeneous_poisson":
        rate = count / duration
        log_likelihood = _poisson_log_likelihood(rate, count, duration)
        parameters = {"rate_per_day": rate}
        converged = True
        message = "closed-form maximum likelihood"
    elif family == "weibull_renewal":
        intervals = np.diff(np.concatenate(([0.0], events)))
        intervals = np.clip(intervals, 1e-9, None)

        def objective(values: np.ndarray) -> float:
            shape, scale = np.exp(values)
            likelihood, _, _ = _weibull_components(
                events,
                start=0.0,
                end=duration,
                history=np.asarray([], dtype=float),
                shape=float(shape),
                scale=float(scale),
            )
            return -likelihood if math.isfinite(likelihood) else 1e100

        initial = np.log([1.0, float(np.mean(intervals))])
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=[(math.log(0.1), math.log(10.0)), (math.log(1e-4), math.log(100.0))],
        )
        shape, scale = np.exp(result.x)
        log_likelihood = -float(result.fun)
        parameters = {"shape": float(shape), "scale_days": float(scale)}
        converged = bool(result.success and math.isfinite(log_likelihood))
        message = str(result.message)
    else:
        rate = count / duration
        candidate_initial = [
            rate * (1.0 - candidate.hawkes_branching_initial),
            candidate.hawkes_branching_initial,
            1.0 / candidate.hawkes_decay_days_initial,
        ]
        starts = [
            candidate_initial,
            [rate * 0.8, 0.2, 1.0 / 2.0],
            [rate * 0.5, 0.5, 1.0 / 14.0],
        ]

        def objective(values: np.ndarray) -> float:
            likelihood, _, _ = _hawkes_components(
                events,
                start=0.0,
                end=duration,
                history=np.asarray([], dtype=float),
                background=float(values[0]),
                branching=float(values[1]),
                decay=float(values[2]),
            )
            return -likelihood if math.isfinite(likelihood) else 1e100

        upper_background = max(10.0, 10.0 * rate)
        results = [
            minimize(
                objective,
                np.asarray(start_values, dtype=float),
                method="L-BFGS-B",
                bounds=[(1e-6, upper_background), (1e-6, 0.949999), (1.0 / 60.0, 10.0)],
            )
            for start_values in starts
        ]
        result = min(results, key=lambda item: float(item.fun))
        log_likelihood = -float(result.fun)
        parameters = {
            "background_rate_per_day": float(result.x[0]),
            "branching_ratio": float(result.x[1]),
            "decay_rate_per_day": float(result.x[2]),
        }
        converged = bool(result.success and math.isfinite(log_likelihood))
        message = str(result.message)
    parameter_count = len(parameters)
    bic = parameter_count * math.log(count) - 2.0 * log_likelihood
    return FittedEventProcessV40.seal(
        candidate_hash=candidate.candidate_hash,
        development_snapshot_hash=development.snapshot_hash,
        family=family,
        parameters=parameters,
        optimizer_converged=converged,
        optimizer_message=message,
        development_log_likelihood=log_likelihood,
        development_bic=bic,
        parameter_count=parameter_count,
    )


def _confirmation_components(
    fit: FittedEventProcessV40,
    development: USGSCatalogSnapshotV40,
    confirmation: USGSCatalogSnapshotV40,
) -> tuple[float, list[float], float]:
    start = 0.0
    duration = _duration_days(confirmation.query)
    confirmation_events = _relative_times(confirmation)
    history = np.asarray(
        [
            (event.origin_time - confirmation.query.start).total_seconds() / 86400.0
            for event in development.events
        ],
        dtype=float,
    )
    if fit.family == "homogeneous_poisson":
        rate = fit.parameters["rate_per_day"]
        likelihood = _poisson_log_likelihood(
            rate, len(confirmation_events), duration
        )
        rescaled = np.diff(
            np.concatenate(([start], confirmation_events))
        ) * rate
        return likelihood, [float(value) for value in rescaled], rate * duration
    if fit.family == "weibull_renewal":
        return _weibull_components(
            confirmation_events,
            start=start,
            end=duration,
            history=history,
            shape=fit.parameters["shape"],
            scale=fit.parameters["scale_days"],
        )
    return _hawkes_components(
        confirmation_events,
        start=start,
        end=duration,
        history=history,
        background=fit.parameters["background_rate_per_day"],
        branching=fit.parameters["branching_ratio"],
        decay=fit.parameters["decay_rate_per_day"],
    )


def evaluate_event_process_v40(
    spec: UnseenEventProcessSpecV40,
    candidate: EventProcessCandidateV40,
    fit: FittedEventProcessV40,
    development: USGSCatalogSnapshotV40,
    confirmation: USGSCatalogSnapshotV40,
) -> EventProcessPrivateEvaluationV40:
    spec.assert_sealed()
    candidate.assert_sealed()
    fit.assert_sealed()
    development.assert_sealed()
    confirmation.assert_sealed()
    if fit.candidate_hash != candidate.candidate_hash:
        raise ValueError("fit is bound to another candidate")
    if fit.development_snapshot_hash != development.snapshot_hash:
        raise ValueError("fit is bound to another development snapshot")
    development_duration = _duration_days(development.query)
    baseline_rate = len(development.events) / development_duration
    baseline_development_ll = _poisson_log_likelihood(
        baseline_rate, len(development.events), development_duration
    )
    baseline_development_bic = (
        math.log(len(development.events)) - 2.0 * baseline_development_ll
    )
    confirmation_duration = _duration_days(confirmation.query)
    baseline_confirmation_ll = _poisson_log_likelihood(
        baseline_rate, len(confirmation.events), confirmation_duration
    )
    candidate_confirmation_ll, rescaled, compensator = _confirmation_components(
        fit, development, confirmation
    )
    count = len(confirmation.events)
    lift = (candidate_confirmation_ll - baseline_confirmation_ll) / count
    ks_pvalue = (
        float(kstest(np.asarray(rescaled, dtype=float), "expon").pvalue)
        if rescaled and all(math.isfinite(value) and value >= 0 for value in rescaled)
        else 0.0
    )
    compensator_error = abs(compensator - count) / count
    branching = fit.parameters.get("branching_ratio", 0.0)
    gates = {
        "data_contract": (
            spec.minimum_events_per_phase
            <= len(development.events)
            <= spec.maximum_events_per_phase
            and spec.minimum_events_per_phase
            <= len(confirmation.events)
            <= spec.maximum_events_per_phase
        ),
        "optimizer_converged": fit.optimizer_converged,
        "development_bic_not_worse": (
            fit.development_bic <= baseline_development_bic
        ),
        "confirmation_log_score_lift": (
            lift >= spec.minimum_log_score_lift_nat_per_event
        ),
        "time_rescaling_calibration": (
            ks_pvalue >= spec.minimum_time_rescaling_ks_pvalue
        ),
        "compensator_count_calibration": (
            compensator_error <= spec.maximum_compensator_count_relative_error
        ),
        "hawkes_stationarity": (
            fit.family != "exponential_hawkes"
            or branching < spec.maximum_hawkes_branching_ratio
        ),
    }
    return EventProcessPrivateEvaluationV40.seal(
        evaluator_epoch=spec.evaluator_epoch,
        candidate_hash=candidate.candidate_hash,
        fit_hash=fit.fit_hash,
        confirmation_snapshot_hash=confirmation.snapshot_hash,
        baseline_development_bic=baseline_development_bic,
        candidate_development_bic=fit.development_bic,
        baseline_confirmation_log_likelihood=baseline_confirmation_ll,
        candidate_confirmation_log_likelihood=candidate_confirmation_ll,
        confirmation_log_score_lift_nat_per_event=lift,
        time_rescaling_ks_pvalue=ks_pvalue,
        compensator_total=compensator,
        observed_confirmation_count=count,
        compensator_count_relative_error=compensator_error,
        gates=gates,
        decision="qualified" if all(gates.values()) else "rejected",
    )


def _artifact_refs(store: RunStore, kind: str) -> list[ArtifactRef]:
    refs: list[ArtifactRef] = []
    for line in store.event_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event["event_type"] != "artifact_committed":
            continue
        ref = ArtifactRef.model_validate(event["payload"])
        if ref.kind == kind:
            refs.append(ref)
    return refs


def _load_one(store: RunStore, kind: str, model):
    refs = _artifact_refs(store, kind)
    if len(refs) != 1:
        raise RuntimeError(f"unseen-task run needs exactly one {kind}")
    return model.model_validate(store.load_artifact(refs[0]))


def _load_candidate_from_draft(graph: GraphLoopStoreV40) -> EventProcessCandidateV40:
    existing = _artifact_refs(graph.store, "event_process_candidate_v40")
    if existing:
        if len(existing) != 1:
            raise RuntimeError("unseen-task run contains duplicate candidates")
        return EventProcessCandidateV40.model_validate(
            graph.store.load_artifact(existing[0])
        )
    drafts = _artifact_refs(graph.store, "codex_frontier_draft_v40")
    if len(drafts) != 1:
        raise RuntimeError("unseen-task run needs one Codex frontier draft")
    payload = graph.store.load_artifact(drafts[0])
    raw_candidate = json.loads(str(payload["draft"]))
    candidate = EventProcessCandidateV40.seal(**raw_candidate)
    graph.put_output("event_process_candidate_v40", candidate)
    return candidate


def _public_node_purpose(summary: EventProcessPublicSummaryV40) -> str:
    public = summary.model_dump(mode="json")
    public.pop("summary_hash")
    schema = {
        "family": [
            "homogeneous_poisson",
            "weibull_renewal",
            "exponential_hawkes",
        ],
        "hawkes_branching_initial": "number strictly between 0 and 0.95",
        "hawkes_decay_days_initial": "number from 0.1 to 60",
        "rationale": "string of at least 10 characters",
        "expected_failure_modes": "nonempty string list",
    }
    return (
        "Choose one continuous-time point-process family from the allowed schema. "
        "The development summary is public evidence; do not claim confirmation. "
        "Set draft to one strict JSON object with exactly these fields and no markdown. "
        f"PUBLIC_SUMMARY={json.dumps(public, sort_keys=True)} "
        f"CANDIDATE_SCHEMA={json.dumps(schema, sort_keys=True)}"
    )


def _new_graph(
    output_root: Path,
    spec: UnseenEventProcessSpecV40,
    development: USGSCatalogSnapshotV40,
    confirmation: USGSCatalogSnapshotV40,
    summary: EventProcessPublicSummaryV40,
    development_raw: USGSCatalogRawV40,
    confirmation_raw: USGSCatalogRawV40,
) -> GraphLoopStoreV40:
    contract = GraphLoopContractV40.seal(
        graph_id=spec.experiment_id,
        layer="modeling",
        evaluator_epoch=spec.evaluator_epoch,
        objective="select and privately verify an unseen earthquake event-time model",
        max_nodes=5,
        max_outcomes=5,
        max_failures=2,
        max_promotions=1,
        created_at=spec.created_at,
    )
    graph = GraphLoopStoreV40(output_root, contract)
    graph.put_output("unseen_event_process_spec_v40", spec)
    graph.put_output("usgs_catalog_raw_development_v40", development_raw)
    graph.put_output("usgs_catalog_raw_confirmation_v40", confirmation_raw)
    graph.put_output("usgs_catalog_snapshot_development_v40", development)
    graph.put_output("usgs_catalog_snapshot_confirmation_v40", confirmation)
    graph.put_output("event_process_public_summary_v40", summary)
    candidate = GraphNodeV40.seal(
        node_id="select_event_process_candidate",
        layer="modeling",
        node_kind="model_candidate",
        executor="model",
        created_by="harness",
        artifact_hash=summary.summary_hash,
        purpose=_public_node_purpose(summary),
        created_at=spec.created_at,
    )
    fit = GraphNodeV40.seal(
        node_id="fit_event_process_candidate",
        layer="modeling",
        node_kind="execution",
        executor="harness",
        created_by="harness",
        artifact_hash=sha256_value(
            {"executor": "continuous-event-process-mle-v1", "spec": spec.spec_hash}
        ),
        purpose="fit the frozen model family on development event times",
        created_at=spec.created_at,
    )
    evaluate = GraphNodeV40.seal(
        node_id="private_confirmation_evaluation",
        layer="modeling",
        node_kind="evaluation",
        executor="verifier",
        created_by="harness",
        artifact_hash=sha256_value(
            {
                "evaluator_epoch": spec.evaluator_epoch,
                "confirmation_snapshot": confirmation.snapshot_hash,
                "spec": spec.spec_hash,
            }
        ),
        purpose="apply the frozen private future-year point-process gates",
        created_at=spec.created_at,
    )
    for node in (candidate, fit, evaluate):
        graph.add_node(node)
    graph.add_edge(
        GraphEdgeV40.seal(
            edge_id="candidate_requires_fit",
            layer="modeling",
            source_node_hash=candidate.node_hash,
            target_node_hash=fit.node_hash,
            relation="requires_success",
            rationale="only a sealed model candidate may be fitted",
            created_at=spec.created_at,
        )
    )
    graph.add_edge(
        GraphEdgeV40.seal(
            edge_id="fit_requires_private_evaluation",
            layer="modeling",
            source_node_hash=fit.node_hash,
            target_node_hash=evaluate.node_hash,
            relation="requires_success",
            rationale="private evaluation requires a successful frozen fit",
            created_at=spec.created_at,
        )
    )
    graph.add_edge(
        GraphEdgeV40.seal(
            edge_id="candidate_evaluated_by_private_confirmation",
            layer="modeling",
            source_node_hash=candidate.node_hash,
            target_node_hash=evaluate.node_hash,
            relation="evaluated_by",
            rationale="candidate promotion is bound to the private evaluator",
            created_at=spec.created_at,
        )
    )
    return graph


def _check_event_counts(
    spec: UnseenEventProcessSpecV40,
    development: USGSCatalogSnapshotV40,
    confirmation: USGSCatalogSnapshotV40,
) -> None:
    for snapshot in (development, confirmation):
        count = len(snapshot.events)
        if not spec.minimum_events_per_phase <= count <= spec.maximum_events_per_phase:
            raise ValueError(
                f"{snapshot.query.phase} event count {count} violates frozen bounds"
            )


def run_unseen_event_process_experiment_v40(
    output_root: str | Path,
    spec: UnseenEventProcessSpecV40,
    transport: FrontierProposalTransportV40,
    *,
    fetch_bytes: FetchBytes | None = None,
    retrieved_at: datetime | None = None,
) -> UnseenEventProcessOutcomeV40:
    spec.assert_sealed()
    root = Path(output_root).resolve()
    run_directory = root / spec.experiment_id
    if run_directory.is_dir():
        graph = GraphLoopStoreV40.open_existing(run_directory)
        frozen_spec = _load_one(
            graph.store, "unseen_event_process_spec_v40", UnseenEventProcessSpecV40
        )
        if frozen_spec != spec:
            raise ValueError("resumed unseen-task run uses another frozen spec")
        development = _load_one(
            graph.store,
            "usgs_catalog_snapshot_development_v40",
            USGSCatalogSnapshotV40,
        )
        confirmation = _load_one(
            graph.store,
            "usgs_catalog_snapshot_confirmation_v40",
            USGSCatalogSnapshotV40,
        )
    else:
        fetch_time = retrieved_at or datetime.now(timezone.utc)
        development_raw, development = fetch_usgs_catalog_snapshot_v40(
            spec.development_query,
            fetch_bytes=fetch_bytes,
            retrieved_at=fetch_time,
        )
        confirmation_raw, confirmation = fetch_usgs_catalog_snapshot_v40(
            spec.confirmation_query,
            fetch_bytes=fetch_bytes,
            retrieved_at=fetch_time,
        )
        _check_event_counts(spec, development, confirmation)
        summary = summarize_development_snapshot_v40(development)
        graph = _new_graph(
            root,
            spec,
            development,
            confirmation,
            summary,
            development_raw,
            confirmation_raw,
        )

    reports = _artifact_refs(graph.store, "unseen_event_process_report_v40")
    if reports:
        if len(reports) != 1:
            raise RuntimeError("unseen-task run contains duplicate reports")
        report = UnseenEventProcessReportV40.model_validate(
            graph.store.load_artifact(reports[0])
        )
        candidate = _load_one(
            graph.store, "event_process_candidate_v40", EventProcessCandidateV40
        )
        fit = _load_one(
            graph.store, "fitted_event_process_v40", FittedEventProcessV40
        )
        evaluation = _load_one(
            graph.store,
            "event_process_private_evaluation_v40",
            EventProcessPrivateEvaluationV40,
        )
        return UnseenEventProcessOutcomeV40(
            graph, report, candidate, fit, evaluation
        )

    state = graph.project_state()
    candidate_node = next(node for node in state.nodes if node.node_kind == "model_candidate")
    fit_node = next(node for node in state.nodes if node.node_kind == "execution")
    evaluator_node = next(node for node in state.nodes if node.node_kind == "evaluation")
    if state.snapshot.node_statuses[candidate_node.node_hash] == "pending":
        driver = CodexFrontierDriverV40(transport)
        driver_outcome = driver.run_once(
            graph,
            receipt_id="unseen_event_process_candidate_call",
            created_at=spec.created_at,
        )
        if driver_outcome.receipt.status != "executed":
            raise RuntimeError(
                "unseen-task Codex candidate call did not execute: "
                + driver_outcome.receipt.status
            )
    candidate = _load_candidate_from_draft(graph)

    state = graph.project_state()
    if state.snapshot.node_statuses[fit_node.node_hash] == "pending":
        fit = fit_event_process_v40(candidate, development)
        fit_ref = graph.put_output("fitted_event_process_v40", fit)
        graph.record_outcome(
            fit_node.node_hash,
            actor="harness",
            status="succeeded" if fit.optimizer_converged else "failed",
            output_artifacts=[fit_ref],
            summary="frozen continuous event-process maximum likelihood fit",
            outcome_id="fit_unseen_event_process_candidate",
            started_at=spec.created_at,
            finished_at=spec.created_at,
        )
        if not fit.optimizer_converged:
            raise RuntimeError("unseen-task optimizer did not converge")
    else:
        fit = _load_one(
            graph.store, "fitted_event_process_v40", FittedEventProcessV40
        )

    state = graph.project_state()
    if state.snapshot.node_statuses[evaluator_node.node_hash] == "pending":
        evaluation = evaluate_event_process_v40(
            spec, candidate, fit, development, confirmation
        )
        evaluation_ref = graph.put_output(
            "event_process_private_evaluation_v40", evaluation
        )
        graph.record_outcome(
            evaluator_node.node_hash,
            actor="verifier",
            status="succeeded",
            output_artifacts=[evaluation_ref],
            summary="independent future-year event-process evaluation completed",
            outcome_id="evaluate_unseen_event_process_candidate",
            started_at=spec.created_at,
            finished_at=spec.created_at,
        )
    else:
        evaluation = _load_one(
            graph.store,
            "event_process_private_evaluation_v40",
            EventProcessPrivateEvaluationV40,
        )

    state = graph.project_state()
    if state.promotions:
        if len(state.promotions) != 1:
            raise RuntimeError("unseen-task run contains duplicate promotions")
        promotion = state.promotions[0]
    else:
        promotion = graph.decide_promotion(
            candidate_node.node_hash,
            evaluator_node.node_hash,
            evidence_node_hashes=[fit_node.node_hash, evaluator_node.node_hash],
            decision=evaluation.decision,
            authority="verifier",
            independent_gate_passed=True,
            private_scientific_gate_passed=(evaluation.decision == "qualified"),
            scope="qualified@usgs_catalog_shadow",
            promotion_id="promote_unseen_event_process_candidate",
            decided_at=spec.created_at,
        )
    replay_verified = graph.verify()
    report = UnseenEventProcessReportV40.seal(
        experiment_id=spec.experiment_id,
        spec_hash=spec.spec_hash,
        development_snapshot_hash=development.snapshot_hash,
        confirmation_snapshot_hash=confirmation.snapshot_hash,
        candidate_hash=candidate.candidate_hash,
        fit_hash=fit.fit_hash,
        evaluation_hash=evaluation.evaluation_hash,
        promotion_receipt_hash=promotion.receipt_hash,
        graph_snapshot_hash=graph.project_state().snapshot.snapshot_hash,
        transport=transport.transport_name,
        decision=evaluation.decision,
        graph_replay_verified=replay_verified,
        created_at=spec.created_at,
    )
    graph.put_output("unseen_event_process_report_v40", report)
    outcome = UnseenEventProcessOutcomeV40(
        graph, report, candidate, fit, evaluation
    )
    if not verify_unseen_event_process_experiment_v40(outcome, spec):
        raise RuntimeError("unseen-task run failed independent replay")
    return outcome


def verify_unseen_event_process_experiment_v40(
    outcome: UnseenEventProcessOutcomeV40,
    spec: UnseenEventProcessSpecV40,
) -> bool:
    try:
        spec.assert_sealed()
        outcome.report.assert_sealed()
        outcome.candidate.assert_sealed()
        outcome.fit.assert_sealed()
        outcome.evaluation.assert_sealed()
        if not outcome.graph.verify():
            return False
        store = outcome.graph.store
        frozen_spec = _load_one(
            store, "unseen_event_process_spec_v40", UnseenEventProcessSpecV40
        )
        development = _load_one(
            store,
            "usgs_catalog_snapshot_development_v40",
            USGSCatalogSnapshotV40,
        )
        confirmation = _load_one(
            store,
            "usgs_catalog_snapshot_confirmation_v40",
            USGSCatalogSnapshotV40,
        )
        development_raw = _load_one(
            store, "usgs_catalog_raw_development_v40", USGSCatalogRawV40
        )
        confirmation_raw = _load_one(
            store, "usgs_catalog_raw_confirmation_v40", USGSCatalogRawV40
        )
        _, replay_development = parse_usgs_catalog_response_v40(
            spec.development_query,
            development_raw.response_body.encode("utf-8"),
            retrieved_at=development.retrieved_at,
        )
        _, replay_confirmation = parse_usgs_catalog_response_v40(
            spec.confirmation_query,
            confirmation_raw.response_body.encode("utf-8"),
            retrieved_at=confirmation.retrieved_at,
        )
        replay_evaluation = evaluate_event_process_v40(
            spec,
            outcome.candidate,
            outcome.fit,
            development,
            confirmation,
        )
        state = outcome.graph.project_state()
        promotion: PromotionReceiptV40 = state.promotions[0]
        return bool(
            frozen_spec == spec
            and development_raw.query_hash == spec.development_query.query_hash
            and confirmation_raw.query_hash == spec.confirmation_query.query_hash
            and replay_development == development
            and replay_confirmation == confirmation
            and replay_evaluation == outcome.evaluation
            and outcome.report.spec_hash == spec.spec_hash
            and outcome.report.development_snapshot_hash == development.snapshot_hash
            and outcome.report.confirmation_snapshot_hash == confirmation.snapshot_hash
            and outcome.report.candidate_hash == outcome.candidate.candidate_hash
            and outcome.report.fit_hash == outcome.fit.fit_hash
            and outcome.report.evaluation_hash == outcome.evaluation.evaluation_hash
            and outcome.report.promotion_receipt_hash == promotion.receipt_hash
            and outcome.report.graph_snapshot_hash == state.snapshot.snapshot_hash
            and outcome.report.decision == promotion.decision
            and outcome.report.graph_replay_verified
        )
    except (
        IndexError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False
