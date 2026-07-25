from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Union

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel

from .schemas import Identifier, Sha256, _assert_timezone


FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFinite = Annotated[float, Field(ge=0, allow_inf_nan=False)]


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class TimeSeriesPoint(StrictModel):
    timestamp: datetime
    value: NonNegativeFinite

    @model_validator(mode="after")
    def validate_point(self) -> "TimeSeriesPoint":
        _assert_timezone(self.timestamp, "timestamp")
        return self


class TimeSeriesDataContract(StrictModel):
    """Frozen interpretation of one historical univariate data source."""

    schema_version: Literal["2.1"] = "2.1"
    dataset_id: Identifier
    mission_spec_hash: Sha256
    source_kind: Literal["fixture", "local_file", "official_api"]
    source_ref: Annotated[str, Field(min_length=1, max_length=512)]
    timestamp_column: Identifier = "timestamp"
    value_column: Identifier = "value"
    frequency: Literal["hourly", "daily", "weekly", "monthly", "quarterly"]
    value_unit: Annotated[str, Field(min_length=1, max_length=80)]
    minimum_points: Annotated[int, Field(ge=12, le=1_000_000)] = 24
    missing_policy: Literal["reject"] = "reject"
    nonnegative_required: Literal[True] = True
    intended_use: Literal["retrospective_forecast_validation"] = (
        "retrospective_forecast_validation"
    )
    created_at: datetime
    data_contract_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "TimeSeriesDataContract":
        _assert_timezone(self.created_at, "created_at")
        if self.timestamp_column == self.value_column:
            raise ValueError("timestamp and value columns must be distinct")
        if self.data_contract_hash and self.data_contract_hash != self.content_hash():
            raise ValueError("data_contract_hash does not match contract content")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "data_contract_hash")

    def assert_sealed(self) -> None:
        if not self.data_contract_hash or self.data_contract_hash != self.content_hash():
            raise ValueError("time-series data contract is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "TimeSeriesDataContract":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"data_contract_hash"}),
            data_contract_hash=draft.content_hash(),
        )


class TimeSeriesSnapshot(StrictModel):
    """Validated, immutable observations; raw source bytes remain separately hashed."""

    schema_version: Literal["2.1"] = "2.1"
    snapshot_id: Identifier
    data_contract_hash: Sha256
    source_content_hash: Sha256
    points: list[TimeSeriesPoint] = Field(min_length=12, max_length=1_000_000)
    quality_status: Literal["passed"] = "passed"
    quality_checks: list[Literal[
        "utf8_decodable",
        "required_columns_exact",
        "no_missing_values",
        "finite_nonnegative_values",
        "timestamps_strictly_increasing",
        "minimum_length_met",
        "source_identity_verified",
        "frequency_complete",
    ]] = Field(min_length=6, max_length=8)
    collected_at: datetime
    snapshot_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_snapshot(self) -> "TimeSeriesSnapshot":
        _assert_timezone(self.collected_at, "collected_at")
        timestamps = [point.timestamp for point in self.points]
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError("timestamps must be unique and strictly increasing")
        required_checks = {
            "utf8_decodable",
            "required_columns_exact",
            "no_missing_values",
            "finite_nonnegative_values",
            "timestamps_strictly_increasing",
            "minimum_length_met",
        }
        if not required_checks.issubset(set(self.quality_checks)):
            raise ValueError("all six base data-quality checks are required")
        if len(self.quality_checks) != len(set(self.quality_checks)):
            raise ValueError("data-quality checks must be distinct")
        if self.snapshot_hash and self.snapshot_hash != self.content_hash():
            raise ValueError("snapshot_hash does not match snapshot content")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "snapshot_hash")

    def assert_sealed(self) -> None:
        if not self.snapshot_hash or self.snapshot_hash != self.content_hash():
            raise ValueError("time-series snapshot is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "TimeSeriesSnapshot":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"snapshot_hash"}),
            snapshot_hash=draft.content_hash(),
        )


class ForecastCandidateSpec(StrictModel):
    schema_version: Literal["2.1"] = "2.1"
    candidate_id: Identifier
    data_contract_hash: Sha256
    family: Literal["last_value", "mean_level", "linear_trend", "seasonal_naive"]
    seasonal_period: Annotated[int, Field(ge=2, le=365)] | None = None
    assumptions: list[Annotated[str, Field(min_length=3)]] = Field(min_length=1)
    role: Literal["baseline", "challenger"]
    candidate_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_candidate(self) -> "ForecastCandidateSpec":
        if (self.family == "seasonal_naive") != (self.seasonal_period is not None):
            raise ValueError("only seasonal_naive requires seasonal_period")
        if self.family == "last_value" and self.role != "baseline":
            raise ValueError("last_value is the frozen baseline")
        if self.family != "last_value" and self.role != "challenger":
            raise ValueError("non-baseline families must be challengers")
        if self.candidate_hash and self.candidate_hash != self.content_hash():
            raise ValueError("candidate_hash does not match candidate content")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "candidate_hash")

    def assert_sealed(self) -> None:
        if not self.candidate_hash or self.candidate_hash != self.content_hash():
            raise ValueError("forecast candidate is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ForecastCandidateSpec":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"candidate_hash"}),
            candidate_hash=draft.content_hash(),
        )


class ForecastCandidateSpecV22(StrictModel):
    """Versioned evolution dialect; V2.1 hash semantics remain untouched."""

    schema_version: Literal["2.2"] = "2.2"
    candidate_id: Identifier
    data_contract_hash: Sha256
    family: Literal[
        "last_value",
        "seasonal_naive",
        "window_mean",
        "window_linear_trend",
        "exponential_smoothing",
    ]
    seasonal_period: Annotated[int, Field(ge=2, le=365)] | None = None
    window_length: Annotated[int, Field(ge=4, le=10_000)] | None = None
    smoothing_alpha: Annotated[
        float, Field(gt=0, lt=1, allow_inf_nan=False)
    ] | None = None
    assumptions: list[Annotated[str, Field(min_length=3)]] = Field(min_length=1)
    role: Literal["baseline", "challenger"]
    candidate_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_candidate(self) -> "ForecastCandidateSpecV22":
        if (self.family == "seasonal_naive") != (self.seasonal_period is not None):
            raise ValueError("only seasonal_naive requires seasonal_period")
        if (self.family in {"window_mean", "window_linear_trend"}) != (
            self.window_length is not None
        ):
            raise ValueError("only windowed families require window_length")
        if (self.family == "exponential_smoothing") != (
            self.smoothing_alpha is not None
        ):
            raise ValueError("only exponential_smoothing requires smoothing_alpha")
        if self.family == "last_value" and self.role != "baseline":
            raise ValueError("last_value is the frozen baseline")
        if self.family != "last_value" and self.role != "challenger":
            raise ValueError("non-baseline families must be challengers")
        if self.candidate_hash and self.candidate_hash != self.content_hash():
            raise ValueError("candidate_hash does not match candidate content")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "candidate_hash")

    def assert_sealed(self) -> None:
        if not self.candidate_hash or self.candidate_hash != self.content_hash():
            raise ValueError("forecast candidate V2.2 is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ForecastCandidateSpecV22":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"candidate_hash"}),
            candidate_hash=draft.content_hash(),
        )


ForecastCandidate = Annotated[
    Union[ForecastCandidateSpec, ForecastCandidateSpecV22],
    Field(discriminator="schema_version"),
]


class ForecastCandidatePortfolio(StrictModel):
    schema_version: Literal["2.1"] = "2.1"
    portfolio_id: Identifier
    data_contract_hash: Sha256
    candidates: list[ForecastCandidate] = Field(min_length=3, max_length=16)
    generator_id: Literal[
        "deterministic_forecast_generator_v1",
        "failure_evolved_forecast_generator_v1",
    ] = "deterministic_forecast_generator_v1"
    generated_at: datetime
    portfolio_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_portfolio(self) -> "ForecastCandidatePortfolio":
        _assert_timezone(self.generated_at, "generated_at")
        ids = [candidate.candidate_id for candidate in self.candidates]
        families = [candidate.family for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate ids must be unique")
        if len(set(families)) < 3:
            raise ValueError("portfolio needs at least three distinct model families")
        if families.count("last_value") != 1:
            raise ValueError("portfolio needs exactly one last_value baseline")
        for candidate in self.candidates:
            candidate.assert_sealed()
            if candidate.data_contract_hash != self.data_contract_hash:
                raise ValueError("candidate is bound to another data contract")
        if self.portfolio_hash and self.portfolio_hash != self.content_hash():
            raise ValueError("portfolio_hash does not match portfolio content")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "portfolio_hash")

    def assert_sealed(self) -> None:
        if not self.portfolio_hash or self.portfolio_hash != self.content_hash():
            raise ValueError("candidate portfolio is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ForecastCandidatePortfolio":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"portfolio_hash"}),
            portfolio_hash=draft.content_hash(),
        )


class TemporalValidationSpec(StrictModel):
    schema_version: Literal["2.1"] = "2.1"
    validation_id: Identifier
    data_snapshot_hash: Sha256
    portfolio_hash: Sha256
    strategy: Literal["expanding_window_rolling_origin"] = (
        "expanding_window_rolling_origin"
    )
    horizon: Literal[1] = 1
    holdout_points: Annotated[int, Field(ge=4, le=10_000)]
    minimum_training_points: Annotated[int, Field(ge=8, le=1_000_000)]
    minimum_calibration_residuals: Annotated[int, Field(ge=4, le=100_000)]
    interval_level: Annotated[float, Field(gt=0.5, lt=1, allow_inf_nan=False)]
    minimum_empirical_coverage: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ]
    max_mae_ratio_to_last_value: Annotated[
        float, Field(gt=0, allow_inf_nan=False)
    ]
    required_passing_candidates: Annotated[int, Field(ge=2, le=16)]
    frozen_at: datetime
    validation_spec_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "TemporalValidationSpec":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.minimum_calibration_residuals >= self.minimum_training_points:
            raise ValueError("calibration residual count must be smaller than training length")
        if self.validation_spec_hash and self.validation_spec_hash != self.content_hash():
            raise ValueError("validation_spec_hash does not match validation spec")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "validation_spec_hash")

    def assert_sealed(self) -> None:
        if not self.validation_spec_hash or self.validation_spec_hash != self.content_hash():
            raise ValueError("temporal validation spec is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "TemporalValidationSpec":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"validation_spec_hash"}),
            validation_spec_hash=draft.content_hash(),
        )


class ForecastRecord(StrictModel):
    origin_index: Annotated[int, Field(ge=1)]
    target_timestamp: datetime
    actual: NonNegativeFinite
    point_prediction: NonNegativeFinite
    interval_lower: NonNegativeFinite
    interval_upper: NonNegativeFinite
    calibration_residual_count: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def validate_record(self) -> "ForecastRecord":
        _assert_timezone(self.target_timestamp, "target_timestamp")
        if not self.interval_lower <= self.point_prediction <= self.interval_upper:
            raise ValueError("point prediction must lie inside its interval")
        return self


class CandidateValidationResult(StrictModel):
    candidate_id: Identifier
    candidate_hash: Sha256
    records: list[ForecastRecord] = Field(min_length=4)
    mae: NonNegativeFinite
    rmse: NonNegativeFinite
    mean_bias: FiniteNumber
    empirical_coverage: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    mean_interval_width: NonNegativeFinite
    mae_ratio_to_last_value: NonNegativeFinite | None
    status: Literal["passed", "failed"]
    reason_codes: list[Literal[
        "mae_gate_failed",
        "coverage_gate_failed",
        "calibration_unavailable",
    ]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status(self) -> "CandidateValidationResult":
        if (self.status == "passed") == bool(self.reason_codes):
            raise ValueError("passed results need no reasons; failed results need reasons")
        return self


class ForecastValidationReport(StrictModel):
    schema_version: Literal["2.1"] = "2.1"
    report_id: Identifier
    data_snapshot_hash: Sha256
    portfolio_hash: Sha256
    validation_spec_hash: Sha256
    evaluator_id: Literal["independent_rolling_origin_evaluator_v1"] = (
        "independent_rolling_origin_evaluator_v1"
    )
    results: list[CandidateValidationResult] = Field(min_length=3, max_length=16)
    passing_candidate_ids: list[Identifier]
    status: Literal["sufficient", "needs_evidence"]
    evaluated_at: datetime
    report_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "ForecastValidationReport":
        _assert_timezone(self.evaluated_at, "evaluated_at")
        ids = [result.candidate_id for result in self.results]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate validation results must be unique")
        expected = [result.candidate_id for result in self.results if result.status == "passed"]
        if self.passing_candidate_ids != expected:
            raise ValueError("passing candidate ids do not match result statuses")
        if self.report_hash and self.report_hash != self.content_hash():
            raise ValueError("report_hash does not match validation report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "report_hash")

    def assert_sealed(self) -> None:
        if not self.report_hash or self.report_hash != self.content_hash():
            raise ValueError("forecast validation report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ForecastValidationReport":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"report_hash"}),
            report_hash=draft.content_hash(),
        )


class CapacityDecisionSpec(StrictModel):
    schema_version: Literal["2.1"] = "2.1"
    decision_id: Identifier
    validation_report_hash: Sha256
    shortage_cost: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    overage_cost: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    maximum_capacity: Annotated[int, Field(ge=1, le=1_000_000)]
    max_mean_holdout_regret: NonNegativeFinite
    minimum_passing_models: Annotated[int, Field(ge=2, le=16)]
    require_exact_action_agreement: Literal[True] = True
    frozen_at: datetime
    decision_spec_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_decision_spec(self) -> "CapacityDecisionSpec":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.decision_spec_hash and self.decision_spec_hash != self.content_hash():
            raise ValueError("decision_spec_hash does not match decision spec")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "decision_spec_hash")

    def assert_sealed(self) -> None:
        if not self.decision_spec_hash or self.decision_spec_hash != self.content_hash():
            raise ValueError("capacity decision spec is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "CapacityDecisionSpec":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"decision_spec_hash"}),
            decision_spec_hash=draft.content_hash(),
        )


class CandidateCapacityDecision(StrictModel):
    candidate_id: Identifier
    next_point_forecast: NonNegativeFinite
    next_interval_lower: NonNegativeFinite
    next_interval_upper: NonNegativeFinite
    recommended_capacity: Annotated[int, Field(ge=0)]
    mean_holdout_regret: NonNegativeFinite
    maximum_holdout_regret: NonNegativeFinite
    status: Literal["passed", "failed"]


class DecisionStabilityReport(StrictModel):
    schema_version: Literal["2.1"] = "2.1"
    report_id: Identifier
    data_snapshot_hash: Sha256
    validation_report_hash: Sha256
    decision_spec_hash: Sha256
    candidate_decisions: list[CandidateCapacityDecision]
    status: Literal["decision_eligible", "needs_evidence"]
    reason_codes: list[Literal[
        "forecast_validation_insufficient",
        "too_few_passing_models",
        "holdout_regret_exceeded",
        "candidate_actions_disagree",
    ]]
    accreditation_status: Literal["not_accredited"] = "not_accredited"
    permissible_uses: list[Literal[
        "synthetic_fixture_analysis",
        "retrospective_shadow_analysis",
    ]] = Field(min_length=1, max_length=1)
    real_world_action_authorized: Literal[False] = False
    evaluated_at: datetime
    decision_report_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_decision_report(self) -> "DecisionStabilityReport":
        _assert_timezone(self.evaluated_at, "evaluated_at")
        if (self.status == "decision_eligible") == bool(self.reason_codes):
            raise ValueError("eligible reports need no reasons; abstentions need reasons")
        if self.decision_report_hash and self.decision_report_hash != self.content_hash():
            raise ValueError("decision_report_hash does not match decision report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "decision_report_hash")

    def assert_sealed(self) -> None:
        if not self.decision_report_hash or self.decision_report_hash != self.content_hash():
            raise ValueError("decision stability report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DecisionStabilityReport":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"decision_report_hash"}),
            decision_report_hash=draft.content_hash(),
        )


class EmpiricalRunManifest(StrictModel):
    schema_version: Literal["2.1"] = "2.1"
    run_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")]
    artifact_refs: list[ArtifactRef] = Field(min_length=7)
    terminal_status: Literal["decision_eligible", "needs_evidence"]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "EmpiricalRunManifest":
        _assert_timezone(self.created_at, "created_at")
        if len({(ref.kind, ref.sha256) for ref in self.artifact_refs}) != len(
            self.artifact_refs
        ):
            raise ValueError("manifest artifact references must be unique")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match manifest")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("empirical run manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EmpiricalRunManifest":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )
