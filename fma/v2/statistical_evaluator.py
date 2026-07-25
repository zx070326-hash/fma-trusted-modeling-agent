from __future__ import annotations

from datetime import datetime, timezone
from random import Random
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel

from .empirical_schemas import ForecastValidationReport
from .schemas import Identifier, Sha256, _assert_timezone


FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFinite = Annotated[float, Field(ge=0, allow_inf_nan=False)]


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class MovingBlockBootstrapSpec(StrictModel):
    """Frozen paired time-series resampling plan; not an IID bootstrap."""

    schema_version: Literal["2.1"] = "2.1"
    bootstrap_id: Identifier
    validation_report_hash: Sha256
    baseline_candidate_id: Identifier
    block_length: Annotated[int, Field(ge=1, le=10_000)]
    replicates_per_seed: Annotated[int, Field(ge=100, le=100_000)]
    seeds: list[Annotated[int, Field(ge=0, le=2_147_483_647)]] = Field(
        min_length=3, max_length=16
    )
    confidence_level: Annotated[float, Field(gt=0.8, lt=1, allow_inf_nan=False)]
    frozen_at: datetime
    bootstrap_spec_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "MovingBlockBootstrapSpec":
        _assert_timezone(self.frozen_at, "frozen_at")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("bootstrap seeds must be unique")
        if self.bootstrap_spec_hash and self.bootstrap_spec_hash != self.content_hash():
            raise ValueError("bootstrap_spec_hash does not match bootstrap spec")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "bootstrap_spec_hash")

    def assert_sealed(self) -> None:
        if not self.bootstrap_spec_hash or self.bootstrap_spec_hash != self.content_hash():
            raise ValueError("moving-block bootstrap spec is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "MovingBlockBootstrapSpec":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"bootstrap_spec_hash"}),
            bootstrap_spec_hash=draft.content_hash(),
        )


class CandidateSkillInterval(StrictModel):
    candidate_id: Identifier
    holdout_points: Annotated[int, Field(ge=4)]
    mean_mae_improvement: FiniteNumber
    relative_mae_improvement: FiniteNumber | None
    confidence_lower: FiniteNumber
    confidence_upper: FiniteNumber
    bootstrap_standard_error: NonNegativeFinite
    probability_improvement_positive: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ]
    interpretation: Literal[
        "evidence_better_than_baseline",
        "inconclusive_vs_baseline",
        "evidence_worse_than_baseline",
    ]

    @model_validator(mode="after")
    def validate_interval(self) -> "CandidateSkillInterval":
        if self.confidence_lower > self.confidence_upper:
            raise ValueError("bootstrap confidence interval is reversed")
        return self


class ForecastSkillUncertaintyReport(StrictModel):
    schema_version: Literal["2.1"] = "2.1"
    report_id: Identifier
    validation_report_hash: Sha256
    bootstrap_spec_hash: Sha256
    method: Literal["paired_circular_moving_block_bootstrap"] = (
        "paired_circular_moving_block_bootstrap"
    )
    results: list[CandidateSkillInterval] = Field(min_length=1, max_length=15)
    evaluated_at: datetime
    report_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "ForecastSkillUncertaintyReport":
        _assert_timezone(self.evaluated_at, "evaluated_at")
        ids = [result.candidate_id for result in self.results]
        if len(ids) != len(set(ids)):
            raise ValueError("skill intervals must have unique candidate ids")
        if self.report_hash and self.report_hash != self.content_hash():
            raise ValueError("report_hash does not match skill uncertainty report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "report_hash")

    def assert_sealed(self) -> None:
        if not self.report_hash or self.report_hash != self.content_hash():
            raise ValueError("forecast skill uncertainty report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ForecastSkillUncertaintyReport":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"report_hash"}),
            report_hash=draft.content_hash(),
        )


class ForecastSkillUncertaintyEvaluator:
    def evaluate(
        self,
        validation_report: ForecastValidationReport,
        spec: MovingBlockBootstrapSpec,
        *,
        evaluated_at: datetime | None = None,
    ) -> ForecastSkillUncertaintyReport:
        validation_report.assert_sealed()
        spec.assert_sealed()
        if spec.validation_report_hash != validation_report.report_hash:
            raise ValueError("bootstrap spec is bound to another validation report")
        by_id = {result.candidate_id: result for result in validation_report.results}
        if spec.baseline_candidate_id not in by_id:
            raise ValueError("bootstrap baseline is absent from the validation report")
        baseline = by_id[spec.baseline_candidate_id]
        baseline_errors = [
            abs(record.point_prediction - record.actual) for record in baseline.records
        ]
        if spec.block_length > len(baseline_errors):
            raise ValueError("bootstrap block length exceeds the holdout length")
        results: list[CandidateSkillInterval] = []
        for candidate in validation_report.results:
            if candidate.candidate_id == spec.baseline_candidate_id:
                continue
            if [record.target_timestamp for record in candidate.records] != [
                record.target_timestamp for record in baseline.records
            ]:
                raise ValueError("candidate and baseline holdouts are not paired")
            candidate_errors = [
                abs(record.point_prediction - record.actual)
                for record in candidate.records
            ]
            paired_improvements = [
                base - challenger
                for base, challenger in zip(baseline_errors, candidate_errors)
            ]
            draws = _moving_block_means(paired_improvements, spec)
            alpha = (1.0 - spec.confidence_level) / 2.0
            lower, upper = np.quantile(draws, [alpha, 1.0 - alpha], method="linear")
            observed = float(np.mean(paired_improvements))
            baseline_mae = float(np.mean(baseline_errors))
            if lower > 0:
                interpretation = "evidence_better_than_baseline"
            elif upper < 0:
                interpretation = "evidence_worse_than_baseline"
            else:
                interpretation = "inconclusive_vs_baseline"
            results.append(
                CandidateSkillInterval(
                    candidate_id=candidate.candidate_id,
                    holdout_points=len(paired_improvements),
                    mean_mae_improvement=observed,
                    relative_mae_improvement=(
                        None if baseline_mae <= 1e-15 else observed / baseline_mae
                    ),
                    confidence_lower=float(lower),
                    confidence_upper=float(upper),
                    bootstrap_standard_error=float(np.std(draws, ddof=1)),
                    probability_improvement_positive=float(np.mean(draws > 0)),
                    interpretation=interpretation,
                )
            )
        assert spec.bootstrap_spec_hash is not None
        assert validation_report.report_hash is not None
        return ForecastSkillUncertaintyReport.seal(
            report_id="forecast_skill_uncertainty",
            validation_report_hash=validation_report.report_hash,
            bootstrap_spec_hash=spec.bootstrap_spec_hash,
            results=results,
            evaluated_at=evaluated_at or datetime.now(timezone.utc),
        )


def _moving_block_means(
    values: list[float], spec: MovingBlockBootstrapSpec
) -> np.ndarray:
    count = len(values)
    draws: list[float] = []
    for seed in spec.seeds:
        random = Random(seed)
        for _ in range(spec.replicates_per_seed):
            sample: list[float] = []
            while len(sample) < count:
                start = random.randrange(count)
                sample.extend(
                    values[(start + offset) % count]
                    for offset in range(spec.block_length)
                )
            draws.append(sum(sample[:count]) / count)
    return np.asarray(draws, dtype=float)
