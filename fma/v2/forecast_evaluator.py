from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from .empirical_schemas import (
    CandidateValidationResult,
    ForecastCandidatePortfolio,
    ForecastCandidate,
    ForecastRecord,
    ForecastValidationReport,
    TemporalValidationSpec,
    TimeSeriesSnapshot,
)


class RollingOriginForecastEvaluator:
    """Code-owned evaluator. Candidates cannot modify splits, metrics, or gates."""

    evaluator_id = "independent_rolling_origin_evaluator_v1"

    def evaluate(
        self,
        snapshot: TimeSeriesSnapshot,
        portfolio: ForecastCandidatePortfolio,
        spec: TemporalValidationSpec,
        *,
        evaluated_at: datetime | None = None,
    ) -> ForecastValidationReport:
        snapshot.assert_sealed()
        portfolio.assert_sealed()
        spec.assert_sealed()
        if spec.data_snapshot_hash != snapshot.snapshot_hash:
            raise ValueError("validation spec is bound to another data snapshot")
        if spec.portfolio_hash != portfolio.portfolio_hash:
            raise ValueError("validation spec is bound to another candidate portfolio")
        if portfolio.data_contract_hash != snapshot.data_contract_hash:
            raise ValueError("portfolio and data snapshot use different contracts")
        point_count = len(snapshot.points)
        first_origin = point_count - spec.holdout_points
        if first_origin < spec.minimum_training_points:
            raise ValueError("frozen holdout leaves too few training observations")

        preliminary: list[tuple[ForecastCandidate, list[ForecastRecord]]] = []
        for candidate in portfolio.candidates:
            records = self._rolling_records(snapshot, candidate, spec, first_origin)
            preliminary.append((candidate, records))
        baseline_records = next(
            records for candidate, records in preliminary if candidate.family == "last_value"
        )
        baseline_mae = _mae(baseline_records)

        results: list[CandidateValidationResult] = []
        for candidate, records in preliminary:
            mae = _mae(records)
            rmse = math.sqrt(sum((r.point_prediction - r.actual) ** 2 for r in records) / len(records))
            bias = sum(r.point_prediction - r.actual for r in records) / len(records)
            coverage = sum(
                r.interval_lower - 1e-12 <= r.actual <= r.interval_upper + 1e-12
                for r in records
            ) / len(records)
            mean_width = sum(r.interval_upper - r.interval_lower for r in records) / len(records)
            ratio = None if baseline_mae <= 1e-15 else mae / baseline_mae
            reasons: list[str] = []
            if mae > baseline_mae * spec.max_mae_ratio_to_last_value + 1e-12:
                reasons.append("mae_gate_failed")
            if coverage + 1e-12 < spec.minimum_empirical_coverage:
                reasons.append("coverage_gate_failed")
            candidate_hash = candidate.candidate_hash
            assert candidate_hash is not None
            results.append(
                CandidateValidationResult(
                    candidate_id=candidate.candidate_id,
                    candidate_hash=candidate_hash,
                    records=records,
                    mae=mae,
                    rmse=rmse,
                    mean_bias=bias,
                    empirical_coverage=coverage,
                    mean_interval_width=mean_width,
                    mae_ratio_to_last_value=ratio,
                    status="failed" if reasons else "passed",
                    reason_codes=reasons,
                )
            )
        passing = [result.candidate_id for result in results if result.status == "passed"]
        return ForecastValidationReport.seal(
            report_id="rolling_origin_forecast_validation",
            data_snapshot_hash=snapshot.snapshot_hash,
            portfolio_hash=portfolio.portfolio_hash,
            validation_spec_hash=spec.validation_spec_hash,
            results=results,
            passing_candidate_ids=passing,
            status=(
                "sufficient"
                if len(passing) >= spec.required_passing_candidates
                else "needs_evidence"
            ),
            evaluated_at=evaluated_at or datetime.now(timezone.utc),
        )

    def forecast_with_interval(
        self,
        values: list[float],
        candidate: ForecastCandidate,
        spec: TemporalValidationSpec,
    ) -> tuple[float, float, float, int]:
        prediction = forecast_one(values, candidate)
        residuals = calibration_residuals(values, candidate)
        if len(residuals) < spec.minimum_calibration_residuals:
            raise ValueError("insufficient prior residuals for interval calibration")
        radius = _higher_quantile(
            [abs(residual) for residual in residuals], spec.interval_level
        )
        return prediction, max(0.0, prediction - radius), prediction + radius, len(residuals)

    def _rolling_records(
        self,
        snapshot: TimeSeriesSnapshot,
        candidate: ForecastCandidate,
        spec: TemporalValidationSpec,
        first_origin: int,
    ) -> list[ForecastRecord]:
        values = [point.value for point in snapshot.points]
        records: list[ForecastRecord] = []
        residuals = calibration_residuals(values[:first_origin], candidate)
        for origin in range(first_origin, len(values)):
            prediction = forecast_one(values[:origin], candidate)
            if len(residuals) < spec.minimum_calibration_residuals:
                raise ValueError("insufficient prior residuals for interval calibration")
            radius = _higher_quantile(
                [abs(residual) for residual in residuals], spec.interval_level
            )
            lower = max(0.0, prediction - radius)
            upper = prediction + radius
            records.append(
                ForecastRecord(
                    origin_index=origin,
                    target_timestamp=snapshot.points[origin].timestamp,
                    actual=values[origin],
                    point_prediction=prediction,
                    interval_lower=lower,
                    interval_upper=upper,
                    calibration_residual_count=len(residuals),
                )
            )
            residuals.append(values[origin] - prediction)
        return records


def forecast_one(values: list[float], candidate: ForecastCandidate) -> float:
    if len(values) < 2:
        raise ValueError("at least two observations are required")
    candidate.assert_sealed()
    data = np.asarray(values, dtype=float)
    if not np.isfinite(data).all() or (data < 0).any():
        raise ValueError("forecast inputs must be finite and nonnegative")
    if candidate.family == "last_value":
        prediction = float(data[-1])
    elif candidate.family == "mean_level":
        prediction = float(data.mean())
    elif candidate.family == "linear_trend":
        x = np.arange(len(data), dtype=float)
        slope, intercept = np.polyfit(x, data, deg=1)
        prediction = float(intercept + slope * len(data))
    elif candidate.family == "window_mean":
        assert candidate.window_length is not None
        if len(data) < candidate.window_length:
            raise ValueError("window-mean candidate has insufficient history")
        prediction = float(data[-candidate.window_length :].mean())
    elif candidate.family == "window_linear_trend":
        assert candidate.window_length is not None
        if len(data) < candidate.window_length:
            raise ValueError("window-trend candidate has insufficient history")
        local = data[-candidate.window_length :]
        x = np.arange(len(local), dtype=float)
        slope, intercept = np.polyfit(x, local, deg=1)
        prediction = float(intercept + slope * len(local))
    elif candidate.family == "exponential_smoothing":
        assert candidate.smoothing_alpha is not None
        level = float(data[0])
        for observation in data[1:]:
            level = candidate.smoothing_alpha * float(observation) + (
                1.0 - candidate.smoothing_alpha
            ) * level
        prediction = level
    else:
        assert candidate.seasonal_period is not None
        if len(data) < candidate.seasonal_period:
            raise ValueError("seasonal candidate has insufficient history")
        prediction = float(data[-candidate.seasonal_period])
    if not math.isfinite(prediction):
        raise ValueError("forecast is not finite")
    return max(0.0, prediction)


def calibration_residuals(
    values: list[float], candidate: ForecastCandidate
) -> list[float]:
    minimum_fit = max(
        3,
        candidate.seasonal_period or 0,
        getattr(candidate, "window_length", None) or 0,
    )
    residuals: list[float] = []
    for origin in range(minimum_fit, len(values)):
        predicted = forecast_one(values[:origin], candidate)
        residuals.append(float(values[origin] - predicted))
    return residuals


def _higher_quantile(values: list[float], level: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile needs observations")
    index = min(len(ordered) - 1, math.ceil(level * len(ordered)) - 1)
    return float(ordered[index])


def _mae(records: list[ForecastRecord]) -> float:
    return sum(abs(record.point_prediction - record.actual) for record in records) / len(records)
