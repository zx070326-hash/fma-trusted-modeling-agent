from __future__ import annotations

from datetime import datetime, timezone

from .empirical_schemas import (
    CandidateCapacityDecision,
    CapacityDecisionSpec,
    DecisionStabilityReport,
    ForecastCandidatePortfolio,
    ForecastValidationReport,
    TemporalValidationSpec,
    TimeSeriesDataContract,
    TimeSeriesSnapshot,
)
from .forecast_evaluator import RollingOriginForecastEvaluator


class CapacityDecisionEvaluator:
    """Translate validated forecasts into shadow decisions and fail closed on disagreement."""

    def evaluate(
        self,
        data_contract: TimeSeriesDataContract,
        snapshot: TimeSeriesSnapshot,
        portfolio: ForecastCandidatePortfolio,
        validation_spec: TemporalValidationSpec,
        validation_report: ForecastValidationReport,
        decision_spec: CapacityDecisionSpec,
        *,
        evaluated_at: datetime | None = None,
    ) -> DecisionStabilityReport:
        data_contract.assert_sealed()
        snapshot.assert_sealed()
        portfolio.assert_sealed()
        validation_spec.assert_sealed()
        validation_report.assert_sealed()
        decision_spec.assert_sealed()
        if data_contract.data_contract_hash != snapshot.data_contract_hash:
            raise ValueError("data snapshot is bound to another data contract")
        if len(snapshot.points) < data_contract.minimum_points:
            raise ValueError("data snapshot violates its contract minimum length")
        if decision_spec.validation_report_hash != validation_report.report_hash:
            raise ValueError("decision spec is bound to another validation report")
        if validation_report.data_snapshot_hash != snapshot.snapshot_hash:
            raise ValueError("validation report is bound to another data snapshot")
        if validation_report.portfolio_hash != portfolio.portfolio_hash:
            raise ValueError("validation report is bound to another portfolio")

        forecast_evaluator = RollingOriginForecastEvaluator()
        recomputed_validation = forecast_evaluator.evaluate(
            snapshot,
            portfolio,
            validation_spec,
            evaluated_at=validation_report.evaluated_at,
        )
        if recomputed_validation.report_hash != validation_report.report_hash:
            raise ValueError("forecast validation report failed independent recomputation")

        by_id = {candidate.candidate_id: candidate for candidate in portfolio.candidates}
        result_by_id = {result.candidate_id: result for result in validation_report.results}
        values = [point.value for point in snapshot.points]
        decisions: list[CandidateCapacityDecision] = []
        for candidate_id in validation_report.passing_candidate_ids:
            candidate = by_id[candidate_id]
            result = result_by_id[candidate_id]
            point, lower, upper, _ = forecast_evaluator.forecast_with_interval(
                values, candidate, validation_spec
            )
            capacity = _optimal_capacity(point, decision_spec)
            regrets = [
                _decision_loss(
                    _optimal_capacity(record.point_prediction, decision_spec),
                    record.actual,
                    decision_spec,
                )
                - _decision_loss(
                    _optimal_capacity(record.actual, decision_spec),
                    record.actual,
                    decision_spec,
                )
                for record in result.records
            ]
            regrets = [max(0.0, regret) for regret in regrets]
            mean_regret = sum(regrets) / len(regrets)
            decisions.append(
                CandidateCapacityDecision(
                    candidate_id=candidate_id,
                    next_point_forecast=point,
                    next_interval_lower=lower,
                    next_interval_upper=upper,
                    recommended_capacity=capacity,
                    mean_holdout_regret=mean_regret,
                    maximum_holdout_regret=max(regrets),
                    status=(
                        "passed"
                        if mean_regret <= decision_spec.max_mean_holdout_regret + 1e-12
                        else "failed"
                    ),
                )
            )

        reasons: list[str] = []
        if validation_report.status != "sufficient":
            reasons.append("forecast_validation_insufficient")
        passed_decisions = [decision for decision in decisions if decision.status == "passed"]
        if len(passed_decisions) < decision_spec.minimum_passing_models:
            reasons.append("too_few_passing_models")
        if any(decision.status == "failed" for decision in decisions):
            reasons.append("holdout_regret_exceeded")
        if len({decision.recommended_capacity for decision in passed_decisions}) > 1:
            reasons.append("candidate_actions_disagree")
        permissible_use = (
            "synthetic_fixture_analysis"
            if data_contract.source_kind == "fixture"
            else "retrospective_shadow_analysis"
        )
        return DecisionStabilityReport.seal(
            report_id="capacity_decision_stability",
            data_snapshot_hash=snapshot.snapshot_hash,
            validation_report_hash=validation_report.report_hash,
            decision_spec_hash=decision_spec.decision_spec_hash,
            candidate_decisions=decisions,
            status="needs_evidence" if reasons else "decision_eligible",
            reason_codes=list(dict.fromkeys(reasons)),
            permissible_uses=[permissible_use],
            evaluated_at=evaluated_at or datetime.now(timezone.utc),
        )


def _optimal_capacity(point_demand: float, spec: CapacityDecisionSpec) -> int:
    return min(
        range(spec.maximum_capacity + 1),
        key=lambda capacity: (
            _decision_loss(capacity, point_demand, spec),
            capacity,
        ),
    )


def _decision_loss(
    capacity: int, actual_demand: float, spec: CapacityDecisionSpec
) -> float:
    return spec.overage_cost * max(capacity - actual_demand, 0.0) + spec.shortage_cost * max(
        actual_demand - capacity, 0.0
    )
