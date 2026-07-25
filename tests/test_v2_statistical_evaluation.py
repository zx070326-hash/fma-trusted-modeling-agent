from __future__ import annotations

from fma.v2.empirical_capacity import (
    FIXTURE_TIME,
    empirical_capacity_mission,
    fixture_data_contract,
    fixture_snapshot,
)
from fma.v2.empirical_schemas import TemporalValidationSpec
from fma.v2.forecast_evaluator import RollingOriginForecastEvaluator
from fma.v2.forecast_generators import generate_default_forecast_portfolio
from fma.v2.shift_diagnostics import WindowShiftEvaluator, WindowShiftSpec
from fma.v2.statistical_evaluator import (
    ForecastSkillUncertaintyEvaluator,
    MovingBlockBootstrapSpec,
)


def _validation(scenario: str):
    contract = fixture_data_contract(empirical_capacity_mission())
    snapshot = fixture_snapshot(contract, scenario=scenario)
    portfolio = generate_default_forecast_portfolio(
        contract, seasonal_period=4, generated_at=FIXTURE_TIME
    )
    assert snapshot.snapshot_hash is not None
    assert portfolio.portfolio_hash is not None
    spec = TemporalValidationSpec.seal(
        validation_id=f"{scenario}_statistical_test",
        data_snapshot_hash=snapshot.snapshot_hash,
        portfolio_hash=portfolio.portfolio_hash,
        holdout_points=8,
        minimum_training_points=16,
        minimum_calibration_residuals=8,
        interval_level=0.8,
        minimum_empirical_coverage=0.75,
        max_mae_ratio_to_last_value=1.05,
        required_passing_candidates=3,
        frozen_at=FIXTURE_TIME,
    )
    report = RollingOriginForecastEvaluator().evaluate(
        snapshot, portfolio, spec, evaluated_at=FIXTURE_TIME
    )
    return snapshot, report


def test_paired_moving_block_bootstrap_is_seeded_and_reports_effect_interval() -> None:
    _, validation = _validation("stable")
    assert validation.report_hash is not None
    spec = MovingBlockBootstrapSpec.seal(
        bootstrap_id="stable_skill_interval",
        validation_report_hash=validation.report_hash,
        baseline_candidate_id="last_value_baseline",
        block_length=2,
        replicates_per_seed=200,
        seeds=[11, 29, 47],
        confidence_level=0.95,
        frozen_at=FIXTURE_TIME,
    )
    evaluator = ForecastSkillUncertaintyEvaluator()
    first = evaluator.evaluate(validation, spec, evaluated_at=FIXTURE_TIME)
    second = evaluator.evaluate(validation, spec, evaluated_at=FIXTURE_TIME)

    assert first.report_hash == second.report_hash
    seasonal = next(
        result
        for result in first.results
        if result.candidate_id == "seasonal_naive_challenger"
    )
    assert seasonal.mean_mae_improvement > 0
    assert seasonal.confidence_lower > 0
    assert seasonal.interpretation == "evidence_better_than_baseline"
    assert seasonal.holdout_points == 8


def test_shift_diagnostic_separates_stable_pattern_from_regime_change() -> None:
    stable, _ = _validation("stable")
    changed, _ = _validation("regime_shift")
    assert stable.snapshot_hash is not None
    assert changed.snapshot_hash is not None

    def spec(snapshot_hash: str, diagnostic_id: str) -> WindowShiftSpec:
        return WindowShiftSpec.seal(
            diagnostic_id=diagnostic_id,
            data_snapshot_hash=snapshot_hash,
            holdout_points=8,
            reference_points=8,
            max_standardized_mean_shift=1.0,
            minimum_scale_ratio=0.5,
            maximum_scale_ratio=2.0,
            max_reference_range_exceedance=0.25,
            frozen_at=FIXTURE_TIME,
        )

    evaluator = WindowShiftEvaluator()
    stable_report = evaluator.evaluate(
        stable, spec(stable.snapshot_hash, "stable_window"), evaluated_at=FIXTURE_TIME
    )
    changed_report = evaluator.evaluate(
        changed, spec(changed.snapshot_hash, "changed_window"), evaluated_at=FIXTURE_TIME
    )

    assert stable_report.status == "stable_by_frozen_diagnostics"
    assert changed_report.status == "shift_detected"
    assert "mean_shift_exceeded" in changed_report.reason_codes
    assert "reference_range_exceedance" in changed_report.reason_codes
