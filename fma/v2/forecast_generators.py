from __future__ import annotations

from datetime import datetime, timezone

from .empirical_schemas import (
    ForecastCandidatePortfolio,
    ForecastCandidateSpec,
    ForecastCandidateSpecV22,
    TimeSeriesDataContract,
)


def generate_default_forecast_portfolio(
    contract: TimeSeriesDataContract,
    *,
    seasonal_period: int = 4,
    generated_at: datetime | None = None,
) -> ForecastCandidatePortfolio:
    """Generate model specifications only; this module cannot score or promote them."""

    contract.assert_sealed()
    contract_hash = contract.data_contract_hash
    assert contract_hash is not None
    candidates = [
        ForecastCandidateSpec.seal(
            candidate_id="last_value_baseline",
            data_contract_hash=contract_hash,
            family="last_value",
            assumptions=["The most recent level persists for one step"],
            role="baseline",
        ),
        ForecastCandidateSpec.seal(
            candidate_id="mean_level_challenger",
            data_contract_hash=contract_hash,
            family="mean_level",
            assumptions=["The series fluctuates around a stable long-run level"],
            role="challenger",
        ),
        ForecastCandidateSpec.seal(
            candidate_id="linear_trend_challenger",
            data_contract_hash=contract_hash,
            family="linear_trend",
            assumptions=["A linear local trend remains informative one step ahead"],
            role="challenger",
        ),
        ForecastCandidateSpec.seal(
            candidate_id="seasonal_naive_challenger",
            data_contract_hash=contract_hash,
            family="seasonal_naive",
            seasonal_period=seasonal_period,
            assumptions=["The value repeats after the frozen seasonal period"],
            role="challenger",
        ),
    ]
    return ForecastCandidatePortfolio.seal(
        portfolio_id="default_forecast_portfolio",
        data_contract_hash=contract_hash,
        candidates=candidates,
        generated_at=generated_at or datetime.now(timezone.utc),
    )


def generate_failure_evolved_forecast_portfolio(
    contract: TimeSeriesDataContract,
    *,
    seasonal_period: int,
    local_window: int,
    smoothing_alpha: float,
    generated_at: datetime | None = None,
) -> ForecastCandidatePortfolio:
    """Freeze general evolution operators learned from prior failure signatures.

    This generator must be evaluated on a different series/site from the runs
    that motivated it; it still has no access to scoring or promotion.
    """

    contract.assert_sealed()
    contract_hash = contract.data_contract_hash
    assert contract_hash is not None
    candidates = [
        ForecastCandidateSpecV22.seal(
            candidate_id="last_value_baseline",
            data_contract_hash=contract_hash,
            family="last_value",
            assumptions=["The most recent level persists for one step"],
            role="baseline",
        ),
        ForecastCandidateSpecV22.seal(
            candidate_id="local_mean_challenger",
            data_contract_hash=contract_hash,
            family="window_mean",
            window_length=local_window,
            assumptions=["Recent local level is more relevant than remote history"],
            role="challenger",
        ),
        ForecastCandidateSpecV22.seal(
            candidate_id="local_trend_challenger",
            data_contract_hash=contract_hash,
            family="window_linear_trend",
            window_length=local_window,
            assumptions=["A recent local linear trend persists for one step"],
            role="challenger",
        ),
        ForecastCandidateSpecV22.seal(
            candidate_id="exponential_level_challenger",
            data_contract_hash=contract_hash,
            family="exponential_smoothing",
            smoothing_alpha=smoothing_alpha,
            assumptions=["Recent observations deserve geometrically greater weight"],
            role="challenger",
        ),
        ForecastCandidateSpecV22.seal(
            candidate_id="seasonal_naive_challenger",
            data_contract_hash=contract_hash,
            family="seasonal_naive",
            seasonal_period=seasonal_period,
            assumptions=["The value repeats after the frozen domain seasonal period"],
            role="challenger",
        ),
    ]
    return ForecastCandidatePortfolio.seal(
        portfolio_id="failure_evolved_forecast_portfolio",
        data_contract_hash=contract_hash,
        candidates=candidates,
        generator_id="failure_evolved_forecast_generator_v1",
        generated_at=generated_at or datetime.now(timezone.utc),
    )
