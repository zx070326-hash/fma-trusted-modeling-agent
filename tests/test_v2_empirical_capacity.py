from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from fma.v2.capacity_decision import CapacityDecisionEvaluator
from fma.v2.empirical_capacity import (
    FIXTURE_TIME,
    empirical_capacity_mission,
    fixture_data_contract,
    fixture_snapshot,
    run_empirical_capacity_fixture,
    verify_empirical_run,
)
from fma.v2.empirical_schemas import (
    CapacityDecisionSpec,
    ForecastValidationReport,
    TemporalValidationSpec,
    TimeSeriesDataContract,
    TimeSeriesPoint,
    TimeSeriesSnapshot,
)
from fma.v2.forecast_evaluator import RollingOriginForecastEvaluator
from fma.v2.forecast_generators import generate_default_forecast_portfolio
from fma.v2.timeseries_intake import QUALITY_CHECKS, ingest_local_timeseries_csv


def test_stable_fixture_reaches_only_a_shadow_decision_eligibility(tmp_path: Path) -> None:
    outcome = run_empirical_capacity_fixture(tmp_path, scenario="stable")

    assert outcome.validation_report.status == "sufficient"
    assert len(outcome.validation_report.passing_candidate_ids) == 4
    assert outcome.decision_report.status == "decision_eligible"
    assert outcome.decision_report.permissible_uses == ["synthetic_fixture_analysis"]
    assert outcome.decision_report.real_world_action_authorized is False
    assert {
        decision.recommended_capacity
        for decision in outcome.decision_report.candidate_decisions
    } == {10}
    assert verify_empirical_run(outcome.store.run_directory)


def test_regime_shift_abstains_instead_of_promoting_one_surviving_model(
    tmp_path: Path,
) -> None:
    outcome = run_empirical_capacity_fixture(tmp_path, scenario="regime_shift")

    assert outcome.validation_report.status == "needs_evidence"
    assert outcome.decision_report.status == "needs_evidence"
    assert "forecast_validation_insufficient" in outcome.decision_report.reason_codes
    assert "too_few_passing_models" in outcome.decision_report.reason_codes
    assert outcome.decision_report.real_world_action_authorized is False


def test_rolling_origin_predictions_cannot_see_a_mutated_future_actual() -> None:
    mission = empirical_capacity_mission()
    contract = fixture_data_contract(mission)
    original = fixture_snapshot(contract)
    portfolio = generate_default_forecast_portfolio(
        contract, seasonal_period=4, generated_at=FIXTURE_TIME
    )
    assert original.snapshot_hash is not None
    assert portfolio.portfolio_hash is not None

    changed_points = list(original.points)
    final = changed_points[-1]
    changed_points[-1] = TimeSeriesPoint(timestamp=final.timestamp, value=30.0)
    changed = TimeSeriesSnapshot.seal(
        snapshot_id="future_actual_mutated",
        data_contract_hash=original.data_contract_hash,
        source_content_hash="f" * 64,
        points=changed_points,
        quality_checks=QUALITY_CHECKS,
        collected_at=FIXTURE_TIME,
    )
    assert changed.snapshot_hash is not None

    def spec_for(snapshot_hash: str, validation_id: str) -> TemporalValidationSpec:
        return TemporalValidationSpec.seal(
            validation_id=validation_id,
            data_snapshot_hash=snapshot_hash,
            portfolio_hash=portfolio.portfolio_hash,
            holdout_points=8,
            minimum_training_points=16,
            minimum_calibration_residuals=8,
            interval_level=0.8,
            minimum_empirical_coverage=0.5,
            max_mae_ratio_to_last_value=100.0,
            required_passing_candidates=2,
            frozen_at=FIXTURE_TIME,
        )

    evaluator = RollingOriginForecastEvaluator()
    before = evaluator.evaluate(
        original,
        portfolio,
        spec_for(original.snapshot_hash, "before_future_mutation"),
        evaluated_at=FIXTURE_TIME,
    )
    after = evaluator.evaluate(
        changed,
        portfolio,
        spec_for(changed.snapshot_hash, "after_future_mutation"),
        evaluated_at=FIXTURE_TIME,
    )
    for before_result, after_result in zip(before.results, after.results):
        assert [record.point_prediction for record in before_result.records] == [
            record.point_prediction for record in after_result.records
        ]
        assert [record.interval_lower for record in before_result.records] == [
            record.interval_lower for record in after_result.records
        ]
        assert [record.interval_upper for record in before_result.records] == [
            record.interval_upper for record in after_result.records
        ]


def test_local_csv_intake_is_contract_bound_and_fail_closed(tmp_path: Path) -> None:
    contract = TimeSeriesDataContract.seal(
        dataset_id="local_weekly_demand",
        mission_spec_hash="a" * 64,
        source_kind="local_file",
        source_ref="workspace:demand.csv",
        frequency="weekly",
        value_unit="product_unit",
        minimum_points=12,
        created_at=FIXTURE_TIME,
    )
    valid_path = tmp_path / "demand.csv"
    rows = ["timestamp,value"] + [
        f"{(FIXTURE_TIME + timedelta(weeks=index)).isoformat()},{10 + index / 10}"
        for index in range(12)
    ]
    valid_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    snapshot = ingest_local_timeseries_csv(
        valid_path,
        workspace_root=tmp_path,
        contract=contract,
        collected_at=FIXTURE_TIME,
    )
    snapshot.assert_sealed()
    assert len(snapshot.points) == 12
    assert snapshot.data_contract_hash == contract.data_contract_hash

    duplicate_path = tmp_path / "duplicate.csv"
    duplicate_path.write_text(
        "timestamp,value\n"
        f"{FIXTURE_TIME.isoformat()},10\n"
        f"{FIXTURE_TIME.isoformat()},11\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        ingest_local_timeseries_csv(
            duplicate_path,
            workspace_root=tmp_path,
            contract=contract,
            collected_at=FIXTURE_TIME,
        )


def test_empirical_run_verifier_detects_artifact_tampering(tmp_path: Path) -> None:
    outcome = run_empirical_capacity_fixture(tmp_path, scenario="stable")
    decision_ref = next(
        ref
        for ref in outcome.manifest.artifact_refs
        if ref.kind == "decision_stability_report"
    )
    path = outcome.store.run_directory / decision_ref.relative_path
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["status"] = "needs_evidence"
    path.write_text(json.dumps(envelope), encoding="utf-8")

    assert verify_empirical_run(outcome.store.run_directory) is False


def test_decision_gate_recomputes_instead_of_trusting_a_resealed_report() -> None:
    mission = empirical_capacity_mission()
    contract = fixture_data_contract(mission)
    snapshot = fixture_snapshot(contract)
    portfolio = generate_default_forecast_portfolio(
        contract, seasonal_period=4, generated_at=FIXTURE_TIME
    )
    assert snapshot.snapshot_hash is not None
    assert portfolio.portfolio_hash is not None
    validation_spec = TemporalValidationSpec.seal(
        validation_id="recomputation_gate",
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
    valid = RollingOriginForecastEvaluator().evaluate(
        snapshot, portfolio, validation_spec, evaluated_at=FIXTURE_TIME
    )
    forged_payload = valid.model_dump(exclude={"report_hash"})
    forged_payload["status"] = "needs_evidence"
    forged = ForecastValidationReport.seal(**forged_payload)
    assert forged.report_hash is not None
    decision_spec = CapacityDecisionSpec.seal(
        decision_id="recomputation_decision_gate",
        validation_report_hash=forged.report_hash,
        shortage_cost=4.0,
        overage_cost=1.0,
        maximum_capacity=30,
        max_mean_holdout_regret=0.25,
        minimum_passing_models=3,
        frozen_at=FIXTURE_TIME,
    )

    with pytest.raises(ValueError, match="independent recomputation"):
        CapacityDecisionEvaluator().evaluate(
            contract,
            snapshot,
            portfolio,
            validation_spec,
            forged,
            decision_spec,
            evaluated_at=FIXTURE_TIME,
        )


def test_snapshot_seal_detects_post_hoc_point_change() -> None:
    contract = fixture_data_contract(empirical_capacity_mission())
    snapshot = fixture_snapshot(contract)
    tampered = snapshot.model_copy(
        update={
            "points": [
                *snapshot.points[:-1],
                TimeSeriesPoint(timestamp=snapshot.points[-1].timestamp, value=99.0),
            ]
        }
    )
    with pytest.raises(ValueError, match="not sealed"):
        tampered.assert_sealed()


def test_v21_candidate_and_portfolio_hashes_remain_backward_compatible() -> None:
    contract = fixture_data_contract(empirical_capacity_mission())
    portfolio = generate_default_forecast_portfolio(
        contract, seasonal_period=4, generated_at=FIXTURE_TIME
    )

    assert portfolio.portfolio_hash == (
        "e5c9c272c9909cd3f7db2a2a4622382331da02e785a92ff0fd17a9bcfa418823"
    )
    assert portfolio.candidates[0].candidate_hash == (
        "9687d94e8fc1fc763a3f09220518c9c8f118317e664a6b2d1fada7efa8619876"
    )
    assert {candidate.schema_version for candidate in portfolio.candidates} == {"2.1"}
