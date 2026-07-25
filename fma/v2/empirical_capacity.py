from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef
from fma.storage import RunStore

from .capacity_decision import CapacityDecisionEvaluator
from .empirical_schemas import (
    CapacityDecisionSpec,
    DecisionStabilityReport,
    EmpiricalRunManifest,
    ForecastCandidatePortfolio,
    ForecastValidationReport,
    TemporalValidationSpec,
    TimeSeriesDataContract,
    TimeSeriesPoint,
    TimeSeriesSnapshot,
)
from .forecast_evaluator import RollingOriginForecastEvaluator
from .forecast_generators import generate_default_forecast_portfolio
from .schemas import ApprovalRecord, MissionContract, MissionSpec
from .timeseries_intake import QUALITY_CHECKS


FIXTURE_TIME = datetime(2026, 7, 22, tzinfo=timezone.utc)


@dataclass(frozen=True)
class EmpiricalCapacityOutcome:
    store: RunStore
    manifest: EmpiricalRunManifest
    manifest_ref: ArtifactRef
    validation_report: ForecastValidationReport
    decision_report: DecisionStabilityReport


def empirical_capacity_mission() -> MissionContract:
    mission = MissionSpec.seal(
        mission_id="empirical_capacity_mission",
        version=1,
        knowledge_objectives=["Compare distinct demand mechanisms on frozen temporal holdouts"],
        intended_decisions=["Assess whether a next-period capacity choice is stable"],
        stakeholders_and_value_owners=["fixture_operations_owner"],
        spatial_temporal_scope="One synthetic weekly demand series; retrospective shadow only",
        approved_evidence_sources=["capacity_fixture:historical_weekly_demand"],
        resource_budget={"candidate_models": 4, "forecast_horizon": 1},
        validation_budget_reserve={"holdout_points": 8},
        allowed_actions=["local_compute", "write_local_run_artifacts"],
        forbidden_actions=["external_action", "real_world_capacity_change"],
        stopping_policy={"when": "decision_stable_or_needs_evidence"},
        created_at=FIXTURE_TIME,
    )
    assert mission.mission_spec_hash is not None
    approval = ApprovalRecord.seal(
        approval_id="empirical_capacity_mission_approval",
        mission_spec_hash=mission.mission_spec_hash,
        sequence=1,
        policy_version="empirical_capacity_fixture_policy_v1",
        decision="approved",
        approved_scope={"allowed_actions": ["local_compute", "write_local_run_artifacts"]},
        approver_ref="fixture_operations_owner",
        issued_at=FIXTURE_TIME,
    )
    return MissionContract(mission=mission, approval=approval)


def fixture_data_contract(mission: MissionContract) -> TimeSeriesDataContract:
    mission.assert_active(FIXTURE_TIME)
    mission_hash = mission.mission.mission_spec_hash
    assert mission_hash is not None
    return TimeSeriesDataContract.seal(
        dataset_id="weekly_capacity_demand",
        mission_spec_hash=mission_hash,
        source_kind="fixture",
        source_ref="capacity_fixture:historical_weekly_demand",
        frequency="weekly",
        value_unit="product_unit",
        minimum_points=32,
        created_at=FIXTURE_TIME,
    )


def fixture_snapshot(
    contract: TimeSeriesDataContract, *, scenario: str = "stable"
) -> TimeSeriesSnapshot:
    contract.assert_sealed()
    if scenario not in {"stable", "regime_shift"}:
        raise ValueError("scenario must be stable or regime_shift")
    pattern = [-0.2, 0.1, 0.2, -0.1]
    values = [10.0 + pattern[index % 4] for index in range(32)]
    if scenario == "regime_shift":
        values[-8:] = [15.0 + pattern[index % 4] for index in range(8)]
    points = [
        TimeSeriesPoint(
            timestamp=FIXTURE_TIME - timedelta(weeks=31 - index),
            value=value,
        )
        for index, value in enumerate(values)
    ]
    assert contract.data_contract_hash is not None
    return TimeSeriesSnapshot.seal(
        snapshot_id=f"{scenario}_weekly_demand",
        data_contract_hash=contract.data_contract_hash,
        source_content_hash=sha256_value(
            {"fixture": scenario, "points": [point.model_dump(mode="json") for point in points]}
        ),
        points=points,
        quality_checks=QUALITY_CHECKS,
        collected_at=FIXTURE_TIME,
    )


def run_empirical_capacity_fixture(
    output_root: str | Path,
    *,
    scenario: str = "stable",
    run_id: str | None = None,
) -> EmpiricalCapacityOutcome:
    mission = empirical_capacity_mission()
    contract = fixture_data_contract(mission)
    snapshot = fixture_snapshot(contract, scenario=scenario)
    portfolio = generate_default_forecast_portfolio(
        contract, seasonal_period=4, generated_at=FIXTURE_TIME
    )
    assert snapshot.snapshot_hash is not None
    assert portfolio.portfolio_hash is not None
    validation_spec = TemporalValidationSpec.seal(
        validation_id=f"{scenario}_rolling_origin_gate",
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
    validation_report = RollingOriginForecastEvaluator().evaluate(
        snapshot, portfolio, validation_spec, evaluated_at=FIXTURE_TIME
    )
    assert validation_report.report_hash is not None
    decision_spec = CapacityDecisionSpec.seal(
        decision_id=f"{scenario}_capacity_gate",
        validation_report_hash=validation_report.report_hash,
        shortage_cost=4.0,
        overage_cost=1.0,
        maximum_capacity=30,
        max_mean_holdout_regret=0.25,
        minimum_passing_models=3,
        frozen_at=FIXTURE_TIME,
    )
    decision_report = CapacityDecisionEvaluator().evaluate(
        contract,
        snapshot,
        portfolio,
        validation_spec,
        validation_report,
        decision_spec,
        evaluated_at=FIXTURE_TIME,
    )

    store = RunStore(
        output_root,
        run_id=run_id or f"empirical-capacity-{scenario}-{uuid4().hex[:12]}",
    )
    refs = [
        store.put_artifact("mission_spec", mission.mission),
        store.put_artifact("approval_record", mission.approval),
        store.put_artifact("timeseries_data_contract", contract),
        store.put_artifact("timeseries_snapshot", snapshot),
        store.put_artifact("forecast_candidate_portfolio", portfolio),
        store.put_artifact("temporal_validation_spec", validation_spec),
        store.put_artifact("forecast_validation_report", validation_report),
        store.put_artifact("capacity_decision_spec", decision_spec),
        store.put_artifact("decision_stability_report", decision_report),
    ]
    manifest = EmpiricalRunManifest.seal(
        run_id=store.run_id,
        artifact_refs=refs,
        terminal_status=decision_report.status,
        created_at=FIXTURE_TIME,
    )
    manifest_ref = store.put_artifact("empirical_run_manifest", manifest)
    store.emit(
        "empirical_run_completed",
        {"manifest_ref": manifest_ref.model_dump(mode="json")},
    )
    if not verify_empirical_run(store.run_directory):
        raise RuntimeError("empirical fixture failed independent run verification")
    return EmpiricalCapacityOutcome(
        store=store,
        manifest=manifest,
        manifest_ref=manifest_ref,
        validation_report=validation_report,
        decision_report=decision_report,
    )


def verify_empirical_run(run_directory: str | Path) -> bool:
    try:
        store = RunStore.open_existing(run_directory)
        events = [
            json.loads(line)
            for line in store.event_path.read_text(encoding="utf-8").splitlines()
        ]
        refs = [
            ArtifactRef.model_validate(event["payload"])
            for event in events
            if event["event_type"] == "artifact_committed"
        ]
        for ref in refs:
            store.load_artifact(ref)
        manifest_refs = [ref for ref in refs if ref.kind == "empirical_run_manifest"]
        if len(manifest_refs) != 1:
            return False
        manifest = EmpiricalRunManifest.model_validate(store.load_artifact(manifest_refs[0]))
        manifest.assert_sealed()
        if manifest.run_id != store.run_id:
            return False
        committed = {(ref.kind, ref.sha256) for ref in refs}
        if any((ref.kind, ref.sha256) not in committed for ref in manifest.artifact_refs):
            return False
        def load_one(kind: str, model: type):
            matching = [ref for ref in manifest.artifact_refs if ref.kind == kind]
            if len(matching) != 1:
                raise RuntimeError(f"manifest needs exactly one {kind} artifact")
            return model.model_validate(store.load_artifact(matching[0]))

        mission_spec = load_one("mission_spec", MissionSpec)
        approval_record = load_one("approval_record", ApprovalRecord)
        mission = MissionContract(mission=mission_spec, approval=approval_record)
        data_contract = load_one("timeseries_data_contract", TimeSeriesDataContract)
        snapshot = load_one("timeseries_snapshot", TimeSeriesSnapshot)
        portfolio = load_one("forecast_candidate_portfolio", ForecastCandidatePortfolio)
        validation_spec = load_one("temporal_validation_spec", TemporalValidationSpec)
        validation_report = load_one(
            "forecast_validation_report", ForecastValidationReport
        )
        decision_spec = load_one("capacity_decision_spec", CapacityDecisionSpec)
        decision_report = load_one(
            "decision_stability_report", DecisionStabilityReport
        )
        mission.assert_active(snapshot.collected_at)
        if data_contract.mission_spec_hash != mission_spec.mission_spec_hash:
            return False
        data_contract.assert_sealed()
        snapshot.assert_sealed()
        portfolio.assert_sealed()
        validation_spec.assert_sealed()
        validation_report.assert_sealed()
        decision_spec.assert_sealed()
        decision_report.assert_sealed()
        recomputed_validation = RollingOriginForecastEvaluator().evaluate(
            snapshot,
            portfolio,
            validation_spec,
            evaluated_at=validation_report.evaluated_at,
        )
        if recomputed_validation.report_hash != validation_report.report_hash:
            return False
        recomputed_decision = CapacityDecisionEvaluator().evaluate(
            data_contract,
            snapshot,
            portfolio,
            validation_spec,
            validation_report,
            decision_spec,
            evaluated_at=decision_report.evaluated_at,
        )
        return (
            recomputed_decision.decision_report_hash
            == decision_report.decision_report_hash
            and decision_report.status == manifest.terminal_status
        )
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError):
        return False
