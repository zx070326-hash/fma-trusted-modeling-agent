from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore

from .empirical_schemas import (
    ForecastCandidatePortfolio,
    ForecastValidationReport,
    TemporalValidationSpec,
    TimeSeriesDataContract,
    TimeSeriesSnapshot,
)
from .forecast_evaluator import RollingOriginForecastEvaluator
from .forecast_generators import (
    generate_default_forecast_portfolio,
    generate_failure_evolved_forecast_portfolio,
)
from .official_data import (
    OfficialDataReceipt,
    fetch_bls_monthly_series,
    fetch_usgs_daily_values,
)
from .schemas import ApprovalRecord, Identifier, MissionContract, MissionSpec, Sha256, _assert_timezone
from .shift_diagnostics import WindowShiftEvaluator, WindowShiftReport, WindowShiftSpec
from .statistical_evaluator import (
    ForecastSkillUncertaintyEvaluator,
    ForecastSkillUncertaintyReport,
    MovingBlockBootstrapSpec,
)


ITERATION_TIME = datetime(2026, 7, 22, tzinfo=timezone.utc)
DatasetName = Literal[
    "bls_nonfarm_employment",
    "usgs_potomac_discharge",
    "bls_private_weekly_hours",
    "usgs_point_of_rocks_discharge",
]


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class OfficialShadowReport(StrictModel):
    schema_version: Literal["2.1"] = "2.1"
    report_id: Identifier
    dataset_name: DatasetName
    data_snapshot_hash: Sha256
    official_receipt_hash: Sha256
    forecast_validation_hash: Sha256
    skill_uncertainty_hash: Sha256
    shift_report_hash: Sha256
    supported_challenger_ids: list[Identifier]
    status: Literal["shadow_evidence_sufficient", "needs_evidence"]
    reason_codes: list[Literal[
        "forecast_validation_insufficient",
        "distribution_shift_detected",
        "no_challenger_has_positive_skill_interval",
    ]]
    warnings: list[Literal[
        "retrospective_only",
        "published_source_is_revision_prone",
        "no_real_world_decision_contract",
    ]] = Field(min_length=3, max_length=3)
    real_world_action_authorized: Literal[False] = False
    evaluated_at: datetime
    report_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "OfficialShadowReport":
        _assert_timezone(self.evaluated_at, "evaluated_at")
        if (self.status == "shadow_evidence_sufficient") == bool(self.reason_codes):
            raise ValueError("sufficient reports need no reasons; abstentions need reasons")
        if self.report_hash and self.report_hash != self.content_hash():
            raise ValueError("report_hash does not match official shadow report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "report_hash")

    def assert_sealed(self) -> None:
        if not self.report_hash or self.report_hash != self.content_hash():
            raise ValueError("official shadow report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "OfficialShadowReport":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"report_hash"}),
            report_hash=draft.content_hash(),
        )


class OfficialShadowManifest(StrictModel):
    schema_version: Literal["2.1"] = "2.1"
    run_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")]
    dataset_name: DatasetName
    artifact_refs: list[ArtifactRef] = Field(min_length=14)
    terminal_status: Literal["shadow_evidence_sufficient", "needs_evidence"]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "OfficialShadowManifest":
        _assert_timezone(self.created_at, "created_at")
        if len({(ref.kind, ref.sha256) for ref in self.artifact_refs}) != len(
            self.artifact_refs
        ):
            raise ValueError("official shadow manifest references must be unique")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match official shadow manifest")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("official shadow manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "OfficialShadowManifest":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class OfficialShadowOutcome:
    store: RunStore
    manifest: OfficialShadowManifest
    report: OfficialShadowReport
    validation_report: ForecastValidationReport
    skill_report: ForecastSkillUncertaintyReport
    shift_report: WindowShiftReport


def official_shadow_mission() -> MissionContract:
    sources = [
        "bls:CES0000000001:2016:2025",
        "usgs:01646500:00060:00003:2023-01-01:2024-12-31",
        "bls:CES0500000002:2016:2025",
        "usgs:01638500:00060:00003:2023-01-01:2024-12-31",
    ]
    mission = MissionSpec.seal(
        mission_id="official_timeseries_shadow_mission",
        version=1,
        knowledge_objectives=[
            "Test whether the empirical evaluator survives two revision-prone official domains"
        ],
        intended_decisions=["Decide whether evidence is sufficient for further shadow study"],
        stakeholders_and_value_owners=["fma_research_owner"],
        spatial_temporal_scope="Frozen BLS monthly and USGS daily historical cutoffs",
        approved_evidence_sources=sources,
        resource_budget={"official_api_calls": 2, "candidate_models_per_dataset": 4},
        validation_budget_reserve={"bootstrap_seeds": 3, "tamper_replay": 1},
        allowed_actions=[
            "official_api_read",
            "local_compute",
            "write_local_run_artifacts",
        ],
        forbidden_actions=["external_action", "operational_decision"],
        stopping_policy={"when": "shadow_gate_or_needs_evidence"},
        created_at=ITERATION_TIME,
    )
    assert mission.mission_spec_hash is not None
    approval = ApprovalRecord.seal(
        approval_id="official_timeseries_shadow_approval",
        mission_spec_hash=mission.mission_spec_hash,
        sequence=1,
        policy_version="official_shadow_policy_v1",
        decision="approved",
        approved_scope={
            "allowed_actions": [
                "official_api_read",
                "local_compute",
                "write_local_run_artifacts",
            ]
        },
        approver_ref="fma_research_owner",
        issued_at=ITERATION_TIME,
    )
    return MissionContract(mission=mission, approval=approval)


def run_official_shadow_benchmark(
    output_root: str | Path,
    *,
    dataset_name: DatasetName,
    fetched_at: datetime | None = None,
    fetcher=None,
    run_id: str | None = None,
) -> OfficialShadowOutcome:
    mission = official_shadow_mission()
    observed_at = fetched_at or datetime.now(timezone.utc)
    mission.assert_active(observed_at)
    contract, settings = _dataset_contract_and_settings(mission, dataset_name)
    raw_responses: list[bytes] = []
    if settings["provider"] == "bls":
        snapshot, receipt = fetch_bls_monthly_series(
            contract,
            series_id=settings["series_id"],
            start_year=settings["start_year"],
            end_year=settings["end_year"],
            fetched_at=observed_at,
            fetcher=fetcher,
            _raw_sink=raw_responses.append,
        )
    else:
        snapshot, receipt = fetch_usgs_daily_values(
            contract,
            site_id=settings["site_id"],
            start_date=settings["start_date"],
            end_date=settings["end_date"],
            fetched_at=observed_at,
            fetcher=fetcher,
            _raw_sink=raw_responses.append,
        )
    if len(raw_responses) != 1:
        raise RuntimeError("official adapter did not expose exactly one raw response")
    if settings["portfolio"] == "evolved":
        portfolio = generate_failure_evolved_forecast_portfolio(
            contract,
            seasonal_period=settings["seasonal_period"],
            local_window=settings["local_window"],
            smoothing_alpha=settings["smoothing_alpha"],
            generated_at=ITERATION_TIME,
        )
    else:
        portfolio = generate_default_forecast_portfolio(
            contract,
            seasonal_period=settings["seasonal_period"],
            generated_at=ITERATION_TIME,
        )
    assert snapshot.snapshot_hash is not None
    assert portfolio.portfolio_hash is not None
    validation_spec = TemporalValidationSpec.seal(
        validation_id=f"{dataset_name}_rolling_gate",
        data_snapshot_hash=snapshot.snapshot_hash,
        portfolio_hash=portfolio.portfolio_hash,
        holdout_points=settings["holdout_points"],
        minimum_training_points=settings["minimum_training_points"],
        minimum_calibration_residuals=settings["minimum_calibration_residuals"],
        interval_level=0.8,
        minimum_empirical_coverage=0.7,
        max_mae_ratio_to_last_value=1.1,
        required_passing_candidates=2,
        frozen_at=ITERATION_TIME,
    )
    validation = RollingOriginForecastEvaluator().evaluate(
        snapshot, portfolio, validation_spec, evaluated_at=observed_at
    )
    assert validation.report_hash is not None
    bootstrap_spec = MovingBlockBootstrapSpec.seal(
        bootstrap_id=f"{dataset_name}_paired_skill",
        validation_report_hash=validation.report_hash,
        baseline_candidate_id="last_value_baseline",
        block_length=settings["bootstrap_block_length"],
        replicates_per_seed=500,
        seeds=[11, 29, 47],
        confidence_level=0.95,
        frozen_at=ITERATION_TIME,
    )
    skill = ForecastSkillUncertaintyEvaluator().evaluate(
        validation, bootstrap_spec, evaluated_at=observed_at
    )
    shift_spec = WindowShiftSpec.seal(
        diagnostic_id=f"{dataset_name}_adjacent_window_shift",
        data_snapshot_hash=snapshot.snapshot_hash,
        holdout_points=settings["holdout_points"],
        reference_points=settings["holdout_points"],
        max_standardized_mean_shift=1.0,
        minimum_scale_ratio=0.5,
        maximum_scale_ratio=2.0,
        max_reference_range_exceedance=0.25,
        frozen_at=ITERATION_TIME,
    )
    shift = WindowShiftEvaluator().evaluate(
        snapshot, shift_spec, evaluated_at=observed_at
    )
    report = _build_official_shadow_report(
        dataset_name, snapshot, receipt, validation, skill, shift, observed_at
    )

    store = RunStore(
        output_root,
        run_id=run_id or f"official-shadow-{dataset_name}-{uuid4().hex[:12]}",
    )
    raw_text = raw_responses[0].decode("utf-8")
    refs = [
        store.put_artifact("mission_spec", mission.mission),
        store.put_artifact("approval_record", mission.approval),
        store.put_artifact("timeseries_data_contract", contract),
        store.put_artifact("official_api_raw_response", {"utf8": raw_text}),
        store.put_artifact("official_data_receipt", receipt),
        store.put_artifact("timeseries_snapshot", snapshot),
        store.put_artifact("forecast_candidate_portfolio", portfolio),
        store.put_artifact("temporal_validation_spec", validation_spec),
        store.put_artifact("forecast_validation_report", validation),
        store.put_artifact("moving_block_bootstrap_spec", bootstrap_spec),
        store.put_artifact("forecast_skill_uncertainty_report", skill),
        store.put_artifact("window_shift_spec", shift_spec),
        store.put_artifact("window_shift_report", shift),
        store.put_artifact("official_shadow_report", report),
    ]
    manifest = OfficialShadowManifest.seal(
        run_id=store.run_id,
        dataset_name=dataset_name,
        artifact_refs=refs,
        terminal_status=report.status,
        created_at=observed_at,
    )
    manifest_ref = store.put_artifact("official_shadow_manifest", manifest)
    store.emit(
        "official_shadow_run_completed",
        {"manifest_ref": manifest_ref.model_dump(mode="json")},
    )
    if not verify_official_shadow_run(store.run_directory):
        raise RuntimeError("official shadow run failed independent verification")
    return OfficialShadowOutcome(
        store=store,
        manifest=manifest,
        report=report,
        validation_report=validation,
        skill_report=skill,
        shift_report=shift,
    )


def verify_official_shadow_run(run_directory: str | Path) -> bool:
    try:
        store = RunStore.open_existing(run_directory)
        events = [
            json.loads(line)
            for line in store.event_path.read_text(encoding="utf-8").splitlines()
        ]
        committed_refs = [
            ArtifactRef.model_validate(event["payload"])
            for event in events
            if event["event_type"] == "artifact_committed"
        ]
        for ref in committed_refs:
            store.load_artifact(ref)
        manifest_refs = [
            ref for ref in committed_refs if ref.kind == "official_shadow_manifest"
        ]
        if len(manifest_refs) != 1:
            return False
        manifest = OfficialShadowManifest.model_validate(
            store.load_artifact(manifest_refs[0])
        )
        manifest.assert_sealed()
        if manifest.run_id != store.run_id:
            return False
        committed = {(ref.kind, ref.sha256) for ref in committed_refs}
        if any((ref.kind, ref.sha256) not in committed for ref in manifest.artifact_refs):
            return False

        def load_one(kind: str, model=None):
            refs = [ref for ref in manifest.artifact_refs if ref.kind == kind]
            if len(refs) != 1:
                raise RuntimeError(f"manifest needs exactly one {kind}")
            payload = store.load_artifact(refs[0])
            return payload if model is None else model.model_validate(payload)

        mission = MissionContract(
            mission=load_one("mission_spec", MissionSpec),
            approval=load_one("approval_record", ApprovalRecord),
        )
        contract = load_one("timeseries_data_contract", TimeSeriesDataContract)
        raw_payload = load_one("official_api_raw_response")
        receipt = load_one("official_data_receipt", OfficialDataReceipt)
        snapshot = load_one("timeseries_snapshot", TimeSeriesSnapshot)
        portfolio = load_one("forecast_candidate_portfolio", ForecastCandidatePortfolio)
        validation_spec = load_one("temporal_validation_spec", TemporalValidationSpec)
        validation = load_one("forecast_validation_report", ForecastValidationReport)
        bootstrap_spec = load_one(
            "moving_block_bootstrap_spec", MovingBlockBootstrapSpec
        )
        skill = load_one(
            "forecast_skill_uncertainty_report", ForecastSkillUncertaintyReport
        )
        shift_spec = load_one("window_shift_spec", WindowShiftSpec)
        shift = load_one("window_shift_report", WindowShiftReport)
        report = load_one("official_shadow_report", OfficialShadowReport)
        raw_text = str(raw_payload["utf8"])
        if sha256_value({"raw_response_utf8": raw_text}) != receipt.response_content_hash:
            return False
        mission.assert_active(receipt.fetched_at)
        if contract.mission_spec_hash != mission.mission.mission_spec_hash:
            return False
        reparsed_snapshot, reparsed_receipt = _reparse_official_raw(
            contract, receipt, raw_text.encode("utf-8")
        )
        if (
            reparsed_snapshot.snapshot_hash != snapshot.snapshot_hash
            or reparsed_receipt.receipt_hash != receipt.receipt_hash
        ):
            return False
        recomputed_validation = RollingOriginForecastEvaluator().evaluate(
            snapshot,
            portfolio,
            validation_spec,
            evaluated_at=validation.evaluated_at,
        )
        if recomputed_validation.report_hash != validation.report_hash:
            return False
        recomputed_skill = ForecastSkillUncertaintyEvaluator().evaluate(
            validation, bootstrap_spec, evaluated_at=skill.evaluated_at
        )
        if recomputed_skill.report_hash != skill.report_hash:
            return False
        recomputed_shift = WindowShiftEvaluator().evaluate(
            snapshot, shift_spec, evaluated_at=shift.evaluated_at
        )
        if recomputed_shift.report_hash != shift.report_hash:
            return False
        recomputed_report = _build_official_shadow_report(
            manifest.dataset_name,
            snapshot,
            receipt,
            validation,
            skill,
            shift,
            report.evaluated_at,
        )
        return (
            recomputed_report.report_hash == report.report_hash
            and report.status == manifest.terminal_status
        )
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ):
        return False


def _dataset_contract_and_settings(
    mission: MissionContract, dataset_name: DatasetName
) -> tuple[TimeSeriesDataContract, dict[str, object]]:
    mission_hash = mission.mission.mission_spec_hash
    assert mission_hash is not None
    if dataset_name == "bls_nonfarm_employment":
        return (
            TimeSeriesDataContract.seal(
                dataset_id="bls_nonfarm_employment",
                mission_spec_hash=mission_hash,
                source_kind="official_api",
                source_ref="bls:CES0000000001:2016:2025",
                frequency="monthly",
                value_unit="thousand_persons",
                minimum_points=120,
                created_at=ITERATION_TIME,
            ),
            {
                "provider": "bls",
                "series_id": "CES0000000001",
                "start_year": 2016,
                "end_year": 2025,
                "portfolio": "default",
                "seasonal_period": 12,
                "holdout_points": 24,
                "minimum_training_points": 60,
                "minimum_calibration_residuals": 24,
                "bootstrap_block_length": 6,
            },
        )
    if dataset_name == "usgs_potomac_discharge":
        return (
            TimeSeriesDataContract.seal(
                dataset_id="usgs_potomac_discharge",
                mission_spec_hash=mission_hash,
                source_kind="official_api",
                source_ref="usgs:01646500:00060:00003:2023-01-01:2024-12-31",
                frequency="daily",
                value_unit="cubic_feet_per_second",
                minimum_points=731,
                created_at=ITERATION_TIME,
            ),
            {
                "provider": "usgs",
                "site_id": "01646500",
                "start_date": date(2023, 1, 1),
                "end_date": date(2024, 12, 31),
                "portfolio": "default",
                "seasonal_period": 7,
                "holdout_points": 60,
                "minimum_training_points": 365,
                "minimum_calibration_residuals": 60,
                "bootstrap_block_length": 7,
            },
        )
    if dataset_name == "bls_private_weekly_hours":
        return (
            TimeSeriesDataContract.seal(
                dataset_id="bls_private_weekly_hours",
                mission_spec_hash=mission_hash,
                source_kind="official_api",
                source_ref="bls:CES0500000002:2016:2025",
                frequency="monthly",
                value_unit="hours_per_week",
                minimum_points=120,
                created_at=ITERATION_TIME,
            ),
            {
                "provider": "bls",
                "series_id": "CES0500000002",
                "start_year": 2016,
                "end_year": 2025,
                "portfolio": "evolved",
                "seasonal_period": 12,
                "local_window": 12,
                "smoothing_alpha": 0.5,
                "holdout_points": 24,
                "minimum_training_points": 60,
                "minimum_calibration_residuals": 24,
                "bootstrap_block_length": 6,
            },
        )
    return (
        TimeSeriesDataContract.seal(
            dataset_id="usgs_point_of_rocks_discharge",
            mission_spec_hash=mission_hash,
            source_kind="official_api",
            source_ref="usgs:01638500:00060:00003:2023-01-01:2024-12-31",
            frequency="daily",
            value_unit="cubic_feet_per_second",
            minimum_points=731,
            created_at=ITERATION_TIME,
        ),
        {
            "provider": "usgs",
            "site_id": "01638500",
            "start_date": date(2023, 1, 1),
            "end_date": date(2024, 12, 31),
            "portfolio": "evolved",
            "seasonal_period": 365,
            "local_window": 30,
            "smoothing_alpha": 0.35,
            "holdout_points": 60,
            "minimum_training_points": 365,
            "minimum_calibration_residuals": 60,
            "bootstrap_block_length": 7,
        },
    )


def _build_official_shadow_report(
    dataset_name: DatasetName,
    snapshot: TimeSeriesSnapshot,
    receipt: OfficialDataReceipt,
    validation: ForecastValidationReport,
    skill: ForecastSkillUncertaintyReport,
    shift: WindowShiftReport,
    evaluated_at: datetime,
) -> OfficialShadowReport:
    supported = [
        result.candidate_id
        for result in skill.results
        if result.interpretation == "evidence_better_than_baseline"
    ]
    reasons: list[str] = []
    if validation.status != "sufficient":
        reasons.append("forecast_validation_insufficient")
    if shift.status == "shift_detected":
        reasons.append("distribution_shift_detected")
    if not supported:
        reasons.append("no_challenger_has_positive_skill_interval")
    assert snapshot.snapshot_hash is not None
    assert receipt.receipt_hash is not None
    assert validation.report_hash is not None
    assert skill.report_hash is not None
    assert shift.report_hash is not None
    return OfficialShadowReport.seal(
        report_id=f"{dataset_name}_shadow_report",
        dataset_name=dataset_name,
        data_snapshot_hash=snapshot.snapshot_hash,
        official_receipt_hash=receipt.receipt_hash,
        forecast_validation_hash=validation.report_hash,
        skill_uncertainty_hash=skill.report_hash,
        shift_report_hash=shift.report_hash,
        supported_challenger_ids=supported,
        status="needs_evidence" if reasons else "shadow_evidence_sufficient",
        reason_codes=reasons,
        warnings=[
            "retrospective_only",
            "published_source_is_revision_prone",
            "no_real_world_decision_contract",
        ],
        evaluated_at=evaluated_at,
    )


def _reparse_official_raw(
    contract: TimeSeriesDataContract,
    receipt: OfficialDataReceipt,
    raw: bytes,
) -> tuple[TimeSeriesSnapshot, OfficialDataReceipt]:
    if receipt.provider == "bls":
        prefix, series_id, start_year, end_year = contract.source_ref.split(":")
        if prefix != "bls":
            raise ValueError("BLS receipt and contract disagree")
        return fetch_bls_monthly_series(
            contract,
            series_id=series_id,
            start_year=int(start_year),
            end_year=int(end_year),
            fetched_at=receipt.fetched_at,
            fetcher=lambda *_: raw,
        )
    prefix, site, parameter, statistic, start, end = contract.source_ref.split(":")
    if prefix != "usgs":
        raise ValueError("USGS receipt and contract disagree")
    return fetch_usgs_daily_values(
        contract,
        site_id=site,
        parameter_code=parameter,
        statistic_code=statistic,
        start_date=date.fromisoformat(start),
        end_date=date.fromisoformat(end),
        fetched_at=receipt.fetched_at,
        fetcher=lambda *_: raw,
    )
