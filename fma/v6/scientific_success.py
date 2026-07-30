"""V6.1 claim-relative scientific-success evaluation.

This module is additive over V5 workflow certificates and V6 recovery state.
It does not reinterpret an S0--S6 gate as scientific qualification.  Instead,
it freezes a narrow predictive-success contract before computation and later
evaluates a leakage-safe rolling-origin confirmation of the registered
positive-scalar capability pack.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Field, model_validator

from fma.hashing import canonical_json, sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5.stage_workspace import StageWorkspaceV50
from fma.v5.workspace_schemas import StageId
from fma.v5_2.ode_system import (
    ODEScientificBundleV52,
    ODEThresholdsV52,
    ODETimeSeriesSnapshotV52,
    build_ode_bundle_v52,
)
from fma.v5_6.hybrid_ode import HybridODEThresholdsV56
from fma.v5_7.adaptive_positive_series import (
    AdaptivePositiveSeriesBundleV57,
    AdaptiveThresholdsV57,
    build_adaptive_positive_series_bundle_v57,
)


ODE_ADAPTER_ID = "scalar_autonomous_ode_v52"
ADAPTIVE_ADAPTER_ID = "adaptive_positive_series_v57"
SUCCESS_CONTRACT_PATH = "docs/scientific_success_contract_v61.json"
ROLLING_CONFIRMATION_PATH = "results/rolling_confirmation_v61.json"
SUCCESS_PROJECTION_PATH = ".fma/scientific_success_v61/report.json"

DimensionStatusV61 = Literal["PASS", "FAIL", "NOT_RUN", "HUMAN"]
ClaimKindV61 = Literal[
    "descriptive",
    "predictive",
    "mechanistic",
    "prescriptive",
    "generalization",
]
ClaimCeilingV61 = Literal[
    "no_scientific_claim",
    "workflow_integrity_only",
    "fixture_protocol_only",
    "local_retrospective_adapter_evidence",
    "local_leakage_safe_predictive_evidence",
    "externally_qualified_predictive_evidence",
]

_DIMENSION_IDS: tuple[str, ...] = (
    "workflow_integrity",
    "data_provenance",
    "local_adapter_checks",
    "leakage_safe_confirmation",
    "decision_value",
    "mechanism_identification",
    "external_generalization",
    "scientific_qualification",
)
_STAGES: tuple[StageId, ...] = ("S0", "S1", "S2", "S3", "S4", "S5", "S6")


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lag1(values: np.ndarray) -> float:
    if len(values) < 3:
        return 0.0
    left = values[:-1]
    right = values[1:]
    if float(np.std(left)) <= 1e-15 or float(np.std(right)) <= 1e-15:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else 0.0


class ScientificSuccessThresholdsV61(StrictModel):
    schema_version: Literal["6.1-success-thresholds"] = (
        "6.1-success-thresholds"
    )
    ode_minimum_history_points: Annotated[int, Field(ge=12)] = 17
    ode_confirmation_folds: Annotated[int, Field(ge=4)] = 6
    adaptive_minimum_history_points: Annotated[int, Field(ge=26)] = 26
    adaptive_confirmation_folds: Annotated[int, Field(ge=6)] = 8
    maximum_confirmation_relative_rmse: Annotated[
        float, Field(gt=0, allow_inf_nan=False)
    ] = 0.20
    minimum_persistence_relative_improvement: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ] = 0.10
    maximum_absolute_residual_lag1_correlation: Annotated[
        float, Field(gt=0, le=1, allow_inf_nan=False)
    ] = 0.85
    minimum_confirmation_interval_coverage: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ] = 0.50
    minimum_admissible_fold_fraction: Annotated[
        float, Field(gt=0, le=1, allow_inf_nan=False)
    ] = 1.0
    thresholds_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_thresholds(self) -> "ScientificSuccessThresholdsV61":
        if (
            self.thresholds_hash
            and self.thresholds_hash != self.content_hash()
        ):
            raise ValueError("V6.1 success threshold hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "thresholds_hash")

    def assert_sealed(self) -> None:
        if (
            not self.thresholds_hash
            or self.thresholds_hash != self.content_hash()
        ):
            raise ValueError("V6.1 success thresholds are not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ScientificSuccessThresholdsV61":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"thresholds_hash"})
        payload["thresholds_hash"] = draft.content_hash()
        return cls(**payload)


class ScientificSuccessContractV61(StrictModel):
    schema_version: Literal["6.1-success-contract"] = "6.1-success-contract"
    workspace_spec_hash: Sha256
    adapter_id: Literal[
        "scalar_autonomous_ode_v52",
        "adaptive_positive_series_v57",
    ]
    claim_kind: ClaimKindV61 = "predictive"
    confirmation_method: Literal[
        "nested_rolling_origin_one_step"
    ] = "nested_rolling_origin_one_step"
    required_dimension_ids: list[Identifier]
    thresholds: ScientificSuccessThresholdsV61
    private_feedback_permitted: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    contract_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "ScientificSuccessContractV61":
        if self.required_dimension_ids != sorted(
            set(self.required_dimension_ids)
        ):
            raise ValueError(
                "success-contract dimensions must be sorted and unique"
            )
        if (
            self.contract_hash
            and self.contract_hash != self.content_hash()
        ):
            raise ValueError("V6.1 success contract hash differs")
        self.thresholds.assert_sealed()
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "contract_hash")

    @classmethod
    def seal(cls, **data: object) -> "ScientificSuccessContractV61":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"contract_hash"})
        payload["contract_hash"] = draft.content_hash()
        return cls(**payload)


class RollingConfirmationV61(StrictModel):
    schema_version: Literal["6.1-rolling-confirmation"] = (
        "6.1-rolling-confirmation"
    )
    adapter_id: Literal[
        "scalar_autonomous_ode_v52",
        "adaptive_positive_series_v57",
    ]
    status: DimensionStatusV61
    observation_count: Annotated[int, Field(ge=0)]
    requested_fold_count: Annotated[int, Field(ge=1)]
    completed_fold_count: Annotated[int, Field(ge=0)]
    admissible_fold_count: Annotated[int, Field(ge=0)]
    selected_model_ids: list[str]
    checks: dict[Identifier, bool]
    metrics: dict[Identifier, float | int | None]
    thresholds: dict[Identifier, float | int]
    reason_codes: list[Identifier]
    actual_values_hash: Sha256 | None = None
    prediction_values_hash: Sha256 | None = None
    persistence_values_hash: Sha256 | None = None
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_confirmation(self) -> "RollingConfirmationV61":
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError(
                "rolling-confirmation reasons must be sorted and unique"
            )
        if self.status == "PASS" and (
            not self.checks or not all(self.checks.values())
        ):
            raise ValueError(
                "passing rolling confirmation contains a failed check"
            )
        if (
            self.evidence_hash
            and self.evidence_hash != self.content_hash()
        ):
            raise ValueError("rolling-confirmation hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evidence_hash")

    @classmethod
    def seal(cls, **data: object) -> "RollingConfirmationV61":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"evidence_hash"})
        payload["evidence_hash"] = draft.content_hash()
        return cls(**payload)


class ScientificSuccessDimensionV61(StrictModel):
    dimension_id: Identifier
    status: DimensionStatusV61
    required_for_claim: bool
    reason_codes: list[Identifier]
    evidence_refs: list[Sha256]
    metrics: dict[Identifier, float | int | bool | None] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_dimension(self) -> "ScientificSuccessDimensionV61":
        if self.dimension_id not in _DIMENSION_IDS:
            raise ValueError("unknown scientific-success dimension")
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("dimension reasons must be sorted and unique")
        if self.evidence_refs != sorted(set(self.evidence_refs)):
            raise ValueError("dimension evidence refs must be sorted and unique")
        return self


class ScientificSuccessReportV61(StrictModel):
    schema_version: Literal["6.1-success-report"] = "6.1-success-report"
    workspace_spec_hash: Sha256
    contract_hash: Sha256
    adapter_id: Literal[
        "scalar_autonomous_ode_v52",
        "adaptive_positive_series_v57",
    ]
    claim_kind: ClaimKindV61
    current_gate_hashes: dict[StageId, Sha256]
    adapter_binding_hash: Sha256
    scientific_bundle_hash: Sha256
    fixture_only: bool
    dimensions: list[ScientificSuccessDimensionV61]
    rolling_confirmation: RollingConfirmationV61
    local_predictive_gate_status: DimensionStatusV61
    scientific_success_status: DimensionStatusV61
    claim_ceiling: ClaimCeilingV61
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    report_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "ScientificSuccessReportV61":
        dimension_ids = [item.dimension_id for item in self.dimensions]
        if dimension_ids != sorted(_DIMENSION_IDS):
            raise ValueError(
                "success report must contain every ordered dimension"
            )
        if (
            self.report_hash
            and self.report_hash != self.content_hash()
        ):
            raise ValueError("V6.1 success report hash differs")
        if self.scientific_qualification_granted:
            raise ValueError("V6.1 local success gate cannot grant qualification")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "report_hash")

    @classmethod
    def seal(cls, **data: object) -> "ScientificSuccessReportV61":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"report_hash"})
        payload["report_hash"] = draft.content_hash()
        return cls(**payload)


def default_scientific_success_contract_v61(
    *,
    workspace_spec_hash: str,
    adapter_id: str,
) -> ScientificSuccessContractV61:
    if adapter_id not in {ODE_ADAPTER_ID, ADAPTIVE_ADAPTER_ID}:
        raise ValueError("no V6.1 success contract for adapter")
    return ScientificSuccessContractV61.seal(
        workspace_spec_hash=workspace_spec_hash,
        adapter_id=adapter_id,
        claim_kind="predictive",
        required_dimension_ids=sorted(
            [
                "workflow_integrity",
                "data_provenance",
                "local_adapter_checks",
                "leakage_safe_confirmation",
            ]
        ),
        thresholds=ScientificSuccessThresholdsV61.seal(),
    )


def _frozen_adaptive_thresholds() -> tuple[
    HybridODEThresholdsV56,
    AdaptiveThresholdsV57,
]:
    root = Path(__file__).resolve().parents[2]
    primary = HybridODEThresholdsV56.seal(
        **json.loads(
            (root / "V5_6_HYBRID_THRESHOLDS.json").read_text(
                encoding="utf-8"
            )
        )
    )
    adaptive = AdaptiveThresholdsV57.seal(
        **json.loads(
            (root / "V5_7_ADAPTIVE_THRESHOLDS.json").read_text(
                encoding="utf-8"
            )
        )
    )
    primary.assert_sealed()
    adaptive.assert_sealed()
    return primary, adaptive


def _fold_forecast(
    *,
    adapter_id: str,
    snapshot: ODETimeSeriesSnapshotV52,
) -> tuple[float, float | None, float | None, str, bool]:
    if adapter_id == ODE_ADAPTER_ID:
        bundle = build_ode_bundle_v52(
            snapshot=snapshot,
            thresholds=ODEThresholdsV52.seal(bootstrap_replicates=40),
        )
        selected = next(
            item
            for item in bundle.candidates
            if item.candidate_id == bundle.selected_candidate_id
        )
        l4 = next(item for item in bundle.levels if item.level == "L4")
        admissible = all(
            item.status == "PASS"
            for item in bundle.levels
            if item.level != "L0"
        )
        return (
            float(selected.forecast_value),
            _finite_metric(l4.metrics.get("forecast_interval_low")),
            _finite_metric(l4.metrics.get("forecast_interval_high")),
            selected.candidate_id,
            admissible,
        )
    if adapter_id == ADAPTIVE_ADAPTER_ID:
        primary, adaptive = _frozen_adaptive_thresholds()
        bundle = build_adaptive_positive_series_bundle_v57(
            snapshot=snapshot,
            primary_thresholds=primary,
            adaptive_thresholds=adaptive,
        )
        branch = bundle.graph.selected_branch
        model_id = bundle.graph.selected_model_id
        if branch == "log_growth":
            forecast = next(
                item.forecast_value
                for item in bundle.growth_candidates
                if item.candidate_id == model_id
            )
        elif branch == "hybrid_ode":
            forecast = next(
                item.forecast_value
                for item in bundle.primary_bundle.candidates
                if item.candidate_id == model_id
            )
        else:
            forecast = math.nan
        l4 = next(item for item in bundle.levels if item.level == "L4")
        admissible = branch != "unresolved" and all(
            item.status == "PASS"
            for item in bundle.levels
            if item.level != "L0"
        )
        return (
            float(forecast),
            _finite_metric(l4.metrics.get("forecast_interval_low")),
            _finite_metric(l4.metrics.get("forecast_interval_high")),
            f"{branch}:{model_id}",
            admissible,
        )
    raise ValueError("rolling confirmation adapter is not registered")


def _finite_metric(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def evaluate_rolling_confirmation_v61(
    *,
    snapshot: ODETimeSeriesSnapshotV52,
    contract: ScientificSuccessContractV61,
) -> RollingConfirmationV61:
    snapshot.assert_sealed()
    contract.thresholds.assert_sealed()
    thresholds = contract.thresholds
    adaptive = contract.adapter_id == ADAPTIVE_ADAPTER_ID
    fold_count = (
        thresholds.adaptive_confirmation_folds
        if adaptive
        else thresholds.ode_confirmation_folds
    )
    minimum_history = (
        thresholds.adaptive_minimum_history_points
        if adaptive
        else thresholds.ode_minimum_history_points
    )
    required = minimum_history + fold_count
    observation_count = len(snapshot.observations)
    common_thresholds = {
        "minimum_observation_count": required,
        "requested_fold_count": fold_count,
        "maximum_confirmation_relative_rmse": (
            thresholds.maximum_confirmation_relative_rmse
        ),
        "minimum_persistence_relative_improvement": (
            thresholds.minimum_persistence_relative_improvement
        ),
        "maximum_absolute_residual_lag1_correlation": (
            thresholds.maximum_absolute_residual_lag1_correlation
        ),
        "minimum_confirmation_interval_coverage": (
            thresholds.minimum_confirmation_interval_coverage
        ),
        "minimum_admissible_fold_fraction": (
            thresholds.minimum_admissible_fold_fraction
        ),
    }
    if observation_count < required:
        return RollingConfirmationV61.seal(
            adapter_id=contract.adapter_id,
            status="NOT_RUN",
            observation_count=observation_count,
            requested_fold_count=fold_count,
            completed_fold_count=0,
            admissible_fold_count=0,
            selected_model_ids=[],
            checks={},
            metrics={
                "minimum_observation_count": required,
                "observed_observation_count": observation_count,
            },
            thresholds=common_thresholds,
            reason_codes=["insufficient_observations"],
        )

    times = np.asarray(snapshot.times, dtype=float)
    observations = np.asarray(snapshot.observations, dtype=float)
    predictions: list[float] = []
    actuals: list[float] = []
    persistence: list[float] = []
    selected_ids: list[str] = []
    covered: list[bool] = []
    admissible_count = 0
    failure_count = 0
    first = observation_count - fold_count
    for index in range(first, observation_count):
        prefix = ODETimeSeriesSnapshotV52.seal(
            task_id=f"{snapshot.task_id}-v61-fold-{index}",
            time_unit=snapshot.time_unit,
            state_unit=snapshot.state_unit,
            times=times[:index].tolist(),
            observations=observations[:index].tolist(),
            source_id=snapshot.source_id,
            fixture_only=snapshot.fixture_only,
        )
        try:
            prediction, low, high, selected_id, admissible = _fold_forecast(
                adapter_id=contract.adapter_id,
                snapshot=prefix,
            )
        except (ArithmeticError, RuntimeError, ValueError):
            failure_count += 1
            continue
        if not math.isfinite(prediction) or prediction <= 0:
            failure_count += 1
            continue
        actual = float(observations[index])
        predictions.append(prediction)
        actuals.append(actual)
        persistence.append(float(observations[index - 1]))
        selected_ids.append(selected_id)
        admissible_count += int(admissible)
        if low is not None and high is not None:
            covered.append(low <= actual <= high)

    completed = len(predictions)
    if completed == 0:
        return RollingConfirmationV61.seal(
            adapter_id=contract.adapter_id,
            status="FAIL",
            observation_count=observation_count,
            requested_fold_count=fold_count,
            completed_fold_count=0,
            admissible_fold_count=0,
            selected_model_ids=selected_ids,
            checks={
                "all_requested_folds_completed": False,
                "all_predictions_finite_positive": False,
            },
            metrics={"fold_failure_count": failure_count},
            thresholds=common_thresholds,
            reason_codes=["no_valid_confirmation_forecast"],
        )

    prediction_array = np.asarray(predictions, dtype=float)
    actual_array = np.asarray(actuals, dtype=float)
    persistence_array = np.asarray(persistence, dtype=float)
    residuals = actual_array - prediction_array
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    persistence_rmse = float(
        np.sqrt(np.mean(np.square(actual_array - persistence_array)))
    )
    relative_rmse = rmse / max(float(np.mean(np.abs(actual_array))), 1e-12)
    improvement = 1.0 - rmse / max(persistence_rmse, 1e-12)
    residual_lag = abs(_lag1(residuals))
    coverage = (
        float(np.mean(covered)) if len(covered) == fold_count else None
    )
    admissible_fraction = admissible_count / fold_count
    checks = {
        "all_requested_folds_completed": completed == fold_count,
        "all_predictions_finite_positive": (
            completed == fold_count
            and bool(np.all(np.isfinite(prediction_array)))
            and bool(np.all(prediction_array > 0))
        ),
        "all_inner_selections_admissible": (
            admissible_fraction
            >= thresholds.minimum_admissible_fold_fraction
        ),
        "confirmation_error_bounded": (
            relative_rmse
            <= thresholds.maximum_confirmation_relative_rmse
        ),
        "persistence_baseline_improved": (
            improvement
            >= thresholds.minimum_persistence_relative_improvement
        ),
        "confirmation_residual_lag_bounded": (
            residual_lag
            <= thresholds.maximum_absolute_residual_lag1_correlation
        ),
        "confirmation_interval_coverage": (
            coverage is not None
            and coverage
            >= thresholds.minimum_confirmation_interval_coverage
        ),
    }
    reasons = sorted(
        check_id for check_id, passed in checks.items() if not passed
    )
    return RollingConfirmationV61.seal(
        adapter_id=contract.adapter_id,
        status="PASS" if all(checks.values()) else "FAIL",
        observation_count=observation_count,
        requested_fold_count=fold_count,
        completed_fold_count=completed,
        admissible_fold_count=admissible_count,
        selected_model_ids=selected_ids,
        checks=checks,
        metrics={
            "confirmation_rmse": rmse,
            "confirmation_relative_rmse": relative_rmse,
            "persistence_rmse": persistence_rmse,
            "persistence_relative_improvement": improvement,
            "absolute_confirmation_residual_lag1_correlation": residual_lag,
            "confirmation_interval_coverage": coverage,
            "admissible_fold_fraction": admissible_fraction,
            "fold_failure_count": failure_count,
        },
        thresholds=common_thresholds,
        reason_codes=reasons,
        actual_values_hash=sha256_value(actuals),
        prediction_values_hash=sha256_value(predictions),
        persistence_values_hash=sha256_value(persistence),
    )


def _dimension(
    *,
    dimension_id: str,
    status: DimensionStatusV61,
    required: bool,
    reasons: list[str],
    evidence_refs: list[str],
    metrics: dict[str, float | int | bool | None] | None = None,
) -> ScientificSuccessDimensionV61:
    return ScientificSuccessDimensionV61(
        dimension_id=dimension_id,
        status=status,
        required_for_claim=required,
        reason_codes=sorted(set(reasons)),
        evidence_refs=sorted(set(evidence_refs)),
        metrics=metrics or {},
    )


def _aggregate_status(
    dimensions: list[ScientificSuccessDimensionV61],
    required_ids: set[str],
) -> DimensionStatusV61:
    statuses = [
        item.status
        for item in dimensions
        if item.dimension_id in required_ids
    ]
    if any(item == "FAIL" for item in statuses):
        return "FAIL"
    if any(item == "HUMAN" for item in statuses):
        return "HUMAN"
    if any(item == "NOT_RUN" for item in statuses):
        return "NOT_RUN"
    return "PASS"


def evaluate_scientific_success_v61(
    workspace: StageWorkspaceV50,
) -> ScientificSuccessReportV61:
    root = workspace.root
    contract_path = root / SUCCESS_CONTRACT_PATH
    binding_path = root / "docs" / "adapter_binding.json"
    snapshot_path = root / "data" / "processed" / "ode_snapshot.json"
    if not all(
        path.is_file()
        for path in (contract_path, binding_path, snapshot_path)
    ):
        raise FileNotFoundError(
            "V6.1 success evaluation needs contract, binding, and snapshot"
        )
    contract = ScientificSuccessContractV61.model_validate_json(
        contract_path.read_text(encoding="utf-8")
    )
    if contract.workspace_spec_hash != workspace.spec.spec_hash:
        raise ValueError("success contract belongs to another workspace")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if binding.get("adapter_id") != contract.adapter_id:
        raise ValueError("success contract and adapter binding differ")
    snapshot = ODETimeSeriesSnapshotV52.model_validate_json(
        snapshot_path.read_text(encoding="utf-8")
    )
    if contract.adapter_id == ADAPTIVE_ADAPTER_ID:
        bundle_path = root / "results" / "adaptive_positive_series_bundle.json"
        bundle = AdaptivePositiveSeriesBundleV57.model_validate_json(
            bundle_path.read_text(encoding="utf-8")
        )
    else:
        bundle_path = root / "results" / "ode_scientific_bundle.json"
        bundle = ODEScientificBundleV52.model_validate_json(
            bundle_path.read_text(encoding="utf-8")
        )

    current_gate_hashes = {
        stage: str(gate)
        for stage in _STAGES
        if (gate := workspace.current_gate(stage)) is not None
    }
    workflow_pass = workspace.verify() and set(current_gate_hashes) == set(
        _STAGES
    )
    recomputed_confirmation = evaluate_rolling_confirmation_v61(
        snapshot=snapshot,
        contract=contract,
    )
    rolling_path = root / ROLLING_CONFIRMATION_PATH
    if rolling_path.is_file():
        confirmation = RollingConfirmationV61.model_validate_json(
            rolling_path.read_text(encoding="utf-8")
        )
        if confirmation != recomputed_confirmation:
            raise ValueError(
                "persisted rolling confirmation differs from deterministic "
                "recomputation"
            )
    else:
        confirmation = recomputed_confirmation
    required_ids = set(contract.required_dimension_ids)
    dimensions = [
        _dimension(
            dimension_id="workflow_integrity",
            status="PASS" if workflow_pass else "FAIL",
            required=True,
            reasons=[] if workflow_pass else ["workflow_not_fully_verified"],
            evidence_refs=list(current_gate_hashes.values()),
            metrics={
                "current_gate_count": len(current_gate_hashes),
                "graph_verified": workspace.verify(),
            },
        ),
        _dimension(
            dimension_id="data_provenance",
            status="NOT_RUN" if snapshot.fixture_only else "HUMAN",
            required=True,
            reasons=(
                ["fixture_only"]
                if snapshot.fixture_only
                else ["independent_source_review_absent"]
            ),
            evidence_refs=[str(snapshot.snapshot_hash)],
            metrics={"fixture_only": snapshot.fixture_only},
        ),
        _dimension(
            dimension_id="local_adapter_checks",
            status="PASS" if bundle.scientific_acceptance else "FAIL",
            required=True,
            reasons=(
                []
                if bundle.scientific_acceptance
                else ["local_l0_l4_not_all_pass"]
            ),
            evidence_refs=[str(bundle.bundle_hash)],
            metrics={
                "local_l0_l4_all_pass": bundle.scientific_acceptance,
            },
        ),
        _dimension(
            dimension_id="leakage_safe_confirmation",
            status=confirmation.status,
            required=True,
            reasons=confirmation.reason_codes,
            evidence_refs=(
                [str(confirmation.evidence_hash)]
                if confirmation.evidence_hash
                else []
            ),
            metrics={
                "completed_fold_count": confirmation.completed_fold_count,
                "requested_fold_count": confirmation.requested_fold_count,
                "confirmation_relative_rmse": confirmation.metrics.get(
                    "confirmation_relative_rmse"
                ),
                "persistence_relative_improvement": confirmation.metrics.get(
                    "persistence_relative_improvement"
                ),
            },
        ),
        _dimension(
            dimension_id="decision_value",
            status="NOT_RUN",
            required=False,
            reasons=["decision_regret_evaluator_absent"],
            evidence_refs=[],
        ),
        _dimension(
            dimension_id="mechanism_identification",
            status="NOT_RUN",
            required=False,
            reasons=["causal_identification_not_established"],
            evidence_refs=[str(bundle.bundle_hash)],
        ),
        _dimension(
            dimension_id="external_generalization",
            status="NOT_RUN",
            required=False,
            reasons=["external_dataset_evaluation_absent"],
            evidence_refs=[],
        ),
        _dimension(
            dimension_id="scientific_qualification",
            status="NOT_RUN",
            required=False,
            reasons=["external_private_promotion_absent"],
            evidence_refs=[],
        ),
    ]
    dimensions = sorted(dimensions, key=lambda item: item.dimension_id)
    local_predictive_status = _aggregate_status(
        dimensions,
        {
            "workflow_integrity",
            "local_adapter_checks",
            "leakage_safe_confirmation",
        },
    )
    scientific_status = _aggregate_status(dimensions, required_ids)
    if snapshot.fixture_only:
        ceiling: ClaimCeilingV61 = "fixture_protocol_only"
    elif scientific_status == "PASS":
        ceiling = "local_leakage_safe_predictive_evidence"
    elif bundle.scientific_acceptance:
        ceiling = "local_retrospective_adapter_evidence"
    elif workflow_pass:
        ceiling = "workflow_integrity_only"
    else:
        ceiling = "no_scientific_claim"
    return ScientificSuccessReportV61.seal(
        workspace_spec_hash=workspace.spec.spec_hash,
        contract_hash=contract.contract_hash,
        adapter_id=contract.adapter_id,
        claim_kind=contract.claim_kind,
        current_gate_hashes=current_gate_hashes,
        adapter_binding_hash=_file_hash(binding_path),
        scientific_bundle_hash=bundle.bundle_hash,
        fixture_only=snapshot.fixture_only,
        dimensions=dimensions,
        rolling_confirmation=confirmation,
        local_predictive_gate_status=local_predictive_status,
        scientific_success_status=scientific_status,
        claim_ceiling=ceiling,
    )


def materialize_scientific_success_v61(
    workspace: StageWorkspaceV50,
) -> tuple[ScientificSuccessReportV61, str]:
    report = evaluate_scientific_success_v61(workspace)
    evidence = workspace.commit_evidence(
        "scientific_success_report_v61",
        report.model_dump(mode="json"),
    )
    path = workspace.root / SUCCESS_PROJECTION_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        canonical_json(report) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)
    return report, evidence.sha256


def scientific_success_summary_v61(
    workspace: StageWorkspaceV50,
) -> dict[str, Any]:
    path = workspace.root / SUCCESS_PROJECTION_PATH
    if not path.is_file():
        return {
            "schema_version": "6.1",
            "evaluated": False,
            "claim_kind": "predictive",
            "local_predictive_gate_status": "NOT_RUN",
            "scientific_success_status": "NOT_RUN",
            "claim_ceiling": "no_scientific_claim",
            "dimensions": {},
            "confirmation": None,
            "scientific_qualification_granted": False,
            "real_world_action_authorized": False,
        }
    try:
        report = ScientificSuccessReportV61.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return {
            "schema_version": "6.1",
            "evaluated": False,
            "claim_kind": "predictive",
            "local_predictive_gate_status": "NOT_RUN",
            "scientific_success_status": "NOT_RUN",
            "claim_ceiling": "no_scientific_claim",
            "dimensions": {},
            "confirmation": None,
            "scientific_qualification_granted": False,
            "real_world_action_authorized": False,
        }
    current_gates = {
        stage: str(gate)
        for stage in _STAGES
        if (gate := workspace.current_gate(stage)) is not None
    }
    current = (
        report.workspace_spec_hash == workspace.spec.spec_hash
        and report.current_gate_hashes == current_gates
        and report.report_hash == report.content_hash()
    )
    if not current:
        return {
            "schema_version": "6.1",
            "evaluated": False,
            "claim_kind": report.claim_kind,
            "local_predictive_gate_status": "NOT_RUN",
            "scientific_success_status": "NOT_RUN",
            "claim_ceiling": "no_scientific_claim",
            "dimensions": {},
            "confirmation": None,
            "scientific_qualification_granted": False,
            "real_world_action_authorized": False,
        }
    return {
        "schema_version": "6.1",
        "evaluated": True,
        "claim_kind": report.claim_kind,
        "local_predictive_gate_status": report.local_predictive_gate_status,
        "scientific_success_status": report.scientific_success_status,
        "claim_ceiling": report.claim_ceiling,
        "fixture_only": report.fixture_only,
        "dimensions": {
            item.dimension_id: {
                "status": item.status,
                "required_for_claim": item.required_for_claim,
                "reason_codes": item.reason_codes,
                "metrics": item.metrics,
            }
            for item in report.dimensions
        },
        "confirmation": {
            "status": report.rolling_confirmation.status,
            "completed_fold_count": (
                report.rolling_confirmation.completed_fold_count
            ),
            "requested_fold_count": (
                report.rolling_confirmation.requested_fold_count
            ),
            "selected_model_ids": (
                report.rolling_confirmation.selected_model_ids
            ),
            "metrics": report.rolling_confirmation.metrics,
            "reason_codes": report.rolling_confirmation.reason_codes,
        },
        "report_hash": report.report_hash,
        "scientific_qualification_granted": False,
        "real_world_action_authorized": False,
    }


__all__ = [
    "ADAPTIVE_ADAPTER_ID",
    "ODE_ADAPTER_ID",
    "ROLLING_CONFIRMATION_PATH",
    "SUCCESS_CONTRACT_PATH",
    "SUCCESS_PROJECTION_PATH",
    "RollingConfirmationV61",
    "ScientificSuccessContractV61",
    "ScientificSuccessDimensionV61",
    "ScientificSuccessReportV61",
    "ScientificSuccessThresholdsV61",
    "default_scientific_success_contract_v61",
    "evaluate_rolling_confirmation_v61",
    "evaluate_scientific_success_v61",
    "materialize_scientific_success_v61",
    "scientific_success_summary_v61",
]
