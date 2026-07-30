"""Thin development-only V6.9 runtime for the frozen V6.8 two-pack portfolio.

The module is intentionally an additive wrapper.  It does not alter V6.8
schemas, admit a capability to the stage workflow, close L0, grant scientific
qualification, or authorize a real-world action.

Application code owns the public series snapshot, common rolling origins,
input-bound recomputation, persistence baseline, and final abstention rule.
The two existing V6.8 packs only fit their registered candidates and emit
one-step forecasts.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5_2.ode_system import ODEThresholdsV52, ODETimeSeriesSnapshotV52

from .capability_catalog_v68 import (
    POSITIVE_LOG_INCREMENT_MANIFEST_ID_V68,
    SCALAR_ODE_MANIFEST_ID_V68,
)
from .capability_sdk_v68 import CapabilityQueryV68, CapabilityRegistryV68
from .portfolio_protocol_v68 import (
    BranchBudgetV68,
    BranchOuterEvaluationV68,
    CommonLossContractV68,
    ModelingPortfolioProtocolV68,
    OuterSelectionPolicyV68,
    PortfolioBranchRequestV68,
    PortfolioBranchV68,
    PortfolioBudgetV68,
    PortfolioSelectionDecisionV68,
    compile_modeling_portfolio_protocol_v68,
    execute_outer_selection_v68,
    one_step_rmse_semantic_hash_v68,
    one_step_rmse_v68,
    outer_selector_semantic_hash_v68,
)
from .positive_log_increment_v68 import (
    PositiveLogIncrementThresholdsV68,
    PositiveScalarSeriesSnapshotV68,
    compile_positive_log_increment_ir_v68,
    execute_positive_log_increment_ir_v68,
)
from .positive_log_increment_verifier_v68 import (
    recompute_positive_log_increment_level_v68,
)
from .scalar_ode_pack_v68 import (
    compile_scalar_autonomous_ode_ir_v68,
    execute_scalar_autonomous_ode_ir_v68,
)
from .scalar_ode_verifier_v68 import (
    recompute_scalar_autonomous_ode_level_v68,
)


MINIMUM_INITIAL_TRAINING_V69 = 34
MINIMUM_SERIES_LENGTH_V69 = MINIMUM_INITIAL_TRAINING_V69 + 1
MAXIMUM_ROLLING_ORIGINS_V69 = 32
_ODE_PACK_ID = "scalar_autonomous_ode_v52"
_LOG_INCREMENT_PACK_ID = "positive_log_increment_v68"
_EXPECTED_PACK_IDS = {_ODE_PACK_ID, _LOG_INCREMENT_PACK_ID}
_ODE_PARAMETER_COUNTS = {
    "constant": 0,
    "exponential": 1,
    "gompertz": 2,
    "logistic": 2,
}
_LOG_INCREMENT_PARAMETER_COUNTS = {
    "log_growth_ar1": 2,
    "log_random_walk_drift": 1,
}

FiniteNumberV69 = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFiniteV69 = Annotated[
    float,
    Field(ge=0, allow_inf_nan=False),
]
ExecutionStatusV69 = Literal["PASS", "FAIL"]
FinalDecisionV69 = Literal["SELECT", "ABSTAIN"]
FailurePhaseV69 = Literal[
    "pack-execution",
    "input-bound-verification",
    "inner-selection",
]


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _sealed(model_type, data: dict[str, object], hash_field: str):
    draft = model_type(**data)
    payload = draft.model_dump(mode="json", exclude={hash_field})
    payload[hash_field] = draft.content_hash()
    return model_type(**payload)


class PositiveSeriesSnapshotV69(StrictModel):
    """One public, ordered, positive scalar series used only in development."""

    schema_version: Literal["6.9-positive-series-snapshot"] = (
        "6.9-positive-series-snapshot"
    )
    task_id: Identifier
    time_unit: Identifier
    state_unit: Identifier
    times: Annotated[
        list[FiniteNumberV69],
        Field(min_length=MINIMUM_SERIES_LENGTH_V69),
    ]
    observations: Annotated[
        list[FiniteNumberV69],
        Field(min_length=MINIMUM_SERIES_LENGTH_V69),
    ]
    source_id: Annotated[str, Field(min_length=3, max_length=500)]
    public_data_only: Literal[True] = True
    private_acceptance_data_accessed: Literal[False] = False
    development_only: Literal[True] = True
    snapshot_is_scientific_evidence: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    snapshot_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_snapshot(self) -> "PositiveSeriesSnapshotV69":
        if len(self.times) != len(self.observations):
            raise ValueError("V6.9 times and observations must have equal length")
        if any(
            right <= left for left, right in zip(self.times, self.times[1:])
        ):
            raise ValueError("V6.9 times must be strictly increasing")
        if any(value <= 0 for value in self.observations):
            raise ValueError("V6.9 observations must be positive")
        if self.snapshot_hash and self.snapshot_hash != self.content_hash():
            raise ValueError("V6.9 snapshot hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "snapshot_hash")

    @classmethod
    def seal(cls, **data: object) -> "PositiveSeriesSnapshotV69":
        return _sealed(cls, data, "snapshot_hash")

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if not self.snapshot_hash or self.snapshot_hash != self.content_hash():
            raise ValueError("V6.9 snapshot is not sealed")


class RollingOriginV69(StrictModel):
    """One one-step target with an exact common training prefix."""

    schema_version: Literal["6.9-rolling-origin"] = "6.9-rolling-origin"
    origin_id: Identifier
    training_count: Annotated[
        int,
        Field(ge=MINIMUM_INITIAL_TRAINING_V69),
    ]
    target_index: Annotated[int, Field(ge=MINIMUM_INITIAL_TRAINING_V69)]
    training_view_hash: Sha256

    @model_validator(mode="after")
    def validate_origin(self) -> "RollingOriginV69":
        if self.target_index != self.training_count:
            raise ValueError("V6.9 rolling origin must be one step ahead")
        return self


class CommonRollingOriginPlanV69(StrictModel):
    """A sealed public plan shared exactly by every portfolio branch."""

    schema_version: Literal["6.9-common-rolling-origin-plan"] = (
        "6.9-common-rolling-origin-plan"
    )
    snapshot_hash: Sha256
    observation_count: Annotated[
        int,
        Field(ge=MINIMUM_SERIES_LENGTH_V69),
    ]
    initial_training_count: Annotated[
        int,
        Field(ge=MINIMUM_INITIAL_TRAINING_V69),
    ]
    forecast_horizon_steps: Literal[1] = 1
    origins: Annotated[list[RollingOriginV69], Field(min_length=1)]
    common_prefix_rule: Literal[
        "all_branches_receive_identical_ordered_prefixes"
    ] = "all_branches_receive_identical_ordered_prefixes"
    plan_is_scientific_evidence: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    plan_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_plan(self) -> "CommonRollingOriginPlanV69":
        expected_counts = list(
            range(
                self.initial_training_count,
                self.initial_training_count + len(self.origins),
            )
        )
        if [item.training_count for item in self.origins] != expected_counts:
            raise ValueError("V6.9 rolling origins must be contiguous and ordered")
        if self.origins[-1].target_index >= self.observation_count:
            raise ValueError("V6.9 rolling origin exceeds the snapshot")
        origin_ids = [item.origin_id for item in self.origins]
        if origin_ids != list(dict.fromkeys(origin_ids)):
            raise ValueError("V6.9 rolling origin IDs must be ordered and unique")
        if self.plan_hash and self.plan_hash != self.content_hash():
            raise ValueError("V6.9 rolling-origin plan hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "plan_hash")

    @classmethod
    def seal(cls, **data: object) -> "CommonRollingOriginPlanV69":
        return _sealed(cls, data, "plan_hash")

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if not self.plan_hash or self.plan_hash != self.content_hash():
            raise ValueError("V6.9 rolling-origin plan is not sealed")


def _training_view_hash_v69(
    snapshot: PositiveSeriesSnapshotV69,
    training_count: int,
) -> str:
    return sha256_value(
        {
            "snapshot_hash": snapshot.snapshot_hash,
            "training_count": training_count,
            "times": snapshot.times[:training_count],
            "observations": snapshot.observations[:training_count],
        }
    )


def build_common_rolling_origin_plan_v69(
    snapshot: PositiveSeriesSnapshotV69,
    *,
    initial_training_count: int = MINIMUM_INITIAL_TRAINING_V69,
    maximum_origins: int | None = None,
) -> CommonRollingOriginPlanV69:
    """Freeze deterministic, contiguous one-step origins over one snapshot."""

    snapshot.assert_sealed()
    if initial_training_count < MINIMUM_INITIAL_TRAINING_V69:
        raise ValueError("V6.9 initial training count must be at least 34")
    available = len(snapshot.observations) - initial_training_count
    if available < 1:
        raise ValueError("V6.9 snapshot has no observation after initial training")
    if maximum_origins is not None and not (
        1 <= maximum_origins <= MAXIMUM_ROLLING_ORIGINS_V69
    ):
        raise ValueError(
            "V6.9 maximum_origins must be between 1 and "
            f"{MAXIMUM_ROLLING_ORIGINS_V69}"
        )
    origin_count = (
        min(available, MAXIMUM_ROLLING_ORIGINS_V69)
        if maximum_origins is None
        else min(available, maximum_origins)
    )
    origins = [
        RollingOriginV69(
            origin_id=f"origin-{target_index:06d}",
            training_count=target_index,
            target_index=target_index,
            training_view_hash=_training_view_hash_v69(
                snapshot,
                target_index,
            ),
        )
        for target_index in range(
            initial_training_count,
            initial_training_count + origin_count,
        )
    ]
    return CommonRollingOriginPlanV69.seal(
        snapshot_hash=str(snapshot.snapshot_hash),
        observation_count=len(snapshot.observations),
        initial_training_count=initial_training_count,
        origins=origins,
    )


class PersistenceBaselinePolicyV69(StrictModel):
    """Positive improvement required beyond the common persistence baseline."""

    schema_version: Literal["6.9-persistence-baseline-policy"] = (
        "6.9-persistence-baseline-policy"
    )
    baseline_id: Literal["persistence-last-value"] = "persistence-last-value"
    minimum_relative_improvement: Annotated[
        float,
        Field(gt=0, le=1, allow_inf_nan=False),
    ] = 0.01
    equality_passes: Literal[False] = False
    policy_is_scientific_evidence: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "PersistenceBaselinePolicyV69":
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("V6.9 persistence policy hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    @classmethod
    def seal(cls, **data: object) -> "PersistenceBaselinePolicyV69":
        return _sealed(cls, data, "policy_hash")

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("V6.9 persistence policy is not sealed")


class OriginExecutionReceiptV69(StrictModel):
    """One input-bound pack execution at one frozen rolling origin."""

    schema_version: Literal["6.9-origin-execution-receipt"] = (
        "6.9-origin-execution-receipt"
    )
    protocol_hash: Sha256
    snapshot_hash: Sha256
    origin_plan_hash: Sha256
    branch_id: Identifier
    manifest_id: Identifier
    manifest_hash: Sha256
    capability_pack_id: Identifier
    origin_id: Identifier
    training_view_hash: Sha256
    training_snapshot_hash: Sha256
    model_ir_hash: Sha256
    output_bundle_hash: Sha256 | None
    verifier_evidence_hash: Sha256 | None
    selected_model_id: Identifier | None
    prediction: FiniteNumberV69 | None
    parameter_count: Annotated[int, Field(ge=0)] | None
    execution_status: ExecutionStatusV69
    failure_reason: Identifier | None = None
    failure_phase: FailurePhaseV69 | None = None
    failure_evidence_hash: Sha256 | None = None
    underlying_pack_forced_fixture_control: Literal[True] = True
    receipt_is_scientific_evidence: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "OriginExecutionReceiptV69":
        if self.execution_status == "PASS":
            if (
                self.selected_model_id is None
                or self.prediction is None
                or self.parameter_count is None
                or self.output_bundle_hash is None
                or self.verifier_evidence_hash is None
                or self.failure_reason is not None
                or self.failure_phase is not None
                or self.failure_evidence_hash is not None
            ):
                raise ValueError("passing V6.9 origin receipt is incomplete")
            if self.prediction <= 0:
                raise ValueError("V6.9 origin prediction must be positive")
        elif (
            self.selected_model_id is not None
            or self.prediction is not None
            or self.parameter_count is not None
            or self.failure_reason is None
            or self.failure_phase is None
            or self.failure_evidence_hash is None
        ):
            raise ValueError("failed V6.9 origin receipt is inconsistent")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("V6.9 origin receipt hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    @classmethod
    def seal(cls, **data: object) -> "OriginExecutionReceiptV69":
        return _sealed(cls, data, "receipt_hash")

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("V6.9 origin receipt is not sealed")


class BranchExecutionReceiptV69(StrictModel):
    """Aggregate execution evidence used as the V6.8 evidence hash."""

    schema_version: Literal["6.9-branch-execution-receipt"] = (
        "6.9-branch-execution-receipt"
    )
    protocol_hash: Sha256
    snapshot_hash: Sha256
    origin_plan_hash: Sha256
    branch_id: Identifier
    manifest_id: Identifier
    manifest_hash: Sha256
    capability_pack_id: Identifier
    origin_receipts: Annotated[
        list[OriginExecutionReceiptV69],
        Field(min_length=1),
    ]
    outer_origin_ids: Annotated[list[Identifier], Field(min_length=1)]
    predictions: list[FiniteNumberV69]
    parameter_count: Annotated[int, Field(ge=0)]
    execution_status: ExecutionStatusV69
    failed_origin_ids: list[Identifier]
    failure_reason: Identifier | None = None
    receipt_is_scientific_evidence: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "BranchExecutionReceiptV69":
        for item in self.origin_receipts:
            item.assert_sealed()
            if (
                item.protocol_hash != self.protocol_hash
                or item.snapshot_hash != self.snapshot_hash
                or item.origin_plan_hash != self.origin_plan_hash
                or item.branch_id != self.branch_id
                or item.manifest_id != self.manifest_id
                or item.manifest_hash != self.manifest_hash
                or item.capability_pack_id != self.capability_pack_id
            ):
                raise ValueError("V6.9 origin receipt branch binding differs")
        receipt_origin_ids = [
            item.origin_id for item in self.origin_receipts
        ]
        if receipt_origin_ids != self.outer_origin_ids:
            raise ValueError("V6.9 branch origin view differs")
        expected_failed = [
            item.origin_id
            for item in self.origin_receipts
            if item.execution_status == "FAIL"
        ]
        if self.failed_origin_ids != expected_failed:
            raise ValueError("V6.9 failed origin IDs differ")
        if self.execution_status == "PASS":
            expected_predictions = [
                float(item.prediction)
                for item in self.origin_receipts
                if item.prediction is not None
            ]
            expected_parameters = max(
                int(item.parameter_count)
                for item in self.origin_receipts
                if item.parameter_count is not None
            )
            if (
                self.failed_origin_ids
                or self.predictions != expected_predictions
                or self.parameter_count != expected_parameters
                or self.failure_reason is not None
            ):
                raise ValueError("passing V6.9 branch receipt is inconsistent")
        elif (
            not self.failed_origin_ids
            or self.predictions
            or self.parameter_count != 0
            or self.failure_reason is None
        ):
            raise ValueError("failed V6.9 branch receipt is inconsistent")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("V6.9 branch receipt hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    @classmethod
    def seal(cls, **data: object) -> "BranchExecutionReceiptV69":
        return _sealed(cls, data, "receipt_hash")

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("V6.9 branch receipt is not sealed")


class PersistenceBaselineEvaluationV69(StrictModel):
    """Code-owned persistence score over the exact common outer view."""

    schema_version: Literal["6.9-persistence-baseline-evaluation"] = (
        "6.9-persistence-baseline-evaluation"
    )
    snapshot_hash: Sha256
    origin_plan_hash: Sha256
    baseline_id: Literal["persistence-last-value"] = "persistence-last-value"
    outer_origin_ids: Annotated[list[Identifier], Field(min_length=1)]
    observations: Annotated[list[FiniteNumberV69], Field(min_length=1)]
    predictions: Annotated[list[FiniteNumberV69], Field(min_length=1)]
    rmse: NonNegativeFiniteV69
    evaluation_is_scientific_evidence: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    evaluation_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_evaluation(self) -> "PersistenceBaselineEvaluationV69":
        if not (
            len(self.outer_origin_ids)
            == len(self.observations)
            == len(self.predictions)
        ):
            raise ValueError("V6.9 persistence data views differ")
        expected_rmse = one_step_rmse_v68(
            self.observations,
            self.predictions,
        )
        if not math.isclose(
            self.rmse,
            expected_rmse,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("V6.9 persistence RMSE differs")
        if self.evaluation_hash and (
            self.evaluation_hash != self.content_hash()
        ):
            raise ValueError("V6.9 persistence evaluation hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evaluation_hash")

    @classmethod
    def seal(cls, **data: object) -> "PersistenceBaselineEvaluationV69":
        return _sealed(cls, data, "evaluation_hash")

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if (
            not self.evaluation_hash
            or self.evaluation_hash != self.content_hash()
        ):
            raise ValueError("V6.9 persistence evaluation is not sealed")


class DevelopmentPortfolioRunV69(StrictModel):
    """Complete local run; a guarded selection is still not qualification."""

    schema_version: Literal["6.9-development-portfolio-run"] = (
        "6.9-development-portfolio-run"
    )
    protocol_hash: Sha256
    snapshot_hash: Sha256
    origin_plan_hash: Sha256
    baseline_policy: PersistenceBaselinePolicyV69
    branch_receipts: Annotated[
        list[BranchExecutionReceiptV69],
        Field(min_length=2, max_length=2),
    ]
    evaluations: Annotated[
        list[BranchOuterEvaluationV68],
        Field(min_length=2, max_length=2),
    ]
    persistence_baseline: PersistenceBaselineEvaluationV69
    inner_selection: PortfolioSelectionDecisionV68
    final_decision: FinalDecisionV69
    selected_branch_id: Identifier | None
    selected_common_loss: NonNegativeFiniteV69 | None
    persistence_relative_improvement: FiniteNumberV69 | None
    reason_code: Identifier
    development_only: Literal[True] = True
    run_is_stage_evidence: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    run_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_run(self) -> "DevelopmentPortfolioRunV69":
        self.baseline_policy.assert_sealed()
        self.persistence_baseline.assert_sealed()
        self.inner_selection.assert_sealed()
        receipt_ids = [item.branch_id for item in self.branch_receipts]
        evaluation_ids = [item.branch_id for item in self.evaluations]
        if receipt_ids != sorted(set(receipt_ids)):
            raise ValueError("V6.9 branch receipts must be sorted and unique")
        if evaluation_ids != receipt_ids:
            raise ValueError("V6.9 branch receipt/evaluation IDs differ")
        receipts = {item.branch_id: item for item in self.branch_receipts}
        for evaluation in self.evaluations:
            evaluation.assert_sealed()
            receipt = receipts[evaluation.branch_id]
            receipt.assert_sealed()
            if (
                evaluation.protocol_hash != self.protocol_hash
                or evaluation.execution_evidence_hash != receipt.receipt_hash
                or evaluation.execution_status != receipt.execution_status
                or evaluation.parameter_count != receipt.parameter_count
            ):
                raise ValueError("V6.9 outer evaluation receipt binding differs")
        if (
            self.persistence_baseline.snapshot_hash != self.snapshot_hash
            or self.persistence_baseline.origin_plan_hash
            != self.origin_plan_hash
            or self.inner_selection.protocol_hash != self.protocol_hash
        ):
            raise ValueError("V6.9 run input binding differs")
        if self.inner_selection.decision == "ABSTAIN":
            if (
                self.final_decision != "ABSTAIN"
                or self.selected_branch_id is not None
                or self.selected_common_loss is not None
                or self.persistence_relative_improvement is not None
                or self.reason_code != "inner-selection-abstained"
            ):
                raise ValueError("V6.9 inner abstention projection differs")
        else:
            inner_selected = str(self.inner_selection.selected_branch_id)
            expected_loss = self.inner_selection.common_loss_by_branch[
                inner_selected
            ]
            expected_improvement = _relative_improvement_v69(
                expected_loss,
                self.persistence_baseline.rmse,
            )
            if (
                self.selected_common_loss is None
                or not math.isclose(
                    self.selected_common_loss,
                    expected_loss,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
                or self.persistence_relative_improvement is None
                or not math.isclose(
                    self.persistence_relative_improvement,
                    expected_improvement,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("V6.9 baseline recomputation differs")
            passes = (
                expected_improvement
                > self.baseline_policy.minimum_relative_improvement
            )
            if passes:
                if (
                    self.final_decision != "SELECT"
                    or self.selected_branch_id != inner_selected
                    or self.reason_code != "baseline-guarded-unique-winner"
                ):
                    raise ValueError("V6.9 guarded selection differs")
            elif (
                self.final_decision != "ABSTAIN"
                or self.selected_branch_id is not None
                or self.reason_code != "persistence-baseline-not-beaten"
            ):
                raise ValueError("V6.9 baseline abstention differs")
        if self.run_hash and self.run_hash != self.content_hash():
            raise ValueError("V6.9 portfolio run hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "run_hash")

    @classmethod
    def seal(cls, **data: object) -> "DevelopmentPortfolioRunV69":
        return _sealed(cls, data, "run_hash")

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if not self.run_hash or self.run_hash != self.content_hash():
            raise ValueError("V6.9 portfolio run is not sealed")


def compile_default_development_portfolio_protocol_v69(
    *,
    query: CapabilityQueryV68,
    registry: CapabilityRegistryV68,
    branch_budget: BranchBudgetV68,
    portfolio_budget: PortfolioBudgetV68,
    tie_tolerance: float = 1e-12,
) -> ModelingPortfolioProtocolV68:
    """Freeze the exact two-pack protocol without accepting observations."""

    query.assert_sealed()
    branch_budget.assert_sealed()
    portfolio_budget.assert_sealed()
    if registry.runtime_mode != "development_sandbox":
        raise PermissionError("V6.9 default portfolio is development-only")
    registry_snapshot = registry.snapshot()
    entries = {item.manifest_id: item for item in registry_snapshot.entries}
    required = {
        POSITIVE_LOG_INCREMENT_MANIFEST_ID_V68,
        SCALAR_ODE_MANIFEST_ID_V68,
    }
    if not required <= set(entries):
        raise KeyError("V6.9 default registry lacks an exact capability manifest")
    common_loss = CommonLossContractV68.seal(
        loss_id="one_step_rmse",
        loss_implementation_ref="fma.v68.common_loss.one_step_rmse",
        loss_semantic_hash=one_step_rmse_semantic_hash_v68(),
        direction="minimize",
        loss_unit=query.measurement.measurement_unit,
        common_data_view_rule=(
            "Every branch receives the same content-addressed ordered series."
        ),
        common_evaluation_origin_rule=(
            "Every branch is scored on the same frozen outer rolling origins."
        ),
    )
    outer_selection = OuterSelectionPolicyV68.seal(
        policy_id="nested_one_step_rmse_v68",
        implementation_ref="fma.v68.selection.nested_one_step_rmse",
        implementation_semantic_hash=outer_selector_semantic_hash_v68(),
        tie_tolerance=tie_tolerance,
        minimum_completed_branches=2,
    )
    return compile_modeling_portfolio_protocol_v68(
        query=query,
        registry=registry,
        branch_requests=[
            PortfolioBranchRequestV68(
                branch_id="branch-log-increment",
                manifest_id=POSITIVE_LOG_INCREMENT_MANIFEST_ID_V68,
                manifest_hash=str(
                    entries[
                        POSITIVE_LOG_INCREMENT_MANIFEST_ID_V68
                    ].manifest_hash
                ),
                budget=branch_budget,
            ),
            PortfolioBranchRequestV68(
                branch_id="branch-scalar-ode",
                manifest_id=SCALAR_ODE_MANIFEST_ID_V68,
                manifest_hash=str(
                    entries[SCALAR_ODE_MANIFEST_ID_V68].manifest_hash
                ),
                budget=branch_budget,
            ),
        ],
        budget=portfolio_budget,
        common_loss=common_loss,
        outer_selection=outer_selection,
    )


def _assert_exact_development_protocol_v69(
    protocol: ModelingPortfolioProtocolV68,
) -> dict[str, PortfolioBranchV68]:
    protocol.assert_sealed()
    if protocol.runtime_mode != "development_sandbox":
        raise PermissionError("V6.9 runtime accepts development protocols only")
    branches = {item.capability_pack_id: item for item in protocol.branches}
    if len(protocol.branches) != 2 or set(branches) != _EXPECTED_PACK_IDS:
        raise ValueError("V6.9 runtime requires the exact ODE and log packs")
    return branches


def _assert_common_plan_v69(
    snapshot: PositiveSeriesSnapshotV69,
    plan: CommonRollingOriginPlanV69,
) -> None:
    snapshot.assert_sealed()
    plan.assert_sealed()
    if (
        plan.snapshot_hash != snapshot.snapshot_hash
        or plan.observation_count != len(snapshot.observations)
    ):
        raise ValueError("V6.9 plan uses a different snapshot data view")
    expected = build_common_rolling_origin_plan_v69(
        snapshot,
        initial_training_count=plan.initial_training_count,
        maximum_origins=len(plan.origins),
    )
    if expected != plan:
        raise ValueError("V6.9 plan differs from code-owned rolling origins")


def _origin_common_fields(
    *,
    protocol: ModelingPortfolioProtocolV68,
    snapshot: PositiveSeriesSnapshotV69,
    plan: CommonRollingOriginPlanV69,
    branch: PortfolioBranchV68,
    origin: RollingOriginV69,
    training_snapshot_hash: str,
    model_ir_hash: str,
    output_bundle_hash: str | None,
    verifier_evidence_hash: str | None,
) -> dict[str, object]:
    return {
        "protocol_hash": str(protocol.protocol_hash),
        "snapshot_hash": str(snapshot.snapshot_hash),
        "origin_plan_hash": str(plan.plan_hash),
        "branch_id": branch.branch_id,
        "manifest_id": branch.manifest_id,
        "manifest_hash": branch.manifest_hash,
        "capability_pack_id": branch.capability_pack_id,
        "origin_id": origin.origin_id,
        "training_view_hash": origin.training_view_hash,
        "training_snapshot_hash": training_snapshot_hash,
        "model_ir_hash": model_ir_hash,
        "output_bundle_hash": output_bundle_hash,
        "verifier_evidence_hash": verifier_evidence_hash,
    }


def _is_controlled_pack_failure_v69(error: Exception) -> bool:
    if isinstance(error, ArithmeticError):
        return True
    if not isinstance(error, (RuntimeError, ValueError)):
        return False
    message = str(error).lower()
    numeric_markers = (
        "array must not contain inf",
        "convergence",
        "invalid forecast",
        "not finite",
        "optimal parameters not found",
        "optimizer",
        "overflow",
        "residuals are not finite",
        "singular",
        "svd did not converge",
    )
    return any(marker in message for marker in numeric_markers)


def _controlled_failure_origin_receipt_v69(
    *,
    common: dict[str, object],
    phase: Literal["pack-execution", "input-bound-verification"],
    error: Exception,
) -> OriginExecutionReceiptV69:
    if not _is_controlled_pack_failure_v69(error):
        raise error
    failure_evidence_hash = sha256_value(
        {
            "schema_version": "6.9-controlled-pack-failure",
            "protocol_hash": common["protocol_hash"],
            "snapshot_hash": common["snapshot_hash"],
            "origin_plan_hash": common["origin_plan_hash"],
            "branch_id": common["branch_id"],
            "origin_id": common["origin_id"],
            "training_snapshot_hash": common["training_snapshot_hash"],
            "model_ir_hash": common["model_ir_hash"],
            "output_bundle_hash": common["output_bundle_hash"],
            "failure_phase": phase,
            "exception_type": type(error).__name__,
            "exception_message_hash": sha256_value(str(error)),
        }
    )
    return OriginExecutionReceiptV69.seal(
        **common,
        selected_model_id=None,
        prediction=None,
        parameter_count=None,
        execution_status="FAIL",
        failure_reason="controlled-pack-runtime-failure",
        failure_phase=phase,
        failure_evidence_hash=failure_evidence_hash,
    )


def _level_gate_failure_origin_receipt_v69(
    *,
    common: dict[str, object],
    level_statuses: dict[str, str],
    level_evidence_hashes: dict[str, str],
) -> OriginExecutionReceiptV69:
    return OriginExecutionReceiptV69.seal(
        **common,
        selected_model_id=None,
        prediction=None,
        parameter_count=None,
        execution_status="FAIL",
        failure_reason="pack-level-obligation-not-passed",
        failure_phase="input-bound-verification",
        failure_evidence_hash=sha256_value(
            {
                "schema_version": "6.9-pack-level-gate-failure",
                "protocol_hash": common["protocol_hash"],
                "snapshot_hash": common["snapshot_hash"],
                "origin_plan_hash": common["origin_plan_hash"],
                "branch_id": common["branch_id"],
                "origin_id": common["origin_id"],
                "level_statuses": level_statuses,
                "level_evidence_hashes": level_evidence_hashes,
            }
        ),
    )


def _run_ode_origin_v69(
    *,
    protocol: ModelingPortfolioProtocolV68,
    snapshot: PositiveSeriesSnapshotV69,
    plan: CommonRollingOriginPlanV69,
    branch: PortfolioBranchV68,
    origin: RollingOriginV69,
    thresholds: ODEThresholdsV52,
) -> OriginExecutionReceiptV69:
    prefix = slice(0, origin.training_count)
    training_snapshot = ODETimeSeriesSnapshotV52.seal(
        task_id=f"{snapshot.task_id}-{origin.origin_id}-ode",
        time_unit=snapshot.time_unit,
        state_unit=snapshot.state_unit,
        times=snapshot.times[prefix],
        observations=snapshot.observations[prefix],
        source_id=f"{snapshot.source_id}:{origin.origin_id}:ode",
        fixture_only=True,
    )
    model_ir = compile_scalar_autonomous_ode_ir_v68(thresholds)
    try:
        bundle = execute_scalar_autonomous_ode_ir_v68(
            model_ir=model_ir,
            snapshot=training_snapshot,
            thresholds=thresholds,
        )
    except Exception as error:
        return _controlled_failure_origin_receipt_v69(
            common=_origin_common_fields(
                protocol=protocol,
                snapshot=snapshot,
                plan=plan,
                branch=branch,
                origin=origin,
                training_snapshot_hash=str(training_snapshot.snapshot_hash),
                model_ir_hash=str(model_ir.ir_hash),
                output_bundle_hash=None,
                verifier_evidence_hash=None,
            ),
            phase="pack-execution",
            error=error,
        )
    try:
        verified = recompute_scalar_autonomous_ode_level_v68(
            bundle=bundle,
            level="L4",
            model_ir=model_ir,
            snapshot=training_snapshot,
            thresholds=thresholds,
        )
    except Exception as error:
        return _controlled_failure_origin_receipt_v69(
            common=_origin_common_fields(
                protocol=protocol,
                snapshot=snapshot,
                plan=plan,
                branch=branch,
                origin=origin,
                training_snapshot_hash=str(training_snapshot.snapshot_hash),
                model_ir_hash=str(model_ir.ir_hash),
                output_bundle_hash=str(bundle.bundle_hash),
                verifier_evidence_hash=None,
            ),
            phase="input-bound-verification",
            error=error,
        )
    common = _origin_common_fields(
        protocol=protocol,
        snapshot=snapshot,
        plan=plan,
        branch=branch,
        origin=origin,
        training_snapshot_hash=str(training_snapshot.snapshot_hash),
        model_ir_hash=str(model_ir.ir_hash),
        output_bundle_hash=str(bundle.bundle_hash),
        verifier_evidence_hash=str(verified.evidence_hash),
    )
    level_statuses = {
        item.level: item.status for item in bundle.legacy_bundle.levels
    }
    # V6.9 replaces each pack's native L3 split with the shared outer-origin
    # comparison.  Applicability (L1), mathematical checks (L2), and support/UQ
    # checks (L4) still gate eligibility; an L0 FAIL is never downgraded.
    if (
        level_statuses["L0"] == "FAIL"
        or any(
            level_statuses[level] != "PASS"
            for level in ("L1", "L2", "L4")
        )
    ):
        return _level_gate_failure_origin_receipt_v69(
            common=common,
            level_statuses=level_statuses,
            level_evidence_hashes={
                item.level: str(item.evidence_hash)
                for item in bundle.legacy_bundle.levels
            },
        )
    selected = next(
        item
        for item in bundle.legacy_bundle.candidates
        if item.candidate_id == bundle.legacy_bundle.selected_candidate_id
    )
    expected_parameters = _ODE_PARAMETER_COUNTS[selected.candidate_id]
    if (
        len(selected.fit.parameter_names) != expected_parameters
        or len(selected.fit.parameter_values) != expected_parameters
    ):
        raise ValueError("V6.9 independently recomputed ODE complexity differs")
    return OriginExecutionReceiptV69.seal(
        **common,
        selected_model_id=selected.candidate_id,
        prediction=selected.forecast_value,
        parameter_count=expected_parameters,
        execution_status="PASS",
    )


def _run_log_increment_origin_v69(
    *,
    protocol: ModelingPortfolioProtocolV68,
    snapshot: PositiveSeriesSnapshotV69,
    plan: CommonRollingOriginPlanV69,
    branch: PortfolioBranchV68,
    origin: RollingOriginV69,
    thresholds: PositiveLogIncrementThresholdsV68,
) -> OriginExecutionReceiptV69:
    prefix = slice(0, origin.training_count)
    training_snapshot = PositiveScalarSeriesSnapshotV68.seal(
        task_id=f"{snapshot.task_id}-{origin.origin_id}-log",
        time_unit=snapshot.time_unit,
        state_unit=snapshot.state_unit,
        times=snapshot.times[prefix],
        observations=snapshot.observations[prefix],
        source_id=f"{snapshot.source_id}:{origin.origin_id}:log",
        fixture_only=True,
    )
    model_ir = compile_positive_log_increment_ir_v68(thresholds)
    try:
        bundle = execute_positive_log_increment_ir_v68(
            model_ir=model_ir,
            snapshot=training_snapshot,
            thresholds=thresholds,
        )
    except Exception as error:
        return _controlled_failure_origin_receipt_v69(
            common=_origin_common_fields(
                protocol=protocol,
                snapshot=snapshot,
                plan=plan,
                branch=branch,
                origin=origin,
                training_snapshot_hash=str(training_snapshot.snapshot_hash),
                model_ir_hash=str(model_ir.ir_hash),
                output_bundle_hash=None,
                verifier_evidence_hash=None,
            ),
            phase="pack-execution",
            error=error,
        )
    try:
        verified = recompute_positive_log_increment_level_v68(
            bundle=bundle,
            level="L4",
            model_ir=model_ir,
            snapshot=training_snapshot,
            thresholds=thresholds,
        )
    except Exception as error:
        return _controlled_failure_origin_receipt_v69(
            common=_origin_common_fields(
                protocol=protocol,
                snapshot=snapshot,
                plan=plan,
                branch=branch,
                origin=origin,
                training_snapshot_hash=str(training_snapshot.snapshot_hash),
                model_ir_hash=str(model_ir.ir_hash),
                output_bundle_hash=str(bundle.bundle_hash),
                verifier_evidence_hash=None,
            ),
            phase="input-bound-verification",
            error=error,
        )
    common = _origin_common_fields(
        protocol=protocol,
        snapshot=snapshot,
        plan=plan,
        branch=branch,
        origin=origin,
        training_snapshot_hash=str(training_snapshot.snapshot_hash),
        model_ir_hash=str(model_ir.ir_hash),
        output_bundle_hash=str(bundle.bundle_hash),
        verifier_evidence_hash=str(verified.evidence_hash),
    )
    level_statuses = {
        item.level: item.status for item in bundle.levels
    }
    # The common V6.9 rolling-origin selector is the portfolio's L3 evidence.
    if (
        level_statuses["L0"] == "FAIL"
        or any(
            level_statuses[level] != "PASS"
            for level in ("L1", "L2", "L4")
        )
    ):
        return _level_gate_failure_origin_receipt_v69(
            common=common,
            level_statuses=level_statuses,
            level_evidence_hashes={
                item.level: str(item.evidence_hash)
                for item in bundle.levels
            },
        )
    if bundle.selection_status == "ABSTAIN":
        return OriginExecutionReceiptV69.seal(
            **common,
            selected_model_id=None,
            prediction=None,
            parameter_count=None,
            execution_status="FAIL",
            failure_reason="inner-pack-abstained",
            failure_phase="inner-selection",
            failure_evidence_hash=sha256_value(
                {
                    "bundle_hash": bundle.bundle_hash,
                    "selection_status": bundle.selection_status,
                    "diagnostic_model_id": bundle.diagnostic_model_id,
                }
            ),
        )
    selected = next(
        item
        for item in bundle.candidates
        if item.candidate_id == bundle.selected_model_id
    )
    expected_parameters = _LOG_INCREMENT_PARAMETER_COUNTS[selected.candidate_id]
    if selected.parameter_count != expected_parameters:
        raise ValueError(
            "V6.9 independently recomputed log-increment complexity differs"
        )
    return OriginExecutionReceiptV69.seal(
        **common,
        selected_model_id=selected.candidate_id,
        prediction=selected.forecast_value,
        parameter_count=expected_parameters,
        execution_status="PASS",
    )


def _aggregate_branch_receipt_v69(
    *,
    protocol: ModelingPortfolioProtocolV68,
    snapshot: PositiveSeriesSnapshotV69,
    plan: CommonRollingOriginPlanV69,
    branch: PortfolioBranchV68,
    origin_receipts: list[OriginExecutionReceiptV69],
) -> BranchExecutionReceiptV69:
    failed = [
        item.origin_id
        for item in origin_receipts
        if item.execution_status == "FAIL"
    ]
    if failed:
        return BranchExecutionReceiptV69.seal(
            protocol_hash=str(protocol.protocol_hash),
            snapshot_hash=str(snapshot.snapshot_hash),
            origin_plan_hash=str(plan.plan_hash),
            branch_id=branch.branch_id,
            manifest_id=branch.manifest_id,
            manifest_hash=branch.manifest_hash,
            capability_pack_id=branch.capability_pack_id,
            origin_receipts=origin_receipts,
            outer_origin_ids=[item.origin_id for item in origin_receipts],
            predictions=[],
            parameter_count=0,
            execution_status="FAIL",
            failed_origin_ids=failed,
            failure_reason="one-or-more-origins-failed",
        )
    return BranchExecutionReceiptV69.seal(
        protocol_hash=str(protocol.protocol_hash),
        snapshot_hash=str(snapshot.snapshot_hash),
        origin_plan_hash=str(plan.plan_hash),
        branch_id=branch.branch_id,
        manifest_id=branch.manifest_id,
        manifest_hash=branch.manifest_hash,
        capability_pack_id=branch.capability_pack_id,
        origin_receipts=origin_receipts,
        outer_origin_ids=[item.origin_id for item in origin_receipts],
        predictions=[
            float(item.prediction)
            for item in origin_receipts
            if item.prediction is not None
        ],
        parameter_count=max(
            int(item.parameter_count)
            for item in origin_receipts
            if item.parameter_count is not None
        ),
        execution_status="PASS",
        failed_origin_ids=[],
    )


def _branch_evaluation_v69(
    *,
    protocol: ModelingPortfolioProtocolV68,
    snapshot: PositiveSeriesSnapshotV69,
    plan: CommonRollingOriginPlanV69,
    receipt: BranchExecutionReceiptV69,
) -> BranchOuterEvaluationV68:
    observations = [
        snapshot.observations[item.target_index]
        for item in plan.origins
    ]
    return BranchOuterEvaluationV68.seal(
        protocol_hash=str(protocol.protocol_hash),
        branch_id=receipt.branch_id,
        manifest_id=receipt.manifest_id,
        manifest_hash=receipt.manifest_hash,
        execution_evidence_hash=str(receipt.receipt_hash),
        outer_origin_ids=receipt.outer_origin_ids,
        observations=observations,
        predictions=receipt.predictions,
        parameter_count=receipt.parameter_count,
        execution_status=receipt.execution_status,
        failure_reason=receipt.failure_reason,
    )


def _persistence_baseline_v69(
    *,
    snapshot: PositiveSeriesSnapshotV69,
    plan: CommonRollingOriginPlanV69,
) -> PersistenceBaselineEvaluationV69:
    observations = [
        snapshot.observations[item.target_index]
        for item in plan.origins
    ]
    predictions = [
        snapshot.observations[item.training_count - 1]
        for item in plan.origins
    ]
    return PersistenceBaselineEvaluationV69.seal(
        snapshot_hash=str(snapshot.snapshot_hash),
        origin_plan_hash=str(plan.plan_hash),
        outer_origin_ids=[item.origin_id for item in plan.origins],
        observations=observations,
        predictions=predictions,
        rmse=one_step_rmse_v68(observations, predictions),
    )


def _relative_improvement_v69(candidate: float, baseline: float) -> float:
    scale = 1e-12
    if baseline <= scale:
        return 0.0 if candidate <= scale else 1.0 - candidate / scale
    return 1.0 - candidate / baseline


def execute_development_portfolio_v69(
    *,
    protocol: ModelingPortfolioProtocolV68,
    snapshot: PositiveSeriesSnapshotV69,
    origin_plan: CommonRollingOriginPlanV69,
    ode_thresholds: ODEThresholdsV52,
    log_increment_thresholds: PositiveLogIncrementThresholdsV68,
    baseline_policy: PersistenceBaselinePolicyV69,
) -> DevelopmentPortfolioRunV69:
    """Execute, recompute, compare, and guard the exact local two-pack run."""

    branches = _assert_exact_development_protocol_v69(protocol)
    _assert_common_plan_v69(snapshot, origin_plan)
    ode_thresholds.assert_sealed()
    log_increment_thresholds.assert_sealed()
    baseline_policy.assert_sealed()
    if protocol.common_loss.loss_unit != snapshot.state_unit:
        raise ValueError("V6.9 snapshot unit differs from the common loss")

    branch_receipts: list[BranchExecutionReceiptV69] = []
    for pack_id, branch in sorted(
        branches.items(),
        key=lambda item: item[1].branch_id,
    ):
        origin_receipts: list[OriginExecutionReceiptV69] = []
        for origin in origin_plan.origins:
            if pack_id == _ODE_PACK_ID:
                receipt = _run_ode_origin_v69(
                    protocol=protocol,
                    snapshot=snapshot,
                    plan=origin_plan,
                    branch=branch,
                    origin=origin,
                    thresholds=ode_thresholds,
                )
            else:
                receipt = _run_log_increment_origin_v69(
                    protocol=protocol,
                    snapshot=snapshot,
                    plan=origin_plan,
                    branch=branch,
                    origin=origin,
                    thresholds=log_increment_thresholds,
                )
            origin_receipts.append(receipt)
        branch_receipts.append(
            _aggregate_branch_receipt_v69(
                protocol=protocol,
                snapshot=snapshot,
                plan=origin_plan,
                branch=branch,
                origin_receipts=origin_receipts,
            )
        )
    branch_receipts.sort(key=lambda item: item.branch_id)
    evaluations = [
        _branch_evaluation_v69(
            protocol=protocol,
            snapshot=snapshot,
            plan=origin_plan,
            receipt=receipt,
        )
        for receipt in branch_receipts
    ]
    inner = execute_outer_selection_v68(
        protocol=protocol,
        evaluations=evaluations,
    )
    baseline = _persistence_baseline_v69(
        snapshot=snapshot,
        plan=origin_plan,
    )
    if inner.decision == "ABSTAIN":
        final_decision: FinalDecisionV69 = "ABSTAIN"
        selected_branch_id = None
        selected_loss = None
        improvement = None
        reason = "inner-selection-abstained"
    else:
        inner_selected = str(inner.selected_branch_id)
        selected_loss = inner.common_loss_by_branch[inner_selected]
        improvement = _relative_improvement_v69(
            selected_loss,
            baseline.rmse,
        )
        if improvement > baseline_policy.minimum_relative_improvement:
            final_decision = "SELECT"
            selected_branch_id = inner_selected
            reason = "baseline-guarded-unique-winner"
        else:
            final_decision = "ABSTAIN"
            selected_branch_id = None
            reason = "persistence-baseline-not-beaten"
    return DevelopmentPortfolioRunV69.seal(
        protocol_hash=str(protocol.protocol_hash),
        snapshot_hash=str(snapshot.snapshot_hash),
        origin_plan_hash=str(origin_plan.plan_hash),
        baseline_policy=baseline_policy,
        branch_receipts=branch_receipts,
        evaluations=evaluations,
        persistence_baseline=baseline,
        inner_selection=inner,
        final_decision=final_decision,
        selected_branch_id=selected_branch_id,
        selected_common_loss=selected_loss,
        persistence_relative_improvement=improvement,
        reason_code=reason,
    )


def verify_development_portfolio_run_v69(
    *,
    protocol: ModelingPortfolioProtocolV68,
    snapshot: PositiveSeriesSnapshotV69,
    origin_plan: CommonRollingOriginPlanV69,
    ode_thresholds: ODEThresholdsV52,
    log_increment_thresholds: PositiveLogIncrementThresholdsV68,
    baseline_policy: PersistenceBaselinePolicyV69,
    run: DevelopmentPortfolioRunV69,
) -> bool:
    """Replay the full development run; never repair a supplied artifact."""

    try:
        run.assert_sealed()
        expected = execute_development_portfolio_v69(
            protocol=protocol,
            snapshot=snapshot,
            origin_plan=origin_plan,
            ode_thresholds=ode_thresholds,
            log_increment_thresholds=log_increment_thresholds,
            baseline_policy=baseline_policy,
        )
    except (
        ArithmeticError,
        KeyError,
        PermissionError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return False
    return expected == run


__all__ = [
    "BranchExecutionReceiptV69",
    "CommonRollingOriginPlanV69",
    "DevelopmentPortfolioRunV69",
    "MAXIMUM_ROLLING_ORIGINS_V69",
    "OriginExecutionReceiptV69",
    "PersistenceBaselineEvaluationV69",
    "PersistenceBaselinePolicyV69",
    "PositiveSeriesSnapshotV69",
    "RollingOriginV69",
    "build_common_rolling_origin_plan_v69",
    "compile_default_development_portfolio_protocol_v69",
    "execute_development_portfolio_v69",
    "verify_development_portfolio_run_v69",
]
