from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Annotated, Literal
from uuid import uuid4

import numpy as np
from pydantic import Field, model_validator
from scipy.stats import beta

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore

from .dynamics_integral import (
    DynamicsEstimatorPolicyV25,
    DynamicsSelectionReceiptV25,
    assert_single_component_estimator_ablation_v25,
    fit_dynamics_candidate_v25,
    select_dynamics_candidate_v25,
    simulate_dynamics_model_v25,
)
from .dynamics_ir import DynamicsDataSnapshotV24, support_f1, trajectory_nrmse
from .dynamics_worldpack import (
    DynamicsWorldPackSpecV24,
    Mechanism,
    PrivateDynamicsCaseV24,
    PrivateDynamicsWorldPackV24,
    generate_private_dynamics_worldpack,
)
from .schemas import Identifier, Sha256, _assert_timezone

if TYPE_CHECKING:
    from .dynamics_stability import DynamicsStabilityProtocolV25, DynamicsStabilityReportV25


EXPLORATORY_ESTIMATOR_SEEDS_V25 = (
    4001,
    4051,
    4099,
    4153,
    4201,
    4253,
    4303,
    4357,
)
EVOLVED_ESTIMATOR_SEEDS_V25 = (
    4409,
    4451,
    4507,
    4561,
    4603,
    4657,
    4703,
    4751,
)
CONFIRMATION_ESTIMATOR_SEEDS_V25 = (
    5003,
    5051,
    5101,
    5153,
    5209,
    5261,
    5309,
    5351,
    5407,
    5459,
    5501,
    5557,
    5609,
    5651,
    5701,
    5753,
    5801,
    5851,
    5903,
    5953,
)


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class DynamicsEstimatorExperimentSpecV25(StrictModel):
    schema_version: Literal["2.5"] = "2.5"
    experiment_id: Identifier
    phase: Literal["exploratory", "confirmation"]
    data_spec: DynamicsWorldPackSpecV24
    knowledge_bundle_hash: Sha256
    prior_failure_report_hash: Sha256
    prior_estimator_report_hash: Sha256 | None = None
    method_evidence_hash: Sha256
    point_policy_hash: Sha256
    integral_policy_hash: Sha256
    frozen_delta: Literal[
        "point_derivative_equations_vs_window_integral_equations_only"
    ] = "point_derivative_equations_vs_window_integral_equations_only"
    frozen_at: datetime
    spec_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "DynamicsEstimatorExperimentSpecV25":
        _assert_timezone(self.frozen_at, "frozen_at")
        self.data_spec.assert_sealed()
        if self.data_spec.phase != self.phase:
            raise ValueError("V2.5 estimator phase must match its frozen data spec")
        if self.data_spec.knowledge_bundle_hash != self.knowledge_bundle_hash:
            raise ValueError("V2.5 estimator data spec uses another knowledge bundle")
        if self.phase == "exploratory" and self.prior_estimator_report_hash is not None:
            raise ValueError("exploratory V2.5 estimator spec cannot bind a prior V2.5 report")
        if self.phase == "confirmation":
            if self.prior_estimator_report_hash is None:
                raise ValueError("confirmation V2.5 estimator spec needs exploratory evidence")
            if set(self.data_spec.seeds) & set(EXPLORATORY_ESTIMATOR_SEEDS_V25):
                raise ValueError("confirmation V2.5 seeds overlap exploratory V2.5 seeds")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match V2.5 estimator experiment")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "spec_hash")

    def assert_sealed(self) -> None:
        if not self.spec_hash or self.spec_hash != self.content_hash():
            raise ValueError("V2.5 estimator experiment spec is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DynamicsEstimatorExperimentSpecV25":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"spec_hash"}),
            spec_hash=draft.content_hash(),
        )


class DynamicsEstimatorSelectionBundleV25(StrictModel):
    schema_version: Literal["2.5"] = "2.5"
    bundle_id: Identifier
    experiment_spec_hash: Sha256
    private_pack_hash: Sha256
    estimator_arm: Literal["point_savgol", "window_integral_matching"]
    policy_hash: Sha256
    case_receipts: list[DynamicsSelectionReceiptV25] = Field(min_length=16)
    sentinel_receipts: list[DynamicsSelectionReceiptV25] = Field(min_length=2, max_length=2)
    bundle_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bundle(self) -> "DynamicsEstimatorSelectionBundleV25":
        receipts = [*self.case_receipts, *self.sentinel_receipts]
        if len({receipt.receipt_id for receipt in receipts}) != len(receipts):
            raise ValueError("V2.5 estimator receipt ids must be unique")
        if any(
            receipt.estimator_arm != self.estimator_arm
            or receipt.policy_hash != self.policy_hash
            for receipt in receipts
        ):
            raise ValueError("V2.5 selections are bound to another estimator policy")
        for receipt in receipts:
            receipt.assert_sealed()
        if self.bundle_hash and self.bundle_hash != self.content_hash():
            raise ValueError("bundle_hash does not match V2.5 selections")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "bundle_hash")

    def assert_sealed(self) -> None:
        if not self.bundle_hash or self.bundle_hash != self.content_hash():
            raise ValueError("V2.5 estimator selection bundle is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DynamicsEstimatorSelectionBundleV25":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"bundle_hash"}),
            bundle_hash=draft.content_hash(),
        )


class DynamicsEstimatorCaseResultV25(StrictModel):
    case_id: Identifier
    mechanism: Mechanism
    seed: Annotated[int, Field(ge=0)]
    point_candidate_id: Identifier
    integral_candidate_id: Identifier
    point_outer_nrmse: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    integral_outer_nrmse: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    point_counterfactual_nrmse: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    integral_counterfactual_nrmse: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    point_combined_nrmse: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    integral_combined_nrmse: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    point_structure_f1: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    integral_structure_f1: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    relative_improvement: Annotated[float, Field(allow_inf_nan=False)]
    structure_f1_improvement: Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)]
    outcome: Literal["integral_win", "tie", "integral_loss"]
    negative_transfer: bool
    point_total_fit_count: Literal[5] = 5
    integral_total_fit_count: Literal[5] = 5


class DynamicsEstimatorMechanismResultV25(StrictModel):
    mechanism: Mechanism
    case_count: Annotated[int, Field(ge=4)]
    mean_relative_improvement: Annotated[float, Field(allow_inf_nan=False)]
    confidence_lower: Annotated[float, Field(allow_inf_nan=False)]
    confidence_upper: Annotated[float, Field(allow_inf_nan=False)]
    noninferiority_margin: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    noninferior: bool

    @model_validator(mode="after")
    def validate_interval(self) -> "DynamicsEstimatorMechanismResultV25":
        if self.confidence_lower > self.confidence_upper:
            raise ValueError("V2.5 estimator mechanism interval is reversed")
        return self


EstimatorReportReason = Literal[
    "exploratory_not_eligible",
    "macro_improvement_interval_not_positive",
    "mechanism_noninferiority_failed",
    "negative_transfer_rate_bound_failed",
    "structure_recovery_not_improved",
    "identifiability_sentinel_false_promotion",
    "candidate_budget_mismatch",
    "single_component_ablation_broken",
]


class DynamicsEstimatorAblationReportV25(StrictModel):
    schema_version: Literal["2.5"] = "2.5"
    report_id: Identifier
    phase: Literal["exploratory", "confirmation"]
    experiment_spec_hash: Sha256
    knowledge_bundle_hash: Sha256
    prior_failure_report_hash: Sha256
    private_pack_hash: Sha256
    point_policy_hash: Sha256
    integral_policy_hash: Sha256
    point_bundle_hash: Sha256
    integral_bundle_hash: Sha256
    cases: list[DynamicsEstimatorCaseResultV25] = Field(min_length=16)
    mechanism_results: list[DynamicsEstimatorMechanismResultV25] = Field(
        min_length=4, max_length=4
    )
    macro_mean_relative_improvement: Annotated[float, Field(allow_inf_nan=False)]
    macro_confidence_lower: Annotated[float, Field(allow_inf_nan=False)]
    macro_confidence_upper: Annotated[float, Field(allow_inf_nan=False)]
    macro_mean_structure_f1_improvement: Annotated[float, Field(allow_inf_nan=False)]
    structure_confidence_lower: Annotated[float, Field(allow_inf_nan=False)]
    structure_confidence_upper: Annotated[float, Field(allow_inf_nan=False)]
    win_count: Annotated[int, Field(ge=0)]
    tie_count: Annotated[int, Field(ge=0)]
    loss_count: Annotated[int, Field(ge=0)]
    negative_transfer_count: Annotated[int, Field(ge=0)]
    negative_transfer_rate: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    negative_transfer_rate_upper: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ]
    sentinel_false_promotion_count: Annotated[int, Field(ge=0, le=4)]
    same_candidate_and_fit_budget: bool
    single_component_ablation: bool
    status: Literal[
        "exploratory_only",
        "candidate_rejected_estimator_v25",
        "promoted_for_synthetic_estimator_worldpack_v25",
    ]
    reason_codes: list[EstimatorReportReason]
    evaluated_at: datetime
    report_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "DynamicsEstimatorAblationReportV25":
        _assert_timezone(self.evaluated_at, "evaluated_at")
        if len({item.mechanism for item in self.mechanism_results}) != 4:
            raise ValueError("V2.5 estimator report needs four mechanisms")
        if self.win_count + self.tie_count + self.loss_count != len(self.cases):
            raise ValueError("V2.5 estimator outcome counts do not cover all cases")
        promoted = self.status == "promoted_for_synthetic_estimator_worldpack_v25"
        if promoted == bool(self.reason_codes):
            raise ValueError("V2.5 promoted report needs no reasons; other reports need reasons")
        if self.report_hash and self.report_hash != self.content_hash():
            raise ValueError("report_hash does not match V2.5 estimator report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "report_hash")

    def assert_sealed(self) -> None:
        if not self.report_hash or self.report_hash != self.content_hash():
            raise ValueError("V2.5 estimator report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DynamicsEstimatorAblationReportV25":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"report_hash"}),
            report_hash=draft.content_hash(),
        )


class DynamicsEstimatorQualificationV25(StrictModel):
    schema_version: Literal["2.5"] = "2.5"
    qualification_id: Identifier
    policy_hash: Sha256
    confirmation_report_hash: Sha256
    stability_report_hash: Sha256
    qualification_scope: Literal["synthetic_estimator_worldpack_v25"] = (
        "synthetic_estimator_worldpack_v25"
    )
    status: Literal["qualified"] = "qualified"
    limitations: list[Annotated[str, Field(min_length=12)]] = Field(min_length=5)
    qualified_at: datetime
    qualification_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_qualification(self) -> "DynamicsEstimatorQualificationV25":
        _assert_timezone(self.qualified_at, "qualified_at")
        if self.qualification_hash and self.qualification_hash != self.content_hash():
            raise ValueError("qualification_hash does not match V2.5 estimator qualification")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "qualification_hash")

    def assert_sealed(self) -> None:
        if not self.qualification_hash or self.qualification_hash != self.content_hash():
            raise ValueError("V2.5 estimator qualification is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DynamicsEstimatorQualificationV25":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"qualification_hash"}),
            qualification_hash=draft.content_hash(),
        )


class DynamicsEstimatorManifestV25(StrictModel):
    schema_version: Literal["2.5"] = "2.5"
    run_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")]
    artifact_refs: list[ArtifactRef] = Field(min_length=7, max_length=10)
    terminal_status: Literal[
        "exploratory_only",
        "candidate_rejected_estimator_v25",
        "promoted_for_synthetic_estimator_worldpack_v25",
    ]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "DynamicsEstimatorManifestV25":
        _assert_timezone(self.created_at, "created_at")
        if len({(ref.kind, ref.sha256) for ref in self.artifact_refs}) != len(
            self.artifact_refs
        ):
            raise ValueError("V2.5 estimator manifest refs must be unique")
        qualification_count = sum(
            ref.kind == "dynamics_estimator_qualification_v25"
            for ref in self.artifact_refs
        )
        promoted = self.terminal_status == "promoted_for_synthetic_estimator_worldpack_v25"
        if promoted != (qualification_count == 1):
            raise ValueError("V2.5 estimator qualification presence must match status")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match V2.5 estimator manifest")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("V2.5 estimator manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DynamicsEstimatorManifestV25":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class DynamicsEstimatorOutcomeV25:
    store: RunStore
    spec: DynamicsEstimatorExperimentSpecV25
    private_pack: PrivateDynamicsWorldPackV24
    point_policy: DynamicsEstimatorPolicyV25
    integral_policy: DynamicsEstimatorPolicyV25
    point_selections: DynamicsEstimatorSelectionBundleV25
    integral_selections: DynamicsEstimatorSelectionBundleV25
    report: DynamicsEstimatorAblationReportV25
    stability_protocol: DynamicsStabilityProtocolV25 | None
    stability_report: DynamicsStabilityReportV25 | None
    qualification: DynamicsEstimatorQualificationV25 | None
    manifest: DynamicsEstimatorManifestV25


def default_estimator_exploratory_spec_v25(
    *,
    knowledge_bundle_hash: str,
    prior_failure_report_hash: str,
    method_evidence_hash: str,
    point_policy_hash: str,
    integral_policy_hash: str,
    frozen_at: datetime | None = None,
) -> DynamicsEstimatorExperimentSpecV25:
    at = frozen_at or datetime.now(timezone.utc)
    data_spec = DynamicsWorldPackSpecV24.seal(
        pack_id="dynamics_estimator_exploratory_data_v25",
        phase="exploratory",
        knowledge_bundle_hash=knowledge_bundle_hash,
        mechanisms=[
            "exponential_decay",
            "logistic_growth",
            "damped_oscillator",
            "lotka_volterra",
        ],
        seeds=list(EXPLORATORY_ESTIMATOR_SEEDS_V25),
        training_points=121,
        inner_validation_points=40,
        outer_holdout_points=60,
        time_step=0.05,
        observation_noise_fraction=0.01,
        confidence_level=0.95,
        bootstrap_replicates=5_000,
        bootstrap_seed=270_722,
        negative_transfer_relative_margin=0.05,
        maximum_negative_transfer_rate_upper=0.05,
        mechanism_noninferiority_margin=0.05,
        minimum_structure_f1_improvement_lower=0.0,
        frozen_at=at,
    )
    return DynamicsEstimatorExperimentSpecV25.seal(
        experiment_id="dynamics_estimator_ablation_exploratory_v25",
        phase="exploratory",
        data_spec=data_spec,
        knowledge_bundle_hash=knowledge_bundle_hash,
        prior_failure_report_hash=prior_failure_report_hash,
        method_evidence_hash=method_evidence_hash,
        point_policy_hash=point_policy_hash,
        integral_policy_hash=integral_policy_hash,
        frozen_at=at,
    )


def default_estimator_confirmation_spec_v25(
    *,
    knowledge_bundle_hash: str,
    prior_failure_report_hash: str,
    prior_estimator_report_hash: str,
    method_evidence_hash: str,
    point_policy_hash: str,
    integral_policy_hash: str,
    frozen_at: datetime | None = None,
) -> DynamicsEstimatorExperimentSpecV25:
    at = frozen_at or datetime.now(timezone.utc)
    data_spec = DynamicsWorldPackSpecV24.seal(
        pack_id="dynamics_estimator_confirmation_data_v25",
        phase="confirmation",
        knowledge_bundle_hash=knowledge_bundle_hash,
        prior_exploratory_report_hash=prior_estimator_report_hash,
        prior_failure_report_hash=prior_failure_report_hash,
        mechanisms=[
            "exponential_decay",
            "logistic_growth",
            "damped_oscillator",
            "lotka_volterra",
        ],
        seeds=list(CONFIRMATION_ESTIMATOR_SEEDS_V25),
        training_points=121,
        inner_validation_points=40,
        outer_holdout_points=60,
        time_step=0.05,
        observation_noise_fraction=0.01,
        confidence_level=0.95,
        bootstrap_replicates=10_000,
        bootstrap_seed=271_722,
        negative_transfer_relative_margin=0.05,
        maximum_negative_transfer_rate_upper=0.05,
        mechanism_noninferiority_margin=0.05,
        minimum_structure_f1_improvement_lower=0.0,
        frozen_at=at,
    )
    return DynamicsEstimatorExperimentSpecV25.seal(
        experiment_id="dynamics_estimator_ablation_confirmation_v25",
        phase="confirmation",
        data_spec=data_spec,
        knowledge_bundle_hash=knowledge_bundle_hash,
        prior_failure_report_hash=prior_failure_report_hash,
        prior_estimator_report_hash=prior_estimator_report_hash,
        method_evidence_hash=method_evidence_hash,
        point_policy_hash=point_policy_hash,
        integral_policy_hash=integral_policy_hash,
        frozen_at=at,
    )


def failure_evolved_estimator_exploratory_spec_v25(
    *,
    knowledge_bundle_hash: str,
    failed_estimator_report_hash: str,
    method_evidence_hash: str,
    point_policy_hash: str,
    integral_policy_hash: str,
    frozen_at: datetime | None = None,
) -> DynamicsEstimatorExperimentSpecV25:
    """Fresh exploratory tranche after the first integral-window failure."""

    if set(EVOLVED_ESTIMATOR_SEEDS_V25) & set(EXPLORATORY_ESTIMATOR_SEEDS_V25):
        raise RuntimeError("failure-evolved V2.5 seeds overlap the first exploration")
    at = frozen_at or datetime.now(timezone.utc)
    data_spec = DynamicsWorldPackSpecV24.seal(
        pack_id="dynamics_estimator_evolved_exploratory_data_v25",
        phase="exploratory",
        knowledge_bundle_hash=knowledge_bundle_hash,
        mechanisms=[
            "exponential_decay",
            "logistic_growth",
            "damped_oscillator",
            "lotka_volterra",
        ],
        seeds=list(EVOLVED_ESTIMATOR_SEEDS_V25),
        training_points=121,
        inner_validation_points=40,
        outer_holdout_points=60,
        time_step=0.05,
        observation_noise_fraction=0.01,
        confidence_level=0.95,
        bootstrap_replicates=5_000,
        bootstrap_seed=272_722,
        negative_transfer_relative_margin=0.05,
        maximum_negative_transfer_rate_upper=0.05,
        mechanism_noninferiority_margin=0.05,
        minimum_structure_f1_improvement_lower=0.0,
        frozen_at=at,
    )
    return DynamicsEstimatorExperimentSpecV25.seal(
        experiment_id="dynamics_estimator_evolved_exploratory_v25",
        phase="exploratory",
        data_spec=data_spec,
        knowledge_bundle_hash=knowledge_bundle_hash,
        prior_failure_report_hash=failed_estimator_report_hash,
        method_evidence_hash=method_evidence_hash,
        point_policy_hash=point_policy_hash,
        integral_policy_hash=integral_policy_hash,
        frozen_at=at,
    )


def select_estimator_worldpack_v25(
    spec: DynamicsEstimatorExperimentSpecV25,
    private_pack: PrivateDynamicsWorldPackV24,
    policy: DynamicsEstimatorPolicyV25,
    *,
    selected_at: datetime | None = None,
) -> DynamicsEstimatorSelectionBundleV25:
    spec.assert_sealed()
    private_pack.assert_sealed()
    policy.assert_sealed()
    if private_pack.pack_spec_hash != spec.data_spec.spec_hash:
        raise ValueError("private pack is bound to another V2.5 estimator data spec")
    expected_hash = (
        spec.point_policy_hash
        if policy.estimator_arm == "point_savgol"
        else spec.integral_policy_hash
    )
    if policy.policy_hash != expected_hash:
        raise ValueError("V2.5 experiment is bound to another estimator policy")
    at = selected_at or datetime.now(timezone.utc)
    cases = [
        select_dynamics_candidate_v25(
            case.public_projection(spec.data_spec),
            policy,
            training_points=spec.data_spec.training_points,
            selected_at=at,
        )
        for case in private_pack.cases
    ]
    sentinels = [
        select_dynamics_candidate_v25(
            sentinel.public_projection(),
            policy,
            training_points=spec.data_spec.training_points,
            selected_at=at,
        )
        for sentinel in private_pack.sentinels
    ]
    assert spec.spec_hash is not None
    assert private_pack.pack_hash is not None
    assert policy.policy_hash is not None
    return DynamicsEstimatorSelectionBundleV25.seal(
        bundle_id=f"{spec.experiment_id}_{policy.estimator_arm}_selections",
        experiment_spec_hash=spec.spec_hash,
        private_pack_hash=private_pack.pack_hash,
        estimator_arm=policy.estimator_arm,
        policy_hash=policy.policy_hash,
        case_receipts=cases,
        sentinel_receipts=sentinels,
    )


def evaluate_estimator_worldpack_v25(
    spec: DynamicsEstimatorExperimentSpecV25,
    private_pack: PrivateDynamicsWorldPackV24,
    point_policy: DynamicsEstimatorPolicyV25,
    integral_policy: DynamicsEstimatorPolicyV25,
    point: DynamicsEstimatorSelectionBundleV25,
    integral: DynamicsEstimatorSelectionBundleV25,
    *,
    evaluated_at: datetime | None = None,
) -> DynamicsEstimatorAblationReportV25:
    for item in [spec, private_pack, point_policy, integral_policy, point, integral]:
        item.assert_sealed()
    single_component = True
    try:
        assert_single_component_estimator_ablation_v25(point_policy, integral_policy)
    except ValueError:
        single_component = False
    if point_policy.policy_hash != spec.point_policy_hash:
        raise ValueError("point policy is not frozen in the V2.5 experiment")
    if integral_policy.policy_hash != spec.integral_policy_hash:
        raise ValueError("integral policy is not frozen in the V2.5 experiment")
    if point.private_pack_hash != private_pack.pack_hash:
        raise ValueError("point selections use another private pack")
    if integral.private_pack_hash != private_pack.pack_hash:
        raise ValueError("integral selections use another private pack")
    point_by_data = {receipt.public_data_hash: receipt for receipt in point.case_receipts}
    integral_by_data = {
        receipt.public_data_hash: receipt for receipt in integral.case_receipts
    }
    results: list[DynamicsEstimatorCaseResultV25] = []
    for case in private_pack.cases:
        public = case.public_projection(spec.data_spec)
        assert public.snapshot_hash is not None
        point_receipt = point_by_data[public.snapshot_hash]
        integral_receipt = integral_by_data[public.snapshot_hash]
        replay_point = select_dynamics_candidate_v25(
            public,
            point_policy,
            training_points=spec.data_spec.training_points,
            selected_at=point_receipt.selected_at,
        )
        replay_integral = select_dynamics_candidate_v25(
            public,
            integral_policy,
            training_points=spec.data_spec.training_points,
            selected_at=integral_receipt.selected_at,
        )
        if replay_point.receipt_hash != point_receipt.receipt_hash:
            raise ValueError("point V2.5 selection does not replay")
        if replay_integral.receipt_hash != integral_receipt.receipt_hash:
            raise ValueError("integral V2.5 selection does not replay")
        if point_receipt.status != "selected" or integral_receipt.status != "selected":
            raise ValueError("primary V2.5 case abstained; no paired effect is valid")
        point_metrics = _outer_metrics_v25(
            case, spec.data_spec, public, point_policy, point_receipt
        )
        integral_metrics = _outer_metrics_v25(
            case, spec.data_spec, public, integral_policy, integral_receipt
        )
        point_combined = (point_metrics[0] + point_metrics[1]) / 2.0
        integral_combined = (integral_metrics[0] + integral_metrics[1]) / 2.0
        relative = (point_combined - integral_combined) / max(point_combined, 1e-12)
        tolerance = max(1e-12, point_combined * 1e-9)
        difference = point_combined - integral_combined
        outcome = (
            "integral_win"
            if difference > tolerance
            else "integral_loss"
            if difference < -tolerance
            else "tie"
        )
        negative = integral_combined > point_combined * (
            1.0 + spec.data_spec.negative_transfer_relative_margin
        ) + 1e-12
        assert point_receipt.selected_candidate_id is not None
        assert integral_receipt.selected_candidate_id is not None
        results.append(
            DynamicsEstimatorCaseResultV25(
                case_id=case.case_id,
                mechanism=case.mechanism,
                seed=case.seed,
                point_candidate_id=point_receipt.selected_candidate_id,
                integral_candidate_id=integral_receipt.selected_candidate_id,
                point_outer_nrmse=point_metrics[0],
                integral_outer_nrmse=integral_metrics[0],
                point_counterfactual_nrmse=point_metrics[1],
                integral_counterfactual_nrmse=integral_metrics[1],
                point_combined_nrmse=point_combined,
                integral_combined_nrmse=integral_combined,
                point_structure_f1=point_metrics[2],
                integral_structure_f1=integral_metrics[2],
                relative_improvement=relative,
                structure_f1_improvement=integral_metrics[2] - point_metrics[2],
                outcome=outcome,
                negative_transfer=negative,
            )
        )
    grouped_relative: dict[str, list[float]] = {
        mechanism: [] for mechanism in spec.data_spec.mechanisms
    }
    grouped_structure: dict[str, list[float]] = {
        mechanism: [] for mechanism in spec.data_spec.mechanisms
    }
    for result in results:
        grouped_relative[result.mechanism].append(result.relative_improvement)
        grouped_structure[result.mechanism].append(result.structure_f1_improvement)
    relative_draws = _stratified_bootstrap(grouped_relative, spec.data_spec, 0)
    structure_draws = _stratified_bootstrap(grouped_structure, spec.data_spec, 1)
    alpha = (1.0 - spec.data_spec.confidence_level) / 2.0
    macro_lower, macro_upper = np.quantile(
        relative_draws, [alpha, 1.0 - alpha], method="linear"
    )
    structure_lower, structure_upper = np.quantile(
        structure_draws, [alpha, 1.0 - alpha], method="linear"
    )
    mechanisms: list[DynamicsEstimatorMechanismResultV25] = []
    for offset, mechanism in enumerate(spec.data_spec.mechanisms, start=11):
        values = grouped_relative[mechanism]
        lower, upper = _bootstrap_interval(values, spec.data_spec, offset)
        mechanisms.append(
            DynamicsEstimatorMechanismResultV25(
                mechanism=mechanism,
                case_count=len(values),
                mean_relative_improvement=float(np.mean(values)),
                confidence_lower=lower,
                confidence_upper=upper,
                noninferiority_margin=spec.data_spec.mechanism_noninferiority_margin,
                noninferior=lower >= -spec.data_spec.mechanism_noninferiority_margin,
            )
        )
    negative_count = sum(result.negative_transfer for result in results)
    negative_upper = _clopper_pearson_upper(
        negative_count, len(results), spec.data_spec.confidence_level
    )
    sentinel_false = _sentinel_false_promotions_v25(
        spec, private_pack, point_policy, integral_policy, point, integral
    )
    same_budget = (
        point_policy.candidate_budget
        == integral_policy.candidate_budget
        == spec.data_spec.candidate_budget_per_arm
        and all(
            result.point_total_fit_count == result.integral_total_fit_count
            for result in results
        )
    )
    reasons: list[EstimatorReportReason] = []
    if spec.phase == "exploratory":
        reasons.append("exploratory_not_eligible")
        status = "exploratory_only"
    else:
        if float(macro_lower) <= 0:
            reasons.append("macro_improvement_interval_not_positive")
        if not all(result.noninferior for result in mechanisms):
            reasons.append("mechanism_noninferiority_failed")
        if negative_upper > spec.data_spec.maximum_negative_transfer_rate_upper:
            reasons.append("negative_transfer_rate_bound_failed")
        if float(structure_lower) < spec.data_spec.minimum_structure_f1_improvement_lower:
            reasons.append("structure_recovery_not_improved")
        if sentinel_false:
            reasons.append("identifiability_sentinel_false_promotion")
        if not same_budget:
            reasons.append("candidate_budget_mismatch")
        if not single_component:
            reasons.append("single_component_ablation_broken")
        status = (
            "candidate_rejected_estimator_v25"
            if reasons
            else "promoted_for_synthetic_estimator_worldpack_v25"
        )
    assert spec.spec_hash is not None
    assert private_pack.pack_hash is not None
    assert point_policy.policy_hash is not None
    assert integral_policy.policy_hash is not None
    assert point.bundle_hash is not None
    assert integral.bundle_hash is not None
    return DynamicsEstimatorAblationReportV25.seal(
        report_id=f"{spec.experiment_id}_report",
        phase=spec.phase,
        experiment_spec_hash=spec.spec_hash,
        knowledge_bundle_hash=spec.knowledge_bundle_hash,
        prior_failure_report_hash=spec.prior_failure_report_hash,
        private_pack_hash=private_pack.pack_hash,
        point_policy_hash=point_policy.policy_hash,
        integral_policy_hash=integral_policy.policy_hash,
        point_bundle_hash=point.bundle_hash,
        integral_bundle_hash=integral.bundle_hash,
        cases=results,
        mechanism_results=mechanisms,
        macro_mean_relative_improvement=float(
            np.mean(
                [
                    np.mean(grouped_relative[mechanism])
                    for mechanism in spec.data_spec.mechanisms
                ]
            )
        ),
        macro_confidence_lower=float(macro_lower),
        macro_confidence_upper=float(macro_upper),
        macro_mean_structure_f1_improvement=float(
            np.mean(
                [
                    np.mean(grouped_structure[mechanism])
                    for mechanism in spec.data_spec.mechanisms
                ]
            )
        ),
        structure_confidence_lower=float(structure_lower),
        structure_confidence_upper=float(structure_upper),
        win_count=sum(result.outcome == "integral_win" for result in results),
        tie_count=sum(result.outcome == "tie" for result in results),
        loss_count=sum(result.outcome == "integral_loss" for result in results),
        negative_transfer_count=negative_count,
        negative_transfer_rate=negative_count / len(results),
        negative_transfer_rate_upper=negative_upper,
        sentinel_false_promotion_count=sentinel_false,
        same_candidate_and_fit_budget=same_budget,
        single_component_ablation=single_component,
        status=status,
        reason_codes=reasons,
        evaluated_at=evaluated_at or datetime.now(timezone.utc),
    )


def qualify_estimator_policy_v25(
    policy: DynamicsEstimatorPolicyV25,
    report: DynamicsEstimatorAblationReportV25,
    stability_report: DynamicsStabilityReportV25 | None = None,
) -> DynamicsEstimatorQualificationV25:
    policy.assert_sealed()
    report.assert_sealed()
    if report.status != "promoted_for_synthetic_estimator_worldpack_v25":
        raise ValueError("a rejected V2.5 estimator policy cannot be qualified")
    if policy.estimator_arm != "window_integral_matching":
        raise ValueError("only the evaluated integral estimator may be qualified")
    if report.integral_policy_hash != policy.policy_hash:
        raise ValueError("V2.5 estimator report is bound to another policy")
    if stability_report is None:
        raise ValueError("V2.5 estimator qualification requires a stability report")
    stability_report.assert_sealed()
    if stability_report.estimator_report_hash != report.report_hash:
        raise ValueError("V2.5 stability evidence is bound to another estimator report")
    if stability_report.integral_policy_hash != policy.policy_hash:
        raise ValueError("V2.5 stability evidence is bound to another policy")
    if stability_report.status != "stability_gate_passed":
        raise ValueError("a stability-failed V2.5 estimator cannot be qualified")
    assert policy.policy_hash is not None
    assert report.report_hash is not None
    assert stability_report.report_hash is not None
    return DynamicsEstimatorQualificationV25.seal(
        qualification_id=f"{policy.policy_id}_qualification",
        policy_hash=policy.policy_hash,
        confirmation_report_hash=report.report_hash,
        stability_report_hash=stability_report.report_hash,
        limitations=[
            "valid only for four frozen synthetic polynomial ODE mechanisms with full state observation",
            "window integral matching is a constant-test-function special case and not full Weak SINDy",
            "overlapping windows are dependent and parameter intervals are not yet calibrated confidence intervals",
            "does not cover irregular sampling, controls, stochastic dynamics, delays, PDEs, or hybrid events",
            "does not authorize real-data parameter claims, interventions, or external decisions",
        ],
        qualified_at=report.evaluated_at,
    )


def run_estimator_worldpack_v25(
    output_root: str | Path,
    *,
    spec: DynamicsEstimatorExperimentSpecV25,
    point_policy: DynamicsEstimatorPolicyV25,
    integral_policy: DynamicsEstimatorPolicyV25,
    stability_protocol: DynamicsStabilityProtocolV25 | None = None,
    evaluated_at: datetime | None = None,
    run_id: str | None = None,
) -> DynamicsEstimatorOutcomeV25:
    spec.assert_sealed()
    point_policy.assert_sealed()
    integral_policy.assert_sealed()
    assert_single_component_estimator_ablation_v25(point_policy, integral_policy)
    if point_policy.policy_hash != spec.point_policy_hash:
        raise ValueError("V2.5 run point policy is not frozen in the spec")
    if integral_policy.policy_hash != spec.integral_policy_hash:
        raise ValueError("V2.5 run integral policy is not frozen in the spec")
    if spec.phase == "confirmation" and stability_protocol is None:
        raise ValueError("V2.5 confirmation requires a frozen stability protocol")
    if stability_protocol is not None:
        stability_protocol.assert_sealed()
        if stability_protocol.experiment_spec_hash != spec.spec_hash:
            raise ValueError("V2.5 stability protocol is bound to another experiment")
    at = evaluated_at or datetime.now(timezone.utc)
    store = RunStore(
        output_root, run_id=run_id or f"dynamics-estimator-{uuid4().hex[:10]}"
    )
    refs = [
        store.put_artifact("dynamics_estimator_spec_v25", spec),
        store.put_artifact("dynamics_point_policy_v25", point_policy),
        store.put_artifact("dynamics_integral_policy_v25", integral_policy),
    ]
    if stability_protocol is not None:
        refs.append(store.put_artifact("dynamics_stability_protocol_v25", stability_protocol))
    store.emit(
        "dynamics_estimator_protocol_frozen_before_private_pack",
        {
            "spec_hash": spec.spec_hash,
            "point_policy_hash": point_policy.policy_hash,
            "integral_policy_hash": integral_policy.policy_hash,
            "stability_protocol_hash": (
                None if stability_protocol is None else stability_protocol.protocol_hash
            ),
            "frozen_delta": spec.frozen_delta,
        },
    )
    private_pack = generate_private_dynamics_worldpack(
        spec.data_spec, generated_at=at
    )
    point = select_estimator_worldpack_v25(
        spec, private_pack, point_policy, selected_at=at
    )
    integral = select_estimator_worldpack_v25(
        spec, private_pack, integral_policy, selected_at=at
    )
    report = evaluate_estimator_worldpack_v25(
        spec,
        private_pack,
        point_policy,
        integral_policy,
        point,
        integral,
        evaluated_at=at,
    )
    refs.extend(
        [
            store.put_artifact("private_dynamics_worldpack_v24", private_pack),
            store.put_artifact("dynamics_point_selections_v25", point),
            store.put_artifact("dynamics_integral_selections_v25", integral),
            store.put_artifact("dynamics_estimator_report_v25", report),
        ]
    )
    stability_report = None
    if stability_protocol is not None:
        from .dynamics_stability import evaluate_dynamics_stability_v25

        stability_report = evaluate_dynamics_stability_v25(
            stability_protocol,
            spec,
            report,
            private_pack,
            point_policy,
            integral_policy,
            point,
            integral,
            evaluated_at=at,
        )
        refs.append(store.put_artifact("dynamics_stability_report_v25", stability_report))
    qualification = None
    terminal_status = report.status
    if report.status == "promoted_for_synthetic_estimator_worldpack_v25":
        if stability_report is not None and stability_report.status == "stability_gate_passed":
            qualification = qualify_estimator_policy_v25(
                integral_policy, report, stability_report
            )
        else:
            terminal_status = "candidate_rejected_estimator_v25"
    if qualification is not None:
        refs.append(
            store.put_artifact("dynamics_estimator_qualification_v25", qualification)
        )
    manifest = DynamicsEstimatorManifestV25.seal(
        run_id=store.run_id,
        artifact_refs=refs,
        terminal_status=terminal_status,
        created_at=at,
    )
    manifest_ref = store.put_artifact("dynamics_estimator_manifest_v25", manifest)
    store.emit(
        "dynamics_estimator_worldpack_adjudicated",
        {"manifest_ref": manifest_ref.model_dump(mode="json")},
    )
    if not verify_estimator_worldpack_run_v25(store.run_directory):
        raise RuntimeError("V2.5 estimator run failed independent verification")
    return DynamicsEstimatorOutcomeV25(
        store,
        spec,
        private_pack,
        point_policy,
        integral_policy,
        point,
        integral,
        report,
        stability_protocol,
        stability_report,
        qualification,
        manifest,
    )


def verify_estimator_worldpack_run_v25(run_directory: str | Path) -> bool:
    try:
        store = RunStore.open_existing(run_directory)
        events = [
            json.loads(line)
            for line in store.event_path.read_text(encoding="utf-8").splitlines()
        ]
        committed = [
            ArtifactRef.model_validate(event["payload"])
            for event in events
            if event["event_type"] == "artifact_committed"
        ]
        for ref in committed:
            store.load_artifact(ref)
        manifest_refs = [
            ref for ref in committed if ref.kind == "dynamics_estimator_manifest_v25"
        ]
        if len(manifest_refs) != 1:
            return False
        manifest = DynamicsEstimatorManifestV25.model_validate(
            store.load_artifact(manifest_refs[0])
        )
        manifest.assert_sealed()
        if manifest.run_id != store.run_id:
            return False

        def load_one(kind: str, model):
            refs = [ref for ref in manifest.artifact_refs if ref.kind == kind]
            if len(refs) != 1:
                raise RuntimeError(f"manifest needs exactly one {kind}")
            return model.model_validate(store.load_artifact(refs[0]))

        spec = load_one("dynamics_estimator_spec_v25", DynamicsEstimatorExperimentSpecV25)
        point_policy = load_one("dynamics_point_policy_v25", DynamicsEstimatorPolicyV25)
        integral_policy = load_one(
            "dynamics_integral_policy_v25", DynamicsEstimatorPolicyV25
        )
        private_pack = load_one("private_dynamics_worldpack_v24", PrivateDynamicsWorldPackV24)
        point = load_one(
            "dynamics_point_selections_v25", DynamicsEstimatorSelectionBundleV25
        )
        integral = load_one(
            "dynamics_integral_selections_v25", DynamicsEstimatorSelectionBundleV25
        )
        report = load_one(
            "dynamics_estimator_report_v25", DynamicsEstimatorAblationReportV25
        )
        for item in [
            spec,
            point_policy,
            integral_policy,
            private_pack,
            point,
            integral,
            report,
        ]:
            item.assert_sealed()
        regenerated = generate_private_dynamics_worldpack(
            spec.data_spec, generated_at=private_pack.generated_at
        )
        if regenerated.pack_hash != private_pack.pack_hash:
            return False
        replay_point = select_estimator_worldpack_v25(
            spec,
            private_pack,
            point_policy,
            selected_at=point.case_receipts[0].selected_at,
        )
        replay_integral = select_estimator_worldpack_v25(
            spec,
            private_pack,
            integral_policy,
            selected_at=integral.case_receipts[0].selected_at,
        )
        if replay_point.bundle_hash != point.bundle_hash:
            return False
        if replay_integral.bundle_hash != integral.bundle_hash:
            return False
        recomputed = evaluate_estimator_worldpack_v25(
            spec,
            private_pack,
            point_policy,
            integral_policy,
            point,
            integral,
            evaluated_at=report.evaluated_at,
        )
        if recomputed.report_hash != report.report_hash:
            return False
        stability_protocol_refs = [
            ref for ref in manifest.artifact_refs if ref.kind == "dynamics_stability_protocol_v25"
        ]
        stability_report_refs = [
            ref for ref in manifest.artifact_refs if ref.kind == "dynamics_stability_report_v25"
        ]
        stability_report = None
        if stability_protocol_refs or stability_report_refs:
            if len(stability_protocol_refs) != 1 or len(stability_report_refs) != 1:
                return False
            from .dynamics_stability import (
                DynamicsStabilityProtocolV25,
                DynamicsStabilityReportV25,
                evaluate_dynamics_stability_v25,
            )

            stability_protocol = DynamicsStabilityProtocolV25.model_validate(
                store.load_artifact(stability_protocol_refs[0])
            )
            stability_report = DynamicsStabilityReportV25.model_validate(
                store.load_artifact(stability_report_refs[0])
            )
            stability_protocol.assert_sealed()
            stability_report.assert_sealed()
            expected_stability = evaluate_dynamics_stability_v25(
                stability_protocol,
                spec,
                report,
                private_pack,
                point_policy,
                integral_policy,
                point,
                integral,
                evaluated_at=stability_report.evaluated_at,
            )
            if expected_stability.report_hash != stability_report.report_hash:
                return False
        expected_terminal = report.status
        if (
            report.status == "promoted_for_synthetic_estimator_worldpack_v25"
            and (stability_report is None or stability_report.status != "stability_gate_passed")
        ):
            expected_terminal = "candidate_rejected_estimator_v25"
        if manifest.terminal_status != expected_terminal:
            return False
        qualifications = [
            ref
            for ref in manifest.artifact_refs
            if ref.kind == "dynamics_estimator_qualification_v25"
        ]
        if report.status == "promoted_for_synthetic_estimator_worldpack_v25":
            if len(qualifications) != 1:
                return False
            stored = DynamicsEstimatorQualificationV25.model_validate(
                store.load_artifact(qualifications[0])
            )
            stored.assert_sealed()
            expected = qualify_estimator_policy_v25(
                integral_policy, report, stability_report
            )
            if expected.qualification_hash != stored.qualification_hash:
                return False
        elif qualifications:
            return False
        freeze = [
            event
            for event in events
            if event["event_type"]
            == "dynamics_estimator_protocol_frozen_before_private_pack"
        ]
        return len(freeze) == 1
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        FloatingPointError,
    ):
        return False


def _outer_metrics_v25(
    case: PrivateDynamicsCaseV24,
    data_spec: DynamicsWorldPackSpecV24,
    public: DynamicsDataSnapshotV24,
    policy: DynamicsEstimatorPolicyV25,
    receipt: DynamicsSelectionReceiptV25,
) -> tuple[float, float, float]:
    definition = next(
        candidate
        for candidate in policy.candidates
        if candidate.candidate_id == receipt.selected_candidate_id
    )
    fit = fit_dynamics_candidate_v25(
        public, definition, policy, fitted_at=receipt.selected_at
    )
    if fit.status != "fit_succeeded" or fit.model is None:
        raise ValueError("selected V2.5 estimator candidate did not refit")
    public_count = data_spec.training_points + data_spec.inner_validation_points
    future_times = case.times[public_count - 1 :]
    predicted = simulate_dynamics_model_v25(
        fit.model, case.public_observed_values[-1], future_times
    )[1:]
    outer_nrmse = trajectory_nrmse(case.clean_values[public_count:], predicted)
    counter_times = [
        index * data_spec.time_step
        for index in range(data_spec.outer_holdout_points + 1)
    ]
    counter_predicted = simulate_dynamics_model_v25(
        fit.model, case.counterfactual_initial_state, counter_times
    )[1:]
    counter_nrmse = trajectory_nrmse(
        case.counterfactual_values[1:], counter_predicted
    )
    structure = support_f1(fit.model, case.truth_coefficients)  # compatible sealed IR
    return outer_nrmse, counter_nrmse, structure


def _sentinel_false_promotions_v25(
    spec: DynamicsEstimatorExperimentSpecV25,
    private_pack: PrivateDynamicsWorldPackV24,
    point_policy: DynamicsEstimatorPolicyV25,
    integral_policy: DynamicsEstimatorPolicyV25,
    point: DynamicsEstimatorSelectionBundleV25,
    integral: DynamicsEstimatorSelectionBundleV25,
) -> int:
    false = 0
    for sentinel, point_receipt, integral_receipt in zip(
        private_pack.sentinels,
        point.sentinel_receipts,
        integral.sentinel_receipts,
        strict=True,
    ):
        public = sentinel.public_projection()
        for policy, receipt in (
            (point_policy, point_receipt),
            (integral_policy, integral_receipt),
        ):
            replay = select_dynamics_candidate_v25(
                public,
                policy,
                training_points=spec.data_spec.training_points,
                selected_at=receipt.selected_at,
            )
            if replay.receipt_hash != receipt.receipt_hash:
                raise ValueError("V2.5 sentinel selection does not replay")
            false += receipt.status == "selected"
    return false


def _stratified_bootstrap(
    grouped: dict[str, list[float]],
    spec: DynamicsWorldPackSpecV24,
    seed_offset: int,
) -> np.ndarray:
    random = Random(spec.bootstrap_seed + seed_offset)
    draws = np.empty(spec.bootstrap_replicates, dtype=float)
    for draw in range(spec.bootstrap_replicates):
        mechanism_means = []
        for mechanism in spec.mechanisms:
            values = grouped[mechanism]
            mechanism_means.append(
                sum(values[random.randrange(len(values))] for _ in values) / len(values)
            )
        draws[draw] = sum(mechanism_means) / len(mechanism_means)
    return draws


def _bootstrap_interval(
    values: list[float],
    spec: DynamicsWorldPackSpecV24,
    seed_offset: int,
) -> tuple[float, float]:
    random = Random(spec.bootstrap_seed + seed_offset)
    draws = [
        sum(values[random.randrange(len(values))] for _ in values) / len(values)
        for _ in range(spec.bootstrap_replicates)
    ]
    alpha = (1.0 - spec.confidence_level) / 2.0
    lower, upper = np.quantile(draws, [alpha, 1.0 - alpha], method="linear")
    return float(lower), float(upper)


def _clopper_pearson_upper(successes: int, trials: int, confidence: float) -> float:
    if successes == trials:
        return 1.0
    return float(beta.ppf(confidence, successes + 1, trials - successes))
