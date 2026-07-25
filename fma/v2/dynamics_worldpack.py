from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Annotated, Literal
from uuid import uuid4

import numpy as np
from pydantic import Field, model_validator
from scipy.integrate import solve_ivp
from scipy.stats import beta

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore

from .dynamics_ir import (
    Arm,
    DynamicsArmPolicyV24,
    DynamicsDataSnapshotV24,
    DynamicsSelectionReceiptV24,
    default_dynamics_arm_policy,
    fit_dynamics_candidate,
    select_dynamics_candidate,
    simulate_dynamics_model,
    support_f1,
    trajectory_nrmse,
)
from .schemas import Identifier, Sha256, _assert_timezone


Mechanism = Literal[
    "exponential_decay",
    "logistic_growth",
    "damped_oscillator",
    "lotka_volterra",
]
SentinelKind = Literal["partial_observation", "rank_deficient_excitation"]


EXPLORATORY_DYNAMICS_SEEDS = (17, 53, 97, 149, 211, 277)
CONFIRMATION_DYNAMICS_SEEDS = (
    1009,
    1061,
    1117,
    1171,
    1223,
    1277,
    1327,
    1381,
    1433,
    1487,
    1543,
    1597,
    1657,
    1709,
    1759,
    1811,
    1871,
    1931,
    1993,
    2053,
)
GUARDED_CONFIRMATION_DYNAMICS_SEEDS = (
    2111,
    2161,
    2213,
    2269,
    2311,
    2371,
    2423,
    2477,
    2531,
    2591,
    2647,
    2707,
    2767,
    2821,
    2879,
    2939,
    2999,
    3061,
    3121,
    3181,
)


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class DynamicsWorldPackSpecV24(StrictModel):
    schema_version: Literal["2.4"] = "2.4"
    pack_id: Identifier
    phase: Literal["exploratory", "confirmation"]
    knowledge_bundle_hash: Sha256
    prior_exploratory_report_hash: Sha256 | None = None
    prior_failure_report_hash: Sha256 | None = None
    mechanisms: list[Mechanism] = Field(min_length=4, max_length=4)
    seeds: list[Annotated[int, Field(ge=0, le=2_147_483_647)]] = Field(
        min_length=4, max_length=64
    )
    training_points: Annotated[int, Field(ge=61, le=1_000)]
    inner_validation_points: Annotated[int, Field(ge=20, le=500)]
    outer_holdout_points: Annotated[int, Field(ge=20, le=500)]
    time_step: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
    observation_noise_fraction: Annotated[
        float, Field(gt=0, le=0.2, allow_inf_nan=False)
    ]
    candidate_budget_per_arm: Literal[4] = 4
    confidence_level: Annotated[float, Field(gt=0.8, lt=1, allow_inf_nan=False)]
    bootstrap_replicates: Annotated[int, Field(ge=5_000, le=100_000)]
    bootstrap_seed: Annotated[int, Field(ge=0, le=2_147_483_647)]
    negative_transfer_relative_margin: Annotated[
        float, Field(gt=0, lt=1, allow_inf_nan=False)
    ]
    maximum_negative_transfer_rate_upper: Annotated[
        float, Field(gt=0, lt=0.5, allow_inf_nan=False)
    ]
    mechanism_noninferiority_margin: Annotated[
        float, Field(gt=0, lt=0.5, allow_inf_nan=False)
    ]
    minimum_structure_f1_improvement_lower: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ]
    frozen_at: datetime
    spec_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "DynamicsWorldPackSpecV24":
        _assert_timezone(self.frozen_at, "frozen_at")
        if set(self.mechanisms) != {
            "exponential_decay",
            "logistic_growth",
            "damped_oscillator",
            "lotka_volterra",
        }:
            raise ValueError("Dynamics WorldPack needs the four frozen mechanisms")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("Dynamics WorldPack seeds must be unique")
        if self.phase == "exploratory" and (
            self.prior_exploratory_report_hash is not None
            or self.prior_failure_report_hash is not None
        ):
            raise ValueError("exploratory Dynamics WorldPack cannot bind a prior report")
        if self.phase == "confirmation":
            if self.prior_exploratory_report_hash is None:
                raise ValueError("confirmation Dynamics WorldPack must bind exploratory evidence")
            if len(self.seeds) < 20:
                raise ValueError("confirmation Dynamics WorldPack needs at least 20 seeds")
            if set(self.seeds) & set(EXPLORATORY_DYNAMICS_SEEDS):
                raise ValueError("confirmation Dynamics seeds overlap exploratory seeds")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match Dynamics WorldPack spec")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "spec_hash")

    def assert_sealed(self) -> None:
        if not self.spec_hash or self.spec_hash != self.content_hash():
            raise ValueError("Dynamics WorldPack spec is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DynamicsWorldPackSpecV24":
        data.setdefault("frozen_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"spec_hash"}),
            spec_hash=draft.content_hash(),
        )


class PrivateDynamicsCaseV24(StrictModel):
    schema_version: Literal["2.4"] = "2.4"
    case_id: Identifier
    mechanism: Mechanism
    seed: Annotated[int, Field(ge=0)]
    state_names: list[Identifier] = Field(min_length=1, max_length=2)
    times: list[Annotated[float, Field(allow_inf_nan=False)]] = Field(min_length=101)
    public_observed_values: list[list[Annotated[float, Field(allow_inf_nan=False)]]]
    clean_values: list[list[Annotated[float, Field(allow_inf_nan=False)]]]
    counterfactual_initial_state: list[
        Annotated[float, Field(allow_inf_nan=False)]
    ] = Field(min_length=1, max_length=2)
    counterfactual_values: list[list[Annotated[float, Field(allow_inf_nan=False)]]]
    truth_coefficients: dict[Identifier, dict[Identifier, Annotated[float, Field(allow_inf_nan=False)]]]
    case_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_case(self) -> "PrivateDynamicsCaseV24":
        dimensions = len(self.state_names)
        if len(self.public_observed_values) >= len(self.clean_values):
            raise ValueError("private Dynamics case must retain an outer holdout")
        if len(self.times) != len(self.clean_values):
            raise ValueError("private Dynamics times and clean values differ in length")
        if any(len(row) != dimensions for row in self.clean_values):
            raise ValueError("clean Dynamics values do not match state dimension")
        if any(len(row) != dimensions for row in self.public_observed_values):
            raise ValueError("public Dynamics values do not match state dimension")
        if len(self.counterfactual_initial_state) != dimensions:
            raise ValueError("counterfactual initial state has wrong dimension")
        if any(len(row) != dimensions for row in self.counterfactual_values):
            raise ValueError("counterfactual Dynamics values have wrong dimension")
        if set(self.truth_coefficients) != set(self.state_names):
            raise ValueError("truth coefficients must cover each state equation")
        if self.case_hash and self.case_hash != self.content_hash():
            raise ValueError("case_hash does not match private Dynamics case")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "case_hash")

    def assert_sealed(self) -> None:
        if not self.case_hash or self.case_hash != self.content_hash():
            raise ValueError("private Dynamics case is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PrivateDynamicsCaseV24":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"case_hash"}),
            case_hash=draft.content_hash(),
        )

    def public_projection(
        self, spec: DynamicsWorldPackSpecV24
    ) -> DynamicsDataSnapshotV24:
        self.assert_sealed()
        spec.assert_sealed()
        public_count = spec.training_points + spec.inner_validation_points
        if len(self.public_observed_values) != public_count:
            raise ValueError("private Dynamics case is bound to another split spec")
        return DynamicsDataSnapshotV24.seal(
            snapshot_id=f"{self.case_id}_public",
            declared_state_names=self.state_names,
            observed_state_names=self.state_names,
            times=self.times[:public_count],
            values=self.public_observed_values,
        )


class PrivateDynamicsSentinelV24(StrictModel):
    schema_version: Literal["2.4"] = "2.4"
    sentinel_id: Identifier
    sentinel_kind: SentinelKind
    declared_state_names: list[Identifier] = Field(min_length=2, max_length=2)
    observed_state_names: list[Identifier] = Field(min_length=1, max_length=2)
    times: list[Annotated[float, Field(allow_inf_nan=False)]] = Field(min_length=81)
    values: list[list[Annotated[float, Field(allow_inf_nan=False)]]] = Field(min_length=81)
    hidden_failure_reason: Literal[
        "latent_state_not_observed", "candidate_library_not_excited"
    ]
    sentinel_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_sentinel(self) -> "PrivateDynamicsSentinelV24":
        if len(self.times) != len(self.values):
            raise ValueError("Dynamics sentinel times and values differ in length")
        if any(len(row) != len(self.observed_state_names) for row in self.values):
            raise ValueError("Dynamics sentinel values do not match observed states")
        if not set(self.observed_state_names).issubset(self.declared_state_names):
            raise ValueError("Dynamics sentinel observed states must be declared")
        if self.sentinel_kind == "partial_observation" and len(self.observed_state_names) != 1:
            raise ValueError("partial-observation sentinel must hide one state")
        if self.sentinel_kind == "rank_deficient_excitation" and len(self.observed_state_names) != 2:
            raise ValueError("rank-deficient sentinel must expose both collinear states")
        if self.sentinel_hash and self.sentinel_hash != self.content_hash():
            raise ValueError("sentinel_hash does not match Dynamics sentinel")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "sentinel_hash")

    def assert_sealed(self) -> None:
        if not self.sentinel_hash or self.sentinel_hash != self.content_hash():
            raise ValueError("Dynamics sentinel is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PrivateDynamicsSentinelV24":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"sentinel_hash"}),
            sentinel_hash=draft.content_hash(),
        )

    def public_projection(self) -> DynamicsDataSnapshotV24:
        self.assert_sealed()
        return DynamicsDataSnapshotV24.seal(
            snapshot_id=f"{self.sentinel_id}_public",
            declared_state_names=self.declared_state_names,
            observed_state_names=self.observed_state_names,
            times=self.times,
            values=self.values,
        )


class PrivateDynamicsWorldPackV24(StrictModel):
    schema_version: Literal["2.4"] = "2.4"
    pack_spec_hash: Sha256
    cases: list[PrivateDynamicsCaseV24] = Field(min_length=16)
    sentinels: list[PrivateDynamicsSentinelV24] = Field(min_length=2, max_length=2)
    generated_at: datetime
    pack_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_pack(self) -> "PrivateDynamicsWorldPackV24":
        _assert_timezone(self.generated_at, "generated_at")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("private Dynamics case ids must be unique")
        if {sentinel.sentinel_kind for sentinel in self.sentinels} != {
            "partial_observation",
            "rank_deficient_excitation",
        }:
            raise ValueError("private Dynamics pack needs both identifiability sentinels")
        for item in [*self.cases, *self.sentinels]:
            item.assert_sealed()
        if self.pack_hash and self.pack_hash != self.content_hash():
            raise ValueError("pack_hash does not match private Dynamics WorldPack")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "pack_hash")

    def assert_sealed(self) -> None:
        if not self.pack_hash or self.pack_hash != self.content_hash():
            raise ValueError("private Dynamics WorldPack is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PrivateDynamicsWorldPackV24":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"pack_hash"}),
            pack_hash=draft.content_hash(),
        )


class DynamicsSelectionBundleV24(StrictModel):
    schema_version: Literal["2.4"] = "2.4"
    bundle_id: Identifier
    pack_spec_hash: Sha256
    private_pack_hash: Sha256
    arm: Arm
    arm_policy_hash: Sha256
    case_receipts: list[DynamicsSelectionReceiptV24] = Field(min_length=16)
    sentinel_receipts: list[DynamicsSelectionReceiptV24] = Field(min_length=2, max_length=2)
    bundle_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bundle(self) -> "DynamicsSelectionBundleV24":
        receipts = [*self.case_receipts, *self.sentinel_receipts]
        if len({receipt.receipt_id for receipt in receipts}) != len(receipts):
            raise ValueError("Dynamics selection receipt ids must be unique")
        if any(receipt.arm != self.arm or receipt.arm_policy_hash != self.arm_policy_hash for receipt in receipts):
            raise ValueError("Dynamics selections are bound to another arm policy")
        for receipt in receipts:
            receipt.assert_sealed()
        if self.bundle_hash and self.bundle_hash != self.content_hash():
            raise ValueError("bundle_hash does not match Dynamics selections")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "bundle_hash")

    def assert_sealed(self) -> None:
        if not self.bundle_hash or self.bundle_hash != self.content_hash():
            raise ValueError("Dynamics selection bundle is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DynamicsSelectionBundleV24":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"bundle_hash"}),
            bundle_hash=draft.content_hash(),
        )


class DynamicsWorldPackCaseResultV24(StrictModel):
    case_id: Identifier
    mechanism: Mechanism
    seed: Annotated[int, Field(ge=0)]
    direct_candidate_id: Identifier
    memory_candidate_id: Identifier
    direct_outer_nrmse: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    memory_outer_nrmse: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    direct_counterfactual_nrmse: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    memory_counterfactual_nrmse: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    direct_combined_nrmse: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    memory_combined_nrmse: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    direct_structure_f1: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    memory_structure_f1: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    relative_improvement: Annotated[float, Field(allow_inf_nan=False)]
    structure_f1_improvement: Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)]
    outcome: Literal["memory_win", "tie", "memory_loss"]
    negative_transfer: bool
    direct_total_fit_count: Literal[5] = 5
    memory_total_fit_count: Literal[5] = 5


class DynamicsMechanismResultV24(StrictModel):
    mechanism: Mechanism
    case_count: Annotated[int, Field(ge=4)]
    mean_relative_improvement: Annotated[float, Field(allow_inf_nan=False)]
    confidence_lower: Annotated[float, Field(allow_inf_nan=False)]
    confidence_upper: Annotated[float, Field(allow_inf_nan=False)]
    noninferiority_margin: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    noninferior: bool

    @model_validator(mode="after")
    def validate_interval(self) -> "DynamicsMechanismResultV24":
        if self.confidence_lower > self.confidence_upper:
            raise ValueError("Dynamics mechanism interval is reversed")
        return self


ReportReason = Literal[
    "exploratory_not_eligible",
    "macro_improvement_interval_not_positive",
    "mechanism_noninferiority_failed",
    "negative_transfer_rate_bound_failed",
    "structure_recovery_not_improved",
    "identifiability_sentinel_false_promotion",
    "candidate_budget_mismatch",
]


class DynamicsWorldPackReportV24(StrictModel):
    schema_version: Literal["2.4"] = "2.4"
    report_id: Identifier
    phase: Literal["exploratory", "confirmation"]
    worldpack_spec_hash: Sha256
    knowledge_bundle_hash: Sha256
    private_pack_hash: Sha256
    direct_policy_hash: Sha256
    memory_policy_hash: Sha256
    direct_bundle_hash: Sha256
    memory_bundle_hash: Sha256
    cases: list[DynamicsWorldPackCaseResultV24] = Field(min_length=16)
    mechanism_results: list[DynamicsMechanismResultV24] = Field(min_length=4, max_length=4)
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
    negative_transfer_rate_upper: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    sentinel_false_promotion_count: Annotated[int, Field(ge=0, le=4)]
    same_candidate_and_fit_budget: bool
    status: Literal[
        "exploratory_only",
        "candidate_rejected_dynamics_v24",
        "promoted_for_synthetic_dynamics_worldpack_v24",
    ]
    reason_codes: list[ReportReason]
    evaluated_at: datetime
    report_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "DynamicsWorldPackReportV24":
        _assert_timezone(self.evaluated_at, "evaluated_at")
        if len({item.mechanism for item in self.mechanism_results}) != 4:
            raise ValueError("Dynamics report needs four unique mechanism results")
        if self.win_count + self.tie_count + self.loss_count != len(self.cases):
            raise ValueError("Dynamics outcome counts do not cover all cases")
        if self.status == "promoted_for_synthetic_dynamics_worldpack_v24" and self.reason_codes:
            raise ValueError("promoted Dynamics report cannot contain failure reasons")
        if self.status != "promoted_for_synthetic_dynamics_worldpack_v24" and not self.reason_codes:
            raise ValueError("non-promoted Dynamics report needs explicit reasons")
        if self.report_hash and self.report_hash != self.content_hash():
            raise ValueError("report_hash does not match Dynamics WorldPack report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "report_hash")

    def assert_sealed(self) -> None:
        if not self.report_hash or self.report_hash != self.content_hash():
            raise ValueError("Dynamics WorldPack report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DynamicsWorldPackReportV24":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"report_hash"}),
            report_hash=draft.content_hash(),
        )


class DynamicsOperatorQualificationV24(StrictModel):
    schema_version: Literal["2.4"] = "2.4"
    qualification_id: Identifier
    knowledge_bundle_hash: Sha256
    policy_hash: Sha256
    confirmation_report_hash: Sha256
    qualification_scope: Literal["synthetic_dynamics_worldpack_v24"] = (
        "synthetic_dynamics_worldpack_v24"
    )
    status: Literal["qualified"] = "qualified"
    limitations: list[Annotated[str, Field(min_length=12)]] = Field(min_length=4)
    qualified_at: datetime
    qualification_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_qualification(self) -> "DynamicsOperatorQualificationV24":
        _assert_timezone(self.qualified_at, "qualified_at")
        if self.qualification_hash and self.qualification_hash != self.content_hash():
            raise ValueError("qualification_hash does not match Dynamics qualification")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "qualification_hash")

    def assert_sealed(self) -> None:
        if not self.qualification_hash or self.qualification_hash != self.content_hash():
            raise ValueError("Dynamics qualification is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DynamicsOperatorQualificationV24":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"qualification_hash"}),
            qualification_hash=draft.content_hash(),
        )


class DynamicsWorldPackManifestV24(StrictModel):
    schema_version: Literal["2.4"] = "2.4"
    run_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")]
    artifact_refs: list[ArtifactRef] = Field(min_length=7, max_length=8)
    terminal_status: Literal[
        "exploratory_only",
        "candidate_rejected_dynamics_v24",
        "promoted_for_synthetic_dynamics_worldpack_v24",
    ]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "DynamicsWorldPackManifestV24":
        _assert_timezone(self.created_at, "created_at")
        if len({(ref.kind, ref.sha256) for ref in self.artifact_refs}) != len(self.artifact_refs):
            raise ValueError("Dynamics manifest references must be unique")
        qualification_count = sum(ref.kind == "dynamics_operator_qualification_v24" for ref in self.artifact_refs)
        if (self.terminal_status == "promoted_for_synthetic_dynamics_worldpack_v24") != (qualification_count == 1):
            raise ValueError("Dynamics qualification presence must match terminal status")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match Dynamics WorldPack manifest")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("Dynamics WorldPack manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DynamicsWorldPackManifestV24":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class DynamicsWorldPackOutcome:
    store: RunStore
    spec: DynamicsWorldPackSpecV24
    private_pack: PrivateDynamicsWorldPackV24
    direct_policy: DynamicsArmPolicyV24
    memory_policy: DynamicsArmPolicyV24
    direct_selections: DynamicsSelectionBundleV24
    memory_selections: DynamicsSelectionBundleV24
    report: DynamicsWorldPackReportV24
    qualification: DynamicsOperatorQualificationV24 | None
    manifest: DynamicsWorldPackManifestV24


def default_exploratory_dynamics_spec_v24(
    *, knowledge_bundle_hash: str, frozen_at: datetime | None = None
) -> DynamicsWorldPackSpecV24:
    return DynamicsWorldPackSpecV24.seal(
        pack_id="dynamics_cross_dialect_exploratory_v24",
        phase="exploratory",
        knowledge_bundle_hash=knowledge_bundle_hash,
        mechanisms=[
            "exponential_decay",
            "logistic_growth",
            "damped_oscillator",
            "lotka_volterra",
        ],
        seeds=list(EXPLORATORY_DYNAMICS_SEEDS),
        training_points=121,
        inner_validation_points=40,
        outer_holdout_points=60,
        time_step=0.05,
        observation_noise_fraction=0.01,
        confidence_level=0.95,
        bootstrap_replicates=5_000,
        bootstrap_seed=260_722,
        negative_transfer_relative_margin=0.05,
        maximum_negative_transfer_rate_upper=0.05,
        mechanism_noninferiority_margin=0.05,
        minimum_structure_f1_improvement_lower=0.0,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


def default_confirmation_dynamics_spec_v24(
    *,
    knowledge_bundle_hash: str,
    prior_exploratory_report_hash: str,
    frozen_at: datetime | None = None,
) -> DynamicsWorldPackSpecV24:
    return DynamicsWorldPackSpecV24.seal(
        pack_id="dynamics_cross_dialect_confirmation_v24",
        phase="confirmation",
        knowledge_bundle_hash=knowledge_bundle_hash,
        prior_exploratory_report_hash=prior_exploratory_report_hash,
        mechanisms=[
            "exponential_decay",
            "logistic_growth",
            "damped_oscillator",
            "lotka_volterra",
        ],
        seeds=list(CONFIRMATION_DYNAMICS_SEEDS),
        training_points=121,
        inner_validation_points=40,
        outer_holdout_points=60,
        time_step=0.05,
        observation_noise_fraction=0.01,
        confidence_level=0.95,
        bootstrap_replicates=10_000,
        bootstrap_seed=261_722,
        negative_transfer_relative_margin=0.05,
        maximum_negative_transfer_rate_upper=0.05,
        mechanism_noninferiority_margin=0.05,
        minimum_structure_f1_improvement_lower=0.0,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


def guarded_confirmation_dynamics_spec_v24(
    *,
    knowledge_bundle_hash: str,
    prior_exploratory_report_hash: str,
    prior_failure_report_hash: str,
    frozen_at: datetime | None = None,
) -> DynamicsWorldPackSpecV24:
    if (
        set(GUARDED_CONFIRMATION_DYNAMICS_SEEDS) & set(EXPLORATORY_DYNAMICS_SEEDS)
        or set(GUARDED_CONFIRMATION_DYNAMICS_SEEDS) & set(CONFIRMATION_DYNAMICS_SEEDS)
    ):
        raise RuntimeError("guarded Dynamics confirmation seeds overlap prior tranches")
    return DynamicsWorldPackSpecV24.seal(
        pack_id="dynamics_guarded_confirmation_v24",
        phase="confirmation",
        knowledge_bundle_hash=knowledge_bundle_hash,
        prior_exploratory_report_hash=prior_exploratory_report_hash,
        prior_failure_report_hash=prior_failure_report_hash,
        mechanisms=[
            "exponential_decay",
            "logistic_growth",
            "damped_oscillator",
            "lotka_volterra",
        ],
        seeds=list(GUARDED_CONFIRMATION_DYNAMICS_SEEDS),
        training_points=121,
        inner_validation_points=40,
        outer_holdout_points=60,
        time_step=0.05,
        observation_noise_fraction=0.01,
        confidence_level=0.95,
        bootstrap_replicates=10_000,
        bootstrap_seed=262_722,
        negative_transfer_relative_margin=0.05,
        maximum_negative_transfer_rate_upper=0.05,
        mechanism_noninferiority_margin=0.05,
        minimum_structure_f1_improvement_lower=0.0,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


def generate_private_dynamics_worldpack(
    spec: DynamicsWorldPackSpecV24,
    *,
    generated_at: datetime | None = None,
) -> PrivateDynamicsWorldPackV24:
    spec.assert_sealed()
    cases: list[PrivateDynamicsCaseV24] = []
    index = 0
    for seed in spec.seeds:
        for mechanism in spec.mechanisms:
            index += 1
            cases.append(_generate_case(spec, mechanism, seed, index))
    public_count = spec.training_points + spec.inner_validation_points
    times = [index * spec.time_step for index in range(public_count)]
    partial_values = [[1.0 + 0.2 * math.sin(time)] for time in times]
    rank_values = [
        [1.0 + 0.1 * math.sin(time), 2.0 + 0.2 * math.sin(time)] for time in times
    ]
    sentinels = [
        PrivateDynamicsSentinelV24.seal(
            sentinel_id="dynamics_partial_observation_sentinel",
            sentinel_kind="partial_observation",
            declared_state_names=["observed_sum", "latent_difference"],
            observed_state_names=["observed_sum"],
            times=times,
            values=partial_values,
            hidden_failure_reason="latent_state_not_observed",
        ),
        PrivateDynamicsSentinelV24.seal(
            sentinel_id="dynamics_rank_deficient_sentinel",
            sentinel_kind="rank_deficient_excitation",
            declared_state_names=["x", "y"],
            observed_state_names=["x", "y"],
            times=times,
            values=rank_values,
            hidden_failure_reason="candidate_library_not_excited",
        ),
    ]
    return PrivateDynamicsWorldPackV24.seal(
        pack_spec_hash=spec.spec_hash,
        cases=cases,
        sentinels=sentinels,
        generated_at=generated_at or spec.frozen_at,
    )


def select_dynamics_worldpack(
    spec: DynamicsWorldPackSpecV24,
    private_pack: PrivateDynamicsWorldPackV24,
    policy: DynamicsArmPolicyV24,
    *,
    selected_at: datetime | None = None,
) -> DynamicsSelectionBundleV24:
    spec.assert_sealed()
    private_pack.assert_sealed()
    policy.assert_sealed()
    if private_pack.pack_spec_hash != spec.spec_hash:
        raise ValueError("private Dynamics pack is bound to another spec")
    at = selected_at or datetime.now(timezone.utc)
    case_receipts = [
        select_dynamics_candidate(
            case.public_projection(spec),
            policy,
            training_points=spec.training_points,
            selected_at=at,
        )
        for case in private_pack.cases
    ]
    sentinel_receipts = [
        select_dynamics_candidate(
            sentinel.public_projection(),
            policy,
            training_points=spec.training_points,
            selected_at=at,
        )
        for sentinel in private_pack.sentinels
    ]
    assert spec.spec_hash is not None
    assert private_pack.pack_hash is not None
    assert policy.policy_hash is not None
    return DynamicsSelectionBundleV24.seal(
        bundle_id=f"{spec.pack_id}_{policy.arm}_selections",
        pack_spec_hash=spec.spec_hash,
        private_pack_hash=private_pack.pack_hash,
        arm=policy.arm,
        arm_policy_hash=policy.policy_hash,
        case_receipts=case_receipts,
        sentinel_receipts=sentinel_receipts,
    )


def evaluate_dynamics_worldpack(
    spec: DynamicsWorldPackSpecV24,
    private_pack: PrivateDynamicsWorldPackV24,
    direct_policy: DynamicsArmPolicyV24,
    memory_policy: DynamicsArmPolicyV24,
    direct: DynamicsSelectionBundleV24,
    memory: DynamicsSelectionBundleV24,
    *,
    evaluated_at: datetime | None = None,
) -> DynamicsWorldPackReportV24:
    for item in [spec, private_pack, direct_policy, memory_policy, direct, memory]:
        item.assert_sealed()
    if direct_policy.arm != "direct_generation" or memory_policy.arm != "retrieval_evolution_memory":
        raise ValueError("Dynamics report requires direct and memory arm policies")
    if memory_policy.knowledge_bundle_hash != spec.knowledge_bundle_hash:
        raise ValueError("Dynamics memory policy is bound to another knowledge bundle")
    if direct.arm_policy_hash != direct_policy.policy_hash or memory.arm_policy_hash != memory_policy.policy_hash:
        raise ValueError("Dynamics selection bundles are bound to other policies")
    if direct.private_pack_hash != private_pack.pack_hash or memory.private_pack_hash != private_pack.pack_hash:
        raise ValueError("Dynamics selections are bound to another private pack")
    direct_by_data = {receipt.public_data_hash: receipt for receipt in direct.case_receipts}
    memory_by_data = {receipt.public_data_hash: receipt for receipt in memory.case_receipts}
    results: list[DynamicsWorldPackCaseResultV24] = []
    for case in private_pack.cases:
        public = case.public_projection(spec)
        assert public.snapshot_hash is not None
        direct_receipt = direct_by_data[public.snapshot_hash]
        memory_receipt = memory_by_data[public.snapshot_hash]
        expected_direct = select_dynamics_candidate(
            public,
            direct_policy,
            training_points=spec.training_points,
            selected_at=direct_receipt.selected_at,
        )
        expected_memory = select_dynamics_candidate(
            public,
            memory_policy,
            training_points=spec.training_points,
            selected_at=memory_receipt.selected_at,
        )
        if direct_receipt.receipt_hash != expected_direct.receipt_hash:
            raise ValueError("direct Dynamics selection does not replay")
        if memory_receipt.receipt_hash != expected_memory.receipt_hash:
            raise ValueError("memory Dynamics selection does not replay")
        if direct_receipt.status != "selected" or memory_receipt.status != "selected":
            raise ValueError("a primary Dynamics case abstained; report cannot claim a paired effect")
        direct_metrics = _outer_metrics(case, spec, public, direct_policy, direct_receipt)
        memory_metrics = _outer_metrics(case, spec, public, memory_policy, memory_receipt)
        direct_combined = (direct_metrics[0] + direct_metrics[1]) / 2.0
        memory_combined = (memory_metrics[0] + memory_metrics[1]) / 2.0
        relative = (direct_combined - memory_combined) / max(direct_combined, 1e-12)
        tolerance = max(1e-12, direct_combined * 1e-9)
        improvement = direct_combined - memory_combined
        outcome = (
            "memory_win"
            if improvement > tolerance
            else "memory_loss"
            if improvement < -tolerance
            else "tie"
        )
        negative = memory_combined > direct_combined * (
            1.0 + spec.negative_transfer_relative_margin
        ) + 1e-12
        results.append(
            DynamicsWorldPackCaseResultV24(
                case_id=case.case_id,
                mechanism=case.mechanism,
                seed=case.seed,
                direct_candidate_id=direct_receipt.selected_candidate_id,
                memory_candidate_id=memory_receipt.selected_candidate_id,
                direct_outer_nrmse=direct_metrics[0],
                memory_outer_nrmse=memory_metrics[0],
                direct_counterfactual_nrmse=direct_metrics[1],
                memory_counterfactual_nrmse=memory_metrics[1],
                direct_combined_nrmse=direct_combined,
                memory_combined_nrmse=memory_combined,
                direct_structure_f1=direct_metrics[2],
                memory_structure_f1=memory_metrics[2],
                relative_improvement=relative,
                structure_f1_improvement=memory_metrics[2] - direct_metrics[2],
                outcome=outcome,
                negative_transfer=negative,
            )
        )

    grouped_relative: dict[str, list[float]] = {mechanism: [] for mechanism in spec.mechanisms}
    grouped_structure: dict[str, list[float]] = {mechanism: [] for mechanism in spec.mechanisms}
    for result in results:
        grouped_relative[result.mechanism].append(result.relative_improvement)
        grouped_structure[result.mechanism].append(result.structure_f1_improvement)
    relative_draws = _stratified_macro_bootstrap(grouped_relative, spec, seed_offset=0)
    structure_draws = _stratified_macro_bootstrap(grouped_structure, spec, seed_offset=1)
    alpha = (1.0 - spec.confidence_level) / 2.0
    macro_lower, macro_upper = np.quantile(relative_draws, [alpha, 1.0 - alpha], method="linear")
    structure_lower, structure_upper = np.quantile(structure_draws, [alpha, 1.0 - alpha], method="linear")
    mechanism_results: list[DynamicsMechanismResultV24] = []
    for offset, mechanism in enumerate(spec.mechanisms, start=11):
        values = grouped_relative[mechanism]
        lower, upper = _bootstrap_interval(values, spec, seed_offset=offset)
        mechanism_results.append(
            DynamicsMechanismResultV24(
                mechanism=mechanism,
                case_count=len(values),
                mean_relative_improvement=float(np.mean(values)),
                confidence_lower=lower,
                confidence_upper=upper,
                noninferiority_margin=spec.mechanism_noninferiority_margin,
                noninferior=lower >= -spec.mechanism_noninferiority_margin,
            )
        )
    negative_count = sum(result.negative_transfer for result in results)
    negative_upper = _clopper_pearson_upper(negative_count, len(results), spec.confidence_level)
    sentinel_false_promotions = _sentinel_false_promotions(
        spec, private_pack, direct_policy, memory_policy, direct, memory
    )
    same_budget = (
        direct_policy.candidate_budget == memory_policy.candidate_budget == spec.candidate_budget_per_arm
        and all(result.direct_total_fit_count == result.memory_total_fit_count for result in results)
    )
    reasons: list[ReportReason] = []
    if spec.phase == "exploratory":
        reasons.append("exploratory_not_eligible")
        status = "exploratory_only"
    else:
        if float(macro_lower) <= 0:
            reasons.append("macro_improvement_interval_not_positive")
        if not all(result.noninferior for result in mechanism_results):
            reasons.append("mechanism_noninferiority_failed")
        if negative_upper > spec.maximum_negative_transfer_rate_upper:
            reasons.append("negative_transfer_rate_bound_failed")
        if float(structure_lower) < spec.minimum_structure_f1_improvement_lower:
            reasons.append("structure_recovery_not_improved")
        if sentinel_false_promotions:
            reasons.append("identifiability_sentinel_false_promotion")
        if not same_budget:
            reasons.append("candidate_budget_mismatch")
        status = (
            "candidate_rejected_dynamics_v24"
            if reasons
            else "promoted_for_synthetic_dynamics_worldpack_v24"
        )
    assert spec.spec_hash is not None
    assert private_pack.pack_hash is not None
    assert direct_policy.policy_hash is not None
    assert memory_policy.policy_hash is not None
    assert direct.bundle_hash is not None
    assert memory.bundle_hash is not None
    return DynamicsWorldPackReportV24.seal(
        report_id=f"{spec.pack_id}_report",
        phase=spec.phase,
        worldpack_spec_hash=spec.spec_hash,
        knowledge_bundle_hash=spec.knowledge_bundle_hash,
        private_pack_hash=private_pack.pack_hash,
        direct_policy_hash=direct_policy.policy_hash,
        memory_policy_hash=memory_policy.policy_hash,
        direct_bundle_hash=direct.bundle_hash,
        memory_bundle_hash=memory.bundle_hash,
        cases=results,
        mechanism_results=mechanism_results,
        macro_mean_relative_improvement=float(
            np.mean([np.mean(grouped_relative[mechanism]) for mechanism in spec.mechanisms])
        ),
        macro_confidence_lower=float(macro_lower),
        macro_confidence_upper=float(macro_upper),
        macro_mean_structure_f1_improvement=float(
            np.mean([np.mean(grouped_structure[mechanism]) for mechanism in spec.mechanisms])
        ),
        structure_confidence_lower=float(structure_lower),
        structure_confidence_upper=float(structure_upper),
        win_count=sum(result.outcome == "memory_win" for result in results),
        tie_count=sum(result.outcome == "tie" for result in results),
        loss_count=sum(result.outcome == "memory_loss" for result in results),
        negative_transfer_count=negative_count,
        negative_transfer_rate=negative_count / len(results),
        negative_transfer_rate_upper=negative_upper,
        sentinel_false_promotion_count=sentinel_false_promotions,
        same_candidate_and_fit_budget=same_budget,
        status=status,
        reason_codes=reasons,
        evaluated_at=evaluated_at or datetime.now(timezone.utc),
    )


def qualify_dynamics_policy_v24(
    policy: DynamicsArmPolicyV24,
    report: DynamicsWorldPackReportV24,
) -> DynamicsOperatorQualificationV24:
    policy.assert_sealed()
    report.assert_sealed()
    if report.status != "promoted_for_synthetic_dynamics_worldpack_v24":
        raise ValueError("a rejected Dynamics policy cannot be qualified")
    if report.memory_policy_hash != policy.policy_hash:
        raise ValueError("Dynamics report is bound to another memory policy")
    if report.knowledge_bundle_hash != policy.knowledge_bundle_hash:
        raise ValueError("Dynamics policy and report use different knowledge bundles")
    assert policy.policy_hash is not None
    assert policy.knowledge_bundle_hash is not None
    assert report.report_hash is not None
    return DynamicsOperatorQualificationV24.seal(
        qualification_id=f"{policy.policy_id}_qualification",
        knowledge_bundle_hash=policy.knowledge_bundle_hash,
        policy_hash=policy.policy_hash,
        confirmation_report_hash=report.report_hash,
        limitations=[
            "valid only for four frozen synthetic polynomial ODE mechanisms with full state observation",
            "does not establish symbolic structural identifiability or hidden-state observability",
            "does not cover irregular sampling, control inputs, stochastic dynamics, delays, PDEs, or hybrid events",
            "does not authorize real-data parameter claims, interventions, or external decisions",
        ],
        qualified_at=report.evaluated_at,
    )


def run_dynamics_worldpack(
    output_root: str | Path,
    *,
    spec: DynamicsWorldPackSpecV24,
    direct_policy: DynamicsArmPolicyV24,
    memory_policy: DynamicsArmPolicyV24,
    evaluated_at: datetime | None = None,
    run_id: str | None = None,
) -> DynamicsWorldPackOutcome:
    spec.assert_sealed()
    direct_policy.assert_sealed()
    memory_policy.assert_sealed()
    if direct_policy.arm != "direct_generation" or memory_policy.arm != "retrieval_evolution_memory":
        raise ValueError("Dynamics WorldPack run needs direct and memory policies")
    if memory_policy.knowledge_bundle_hash != spec.knowledge_bundle_hash:
        raise ValueError("Dynamics memory policy is bound to another knowledge bundle")
    at = evaluated_at or datetime.now(timezone.utc)
    store = RunStore(output_root, run_id=run_id or f"dynamics-worldpack-{uuid4().hex[:10]}")
    refs = [
        store.put_artifact("dynamics_worldpack_spec_v24", spec),
        store.put_artifact("dynamics_direct_policy_v24", direct_policy),
        store.put_artifact("dynamics_memory_policy_v24", memory_policy),
    ]
    store.emit(
        "dynamics_protocol_frozen_before_private_pack",
        {
            "spec_hash": spec.spec_hash,
            "direct_policy_hash": direct_policy.policy_hash,
            "memory_policy_hash": memory_policy.policy_hash,
        },
    )
    private_pack = generate_private_dynamics_worldpack(spec, generated_at=at)
    direct = select_dynamics_worldpack(spec, private_pack, direct_policy, selected_at=at)
    memory = select_dynamics_worldpack(spec, private_pack, memory_policy, selected_at=at)
    report = evaluate_dynamics_worldpack(
        spec,
        private_pack,
        direct_policy,
        memory_policy,
        direct,
        memory,
        evaluated_at=at,
    )
    refs.extend(
        [
            store.put_artifact("private_dynamics_worldpack_v24", private_pack),
            store.put_artifact("dynamics_direct_selections_v24", direct),
            store.put_artifact("dynamics_memory_selections_v24", memory),
            store.put_artifact("dynamics_worldpack_report_v24", report),
        ]
    )
    qualification = None
    if report.status == "promoted_for_synthetic_dynamics_worldpack_v24":
        qualification = qualify_dynamics_policy_v24(memory_policy, report)
        refs.append(store.put_artifact("dynamics_operator_qualification_v24", qualification))
    manifest = DynamicsWorldPackManifestV24.seal(
        run_id=store.run_id,
        artifact_refs=refs,
        terminal_status=report.status,
        created_at=at,
    )
    manifest_ref = store.put_artifact("dynamics_worldpack_manifest_v24", manifest)
    store.emit(
        "dynamics_worldpack_adjudicated",
        {"manifest_ref": manifest_ref.model_dump(mode="json")},
    )
    if not verify_dynamics_worldpack_run(store.run_directory):
        raise RuntimeError("Dynamics WorldPack run failed independent verification")
    return DynamicsWorldPackOutcome(
        store,
        spec,
        private_pack,
        direct_policy,
        memory_policy,
        direct,
        memory,
        report,
        qualification,
        manifest,
    )


def verify_dynamics_worldpack_run(run_directory: str | Path) -> bool:
    try:
        store = RunStore.open_existing(run_directory)
        events = [json.loads(line) for line in store.event_path.read_text(encoding="utf-8").splitlines()]
        committed = [
            ArtifactRef.model_validate(event["payload"])
            for event in events
            if event["event_type"] == "artifact_committed"
        ]
        for ref in committed:
            store.load_artifact(ref)
        manifests = [ref for ref in committed if ref.kind == "dynamics_worldpack_manifest_v24"]
        if len(manifests) != 1:
            return False
        manifest = DynamicsWorldPackManifestV24.model_validate(store.load_artifact(manifests[0]))
        manifest.assert_sealed()
        if manifest.run_id != store.run_id:
            return False

        def load_one(kind: str, model):
            refs = [ref for ref in manifest.artifact_refs if ref.kind == kind]
            if len(refs) != 1:
                raise RuntimeError(f"manifest needs exactly one {kind}")
            return model.model_validate(store.load_artifact(refs[0]))

        spec = load_one("dynamics_worldpack_spec_v24", DynamicsWorldPackSpecV24)
        direct_policy = load_one("dynamics_direct_policy_v24", DynamicsArmPolicyV24)
        memory_policy = load_one("dynamics_memory_policy_v24", DynamicsArmPolicyV24)
        private_pack = load_one("private_dynamics_worldpack_v24", PrivateDynamicsWorldPackV24)
        direct = load_one("dynamics_direct_selections_v24", DynamicsSelectionBundleV24)
        memory = load_one("dynamics_memory_selections_v24", DynamicsSelectionBundleV24)
        report = load_one("dynamics_worldpack_report_v24", DynamicsWorldPackReportV24)
        for item in [spec, direct_policy, memory_policy, private_pack, direct, memory, report]:
            item.assert_sealed()
        regenerated = generate_private_dynamics_worldpack(spec, generated_at=private_pack.generated_at)
        if regenerated.pack_hash != private_pack.pack_hash:
            return False
        direct_time = direct.case_receipts[0].selected_at
        memory_time = memory.case_receipts[0].selected_at
        replay_direct = select_dynamics_worldpack(spec, private_pack, direct_policy, selected_at=direct_time)
        replay_memory = select_dynamics_worldpack(spec, private_pack, memory_policy, selected_at=memory_time)
        if replay_direct.bundle_hash != direct.bundle_hash or replay_memory.bundle_hash != memory.bundle_hash:
            return False
        recomputed = evaluate_dynamics_worldpack(
            spec,
            private_pack,
            direct_policy,
            memory_policy,
            direct,
            memory,
            evaluated_at=report.evaluated_at,
        )
        if recomputed.report_hash != report.report_hash or manifest.terminal_status != report.status:
            return False
        qualifications = [ref for ref in manifest.artifact_refs if ref.kind == "dynamics_operator_qualification_v24"]
        if report.status == "promoted_for_synthetic_dynamics_worldpack_v24":
            if len(qualifications) != 1:
                return False
            stored = DynamicsOperatorQualificationV24.model_validate(store.load_artifact(qualifications[0]))
            stored.assert_sealed()
            expected = qualify_dynamics_policy_v24(memory_policy, report)
            if stored.qualification_hash != expected.qualification_hash:
                return False
        elif qualifications:
            return False
        freeze_events = [
            event for event in events if event["event_type"] == "dynamics_protocol_frozen_before_private_pack"
        ]
        return len(freeze_events) == 1
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


def _generate_case(
    spec: DynamicsWorldPackSpecV24,
    mechanism: Mechanism,
    seed: int,
    index: int,
) -> PrivateDynamicsCaseV24:
    random = Random(seed * 104_729 + index * 7_919)
    total = spec.training_points + spec.inner_validation_points + spec.outer_holdout_points
    times = np.arange(total, dtype=float) * spec.time_step
    state_names, initial, parameters, truth = _mechanism_parameters(mechanism, random)
    clean = _simulate_truth(mechanism, initial, times, parameters)
    public_count = spec.training_points + spec.inner_validation_points
    rng = np.random.default_rng(seed * 65_537 + index * 257)
    scales = np.std(clean[:public_count], axis=0)
    fallbacks = np.maximum(np.mean(np.abs(clean[:public_count]), axis=0), 1.0)
    scales = np.where(scales > 1e-8, scales, fallbacks)
    noisy = clean[:public_count] + rng.normal(
        0.0,
        spec.observation_noise_fraction * scales,
        size=(public_count, len(state_names)),
    )
    if mechanism in {"exponential_decay", "logistic_growth", "lotka_volterra"}:
        noisy = np.maximum(noisy, 1e-6)
    boundary = clean[public_count - 1].copy()
    if mechanism in {"exponential_decay", "logistic_growth"}:
        counterfactual_initial = boundary * 1.2
    elif mechanism == "damped_oscillator":
        counterfactual_initial = boundary + np.asarray([0.25, -0.2])
    else:
        counterfactual_initial = boundary * np.asarray([1.15, 0.85])
    counter_times = np.arange(spec.outer_holdout_points + 1, dtype=float) * spec.time_step
    counterfactual = _simulate_truth(
        mechanism, counterfactual_initial, counter_times, parameters
    )
    return PrivateDynamicsCaseV24.seal(
        case_id=f"dyn_case_{index:03d}",
        mechanism=mechanism,
        seed=seed,
        state_names=state_names,
        times=times.tolist(),
        public_observed_values=noisy.tolist(),
        clean_values=clean.tolist(),
        counterfactual_initial_state=counterfactual_initial.tolist(),
        counterfactual_values=counterfactual.tolist(),
        truth_coefficients=truth,
    )


def _mechanism_parameters(
    mechanism: Mechanism, random: Random
) -> tuple[list[str], np.ndarray, dict[str, float], dict[str, dict[str, float]]]:
    if mechanism == "exponential_decay":
        rate = 0.35 + 0.25 * random.random()
        return ["x"], np.asarray([1.5 + random.random()]), {"rate": rate}, {"x": {"x": -rate}}
    if mechanism == "logistic_growth":
        rate = 0.55 + 0.35 * random.random()
        capacity = 7.0 + 4.0 * random.random()
        return (
            ["x"],
            np.asarray([0.5 + random.random()]),
            {"rate": rate, "capacity": capacity},
            {"x": {"x": rate, "x2": -rate / capacity}},
        )
    if mechanism == "damped_oscillator":
        omega = 0.75 + 0.5 * random.random()
        damping = 0.08 + 0.25 * random.random()
        return (
            ["position", "velocity"],
            np.asarray([1.0 + 0.4 * random.random(), -0.2 + 0.4 * random.random()]),
            {"omega": omega, "damping": damping},
            {
                "position": {"velocity": 1.0},
                "velocity": {"position": -(omega**2), "velocity": -damping},
            },
        )
    alpha = 0.7 + 0.4 * random.random()
    beta_value = 0.06 + 0.04 * random.random()
    delta = 0.04 + 0.03 * random.random()
    gamma = 0.8 + 0.4 * random.random()
    return (
        ["prey", "predator"],
        np.asarray([8.0 + 4.0 * random.random(), 5.0 + 3.0 * random.random()]),
        {"alpha": alpha, "beta": beta_value, "delta": delta, "gamma": gamma},
        {
            "prey": {"prey": alpha, "prey_predator": -beta_value},
            "predator": {"prey_predator": delta, "predator": -gamma},
        },
    )


def _simulate_truth(
    mechanism: Mechanism,
    initial: np.ndarray,
    times: np.ndarray,
    parameters: dict[str, float],
) -> np.ndarray:
    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        if mechanism == "exponential_decay":
            return np.asarray([-parameters["rate"] * state[0]])
        if mechanism == "logistic_growth":
            return np.asarray(
                [parameters["rate"] * state[0] * (1.0 - state[0] / parameters["capacity"])]
            )
        if mechanism == "damped_oscillator":
            return np.asarray(
                [
                    state[1],
                    -(parameters["omega"] ** 2) * state[0]
                    - parameters["damping"] * state[1],
                ]
            )
        return np.asarray(
            [
                parameters["alpha"] * state[0] - parameters["beta"] * state[0] * state[1],
                parameters["delta"] * state[0] * state[1] - parameters["gamma"] * state[1],
            ]
        )

    solution = solve_ivp(
        rhs,
        (float(times[0]), float(times[-1])),
        initial,
        t_eval=times,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
    )
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError("private Dynamics truth simulator failed")
    return solution.y.T


def _outer_metrics(
    case: PrivateDynamicsCaseV24,
    spec: DynamicsWorldPackSpecV24,
    public: DynamicsDataSnapshotV24,
    policy: DynamicsArmPolicyV24,
    receipt: DynamicsSelectionReceiptV24,
) -> tuple[float, float, float]:
    definition = next(
        candidate for candidate in policy.candidates if candidate.candidate_id == receipt.selected_candidate_id
    )
    fit = fit_dynamics_candidate(public, definition, policy, fitted_at=receipt.selected_at)
    if fit.status != "fit_succeeded" or fit.model is None:
        raise ValueError("selected Dynamics candidate did not refit on public data")
    public_count = spec.training_points + spec.inner_validation_points
    future_times = case.times[public_count - 1 :]
    predicted = simulate_dynamics_model(
        fit.model,
        case.public_observed_values[-1],
        future_times,
    )[1:]
    outer_truth = case.clean_values[public_count:]
    outer_nrmse = trajectory_nrmse(outer_truth, predicted)
    counter_times = [index * spec.time_step for index in range(spec.outer_holdout_points + 1)]
    counter_predicted = simulate_dynamics_model(
        fit.model,
        case.counterfactual_initial_state,
        counter_times,
    )[1:]
    counter_nrmse = trajectory_nrmse(case.counterfactual_values[1:], counter_predicted)
    structure = support_f1(fit.model, case.truth_coefficients)
    return outer_nrmse, counter_nrmse, structure


def _sentinel_false_promotions(
    spec: DynamicsWorldPackSpecV24,
    private_pack: PrivateDynamicsWorldPackV24,
    direct_policy: DynamicsArmPolicyV24,
    memory_policy: DynamicsArmPolicyV24,
    direct: DynamicsSelectionBundleV24,
    memory: DynamicsSelectionBundleV24,
) -> int:
    false_promotions = 0
    for sentinel, direct_receipt, memory_receipt in zip(
        private_pack.sentinels,
        direct.sentinel_receipts,
        memory.sentinel_receipts,
        strict=True,
    ):
        public = sentinel.public_projection()
        for policy, receipt in (
            (direct_policy, direct_receipt),
            (memory_policy, memory_receipt),
        ):
            expected = select_dynamics_candidate(
                public,
                policy,
                training_points=spec.training_points,
                selected_at=receipt.selected_at,
            )
            if expected.receipt_hash != receipt.receipt_hash:
                raise ValueError("Dynamics sentinel selection does not replay")
            false_promotions += receipt.status == "selected"
    return false_promotions


def _stratified_macro_bootstrap(
    grouped: dict[str, list[float]],
    spec: DynamicsWorldPackSpecV24,
    *,
    seed_offset: int,
) -> np.ndarray:
    random = Random(spec.bootstrap_seed + seed_offset)
    mechanisms = list(spec.mechanisms)
    draws = np.empty(spec.bootstrap_replicates, dtype=float)
    for draw in range(spec.bootstrap_replicates):
        mechanism_means = []
        for mechanism in mechanisms:
            values = grouped[mechanism]
            mechanism_means.append(
                sum(values[random.randrange(len(values))] for _ in values) / len(values)
            )
        draws[draw] = sum(mechanism_means) / len(mechanism_means)
    return draws


def _bootstrap_interval(
    values: list[float],
    spec: DynamicsWorldPackSpecV24,
    *,
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
