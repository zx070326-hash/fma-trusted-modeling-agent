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
from scipy.signal import savgol_filter
from scipy.stats import beta

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore

from .dynamics_ir import (
    PolynomialBasisTermV24,
    evaluate_polynomial_library,
    polynomial_basis_terms,
    trajectory_nrmse,
)
from .dynamics_worldpack import Mechanism
from .schemas import Identifier, Sha256, _assert_timezone


DesignArmV26 = Literal[
    "random_safe_catalog",
    "ensemble_disagreement_catalog",
]

EXPLORATORY_ACTIVE_DESIGN_SEEDS_V26 = (
    6101,
    6151,
    6203,
    6257,
    6301,
    6353,
    6401,
    6451,
)
EVOLVED_ACTIVE_DESIGN_SEEDS_V26 = (
    6503,
    6551,
    6607,
    6653,
    6701,
    6751,
    6803,
    6857,
)
CONFIRMATION_ACTIVE_DESIGN_SEEDS_V26 = (
    7001,
    7057,
    7103,
    7151,
    7207,
    7253,
    7307,
    7351,
    7403,
    7451,
    7507,
    7559,
    7603,
    7651,
    7703,
    7757,
    7801,
    7853,
    7901,
    7951,
)


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class ActiveDesignPolicyV26(StrictModel):
    schema_version: Literal["2.6"] = "2.6"
    policy_id: Identifier
    arm: DesignArmV26
    knowledge_bundle_hash: Sha256
    prior_failure_report_hash: Sha256
    acquisition_rule: Literal[
        "prefrozen_random_without_replacement",
        "bootstrap_model_derivative_disagreement",
    ]
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "ActiveDesignPolicyV26":
        expected = {
            "random_safe_catalog": "prefrozen_random_without_replacement",
            "ensemble_disagreement_catalog": (
                "bootstrap_model_derivative_disagreement"
            ),
        }[self.arm]
        if self.acquisition_rule != expected:
            raise ValueError("V2.6 arm and acquisition rule disagree")
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("policy_hash does not match V2.6 active-design policy")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("V2.6 active-design policy is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ActiveDesignPolicyV26":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"policy_hash"}),
            policy_hash=draft.content_hash(),
        )


class ActiveDesignWorldPackSpecV26(StrictModel):
    schema_version: Literal["2.6"] = "2.6"
    experiment_id: Identifier
    phase: Literal["exploratory", "confirmation"]
    mechanisms: list[Mechanism] = Field(min_length=4, max_length=4)
    seeds: list[int] = Field(min_length=2, max_length=20)
    candidate_action_count: Annotated[int, Field(ge=12, le=64)] = 24
    action_budget: Annotated[int, Field(ge=2, le=12)] = 4
    trajectory_points: Annotated[int, Field(ge=15, le=81)] = 25
    time_step: Annotated[float, Field(gt=0, le=0.2, allow_inf_nan=False)] = 0.08
    observation_noise_fraction: Annotated[
        float, Field(ge=0, le=0.1, allow_inf_nan=False)
    ] = 0.02
    polynomial_degree: Literal[2] = 2
    savgol_window: Annotated[int, Field(ge=5, le=21)] = 9
    savgol_polynomial_order: Annotated[int, Field(ge=2, le=5)] = 3
    ridge_alpha: Annotated[float, Field(gt=0, le=0.1, allow_inf_nan=False)] = 1e-4
    sparsity_threshold: Annotated[
        float, Field(gt=0, le=0.2, allow_inf_nan=False)
    ] = 0.025
    maximum_iterations: Annotated[int, Field(ge=2, le=30)] = 12
    ensemble_members: Annotated[int, Field(ge=8, le=64)] = 24
    bootstrap_fraction: Annotated[
        float, Field(ge=0.5, le=1.0, allow_inf_nan=False)
    ] = 0.8
    probe_trajectory_count: Literal[4] = 4
    bootstrap_replicates: Annotated[int, Field(ge=200, le=5000)] = 2000
    bootstrap_seed: int = 260722
    confidence_level: Literal[0.95] = 0.95
    minimum_macro_relative_improvement: Annotated[
        float, Field(ge=0, le=0.2, allow_inf_nan=False)
    ] = 0.0
    maximum_negative_transfer_rate: Annotated[
        float, Field(gt=0, le=0.5, allow_inf_nan=False)
    ] = 0.25
    maximum_mechanism_regression: Annotated[
        float, Field(ge=0, le=0.25, allow_inf_nan=False)
    ] = 0.05
    baseline_policy_hash: Sha256
    active_policy_hash: Sha256
    knowledge_bundle_hash: Sha256
    prior_failure_report_hash: Sha256
    method_evidence_hash: Sha256
    frozen_delta: Literal[
        "prefrozen_random_vs_sequential_ensemble_disagreement_action_selection_only"
    ] = "prefrozen_random_vs_sequential_ensemble_disagreement_action_selection_only"
    frozen_at: datetime
    spec_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "ActiveDesignWorldPackSpecV26":
        _assert_timezone(self.frozen_at, "frozen_at")
        if len(set(self.mechanisms)) != 4:
            raise ValueError("V2.6 requires four distinct mechanisms")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("V2.6 seeds must be unique")
        if self.savgol_window % 2 == 0:
            raise ValueError("V2.6 Savitzky-Golay window must be odd")
        if self.savgol_polynomial_order >= self.savgol_window:
            raise ValueError("V2.6 Savitzky-Golay order is invalid")
        if self.trajectory_points <= self.savgol_window + 4:
            raise ValueError("V2.6 trajectories are too short for the frozen estimator")
        if self.action_budget >= self.candidate_action_count:
            raise ValueError("V2.6 action budget must leave unselected actions")
        exploratory = set(EXPLORATORY_ACTIVE_DESIGN_SEEDS_V26) | set(
            EVOLVED_ACTIVE_DESIGN_SEEDS_V26
        )
        confirmation = set(CONFIRMATION_ACTIVE_DESIGN_SEEDS_V26)
        if self.phase == "exploratory" and not set(self.seeds).issubset(exploratory):
            raise ValueError("exploratory V2.6 seeds are outside the frozen family")
        if self.phase == "confirmation" and not set(self.seeds).issubset(confirmation):
            raise ValueError("confirmation V2.6 seeds are outside the frozen family")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match V2.6 active-design spec")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "spec_hash")

    def assert_sealed(self) -> None:
        if not self.spec_hash or self.spec_hash != self.content_hash():
            raise ValueError("V2.6 active-design spec is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ActiveDesignWorldPackSpecV26":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"spec_hash"}),
            spec_hash=draft.content_hash(),
        )


class SafeInitialConditionActionV26(StrictModel):
    action_id: Identifier
    initial_state: list[Annotated[float, Field(allow_inf_nan=False)]] = Field(
        min_length=1, max_length=2
    )
    action_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_action(self) -> "SafeInitialConditionActionV26":
        if self.action_hash and self.action_hash != self.content_hash():
            raise ValueError("action_hash does not match V2.6 action")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "action_hash")

    def assert_sealed(self) -> None:
        if not self.action_hash or self.action_hash != self.content_hash():
            raise ValueError("V2.6 action is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "SafeInitialConditionActionV26":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"action_hash"}),
            action_hash=draft.content_hash(),
        )


class ActiveDesignObservationV26(StrictModel):
    action_id: Identifier
    action_hash: Sha256
    times: list[Annotated[float, Field(allow_inf_nan=False)]] = Field(min_length=15)
    values: list[list[Annotated[float, Field(allow_inf_nan=False)]]] = Field(
        min_length=15
    )
    trust_class: Literal["untrusted_synthetic_observation"] = (
        "untrusted_synthetic_observation"
    )
    observation_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_observation(self) -> "ActiveDesignObservationV26":
        if len(self.times) != len(self.values):
            raise ValueError("V2.6 observation times and values differ")
        if any(len(row) != len(self.values[0]) for row in self.values):
            raise ValueError("V2.6 observation state dimensions differ")
        if any(right <= left for left, right in zip(self.times, self.times[1:])):
            raise ValueError("V2.6 observation times must increase")
        if self.observation_hash and self.observation_hash != self.content_hash():
            raise ValueError("observation_hash does not match V2.6 observation")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "observation_hash")

    def assert_sealed(self) -> None:
        if not self.observation_hash or self.observation_hash != self.content_hash():
            raise ValueError("V2.6 observation is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ActiveDesignObservationV26":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"observation_hash"}),
            observation_hash=draft.content_hash(),
        )


class ActiveDesignPublicCaseV26(StrictModel):
    schema_version: Literal["2.6"] = "2.6"
    case_id: Identifier
    mechanism: Mechanism
    state_names: list[Identifier] = Field(min_length=1, max_length=2)
    state_lower_bounds: list[Annotated[float, Field(allow_inf_nan=False)]]
    state_upper_bounds: list[Annotated[float, Field(allow_inf_nan=False)]]
    action_catalog: list[SafeInitialConditionActionV26] = Field(min_length=12)
    pilot_observation: ActiveDesignObservationV26
    public_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_public_case(self) -> "ActiveDesignPublicCaseV26":
        dimension = len(self.state_names)
        if not (
            len(self.state_lower_bounds)
            == len(self.state_upper_bounds)
            == dimension
        ):
            raise ValueError("V2.6 state-bound dimensions disagree")
        if any(low >= high for low, high in zip(
            self.state_lower_bounds, self.state_upper_bounds, strict=True
        )):
            raise ValueError("V2.6 state bounds are invalid")
        if len({action.action_id for action in self.action_catalog}) != len(
            self.action_catalog
        ):
            raise ValueError("V2.6 action ids must be unique")
        for action in self.action_catalog:
            action.assert_sealed()
            if len(action.initial_state) != dimension:
                raise ValueError("V2.6 action dimension disagrees with states")
            _assert_action_safe(self, action)
        self.pilot_observation.assert_sealed()
        if len(self.pilot_observation.values[0]) != dimension:
            raise ValueError("V2.6 pilot observation dimension disagrees")
        if self.public_hash and self.public_hash != self.content_hash():
            raise ValueError("public_hash does not match V2.6 public case")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "public_hash")

    def assert_sealed(self) -> None:
        if not self.public_hash or self.public_hash != self.content_hash():
            raise ValueError("V2.6 public case is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ActiveDesignPublicCaseV26":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"public_hash"}),
            public_hash=draft.content_hash(),
        )


class PrivateActiveDesignCaseV26(StrictModel):
    schema_version: Literal["2.6-private"] = "2.6-private"
    public_case: ActiveDesignPublicCaseV26
    hidden_parameters: dict[str, Annotated[float, Field(allow_inf_nan=False)]]
    truth_coefficients: dict[str, dict[str, Annotated[float, Field(allow_inf_nan=False)]]]
    action_observations: dict[Identifier, ActiveDesignObservationV26]
    probe_initial_states: list[list[Annotated[float, Field(allow_inf_nan=False)]]]
    probe_clean_values: list[list[list[Annotated[float, Field(allow_inf_nan=False)]]]]
    case_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_private_case(self) -> "PrivateActiveDesignCaseV26":
        self.public_case.assert_sealed()
        catalog = {action.action_id: action for action in self.public_case.action_catalog}
        if set(self.action_observations) != set(catalog):
            raise ValueError("V2.6 private observations do not cover the action catalog")
        for action_id, observation in self.action_observations.items():
            observation.assert_sealed()
            if observation.action_hash != catalog[action_id].action_hash:
                raise ValueError("V2.6 observation is bound to another action")
        if len(self.probe_initial_states) != len(self.probe_clean_values):
            raise ValueError("V2.6 probe initial states and trajectories differ")
        if self.case_hash and self.case_hash != self.content_hash():
            raise ValueError("case_hash does not match V2.6 private case")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "case_hash")

    def assert_sealed(self) -> None:
        if not self.case_hash or self.case_hash != self.content_hash():
            raise ValueError("V2.6 private case is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PrivateActiveDesignCaseV26":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"case_hash"}),
            case_hash=draft.content_hash(),
        )


class PrivateActiveDesignWorldPackV26(StrictModel):
    schema_version: Literal["2.6-private"] = "2.6-private"
    spec_hash: Sha256
    cases: list[PrivateActiveDesignCaseV26] = Field(min_length=8)
    generated_at: datetime
    pack_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_pack(self) -> "PrivateActiveDesignWorldPackV26":
        _assert_timezone(self.generated_at, "generated_at")
        if len({case.public_case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("V2.6 private case ids must be unique")
        for case in self.cases:
            case.assert_sealed()
        if self.pack_hash and self.pack_hash != self.content_hash():
            raise ValueError("pack_hash does not match V2.6 private pack")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "pack_hash")

    def assert_sealed(self) -> None:
        if not self.pack_hash or self.pack_hash != self.content_hash():
            raise ValueError("V2.6 private pack is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PrivateActiveDesignWorldPackV26":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"pack_hash"}),
            pack_hash=draft.content_hash(),
        )


class ActiveDesignModelV26(StrictModel):
    schema_version: Literal["2.6"] = "2.6"
    model_id: Identifier
    state_names: list[Identifier] = Field(min_length=1, max_length=2)
    basis_terms: list[PolynomialBasisTermV24] = Field(min_length=3, max_length=10)
    coefficient_matrix: list[list[Annotated[float, Field(allow_inf_nan=False)]]]
    source_observation_hashes: list[Sha256] = Field(min_length=3)
    normalized_design_rank: Annotated[int, Field(ge=0, le=10)]
    normalized_condition_number: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    empirical_trajectory_identifiable: bool
    structural_identifiability_proven: Literal[False] = False
    model_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_model(self) -> "ActiveDesignModelV26":
        if len(self.coefficient_matrix) != len(self.state_names):
            raise ValueError("V2.6 coefficient rows disagree with states")
        if any(len(row) != len(self.basis_terms) for row in self.coefficient_matrix):
            raise ValueError("V2.6 coefficient columns disagree with basis")
        if self.model_hash and self.model_hash != self.content_hash():
            raise ValueError("model_hash does not match V2.6 model")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "model_hash")

    def assert_sealed(self) -> None:
        if not self.model_hash or self.model_hash != self.content_hash():
            raise ValueError("V2.6 active-design model is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ActiveDesignModelV26":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"model_hash"}),
            model_hash=draft.content_hash(),
        )


class ActiveDesignStepReceiptV26(StrictModel):
    step_index: Annotated[int, Field(ge=1, le=12)]
    available_action_count: Annotated[int, Field(ge=1)]
    acquisition_scores: dict[Identifier, Annotated[float, Field(ge=0, allow_inf_nan=False)]]
    selected_action_id: Identifier
    selected_action_hash: Sha256
    observation_hash: Sha256


class ActiveDesignCaseReceiptV26(StrictModel):
    schema_version: Literal["2.6"] = "2.6"
    receipt_id: Identifier
    case_id: Identifier
    public_case_hash: Sha256
    policy_hash: Sha256
    arm: DesignArmV26
    pilot_observation_hash: Sha256
    steps: list[ActiveDesignStepReceiptV26] = Field(min_length=2, max_length=12)
    selected_action_ids: list[Identifier] = Field(min_length=2, max_length=12)
    final_model: ActiveDesignModelV26
    action_budget_consumed: Annotated[int, Field(ge=2, le=12)]
    invalid_action_count: Literal[0] = 0
    selected_at: datetime
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "ActiveDesignCaseReceiptV26":
        _assert_timezone(self.selected_at, "selected_at")
        self.final_model.assert_sealed()
        if self.action_budget_consumed != len(self.steps):
            raise ValueError("V2.6 budget does not match step receipts")
        if self.selected_action_ids != [step.selected_action_id for step in self.steps]:
            raise ValueError("V2.6 selected action order disagrees with steps")
        if len(set(self.selected_action_ids)) != len(self.selected_action_ids):
            raise ValueError("V2.6 action selection must be without replacement")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match V2.6 case receipt")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("V2.6 case receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ActiveDesignCaseReceiptV26":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


class ActiveDesignSelectionBundleV26(StrictModel):
    schema_version: Literal["2.6"] = "2.6"
    spec_hash: Sha256
    private_pack_hash: Sha256
    policy_hash: Sha256
    arm: DesignArmV26
    case_receipts: list[ActiveDesignCaseReceiptV26] = Field(min_length=8)
    total_action_budget: Annotated[int, Field(ge=1)]
    invalid_action_count: Literal[0] = 0
    bundle_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bundle(self) -> "ActiveDesignSelectionBundleV26":
        for receipt in self.case_receipts:
            receipt.assert_sealed()
            if receipt.policy_hash != self.policy_hash or receipt.arm != self.arm:
                raise ValueError("V2.6 receipt is bound to another policy")
        if self.total_action_budget != sum(
            receipt.action_budget_consumed for receipt in self.case_receipts
        ):
            raise ValueError("V2.6 total budget disagrees with receipts")
        if self.bundle_hash and self.bundle_hash != self.content_hash():
            raise ValueError("bundle_hash does not match V2.6 selection bundle")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "bundle_hash")

    def assert_sealed(self) -> None:
        if not self.bundle_hash or self.bundle_hash != self.content_hash():
            raise ValueError("V2.6 selection bundle is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ActiveDesignSelectionBundleV26":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"bundle_hash"}),
            bundle_hash=draft.content_hash(),
        )


class ActiveDesignCaseResultV26(StrictModel):
    case_id: Identifier
    mechanism: Mechanism
    baseline_parameter_error: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    active_parameter_error: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    baseline_support_f1: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    active_support_f1: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    baseline_probe_nrmse: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    active_probe_nrmse: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    baseline_joint_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    active_joint_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    relative_improvement: Annotated[float, Field(allow_inf_nan=False)]
    baseline_condition_number: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    active_condition_number: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    log_condition_improvement: Annotated[float, Field(allow_inf_nan=False)]
    material_negative_transfer: bool


class ActiveDesignMechanismResultV26(StrictModel):
    mechanism: Mechanism
    case_count: Annotated[int, Field(ge=2)]
    mean_relative_improvement: Annotated[float, Field(allow_inf_nan=False)]
    mean_log_condition_improvement: Annotated[float, Field(allow_inf_nan=False)]


class ActiveDesignReportV26(StrictModel):
    schema_version: Literal["2.6"] = "2.6"
    spec_hash: Sha256
    private_pack_hash: Sha256
    baseline_bundle_hash: Sha256
    active_bundle_hash: Sha256
    cases: list[ActiveDesignCaseResultV26] = Field(min_length=8)
    mechanisms: list[ActiveDesignMechanismResultV26] = Field(min_length=4, max_length=4)
    same_action_and_fit_budget: bool
    invalid_action_count: Annotated[int, Field(ge=0)]
    material_negative_transfer_count: Annotated[int, Field(ge=0)]
    macro_relative_improvement: Annotated[float, Field(allow_inf_nan=False)]
    macro_relative_improvement_ci_lower: Annotated[float, Field(allow_inf_nan=False)]
    macro_relative_improvement_ci_upper: Annotated[float, Field(allow_inf_nan=False)]
    negative_transfer_rate_upper: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    macro_log_condition_improvement: Annotated[float, Field(allow_inf_nan=False)]
    status: Literal[
        "exploratory_only",
        "candidate_rejected_active_design_v26",
        "promoted_for_synthetic_active_design_worldpack_v26",
    ]
    reason_codes: list[str]
    evaluated_at: datetime
    report_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "ActiveDesignReportV26":
        _assert_timezone(self.evaluated_at, "evaluated_at")
        if self.report_hash and self.report_hash != self.content_hash():
            raise ValueError("report_hash does not match V2.6 active-design report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "report_hash")

    def assert_sealed(self) -> None:
        if not self.report_hash or self.report_hash != self.content_hash():
            raise ValueError("V2.6 active-design report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ActiveDesignReportV26":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"report_hash"}),
            report_hash=draft.content_hash(),
        )


class ActiveDesignQualificationV26(StrictModel):
    schema_version: Literal["2.6"] = "2.6"
    qualification_id: Identifier
    qualification_scope: Literal["synthetic_safe_initial_condition_design_v26"] = (
        "synthetic_safe_initial_condition_design_v26"
    )
    policy_hash: Sha256
    report_hash: Sha256
    real_world_validity_established: Literal[False] = False
    structural_identifiability_proven: Literal[False] = False
    qualified_at: datetime
    qualification_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_qualification(self) -> "ActiveDesignQualificationV26":
        _assert_timezone(self.qualified_at, "qualified_at")
        if self.qualification_hash and self.qualification_hash != self.content_hash():
            raise ValueError("qualification_hash does not match V2.6 qualification")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "qualification_hash")

    def assert_sealed(self) -> None:
        if not self.qualification_hash or self.qualification_hash != self.content_hash():
            raise ValueError("V2.6 qualification is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ActiveDesignQualificationV26":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"qualification_hash"}),
            qualification_hash=draft.content_hash(),
        )


class ActiveDesignManifestV26(StrictModel):
    schema_version: Literal["2.6"] = "2.6"
    run_id: Identifier
    artifact_refs: list[ArtifactRef] = Field(min_length=7)
    terminal_status: Literal[
        "exploratory_only",
        "candidate_rejected_active_design_v26",
        "promoted_for_synthetic_active_design_worldpack_v26",
    ]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "ActiveDesignManifestV26":
        _assert_timezone(self.created_at, "created_at")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match V2.6 manifest")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("V2.6 manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ActiveDesignManifestV26":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class ActiveDesignOutcomeV26:
    store: RunStore
    spec: ActiveDesignWorldPackSpecV26
    private_pack: PrivateActiveDesignWorldPackV26
    baseline_policy: ActiveDesignPolicyV26
    active_policy: ActiveDesignPolicyV26
    baseline: ActiveDesignSelectionBundleV26
    active: ActiveDesignSelectionBundleV26
    report: ActiveDesignReportV26
    qualification: ActiveDesignQualificationV26 | None
    manifest: ActiveDesignManifestV26


def default_active_design_policies_v26(
    *,
    knowledge_bundle_hash: str,
    prior_failure_report_hash: str,
) -> tuple[ActiveDesignPolicyV26, ActiveDesignPolicyV26]:
    baseline = ActiveDesignPolicyV26.seal(
        policy_id="random_safe_catalog_v26",
        arm="random_safe_catalog",
        knowledge_bundle_hash=knowledge_bundle_hash,
        prior_failure_report_hash=prior_failure_report_hash,
        acquisition_rule="prefrozen_random_without_replacement",
    )
    active = ActiveDesignPolicyV26.seal(
        policy_id="ensemble_disagreement_catalog_v26",
        arm="ensemble_disagreement_catalog",
        knowledge_bundle_hash=knowledge_bundle_hash,
        prior_failure_report_hash=prior_failure_report_hash,
        acquisition_rule="bootstrap_model_derivative_disagreement",
    )
    return baseline, active


def default_active_design_exploratory_spec_v26(
    *,
    knowledge_bundle_hash: str,
    prior_failure_report_hash: str,
    method_evidence_hash: str,
    baseline_policy_hash: str,
    active_policy_hash: str,
    frozen_at: datetime | None = None,
) -> ActiveDesignWorldPackSpecV26:
    return ActiveDesignWorldPackSpecV26.seal(
        experiment_id="active_design_exploratory_v26",
        phase="exploratory",
        mechanisms=[
            "exponential_decay",
            "logistic_growth",
            "damped_oscillator",
            "lotka_volterra",
        ],
        seeds=list(EXPLORATORY_ACTIVE_DESIGN_SEEDS_V26),
        baseline_policy_hash=baseline_policy_hash,
        active_policy_hash=active_policy_hash,
        knowledge_bundle_hash=knowledge_bundle_hash,
        prior_failure_report_hash=prior_failure_report_hash,
        method_evidence_hash=method_evidence_hash,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


def default_active_design_confirmation_spec_v26(
    *,
    knowledge_bundle_hash: str,
    prior_failure_report_hash: str,
    method_evidence_hash: str,
    baseline_policy_hash: str,
    active_policy_hash: str,
    frozen_at: datetime | None = None,
) -> ActiveDesignWorldPackSpecV26:
    return ActiveDesignWorldPackSpecV26.seal(
        experiment_id="active_design_confirmation_v26",
        phase="confirmation",
        mechanisms=[
            "exponential_decay",
            "logistic_growth",
            "damped_oscillator",
            "lotka_volterra",
        ],
        seeds=list(CONFIRMATION_ACTIVE_DESIGN_SEEDS_V26),
        baseline_policy_hash=baseline_policy_hash,
        active_policy_hash=active_policy_hash,
        knowledge_bundle_hash=knowledge_bundle_hash,
        prior_failure_report_hash=prior_failure_report_hash,
        method_evidence_hash=method_evidence_hash,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


def failure_evolved_active_design_exploratory_spec_v26(
    *,
    knowledge_bundle_hash: str,
    prior_failure_report_hash: str,
    method_evidence_hash: str,
    baseline_policy_hash: str,
    active_policy_hash: str,
    frozen_at: datetime | None = None,
) -> ActiveDesignWorldPackSpecV26:
    return ActiveDesignWorldPackSpecV26.seal(
        experiment_id="active_design_effect_evolved_exploratory_v26",
        phase="exploratory",
        mechanisms=[
            "exponential_decay",
            "logistic_growth",
            "damped_oscillator",
            "lotka_volterra",
        ],
        seeds=list(EVOLVED_ACTIVE_DESIGN_SEEDS_V26),
        baseline_policy_hash=baseline_policy_hash,
        active_policy_hash=active_policy_hash,
        knowledge_bundle_hash=knowledge_bundle_hash,
        prior_failure_report_hash=prior_failure_report_hash,
        method_evidence_hash=method_evidence_hash,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


def assert_single_component_active_design_v26(
    baseline: ActiveDesignPolicyV26,
    active: ActiveDesignPolicyV26,
) -> None:
    baseline.assert_sealed()
    active.assert_sealed()
    if baseline.arm != "random_safe_catalog" or active.arm != "ensemble_disagreement_catalog":
        raise ValueError("V2.6 requires the frozen baseline and active arms")
    for field in ("knowledge_bundle_hash", "prior_failure_report_hash"):
        if getattr(baseline, field) != getattr(active, field):
            raise ValueError("V2.6 policies differ outside action selection")


def generate_private_active_design_worldpack_v26(
    spec: ActiveDesignWorldPackSpecV26,
    *,
    generated_at: datetime | None = None,
) -> PrivateActiveDesignWorldPackV26:
    spec.assert_sealed()
    cases: list[PrivateActiveDesignCaseV26] = []
    for seed in spec.seeds:
        for mechanism in spec.mechanisms:
            cases.append(_generate_private_case_v26(spec, mechanism, seed))
    return PrivateActiveDesignWorldPackV26.seal(
        spec_hash=spec.spec_hash,
        cases=cases,
        generated_at=generated_at or datetime.now(timezone.utc),
    )


def execute_active_design_policy_v26(
    spec: ActiveDesignWorldPackSpecV26,
    private_pack: PrivateActiveDesignWorldPackV26,
    policy: ActiveDesignPolicyV26,
    *,
    selected_at: datetime | None = None,
) -> ActiveDesignSelectionBundleV26:
    spec.assert_sealed()
    private_pack.assert_sealed()
    policy.assert_sealed()
    if private_pack.spec_hash != spec.spec_hash:
        raise ValueError("V2.6 private pack is bound to another spec")
    expected_hash = (
        spec.baseline_policy_hash
        if policy.arm == "random_safe_catalog"
        else spec.active_policy_hash
    )
    if policy.policy_hash != expected_hash:
        raise ValueError("V2.6 policy is not frozen in the spec")
    at = selected_at or datetime.now(timezone.utc)
    receipts = [
        _execute_case_policy_v26(spec, case, policy, selected_at=at)
        for case in private_pack.cases
    ]
    return ActiveDesignSelectionBundleV26.seal(
        spec_hash=spec.spec_hash,
        private_pack_hash=private_pack.pack_hash,
        policy_hash=policy.policy_hash,
        arm=policy.arm,
        case_receipts=receipts,
        total_action_budget=sum(item.action_budget_consumed for item in receipts),
        invalid_action_count=0,
    )


def evaluate_active_design_worldpack_v26(
    spec: ActiveDesignWorldPackSpecV26,
    private_pack: PrivateActiveDesignWorldPackV26,
    baseline: ActiveDesignSelectionBundleV26,
    active: ActiveDesignSelectionBundleV26,
    *,
    evaluated_at: datetime | None = None,
) -> ActiveDesignReportV26:
    spec.assert_sealed()
    private_pack.assert_sealed()
    baseline.assert_sealed()
    active.assert_sealed()
    if baseline.private_pack_hash != private_pack.pack_hash:
        raise ValueError("V2.6 baseline is bound to another private pack")
    if active.private_pack_hash != private_pack.pack_hash:
        raise ValueError("V2.6 active bundle is bound to another private pack")
    baseline_by_case = {item.case_id: item for item in baseline.case_receipts}
    active_by_case = {item.case_id: item for item in active.case_receipts}
    results: list[ActiveDesignCaseResultV26] = []
    grouped_improvement: dict[str, list[float]] = {
        mechanism: [] for mechanism in spec.mechanisms
    }
    grouped_condition: dict[str, list[float]] = {
        mechanism: [] for mechanism in spec.mechanisms
    }
    for case in private_pack.cases:
        case_id = case.public_case.case_id
        base_receipt = baseline_by_case[case_id]
        active_receipt = active_by_case[case_id]
        base_metrics = _hidden_model_metrics_v26(case, base_receipt.final_model, spec)
        active_metrics = _hidden_model_metrics_v26(case, active_receipt.final_model, spec)
        base_loss = _joint_loss(*base_metrics)
        active_loss = _joint_loss(*active_metrics)
        improvement = (base_loss - active_loss) / max(base_loss, 1e-9)
        condition_improvement = math.log1p(
            base_receipt.final_model.normalized_condition_number
        ) - math.log1p(active_receipt.final_model.normalized_condition_number)
        negative = active_loss > base_loss * 1.02
        result = ActiveDesignCaseResultV26(
            case_id=case_id,
            mechanism=case.public_case.mechanism,
            baseline_parameter_error=base_metrics[0],
            active_parameter_error=active_metrics[0],
            baseline_support_f1=base_metrics[1],
            active_support_f1=active_metrics[1],
            baseline_probe_nrmse=base_metrics[2],
            active_probe_nrmse=active_metrics[2],
            baseline_joint_loss=base_loss,
            active_joint_loss=active_loss,
            relative_improvement=improvement,
            baseline_condition_number=(
                base_receipt.final_model.normalized_condition_number
            ),
            active_condition_number=(
                active_receipt.final_model.normalized_condition_number
            ),
            log_condition_improvement=condition_improvement,
            material_negative_transfer=negative,
        )
        results.append(result)
        grouped_improvement[result.mechanism].append(improvement)
        grouped_condition[result.mechanism].append(condition_improvement)
    mechanism_results = [
        ActiveDesignMechanismResultV26(
            mechanism=mechanism,
            case_count=len(grouped_improvement[mechanism]),
            mean_relative_improvement=float(np.mean(grouped_improvement[mechanism])),
            mean_log_condition_improvement=float(np.mean(grouped_condition[mechanism])),
        )
        for mechanism in spec.mechanisms
    ]
    macro = float(np.mean([item.mean_relative_improvement for item in mechanism_results]))
    macro_condition = float(
        np.mean([item.mean_log_condition_improvement for item in mechanism_results])
    )
    draws = _stratified_macro_bootstrap(grouped_improvement, spec)
    alpha = (1.0 - spec.confidence_level) / 2.0
    ci_lower, ci_upper = np.quantile(draws, [alpha, 1.0 - alpha], method="linear")
    negative_count = sum(item.material_negative_transfer for item in results)
    negative_upper = _clopper_pearson_upper(
        negative_count, len(results), spec.confidence_level
    )
    same_budget = (
        baseline.total_action_budget == active.total_action_budget
        and all(
            baseline_by_case[item.case_id].action_budget_consumed
            == active_by_case[item.case_id].action_budget_consumed
            == spec.action_budget
            for item in results
        )
    )
    invalid_count = baseline.invalid_action_count + active.invalid_action_count
    reasons: list[str] = []
    if spec.phase == "exploratory":
        status = "exploratory_only"
        reasons.append("exploratory_not_eligible")
    else:
        if not same_budget:
            reasons.append("unequal_action_or_fit_budget")
        if invalid_count:
            reasons.append("invalid_or_unsafe_action")
        if float(ci_lower) <= spec.minimum_macro_relative_improvement:
            reasons.append("macro_improvement_gate_failed")
        if negative_upper > spec.maximum_negative_transfer_rate:
            reasons.append("negative_transfer_gate_failed")
        if any(
            item.mean_relative_improvement < -spec.maximum_mechanism_regression
            for item in mechanism_results
        ):
            reasons.append("mechanism_noninferiority_gate_failed")
        if macro_condition <= 0:
            reasons.append("empirical_information_gate_failed")
        status = (
            "candidate_rejected_active_design_v26"
            if reasons
            else "promoted_for_synthetic_active_design_worldpack_v26"
        )
    return ActiveDesignReportV26.seal(
        spec_hash=spec.spec_hash,
        private_pack_hash=private_pack.pack_hash,
        baseline_bundle_hash=baseline.bundle_hash,
        active_bundle_hash=active.bundle_hash,
        cases=results,
        mechanisms=mechanism_results,
        same_action_and_fit_budget=same_budget,
        invalid_action_count=invalid_count,
        material_negative_transfer_count=negative_count,
        macro_relative_improvement=macro,
        macro_relative_improvement_ci_lower=float(ci_lower),
        macro_relative_improvement_ci_upper=float(ci_upper),
        negative_transfer_rate_upper=negative_upper,
        macro_log_condition_improvement=macro_condition,
        status=status,
        reason_codes=reasons,
        evaluated_at=evaluated_at or datetime.now(timezone.utc),
    )


def qualify_active_design_policy_v26(
    active_policy: ActiveDesignPolicyV26,
    report: ActiveDesignReportV26,
    *,
    qualified_at: datetime | None = None,
) -> ActiveDesignQualificationV26:
    active_policy.assert_sealed()
    report.assert_sealed()
    if report.status != "promoted_for_synthetic_active_design_worldpack_v26":
        raise ValueError("cannot qualify a rejected V2.6 active-design policy")
    return ActiveDesignQualificationV26.seal(
        qualification_id="synthetic_safe_initial_condition_design_v26",
        policy_hash=active_policy.policy_hash,
        report_hash=report.report_hash,
        qualified_at=qualified_at or datetime.now(timezone.utc),
    )


def run_active_design_worldpack_v26(
    output_root: str | Path,
    *,
    spec: ActiveDesignWorldPackSpecV26,
    baseline_policy: ActiveDesignPolicyV26,
    active_policy: ActiveDesignPolicyV26,
    evaluated_at: datetime | None = None,
    run_id: str | None = None,
) -> ActiveDesignOutcomeV26:
    spec.assert_sealed()
    assert_single_component_active_design_v26(baseline_policy, active_policy)
    if baseline_policy.policy_hash != spec.baseline_policy_hash:
        raise ValueError("V2.6 baseline policy is not frozen in the spec")
    if active_policy.policy_hash != spec.active_policy_hash:
        raise ValueError("V2.6 active policy is not frozen in the spec")
    at = evaluated_at or datetime.now(timezone.utc)
    store = RunStore(
        output_root, run_id=run_id or f"dynamics-active-design-{uuid4().hex[:10]}"
    )
    refs = [
        store.put_artifact("active_design_spec_v26", spec),
        store.put_artifact("active_design_baseline_policy_v26", baseline_policy),
        store.put_artifact("active_design_candidate_policy_v26", active_policy),
    ]
    store.emit(
        "active_design_protocol_frozen_before_private_pack",
        {
            "spec_hash": spec.spec_hash,
            "baseline_policy_hash": baseline_policy.policy_hash,
            "active_policy_hash": active_policy.policy_hash,
            "frozen_delta": spec.frozen_delta,
        },
    )
    private_pack = generate_private_active_design_worldpack_v26(spec, generated_at=at)
    baseline = execute_active_design_policy_v26(
        spec, private_pack, baseline_policy, selected_at=at
    )
    active = execute_active_design_policy_v26(
        spec, private_pack, active_policy, selected_at=at
    )
    report = evaluate_active_design_worldpack_v26(
        spec, private_pack, baseline, active, evaluated_at=at
    )
    refs.extend(
        [
            store.put_artifact("private_active_design_worldpack_v26", private_pack),
            store.put_artifact("active_design_baseline_bundle_v26", baseline),
            store.put_artifact("active_design_candidate_bundle_v26", active),
            store.put_artifact("active_design_report_v26", report),
        ]
    )
    qualification = None
    if report.status == "promoted_for_synthetic_active_design_worldpack_v26":
        qualification = qualify_active_design_policy_v26(
            active_policy, report, qualified_at=at
        )
        refs.append(store.put_artifact("active_design_qualification_v26", qualification))
    manifest = ActiveDesignManifestV26.seal(
        run_id=store.run_id,
        artifact_refs=refs,
        terminal_status=report.status,
        created_at=at,
    )
    manifest_ref = store.put_artifact("active_design_manifest_v26", manifest)
    store.emit(
        "active_design_worldpack_adjudicated",
        {"manifest_ref": manifest_ref.model_dump(mode="json")},
    )
    if not verify_active_design_worldpack_run_v26(store.run_directory):
        raise RuntimeError("V2.6 active-design run failed independent verification")
    return ActiveDesignOutcomeV26(
        store,
        spec,
        private_pack,
        baseline_policy,
        active_policy,
        baseline,
        active,
        report,
        qualification,
        manifest,
    )


def verify_active_design_worldpack_run_v26(run_directory: str | Path) -> bool:
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
            ref for ref in committed if ref.kind == "active_design_manifest_v26"
        ]
        if len(manifest_refs) != 1:
            return False
        manifest = ActiveDesignManifestV26.model_validate(
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

        spec = load_one("active_design_spec_v26", ActiveDesignWorldPackSpecV26)
        baseline_policy = load_one(
            "active_design_baseline_policy_v26", ActiveDesignPolicyV26
        )
        active_policy = load_one(
            "active_design_candidate_policy_v26", ActiveDesignPolicyV26
        )
        private_pack = load_one(
            "private_active_design_worldpack_v26", PrivateActiveDesignWorldPackV26
        )
        baseline = load_one(
            "active_design_baseline_bundle_v26", ActiveDesignSelectionBundleV26
        )
        active = load_one(
            "active_design_candidate_bundle_v26", ActiveDesignSelectionBundleV26
        )
        report = load_one("active_design_report_v26", ActiveDesignReportV26)
        for item in (
            spec,
            baseline_policy,
            active_policy,
            private_pack,
            baseline,
            active,
            report,
        ):
            item.assert_sealed()
        regenerated = generate_private_active_design_worldpack_v26(
            spec, generated_at=private_pack.generated_at
        )
        if regenerated.pack_hash != private_pack.pack_hash:
            return False
        selected_at = baseline.case_receipts[0].selected_at
        replay_baseline = execute_active_design_policy_v26(
            spec, private_pack, baseline_policy, selected_at=selected_at
        )
        replay_active = execute_active_design_policy_v26(
            spec, private_pack, active_policy, selected_at=selected_at
        )
        if replay_baseline.bundle_hash != baseline.bundle_hash:
            return False
        if replay_active.bundle_hash != active.bundle_hash:
            return False
        recomputed = evaluate_active_design_worldpack_v26(
            spec,
            private_pack,
            baseline,
            active,
            evaluated_at=report.evaluated_at,
        )
        if recomputed.report_hash != report.report_hash:
            return False
        qualifications = [
            ref
            for ref in manifest.artifact_refs
            if ref.kind == "active_design_qualification_v26"
        ]
        if report.status == "promoted_for_synthetic_active_design_worldpack_v26":
            if len(qualifications) != 1:
                return False
            qualification = ActiveDesignQualificationV26.model_validate(
                store.load_artifact(qualifications[0])
            )
            qualification.assert_sealed()
            if qualification.report_hash != report.report_hash:
                return False
        elif qualifications:
            return False
        freeze_events = [
            event
            for event in events
            if event["event_type"]
            == "active_design_protocol_frozen_before_private_pack"
        ]
        return len(freeze_events) == 1 and store.verify_event_chain()
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


def _assert_action_safe(
    public_case: ActiveDesignPublicCaseV26,
    action: SafeInitialConditionActionV26,
) -> None:
    if any(
        value < low or value > high
        for value, low, high in zip(
            action.initial_state,
            public_case.state_lower_bounds,
            public_case.state_upper_bounds,
            strict=True,
        )
    ):
        raise ValueError("V2.6 action violates the public safety envelope")


def _generate_private_case_v26(
    spec: ActiveDesignWorldPackSpecV26,
    mechanism: Mechanism,
    seed: int,
) -> PrivateActiveDesignCaseV26:
    mechanism_index = spec.mechanisms.index(mechanism)
    random = Random(seed * 104_729 + mechanism_index * 7_919)
    state_names, parameters, truth, bounds, pilot_initial = _mechanism_definition_v26(
        mechanism, random
    )
    actions = _latin_hypercube_actions_v26(
        bounds,
        spec.candidate_action_count,
        Random(seed * 65_537 + mechanism_index * 257),
    )
    times = [index * spec.time_step for index in range(spec.trajectory_points)]
    observations: dict[str, ActiveDesignObservationV26] = {}
    safe_actions: list[SafeInitialConditionActionV26] = []
    for index, initial in enumerate(actions):
        action = SafeInitialConditionActionV26.seal(
            action_id=f"{mechanism}_{seed}_action_{index:02d}",
            initial_state=initial,
        )
        clean = _simulate_truth_v26(mechanism, initial, times, parameters)
        noisy = _noisy_values_v26(
            clean,
            spec.observation_noise_fraction,
            seed * 1_000_003 + mechanism_index * 10_007 + index,
            mechanism,
        )
        observation = ActiveDesignObservationV26.seal(
            action_id=action.action_id,
            action_hash=action.action_hash,
            times=times,
            values=noisy,
        )
        safe_actions.append(action)
        observations[action.action_id] = observation
    pilot_clean = _simulate_truth_v26(mechanism, pilot_initial, times, parameters)
    pilot_action = SafeInitialConditionActionV26.seal(
        action_id=f"{mechanism}_{seed}_pilot",
        initial_state=pilot_initial,
    )
    pilot = ActiveDesignObservationV26.seal(
        action_id=pilot_action.action_id,
        action_hash=pilot_action.action_hash,
        times=times,
        values=_noisy_values_v26(
            pilot_clean,
            spec.observation_noise_fraction,
            seed * 2_000_003 + mechanism_index * 20_011,
            mechanism,
        ),
    )
    public = ActiveDesignPublicCaseV26.seal(
        case_id=f"active_{mechanism}_{seed}",
        mechanism=mechanism,
        state_names=state_names,
        state_lower_bounds=[low for low, _ in bounds],
        state_upper_bounds=[high for _, high in bounds],
        action_catalog=safe_actions,
        pilot_observation=pilot,
    )
    probe_initials = _latin_hypercube_actions_v26(
        bounds,
        spec.probe_trajectory_count,
        Random(seed * 131_071 + mechanism_index * 509 + 17),
    )
    probes = [
        _simulate_truth_v26(mechanism, initial, times, parameters)
        for initial in probe_initials
    ]
    return PrivateActiveDesignCaseV26.seal(
        public_case=public,
        hidden_parameters=parameters,
        truth_coefficients=truth,
        action_observations=observations,
        probe_initial_states=probe_initials,
        probe_clean_values=probes,
    )


def _mechanism_definition_v26(
    mechanism: Mechanism,
    random: Random,
) -> tuple[
    list[str],
    dict[str, float],
    dict[str, dict[str, float]],
    list[tuple[float, float]],
    list[float],
]:
    if mechanism == "exponential_decay":
        rate = 0.35 + 0.25 * random.random()
        return (
            ["x"],
            {"rate": rate},
            {"x": {"x": -rate}},
            [(0.2, 3.0)],
            [1.5],
        )
    if mechanism == "logistic_growth":
        rate = 0.55 + 0.35 * random.random()
        capacity = 7.0 + 4.0 * random.random()
        return (
            ["x"],
            {"rate": rate, "capacity": capacity},
            {"x": {"x": rate, "x2": -rate / capacity}},
            [(0.25, 12.0)],
            [0.75],
        )
    if mechanism == "damped_oscillator":
        omega = 0.75 + 0.5 * random.random()
        damping = 0.08 + 0.25 * random.random()
        return (
            ["position", "velocity"],
            {"omega": omega, "damping": damping},
            {
                "position": {"velocity": 1.0},
                "velocity": {"position": -(omega**2), "velocity": -damping},
            },
            [(-2.0, 2.0), (-2.0, 2.0)],
            [1.0, 0.0],
        )
    alpha = 0.7 + 0.4 * random.random()
    beta_value = 0.06 + 0.04 * random.random()
    delta = 0.04 + 0.03 * random.random()
    gamma = 0.8 + 0.4 * random.random()
    return (
        ["prey", "predator"],
        {
            "alpha": alpha,
            "beta": beta_value,
            "delta": delta,
            "gamma": gamma,
        },
        {
            "prey": {"prey": alpha, "prey_predator": -beta_value},
            "predator": {"prey_predator": delta, "predator": -gamma},
        },
        [(2.0, 16.0), (2.0, 12.0)],
        [8.0, 5.0],
    )


def _latin_hypercube_actions_v26(
    bounds: list[tuple[float, float]],
    count: int,
    random: Random,
) -> list[list[float]]:
    dimensions: list[list[float]] = []
    for low, high in bounds:
        bins = list(range(count))
        random.shuffle(bins)
        dimensions.append(
            [low + (bin_index + random.random()) / count * (high - low) for bin_index in bins]
        )
    return [
        [dimensions[dimension][index] for dimension in range(len(bounds))]
        for index in range(count)
    ]


def _simulate_truth_v26(
    mechanism: Mechanism,
    initial_state: list[float],
    times: list[float],
    parameters: dict[str, float],
) -> list[list[float]]:
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
                parameters["alpha"] * state[0]
                - parameters["beta"] * state[0] * state[1],
                parameters["delta"] * state[0] * state[1]
                - parameters["gamma"] * state[1],
            ]
        )

    solution = solve_ivp(
        rhs,
        (float(times[0]), float(times[-1])),
        np.asarray(initial_state, dtype=float),
        t_eval=np.asarray(times, dtype=float),
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
    )
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError("V2.6 private truth simulator failed")
    values = solution.y.T
    if np.max(np.abs(values)) > 100.0:
        raise RuntimeError("V2.6 action left the synthetic safety envelope")
    return values.tolist()


def _noisy_values_v26(
    clean_values: list[list[float]],
    noise_fraction: float,
    seed: int,
    mechanism: Mechanism,
) -> list[list[float]]:
    clean = np.asarray(clean_values, dtype=float)
    scales = np.std(clean, axis=0)
    fallback = np.maximum(np.mean(np.abs(clean), axis=0) * 0.1, 1e-3)
    scales = np.maximum(scales, fallback)
    rng = np.random.default_rng(seed)
    noisy = clean + rng.normal(0.0, noise_fraction * scales, size=clean.shape)
    if mechanism in {"exponential_decay", "logistic_growth", "lotka_volterra"}:
        noisy = np.maximum(noisy, 1e-6)
    return noisy.tolist()


def _execute_case_policy_v26(
    spec: ActiveDesignWorldPackSpecV26,
    private_case: PrivateActiveDesignCaseV26,
    policy: ActiveDesignPolicyV26,
    *,
    selected_at: datetime,
) -> ActiveDesignCaseReceiptV26:
    public = private_case.public_case
    public.assert_sealed()
    collected = [public.pilot_observation]
    selected_ids: list[str] = []
    steps: list[ActiveDesignStepReceiptV26] = []
    for step_index in range(1, spec.action_budget + 1):
        available = [
            action for action in public.action_catalog if action.action_id not in selected_ids
        ]
        scores = _choose_action_scores_v26(
            spec,
            public,
            collected,
            available,
            policy,
            step_index,
        )
        selected = max(
            available,
            key=lambda action: (scores[action.action_id], action.action_id),
        )
        _assert_action_safe(public, selected)
        observation = private_case.action_observations[selected.action_id]
        observation.assert_sealed()
        if observation.action_hash != selected.action_hash:
            raise RuntimeError("V2.6 environment returned an observation for another action")
        collected.append(observation)
        selected_ids.append(selected.action_id)
        steps.append(
            ActiveDesignStepReceiptV26(
                step_index=step_index,
                available_action_count=len(available),
                acquisition_scores=scores,
                selected_action_id=selected.action_id,
                selected_action_hash=selected.action_hash,
                observation_hash=observation.observation_hash,
            )
        )
    model = _fit_active_design_model_v26(
        public.case_id,
        public.state_names,
        collected,
        spec,
    )
    return ActiveDesignCaseReceiptV26.seal(
        receipt_id=f"{public.case_id}_{policy.arm}",
        case_id=public.case_id,
        public_case_hash=public.public_hash,
        policy_hash=policy.policy_hash,
        arm=policy.arm,
        pilot_observation_hash=public.pilot_observation.observation_hash,
        steps=steps,
        selected_action_ids=selected_ids,
        final_model=model,
        action_budget_consumed=spec.action_budget,
        invalid_action_count=0,
        selected_at=selected_at,
    )


def _choose_action_scores_v26(
    spec: ActiveDesignWorldPackSpecV26,
    public: ActiveDesignPublicCaseV26,
    collected: list[ActiveDesignObservationV26],
    available: list[SafeInitialConditionActionV26],
    policy: ActiveDesignPolicyV26,
    step_index: int,
) -> dict[str, float]:
    if policy.arm == "random_safe_catalog":
        random = Random(
            int(sha256_value([spec.spec_hash, public.case_id, policy.arm])[:16], 16)
        )
        order = [action.action_id for action in public.action_catalog]
        random.shuffle(order)
        priority = {action_id: float(len(order) - index) for index, action_id in enumerate(order)}
        return {action.action_id: priority[action.action_id] for action in available}
    ensemble = _bootstrap_coefficients_v26(
        public.state_names,
        collected,
        spec,
        seed=int(
            sha256_value([spec.spec_hash, public.case_id, policy.arm, step_index])[:16],
            16,
        ),
    )
    terms = polynomial_basis_terms(public.state_names, spec.polynomial_degree)
    states = np.asarray([action.initial_state for action in available], dtype=float)
    library = evaluate_polynomial_library(states, terms)
    predictions = np.einsum("ct,bet->bce", library, ensemble)
    variance = np.var(predictions, axis=0)
    mean_prediction = np.mean(np.abs(predictions), axis=0)
    scale = np.maximum(np.median(mean_prediction, axis=0), 1e-6)
    raw = np.mean(variance / (scale[np.newaxis, :] ** 2), axis=1)
    return {
        action.action_id: float(max(score, 0.0))
        for action, score in zip(available, raw, strict=True)
    }


def _observation_regression_arrays_v26(
    state_names: list[str],
    observations: list[ActiveDesignObservationV26],
    spec: ActiveDesignWorldPackSpecV26,
) -> tuple[list[PolynomialBasisTermV24], np.ndarray, np.ndarray]:
    terms = polynomial_basis_terms(state_names, spec.polynomial_degree)
    libraries: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    trim = spec.savgol_window // 2
    for observation in observations:
        observation.assert_sealed()
        values = np.asarray(observation.values, dtype=float)
        times = np.asarray(observation.times, dtype=float)
        dt = float(times[1] - times[0])
        derivatives = np.column_stack(
            [
                savgol_filter(
                    values[:, index],
                    window_length=spec.savgol_window,
                    polyorder=spec.savgol_polynomial_order,
                    deriv=1,
                    delta=dt,
                    mode="interp",
                )
                for index in range(values.shape[1])
            ]
        )
        libraries.append(
            evaluate_polynomial_library(values[trim:-trim], terms)
        )
        targets.append(derivatives[trim:-trim])
    return terms, np.vstack(libraries), np.vstack(targets)


def _ridge_coefficients_v26(
    library: np.ndarray,
    targets: np.ndarray,
    ridge_alpha: float,
) -> np.ndarray:
    scales = np.sqrt(np.mean(library**2, axis=0))
    scales[scales < 1e-12] = 1.0
    normalized = library / scales
    coefficients_normalized = np.linalg.solve(
        normalized.T @ normalized + ridge_alpha * np.eye(normalized.shape[1]),
        normalized.T @ targets,
    ).T
    return coefficients_normalized / scales[np.newaxis, :]


def _bootstrap_coefficients_v26(
    state_names: list[str],
    observations: list[ActiveDesignObservationV26],
    spec: ActiveDesignWorldPackSpecV26,
    *,
    seed: int,
) -> np.ndarray:
    _, library, targets = _observation_regression_arrays_v26(
        state_names, observations, spec
    )
    random = np.random.default_rng(seed)
    sample_count = max(library.shape[1] + 2, int(len(library) * spec.bootstrap_fraction))
    ensemble = []
    for _ in range(spec.ensemble_members):
        indices = random.integers(0, len(library), size=sample_count)
        ensemble.append(
            _ridge_coefficients_v26(
                library[indices], targets[indices], spec.ridge_alpha
            )
        )
    return np.asarray(ensemble)


def _fit_active_design_model_v26(
    case_id: str,
    state_names: list[str],
    observations: list[ActiveDesignObservationV26],
    spec: ActiveDesignWorldPackSpecV26,
) -> ActiveDesignModelV26:
    terms, library, targets = _observation_regression_arrays_v26(
        state_names, observations, spec
    )
    coefficients = _ridge_coefficients_v26(library, targets, spec.ridge_alpha)
    scales = np.sqrt(np.mean(library**2, axis=0))
    scales[scales < 1e-12] = 1.0
    normalized = library / scales
    for equation in range(targets.shape[1]):
        active = np.abs(coefficients[equation]) >= spec.sparsity_threshold
        for _ in range(spec.maximum_iterations):
            previous = active.copy()
            coefficients[equation, ~active] = 0.0
            if active.any():
                selected = normalized[:, active]
                solved = np.linalg.solve(
                    selected.T @ selected
                    + spec.ridge_alpha * np.eye(selected.shape[1]),
                    selected.T @ targets[:, equation],
                )
                coefficients[equation, active] = solved / scales[active]
            active = np.abs(coefficients[equation]) >= spec.sparsity_threshold
            if np.array_equal(previous, active):
                break
        coefficients[equation, ~active] = 0.0
    coefficients[np.abs(coefficients) < 1e-12] = 0.0
    rank = int(np.linalg.matrix_rank(normalized, tol=1e-10))
    condition = float(np.linalg.cond(normalized))
    if not math.isfinite(condition):
        condition = 1e15
    return ActiveDesignModelV26.seal(
        model_id=f"{case_id}_active_design_model",
        state_names=state_names,
        basis_terms=terms,
        coefficient_matrix=coefficients.tolist(),
        source_observation_hashes=[item.observation_hash for item in observations],
        normalized_design_rank=rank,
        normalized_condition_number=condition,
        empirical_trajectory_identifiable=(rank == normalized.shape[1]),
    )


def _simulate_active_design_model_v26(
    model: ActiveDesignModelV26,
    initial_state: list[float],
    times: list[float],
) -> list[list[float]]:
    model.assert_sealed()
    coefficients = np.asarray(model.coefficient_matrix, dtype=float)

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        if np.max(np.abs(state)) > 1e5:
            raise FloatingPointError("V2.6 fitted model diverged")
        row = evaluate_polynomial_library(state.reshape(1, -1), model.basis_terms)[0]
        derivative = coefficients @ row
        if not np.isfinite(derivative).all():
            raise FloatingPointError("V2.6 fitted derivative is nonfinite")
        return derivative

    try:
        solution = solve_ivp(
            rhs,
            (float(times[0]), float(times[-1])),
            np.asarray(initial_state, dtype=float),
            t_eval=np.asarray(times, dtype=float),
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )
    except (FloatingPointError, ValueError) as exc:
        raise RuntimeError("V2.6 fitted model simulation failed") from exc
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError("V2.6 fitted model did not cover the probe horizon")
    return solution.y.T.tolist()


def _truth_matrix_v26(
    model: ActiveDesignModelV26,
    truth: dict[str, dict[str, float]],
) -> np.ndarray:
    term_ids = [term.term_id for term in model.basis_terms]
    return np.asarray(
        [
            [truth.get(state, {}).get(term, 0.0) for term in term_ids]
            for state in model.state_names
        ],
        dtype=float,
    )


def _hidden_model_metrics_v26(
    case: PrivateActiveDesignCaseV26,
    model: ActiveDesignModelV26,
    spec: ActiveDesignWorldPackSpecV26,
) -> tuple[float, float, float]:
    model.assert_sealed()
    predicted_coefficients = np.asarray(model.coefficient_matrix, dtype=float)
    truth = _truth_matrix_v26(model, case.truth_coefficients)
    parameter_error = float(
        np.linalg.norm(predicted_coefficients - truth) / max(np.linalg.norm(truth), 1e-12)
    )
    predicted_support = set(zip(*np.where(np.abs(predicted_coefficients) > 1e-6)))
    truth_support = set(zip(*np.where(np.abs(truth) > 1e-6)))
    true_positive = len(predicted_support & truth_support)
    precision = true_positive / len(predicted_support) if predicted_support else 0.0
    recall = true_positive / len(truth_support) if truth_support else 0.0
    support = (
        0.0
        if precision + recall == 0
        else 2.0 * precision * recall / (precision + recall)
    )
    times = [index * spec.time_step for index in range(spec.trajectory_points)]
    probe_errors = []
    for initial, clean in zip(
        case.probe_initial_states, case.probe_clean_values, strict=True
    ):
        try:
            prediction = _simulate_active_design_model_v26(model, initial, times)
            error = trajectory_nrmse(clean, prediction)
        except (RuntimeError, ValueError, FloatingPointError):
            error = 100.0
        probe_errors.append(error)
    return parameter_error, support, float(np.mean(probe_errors))


def _joint_loss(parameter_error: float, support_f1: float, probe_nrmse: float) -> float:
    return float(
        0.4 * math.log1p(parameter_error)
        + 0.4 * math.log1p(probe_nrmse)
        + 0.2 * (1.0 - support_f1)
    )


def _stratified_macro_bootstrap(
    grouped: dict[str, list[float]],
    spec: ActiveDesignWorldPackSpecV26,
) -> np.ndarray:
    random = Random(spec.bootstrap_seed)
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


def _clopper_pearson_upper(successes: int, trials: int, confidence: float) -> float:
    if successes == trials:
        return 1.0
    return float(beta.ppf(confidence, successes + 1, trials - successes))
