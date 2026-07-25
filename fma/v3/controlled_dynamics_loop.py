from __future__ import annotations

import itertools
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
from fma.v2.dynamics_ir import (
    PolynomialBasisTermV24,
    evaluate_polynomial_library,
    polynomial_basis_terms,
    trajectory_nrmse,
)
from fma.v2.schemas import Identifier, Sha256, _assert_timezone

from .experiment_ir import (
    ControlledObservationReceiptV31,
    DecisionTargetV31,
    ExperimentAcquisitionReceiptV31,
    ExperimentConstraintEnvelopeV31,
    ExperimentPermissionDecisionV31,
    KnownActuatorMapV31,
    PiecewiseConstantInputActionV31,
    validate_action_against_envelope_v31,
)


MechanismV31 = Literal[
    "exponential_decay",
    "logistic_growth",
    "damped_oscillator",
    "duffing_oscillator",
]
ArmV31 = Literal[
    "random_bounded_inputs",
    "goal_oriented_epistemic_control",
]
RouteLayerV31 = Literal["problem_layer", "model_layer", "data_layer"]

MECHANISMS_V31: tuple[MechanismV31, ...] = (
    "exponential_decay",
    "logistic_growth",
    "damped_oscillator",
    "duffing_oscillator",
)
EXPLORATORY_SEEDS_V31 = (8101, 8153, 8209, 8263, 8311, 8363, 8419, 8467)
CONFIRMATION_SEEDS_V31 = (
    9001, 9059, 9103, 9151, 9209, 9257, 9301, 9353, 9403, 9451,
    9503, 9551, 9601, 9653, 9701, 9751, 9803, 9851, 9901, 9953,
)


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class ControlledDynamicsPolicyV31(StrictModel):
    schema_version: Literal["3.1"] = "3.1"
    policy_id: Identifier
    arm: ArmV31
    selection_rule: Literal[
        "prefrozen_random_without_replacement",
        "clarify_then_goal_information_risk_utility",
    ]
    may_reformulate_problem: bool
    maximum_actions: Literal[2] = 2
    known_actuator_required: Literal[True] = True
    prior_epistemic_qualification_hash: Sha256
    prior_active_design_qualification_hash: Sha256
    method_evidence_hash: Sha256
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "ControlledDynamicsPolicyV31":
        expected = {
            "random_bounded_inputs": (
                "prefrozen_random_without_replacement", False
            ),
            "goal_oriented_epistemic_control": (
                "clarify_then_goal_information_risk_utility", True
            ),
        }[self.arm]
        if (self.selection_rule, self.may_reformulate_problem) != expected:
            raise ValueError("V3.1 arm and policy behavior disagree")
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("policy_hash does not match V3.1 policy")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("V3.1 policy is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsPolicyV31":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"policy_hash"}),
            policy_hash=draft.content_hash(),
        )


class ControlledDynamicsContractV31(StrictModel):
    schema_version: Literal["3.1"] = "3.1"
    contract_id: Identifier
    case_id: Identifier
    version: Annotated[int, Field(ge=1, le=2)]
    decision_target: DecisionTargetV31
    target_status: Literal["default_unverified", "authoritative"]
    unresolved_fields: list[Literal["decision_target"]] = Field(max_length=1)
    parent_contract_hash: Sha256 | None = None
    triggering_evidence_hash: Sha256 | None = None
    real_world_action_authorized: Literal[False] = False
    frozen_at: datetime
    contract_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "ControlledDynamicsContractV31":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.target_status == "default_unverified":
            if self.unresolved_fields != ["decision_target"] or self.version != 1:
                raise ValueError("underspecified V3.1 contract must expose target gap")
        elif self.unresolved_fields:
            raise ValueError("authoritative V3.1 contract cannot remain unresolved")
        if self.version == 1 and (self.parent_contract_hash or self.triggering_evidence_hash):
            raise ValueError("initial V3.1 contract cannot have reformulation lineage")
        if self.version == 2 and not (self.parent_contract_hash and self.triggering_evidence_hash):
            raise ValueError("reformulated V3.1 contract needs evidence lineage")
        if self.contract_hash and self.contract_hash != self.content_hash():
            raise ValueError("contract_hash does not match V3.1 contract")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "contract_hash")

    def assert_sealed(self) -> None:
        if not self.contract_hash or self.contract_hash != self.content_hash():
            raise ValueError("V3.1 contract is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsContractV31":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"contract_hash"}),
            contract_hash=draft.content_hash(),
        )


class TargetClarificationEvidenceV31(StrictModel):
    schema_version: Literal["3.1"] = "3.1"
    evidence_id: Identifier
    case_id: Identifier
    decision_target: DecisionTargetV31
    source_ref: Annotated[str, Field(min_length=3)]
    observed_at: datetime
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> "TargetClarificationEvidenceV31":
        _assert_timezone(self.observed_at, "observed_at")
        if self.evidence_hash and self.evidence_hash != self.content_hash():
            raise ValueError("evidence_hash does not match target clarification")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evidence_hash")

    def assert_sealed(self) -> None:
        if not self.evidence_hash or self.evidence_hash != self.content_hash():
            raise ValueError("target clarification evidence is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "TargetClarificationEvidenceV31":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evidence_hash"}),
            evidence_hash=draft.content_hash(),
        )


class PilotObservationV31(StrictModel):
    schema_version: Literal["3.1"] = "3.1"
    observation_id: Identifier
    case_id: Identifier
    actuator_hash: Sha256
    times: list[Annotated[float, Field(allow_inf_nan=False)]] = Field(min_length=9)
    states: list[list[Annotated[float, Field(allow_inf_nan=False)]]] = Field(min_length=9)
    inputs: list[list[Annotated[float, Field(allow_inf_nan=False)]]] = Field(min_length=9)
    quality_flags: list[Identifier] = Field(default_factory=list)
    observation_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_pilot(self) -> "PilotObservationV31":
        if len(self.times) != len(self.states) or len(self.times) != len(self.inputs):
            raise ValueError("pilot arrays differ in length")
        if any(right <= left for left, right in zip(self.times, self.times[1:])):
            raise ValueError("pilot times must increase")
        if self.observation_hash and self.observation_hash != self.content_hash():
            raise ValueError("observation_hash does not match V3.1 pilot")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "observation_hash")

    def assert_sealed(self) -> None:
        if not self.observation_hash or self.observation_hash != self.content_hash():
            raise ValueError("V3.1 pilot is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PilotObservationV31":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"observation_hash"}),
            observation_hash=draft.content_hash(),
        )


class ControlledDynamicsPublicCaseV31(StrictModel):
    schema_version: Literal["3.1"] = "3.1"
    case_id: Identifier
    state_names: list[Identifier]
    initial_state: list[Annotated[float, Field(allow_inf_nan=False)]]
    actuator: KnownActuatorMapV31
    envelope: ExperimentConstraintEnvelopeV31
    action_catalog: list[PiecewiseConstantInputActionV31] = Field(min_length=8, max_length=8)
    pilot: PilotObservationV31
    initial_contract: ControlledDynamicsContractV31
    public_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_public(self) -> "ControlledDynamicsPublicCaseV31":
        self.actuator.assert_sealed()
        self.envelope.assert_sealed()
        self.pilot.assert_sealed()
        self.initial_contract.assert_sealed()
        if len(self.state_names) != len(self.initial_state):
            raise ValueError("V3.1 public state dimensions differ")
        if self.state_names != self.actuator.state_names:
            raise ValueError("V3.1 actuator and public state names differ")
        hashes = set()
        for action in self.action_catalog:
            action.assert_sealed()
            if validate_action_against_envelope_v31(action, self.actuator, self.envelope):
                raise ValueError("V3.1 action catalog violates the frozen envelope")
            hashes.add(action.action_hash)
        if len(hashes) != len(self.action_catalog):
            raise ValueError("V3.1 action catalog contains duplicates")
        if self.public_hash and self.public_hash != self.content_hash():
            raise ValueError("public_hash does not match V3.1 case")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "public_hash")

    def assert_sealed(self) -> None:
        if not self.public_hash or self.public_hash != self.content_hash():
            raise ValueError("V3.1 public case is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsPublicCaseV31":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"public_hash"}),
            public_hash=draft.content_hash(),
        )


class ControlledDynamicsWorldPackSpecV31(StrictModel):
    schema_version: Literal["3.1"] = "3.1"
    experiment_id: Identifier
    phase: Literal["exploratory", "confirmation"]
    mechanisms: list[MechanismV31] = Field(min_length=4, max_length=4)
    seeds: list[int] = Field(min_length=8, max_length=20)
    action_budget: Literal[2] = 2
    trajectory_points: Literal[49] = 49
    time_step: Literal[0.04] = 0.04
    segment_count: Literal[6] = 6
    segment_duration: Literal[0.32] = 0.32
    input_amplitude: Literal[0.35] = 0.35
    observation_noise_fraction: Literal[0.01] = 0.01
    polynomial_degree: Literal[2] = 2
    savgol_window: Literal[9] = 9
    savgol_order: Literal[3] = 3
    ridge_alpha: Literal[0.0001] = 0.0001
    sparsity_threshold: Literal[0.02] = 0.02
    ensemble_members: Literal[12] = 12
    bootstrap_fraction: Literal[0.8] = 0.8
    maximum_empirical_prediction_risk: Literal[0.25] = 0.25
    model_mismatch_residual_threshold: Literal[0.24] = 0.24
    bootstrap_replicates: Annotated[int, Field(ge=200, le=5000)] = 1200
    bootstrap_seed: int = 310722
    minimum_macro_loss_improvement: Literal[0.0] = 0.0
    maximum_mechanism_regression: Literal[0.02] = 0.02
    material_negative_transfer: Literal[0.02] = 0.02
    maximum_negative_transfer_rate: Literal[0.1] = 0.1
    required_routing_accuracy: Literal[1.0] = 1.0
    baseline_policy_hash: Sha256
    candidate_policy_hash: Sha256
    method_evidence_hash: Sha256
    prior_epistemic_qualification_hash: Sha256
    prior_active_design_qualification_hash: Sha256
    frozen_delta: Literal[
        "random_inputs_vs_goal_oriented_epistemic_control_only"
    ] = "random_inputs_vs_goal_oriented_epistemic_control_only"
    frozen_at: datetime
    spec_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "ControlledDynamicsWorldPackSpecV31":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.mechanisms != list(MECHANISMS_V31):
            raise ValueError("V3.1 requires the frozen four-mechanism order")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("V3.1 seeds must be unique")
        allowed = (
            set(EXPLORATORY_SEEDS_V31)
            if self.phase == "exploratory"
            else set(CONFIRMATION_SEEDS_V31)
        )
        if set(self.seeds) != allowed:
            raise ValueError("V3.1 seeds do not match the frozen phase")
        if not math.isclose(
            (self.trajectory_points - 1) * self.time_step,
            self.segment_count * self.segment_duration,
            abs_tol=1e-12,
        ):
            raise ValueError("V3.1 input segments do not cover the trajectory")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match V3.1 protocol")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "spec_hash")

    def assert_sealed(self) -> None:
        if not self.spec_hash or self.spec_hash != self.content_hash():
            raise ValueError("V3.1 protocol is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsWorldPackSpecV31":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"spec_hash"}),
            spec_hash=draft.content_hash(),
        )


class PrivateControlledDynamicsCaseV31(StrictModel):
    schema_version: Literal["3.1"] = "3.1"
    public_case: ControlledDynamicsPublicCaseV31
    mechanism: MechanismV31
    hidden_parameters: dict[Identifier, Annotated[float, Field(allow_inf_nan=False)]]
    true_decision_target: DecisionTargetV31
    target_was_underspecified: bool
    expected_issue_routes: list[RouteLayerV31]
    performance_eligible: bool
    action_observations: dict[Identifier, ControlledObservationReceiptV31]
    probe_initial_states: list[list[Annotated[float, Field(allow_inf_nan=False)]]]
    probe_input_sequences: list[list[list[Annotated[float, Field(allow_inf_nan=False)]]]]
    probe_clean_states: list[list[list[Annotated[float, Field(allow_inf_nan=False)]]]]
    private_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_private(self) -> "PrivateControlledDynamicsCaseV31":
        self.public_case.assert_sealed()
        catalog = {action.action_id: action for action in self.public_case.action_catalog}
        if set(self.action_observations) != set(catalog):
            raise ValueError("V3.1 private observations do not cover the action catalog")
        for action_id, observation in self.action_observations.items():
            observation.assert_sealed()
            if observation.action_hash != catalog[action_id].action_hash:
                raise ValueError("V3.1 observation is bound to another action")
        if len(self.probe_initial_states) != len(self.probe_clean_states):
            raise ValueError("V3.1 probe initial states and truths differ")
        if len(self.probe_input_sequences) != len(self.probe_clean_states):
            raise ValueError("V3.1 probe inputs and truths differ")
        if self.performance_eligible and self.expected_issue_routes:
            non_problem = set(self.expected_issue_routes) - {"problem_layer"}
            if non_problem:
                raise ValueError("performance-eligible cases cannot have data/model issues")
        if self.private_hash and self.private_hash != self.content_hash():
            raise ValueError("private_hash does not match V3.1 private case")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "private_hash")

    def assert_sealed(self) -> None:
        if not self.private_hash or self.private_hash != self.content_hash():
            raise ValueError("V3.1 private case is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PrivateControlledDynamicsCaseV31":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"private_hash"}),
            private_hash=draft.content_hash(),
        )


class PrivateControlledDynamicsWorldPackV31(StrictModel):
    schema_version: Literal["3.1"] = "3.1"
    spec_hash: Sha256
    cases: list[PrivateControlledDynamicsCaseV31] = Field(min_length=32, max_length=80)
    generated_at: datetime
    pack_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_pack(self) -> "PrivateControlledDynamicsWorldPackV31":
        _assert_timezone(self.generated_at, "generated_at")
        case_ids = [case.public_case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("V3.1 private case ids must be unique")
        for case in self.cases:
            case.assert_sealed()
        if self.pack_hash and self.pack_hash != self.content_hash():
            raise ValueError("pack_hash does not match V3.1 private pack")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "pack_hash")

    def assert_sealed(self) -> None:
        if not self.pack_hash or self.pack_hash != self.content_hash():
            raise ValueError("V3.1 private pack is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PrivateControlledDynamicsWorldPackV31":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"pack_hash"}),
            pack_hash=draft.content_hash(),
        )


class ControlledDriftModelV31(StrictModel):
    schema_version: Literal["3.1"] = "3.1"
    model_id: Identifier
    case_id: Identifier
    state_names: list[Identifier]
    actuator_hash: Sha256
    basis_terms: list[PolynomialBasisTermV24]
    coefficient_matrix: list[list[Annotated[float, Field(allow_inf_nan=False)]]]
    source_observation_hashes: list[Sha256]
    normalized_design_rank: Annotated[int, Field(ge=0)]
    normalized_condition_number: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    normalized_derivative_residual: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    structural_identifiability_proven: Literal[False] = False
    model_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_model(self) -> "ControlledDriftModelV31":
        if len(self.coefficient_matrix) != len(self.state_names):
            raise ValueError("V3.1 model equation count differs from state dimension")
        if any(len(row) != len(self.basis_terms) for row in self.coefficient_matrix):
            raise ValueError("V3.1 model coefficient width differs from basis")
        if self.model_hash and self.model_hash != self.content_hash():
            raise ValueError("model_hash does not match V3.1 drift model")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "model_hash")

    def assert_sealed(self) -> None:
        if not self.model_hash or self.model_hash != self.content_hash():
            raise ValueError("V3.1 drift model is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ControlledDriftModelV31":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"model_hash"}),
            model_hash=draft.content_hash(),
        )


class ControlledDynamicsStepReceiptV31(StrictModel):
    schema_version: Literal["3.1"] = "3.1"
    step_index: Annotated[int, Field(ge=1, le=3)]
    action_kind: Literal["clarify_target", "controlled_experiment", "abstain"]
    contract_before_hash: Sha256
    contract_after_hash: Sha256
    clarification_evidence: TargetClarificationEvidenceV31 | None = None
    acquisition_receipts: list[ExperimentAcquisitionReceiptV31] = Field(default_factory=list)
    selected_action_id: Identifier | None = None
    selected_action_hash: Sha256 | None = None
    permission: ExperimentPermissionDecisionV31 | None = None
    observation_hash: Sha256 | None = None
    step_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_step(self) -> "ControlledDynamicsStepReceiptV31":
        if self.action_kind == "clarify_target":
            if self.clarification_evidence is None or self.permission is not None:
                raise ValueError("clarification step needs evidence and no experiment permission")
            self.clarification_evidence.assert_sealed()
            if self.contract_before_hash == self.contract_after_hash:
                raise ValueError("clarification step must supersede the contract")
        elif self.action_kind == "controlled_experiment":
            if not self.acquisition_receipts or self.permission is None:
                raise ValueError("experiment step needs acquisition and permission")
            for receipt in self.acquisition_receipts:
                receipt.assert_sealed()
            self.permission.assert_sealed()
            if self.permission.decision != "allow_synthetic":
                raise ValueError("controlled experiment step requires allow_synthetic")
            if not (self.selected_action_id and self.selected_action_hash and self.observation_hash):
                raise ValueError("executed experiment needs selected action and observation")
            if self.contract_before_hash != self.contract_after_hash:
                raise ValueError("experiment cannot silently change the problem contract")
        else:
            if self.permission is None or self.permission.decision not in {"deny", "abstain"}:
                raise ValueError("abstention step needs a deny/abstain permission result")
            if self.observation_hash is not None:
                raise ValueError("abstention cannot have an experiment observation")
        if self.step_hash and self.step_hash != self.content_hash():
            raise ValueError("step_hash does not match V3.1 step")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "step_hash")

    def assert_sealed(self) -> None:
        if not self.step_hash or self.step_hash != self.content_hash():
            raise ValueError("V3.1 step is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsStepReceiptV31":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"step_hash"}),
            step_hash=draft.content_hash(),
        )


class ControlledDynamicsCaseReceiptV31(StrictModel):
    schema_version: Literal["3.1"] = "3.1"
    receipt_id: Identifier
    case_id: Identifier
    public_case_hash: Sha256
    policy_hash: Sha256
    arm: ArmV31
    steps: list[ControlledDynamicsStepReceiptV31] = Field(min_length=1, max_length=3)
    final_contract: ControlledDynamicsContractV31
    selected_action_ids: list[Identifier] = Field(max_length=3)
    observation_hashes: list[Sha256] = Field(max_length=3)
    final_model: ControlledDriftModelV31 | None
    issue_routes: list[RouteLayerV31]
    action_budget_consumed: Annotated[int, Field(ge=0, le=3)]
    abstention_count: Annotated[int, Field(ge=0, le=1)]
    invalid_action_count: Literal[0] = 0
    actual_state_violation_count: Literal[0] = 0
    target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
    executed_at: datetime
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "ControlledDynamicsCaseReceiptV31":
        _assert_timezone(self.executed_at, "executed_at")
        self.final_contract.assert_sealed()
        for step in self.steps:
            step.assert_sealed()
        if self.final_model is not None:
            self.final_model.assert_sealed()
        if len(set(self.issue_routes)) != len(self.issue_routes):
            raise ValueError("V3.1 issue routes must be unique")
        allowed = sum(step.action_kind == "controlled_experiment" for step in self.steps)
        clarified = sum(step.action_kind == "clarify_target" for step in self.steps)
        if self.action_budget_consumed != allowed + clarified:
            raise ValueError("V3.1 receipt budget is inconsistent")
        if len(self.selected_action_ids) != allowed or len(self.observation_hashes) != allowed:
            raise ValueError("V3.1 selected actions/observations are inconsistent")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match V3.1 case receipt")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("V3.1 case receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsCaseReceiptV31":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


class ControlledDynamicsSelectionBundleV31(StrictModel):
    schema_version: Literal["3.1"] = "3.1"
    spec_hash: Sha256
    private_pack_hash: Sha256
    policy_hash: Sha256
    arm: ArmV31
    case_receipts: list[ControlledDynamicsCaseReceiptV31] = Field(min_length=32, max_length=80)
    total_action_budget_consumed: Annotated[int, Field(ge=0)]
    total_abstentions: Annotated[int, Field(ge=0)]
    invalid_action_count: Literal[0] = 0
    actual_state_violation_count: Literal[0] = 0
    bundle_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bundle(self) -> "ControlledDynamicsSelectionBundleV31":
        for receipt in self.case_receipts:
            receipt.assert_sealed()
        if self.total_action_budget_consumed != sum(
            receipt.action_budget_consumed for receipt in self.case_receipts
        ):
            raise ValueError("V3.1 bundle budget total is inconsistent")
        if self.total_abstentions != sum(
            receipt.abstention_count for receipt in self.case_receipts
        ):
            raise ValueError("V3.1 bundle abstention total is inconsistent")
        if self.bundle_hash and self.bundle_hash != self.content_hash():
            raise ValueError("bundle_hash does not match V3.1 selection bundle")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "bundle_hash")

    def assert_sealed(self) -> None:
        if not self.bundle_hash or self.bundle_hash != self.content_hash():
            raise ValueError("V3.1 selection bundle is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsSelectionBundleV31":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"bundle_hash"}),
            bundle_hash=draft.content_hash(),
        )


class ControlledDynamicsCaseResultV31(StrictModel):
    case_id: Identifier
    mechanism: MechanismV31
    seed: int
    performance_eligible: bool
    baseline_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
    candidate_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
    absolute_loss_improvement: Annotated[float, Field(allow_inf_nan=False)] | None
    material_negative_transfer: bool | None
    expected_issue_routes: list[RouteLayerV31]
    candidate_issue_routes: list[RouteLayerV31]
    routing_correct: bool
    candidate_reformulated: bool
    candidate_spurious_reformulation: bool


class ControlledDynamicsMechanismResultV31(StrictModel):
    mechanism: MechanismV31
    eligible_case_count: Annotated[int, Field(ge=0)]
    mean_absolute_loss_improvement: Annotated[float, Field(allow_inf_nan=False)] | None
    routing_accuracy: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class ControlledDynamicsReportV31(StrictModel):
    schema_version: Literal["3.1"] = "3.1"
    report_id: Identifier
    phase: Literal["exploratory", "confirmation"]
    spec_hash: Sha256
    private_pack_hash: Sha256
    baseline_bundle_hash: Sha256
    candidate_bundle_hash: Sha256
    case_results: list[ControlledDynamicsCaseResultV31] = Field(min_length=32, max_length=80)
    mechanism_results: list[ControlledDynamicsMechanismResultV31] = Field(min_length=4, max_length=4)
    performance_eligible_case_count: Annotated[int, Field(ge=1)]
    excluded_data_issue_count: Annotated[int, Field(ge=0)]
    excluded_model_issue_count: Annotated[int, Field(ge=0)]
    macro_absolute_loss_improvement: Annotated[float, Field(allow_inf_nan=False)]
    macro_improvement_ci_low: Annotated[float, Field(allow_inf_nan=False)]
    macro_improvement_ci_high: Annotated[float, Field(allow_inf_nan=False)]
    material_negative_transfer_count: Annotated[int, Field(ge=0)]
    material_negative_transfer_rate: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    material_negative_transfer_upper: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    routing_accuracy: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    required_reformulation_count: Annotated[int, Field(ge=0)]
    evidence_bound_reformulation_count: Annotated[int, Field(ge=0)]
    spurious_reformulation_count: Annotated[int, Field(ge=0)]
    data_gate_experiment_count: Literal[0] = 0
    invalid_action_count: Literal[0] = 0
    actual_state_violation_count: Literal[0] = 0
    formal_safety_proven: Literal[False] = False
    gates: dict[Identifier, bool]
    status: Literal[
        "exploratory_only_v31",
        "candidate_rejected_v31",
        "promoted_for_synthetic_controlled_epistemic_worldpack_v31",
    ]
    evaluated_at: datetime
    report_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "ControlledDynamicsReportV31":
        _assert_timezone(self.evaluated_at, "evaluated_at")
        if self.phase == "exploratory" and self.status != "exploratory_only_v31":
            raise ValueError("exploratory V3.1 cannot promote or confirm-reject")
        if self.phase == "confirmation":
            promoted = all(self.gates.values())
            expected = (
                "promoted_for_synthetic_controlled_epistemic_worldpack_v31"
                if promoted
                else "candidate_rejected_v31"
            )
            if self.status != expected:
                raise ValueError("V3.1 confirmation status disagrees with gates")
        if self.report_hash and self.report_hash != self.content_hash():
            raise ValueError("report_hash does not match V3.1 report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "report_hash")

    def assert_sealed(self) -> None:
        if not self.report_hash or self.report_hash != self.content_hash():
            raise ValueError("V3.1 report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsReportV31":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"report_hash"}),
            report_hash=draft.content_hash(),
        )


class ControlledDynamicsQualificationV31(StrictModel):
    schema_version: Literal["3.1"] = "3.1"
    qualification_id: Literal[
        "synthetic_known_actuator_goal_oriented_experiment_routing_v31"
    ] = "synthetic_known_actuator_goal_oriented_experiment_routing_v31"
    policy_hash: Sha256
    report_hash: Sha256
    scope: Literal[
        "synthetic_known_actuator_piecewise_constant_input_worldpack_v31"
    ] = "synthetic_known_actuator_piecewise_constant_input_worldpack_v31"
    known_actuator_only: Literal[True] = True
    empirical_risk_only: Literal[True] = True
    formal_safety_proven: Literal[False] = False
    structural_identifiability_proven: Literal[False] = False
    unknown_actuator_learning_proven: Literal[False] = False
    real_world_validity_proven: Literal[False] = False
    autonomous_modeling_proven: Literal[False] = False
    qualified_at: datetime
    qualification_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_qualification(self) -> "ControlledDynamicsQualificationV31":
        _assert_timezone(self.qualified_at, "qualified_at")
        if self.qualification_hash and self.qualification_hash != self.content_hash():
            raise ValueError("qualification_hash does not match V3.1 qualification")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "qualification_hash")

    def assert_sealed(self) -> None:
        if not self.qualification_hash or self.qualification_hash != self.content_hash():
            raise ValueError("V3.1 qualification is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsQualificationV31":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"qualification_hash"}),
            qualification_hash=draft.content_hash(),
        )


class ControlledDynamicsManifestV31(StrictModel):
    schema_version: Literal["3.1"] = "3.1"
    run_id: Identifier
    artifact_refs: list[ArtifactRef] = Field(min_length=7)
    terminal_status: Literal[
        "exploratory_only_v31",
        "candidate_rejected_v31",
        "promoted_for_synthetic_controlled_epistemic_worldpack_v31",
    ]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "ControlledDynamicsManifestV31":
        _assert_timezone(self.created_at, "created_at")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match V3.1 manifest")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("V3.1 manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsManifestV31":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class ControlledDynamicsOutcomeV31:
    store: RunStore
    spec: ControlledDynamicsWorldPackSpecV31
    private_pack: PrivateControlledDynamicsWorldPackV31
    baseline_policy: ControlledDynamicsPolicyV31
    candidate_policy: ControlledDynamicsPolicyV31
    baseline_bundle: ControlledDynamicsSelectionBundleV31
    candidate_bundle: ControlledDynamicsSelectionBundleV31
    report: ControlledDynamicsReportV31
    qualification: ControlledDynamicsQualificationV31 | None
    manifest: ControlledDynamicsManifestV31


def default_controlled_dynamics_policies_v31(
    *,
    prior_epistemic_qualification_hash: str,
    prior_active_design_qualification_hash: str,
    method_evidence_hash: str,
) -> tuple[ControlledDynamicsPolicyV31, ControlledDynamicsPolicyV31]:
    shared = dict(
        prior_epistemic_qualification_hash=prior_epistemic_qualification_hash,
        prior_active_design_qualification_hash=prior_active_design_qualification_hash,
        method_evidence_hash=method_evidence_hash,
    )
    return (
        ControlledDynamicsPolicyV31.seal(
            policy_id="random_bounded_inputs_v31",
            arm="random_bounded_inputs",
            selection_rule="prefrozen_random_without_replacement",
            may_reformulate_problem=False,
            **shared,
        ),
        ControlledDynamicsPolicyV31.seal(
            policy_id="goal_oriented_epistemic_control_v31",
            arm="goal_oriented_epistemic_control",
            selection_rule="clarify_then_goal_information_risk_utility",
            may_reformulate_problem=True,
            **shared,
        ),
    )


def _default_spec_v31(
    phase: Literal["exploratory", "confirmation"],
    *,
    baseline_policy_hash: str,
    candidate_policy_hash: str,
    method_evidence_hash: str,
    prior_epistemic_qualification_hash: str,
    prior_active_design_qualification_hash: str,
    frozen_at: datetime | None,
) -> ControlledDynamicsWorldPackSpecV31:
    return ControlledDynamicsWorldPackSpecV31.seal(
        experiment_id=f"controlled_dynamics_{phase}_v31",
        phase=phase,
        mechanisms=list(MECHANISMS_V31),
        seeds=list(
            EXPLORATORY_SEEDS_V31
            if phase == "exploratory"
            else CONFIRMATION_SEEDS_V31
        ),
        baseline_policy_hash=baseline_policy_hash,
        candidate_policy_hash=candidate_policy_hash,
        method_evidence_hash=method_evidence_hash,
        prior_epistemic_qualification_hash=prior_epistemic_qualification_hash,
        prior_active_design_qualification_hash=prior_active_design_qualification_hash,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


def default_controlled_dynamics_exploratory_spec_v31(**kwargs: object) -> ControlledDynamicsWorldPackSpecV31:
    return _default_spec_v31("exploratory", **kwargs)


def default_controlled_dynamics_confirmation_spec_v31(**kwargs: object) -> ControlledDynamicsWorldPackSpecV31:
    return _default_spec_v31("confirmation", **kwargs)


def assert_single_component_controlled_dynamics_v31(
    baseline: ControlledDynamicsPolicyV31,
    candidate: ControlledDynamicsPolicyV31,
) -> None:
    baseline.assert_sealed()
    candidate.assert_sealed()
    if baseline.arm != "random_bounded_inputs" or candidate.arm != "goal_oriented_epistemic_control":
        raise ValueError("V3.1 requires the frozen baseline and candidate arms")
    for field in (
        "maximum_actions",
        "known_actuator_required",
        "prior_epistemic_qualification_hash",
        "prior_active_design_qualification_hash",
        "method_evidence_hash",
    ):
        if getattr(baseline, field) != getattr(candidate, field):
            raise ValueError(f"V3.1 policies differ outside acquisition at {field}")


def _mechanism_definition_v31(
    mechanism: MechanismV31,
    random: Random,
) -> tuple[list[str], list[float], list[list[float]], list[tuple[float, float]], dict[str, float]]:
    if mechanism == "exponential_decay":
        return (
            ["x"], [1.6], [[1.0]], [(-0.5, 3.5)],
            {"rate": 0.35 + 0.22 * random.random()},
        )
    if mechanism == "logistic_growth":
        return (
            ["x"], [1.1], [[1.0]], [(-0.5, 12.5)],
            {
                "rate": 0.55 + 0.25 * random.random(),
                "capacity": 7.5 + 3.0 * random.random(),
            },
        )
    if mechanism == "damped_oscillator":
        return (
            ["position", "velocity"], [0.8, 0.0], [[0.0], [1.0]],
            [(-3.0, 3.0), (-4.0, 4.0)],
            {
                "omega": 1.0 + 0.45 * random.random(),
                "damping": 0.08 + 0.14 * random.random(),
            },
        )
    return (
        ["position", "velocity"], [0.85, 0.0], [[0.0], [1.0]],
        [(-2.2, 2.2), (-4.5, 4.5)],
        {
            "linear": -1.0 - 0.35 * random.random(),
            "cubic": 1.15 + 0.55 * random.random(),
            "damping": 0.12 + 0.12 * random.random(),
        },
    )


def _truth_rhs_v31(
    mechanism: MechanismV31,
    state: np.ndarray,
    parameters: dict[str, float],
    control: np.ndarray,
    actuator: np.ndarray,
) -> np.ndarray:
    if mechanism == "exponential_decay":
        drift = np.asarray([-parameters["rate"] * state[0]])
    elif mechanism == "logistic_growth":
        drift = np.asarray([
            parameters["rate"] * state[0] * (1.0 - state[0] / parameters["capacity"])
        ])
    elif mechanism == "damped_oscillator":
        omega = parameters["omega"]
        drift = np.asarray([
            state[1],
            -(omega**2) * state[0] - 2.0 * parameters["damping"] * omega * state[1],
        ])
    else:
        drift = np.asarray([
            state[1],
            -parameters["damping"] * state[1]
            + parameters["linear"] * state[0]
            - parameters["cubic"] * state[0] ** 3,
        ])
    return drift + actuator @ control


def _input_at_time_v31(
    sequence: list[list[float]],
    time: float,
    segment_duration: float,
) -> np.ndarray:
    index = min(int(max(time, 0.0) / segment_duration), len(sequence) - 1)
    return np.asarray(sequence[index], dtype=float)


def _simulate_truth_v31(
    mechanism: MechanismV31,
    initial_state: list[float],
    times: list[float],
    parameters: dict[str, float],
    input_sequence: list[list[float]],
    actuator_matrix: list[list[float]],
    segment_duration: float,
) -> list[list[float]]:
    actuator = np.asarray(actuator_matrix, dtype=float)

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        return _truth_rhs_v31(
            mechanism,
            state,
            parameters,
            _input_at_time_v31(input_sequence, time, segment_duration),
            actuator,
        )

    solution = solve_ivp(
        rhs,
        (float(times[0]), float(times[-1])),
        np.asarray(initial_state, dtype=float),
        t_eval=np.asarray(times, dtype=float),
        rtol=1e-9,
        atol=1e-11,
        max_step=segment_duration / 8.0,
    )
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError("V3.1 hidden dynamics simulation failed")
    return solution.y.T.tolist()


def _noisy_states_v31(
    clean: list[list[float]],
    noise_fraction: float,
    seed: int,
    *,
    calibration_failed: bool,
) -> list[list[float]]:
    values = np.asarray(clean, dtype=float)
    scale = np.maximum(np.std(values, axis=0), 0.1)
    random = np.random.default_rng(seed)
    noisy = values + random.normal(0.0, noise_fraction * scale, size=values.shape)
    if calibration_failed:
        noisy = noisy + 0.4 * scale[np.newaxis, :]
    return noisy.tolist()


def _catalog_sequences_v31(amplitude: float) -> list[list[list[float]]]:
    sequences = []
    for signs in itertools.product((-1.0, 1.0), repeat=6):
        if sum(value > 0 for value in signs) != 3:
            continue
        switches = sum(left != right for left, right in zip(signs, signs[1:]))
        if switches == 3:
            sequences.append([[amplitude * value] for value in signs])
    if len(sequences) != 8:
        raise RuntimeError("V3.1 frozen balanced input catalog changed")
    return sequences


def _state_peak_ratio_v31(
    states: list[list[float]],
    lower: list[float],
    upper: list[float],
) -> float:
    values = np.asarray(states, dtype=float)
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    center = (low + high) / 2.0
    half = (high - low) / 2.0
    return float(np.max(np.abs((values - center) / half)))


def _probe_sequences_v31(
    target: DecisionTargetV31,
    amplitude: float,
) -> list[list[list[float]]]:
    zero = [[0.0] for _ in range(6)]
    if target == "free_run_prediction":
        return [zero, zero, zero]
    patterns = (
        (-1, 1, 1, -1, 1, -1),
        (1, 1, -1, 1, -1, -1),
        (1, -1, -1, -1, 1, 1),
    )
    return [
        [[0.72 * amplitude * sign] for sign in pattern]
        for pattern in patterns
    ]


def _generate_case_v31(
    spec: ControlledDynamicsWorldPackSpecV31,
    mechanism: MechanismV31,
    seed: int,
) -> PrivateControlledDynamicsCaseV31:
    mechanism_index = spec.mechanisms.index(mechanism)
    seed_index = spec.seeds.index(seed)
    random = Random(seed * 104729 + mechanism_index * 7919)
    state_names, initial, actuator_matrix, bounds, parameters = _mechanism_definition_v31(
        mechanism, random
    )
    case_id = f"controlled_{mechanism}_{seed}"
    actuator = KnownActuatorMapV31.seal(
        actuator_id=f"actuator_{mechanism}_{seed}",
        state_names=state_names,
        input_names=["u"],
        matrix=actuator_matrix,
        source_ref=f"synthetic_worldpack_known_actuator:{case_id}",
    )
    sequences = _catalog_sequences_v31(spec.input_amplitude)
    energy = spec.segment_duration * sum(value[0] ** 2 for value in sequences[0])
    envelope = ExperimentConstraintEnvelopeV31.seal(
        envelope_id=f"envelope_{mechanism}_{seed}",
        actuator_hash=actuator.actuator_hash,
        state_lower_bounds=[low for low, _ in bounds],
        state_upper_bounds=[high for _, high in bounds],
        required_peak_amplitude=spec.input_amplitude,
        required_total_energy=energy,
        required_switch_count=3,
        maximum_empirical_prediction_risk=spec.maximum_empirical_prediction_risk,
    )
    actions = [
        PiecewiseConstantInputActionV31.seal(
            action_id=f"input_{mechanism}_{seed}_{index:02d}",
            actuator_hash=actuator.actuator_hash,
            segment_duration=spec.segment_duration,
            input_values=sequence,
            peak_amplitude=spec.input_amplitude,
            total_energy=energy,
            switch_count=3,
        )
        for index, sequence in enumerate(sequences)
    ]
    true_target: DecisionTargetV31 = (
        "controlled_response_prediction"
        if (seed_index + mechanism_index) % 2
        else "free_run_prediction"
    )
    underspecified = (seed_index + mechanism_index) % 4 != 0
    initial_contract = ControlledDynamicsContractV31.seal(
        contract_id=f"contract_{case_id}_v1",
        case_id=case_id,
        version=1,
        decision_target=(
            "free_run_prediction" if underspecified else true_target
        ),
        target_status=("default_unverified" if underspecified else "authoritative"),
        unresolved_fields=(["decision_target"] if underspecified else []),
        frozen_at=spec.frozen_at,
    )
    # Every fifth in-family seed has a pre-frozen calibration failure. Duffing is
    # reserved for the model-layer misspecification sentinel.
    calibration_failed = mechanism != "duffing_oscillator" and seed_index % 5 == 0
    expected_routes: list[RouteLayerV31] = []
    if underspecified:
        expected_routes.append("problem_layer")
    if calibration_failed:
        expected_routes.append("data_layer")
    elif mechanism == "duffing_oscillator":
        expected_routes.append("model_layer")
    performance_eligible = not calibration_failed and mechanism != "duffing_oscillator"

    times = [index * spec.time_step for index in range(spec.trajectory_points)]
    zero_sequence = [[0.0] for _ in range(spec.segment_count)]
    pilot_clean = _simulate_truth_v31(
        mechanism, initial, times, parameters, zero_sequence,
        actuator.matrix, spec.segment_duration,
    )
    pilot = PilotObservationV31.seal(
        observation_id=f"pilot_{case_id}",
        case_id=case_id,
        actuator_hash=actuator.actuator_hash,
        times=times,
        states=_noisy_states_v31(
            pilot_clean,
            spec.observation_noise_fraction,
            seed * 1000003 + mechanism_index * 10007,
            calibration_failed=calibration_failed,
        ),
        inputs=[[0.0] for _ in times],
        quality_flags=(["sensor_calibration_failed"] if calibration_failed else []),
    )
    observations: dict[str, ControlledObservationReceiptV31] = {}
    for index, action in enumerate(actions):
        clean = _simulate_truth_v31(
            mechanism, initial, times, parameters, action.input_values,
            actuator.matrix, spec.segment_duration,
        )
        inputs = [
            _input_at_time_v31(action.input_values, time, spec.segment_duration).tolist()
            for time in times
        ]
        observations[action.action_id] = ControlledObservationReceiptV31.seal(
            observation_id=f"observation_{case_id}_{index:02d}",
            case_id=case_id,
            action_hash=action.action_hash,
            actuator_hash=actuator.actuator_hash,
            times=times,
            states=_noisy_states_v31(
                clean,
                spec.observation_noise_fraction,
                seed * 2000003 + mechanism_index * 20011 + index * 101,
                calibration_failed=False,
            ),
            inputs=inputs,
            empirical_peak_state_ratio=_state_peak_ratio_v31(
                clean, envelope.state_lower_bounds, envelope.state_upper_bounds
            ),
            quality_flags=[],
            observed_at=spec.frozen_at,
        )
    probe_initials: list[list[float]] = []
    if len(initial) == 1:
        probe_initials = [[0.65 * initial[0]], [1.15 * initial[0]], [1.55 * initial[0]]]
    else:
        probe_initials = [
            [0.55 * initial[0], 0.2],
            [1.15 * initial[0], -0.25],
            [-0.65 * initial[0], 0.35],
        ]
    probe_inputs = _probe_sequences_v31(true_target, spec.input_amplitude)
    probe_clean = [
        _simulate_truth_v31(
            mechanism, probe_initial, times, parameters, input_sequence,
            actuator.matrix, spec.segment_duration,
        )
        for probe_initial, input_sequence in zip(probe_initials, probe_inputs, strict=True)
    ]
    public = ControlledDynamicsPublicCaseV31.seal(
        case_id=case_id,
        state_names=state_names,
        initial_state=initial,
        actuator=actuator,
        envelope=envelope,
        action_catalog=actions,
        pilot=pilot,
        initial_contract=initial_contract,
    )
    return PrivateControlledDynamicsCaseV31.seal(
        public_case=public,
        mechanism=mechanism,
        hidden_parameters=parameters,
        true_decision_target=true_target,
        target_was_underspecified=underspecified,
        expected_issue_routes=expected_routes,
        performance_eligible=performance_eligible,
        action_observations=observations,
        probe_initial_states=probe_initials,
        probe_input_sequences=probe_inputs,
        probe_clean_states=probe_clean,
    )


def generate_private_controlled_dynamics_worldpack_v31(
    spec: ControlledDynamicsWorldPackSpecV31,
    *,
    generated_at: datetime | None = None,
) -> PrivateControlledDynamicsWorldPackV31:
    spec.assert_sealed()
    return PrivateControlledDynamicsWorldPackV31.seal(
        spec_hash=spec.spec_hash,
        cases=[
            _generate_case_v31(spec, mechanism, seed)
            for seed in spec.seeds
            for mechanism in spec.mechanisms
        ],
        generated_at=generated_at or datetime.now(timezone.utc),
    )


def _observation_arrays_v31(
    public: ControlledDynamicsPublicCaseV31,
    observations: list[ControlledObservationReceiptV31],
    spec: ControlledDynamicsWorldPackSpecV31,
) -> tuple[list[PolynomialBasisTermV24], np.ndarray, np.ndarray]:
    terms = polynomial_basis_terms(public.state_names, spec.polynomial_degree)
    all_items: list[tuple[list[list[float]], list[list[float]], str]] = [
        (public.pilot.states, public.pilot.inputs, public.pilot.observation_hash)
    ]
    all_items.extend(
        (item.states, item.inputs, item.observation_hash) for item in observations
    )
    libraries: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    actuator = np.asarray(public.actuator.matrix, dtype=float)
    trim = spec.savgol_window // 2
    for states_raw, inputs_raw, _ in all_items:
        states = np.asarray(states_raw, dtype=float)
        inputs = np.asarray(inputs_raw, dtype=float)
        derivatives = np.column_stack([
            savgol_filter(
                states[:, equation],
                window_length=spec.savgol_window,
                polyorder=spec.savgol_order,
                deriv=1,
                delta=spec.time_step,
                mode="interp",
            )
            for equation in range(states.shape[1])
        ])
        drift_targets = derivatives - inputs @ actuator.T
        libraries.append(evaluate_polynomial_library(states[trim:-trim], terms))
        targets.append(drift_targets[trim:-trim])
    return terms, np.vstack(libraries), np.vstack(targets)


def _ridge_v31(library: np.ndarray, targets: np.ndarray, alpha: float) -> np.ndarray:
    scales = np.sqrt(np.mean(library**2, axis=0))
    scales[scales < 1e-12] = 1.0
    normalized = library / scales
    solved = np.linalg.solve(
        normalized.T @ normalized + alpha * np.eye(normalized.shape[1]),
        normalized.T @ targets,
    ).T
    return solved / scales[np.newaxis, :]


def _fit_model_v31(
    public: ControlledDynamicsPublicCaseV31,
    observations: list[ControlledObservationReceiptV31],
    spec: ControlledDynamicsWorldPackSpecV31,
) -> ControlledDriftModelV31:
    terms, library, targets = _observation_arrays_v31(public, observations, spec)
    coefficients = _ridge_v31(library, targets, spec.ridge_alpha)
    scales = np.sqrt(np.mean(library**2, axis=0))
    scales[scales < 1e-12] = 1.0
    normalized = library / scales
    for equation in range(targets.shape[1]):
        active = np.abs(coefficients[equation]) >= spec.sparsity_threshold
        for _ in range(12):
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
            if np.array_equal(active, previous):
                break
        coefficients[equation, ~active] = 0.0
    fitted = library @ coefficients.T
    residual = float(
        np.sqrt(np.mean((fitted - targets) ** 2))
        / max(float(np.sqrt(np.mean(targets**2))), 0.1)
    )
    condition = float(np.linalg.cond(normalized))
    if not math.isfinite(condition):
        condition = 1e15
    return ControlledDriftModelV31.seal(
        model_id=f"model_{public.case_id}_{len(observations)}",
        case_id=public.case_id,
        state_names=public.state_names,
        actuator_hash=public.actuator.actuator_hash,
        basis_terms=terms,
        coefficient_matrix=coefficients.tolist(),
        source_observation_hashes=[
            public.pilot.observation_hash,
            *[item.observation_hash for item in observations],
        ],
        normalized_design_rank=int(np.linalg.matrix_rank(normalized, tol=1e-10)),
        normalized_condition_number=condition,
        normalized_derivative_residual=residual,
    )


def _bootstrap_ensemble_v31(
    public: ControlledDynamicsPublicCaseV31,
    observations: list[ControlledObservationReceiptV31],
    spec: ControlledDynamicsWorldPackSpecV31,
    *,
    seed: int,
) -> tuple[list[PolynomialBasisTermV24], np.ndarray, np.ndarray]:
    terms, library, targets = _observation_arrays_v31(public, observations, spec)
    random = np.random.default_rng(seed)
    sample_count = max(library.shape[1] + 2, int(len(library) * spec.bootstrap_fraction))
    ensemble = []
    for _ in range(spec.ensemble_members):
        indices = random.integers(0, len(library), size=sample_count)
        ensemble.append(_ridge_v31(library[indices], targets[indices], spec.ridge_alpha))
    return terms, library, np.asarray(ensemble)


def _ensemble_paths_v31(
    public: ControlledDynamicsPublicCaseV31,
    action: PiecewiseConstantInputActionV31,
    terms: list[PolynomialBasisTermV24],
    ensemble: np.ndarray,
    spec: ControlledDynamicsWorldPackSpecV31,
) -> np.ndarray:
    state = np.repeat(
        np.asarray(public.initial_state, dtype=float)[np.newaxis, :],
        ensemble.shape[0],
        axis=0,
    )
    actuator = np.asarray(public.actuator.matrix, dtype=float)
    paths = [state.copy()]
    for index in range(1, spec.trajectory_points):
        time = (index - 1) * spec.time_step
        control = _input_at_time_v31(
            action.input_values, time, spec.segment_duration
        )
        library = evaluate_polynomial_library(state, terms)
        drift = np.einsum("bt,bet->be", library, ensemble)
        state = state + spec.time_step * (drift + (actuator @ control)[np.newaxis, :])
        state = np.clip(state, -1e4, 1e4)
        paths.append(state.copy())
    return np.stack(paths, axis=1)


def _d_opt_gain_v31(current: np.ndarray, proposed_states: np.ndarray, terms: list[PolynomialBasisTermV24]) -> float:
    proposed = evaluate_polynomial_library(proposed_states, terms)
    combined = np.vstack([current, proposed])
    scales = np.sqrt(np.mean(combined**2, axis=0))
    scales[scales < 1e-12] = 1.0
    base = current / scales
    added = proposed / scales
    identity = np.eye(current.shape[1])
    base_matrix = identity + base.T @ base / max(len(base), 1)
    full_matrix = base_matrix + added.T @ added / max(len(added), 1)
    base_sign, base_logdet = np.linalg.slogdet(base_matrix)
    full_sign, full_logdet = np.linalg.slogdet(full_matrix)
    if base_sign <= 0 or full_sign <= 0:
        return 0.0
    return float(max(full_logdet - base_logdet, 0.0))


def _acquisition_receipts_v31(
    spec: ControlledDynamicsWorldPackSpecV31,
    public: ControlledDynamicsPublicCaseV31,
    observations: list[ControlledObservationReceiptV31],
    available: list[PiecewiseConstantInputActionV31],
    target: DecisionTargetV31,
    step_index: int,
) -> list[ExperimentAcquisitionReceiptV31]:
    terms, current_library, ensemble = _bootstrap_ensemble_v31(
        public,
        observations,
        spec,
        seed=int(sha256_value([spec.spec_hash, public.case_id, step_index])[:16], 16),
    )
    lower = np.asarray(public.envelope.state_lower_bounds, dtype=float)
    upper = np.asarray(public.envelope.state_upper_bounds, dtype=float)
    receipts: list[ExperimentAcquisitionReceiptV31] = []
    for action in available:
        failures = validate_action_against_envelope_v31(
            action, public.actuator, public.envelope
        )
        if failures:
            raise RuntimeError(f"V3.1 catalog action failed Harness validation: {failures}")
        paths = _ensemble_paths_v31(public, action, terms, ensemble, spec)
        mean_path = np.mean(paths, axis=0)
        outside = np.any((paths < lower) | (paths > upper), axis=(1, 2))
        empirical_risk = float(np.mean(outside))
        state_scale = np.maximum(upper - lower, 1e-6)
        disagreement = float(np.mean(np.var(paths / state_scale, axis=0)))
        d_opt = _d_opt_gain_v31(current_library, mean_path, terms)
        if target == "controlled_response_prediction":
            decision_information = float(
                np.mean(np.linalg.norm(mean_path - mean_path[0], axis=1))
                + np.mean(np.var(paths[:, -1, :] / state_scale, axis=0))
            )
        else:
            mean_library = evaluate_polynomial_library(mean_path, terms)
            drift_predictions = np.einsum("pt,bet->bpe", mean_library, ensemble)
            decision_information = float(
                np.mean(np.var(drift_predictions / state_scale, axis=0))
            )
        admissible = empirical_risk <= public.envelope.maximum_empirical_prediction_risk
        utility = (
            d_opt
            + 0.65 * disagreement
            + 0.35 * decision_information
            - 0.05 * action.action_cost
            - 2.0 * empirical_risk
        )
        receipts.append(ExperimentAcquisitionReceiptV31.seal(
            acquisition_id=f"acq_{public.case_id}_{step_index}_{action.action_id}",
            case_id=public.case_id,
            action_hash=action.action_hash,
            decision_target=target,
            d_optimal_gain=d_opt,
            model_disagreement=disagreement,
            decision_information=decision_information,
            action_cost=action.action_cost,
            empirical_prediction_risk=empirical_risk,
            utility_score=utility,
            admissible=admissible,
            gate_codes=[
                "known_actuator", "peak_equal", "energy_equal", "switch_equal", "cost_equal",
                "empirical_risk_pass" if admissible else "empirical_risk_fail",
            ],
        ))
    return receipts


def _permission_v31(
    acquisition_hash: str,
    envelope_hash: str,
    *,
    data_quality_passed: bool,
    admissible: bool,
    budget_before: int,
    decided_at: datetime,
) -> ExperimentPermissionDecisionV31:
    if not data_quality_passed:
        decision, rule, after = "deny", "deny_data_quality", budget_before
    elif budget_before < 1:
        decision, rule, after = "deny", "deny_budget", budget_before
    elif not admissible:
        decision, rule, after = "abstain", "abstain_no_admissible_action", budget_before
    else:
        decision, rule, after = (
            "allow_synthetic", "allow_bounded_synthetic_experiment", budget_before - 1
        )
    return ExperimentPermissionDecisionV31.seal(
        acquisition_hash=acquisition_hash,
        envelope_hash=envelope_hash,
        decision=decision,
        policy_rule=rule,
        budget_before=budget_before,
        budget_after=after,
        decided_at=decided_at,
    )


def _simulate_model_v31(
    model: ControlledDriftModelV31,
    actuator: KnownActuatorMapV31,
    initial_state: list[float],
    times: list[float],
    input_sequence: list[list[float]],
    segment_duration: float,
) -> list[list[float]]:
    model.assert_sealed()
    actuator.assert_sealed()
    coefficients = np.asarray(model.coefficient_matrix, dtype=float)
    actuator_matrix = np.asarray(actuator.matrix, dtype=float)

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        if np.max(np.abs(state)) > 1e5:
            raise FloatingPointError("V3.1 fitted model diverged")
        library = evaluate_polynomial_library(state.reshape(1, -1), model.basis_terms)[0]
        return coefficients @ library + actuator_matrix @ _input_at_time_v31(
            input_sequence, time, segment_duration
        )

    try:
        solution = solve_ivp(
            rhs,
            (float(times[0]), float(times[-1])),
            np.asarray(initial_state, dtype=float),
            t_eval=np.asarray(times, dtype=float),
            rtol=1e-8,
            atol=1e-10,
            max_step=segment_duration / 8.0,
        )
    except (FloatingPointError, ValueError) as exc:
        raise RuntimeError("V3.1 fitted model simulation failed") from exc
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError("V3.1 fitted model did not cover the probe")
    return solution.y.T.tolist()


def _target_loss_v31(
    private_case: PrivateControlledDynamicsCaseV31,
    model: ControlledDriftModelV31,
    spec: ControlledDynamicsWorldPackSpecV31,
) -> float:
    times = private_case.public_case.pilot.times
    losses = []
    for initial, inputs, truth in zip(
        private_case.probe_initial_states,
        private_case.probe_input_sequences,
        private_case.probe_clean_states,
        strict=True,
    ):
        try:
            predicted = _simulate_model_v31(
                model,
                private_case.public_case.actuator,
                initial,
                times,
                inputs,
                spec.segment_duration,
            )
            losses.append(trajectory_nrmse(truth, predicted))
        except RuntimeError:
            losses.append(10.0)
    return float(np.mean(losses))


def _execute_case_v31(
    spec: ControlledDynamicsWorldPackSpecV31,
    private_case: PrivateControlledDynamicsCaseV31,
    policy: ControlledDynamicsPolicyV31,
    *,
    executed_at: datetime,
) -> ControlledDynamicsCaseReceiptV31:
    public = private_case.public_case
    public.assert_sealed()
    data_quality_passed = not public.pilot.quality_flags
    contract = public.initial_contract
    budget = spec.action_budget
    observations: list[ControlledObservationReceiptV31] = []
    selected_ids: list[str] = []
    steps: list[ControlledDynamicsStepReceiptV31] = []
    issue_routes: list[RouteLayerV31] = []
    abstention_count = 0

    for step_index in range(1, spec.action_budget + 1):
        if (
            policy.may_reformulate_problem
            and contract.target_status == "default_unverified"
        ):
            evidence = TargetClarificationEvidenceV31.seal(
                evidence_id=f"target_evidence_{public.case_id}",
                case_id=public.case_id,
                decision_target=private_case.true_decision_target,
                source_ref=f"synthetic_value_owner:{public.case_id}",
                observed_at=executed_at,
            )
            previous_hash = contract.contract_hash
            contract = ControlledDynamicsContractV31.seal(
                contract_id=f"contract_{public.case_id}_v2",
                case_id=public.case_id,
                version=2,
                decision_target=evidence.decision_target,
                target_status="authoritative",
                unresolved_fields=[],
                parent_contract_hash=previous_hash,
                triggering_evidence_hash=evidence.evidence_hash,
                frozen_at=executed_at,
            )
            budget -= 1
            issue_routes.append("problem_layer")
            steps.append(ControlledDynamicsStepReceiptV31.seal(
                step_index=step_index,
                action_kind="clarify_target",
                contract_before_hash=previous_hash,
                contract_after_hash=contract.contract_hash,
                clarification_evidence=evidence,
            ))
            continue

        if not data_quality_passed:
            permission = _permission_v31(
                sha256_value(["data_gate", public.case_id, policy.arm, step_index]),
                public.envelope.envelope_hash,
                data_quality_passed=False,
                admissible=False,
                budget_before=budget,
                decided_at=executed_at,
            )
            if "data_layer" not in issue_routes:
                issue_routes.append("data_layer")
            abstention_count = 1
            steps.append(ControlledDynamicsStepReceiptV31.seal(
                step_index=step_index,
                action_kind="abstain",
                contract_before_hash=contract.contract_hash,
                contract_after_hash=contract.contract_hash,
                permission=permission,
            ))
            break

        available = [
            action for action in public.action_catalog
            if action.action_id not in selected_ids
        ]
        acquisitions = _acquisition_receipts_v31(
            spec,
            public,
            observations,
            available,
            contract.decision_target,
            step_index,
        )
        by_hash = {receipt.action_hash: receipt for receipt in acquisitions}
        admissible_actions = [
            action for action in available if by_hash[action.action_hash].admissible
        ]
        if policy.arm == "random_bounded_inputs":
            random = Random(int(sha256_value(
                [spec.spec_hash, public.case_id, policy.arm]
            )[:16], 16))
            order = [action.action_id for action in public.action_catalog]
            random.shuffle(order)
            priority = {action_id: index for index, action_id in enumerate(order)}
            ranked = sorted(admissible_actions, key=lambda item: priority[item.action_id])
        else:
            ranked = sorted(
                admissible_actions,
                key=lambda item: (by_hash[item.action_hash].utility_score, item.action_id),
                reverse=True,
            )
        if not ranked:
            best = max(acquisitions, key=lambda item: (item.utility_score, item.action_hash))
            permission = _permission_v31(
                best.acquisition_hash,
                public.envelope.envelope_hash,
                data_quality_passed=True,
                admissible=False,
                budget_before=budget,
                decided_at=executed_at,
            )
            abstention_count = 1
            steps.append(ControlledDynamicsStepReceiptV31.seal(
                step_index=step_index,
                action_kind="abstain",
                contract_before_hash=contract.contract_hash,
                contract_after_hash=contract.contract_hash,
                acquisition_receipts=acquisitions,
                permission=permission,
            ))
            break
        selected = ranked[0]
        selected_acquisition = by_hash[selected.action_hash]
        permission = _permission_v31(
            selected_acquisition.acquisition_hash,
            public.envelope.envelope_hash,
            data_quality_passed=True,
            admissible=True,
            budget_before=budget,
            decided_at=executed_at,
        )
        if permission.decision != "allow_synthetic":
            raise RuntimeError("V3.1 admissible action was not allowed")
        observation = private_case.action_observations[selected.action_id]
        observation.assert_sealed()
        if observation.empirical_peak_state_ratio > 1.0:
            raise RuntimeError("V3.1 hidden Reality Interface detected state-bound violation")
        observations.append(observation)
        selected_ids.append(selected.action_id)
        budget = permission.budget_after
        steps.append(ControlledDynamicsStepReceiptV31.seal(
            step_index=step_index,
            action_kind="controlled_experiment",
            contract_before_hash=contract.contract_hash,
            contract_after_hash=contract.contract_hash,
            acquisition_receipts=acquisitions,
            selected_action_id=selected.action_id,
            selected_action_hash=selected.action_hash,
            permission=permission,
            observation_hash=observation.observation_hash,
        ))

    if contract.target_status == "default_unverified" and "problem_layer" not in issue_routes:
        issue_routes.append("problem_layer")
    model: ControlledDriftModelV31 | None = None
    target_loss: float | None = None
    if data_quality_passed:
        model = _fit_model_v31(public, observations, spec)
        if model.normalized_derivative_residual > spec.model_mismatch_residual_threshold:
            issue_routes.append("model_layer")
        if private_case.performance_eligible:
            target_loss = _target_loss_v31(private_case, model, spec)
    return ControlledDynamicsCaseReceiptV31.seal(
        receipt_id=f"receipt_{policy.arm}_{public.case_id}",
        case_id=public.case_id,
        public_case_hash=public.public_hash,
        policy_hash=policy.policy_hash,
        arm=policy.arm,
        steps=steps,
        final_contract=contract,
        selected_action_ids=selected_ids,
        observation_hashes=[item.observation_hash for item in observations],
        final_model=model,
        issue_routes=issue_routes,
        action_budget_consumed=spec.action_budget - budget,
        abstention_count=abstention_count,
        target_loss=target_loss,
        executed_at=executed_at,
    )


def execute_controlled_dynamics_policy_v31(
    spec: ControlledDynamicsWorldPackSpecV31,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    policy: ControlledDynamicsPolicyV31,
    *,
    executed_at: datetime | None = None,
) -> ControlledDynamicsSelectionBundleV31:
    spec.assert_sealed()
    private_pack.assert_sealed()
    policy.assert_sealed()
    if private_pack.spec_hash != spec.spec_hash:
        raise ValueError("V3.1 private pack belongs to another protocol")
    expected = (
        spec.baseline_policy_hash
        if policy.arm == "random_bounded_inputs"
        else spec.candidate_policy_hash
    )
    if policy.policy_hash != expected:
        raise ValueError("V3.1 policy is not frozen in the protocol")
    at = executed_at or datetime.now(timezone.utc)
    receipts = [
        _execute_case_v31(spec, private_case, policy, executed_at=at)
        for private_case in private_pack.cases
    ]
    return ControlledDynamicsSelectionBundleV31.seal(
        spec_hash=spec.spec_hash,
        private_pack_hash=private_pack.pack_hash,
        policy_hash=policy.policy_hash,
        arm=policy.arm,
        case_receipts=receipts,
        total_action_budget_consumed=sum(item.action_budget_consumed for item in receipts),
        total_abstentions=sum(item.abstention_count for item in receipts),
    )


def _macro_bootstrap_v31(
    values_by_mechanism: dict[str, list[float]],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    random = np.random.default_rng(seed)
    draws = []
    groups = [np.asarray(values, dtype=float) for values in values_by_mechanism.values() if values]
    for _ in range(replicates):
        means = [
            float(np.mean(group[random.integers(0, len(group), size=len(group))]))
            for group in groups
        ]
        draws.append(float(np.mean(means)))
    low, high = np.quantile(np.asarray(draws), [0.025, 0.975])
    return float(low), float(high)


def _clopper_upper_v31(successes: int, trials: int, confidence: float = 0.95) -> float:
    if trials <= 0:
        return 1.0
    if successes == trials:
        return 1.0
    return float(beta.ppf(confidence, successes + 1, trials - successes))


def evaluate_controlled_dynamics_worldpack_v31(
    spec: ControlledDynamicsWorldPackSpecV31,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    baseline: ControlledDynamicsSelectionBundleV31,
    candidate: ControlledDynamicsSelectionBundleV31,
    *,
    evaluated_at: datetime | None = None,
) -> ControlledDynamicsReportV31:
    spec.assert_sealed()
    private_pack.assert_sealed()
    baseline.assert_sealed()
    candidate.assert_sealed()
    if baseline.spec_hash != spec.spec_hash or candidate.spec_hash != spec.spec_hash:
        raise ValueError("V3.1 bundles belong to another protocol")
    private_by_id = {case.public_case.case_id: case for case in private_pack.cases}
    baseline_by_id = {receipt.case_id: receipt for receipt in baseline.case_receipts}
    candidate_by_id = {receipt.case_id: receipt for receipt in candidate.case_receipts}
    if set(private_by_id) != set(baseline_by_id) or set(private_by_id) != set(candidate_by_id):
        raise ValueError("V3.1 evaluation case sets differ")
    results: list[ControlledDynamicsCaseResultV31] = []
    values_by_mechanism: dict[str, list[float]] = {item: [] for item in spec.mechanisms}
    data_gate_experiment_count = 0
    required_reformulations = 0
    completed_reformulations = 0
    spurious_reformulations = 0
    for case_id, private_case in private_by_id.items():
        base = baseline_by_id[case_id]
        cand = candidate_by_id[case_id]
        performance_eligible = private_case.performance_eligible
        improvement = None
        negative = None
        if performance_eligible:
            if base.target_loss is None or cand.target_loss is None:
                raise RuntimeError("eligible V3.1 case lacks target loss")
            improvement = base.target_loss - cand.target_loss
            negative = improvement < -spec.material_negative_transfer
            values_by_mechanism[private_case.mechanism].append(improvement)
        expected = set(private_case.expected_issue_routes)
        observed = set(cand.issue_routes)
        reformulated = cand.final_contract.version == 2
        if private_case.target_was_underspecified:
            required_reformulations += 1
            completed_reformulations += int(reformulated)
        elif reformulated:
            spurious_reformulations += 1
        if "data_layer" in expected:
            data_gate_experiment_count += len(cand.selected_action_ids)
        results.append(ControlledDynamicsCaseResultV31(
            case_id=case_id,
            mechanism=private_case.mechanism,
            seed=int(case_id.rsplit("_", 1)[1]),
            performance_eligible=performance_eligible,
            baseline_loss=base.target_loss,
            candidate_loss=cand.target_loss,
            absolute_loss_improvement=improvement,
            material_negative_transfer=negative,
            expected_issue_routes=private_case.expected_issue_routes,
            candidate_issue_routes=cand.issue_routes,
            routing_correct=(expected == observed),
            candidate_reformulated=reformulated,
            candidate_spurious_reformulation=(reformulated and not private_case.target_was_underspecified),
        ))
    mechanism_results = []
    for mechanism in spec.mechanisms:
        mechanism_cases = [item for item in results if item.mechanism == mechanism]
        values = values_by_mechanism[mechanism]
        mechanism_results.append(ControlledDynamicsMechanismResultV31(
            mechanism=mechanism,
            eligible_case_count=len(values),
            mean_absolute_loss_improvement=(float(np.mean(values)) if values else None),
            routing_accuracy=float(np.mean([item.routing_correct for item in mechanism_cases])),
        ))
    eligible_values = [
        item.absolute_loss_improvement
        for item in results
        if item.absolute_loss_improvement is not None
    ]
    macro = float(np.mean([
        np.mean(values) for values in values_by_mechanism.values() if values
    ]))
    ci_low, ci_high = _macro_bootstrap_v31(
        values_by_mechanism,
        replicates=spec.bootstrap_replicates,
        seed=spec.bootstrap_seed,
    )
    negatives = sum(item.material_negative_transfer is True for item in results)
    negative_rate = negatives / len(eligible_values)
    routing_accuracy = float(np.mean([item.routing_correct for item in results]))
    mechanism_gate = all(
        item.mean_absolute_loss_improvement is None
        or item.mean_absolute_loss_improvement >= -spec.maximum_mechanism_regression
        for item in mechanism_results
    )
    gates = {
        "macro_improvement_lower_bound": ci_low > spec.minimum_macro_loss_improvement,
        "mechanism_non_regression": mechanism_gate,
        "negative_transfer_upper_bound": _clopper_upper_v31(
            negatives, len(eligible_values)
        ) <= spec.maximum_negative_transfer_rate,
        "routing_accuracy": routing_accuracy >= spec.required_routing_accuracy,
        "required_reformulations_evidence_bound": (
            completed_reformulations == required_reformulations
        ),
        "zero_spurious_reformulations": spurious_reformulations == 0,
        "data_gate_prevented_experiments": data_gate_experiment_count == 0,
        "zero_invalid_actions": (
            baseline.invalid_action_count == 0 and candidate.invalid_action_count == 0
        ),
        "zero_actual_state_violations": (
            baseline.actual_state_violation_count == 0
            and candidate.actual_state_violation_count == 0
        ),
    }
    status = (
        "exploratory_only_v31"
        if spec.phase == "exploratory"
        else (
            "promoted_for_synthetic_controlled_epistemic_worldpack_v31"
            if all(gates.values())
            else "candidate_rejected_v31"
        )
    )
    return ControlledDynamicsReportV31.seal(
        report_id=f"controlled_dynamics_report_{spec.phase}_v31",
        phase=spec.phase,
        spec_hash=spec.spec_hash,
        private_pack_hash=private_pack.pack_hash,
        baseline_bundle_hash=baseline.bundle_hash,
        candidate_bundle_hash=candidate.bundle_hash,
        case_results=results,
        mechanism_results=mechanism_results,
        performance_eligible_case_count=len(eligible_values),
        excluded_data_issue_count=sum(
            "data_layer" in case.expected_issue_routes for case in private_pack.cases
        ),
        excluded_model_issue_count=sum(
            "model_layer" in case.expected_issue_routes for case in private_pack.cases
        ),
        macro_absolute_loss_improvement=macro,
        macro_improvement_ci_low=ci_low,
        macro_improvement_ci_high=ci_high,
        material_negative_transfer_count=negatives,
        material_negative_transfer_rate=negative_rate,
        material_negative_transfer_upper=_clopper_upper_v31(negatives, len(eligible_values)),
        routing_accuracy=routing_accuracy,
        required_reformulation_count=required_reformulations,
        evidence_bound_reformulation_count=completed_reformulations,
        spurious_reformulation_count=spurious_reformulations,
        data_gate_experiment_count=data_gate_experiment_count,
        gates=gates,
        status=status,
        evaluated_at=evaluated_at or datetime.now(timezone.utc),
    )


def qualify_controlled_dynamics_policy_v31(
    candidate_policy: ControlledDynamicsPolicyV31,
    report: ControlledDynamicsReportV31,
    *,
    qualified_at: datetime | None = None,
) -> ControlledDynamicsQualificationV31:
    candidate_policy.assert_sealed()
    report.assert_sealed()
    if report.status != "promoted_for_synthetic_controlled_epistemic_worldpack_v31":
        raise ValueError("cannot qualify a rejected V3.1 controlled-dynamics policy")
    return ControlledDynamicsQualificationV31.seal(
        policy_hash=candidate_policy.policy_hash,
        report_hash=report.report_hash,
        qualified_at=qualified_at or datetime.now(timezone.utc),
    )


def run_controlled_dynamics_worldpack_v31(
    output_root: str | Path,
    *,
    spec: ControlledDynamicsWorldPackSpecV31,
    baseline_policy: ControlledDynamicsPolicyV31,
    candidate_policy: ControlledDynamicsPolicyV31,
    evaluated_at: datetime | None = None,
    run_id: str | None = None,
) -> ControlledDynamicsOutcomeV31:
    spec.assert_sealed()
    assert_single_component_controlled_dynamics_v31(baseline_policy, candidate_policy)
    if baseline_policy.policy_hash != spec.baseline_policy_hash:
        raise ValueError("V3.1 baseline policy is not frozen in the protocol")
    if candidate_policy.policy_hash != spec.candidate_policy_hash:
        raise ValueError("V3.1 candidate policy is not frozen in the protocol")
    at = evaluated_at or datetime.now(timezone.utc)
    store = RunStore(
        output_root,
        run_id=run_id or f"controlled-dynamics-{uuid4().hex[:10]}",
    )
    refs = [
        store.put_artifact("controlled_dynamics_spec_v31", spec),
        store.put_artifact("controlled_dynamics_baseline_policy_v31", baseline_policy),
        store.put_artifact("controlled_dynamics_candidate_policy_v31", candidate_policy),
    ]
    store.emit("controlled_dynamics_protocol_frozen_before_private_pack", {
        "spec_hash": spec.spec_hash,
        "baseline_policy_hash": baseline_policy.policy_hash,
        "candidate_policy_hash": candidate_policy.policy_hash,
        "frozen_delta": spec.frozen_delta,
    })
    private_pack = generate_private_controlled_dynamics_worldpack_v31(spec, generated_at=at)
    baseline = execute_controlled_dynamics_policy_v31(
        spec, private_pack, baseline_policy, executed_at=at
    )
    candidate = execute_controlled_dynamics_policy_v31(
        spec, private_pack, candidate_policy, executed_at=at
    )
    report = evaluate_controlled_dynamics_worldpack_v31(
        spec, private_pack, baseline, candidate, evaluated_at=at
    )
    refs.extend([
        store.put_artifact("private_controlled_dynamics_worldpack_v31", private_pack),
        store.put_artifact("controlled_dynamics_baseline_bundle_v31", baseline),
        store.put_artifact("controlled_dynamics_candidate_bundle_v31", candidate),
        store.put_artifact("controlled_dynamics_report_v31", report),
    ])
    qualification = None
    if report.status == "promoted_for_synthetic_controlled_epistemic_worldpack_v31":
        qualification = qualify_controlled_dynamics_policy_v31(
            candidate_policy, report, qualified_at=at
        )
        refs.append(store.put_artifact("controlled_dynamics_qualification_v31", qualification))
    manifest = ControlledDynamicsManifestV31.seal(
        run_id=store.run_id,
        artifact_refs=refs,
        terminal_status=report.status,
        created_at=at,
    )
    manifest_ref = store.put_artifact("controlled_dynamics_manifest_v31", manifest)
    store.emit("controlled_dynamics_worldpack_adjudicated", {
        "manifest_ref": manifest_ref.model_dump(mode="json")
    })
    if not verify_controlled_dynamics_run_v31(store.run_directory):
        raise RuntimeError("V3.1 controlled-dynamics run failed independent verification")
    return ControlledDynamicsOutcomeV31(
        store, spec, private_pack, baseline_policy, candidate_policy,
        baseline, candidate, report, qualification, manifest,
    )


def verify_controlled_dynamics_run_v31(run_directory: str | Path) -> bool:
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
        for reference in committed:
            store.load_artifact(reference)
        manifest_refs = [
            reference for reference in committed
            if reference.kind == "controlled_dynamics_manifest_v31"
        ]
        if len(manifest_refs) != 1:
            return False
        manifest = ControlledDynamicsManifestV31.model_validate(
            store.load_artifact(manifest_refs[0])
        )
        manifest.assert_sealed()
        if manifest.run_id != store.run_id:
            return False

        def load_one(kind: str, model):
            references = [item for item in manifest.artifact_refs if item.kind == kind]
            if len(references) != 1:
                raise RuntimeError(f"V3.1 manifest needs exactly one {kind}")
            return model.model_validate(store.load_artifact(references[0]))

        spec = load_one("controlled_dynamics_spec_v31", ControlledDynamicsWorldPackSpecV31)
        baseline_policy = load_one(
            "controlled_dynamics_baseline_policy_v31", ControlledDynamicsPolicyV31
        )
        candidate_policy = load_one(
            "controlled_dynamics_candidate_policy_v31", ControlledDynamicsPolicyV31
        )
        private_pack = load_one(
            "private_controlled_dynamics_worldpack_v31", PrivateControlledDynamicsWorldPackV31
        )
        baseline = load_one(
            "controlled_dynamics_baseline_bundle_v31", ControlledDynamicsSelectionBundleV31
        )
        candidate = load_one(
            "controlled_dynamics_candidate_bundle_v31", ControlledDynamicsSelectionBundleV31
        )
        report = load_one("controlled_dynamics_report_v31", ControlledDynamicsReportV31)
        for artifact in (
            spec, baseline_policy, candidate_policy, private_pack,
            baseline, candidate, report, manifest,
        ):
            artifact.assert_sealed()
        regenerated = generate_private_controlled_dynamics_worldpack_v31(
            spec, generated_at=private_pack.generated_at
        )
        if regenerated.pack_hash != private_pack.pack_hash:
            return False
        executed_at = baseline.case_receipts[0].executed_at
        replay_baseline = execute_controlled_dynamics_policy_v31(
            spec, private_pack, baseline_policy, executed_at=executed_at
        )
        replay_candidate = execute_controlled_dynamics_policy_v31(
            spec, private_pack, candidate_policy, executed_at=executed_at
        )
        if replay_baseline.bundle_hash != baseline.bundle_hash:
            return False
        if replay_candidate.bundle_hash != candidate.bundle_hash:
            return False
        recomputed = evaluate_controlled_dynamics_worldpack_v31(
            spec, private_pack, baseline, candidate, evaluated_at=report.evaluated_at
        )
        if recomputed.report_hash != report.report_hash:
            return False
        qualifications = [
            item for item in manifest.artifact_refs
            if item.kind == "controlled_dynamics_qualification_v31"
        ]
        if report.status == "promoted_for_synthetic_controlled_epistemic_worldpack_v31":
            if len(qualifications) != 1:
                return False
            qualification = ControlledDynamicsQualificationV31.model_validate(
                store.load_artifact(qualifications[0])
            )
            qualification.assert_sealed()
            if qualification.report_hash != report.report_hash:
                return False
        elif qualifications:
            return False
        freeze_events = [
            event for event in events
            if event["event_type"] == "controlled_dynamics_protocol_frozen_before_private_pack"
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
        np.linalg.LinAlgError,
    ):
        return False
