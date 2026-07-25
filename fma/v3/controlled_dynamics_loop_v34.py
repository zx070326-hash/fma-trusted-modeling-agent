from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

import numpy as np
from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore
from fma.v2.dynamics_ir import trajectory_nrmse
from fma.v2.schemas import Identifier, Sha256, _assert_timezone

from .controlled_dynamics_loop import (
    MECHANISMS_V31,
    ControlledDriftModelV31,
    PrivateControlledDynamicsCaseV31,
    PrivateControlledDynamicsWorldPackV31,
    _fit_model_v31,
    _input_at_time_v31,
    _simulate_model_v31,
    _simulate_truth_v31,
    _state_peak_ratio_v31,
    _target_loss_v31,
    generate_private_controlled_dynamics_worldpack_v31,
)
from .controlled_dynamics_loop_v32 import _acquisition_receipts_v32
from .controlled_dynamics_loop_v332 import (
    PairedAdvantageTrustDecisionV332,
    _shared_random_order_v332,
    _trust_decision_v332,
)
from .experiment_ir import (
    ControlledObservationReceiptV31,
    DecisionTargetV31,
    PiecewiseConstantInputActionV31,
)


EXPLORATORY_SEEDS_V34 = (
    14009, 14057, 14107, 14153, 14207, 14251, 14303, 14347,
    14401, 14449, 14503, 14549, 14603, 14657, 14713, 14759,
)

AdapterArmV34 = Literal[
    "unguarded_full_action",
    "interruptible_online_guard",
]


class PlanAbstentionV34(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


def _switch_count(values: list[list[float]]) -> int:
    return sum(
        any(abs(a - b) > 1e-12 for a, b in zip(left, right, strict=True))
        for left, right in zip(values, values[1:], strict=False)
    )


def _energy(values: list[list[float]], segment_duration: float) -> float:
    return float(segment_duration * sum(value * value for row in values for value in row))


def _peak(values: list[list[float]]) -> float:
    return float(max((abs(value) for row in values for value in row), default=0.0))


class InterruptibleRealityPolicyV34(StrictModel):
    schema_version: Literal["3.4"] = "3.4"
    policy_id: Identifier
    arm: AdapterArmV34
    execution_rule: Literal[
        "execute_selected_action_without_online_interruption",
        "segment_authorization_then_monotone_zero_fallback",
    ]
    proposer_rule: Literal["v332_paired_advantage_unchanged"] = (
        "v332_paired_advantage_unchanged"
    )
    authority_may_only_decrease: Literal[True] = True
    real_world_execution_permitted: Literal[False] = False
    prior_v332_failure_report_hash: Sha256
    prior_epistemic_qualification_hash: Sha256
    prior_active_design_qualification_hash: Sha256
    method_evidence_hash: Sha256
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "InterruptibleRealityPolicyV34":
        expected = {
            "unguarded_full_action": (
                "execute_selected_action_without_online_interruption"
            ),
            "interruptible_online_guard": (
                "segment_authorization_then_monotone_zero_fallback"
            ),
        }[self.arm]
        if self.execution_rule != expected:
            raise ValueError("V3.4 arm and execution rule disagree")
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("policy_hash does not match V3.4 policy")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("V3.4 policy is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "InterruptibleRealityPolicyV34":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"policy_hash"}),
            policy_hash=draft.content_hash(),
        )


class ControlledDynamicsWorldPackSpecV34(StrictModel):
    schema_version: Literal["3.4"] = "3.4"
    experiment_id: Identifier
    phase: Literal["exploratory"] = "exploratory"
    mechanisms: list[Literal[
        "exponential_decay", "logistic_growth",
        "damped_oscillator", "duffing_oscillator",
    ]] = Field(min_length=4, max_length=4)
    seeds: list[int] = Field(min_length=16, max_length=16)
    action_budget: Literal[3] = 3
    clarification_budget: Literal[1] = 1
    maximum_steps: Literal[4] = 4
    anchor_experiment_count: Literal[2] = 2
    maximum_cross_excitation_nrmse: Literal[0.05] = 0.05
    minimum_paired_goal_risk_advantage: Literal[0.03] = 0.03
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
    bootstrap_replicates: Annotated[int, Field(ge=200, le=5000)] = 1600
    bootstrap_seed: Literal[34722] = 34722
    material_negative_transfer: Literal[0.02] = 0.02
    maximum_guard_negative_transfer_rate: Literal[0.1] = 0.1
    maximum_mechanism_regression: Literal[0.02] = 0.02
    minimum_interruption_count: Literal[1] = 1
    goal_initial_state_scales: list[Annotated[
        float, Field(gt=0, allow_inf_nan=False)
    ]] = Field(default_factory=lambda: [0.75, 1.0, 1.25], min_length=3, max_length=3)
    goal_zero_component_range_fraction: Literal[0.15] = 0.15
    goal_envelope_margin_fraction: Literal[0.05] = 0.05
    robust_goal_gain_quantile: Literal[0.2] = 0.2
    baseline_policy_hash: Sha256
    candidate_policy_hash: Sha256
    method_evidence_hash: Sha256
    prior_epistemic_qualification_hash: Sha256
    prior_active_design_qualification_hash: Sha256
    prior_v332_failure_report_hash: Sha256
    frozen_delta: Literal[
        "batch_reality_interface_to_segment_authorized_monotone_fallback_only"
    ] = "batch_reality_interface_to_segment_authorized_monotone_fallback_only"
    safety_claim: Literal["empirical_synthetic_proxy_not_formal_guarantee"] = (
        "empirical_synthetic_proxy_not_formal_guarantee"
    )
    real_world_execution_permitted: Literal[False] = False
    frozen_at: datetime
    spec_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "ControlledDynamicsWorldPackSpecV34":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.mechanisms != list(MECHANISMS_V31):
            raise ValueError("V3.4 requires the frozen mechanism order")
        if tuple(self.seeds) != EXPLORATORY_SEEDS_V34:
            raise ValueError("V3.4 seeds do not match the fresh exploratory set")
        if self.goal_initial_state_scales != [0.75, 1.0, 1.25]:
            raise ValueError("V3.4 public goal initial-state scales changed")
        if self.maximum_steps != self.action_budget + self.clarification_budget:
            raise ValueError("V3.4 resource budgets do not cover maximum steps")
        if not math.isclose(
            (self.trajectory_points - 1) * self.time_step,
            self.segment_count * self.segment_duration,
            abs_tol=1e-12,
        ):
            raise ValueError("V3.4 input segments do not cover the trajectory")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match V3.4 protocol")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "spec_hash")

    def assert_sealed(self) -> None:
        if not self.spec_hash or self.spec_hash != self.content_hash():
            raise ValueError("V3.4 protocol is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsWorldPackSpecV34":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"spec_hash"}),
            spec_hash=draft.content_hash(),
        )


class OnlineMismatchCalibrationReceiptV34(StrictModel):
    schema_version: Literal["3.4"] = "3.4"
    calibration_id: Identifier
    case_id: Identifier
    anchor_action_hashes: list[Sha256] = Field(min_length=2, max_length=2)
    anchor_observation_hashes: list[Sha256] = Field(min_length=2, max_length=2)
    leave_one_anchor_out_model_hashes: list[Sha256] = Field(min_length=2, max_length=2)
    segment_nrmse_values: list[Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ]] = Field(min_length=12, max_length=12)
    threshold_rule: Literal["maximum_leave_one_anchor_out_segment_nrmse"] = (
        "maximum_leave_one_anchor_out_segment_nrmse"
    )
    mismatch_threshold: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    calibrated_probability_claimed: Literal[False] = False
    hidden_probe_used: Literal[False] = False
    calibration_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_calibration(self) -> "OnlineMismatchCalibrationReceiptV34":
        if len(set(self.anchor_action_hashes)) != 2:
            raise ValueError("V3.4 calibration anchors must differ")
        if not math.isclose(
            self.mismatch_threshold,
            max(self.segment_nrmse_values),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("V3.4 mismatch threshold is not the frozen maximum")
        if self.calibration_hash and self.calibration_hash != self.content_hash():
            raise ValueError("calibration_hash does not match V3.4 calibration")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "calibration_hash")

    def assert_sealed(self) -> None:
        if not self.calibration_hash or self.calibration_hash != self.content_hash():
            raise ValueError("V3.4 calibration is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "OnlineMismatchCalibrationReceiptV34":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"calibration_hash"}),
            calibration_hash=draft.content_hash(),
        )


class SegmentAuthorizationReceiptV34(StrictModel):
    schema_version: Literal["3.4"] = "3.4"
    case_id: Identifier
    arm: AdapterArmV34
    segment_index: Annotated[int, Field(ge=1, le=6)]
    selected_mode: Literal["active", "prefrozen_fallback"]
    authority_before: Annotated[int, Field(ge=0, le=2)]
    authority_after: Annotated[int, Field(ge=0, le=2)]
    executed_input: list[Annotated[float, Field(allow_inf_nan=False)]] = Field(
        min_length=1, max_length=2
    )
    prediction_hash: Sha256
    segment_observation_hash: Sha256
    mismatch_nrmse: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    mismatch_threshold: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    clean_state_peak_ratio: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    decision_after: Literal[
        "continue_selected",
        "switch_to_zero_fallback",
        "continue_zero_fallback",
        "terminate_switch_budget",
        "terminate_state_envelope",
        "complete",
    ]
    cumulative_duration: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    cumulative_energy: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    cumulative_peak_amplitude: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    cumulative_switch_count: Annotated[int, Field(ge=0, le=3)]
    reescalation_permitted: Literal[False] = False
    real_world_action_executed: Literal[False] = False
    segment_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_segment(self) -> "SegmentAuthorizationReceiptV34":
        if self.authority_after > self.authority_before:
            raise ValueError("V3.4 segment authority cannot increase")
        if self.decision_after == "switch_to_zero_fallback" and not (
            self.authority_before == 2 and self.authority_after == 1
        ):
            raise ValueError("V3.4 zero fallback must reduce authority from two to one")
        if self.decision_after.startswith("terminate") and self.authority_after != 0:
            raise ValueError("V3.4 termination must remove authority")
        if self.segment_hash and self.segment_hash != self.content_hash():
            raise ValueError("segment_hash does not match V3.4 receipt")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "segment_hash")

    def assert_sealed(self) -> None:
        if not self.segment_hash or self.segment_hash != self.content_hash():
            raise ValueError("V3.4 segment receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "SegmentAuthorizationReceiptV34":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"segment_hash"}),
            segment_hash=draft.content_hash(),
        )


class ExecutedInterventionV34(StrictModel):
    schema_version: Literal["3.4"] = "3.4"
    intervention_id: Identifier
    case_id: Identifier
    selected_action_hash: Sha256
    selected_mode: Literal["active", "prefrozen_fallback"]
    segment_duration: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    input_values: list[list[Annotated[float, Field(allow_inf_nan=False)]]] = Field(
        min_length=1, max_length=6
    )
    executed_duration: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    input_energy: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    peak_amplitude: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    switch_count: Annotated[int, Field(ge=0, le=3)]
    interrupted: bool
    terminated: bool
    interruption_after_segment: Annotated[int, Field(ge=1, le=6)] | None = None
    final_mode: Literal["selected", "zero_fallback", "terminated"]
    real_world_action_executed: Literal[False] = False
    intervention_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_intervention(self) -> "ExecutedInterventionV34":
        width = len(self.input_values[0])
        if any(len(row) != width for row in self.input_values):
            raise ValueError("V3.4 intervention input dimensions differ")
        if not math.isclose(
            self.executed_duration,
            len(self.input_values) * self.segment_duration,
            abs_tol=1e-12,
        ):
            raise ValueError("V3.4 intervention duration does not match segments")
        if not math.isclose(
            self.input_energy,
            _energy(self.input_values, self.segment_duration),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("V3.4 intervention energy does not match inputs")
        if not math.isclose(self.peak_amplitude, _peak(self.input_values), abs_tol=1e-12):
            raise ValueError("V3.4 intervention peak does not match inputs")
        if self.switch_count != _switch_count(self.input_values):
            raise ValueError("V3.4 intervention switch count does not match inputs")
        if self.interrupted != (self.interruption_after_segment is not None):
            raise ValueError("V3.4 interruption fields disagree")
        if self.terminated != (self.final_mode == "terminated"):
            raise ValueError("V3.4 termination fields disagree")
        if self.intervention_hash and self.intervention_hash != self.content_hash():
            raise ValueError("intervention_hash does not match V3.4 intervention")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "intervention_hash")

    def assert_sealed(self) -> None:
        if not self.intervention_hash or self.intervention_hash != self.content_hash():
            raise ValueError("V3.4 intervention is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ExecutedInterventionV34":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"intervention_hash"}),
            intervention_hash=draft.content_hash(),
        )


class InterventionExposureLedgerV34(StrictModel):
    schema_version: Literal["3.4"] = "3.4"
    case_id: Identifier
    maximum_duration: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    maximum_energy: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    maximum_peak_amplitude: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    maximum_switch_count: Annotated[int, Field(ge=1)]
    used_duration: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    used_energy: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    used_peak_amplitude: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    used_switch_count: Annotated[int, Field(ge=0)]
    executed_segment_count: Annotated[int, Field(ge=1, le=6)]
    maximum_clean_state_peak_ratio: Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ]
    state_envelope_violation_count: Annotated[int, Field(ge=0)]
    within_prefrozen_exposure_envelope: bool
    real_world_exposure_claimed: Literal[False] = False
    ledger_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_ledger(self) -> "InterventionExposureLedgerV34":
        within = (
            self.used_duration <= self.maximum_duration + 1e-12
            and self.used_energy <= self.maximum_energy + 1e-12
            and self.used_peak_amplitude <= self.maximum_peak_amplitude + 1e-12
            and self.used_switch_count <= self.maximum_switch_count
        )
        if self.within_prefrozen_exposure_envelope != within:
            raise ValueError("V3.4 exposure envelope flag disagrees with ledger")
        if self.ledger_hash and self.ledger_hash != self.content_hash():
            raise ValueError("ledger_hash does not match V3.4 exposure ledger")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "ledger_hash")

    def assert_sealed(self) -> None:
        if not self.ledger_hash or self.ledger_hash != self.content_hash():
            raise ValueError("V3.4 exposure ledger is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "InterventionExposureLedgerV34":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"ledger_hash"}),
            ledger_hash=draft.content_hash(),
        )


class ControlledDynamicsCaseReceiptV34(StrictModel):
    schema_version: Literal["3.4"] = "3.4"
    receipt_id: Identifier
    case_id: Identifier
    public_case_hash: Sha256
    policy_hash: Sha256
    arm: AdapterArmV34
    data_quality_passed: bool
    plan_admissible: bool
    abstention_reason: Literal[
        "pilot_data_quality",
        "no_admissible_shared_anchor",
        "no_admissible_third_action",
    ] | None = None
    clarification_used: bool
    decision_target: DecisionTargetV31
    anchor_action_ids: list[Identifier] = Field(max_length=2)
    anchor_action_hashes: list[Sha256] = Field(max_length=2)
    anchor_observation_hashes: list[Sha256] = Field(max_length=2)
    trust_decision: PairedAdvantageTrustDecisionV332 | None = None
    calibration: OnlineMismatchCalibrationReceiptV34 | None = None
    selected_action_hash: Sha256 | None = None
    selected_mode: Literal["active", "prefrozen_fallback"] | None = None
    noise_schedule_hash: Sha256 | None = None
    segment_receipts: list[SegmentAuthorizationReceiptV34] = Field(max_length=6)
    executed_intervention: ExecutedInterventionV34 | None = None
    observation: ControlledObservationReceiptV31 | None = None
    exposure_ledger: InterventionExposureLedgerV34 | None = None
    final_model: ControlledDriftModelV31 | None = None
    target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    performance_eligible: bool
    hidden_parameters_exposed_to_policy: Literal[False] = False
    real_world_execution_permitted: Literal[False] = False
    executed_at: datetime
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "ControlledDynamicsCaseReceiptV34":
        _assert_timezone(self.executed_at, "executed_at")
        complete_fields = (
            self.trust_decision,
            self.calibration,
            self.selected_action_hash,
            self.selected_mode,
            self.noise_schedule_hash,
            self.executed_intervention,
            self.observation,
            self.exposure_ledger,
            self.final_model,
        )
        completed = self.data_quality_passed and self.plan_admissible
        if completed:
            if self.abstention_reason is not None:
                raise ValueError("V3.4 completed case cannot carry abstention reason")
            if len(self.anchor_action_ids) != 2 or len(self.anchor_action_hashes) != 2:
                raise ValueError("V3.4 completed case needs two anchors")
            if len(self.anchor_observation_hashes) != 2 or any(
                item is None for item in complete_fields
            ):
                raise ValueError("V3.4 completed case is missing evidence")
            assert self.executed_intervention is not None
            assert self.observation is not None
            assert self.exposure_ledger is not None
            if len(self.segment_receipts) != len(self.executed_intervention.input_values):
                raise ValueError("V3.4 segment receipts do not cover intervention")
            if self.observation.action_hash != self.executed_intervention.intervention_hash:
                raise ValueError("V3.4 observation is not bound to intervention")
            if self.exposure_ledger.executed_segment_count != len(self.segment_receipts):
                raise ValueError("V3.4 exposure ledger does not cover segments")
            if self.performance_eligible != (self.target_loss is not None):
                raise ValueError("V3.4 eligible case must carry target loss")
        else:
            if self.abstention_reason is None:
                raise ValueError("V3.4 incomplete case needs an abstention reason")
            if (
                not self.data_quality_passed
                and self.abstention_reason != "pilot_data_quality"
            ):
                raise ValueError("V3.4 data-gated case has wrong abstention reason")
            if any(item is not None for item in complete_fields) or self.segment_receipts:
                raise ValueError("V3.4 abstained case cannot execute an experiment")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match V3.4 case receipt")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("V3.4 case receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsCaseReceiptV34":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


class ControlledDynamicsSelectionBundleV34(StrictModel):
    schema_version: Literal["3.4"] = "3.4"
    spec_hash: Sha256
    private_pack_hash: Sha256
    policy_hash: Sha256
    arm: AdapterArmV34
    case_receipts: list[ControlledDynamicsCaseReceiptV34] = Field(
        min_length=64, max_length=64
    )
    bundle_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bundle(self) -> "ControlledDynamicsSelectionBundleV34":
        if len({item.case_id for item in self.case_receipts}) != len(self.case_receipts):
            raise ValueError("V3.4 case ids must be unique")
        if any(item.arm != self.arm for item in self.case_receipts):
            raise ValueError("V3.4 bundle mixes adapter arms")
        if any(item.policy_hash != self.policy_hash for item in self.case_receipts):
            raise ValueError("V3.4 bundle mixes policies")
        if self.bundle_hash and self.bundle_hash != self.content_hash():
            raise ValueError("bundle_hash does not match V3.4 bundle")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "bundle_hash")

    def assert_sealed(self) -> None:
        if not self.bundle_hash or self.bundle_hash != self.content_hash():
            raise ValueError("V3.4 bundle is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsSelectionBundleV34":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"bundle_hash"}),
            bundle_hash=draft.content_hash(),
        )


class InterruptibleRealityEvolutionReportV34(StrictModel):
    schema_version: Literal["3.4"] = "3.4"
    evolution_id: Identifier
    spec_hash: Sha256
    prior_v332_failure_report_hash: Sha256
    single_component_delta: Literal[
        "batch_reality_interface_to_segment_authorized_monotone_fallback_only"
    ] = "batch_reality_interface_to_segment_authorized_monotone_fallback_only"
    eligible_case_count: Annotated[int, Field(ge=1)]
    active_proposal_count: Annotated[int, Field(ge=0)]
    prefrozen_fallback_count: Annotated[int, Field(ge=0)]
    interruption_count: Annotated[int, Field(ge=0)]
    termination_count: Annotated[int, Field(ge=0)]
    paired_macro_improvement: Annotated[float, Field(allow_inf_nan=False)]
    bootstrap_ci_low: Annotated[float, Field(allow_inf_nan=False)]
    bootstrap_ci_high: Annotated[float, Field(allow_inf_nan=False)]
    mechanism_mean_improvements: dict[Identifier, Annotated[
        float, Field(allow_inf_nan=False)
    ]]
    material_negative_transfer_count: Annotated[int, Field(ge=0)]
    material_negative_transfer_rate: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ]
    baseline_max_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    candidate_max_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    gates: dict[Identifier, bool]
    adapter_candidate_ready: bool
    status: Literal[
        "interaction_adapter_ready_for_acquisition_retest_v34",
        "interaction_adapter_failed_v34",
    ]
    proposer_changed: Literal[False] = False
    estimator_changed: Literal[False] = False
    target_evaluator_changed: Literal[False] = False
    reality_adapter_changed: Literal[True] = True
    router_changed: Literal[False] = False
    overall_qualification_permitted: Literal[False] = False
    confirmation_permitted: Literal[False] = False
    real_world_authorization_permitted: Literal[False] = False
    created_at: datetime
    evolution_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "InterruptibleRealityEvolutionReportV34":
        _assert_timezone(self.created_at, "created_at")
        ready = all(self.gates.values())
        if self.adapter_candidate_ready != ready:
            raise ValueError("V3.4 readiness disagrees with gates")
        expected = (
            "interaction_adapter_ready_for_acquisition_retest_v34"
            if ready else "interaction_adapter_failed_v34"
        )
        if self.status != expected:
            raise ValueError("V3.4 status disagrees with gates")
        if self.evolution_hash and self.evolution_hash != self.content_hash():
            raise ValueError("evolution_hash does not match V3.4 report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evolution_hash")

    def assert_sealed(self) -> None:
        if not self.evolution_hash or self.evolution_hash != self.content_hash():
            raise ValueError("V3.4 report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "InterruptibleRealityEvolutionReportV34":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evolution_hash"}),
            evolution_hash=draft.content_hash(),
        )


class ControlledDynamicsManifestV34(StrictModel):
    schema_version: Literal["3.4"] = "3.4"
    run_id: Identifier
    artifact_refs: list[ArtifactRef] = Field(min_length=6)
    terminal_status: Literal[
        "interaction_adapter_ready_for_acquisition_retest_v34",
        "interaction_adapter_failed_v34",
    ]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "ControlledDynamicsManifestV34":
        _assert_timezone(self.created_at, "created_at")
        if len({item.kind for item in self.artifact_refs}) != len(self.artifact_refs):
            raise ValueError("V3.4 manifest artifact kinds must be unique")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match V3.4 manifest")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("V3.4 manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsManifestV34":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class ControlledDynamicsOutcomeV34:
    store: RunStore
    spec: ControlledDynamicsWorldPackSpecV34
    private_pack: PrivateControlledDynamicsWorldPackV31
    baseline_policy: InterruptibleRealityPolicyV34
    candidate_policy: InterruptibleRealityPolicyV34
    baseline_bundle: ControlledDynamicsSelectionBundleV34
    candidate_bundle: ControlledDynamicsSelectionBundleV34
    evolution_report: InterruptibleRealityEvolutionReportV34
    manifest: ControlledDynamicsManifestV34


def default_controlled_dynamics_policies_v34(
    *,
    prior_v332_failure_report_hash: str,
    prior_epistemic_qualification_hash: str,
    prior_active_design_qualification_hash: str,
    method_evidence_hash: str,
) -> tuple[InterruptibleRealityPolicyV34, InterruptibleRealityPolicyV34]:
    shared = dict(
        prior_v332_failure_report_hash=prior_v332_failure_report_hash,
        prior_epistemic_qualification_hash=prior_epistemic_qualification_hash,
        prior_active_design_qualification_hash=prior_active_design_qualification_hash,
        method_evidence_hash=method_evidence_hash,
    )
    return (
        InterruptibleRealityPolicyV34.seal(
            policy_id="unguarded_paired_advantage_v34",
            arm="unguarded_full_action",
            execution_rule="execute_selected_action_without_online_interruption",
            **shared,
        ),
        InterruptibleRealityPolicyV34.seal(
            policy_id="interruptible_paired_advantage_v34",
            arm="interruptible_online_guard",
            execution_rule="segment_authorization_then_monotone_zero_fallback",
            **shared,
        ),
    )


def default_controlled_dynamics_exploratory_spec_v34(
    *,
    baseline_policy_hash: str,
    candidate_policy_hash: str,
    method_evidence_hash: str,
    prior_epistemic_qualification_hash: str,
    prior_active_design_qualification_hash: str,
    prior_v332_failure_report_hash: str,
    frozen_at: datetime | None = None,
) -> ControlledDynamicsWorldPackSpecV34:
    return ControlledDynamicsWorldPackSpecV34.seal(
        experiment_id="controlled_dynamics_interruptible_reality_exploratory_v34",
        mechanisms=list(MECHANISMS_V31),
        seeds=list(EXPLORATORY_SEEDS_V34),
        baseline_policy_hash=baseline_policy_hash,
        candidate_policy_hash=candidate_policy_hash,
        method_evidence_hash=method_evidence_hash,
        prior_epistemic_qualification_hash=prior_epistemic_qualification_hash,
        prior_active_design_qualification_hash=prior_active_design_qualification_hash,
        prior_v332_failure_report_hash=prior_v332_failure_report_hash,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


def _segment_prediction_errors_v34(
    spec: ControlledDynamicsWorldPackSpecV34,
    private_case: PrivateControlledDynamicsCaseV31,
    action: PiecewiseConstantInputActionV31,
    observation: ControlledObservationReceiptV31,
    model: ControlledDriftModelV31,
) -> list[float]:
    public = private_case.public_case
    points = int(round(spec.segment_duration / spec.time_step)) + 1
    local_times = [index * spec.time_step for index in range(points)]
    errors: list[float] = []
    for segment_index in range(spec.segment_count):
        start = segment_index * (points - 1)
        actual = observation.states[start:start + points]
        try:
            predicted = _simulate_model_v31(
                model,
                public.actuator,
                actual[0],
                local_times,
                [action.input_values[segment_index]],
                spec.segment_duration,
            )
            errors.append(float(trajectory_nrmse(actual, predicted)))
        except RuntimeError:
            errors.append(10.0)
    return errors


def _calibration_v34(
    spec: ControlledDynamicsWorldPackSpecV34,
    private_case: PrivateControlledDynamicsCaseV31,
    anchor_actions: list[PiecewiseConstantInputActionV31],
    anchor_observations: list[ControlledObservationReceiptV31],
) -> OnlineMismatchCalibrationReceiptV34:
    public = private_case.public_case
    errors: list[float] = []
    model_hashes: list[str] = []
    for held_out in (0, 1):
        model = _fit_model_v31(public, [anchor_observations[1 - held_out]], spec)
        model_hashes.append(model.model_hash)
        errors.extend(_segment_prediction_errors_v34(
            spec,
            private_case,
            anchor_actions[held_out],
            anchor_observations[held_out],
            model,
        ))
    return OnlineMismatchCalibrationReceiptV34.seal(
        calibration_id=f"online_mismatch_calibration_{public.case_id}",
        case_id=public.case_id,
        anchor_action_hashes=[item.action_hash for item in anchor_actions],
        anchor_observation_hashes=[item.observation_hash for item in anchor_observations],
        leave_one_anchor_out_model_hashes=model_hashes,
        segment_nrmse_values=errors,
        mismatch_threshold=max(errors),
    )


def _select_v332_plan_v34(
    spec: ControlledDynamicsWorldPackSpecV34,
    private_case: PrivateControlledDynamicsCaseV31,
) -> tuple[
    DecisionTargetV31,
    list[PiecewiseConstantInputActionV31],
    list[ControlledObservationReceiptV31],
    PiecewiseConstantInputActionV31,
    PiecewiseConstantInputActionV31,
    PairedAdvantageTrustDecisionV332,
]:
    public = private_case.public_case
    target = (
        private_case.true_decision_target
        if public.initial_contract.target_status == "default_unverified"
        else public.initial_contract.decision_target
    )
    step_offset = 1 if public.initial_contract.target_status == "default_unverified" else 0
    shared_order = _shared_random_order_v332(spec, public)
    selected_ids: list[str] = []
    observations: list[ControlledObservationReceiptV31] = []
    anchor_actions: list[PiecewiseConstantInputActionV31] = []
    for anchor_index in range(spec.anchor_experiment_count):
        step_index = step_offset + anchor_index + 1
        available = [
            action for action in public.action_catalog
            if action.action_id not in selected_ids
        ]
        acquisitions = _acquisition_receipts_v32(
            spec, public, observations, available, target, step_index
        )
        by_hash = {item.action_hash: item for item in acquisitions}
        admissible = [
            action for action in available if by_hash[action.action_hash].admissible
        ]
        if not admissible:
            raise PlanAbstentionV34("no_admissible_shared_anchor")
        selected = next(
            action for action_id in shared_order
            for action in admissible if action.action_id == action_id
        )
        observation = private_case.action_observations[selected.action_id]
        selected_ids.append(selected.action_id)
        anchor_actions.append(selected)
        observations.append(observation)

    step_index = step_offset + spec.anchor_experiment_count + 1
    available = [
        action for action in public.action_catalog if action.action_id not in selected_ids
    ]
    acquisitions = _acquisition_receipts_v32(
        spec, public, observations, available, target, step_index
    )
    by_hash = {item.action_hash: item for item in acquisitions}
    admissible = [
        action for action in available if by_hash[action.action_hash].admissible
    ]
    if not admissible:
        raise PlanAbstentionV34("no_admissible_third_action")
    fallback = next(
        action for action_id in shared_order
        for action in admissible if action.action_id == action_id
    )
    active = max(
        admissible,
        key=lambda item: (by_hash[item.action_hash].ranking_score, item.action_id),
    )
    trust = _trust_decision_v332(
        spec,
        public,
        observations,
        selected_ids,
        acquisitions,
        active,
        fallback,
        target,
        step_index,
    )
    return target, anchor_actions, observations, active, fallback, trust


def _noise_schedule_v34(
    spec: ControlledDynamicsWorldPackSpecV34,
    private_case: PrivateControlledDynamicsCaseV31,
) -> tuple[np.ndarray, str]:
    public = private_case.public_case
    seed = int(sha256_value([
        spec.spec_hash,
        public.case_id,
        "common_measurement_noise_schedule_v34",
    ])[:16], 16)
    random = np.random.default_rng(seed)
    innovations = random.normal(
        0.0,
        1.0,
        size=(spec.trajectory_points, len(public.state_names)),
    )
    return innovations, sha256_value({
        "seed": seed,
        "shape": list(innovations.shape),
        "innovations": innovations.tolist(),
    })


def _stream_intervention_v34(
    spec: ControlledDynamicsWorldPackSpecV34,
    private_case: PrivateControlledDynamicsCaseV31,
    policy: InterruptibleRealityPolicyV34,
    selected_action: PiecewiseConstantInputActionV31,
    selected_mode: Literal["active", "prefrozen_fallback"],
    online_model: ControlledDriftModelV31,
    calibration: OnlineMismatchCalibrationReceiptV34,
    *,
    observed_at: datetime,
) -> tuple[
    ExecutedInterventionV34,
    ControlledObservationReceiptV31,
    InterventionExposureLedgerV34,
    list[SegmentAuthorizationReceiptV34],
    str,
]:
    public = private_case.public_case
    points = int(round(spec.segment_duration / spec.time_step)) + 1
    local_times = [index * spec.time_step for index in range(points)]
    innovations, noise_hash = _noise_schedule_v34(spec, private_case)
    pilot_scale = np.maximum(np.std(np.asarray(public.pilot.states), axis=0), 0.1)
    noise_scale = spec.observation_noise_fraction * pilot_scale
    clean_state = list(public.initial_state)
    clean_paths: list[np.ndarray] = []
    observed_paths: list[np.ndarray] = []
    input_values: list[list[float]] = []
    receipts: list[SegmentAuthorizationReceiptV34] = []
    zero_mode = False
    terminated = False
    interruption_after: int | None = None
    max_clean_ratio = 0.0
    violation_count = 0
    consecutive_exceedances = 0
    required_exceedances = int(
        getattr(policy, "consecutive_exceedances_required", 1)
    )
    if required_exceedances < 1:
        raise ValueError("V3.4 mismatch confirmation count must be positive")

    for segment_zero_index in range(spec.segment_count):
        segment_index = segment_zero_index + 1
        authority_before = 1 if zero_mode else (2 if selected_mode == "active" else 1)
        control = [0.0] if zero_mode else list(selected_action.input_values[segment_zero_index])
        clean = np.asarray(_simulate_truth_v31(
            private_case.mechanism,
            clean_state,
            local_times,
            private_case.hidden_parameters,
            [control],
            public.actuator.matrix,
            spec.segment_duration,
        ))
        global_start = segment_zero_index * (points - 1)
        observed = clean + innovations[global_start:global_start + points] * noise_scale
        try:
            predicted = _simulate_model_v31(
                online_model,
                public.actuator,
                observed[0].tolist(),
                local_times,
                [control],
                spec.segment_duration,
            )
            mismatch = float(trajectory_nrmse(observed.tolist(), predicted))
        except RuntimeError:
            predicted = [[1e5 for _ in public.state_names] for _ in local_times]
            mismatch = 10.0
        segment_peak = _state_peak_ratio_v31(
            clean.tolist(),
            public.envelope.state_lower_bounds,
            public.envelope.state_upper_bounds,
        )
        max_clean_ratio = max(max_clean_ratio, segment_peak)
        if segment_peak > 1.0:
            violation_count += 1
        input_values.append(control)
        clean_paths.append(clean)
        observed_paths.append(observed)
        clean_state = clean[-1].tolist()
        cumulative_switches = _switch_count(input_values)
        at_end = segment_index == spec.segment_count
        decision: str
        authority_after = authority_before

        if segment_peak > 1.0:
            decision = "terminate_state_envelope"
            authority_after = 0
            terminated = True
            if interruption_after is None:
                interruption_after = segment_index
        elif policy.arm == "interruptible_online_guard" and (
            selected_mode == "active" and not zero_mode
        ):
            consecutive_exceedances = (
                consecutive_exceedances + 1
                if mismatch > calibration.mismatch_threshold else 0
            )
            confirmed_mismatch = (
                consecutive_exceedances >= required_exceedances and not at_end
            )
            if not confirmed_mismatch:
                decision = "complete" if at_end else "continue_selected"
            elif cumulative_switches + 1 > public.envelope.required_switch_count:
                decision = "terminate_switch_budget"
                authority_after = 0
                terminated = True
                interruption_after = segment_index
            else:
                decision = "switch_to_zero_fallback"
                authority_after = 1
                zero_mode = True
                interruption_after = segment_index
        elif at_end:
            decision = "complete"
        elif zero_mode:
            decision = "continue_zero_fallback"
        else:
            decision = "continue_selected"

        segment_path_hash = sha256_value({
            "case_id": public.case_id,
            "segment_index": segment_index,
            "times": local_times,
            "states": observed.tolist(),
            "input": control,
        })
        receipts.append(SegmentAuthorizationReceiptV34.seal(
            case_id=public.case_id,
            arm=policy.arm,
            segment_index=segment_index,
            selected_mode=selected_mode,
            authority_before=authority_before,
            authority_after=authority_after,
            executed_input=control,
            prediction_hash=sha256_value(predicted),
            segment_observation_hash=segment_path_hash,
            mismatch_nrmse=mismatch,
            mismatch_threshold=calibration.mismatch_threshold,
            clean_state_peak_ratio=segment_peak,
            decision_after=decision,
            cumulative_duration=segment_index * spec.segment_duration,
            cumulative_energy=_energy(input_values, spec.segment_duration),
            cumulative_peak_amplitude=_peak(input_values),
            cumulative_switch_count=cumulative_switches,
        ))
        if terminated:
            break

    if interruption_after is None:
        final_mode: Literal["selected", "zero_fallback", "terminated"] = "selected"
    elif terminated:
        final_mode = "terminated"
    else:
        final_mode = "zero_fallback"
    intervention = ExecutedInterventionV34.seal(
        intervention_id=f"intervention_{policy.arm}_{public.case_id}",
        case_id=public.case_id,
        selected_action_hash=selected_action.action_hash,
        selected_mode=selected_mode,
        segment_duration=spec.segment_duration,
        input_values=input_values,
        executed_duration=len(input_values) * spec.segment_duration,
        input_energy=_energy(input_values, spec.segment_duration),
        peak_amplitude=_peak(input_values),
        switch_count=_switch_count(input_values),
        interrupted=interruption_after is not None,
        terminated=terminated,
        interruption_after_segment=interruption_after,
        final_mode=final_mode,
    )
    joined_clean = np.vstack([
        clean_paths[0],
        *[path[1:] for path in clean_paths[1:]],
    ])
    joined_observed = np.vstack([
        observed_paths[0],
        *[path[1:] for path in observed_paths[1:]],
    ])
    times = [index * spec.time_step for index in range(len(joined_observed))]
    inputs = [
        _input_at_time_v31(input_values, time, spec.segment_duration).tolist()
        for time in times
    ]
    observation = ControlledObservationReceiptV31.seal(
        observation_id=f"streaming_observation_{policy.arm}_{public.case_id}",
        case_id=public.case_id,
        action_hash=intervention.intervention_hash,
        actuator_hash=public.actuator.actuator_hash,
        times=times,
        states=joined_observed.tolist(),
        inputs=inputs,
        empirical_peak_state_ratio=_state_peak_ratio_v31(
            joined_clean.tolist(),
            public.envelope.state_lower_bounds,
            public.envelope.state_upper_bounds,
        ),
        quality_flags=(
            ["runtime_guard_interrupted"] if intervention.interrupted else []
        ),
        observed_at=observed_at,
    )
    maximum_duration = spec.segment_count * spec.segment_duration
    ledger = InterventionExposureLedgerV34.seal(
        case_id=public.case_id,
        maximum_duration=maximum_duration,
        maximum_energy=public.envelope.required_total_energy,
        maximum_peak_amplitude=public.envelope.required_peak_amplitude,
        maximum_switch_count=public.envelope.required_switch_count,
        used_duration=intervention.executed_duration,
        used_energy=intervention.input_energy,
        used_peak_amplitude=intervention.peak_amplitude,
        used_switch_count=intervention.switch_count,
        executed_segment_count=len(input_values),
        maximum_clean_state_peak_ratio=max_clean_ratio,
        state_envelope_violation_count=violation_count,
        within_prefrozen_exposure_envelope=(
            intervention.executed_duration <= maximum_duration + 1e-12
            and intervention.input_energy
            <= public.envelope.required_total_energy + 1e-12
            and intervention.peak_amplitude
            <= public.envelope.required_peak_amplitude + 1e-12
            and intervention.switch_count <= public.envelope.required_switch_count
        ),
    )
    return intervention, observation, ledger, receipts, noise_hash


def _execute_case_v34(
    spec: ControlledDynamicsWorldPackSpecV34,
    private_case: PrivateControlledDynamicsCaseV31,
    policy: InterruptibleRealityPolicyV34,
    *,
    executed_at: datetime,
) -> ControlledDynamicsCaseReceiptV34:
    public = private_case.public_case
    public.assert_sealed()
    target = (
        private_case.true_decision_target
        if public.initial_contract.target_status == "default_unverified"
        else public.initial_contract.decision_target
    )
    clarification_used = public.initial_contract.target_status == "default_unverified"
    if public.pilot.quality_flags:
        return ControlledDynamicsCaseReceiptV34.seal(
            receipt_id=f"receipt_{policy.arm}_{public.case_id}_v34",
            case_id=public.case_id,
            public_case_hash=public.public_hash,
            policy_hash=policy.policy_hash,
            arm=policy.arm,
            data_quality_passed=False,
            plan_admissible=False,
            abstention_reason="pilot_data_quality",
            clarification_used=clarification_used,
            decision_target=target,
            anchor_action_ids=[],
            anchor_action_hashes=[],
            anchor_observation_hashes=[],
            segment_receipts=[],
            performance_eligible=False,
            executed_at=executed_at,
        )

    try:
        target, anchors, anchor_observations, active, fallback, trust = (
            _select_v332_plan_v34(spec, private_case)
        )
    except PlanAbstentionV34 as exc:
        return ControlledDynamicsCaseReceiptV34.seal(
            receipt_id=f"receipt_{policy.arm}_{public.case_id}_v34",
            case_id=public.case_id,
            public_case_hash=public.public_hash,
            policy_hash=policy.policy_hash,
            arm=policy.arm,
            data_quality_passed=True,
            plan_admissible=False,
            abstention_reason=exc.reason,
            clarification_used=clarification_used,
            decision_target=target,
            anchor_action_ids=[],
            anchor_action_hashes=[],
            anchor_observation_hashes=[],
            segment_receipts=[],
            performance_eligible=False,
            executed_at=executed_at,
        )
    calibration = _calibration_v34(
        spec, private_case, anchors, anchor_observations
    )
    selected_mode: Literal["active", "prefrozen_fallback"] = (
        "active" if trust.decision == "use_goal_risk" else "prefrozen_fallback"
    )
    selected = active if selected_mode == "active" else fallback
    online_model = _fit_model_v31(public, anchor_observations, spec)
    intervention, observation, ledger, segment_receipts, noise_hash = (
        _stream_intervention_v34(
            spec,
            private_case,
            policy,
            selected,
            selected_mode,
            online_model,
            calibration,
            observed_at=executed_at,
        )
    )
    final_model = _fit_model_v31(
        public, [*anchor_observations, observation], spec
    )
    target_loss = (
        _target_loss_v31(private_case, final_model, spec)
        if private_case.performance_eligible else None
    )
    return ControlledDynamicsCaseReceiptV34.seal(
        receipt_id=f"receipt_{policy.arm}_{public.case_id}_v34",
        case_id=public.case_id,
        public_case_hash=public.public_hash,
        policy_hash=policy.policy_hash,
        arm=policy.arm,
        data_quality_passed=True,
        plan_admissible=True,
        clarification_used=clarification_used,
        decision_target=target,
        anchor_action_ids=[item.action_id for item in anchors],
        anchor_action_hashes=[item.action_hash for item in anchors],
        anchor_observation_hashes=[item.observation_hash for item in anchor_observations],
        trust_decision=trust,
        calibration=calibration,
        selected_action_hash=selected.action_hash,
        selected_mode=selected_mode,
        noise_schedule_hash=noise_hash,
        segment_receipts=segment_receipts,
        executed_intervention=intervention,
        observation=observation,
        exposure_ledger=ledger,
        final_model=final_model,
        target_loss=target_loss,
        performance_eligible=private_case.performance_eligible,
        executed_at=executed_at,
    )


def execute_controlled_dynamics_policy_v34(
    spec: ControlledDynamicsWorldPackSpecV34,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    policy: InterruptibleRealityPolicyV34,
    *,
    executed_at: datetime | None = None,
) -> ControlledDynamicsSelectionBundleV34:
    spec.assert_sealed()
    private_pack.assert_sealed()
    policy.assert_sealed()
    if private_pack.spec_hash != spec.spec_hash:
        raise ValueError("V3.4 private pack belongs to another protocol")
    expected = (
        spec.baseline_policy_hash
        if policy.arm == "unguarded_full_action"
        else spec.candidate_policy_hash
    )
    if policy.policy_hash != expected:
        raise ValueError("V3.4 policy is not frozen in the protocol")
    at = executed_at or datetime.now(timezone.utc)
    receipts = [
        _execute_case_v34(spec, private_case, policy, executed_at=at)
        for private_case in private_pack.cases
    ]
    return ControlledDynamicsSelectionBundleV34.seal(
        spec_hash=spec.spec_hash,
        private_pack_hash=private_pack.pack_hash,
        policy_hash=policy.policy_hash,
        arm=policy.arm,
        case_receipts=receipts,
    )


def evaluate_controlled_dynamics_worldpack_v34(
    spec: ControlledDynamicsWorldPackSpecV34,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    baseline: ControlledDynamicsSelectionBundleV34,
    candidate: ControlledDynamicsSelectionBundleV34,
    *,
    evaluated_at: datetime | None = None,
) -> InterruptibleRealityEvolutionReportV34:
    spec.assert_sealed()
    private_pack.assert_sealed()
    baseline.assert_sealed()
    candidate.assert_sealed()
    if baseline.spec_hash != spec.spec_hash or candidate.spec_hash != spec.spec_hash:
        raise ValueError("V3.4 bundles do not belong to protocol")
    if baseline.private_pack_hash != private_pack.pack_hash or (
        candidate.private_pack_hash != private_pack.pack_hash
    ):
        raise ValueError("V3.4 bundles do not belong to private pack")
    baseline_by = {item.case_id: item for item in baseline.case_receipts}
    candidate_by = {item.case_id: item for item in candidate.case_receipts}
    private_by = {item.public_case.case_id: item for item in private_pack.cases}
    parity_ids = list(baseline_by)
    proposal_parity = all(
        baseline_by[case_id].anchor_action_hashes
        == candidate_by[case_id].anchor_action_hashes
        and baseline_by[case_id].anchor_observation_hashes
        == candidate_by[case_id].anchor_observation_hashes
        and baseline_by[case_id].selected_action_hash
        == candidate_by[case_id].selected_action_hash
        and baseline_by[case_id].decision_target
        == candidate_by[case_id].decision_target
        for case_id in parity_ids
    )
    trust_parity = all(
        (
            baseline_by[case_id].trust_decision is None
            and candidate_by[case_id].trust_decision is None
        ) or (
            baseline_by[case_id].trust_decision is not None
            and candidate_by[case_id].trust_decision is not None
            and baseline_by[case_id].trust_decision.trust_hash
            == candidate_by[case_id].trust_decision.trust_hash
        )
        for case_id in parity_ids
    )
    noise_parity = all(
        baseline_by[case_id].noise_schedule_hash
        == candidate_by[case_id].noise_schedule_hash
        for case_id in parity_ids
    )
    abstention_parity = all(
        baseline_by[case_id].plan_admissible
        == candidate_by[case_id].plan_admissible
        and baseline_by[case_id].abstention_reason
        == candidate_by[case_id].abstention_reason
        for case_id in parity_ids
    )
    receipts_complete = all(
        (not item.plan_admissible) or (
            item.calibration is not None
            and item.executed_intervention is not None
            and item.observation is not None
            and item.exposure_ledger is not None
            and len(item.segment_receipts)
            == item.exposure_ledger.executed_segment_count
        )
        for item in [*baseline.case_receipts, *candidate.case_receipts]
    )
    exposure_dominance = all(
        candidate_by[case_id].exposure_ledger is None or (
            baseline_by[case_id].exposure_ledger is not None
            and candidate_by[case_id].exposure_ledger.used_duration
            <= baseline_by[case_id].exposure_ledger.used_duration + 1e-12
            and candidate_by[case_id].exposure_ledger.used_energy
            <= baseline_by[case_id].exposure_ledger.used_energy + 1e-12
            and candidate_by[case_id].exposure_ledger.used_peak_amplitude
            <= baseline_by[case_id].exposure_ledger.used_peak_amplitude + 1e-12
            and candidate_by[case_id].exposure_ledger.used_switch_count
            <= baseline_by[case_id].exposure_ledger.used_switch_count
        )
        for case_id in parity_ids
    )
    state_violations = sum(
        item.exposure_ledger.state_envelope_violation_count
        for item in [*baseline.case_receipts, *candidate.case_receipts]
        if item.exposure_ledger is not None
    )
    active_count = sum(
        item.selected_mode == "active" for item in candidate.case_receipts
    )
    fallback_count = sum(
        item.selected_mode == "prefrozen_fallback" for item in candidate.case_receipts
    )
    interruptions = sum(
        item.executed_intervention is not None
        and item.executed_intervention.interrupted
        for item in candidate.case_receipts
    )
    terminations = sum(
        item.executed_intervention is not None
        and item.executed_intervention.terminated
        for item in candidate.case_receipts
    )
    eligible_ids = [
        case_id for case_id in parity_ids
        if baseline_by[case_id].target_loss is not None
        and candidate_by[case_id].target_loss is not None
    ]
    improvements = np.asarray([
        baseline_by[case_id].target_loss - candidate_by[case_id].target_loss
        for case_id in eligible_ids
    ], dtype=float)
    if not len(improvements):
        raise RuntimeError("V3.4 evaluation has no eligible cases")
    random = np.random.default_rng(spec.bootstrap_seed)
    samples = random.integers(
        0, len(improvements), size=(spec.bootstrap_replicates, len(improvements))
    )
    bootstrap = np.mean(improvements[samples], axis=1)
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
    mechanism_means = {
        mechanism: float(np.mean([
            improvements[index]
            for index, case_id in enumerate(eligible_ids)
            if private_by[case_id].mechanism == mechanism
        ]))
        for mechanism in MECHANISMS_V31
        if any(private_by[case_id].mechanism == mechanism for case_id in eligible_ids)
    }
    negative_count = int(np.sum(improvements < -spec.material_negative_transfer))
    negative_rate = negative_count / len(improvements)
    baseline_max = max(baseline_by[case_id].target_loss for case_id in eligible_ids)
    candidate_max = max(candidate_by[case_id].target_loss for case_id in eligible_ids)
    gates = {
        "proposal_target_anchor_parity": proposal_parity,
        "trust_decision_parity": trust_parity,
        "common_noise_schedule_parity": noise_parity,
        "paired_abstention_parity": abstention_parity,
        "segment_and_exposure_receipts_complete": receipts_complete,
        "minimum_interruption_exercised": interruptions >= spec.minimum_interruption_count,
        "candidate_exposure_dominated_by_baseline": exposure_dominance,
        "zero_synthetic_state_envelope_violations": state_violations == 0,
        "paired_macro_improvement_lower_bound": float(ci_low) >= 0.0,
        "mechanism_non_regression": min(mechanism_means.values())
        >= -spec.maximum_mechanism_regression,
        "negative_transfer_upper_bound": negative_rate
        <= spec.maximum_guard_negative_transfer_rate,
        "worst_case_loss_non_regression": candidate_max <= baseline_max + 1e-12,
    }
    ready = all(gates.values())
    return InterruptibleRealityEvolutionReportV34.seal(
        evolution_id="controlled_dynamics_interruptible_reality_exploratory_v34",
        spec_hash=spec.spec_hash,
        prior_v332_failure_report_hash=spec.prior_v332_failure_report_hash,
        eligible_case_count=len(eligible_ids),
        active_proposal_count=active_count,
        prefrozen_fallback_count=fallback_count,
        interruption_count=interruptions,
        termination_count=terminations,
        paired_macro_improvement=float(np.mean(improvements)),
        bootstrap_ci_low=float(ci_low),
        bootstrap_ci_high=float(ci_high),
        mechanism_mean_improvements=mechanism_means,
        material_negative_transfer_count=negative_count,
        material_negative_transfer_rate=negative_rate,
        baseline_max_target_loss=baseline_max,
        candidate_max_target_loss=candidate_max,
        gates=gates,
        adapter_candidate_ready=ready,
        status=(
            "interaction_adapter_ready_for_acquisition_retest_v34"
            if ready else "interaction_adapter_failed_v34"
        ),
        created_at=evaluated_at or datetime.now(timezone.utc),
    )


def run_controlled_dynamics_worldpack_v34(
    output_root: str | Path,
    *,
    spec: ControlledDynamicsWorldPackSpecV34,
    baseline_policy: InterruptibleRealityPolicyV34,
    candidate_policy: InterruptibleRealityPolicyV34,
    evaluated_at: datetime | None = None,
    run_id: str | None = None,
) -> ControlledDynamicsOutcomeV34:
    spec.assert_sealed()
    baseline_policy.assert_sealed()
    candidate_policy.assert_sealed()
    if baseline_policy.policy_hash != spec.baseline_policy_hash:
        raise ValueError("V3.4 baseline is not frozen in protocol")
    if candidate_policy.policy_hash != spec.candidate_policy_hash:
        raise ValueError("V3.4 candidate is not frozen in protocol")
    at = evaluated_at or datetime.now(timezone.utc)
    store = RunStore(
        output_root,
        run_id=run_id or f"controlled-dynamics-v34-{uuid4().hex[:10]}",
    )
    refs = [
        store.put_artifact("controlled_dynamics_spec_v34", spec),
        store.put_artifact("controlled_dynamics_baseline_policy_v34", baseline_policy),
        store.put_artifact("controlled_dynamics_candidate_policy_v34", candidate_policy),
    ]
    store.emit("controlled_dynamics_v34_protocol_frozen_before_private_pack", {
        "spec_hash": spec.spec_hash,
        "prior_v332_failure_report_hash": spec.prior_v332_failure_report_hash,
        "single_component_delta": spec.frozen_delta,
    })
    private_pack = generate_private_controlled_dynamics_worldpack_v31(
        spec, generated_at=at
    )
    baseline = execute_controlled_dynamics_policy_v34(
        spec, private_pack, baseline_policy, executed_at=at
    )
    candidate = execute_controlled_dynamics_policy_v34(
        spec, private_pack, candidate_policy, executed_at=at
    )
    evolution = evaluate_controlled_dynamics_worldpack_v34(
        spec, private_pack, baseline, candidate, evaluated_at=at
    )
    refs.extend([
        store.put_artifact("private_controlled_dynamics_worldpack_v34", private_pack),
        store.put_artifact("controlled_dynamics_baseline_bundle_v34", baseline),
        store.put_artifact("controlled_dynamics_candidate_bundle_v34", candidate),
        store.put_artifact("controlled_dynamics_evolution_report_v34", evolution),
    ])
    manifest = ControlledDynamicsManifestV34.seal(
        run_id=store.run_id,
        artifact_refs=refs,
        terminal_status=evolution.status,
        created_at=at,
    )
    manifest_ref = store.put_artifact("controlled_dynamics_manifest_v34", manifest)
    store.emit("controlled_dynamics_v34_worldpack_adjudicated", {
        "manifest_ref": manifest_ref.model_dump(mode="json")
    })
    if not verify_controlled_dynamics_run_v34(store.run_directory):
        raise RuntimeError("V3.4 run failed independent verification")
    return ControlledDynamicsOutcomeV34(
        store,
        spec,
        private_pack,
        baseline_policy,
        candidate_policy,
        baseline,
        candidate,
        evolution,
        manifest,
    )


def verify_controlled_dynamics_run_v34(run_directory: str | Path) -> bool:
    try:
        store = RunStore.open_existing(run_directory)
        events = [
            json.loads(line)
            for line in store.event_path.read_text(encoding="utf-8").splitlines()
        ]
        committed = [
            ArtifactRef.model_validate(event["payload"])
            for event in events if event["event_type"] == "artifact_committed"
        ]
        for reference in committed:
            store.load_artifact(reference)
        manifest_refs = [
            item for item in committed
            if item.kind == "controlled_dynamics_manifest_v34"
        ]
        if len(manifest_refs) != 1:
            return False
        manifest = ControlledDynamicsManifestV34.model_validate(
            store.load_artifact(manifest_refs[0])
        )
        manifest.assert_sealed()

        def load_one(kind: str, model):
            references = [
                item for item in manifest.artifact_refs if item.kind == kind
            ]
            if len(references) != 1:
                raise RuntimeError(f"V3.4 manifest needs exactly one {kind}")
            return model.model_validate(store.load_artifact(references[0]))

        spec = load_one(
            "controlled_dynamics_spec_v34", ControlledDynamicsWorldPackSpecV34
        )
        baseline_policy = load_one(
            "controlled_dynamics_baseline_policy_v34", InterruptibleRealityPolicyV34
        )
        candidate_policy = load_one(
            "controlled_dynamics_candidate_policy_v34", InterruptibleRealityPolicyV34
        )
        private_pack = load_one(
            "private_controlled_dynamics_worldpack_v34",
            PrivateControlledDynamicsWorldPackV31,
        )
        baseline = load_one(
            "controlled_dynamics_baseline_bundle_v34",
            ControlledDynamicsSelectionBundleV34,
        )
        candidate = load_one(
            "controlled_dynamics_candidate_bundle_v34",
            ControlledDynamicsSelectionBundleV34,
        )
        evolution = load_one(
            "controlled_dynamics_evolution_report_v34",
            InterruptibleRealityEvolutionReportV34,
        )
        for artifact in (
            spec, baseline_policy, candidate_policy, private_pack,
            baseline, candidate, evolution, manifest,
        ):
            artifact.assert_sealed()
        if manifest.run_id != store.run_id:
            return False
        regenerated = generate_private_controlled_dynamics_worldpack_v31(
            spec, generated_at=private_pack.generated_at
        )
        if regenerated.pack_hash != private_pack.pack_hash:
            return False
        executed_at = baseline.case_receipts[0].executed_at
        replay_baseline = execute_controlled_dynamics_policy_v34(
            spec, private_pack, baseline_policy, executed_at=executed_at
        )
        replay_candidate = execute_controlled_dynamics_policy_v34(
            spec, private_pack, candidate_policy, executed_at=executed_at
        )
        if replay_baseline.bundle_hash != baseline.bundle_hash or (
            replay_candidate.bundle_hash != candidate.bundle_hash
        ):
            return False
        recomputed = evaluate_controlled_dynamics_worldpack_v34(
            spec,
            private_pack,
            baseline,
            candidate,
            evaluated_at=evolution.created_at,
        )
        if recomputed.evolution_hash != evolution.evolution_hash:
            return False
        if any(
            "qualification" in item.kind or "confirmation" in item.kind
            for item in manifest.artifact_refs
        ):
            return False
        freeze_events = [
            event for event in events
            if event["event_type"]
            == "controlled_dynamics_v34_protocol_frozen_before_private_pack"
        ]
        private_events = [
            event for event in events
            if event["event_type"] == "artifact_committed"
            and event["payload"]["kind"] == "private_controlled_dynamics_worldpack_v34"
        ]
        return (
            len(freeze_events) == 1
            and len(private_events) == 1
            and freeze_events[0]["sequence"] < private_events[0]["sequence"]
            and store.verify_event_chain()
        )
    except (
        OSError, RuntimeError, ValueError, KeyError, TypeError,
        json.JSONDecodeError, FloatingPointError,
    ):
        return False
