from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore
from fma.v2.dynamics_ir import trajectory_nrmse
from fma.v2.schemas import Identifier, Sha256, _assert_timezone

from .controlled_dynamics_loop import (
    ArmV31,
    MECHANISMS_V31,
    ControlledDynamicsContractV31,
    ControlledDynamicsReportV31,
    PrivateControlledDynamicsCaseV31,
    PrivateControlledDynamicsWorldPackV31,
    RouteLayerV31,
    TargetClarificationEvidenceV31,
    _fit_model_v31,
    _permission_v31,
    _simulate_model_v31,
    _target_loss_v31,
    evaluate_controlled_dynamics_worldpack_v31,
    generate_private_controlled_dynamics_worldpack_v31,
)
from .controlled_dynamics_loop_v32 import _acquisition_receipts_v32
from .controlled_dynamics_loop_v33 import (
    ControlledDynamicsCaseReceiptV33,
    ControlledDynamicsManifestV33,
    ControlledDynamicsSelectionBundleV33,
    ControlledDynamicsStepReceiptV33,
    EpistemicResourceLedgerV33,
)
from .experiment_ir import ControlledObservationReceiptV31


EXPLORATORY_SEEDS_V331 = (
    12203, 12253, 12301, 12347, 12401, 12451, 12503, 12553,
    12601, 12653, 12703, 12757, 12809, 12853, 12907, 12959,
)


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class TrustGatedControlledDynamicsPolicyV331(StrictModel):
    schema_version: Literal["3.3.1"] = "3.3.1"
    policy_id: Identifier
    arm: ArmV31
    selection_rule: Literal[
        "clarify_then_three_shared_random_inputs",
        "clarify_then_two_shared_random_anchors_then_trust_gated_goal_risk",
    ]
    may_reformulate_problem: Literal[True] = True
    clarification_budget: Literal[1] = 1
    controlled_experiment_budget: Literal[3] = 3
    anchor_experiment_count: Literal[2] = 2
    known_actuator_required: Literal[True] = True
    prior_v33_failure_report_hash: Sha256
    prior_epistemic_qualification_hash: Sha256
    prior_active_design_qualification_hash: Sha256
    method_evidence_hash: Sha256
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "TrustGatedControlledDynamicsPolicyV331":
        expected = {
            "random_bounded_inputs": "clarify_then_three_shared_random_inputs",
            "goal_oriented_epistemic_control": (
                "clarify_then_two_shared_random_anchors_then_trust_gated_goal_risk"
            ),
        }[self.arm]
        if self.selection_rule != expected:
            raise ValueError("V3.3.1 arm and selection rule disagree")
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("policy_hash does not match V3.3.1 policy")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("V3.3.1 policy is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "TrustGatedControlledDynamicsPolicyV331":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"policy_hash"}),
            policy_hash=draft.content_hash(),
        )


class ControlledDynamicsWorldPackSpecV331(StrictModel):
    schema_version: Literal["3.3.1"] = "3.3.1"
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
    minimum_goal_risk_margin: Literal[0.03] = 0.03
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
    bootstrap_seed: Literal[331722] = 331722
    minimum_macro_loss_improvement: Literal[0.0] = 0.0
    maximum_mechanism_regression: Literal[0.02] = 0.02
    material_negative_transfer: Literal[0.02] = 0.02
    maximum_negative_transfer_rate: Literal[0.1] = 0.1
    required_routing_accuracy: Literal[1.0] = 1.0
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
    prior_v33_failure_report_hash: Sha256
    frozen_delta: Literal[
        "unconditional_goal_risk_to_cross_excitation_trust_gate_only"
    ] = "unconditional_goal_risk_to_cross_excitation_trust_gate_only"
    frozen_at: datetime
    spec_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "ControlledDynamicsWorldPackSpecV331":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.mechanisms != list(MECHANISMS_V31):
            raise ValueError("V3.3.1 requires the frozen mechanism order")
        if tuple(self.seeds) != EXPLORATORY_SEEDS_V331:
            raise ValueError("V3.3.1 seeds do not match the frozen exploratory set")
        if self.goal_initial_state_scales != [0.75, 1.0, 1.25]:
            raise ValueError("V3.3.1 public goal initial-state scales changed")
        if self.maximum_steps != self.action_budget + self.clarification_budget:
            raise ValueError("V3.3.1 resource budgets do not cover maximum steps")
        if not math.isclose(
            (self.trajectory_points - 1) * self.time_step,
            self.segment_count * self.segment_duration,
            abs_tol=1e-12,
        ):
            raise ValueError("V3.3.1 input segments do not cover the trajectory")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match V3.3.1 protocol")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "spec_hash")

    def assert_sealed(self) -> None:
        if not self.spec_hash or self.spec_hash != self.content_hash():
            raise ValueError("V3.3.1 protocol is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsWorldPackSpecV331":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"spec_hash"}),
            spec_hash=draft.content_hash(),
        )


class AcquisitionTrustDecisionV331(StrictModel):
    schema_version: Literal["3.3.1"] = "3.3.1"
    trust_id: Identifier
    case_id: Identifier
    anchor_action_hashes: list[Sha256] = Field(min_length=2, max_length=2)
    anchor_observation_hashes: list[Sha256] = Field(min_length=2, max_length=2)
    calibration_model_hash: Sha256
    cross_excitation_nrmse: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    maximum_cross_excitation_nrmse: Literal[0.05] = 0.05
    active_action_hash: Sha256
    fallback_action_hash: Sha256
    active_goal_risk_score: Annotated[float, Field(allow_inf_nan=False)]
    fallback_goal_risk_score: Annotated[float, Field(allow_inf_nan=False)]
    goal_risk_margin: Annotated[float, Field(allow_inf_nan=False)]
    minimum_goal_risk_margin: Literal[0.03] = 0.03
    decision: Literal["use_goal_risk", "fallback_prefrozen_random"]
    selected_action_hash: Sha256
    hidden_probe_used: Literal[False] = False
    calibrated_probability_claimed: Literal[False] = False
    trust_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_trust(self) -> "AcquisitionTrustDecisionV331":
        if len(set(self.anchor_action_hashes)) != 2:
            raise ValueError("V3.3.1 anchor actions must differ")
        if not math.isclose(
            self.goal_risk_margin,
            self.active_goal_risk_score - self.fallback_goal_risk_score,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("V3.3.1 goal-risk margin was not recomputed")
        passes = (
            self.cross_excitation_nrmse <= self.maximum_cross_excitation_nrmse
            and self.goal_risk_margin >= self.minimum_goal_risk_margin
        )
        expected_decision = (
            "use_goal_risk" if passes else "fallback_prefrozen_random"
        )
        if self.decision != expected_decision:
            raise ValueError("V3.3.1 trust decision disagrees with frozen gates")
        expected_hash = (
            self.active_action_hash if passes else self.fallback_action_hash
        )
        if self.selected_action_hash != expected_hash:
            raise ValueError("V3.3.1 selected action disagrees with trust decision")
        if self.trust_hash and self.trust_hash != self.content_hash():
            raise ValueError("trust_hash does not match V3.3.1 decision")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "trust_hash")

    def assert_sealed(self) -> None:
        if not self.trust_hash or self.trust_hash != self.content_hash():
            raise ValueError("V3.3.1 trust decision is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "AcquisitionTrustDecisionV331":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"trust_hash"}),
            trust_hash=draft.content_hash(),
        )


class ControlledDynamicsCaseReceiptV331(ControlledDynamicsCaseReceiptV33):
    schema_version: Literal["3.3.1"] = "3.3.1"
    acquisition_trust_decision: AcquisitionTrustDecisionV331 | None = None

    @model_validator(mode="after")
    def validate_trust_binding(self) -> "ControlledDynamicsCaseReceiptV331":
        completed_third_experiment = len(self.selected_action_ids) == 3
        if self.arm == "goal_oriented_epistemic_control" and completed_third_experiment:
            if self.acquisition_trust_decision is None:
                raise ValueError("V3.3.1 completed candidate case needs trust decision")
            self.acquisition_trust_decision.assert_sealed()
            if self.acquisition_trust_decision.selected_action_hash != self.steps[-1].selected_action_hash:
                raise ValueError("V3.3.1 trust decision is not bound to final step")
        elif self.acquisition_trust_decision is not None:
            raise ValueError("V3.3.1 baseline/data-failure case cannot carry trust decision")
        return self


class ControlledDynamicsSelectionBundleV331(ControlledDynamicsSelectionBundleV33):
    schema_version: Literal["3.3.1"] = "3.3.1"
    case_receipts: list[ControlledDynamicsCaseReceiptV331] = Field(
        min_length=64, max_length=64
    )


class AcquisitionTrustEvolutionReportV331(StrictModel):
    schema_version: Literal["3.3.1"] = "3.3.1"
    evolution_id: Identifier
    spec_hash: Sha256
    base_adjudication_report: ControlledDynamicsReportV31
    prior_v33_failure_report_hash: Sha256
    single_component_delta: Literal[
        "unconditional_goal_risk_to_cross_excitation_trust_gate_only"
    ] = "unconditional_goal_risk_to_cross_excitation_trust_gate_only"
    acquisition_gates: dict[Identifier, bool]
    resource_entitlement_parity: bool
    resource_use_parity: bool
    target_contract_parity: bool
    shared_anchor_parity: bool
    trust_receipts_complete: bool
    goal_risk_activation_count: Annotated[int, Field(ge=0)]
    random_fallback_count: Annotated[int, Field(ge=0)]
    acquisition_candidate_ready: bool
    router_gate_passed: bool
    router_accuracy: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    status: Literal[
        "acquisition_candidate_ready_for_router_evolution_v331",
        "acquisition_candidate_failed_v331",
    ]
    acquisition_changed: Literal[True] = True
    estimator_changed: Literal[False] = False
    action_catalog_changed: Literal[False] = False
    risk_gate_changed: Literal[False] = False
    statistical_gate_changed: Literal[False] = False
    model_router_changed: Literal[False] = False
    budget_model_changed: Literal[False] = False
    overall_qualification_permitted: Literal[False] = False
    confirmation_permitted: Literal[False] = False
    created_at: datetime
    evolution_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_evolution(self) -> "AcquisitionTrustEvolutionReportV331":
        _assert_timezone(self.created_at, "created_at")
        self.base_adjudication_report.assert_sealed()
        if self.base_adjudication_report.spec_hash != self.spec_hash:
            raise ValueError("V3.3.1 wrapper is bound to another adjudication")
        ready = all(self.acquisition_gates.values())
        if self.acquisition_candidate_ready != ready:
            raise ValueError("V3.3.1 readiness disagrees with gates")
        expected = (
            "acquisition_candidate_ready_for_router_evolution_v331"
            if ready else "acquisition_candidate_failed_v331"
        )
        if self.status != expected:
            raise ValueError("V3.3.1 status disagrees with gates")
        if self.router_gate_passed != self.base_adjudication_report.gates["routing_accuracy"]:
            raise ValueError("V3.3.1 router status was not inherited unchanged")
        if self.evolution_hash and self.evolution_hash != self.content_hash():
            raise ValueError("evolution_hash does not match V3.3.1 report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evolution_hash")

    def assert_sealed(self) -> None:
        if not self.evolution_hash or self.evolution_hash != self.content_hash():
            raise ValueError("V3.3.1 evolution report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "AcquisitionTrustEvolutionReportV331":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evolution_hash"}),
            evolution_hash=draft.content_hash(),
        )


class ControlledDynamicsManifestV331(ControlledDynamicsManifestV33):
    schema_version: Literal["3.3.1"] = "3.3.1"
    terminal_status: Literal[
        "acquisition_candidate_ready_for_router_evolution_v331",
        "acquisition_candidate_failed_v331",
    ]


@dataclass(frozen=True)
class ControlledDynamicsOutcomeV331:
    store: RunStore
    spec: ControlledDynamicsWorldPackSpecV331
    private_pack: PrivateControlledDynamicsWorldPackV31
    baseline_policy: TrustGatedControlledDynamicsPolicyV331
    candidate_policy: TrustGatedControlledDynamicsPolicyV331
    baseline_bundle: ControlledDynamicsSelectionBundleV331
    candidate_bundle: ControlledDynamicsSelectionBundleV331
    evolution_report: AcquisitionTrustEvolutionReportV331
    manifest: ControlledDynamicsManifestV331


def default_controlled_dynamics_policies_v331(
    *,
    prior_v33_failure_report_hash: str,
    prior_epistemic_qualification_hash: str,
    prior_active_design_qualification_hash: str,
    method_evidence_hash: str,
) -> tuple[
    TrustGatedControlledDynamicsPolicyV331,
    TrustGatedControlledDynamicsPolicyV331,
]:
    shared = dict(
        prior_v33_failure_report_hash=prior_v33_failure_report_hash,
        prior_epistemic_qualification_hash=prior_epistemic_qualification_hash,
        prior_active_design_qualification_hash=prior_active_design_qualification_hash,
        method_evidence_hash=method_evidence_hash,
    )
    return (
        TrustGatedControlledDynamicsPolicyV331.seal(
            policy_id="shared_random_inputs_v331",
            arm="random_bounded_inputs",
            selection_rule="clarify_then_three_shared_random_inputs",
            **shared,
        ),
        TrustGatedControlledDynamicsPolicyV331.seal(
            policy_id="trust_gated_goal_posterior_risk_v331",
            arm="goal_oriented_epistemic_control",
            selection_rule=(
                "clarify_then_two_shared_random_anchors_then_trust_gated_goal_risk"
            ),
            **shared,
        ),
    )


def default_controlled_dynamics_exploratory_spec_v331(
    *,
    baseline_policy_hash: str,
    candidate_policy_hash: str,
    method_evidence_hash: str,
    prior_epistemic_qualification_hash: str,
    prior_active_design_qualification_hash: str,
    prior_v33_failure_report_hash: str,
    frozen_at: datetime | None = None,
) -> ControlledDynamicsWorldPackSpecV331:
    return ControlledDynamicsWorldPackSpecV331.seal(
        experiment_id="controlled_dynamics_trust_gate_exploratory_v331",
        mechanisms=list(MECHANISMS_V31),
        seeds=list(EXPLORATORY_SEEDS_V331),
        baseline_policy_hash=baseline_policy_hash,
        candidate_policy_hash=candidate_policy_hash,
        method_evidence_hash=method_evidence_hash,
        prior_epistemic_qualification_hash=prior_epistemic_qualification_hash,
        prior_active_design_qualification_hash=prior_active_design_qualification_hash,
        prior_v33_failure_report_hash=prior_v33_failure_report_hash,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


def _shared_random_order_v331(spec, public) -> list[str]:
    random = Random(int(sha256_value([
        spec.spec_hash, public.case_id, "shared_random_anchor_order_v331"
    ])[:16], 16))
    order = [action.action_id for action in public.action_catalog]
    random.shuffle(order)
    return order


def _trust_decision_v331(
    spec: ControlledDynamicsWorldPackSpecV331,
    public,
    observations: list[ControlledObservationReceiptV31],
    selected_ids: list[str],
    acquisitions,
    active_action,
    fallback_action,
) -> AcquisitionTrustDecisionV331:
    calibration_model = _fit_model_v31(public, [observations[0]], spec)
    second_action = next(
        item for item in public.action_catalog if item.action_id == selected_ids[1]
    )
    try:
        predicted = _simulate_model_v31(
            calibration_model,
            public.actuator,
            public.initial_state,
            public.pilot.times,
            second_action.input_values,
            spec.segment_duration,
        )
        cross_error = trajectory_nrmse(observations[1].states, predicted)
    except RuntimeError:
        cross_error = 10.0
    by_hash = {item.action_hash: item for item in acquisitions}
    active_score = by_hash[active_action.action_hash].ranking_score
    fallback_score = by_hash[fallback_action.action_hash].ranking_score
    margin = active_score - fallback_score
    use_active = (
        cross_error <= spec.maximum_cross_excitation_nrmse
        and margin >= spec.minimum_goal_risk_margin
    )
    selected_hash = (
        active_action.action_hash if use_active else fallback_action.action_hash
    )
    return AcquisitionTrustDecisionV331.seal(
        trust_id=f"trust_{public.case_id}",
        case_id=public.case_id,
        anchor_action_hashes=[
            next(item for item in public.action_catalog if item.action_id == action_id).action_hash
            for action_id in selected_ids[:2]
        ],
        anchor_observation_hashes=[item.observation_hash for item in observations[:2]],
        calibration_model_hash=calibration_model.model_hash,
        cross_excitation_nrmse=cross_error,
        active_action_hash=active_action.action_hash,
        fallback_action_hash=fallback_action.action_hash,
        active_goal_risk_score=active_score,
        fallback_goal_risk_score=fallback_score,
        goal_risk_margin=margin,
        decision=("use_goal_risk" if use_active else "fallback_prefrozen_random"),
        selected_action_hash=selected_hash,
    )


def _execute_case_v331(
    spec: ControlledDynamicsWorldPackSpecV331,
    private_case: PrivateControlledDynamicsCaseV31,
    policy: TrustGatedControlledDynamicsPolicyV331,
    *,
    executed_at: datetime,
) -> ControlledDynamicsCaseReceiptV331:
    public = private_case.public_case
    public.assert_sealed()
    data_quality_passed = not public.pilot.quality_flags
    contract = public.initial_contract
    experiment_budget = spec.action_budget
    clarification_remaining = spec.clarification_budget
    observations: list[ControlledObservationReceiptV31] = []
    selected_ids: list[str] = []
    steps: list[ControlledDynamicsStepReceiptV33] = []
    issue_routes: list[RouteLayerV31] = []
    abstention_count = 0
    clarification_used = 0
    trust_decision = None
    shared_order = _shared_random_order_v331(spec, public)

    for step_index in range(1, spec.maximum_steps + 1):
        if contract.target_status == "default_unverified":
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
            clarification_remaining -= 1
            clarification_used += 1
            issue_routes.append("problem_layer")
            steps.append(ControlledDynamicsStepReceiptV33.seal(
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
                budget_before=experiment_budget,
                decided_at=executed_at,
            )
            if "data_layer" not in issue_routes:
                issue_routes.append("data_layer")
            abstention_count = 1
            steps.append(ControlledDynamicsStepReceiptV33.seal(
                step_index=step_index,
                action_kind="abstain",
                contract_before_hash=contract.contract_hash,
                contract_after_hash=contract.contract_hash,
                permission=permission,
            ))
            break
        if experiment_budget < 1:
            break

        available = [
            action for action in public.action_catalog
            if action.action_id not in selected_ids
        ]
        acquisitions = _acquisition_receipts_v32(
            spec, public, observations, available,
            contract.decision_target, step_index,
        )
        by_hash = {item.action_hash: item for item in acquisitions}
        admissible_actions = [
            action for action in available if by_hash[action.action_hash].admissible
        ]
        if not admissible_actions:
            best = max(acquisitions, key=lambda item: (item.ranking_score, item.action_hash))
            permission = _permission_v31(
                best.acquisition_hash,
                public.envelope.envelope_hash,
                data_quality_passed=True,
                admissible=False,
                budget_before=experiment_budget,
                decided_at=executed_at,
            )
            abstention_count = 1
            steps.append(ControlledDynamicsStepReceiptV33.seal(
                step_index=step_index,
                action_kind="abstain",
                contract_before_hash=contract.contract_hash,
                contract_after_hash=contract.contract_hash,
                acquisition_receipts=acquisitions,
                permission=permission,
            ))
            break

        fallback = next(
            action for action_id in shared_order
            for action in admissible_actions if action.action_id == action_id
        )
        selected = fallback
        if (
            policy.arm == "goal_oriented_epistemic_control"
            and len(observations) >= spec.anchor_experiment_count
        ):
            active = max(
                admissible_actions,
                key=lambda item: (by_hash[item.action_hash].ranking_score, item.action_id),
            )
            trust_decision = _trust_decision_v331(
                spec, public, observations, selected_ids,
                acquisitions, active, fallback,
            )
            selected = next(
                item for item in admissible_actions
                if item.action_hash == trust_decision.selected_action_hash
            )
        selected_acquisition = by_hash[selected.action_hash]
        permission = _permission_v31(
            selected_acquisition.acquisition_hash,
            public.envelope.envelope_hash,
            data_quality_passed=True,
            admissible=True,
            budget_before=experiment_budget,
            decided_at=executed_at,
        )
        observation = private_case.action_observations[selected.action_id]
        observation.assert_sealed()
        if observation.empirical_peak_state_ratio > 1.0:
            raise RuntimeError("V3.3.1 hidden Reality Interface detected state violation")
        observations.append(observation)
        selected_ids.append(selected.action_id)
        experiment_budget = permission.budget_after
        steps.append(ControlledDynamicsStepReceiptV33.seal(
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

    model = None
    target_loss = None
    if data_quality_passed:
        model = _fit_model_v31(public, observations, spec)
        if model.normalized_derivative_residual > spec.model_mismatch_residual_threshold:
            issue_routes.append("model_layer")
        if private_case.performance_eligible:
            target_loss = _target_loss_v31(private_case, model, spec)
    ledger = EpistemicResourceLedgerV33.seal(
        clarification_used=clarification_used,
        controlled_experiments_used=spec.action_budget - experiment_budget,
        clarification_remaining=clarification_remaining,
        controlled_experiments_remaining=experiment_budget,
    )
    return ControlledDynamicsCaseReceiptV331.seal(
        receipt_id=f"receipt_{policy.arm}_{public.case_id}_v331",
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
        action_budget_consumed=(
            clarification_used + spec.action_budget - experiment_budget
        ),
        abstention_count=abstention_count,
        target_loss=target_loss,
        executed_at=executed_at,
        resource_ledger=ledger,
        acquisition_trust_decision=trust_decision,
    )


def execute_controlled_dynamics_policy_v331(
    spec: ControlledDynamicsWorldPackSpecV331,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    policy: TrustGatedControlledDynamicsPolicyV331,
    *,
    executed_at: datetime | None = None,
) -> ControlledDynamicsSelectionBundleV331:
    spec.assert_sealed()
    private_pack.assert_sealed()
    policy.assert_sealed()
    if private_pack.spec_hash != spec.spec_hash:
        raise ValueError("V3.3.1 private pack belongs to another protocol")
    expected = (
        spec.baseline_policy_hash
        if policy.arm == "random_bounded_inputs"
        else spec.candidate_policy_hash
    )
    if policy.policy_hash != expected:
        raise ValueError("V3.3.1 policy is not frozen in the protocol")
    at = executed_at or datetime.now(timezone.utc)
    receipts = [
        _execute_case_v331(spec, private_case, policy, executed_at=at)
        for private_case in private_pack.cases
    ]
    return ControlledDynamicsSelectionBundleV331.seal(
        spec_hash=spec.spec_hash,
        private_pack_hash=private_pack.pack_hash,
        policy_hash=policy.policy_hash,
        arm=policy.arm,
        case_receipts=receipts,
        total_action_budget_consumed=sum(
            item.action_budget_consumed for item in receipts
        ),
        total_abstentions=sum(item.abstention_count for item in receipts),
    )


def evaluate_controlled_dynamics_worldpack_v331(
    spec: ControlledDynamicsWorldPackSpecV331,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    baseline: ControlledDynamicsSelectionBundleV331,
    candidate: ControlledDynamicsSelectionBundleV331,
    *,
    evaluated_at: datetime | None = None,
) -> AcquisitionTrustEvolutionReportV331:
    report = evaluate_controlled_dynamics_worldpack_v31(
        spec, private_pack, baseline, candidate, evaluated_at=evaluated_at
    )
    baseline_by_id = {item.case_id: item for item in baseline.case_receipts}
    candidate_by_id = {item.case_id: item for item in candidate.case_receipts}
    entitlement_parity = all(
        left.resource_ledger.initial_clarification_budget
        == candidate_by_id[case_id].resource_ledger.initial_clarification_budget
        and left.resource_ledger.initial_controlled_experiment_budget
        == candidate_by_id[case_id].resource_ledger.initial_controlled_experiment_budget
        for case_id, left in baseline_by_id.items()
    )
    use_parity = all(
        left.resource_ledger.clarification_used
        == candidate_by_id[case_id].resource_ledger.clarification_used
        and left.resource_ledger.controlled_experiments_used
        == candidate_by_id[case_id].resource_ledger.controlled_experiments_used
        for case_id, left in baseline_by_id.items()
    )
    contract_parity = all(
        left.final_contract.contract_hash
        == candidate_by_id[case_id].final_contract.contract_hash
        for case_id, left in baseline_by_id.items()
    )
    shared_anchor_ids = [
        case_id for case_id, receipt in candidate_by_id.items()
        if len(receipt.selected_action_ids) >= spec.anchor_experiment_count
    ]
    anchor_parity = all(
        baseline_by_id[case_id].selected_action_ids[:2]
        == candidate_by_id[case_id].selected_action_ids[:2]
        for case_id in shared_anchor_ids
    )
    completed_candidate_ids = [
        case_id for case_id, receipt in candidate_by_id.items()
        if len(receipt.selected_action_ids) == spec.action_budget
    ]
    trust_decisions = [
        candidate_by_id[case_id].acquisition_trust_decision
        for case_id in completed_candidate_ids
    ]
    trust_complete = (
        all(item is not None for item in trust_decisions)
        and all(
            receipt.acquisition_trust_decision is None
            for case_id, receipt in candidate_by_id.items()
            if case_id not in completed_candidate_ids
        )
    )
    activations = sum(
        item is not None and item.decision == "use_goal_risk"
        for item in trust_decisions
    )
    fallbacks = sum(
        item is not None and item.decision == "fallback_prefrozen_random"
        for item in trust_decisions
    )
    acquisition_gate_names = (
        "macro_improvement_lower_bound",
        "mechanism_non_regression",
        "negative_transfer_upper_bound",
        "required_reformulations_evidence_bound",
        "zero_spurious_reformulations",
        "data_gate_prevented_experiments",
        "zero_invalid_actions",
        "zero_actual_state_violations",
    )
    acquisition_gates = {
        name: report.gates[name] for name in acquisition_gate_names
    }
    acquisition_gates.update({
        "resource_entitlement_parity": entitlement_parity,
        "resource_use_parity": use_parity,
        "target_contract_parity": contract_parity,
        "shared_anchor_parity": anchor_parity,
        "trust_receipts_complete": trust_complete,
    })
    ready = all(acquisition_gates.values())
    return AcquisitionTrustEvolutionReportV331.seal(
        evolution_id="controlled_dynamics_trust_gate_exploratory_v331",
        spec_hash=spec.spec_hash,
        base_adjudication_report=report,
        prior_v33_failure_report_hash=spec.prior_v33_failure_report_hash,
        acquisition_gates=acquisition_gates,
        resource_entitlement_parity=entitlement_parity,
        resource_use_parity=use_parity,
        target_contract_parity=contract_parity,
        shared_anchor_parity=anchor_parity,
        trust_receipts_complete=trust_complete,
        goal_risk_activation_count=activations,
        random_fallback_count=fallbacks,
        acquisition_candidate_ready=ready,
        router_gate_passed=report.gates["routing_accuracy"],
        router_accuracy=report.routing_accuracy,
        status=(
            "acquisition_candidate_ready_for_router_evolution_v331"
            if ready else "acquisition_candidate_failed_v331"
        ),
        created_at=evaluated_at or datetime.now(timezone.utc),
    )


def run_controlled_dynamics_worldpack_v331(
    output_root: str | Path,
    *,
    spec: ControlledDynamicsWorldPackSpecV331,
    baseline_policy: TrustGatedControlledDynamicsPolicyV331,
    candidate_policy: TrustGatedControlledDynamicsPolicyV331,
    evaluated_at: datetime | None = None,
    run_id: str | None = None,
) -> ControlledDynamicsOutcomeV331:
    spec.assert_sealed()
    baseline_policy.assert_sealed()
    candidate_policy.assert_sealed()
    if baseline_policy.policy_hash != spec.baseline_policy_hash:
        raise ValueError("V3.3.1 baseline is not frozen in the protocol")
    if candidate_policy.policy_hash != spec.candidate_policy_hash:
        raise ValueError("V3.3.1 candidate is not frozen in the protocol")
    at = evaluated_at or datetime.now(timezone.utc)
    store = RunStore(
        output_root,
        run_id=run_id or f"controlled-dynamics-v331-{uuid4().hex[:10]}",
    )
    refs = [
        store.put_artifact("controlled_dynamics_spec_v331", spec),
        store.put_artifact("controlled_dynamics_baseline_policy_v331", baseline_policy),
        store.put_artifact("controlled_dynamics_candidate_policy_v331", candidate_policy),
    ]
    store.emit("controlled_dynamics_v331_protocol_frozen_before_private_pack", {
        "spec_hash": spec.spec_hash,
        "prior_v33_failure_report_hash": spec.prior_v33_failure_report_hash,
        "single_component_delta": spec.frozen_delta,
    })
    private_pack = generate_private_controlled_dynamics_worldpack_v31(
        spec, generated_at=at
    )
    baseline = execute_controlled_dynamics_policy_v331(
        spec, private_pack, baseline_policy, executed_at=at
    )
    candidate = execute_controlled_dynamics_policy_v331(
        spec, private_pack, candidate_policy, executed_at=at
    )
    evolution = evaluate_controlled_dynamics_worldpack_v331(
        spec, private_pack, baseline, candidate, evaluated_at=at
    )
    refs.extend([
        store.put_artifact("private_controlled_dynamics_worldpack_v331", private_pack),
        store.put_artifact("controlled_dynamics_baseline_bundle_v331", baseline),
        store.put_artifact("controlled_dynamics_candidate_bundle_v331", candidate),
        store.put_artifact(
            "controlled_dynamics_base_report_v331",
            evolution.base_adjudication_report,
        ),
        store.put_artifact("controlled_dynamics_evolution_report_v331", evolution),
    ])
    manifest = ControlledDynamicsManifestV331.seal(
        run_id=store.run_id,
        artifact_refs=refs,
        terminal_status=evolution.status,
        created_at=at,
    )
    manifest_ref = store.put_artifact("controlled_dynamics_manifest_v331", manifest)
    store.emit("controlled_dynamics_v331_worldpack_adjudicated", {
        "manifest_ref": manifest_ref.model_dump(mode="json")
    })
    if not verify_controlled_dynamics_run_v331(store.run_directory):
        raise RuntimeError("V3.3.1 controlled-dynamics run failed independent verification")
    return ControlledDynamicsOutcomeV331(
        store, spec, private_pack, baseline_policy, candidate_policy,
        baseline, candidate, evolution, manifest,
    )


def verify_controlled_dynamics_run_v331(run_directory: str | Path) -> bool:
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
            if item.kind == "controlled_dynamics_manifest_v331"
        ]
        if len(manifest_refs) != 1:
            return False
        manifest = ControlledDynamicsManifestV331.model_validate(
            store.load_artifact(manifest_refs[0])
        )
        manifest.assert_sealed()

        def load_one(kind: str, model):
            references = [item for item in manifest.artifact_refs if item.kind == kind]
            if len(references) != 1:
                raise RuntimeError(f"V3.3.1 manifest needs exactly one {kind}")
            return model.model_validate(store.load_artifact(references[0]))

        spec = load_one(
            "controlled_dynamics_spec_v331", ControlledDynamicsWorldPackSpecV331
        )
        baseline_policy = load_one(
            "controlled_dynamics_baseline_policy_v331",
            TrustGatedControlledDynamicsPolicyV331,
        )
        candidate_policy = load_one(
            "controlled_dynamics_candidate_policy_v331",
            TrustGatedControlledDynamicsPolicyV331,
        )
        private_pack = load_one(
            "private_controlled_dynamics_worldpack_v331",
            PrivateControlledDynamicsWorldPackV31,
        )
        baseline = load_one(
            "controlled_dynamics_baseline_bundle_v331",
            ControlledDynamicsSelectionBundleV331,
        )
        candidate = load_one(
            "controlled_dynamics_candidate_bundle_v331",
            ControlledDynamicsSelectionBundleV331,
        )
        base_report = load_one(
            "controlled_dynamics_base_report_v331", ControlledDynamicsReportV31
        )
        evolution = load_one(
            "controlled_dynamics_evolution_report_v331",
            AcquisitionTrustEvolutionReportV331,
        )
        for artifact in (
            spec, baseline_policy, candidate_policy, private_pack,
            baseline, candidate, base_report, evolution, manifest,
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
        replay_baseline = execute_controlled_dynamics_policy_v331(
            spec, private_pack, baseline_policy, executed_at=executed_at
        )
        replay_candidate = execute_controlled_dynamics_policy_v331(
            spec, private_pack, candidate_policy, executed_at=executed_at
        )
        if (
            replay_baseline.bundle_hash != baseline.bundle_hash
            or replay_candidate.bundle_hash != candidate.bundle_hash
        ):
            return False
        recomputed = evaluate_controlled_dynamics_worldpack_v331(
            spec, private_pack, baseline, candidate,
            evaluated_at=base_report.evaluated_at,
        )
        if recomputed.evolution_hash != evolution.evolution_hash:
            return False
        if any("qualification" in item.kind for item in manifest.artifact_refs):
            return False
        freeze_events = [
            event for event in events
            if event["event_type"]
            == "controlled_dynamics_v331_protocol_frozen_before_private_pack"
        ]
        return len(freeze_events) == 1 and store.verify_event_chain()
    except (
        OSError, RuntimeError, ValueError, KeyError, TypeError,
        json.JSONDecodeError, FloatingPointError,
    ):
        return False
