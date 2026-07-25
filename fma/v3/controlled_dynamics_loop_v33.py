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
    _target_loss_v31,
    evaluate_controlled_dynamics_worldpack_v31,
    generate_private_controlled_dynamics_worldpack_v31,
)
from .controlled_dynamics_loop_v32 import (
    ControlledDynamicsCaseReceiptV32,
    ControlledDynamicsSelectionBundleV32,
    ControlledDynamicsStepReceiptV32,
    GoalPosteriorRiskAcquisitionReceiptV32,
    _acquisition_receipts_v32,
)
from .experiment_ir import ControlledObservationReceiptV31


EXPLORATORY_SEEDS_V33 = (
    11113, 11161, 11213, 11261, 11311, 11369, 11411, 11467,
    11519, 11579, 11617, 11677, 11731, 11779, 11831, 11887,
)


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class ResourceAwareControlledDynamicsPolicyV33(StrictModel):
    schema_version: Literal["3.3"] = "3.3"
    policy_id: Identifier
    arm: ArmV31
    selection_rule: Literal[
        "clarify_then_prefrozen_random_without_replacement",
        "clarify_then_robust_goal_posterior_risk",
    ]
    may_reformulate_problem: Literal[True] = True
    clarification_budget: Literal[1] = 1
    controlled_experiment_budget: Literal[3] = 3
    known_actuator_required: Literal[True] = True
    prior_v32_failure_report_hash: Sha256
    prior_epistemic_qualification_hash: Sha256
    prior_active_design_qualification_hash: Sha256
    method_evidence_hash: Sha256
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "ResourceAwareControlledDynamicsPolicyV33":
        expected = {
            "random_bounded_inputs": (
                "clarify_then_prefrozen_random_without_replacement"
            ),
            "goal_oriented_epistemic_control": (
                "clarify_then_robust_goal_posterior_risk"
            ),
        }[self.arm]
        if self.selection_rule != expected:
            raise ValueError("V3.3 arm and selection rule disagree")
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("policy_hash does not match V3.3 policy")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("V3.3 policy is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ResourceAwareControlledDynamicsPolicyV33":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"policy_hash"}),
            policy_hash=draft.content_hash(),
        )


class ControlledDynamicsWorldPackSpecV33(StrictModel):
    schema_version: Literal["3.3"] = "3.3"
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
    bootstrap_seed: Literal[330722] = 330722
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
    prior_v32_failure_report_hash: Sha256
    frozen_delta: Literal[
        "scalar_action_budget_to_typed_resource_ledger_only"
    ] = "scalar_action_budget_to_typed_resource_ledger_only"
    frozen_at: datetime
    spec_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "ControlledDynamicsWorldPackSpecV33":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.mechanisms != list(MECHANISMS_V31):
            raise ValueError("V3.3 requires the frozen mechanism order")
        if tuple(self.seeds) != EXPLORATORY_SEEDS_V33:
            raise ValueError("V3.3 seeds do not match the frozen exploratory set")
        if self.goal_initial_state_scales != [0.75, 1.0, 1.25]:
            raise ValueError("V3.3 public goal initial-state scales changed")
        if self.maximum_steps != self.action_budget + self.clarification_budget:
            raise ValueError("V3.3 typed resource budgets do not cover maximum steps")
        if not math.isclose(
            (self.trajectory_points - 1) * self.time_step,
            self.segment_count * self.segment_duration,
            abs_tol=1e-12,
        ):
            raise ValueError("V3.3 input segments do not cover the trajectory")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match V3.3 protocol")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "spec_hash")

    def assert_sealed(self) -> None:
        if not self.spec_hash or self.spec_hash != self.content_hash():
            raise ValueError("V3.3 protocol is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsWorldPackSpecV33":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"spec_hash"}),
            spec_hash=draft.content_hash(),
        )


class EpistemicResourceLedgerV33(StrictModel):
    schema_version: Literal["3.3"] = "3.3"
    initial_clarification_budget: Literal[1] = 1
    initial_controlled_experiment_budget: Literal[3] = 3
    clarification_used: Annotated[int, Field(ge=0, le=1)]
    controlled_experiments_used: Annotated[int, Field(ge=0, le=3)]
    clarification_remaining: Annotated[int, Field(ge=0, le=1)]
    controlled_experiments_remaining: Annotated[int, Field(ge=0, le=3)]
    resources_are_not_interchangeable: Literal[True] = True
    real_world_cost_equivalence_claimed: Literal[False] = False
    ledger_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_ledger(self) -> "EpistemicResourceLedgerV33":
        if self.clarification_used + self.clarification_remaining != 1:
            raise ValueError("V3.3 clarification ledger does not balance")
        if (
            self.controlled_experiments_used
            + self.controlled_experiments_remaining != 3
        ):
            raise ValueError("V3.3 experiment ledger does not balance")
        if self.ledger_hash and self.ledger_hash != self.content_hash():
            raise ValueError("ledger_hash does not match V3.3 resource ledger")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "ledger_hash")

    def assert_sealed(self) -> None:
        if not self.ledger_hash or self.ledger_hash != self.content_hash():
            raise ValueError("V3.3 resource ledger is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EpistemicResourceLedgerV33":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"ledger_hash"}),
            ledger_hash=draft.content_hash(),
        )


class ControlledDynamicsStepReceiptV33(ControlledDynamicsStepReceiptV32):
    schema_version: Literal["3.3"] = "3.3"
    step_index: Annotated[int, Field(ge=1, le=4)]
    acquisition_receipts: list[GoalPosteriorRiskAcquisitionReceiptV32] = Field(
        default_factory=list
    )


class ControlledDynamicsCaseReceiptV33(ControlledDynamicsCaseReceiptV32):
    schema_version: Literal["3.3"] = "3.3"
    steps: list[ControlledDynamicsStepReceiptV33] = Field(min_length=1, max_length=4)
    action_budget_consumed: Annotated[int, Field(ge=0, le=4)]
    resource_ledger: EpistemicResourceLedgerV33

    @model_validator(mode="after")
    def validate_resource_binding(self) -> "ControlledDynamicsCaseReceiptV33":
        self.resource_ledger.assert_sealed()
        if self.action_budget_consumed != (
            self.resource_ledger.clarification_used
            + self.resource_ledger.controlled_experiments_used
        ):
            raise ValueError("V3.3 total steps disagree with typed resource ledger")
        return self


class ControlledDynamicsSelectionBundleV33(ControlledDynamicsSelectionBundleV32):
    schema_version: Literal["3.3"] = "3.3"
    case_receipts: list[ControlledDynamicsCaseReceiptV33] = Field(
        min_length=64, max_length=64
    )


class ResourceLedgerEvolutionReportV33(StrictModel):
    schema_version: Literal["3.3"] = "3.3"
    evolution_id: Identifier
    spec_hash: Sha256
    base_adjudication_report: ControlledDynamicsReportV31
    prior_v32_failure_report_hash: Sha256
    single_component_delta: Literal[
        "scalar_action_budget_to_typed_resource_ledger_only"
    ] = "scalar_action_budget_to_typed_resource_ledger_only"
    acquisition_gates: dict[Identifier, bool]
    resource_entitlement_parity: bool
    resource_use_parity: bool
    target_contract_parity: bool
    acquisition_candidate_ready: bool
    router_gate_passed: bool
    router_accuracy: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    status: Literal[
        "acquisition_candidate_ready_for_router_evolution_v33",
        "acquisition_candidate_failed_v33",
    ]
    acquisition_changed: Literal[False] = False
    estimator_changed: Literal[False] = False
    action_catalog_changed: Literal[False] = False
    risk_gate_changed: Literal[False] = False
    statistical_gate_changed: Literal[False] = False
    model_router_changed: Literal[False] = False
    budget_model_changed: Literal[True] = True
    baseline_target_clarification_aligned: Literal[True] = True
    overall_qualification_permitted: Literal[False] = False
    confirmation_permitted: Literal[False] = False
    created_at: datetime
    evolution_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_evolution(self) -> "ResourceLedgerEvolutionReportV33":
        _assert_timezone(self.created_at, "created_at")
        self.base_adjudication_report.assert_sealed()
        if self.base_adjudication_report.spec_hash != self.spec_hash:
            raise ValueError("V3.3 wrapper is bound to another adjudication")
        ready = all(self.acquisition_gates.values())
        if self.acquisition_candidate_ready != ready:
            raise ValueError("V3.3 acquisition readiness disagrees with gates")
        expected = (
            "acquisition_candidate_ready_for_router_evolution_v33"
            if ready else "acquisition_candidate_failed_v33"
        )
        if self.status != expected:
            raise ValueError("V3.3 status disagrees with acquisition gates")
        if self.router_gate_passed != self.base_adjudication_report.gates["routing_accuracy"]:
            raise ValueError("V3.3 router status was not inherited unchanged")
        if self.evolution_hash and self.evolution_hash != self.content_hash():
            raise ValueError("evolution_hash does not match V3.3 report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evolution_hash")

    def assert_sealed(self) -> None:
        if not self.evolution_hash or self.evolution_hash != self.content_hash():
            raise ValueError("V3.3 evolution report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ResourceLedgerEvolutionReportV33":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evolution_hash"}),
            evolution_hash=draft.content_hash(),
        )


class ControlledDynamicsManifestV33(StrictModel):
    schema_version: Literal["3.3"] = "3.3"
    run_id: Identifier
    artifact_refs: list[ArtifactRef] = Field(min_length=8)
    terminal_status: Literal[
        "acquisition_candidate_ready_for_router_evolution_v33",
        "acquisition_candidate_failed_v33",
    ]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "ControlledDynamicsManifestV33":
        _assert_timezone(self.created_at, "created_at")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match V3.3 manifest")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("V3.3 manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsManifestV33":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class ControlledDynamicsOutcomeV33:
    store: RunStore
    spec: ControlledDynamicsWorldPackSpecV33
    private_pack: PrivateControlledDynamicsWorldPackV31
    baseline_policy: ResourceAwareControlledDynamicsPolicyV33
    candidate_policy: ResourceAwareControlledDynamicsPolicyV33
    baseline_bundle: ControlledDynamicsSelectionBundleV33
    candidate_bundle: ControlledDynamicsSelectionBundleV33
    evolution_report: ResourceLedgerEvolutionReportV33
    manifest: ControlledDynamicsManifestV33


def default_controlled_dynamics_policies_v33(
    *,
    prior_v32_failure_report_hash: str,
    prior_epistemic_qualification_hash: str,
    prior_active_design_qualification_hash: str,
    method_evidence_hash: str,
) -> tuple[
    ResourceAwareControlledDynamicsPolicyV33,
    ResourceAwareControlledDynamicsPolicyV33,
]:
    shared = dict(
        prior_v32_failure_report_hash=prior_v32_failure_report_hash,
        prior_epistemic_qualification_hash=prior_epistemic_qualification_hash,
        prior_active_design_qualification_hash=prior_active_design_qualification_hash,
        method_evidence_hash=method_evidence_hash,
    )
    return (
        ResourceAwareControlledDynamicsPolicyV33.seal(
            policy_id="clarified_random_bounded_inputs_v33",
            arm="random_bounded_inputs",
            selection_rule="clarify_then_prefrozen_random_without_replacement",
            **shared,
        ),
        ResourceAwareControlledDynamicsPolicyV33.seal(
            policy_id="clarified_goal_posterior_risk_v33",
            arm="goal_oriented_epistemic_control",
            selection_rule="clarify_then_robust_goal_posterior_risk",
            **shared,
        ),
    )


def default_controlled_dynamics_exploratory_spec_v33(
    *,
    baseline_policy_hash: str,
    candidate_policy_hash: str,
    method_evidence_hash: str,
    prior_epistemic_qualification_hash: str,
    prior_active_design_qualification_hash: str,
    prior_v32_failure_report_hash: str,
    frozen_at: datetime | None = None,
) -> ControlledDynamicsWorldPackSpecV33:
    return ControlledDynamicsWorldPackSpecV33.seal(
        experiment_id="controlled_dynamics_resource_ledger_exploratory_v33",
        mechanisms=list(MECHANISMS_V31),
        seeds=list(EXPLORATORY_SEEDS_V33),
        baseline_policy_hash=baseline_policy_hash,
        candidate_policy_hash=candidate_policy_hash,
        method_evidence_hash=method_evidence_hash,
        prior_epistemic_qualification_hash=prior_epistemic_qualification_hash,
        prior_active_design_qualification_hash=prior_active_design_qualification_hash,
        prior_v32_failure_report_hash=prior_v32_failure_report_hash,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


def _execute_case_v33(
    spec: ControlledDynamicsWorldPackSpecV33,
    private_case: PrivateControlledDynamicsCaseV31,
    policy: ResourceAwareControlledDynamicsPolicyV33,
    *,
    executed_at: datetime,
) -> ControlledDynamicsCaseReceiptV33:
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

    for step_index in range(1, spec.maximum_steps + 1):
        if contract.target_status == "default_unverified":
            if clarification_remaining < 1:
                raise RuntimeError("V3.3 target needs unavailable clarification budget")
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
            ranked = sorted(
                admissible_actions, key=lambda item: priority[item.action_id]
            )
        else:
            ranked = sorted(
                admissible_actions,
                key=lambda item: (
                    by_hash[item.action_hash].ranking_score, item.action_id,
                ),
                reverse=True,
            )
        if not ranked:
            best = max(
                acquisitions,
                key=lambda item: (item.ranking_score, item.action_hash),
            )
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
        selected = ranked[0]
        selected_acquisition = by_hash[selected.action_hash]
        permission = _permission_v31(
            selected_acquisition.acquisition_hash,
            public.envelope.envelope_hash,
            data_quality_passed=True,
            admissible=True,
            budget_before=experiment_budget,
            decided_at=executed_at,
        )
        if permission.decision != "allow_synthetic":
            raise RuntimeError("V3.3 admissible action was not allowed")
        observation = private_case.action_observations[selected.action_id]
        observation.assert_sealed()
        if observation.empirical_peak_state_ratio > 1.0:
            raise RuntimeError("V3.3 hidden Reality Interface detected state-bound violation")
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
    return ControlledDynamicsCaseReceiptV33.seal(
        receipt_id=f"receipt_{policy.arm}_{public.case_id}_v33",
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
    )


def execute_controlled_dynamics_policy_v33(
    spec: ControlledDynamicsWorldPackSpecV33,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    policy: ResourceAwareControlledDynamicsPolicyV33,
    *,
    executed_at: datetime | None = None,
) -> ControlledDynamicsSelectionBundleV33:
    spec.assert_sealed()
    private_pack.assert_sealed()
    policy.assert_sealed()
    if private_pack.spec_hash != spec.spec_hash:
        raise ValueError("V3.3 private pack belongs to another protocol")
    expected = (
        spec.baseline_policy_hash
        if policy.arm == "random_bounded_inputs"
        else spec.candidate_policy_hash
    )
    if policy.policy_hash != expected:
        raise ValueError("V3.3 policy is not frozen in the protocol")
    at = executed_at or datetime.now(timezone.utc)
    receipts = [
        _execute_case_v33(spec, private_case, policy, executed_at=at)
        for private_case in private_pack.cases
    ]
    return ControlledDynamicsSelectionBundleV33.seal(
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


def evaluate_controlled_dynamics_worldpack_v33(
    spec: ControlledDynamicsWorldPackSpecV33,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    baseline: ControlledDynamicsSelectionBundleV33,
    candidate: ControlledDynamicsSelectionBundleV33,
    *,
    evaluated_at: datetime | None = None,
) -> ResourceLedgerEvolutionReportV33:
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
    })
    ready = all(acquisition_gates.values())
    return ResourceLedgerEvolutionReportV33.seal(
        evolution_id="controlled_dynamics_resource_ledger_exploratory_v33",
        spec_hash=spec.spec_hash,
        base_adjudication_report=report,
        prior_v32_failure_report_hash=spec.prior_v32_failure_report_hash,
        acquisition_gates=acquisition_gates,
        resource_entitlement_parity=entitlement_parity,
        resource_use_parity=use_parity,
        target_contract_parity=contract_parity,
        acquisition_candidate_ready=ready,
        router_gate_passed=report.gates["routing_accuracy"],
        router_accuracy=report.routing_accuracy,
        status=(
            "acquisition_candidate_ready_for_router_evolution_v33"
            if ready else "acquisition_candidate_failed_v33"
        ),
        created_at=evaluated_at or datetime.now(timezone.utc),
    )


def run_controlled_dynamics_worldpack_v33(
    output_root: str | Path,
    *,
    spec: ControlledDynamicsWorldPackSpecV33,
    baseline_policy: ResourceAwareControlledDynamicsPolicyV33,
    candidate_policy: ResourceAwareControlledDynamicsPolicyV33,
    evaluated_at: datetime | None = None,
    run_id: str | None = None,
) -> ControlledDynamicsOutcomeV33:
    spec.assert_sealed()
    baseline_policy.assert_sealed()
    candidate_policy.assert_sealed()
    if baseline_policy.policy_hash != spec.baseline_policy_hash:
        raise ValueError("V3.3 baseline is not frozen in the protocol")
    if candidate_policy.policy_hash != spec.candidate_policy_hash:
        raise ValueError("V3.3 candidate is not frozen in the protocol")
    at = evaluated_at or datetime.now(timezone.utc)
    store = RunStore(
        output_root,
        run_id=run_id or f"controlled-dynamics-v33-{uuid4().hex[:10]}",
    )
    refs = [
        store.put_artifact("controlled_dynamics_spec_v33", spec),
        store.put_artifact("controlled_dynamics_baseline_policy_v33", baseline_policy),
        store.put_artifact("controlled_dynamics_candidate_policy_v33", candidate_policy),
    ]
    store.emit("controlled_dynamics_v33_protocol_frozen_before_private_pack", {
        "spec_hash": spec.spec_hash,
        "prior_v32_failure_report_hash": spec.prior_v32_failure_report_hash,
        "single_component_delta": spec.frozen_delta,
    })
    private_pack = generate_private_controlled_dynamics_worldpack_v31(
        spec, generated_at=at
    )
    baseline = execute_controlled_dynamics_policy_v33(
        spec, private_pack, baseline_policy, executed_at=at
    )
    candidate = execute_controlled_dynamics_policy_v33(
        spec, private_pack, candidate_policy, executed_at=at
    )
    evolution = evaluate_controlled_dynamics_worldpack_v33(
        spec, private_pack, baseline, candidate, evaluated_at=at
    )
    refs.extend([
        store.put_artifact("private_controlled_dynamics_worldpack_v33", private_pack),
        store.put_artifact("controlled_dynamics_baseline_bundle_v33", baseline),
        store.put_artifact("controlled_dynamics_candidate_bundle_v33", candidate),
        store.put_artifact(
            "controlled_dynamics_base_report_v33",
            evolution.base_adjudication_report,
        ),
        store.put_artifact("controlled_dynamics_evolution_report_v33", evolution),
    ])
    manifest = ControlledDynamicsManifestV33.seal(
        run_id=store.run_id,
        artifact_refs=refs,
        terminal_status=evolution.status,
        created_at=at,
    )
    manifest_ref = store.put_artifact("controlled_dynamics_manifest_v33", manifest)
    store.emit("controlled_dynamics_v33_worldpack_adjudicated", {
        "manifest_ref": manifest_ref.model_dump(mode="json")
    })
    if not verify_controlled_dynamics_run_v33(store.run_directory):
        raise RuntimeError("V3.3 controlled-dynamics run failed independent verification")
    return ControlledDynamicsOutcomeV33(
        store, spec, private_pack, baseline_policy, candidate_policy,
        baseline, candidate, evolution, manifest,
    )


def verify_controlled_dynamics_run_v33(run_directory: str | Path) -> bool:
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
            if item.kind == "controlled_dynamics_manifest_v33"
        ]
        if len(manifest_refs) != 1:
            return False
        manifest = ControlledDynamicsManifestV33.model_validate(
            store.load_artifact(manifest_refs[0])
        )
        manifest.assert_sealed()

        def load_one(kind: str, model):
            references = [item for item in manifest.artifact_refs if item.kind == kind]
            if len(references) != 1:
                raise RuntimeError(f"V3.3 manifest needs exactly one {kind}")
            return model.model_validate(store.load_artifact(references[0]))

        spec = load_one(
            "controlled_dynamics_spec_v33", ControlledDynamicsWorldPackSpecV33
        )
        baseline_policy = load_one(
            "controlled_dynamics_baseline_policy_v33",
            ResourceAwareControlledDynamicsPolicyV33,
        )
        candidate_policy = load_one(
            "controlled_dynamics_candidate_policy_v33",
            ResourceAwareControlledDynamicsPolicyV33,
        )
        private_pack = load_one(
            "private_controlled_dynamics_worldpack_v33",
            PrivateControlledDynamicsWorldPackV31,
        )
        baseline = load_one(
            "controlled_dynamics_baseline_bundle_v33",
            ControlledDynamicsSelectionBundleV33,
        )
        candidate = load_one(
            "controlled_dynamics_candidate_bundle_v33",
            ControlledDynamicsSelectionBundleV33,
        )
        base_report = load_one(
            "controlled_dynamics_base_report_v33", ControlledDynamicsReportV31
        )
        evolution = load_one(
            "controlled_dynamics_evolution_report_v33",
            ResourceLedgerEvolutionReportV33,
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
        replay_baseline = execute_controlled_dynamics_policy_v33(
            spec, private_pack, baseline_policy, executed_at=executed_at
        )
        replay_candidate = execute_controlled_dynamics_policy_v33(
            spec, private_pack, candidate_policy, executed_at=executed_at
        )
        if (
            replay_baseline.bundle_hash != baseline.bundle_hash
            or replay_candidate.bundle_hash != candidate.bundle_hash
        ):
            return False
        recomputed = evaluate_controlled_dynamics_worldpack_v33(
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
            == "controlled_dynamics_v33_protocol_frozen_before_private_pack"
        ]
        return len(freeze_events) == 1 and store.verify_event_chain()
    except (
        OSError, RuntimeError, ValueError, KeyError, TypeError,
        json.JSONDecodeError, FloatingPointError,
    ):
        return False
