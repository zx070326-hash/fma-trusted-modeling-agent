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

from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore
from fma.v2.schemas import Identifier, Sha256, _assert_timezone

from .controlled_dynamics_loop import (
    MECHANISMS_V31,
    PrivateControlledDynamicsCaseV31,
    PrivateControlledDynamicsWorldPackV31,
    _fit_model_v31,
    _target_loss_v31,
    generate_private_controlled_dynamics_worldpack_v31,
)
from .controlled_dynamics_loop_v34 import (
    AdapterArmV34,
    ControlledDynamicsCaseReceiptV34,
    ControlledDynamicsSelectionBundleV34,
    PlanAbstentionV34,
    _calibration_v34,
    _hash_without,
    _select_v332_plan_v34,
    _stream_intervention_v34,
)
from .controlled_dynamics_loop_v341 import ControlledDynamicsWorldPackSpecV341


EXPLORATORY_SEEDS_V35 = (
    16001, 16057, 16111, 16157, 16217, 16267, 16319, 16369,
    16421, 16477, 16529, 16573, 16631, 16673, 16729, 16787,
)

AcquisitionRoleV35 = Literal[
    "shared_random_baseline",
    "paired_advantage_unguarded",
    "paired_advantage_persistent_guard",
]


class GuardedAcquisitionPolicyV35(StrictModel):
    schema_version: Literal["3.5"] = "3.5"
    policy_id: Identifier
    acquisition_role: AcquisitionRoleV35
    arm: AdapterArmV34
    selection_rule: Literal[
        "two_shared_anchors_then_prefrozen_random",
        "two_shared_anchors_then_v332_paired_advantage",
    ]
    execution_rule: Literal[
        "execute_selected_action_without_online_interruption",
        "two_consecutive_mismatch_then_monotone_zero_fallback",
    ]
    consecutive_exceedances_required: Literal[2] = 2
    authority_may_only_decrease: Literal[True] = True
    real_world_execution_permitted: Literal[False] = False
    prior_v341_adapter_report_hash: Sha256
    prior_v34_failure_report_hash: Sha256
    prior_v332_failure_report_hash: Sha256
    prior_epistemic_qualification_hash: Sha256
    prior_active_design_qualification_hash: Sha256
    method_evidence_hash: Sha256
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "GuardedAcquisitionPolicyV35":
        expected = {
            "shared_random_baseline": (
                "unguarded_full_action",
                "two_shared_anchors_then_prefrozen_random",
                "execute_selected_action_without_online_interruption",
            ),
            "paired_advantage_unguarded": (
                "unguarded_full_action",
                "two_shared_anchors_then_v332_paired_advantage",
                "execute_selected_action_without_online_interruption",
            ),
            "paired_advantage_persistent_guard": (
                "interruptible_online_guard",
                "two_shared_anchors_then_v332_paired_advantage",
                "two_consecutive_mismatch_then_monotone_zero_fallback",
            ),
        }[self.acquisition_role]
        if (self.arm, self.selection_rule, self.execution_rule) != expected:
            raise ValueError("V3.5 policy role disagrees with selection/execution")
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("policy_hash does not match V3.5 policy")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("V3.5 policy is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "GuardedAcquisitionPolicyV35":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"policy_hash"}),
            policy_hash=draft.content_hash(),
        )


class ControlledDynamicsWorldPackSpecV35(ControlledDynamicsWorldPackSpecV341):
    schema_version: Literal["3.5"] = "3.5"
    seeds: list[int] = Field(min_length=16, max_length=16)
    bootstrap_seed: Literal[35722] = 35722
    diagnostic_policy_hash: Sha256
    prior_v341_adapter_report_hash: Sha256
    frozen_delta: Literal[
        "three_arm_random_vs_acquisition_vs_guarded_acquisition_factorial"
    ] = "three_arm_random_vs_acquisition_vs_guarded_acquisition_factorial"

    @model_validator(mode="after")
    def validate_spec(self) -> "ControlledDynamicsWorldPackSpecV35":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.mechanisms != list(MECHANISMS_V31):
            raise ValueError("V3.5 requires the frozen mechanism order")
        if tuple(self.seeds) != EXPLORATORY_SEEDS_V35:
            raise ValueError("V3.5 seeds do not match the fresh exploratory set")
        if self.goal_initial_state_scales != [0.75, 1.0, 1.25]:
            raise ValueError("V3.5 public goal initial-state scales changed")
        if self.maximum_steps != self.action_budget + self.clarification_budget:
            raise ValueError("V3.5 resource budgets do not cover maximum steps")
        if not math.isclose(
            (self.trajectory_points - 1) * self.time_step,
            self.segment_count * self.segment_duration,
            abs_tol=1e-12,
        ):
            raise ValueError("V3.5 input segments do not cover trajectory")
        if len({
            self.baseline_policy_hash,
            self.diagnostic_policy_hash,
            self.candidate_policy_hash,
        }) != 3:
            raise ValueError("V3.5 requires three distinct frozen policies")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match V3.5 protocol")
        return self

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsWorldPackSpecV35":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"spec_hash"}),
            spec_hash=draft.content_hash(),
        )


class GuardedAcquisitionEvolutionReportV35(StrictModel):
    schema_version: Literal["3.5"] = "3.5"
    evolution_id: Identifier
    spec_hash: Sha256
    prior_v341_adapter_report_hash: Sha256
    factorial_design: Literal[
        "R_shared_random_A_paired_advantage_AG_persistent_guard"
    ] = "R_shared_random_A_paired_advantage_AG_persistent_guard"
    eligible_case_count: Annotated[int, Field(ge=1)]
    acquisition_change_count: Annotated[int, Field(ge=0)]
    guard_interruption_count: Annotated[int, Field(ge=0)]
    guard_termination_count: Annotated[int, Field(ge=0)]
    random_vs_unguarded_macro_improvement: Annotated[
        float, Field(allow_inf_nan=False)
    ]
    random_vs_unguarded_ci_low: Annotated[float, Field(allow_inf_nan=False)]
    random_vs_unguarded_ci_high: Annotated[float, Field(allow_inf_nan=False)]
    unguarded_vs_guarded_macro_improvement: Annotated[
        float, Field(allow_inf_nan=False)
    ]
    unguarded_vs_guarded_ci_low: Annotated[float, Field(allow_inf_nan=False)]
    unguarded_vs_guarded_ci_high: Annotated[float, Field(allow_inf_nan=False)]
    package_macro_improvement: Annotated[float, Field(allow_inf_nan=False)]
    package_ci_low: Annotated[float, Field(allow_inf_nan=False)]
    package_ci_high: Annotated[float, Field(allow_inf_nan=False)]
    package_mechanism_mean_improvements: dict[Identifier, Annotated[
        float, Field(allow_inf_nan=False)
    ]]
    package_material_negative_transfer_count: Annotated[int, Field(ge=0)]
    package_material_negative_transfer_rate: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ]
    random_max_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    unguarded_max_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    guarded_max_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    gates: dict[Identifier, bool]
    guarded_acquisition_ready: bool
    router_experiment_permitted: bool
    status: Literal[
        "guarded_acquisition_ready_for_router_experiment_v35",
        "guarded_acquisition_failed_v35",
    ]
    estimator_changed: Literal[False] = False
    target_evaluator_changed: Literal[False] = False
    action_catalog_changed: Literal[False] = False
    statistical_gates_changed: Literal[False] = False
    router_changed: Literal[False] = False
    overall_qualification_permitted: Literal[False] = False
    confirmation_permitted: Literal[False] = False
    real_world_authorization_permitted: Literal[False] = False
    created_at: datetime
    evolution_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "GuardedAcquisitionEvolutionReportV35":
        _assert_timezone(self.created_at, "created_at")
        ready = all(self.gates.values())
        if self.guarded_acquisition_ready != ready:
            raise ValueError("V3.5 readiness disagrees with gates")
        if self.router_experiment_permitted != ready:
            raise ValueError("V3.5 router permission disagrees with readiness")
        expected = (
            "guarded_acquisition_ready_for_router_experiment_v35"
            if ready else "guarded_acquisition_failed_v35"
        )
        if self.status != expected:
            raise ValueError("V3.5 status disagrees with readiness")
        if self.evolution_hash and self.evolution_hash != self.content_hash():
            raise ValueError("evolution_hash does not match V3.5 report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evolution_hash")

    def assert_sealed(self) -> None:
        if not self.evolution_hash or self.evolution_hash != self.content_hash():
            raise ValueError("V3.5 report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "GuardedAcquisitionEvolutionReportV35":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evolution_hash"}),
            evolution_hash=draft.content_hash(),
        )


class ControlledDynamicsManifestV35(StrictModel):
    schema_version: Literal["3.5"] = "3.5"
    run_id: Identifier
    artifact_refs: list[ArtifactRef] = Field(min_length=8)
    terminal_status: Literal[
        "guarded_acquisition_ready_for_router_experiment_v35",
        "guarded_acquisition_failed_v35",
    ]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "ControlledDynamicsManifestV35":
        _assert_timezone(self.created_at, "created_at")
        if len({item.kind for item in self.artifact_refs}) != len(self.artifact_refs):
            raise ValueError("V3.5 manifest artifact kinds must be unique")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match V3.5 manifest")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("V3.5 manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsManifestV35":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class ControlledDynamicsOutcomeV35:
    store: RunStore
    spec: ControlledDynamicsWorldPackSpecV35
    private_pack: PrivateControlledDynamicsWorldPackV31
    baseline_policy: GuardedAcquisitionPolicyV35
    diagnostic_policy: GuardedAcquisitionPolicyV35
    candidate_policy: GuardedAcquisitionPolicyV35
    baseline_bundle: ControlledDynamicsSelectionBundleV34
    diagnostic_bundle: ControlledDynamicsSelectionBundleV34
    candidate_bundle: ControlledDynamicsSelectionBundleV34
    evolution_report: GuardedAcquisitionEvolutionReportV35
    manifest: ControlledDynamicsManifestV35


def default_controlled_dynamics_policies_v35(
    *,
    prior_v341_adapter_report_hash: str,
    prior_v34_failure_report_hash: str,
    prior_v332_failure_report_hash: str,
    prior_epistemic_qualification_hash: str,
    prior_active_design_qualification_hash: str,
    method_evidence_hash: str,
) -> tuple[
    GuardedAcquisitionPolicyV35,
    GuardedAcquisitionPolicyV35,
    GuardedAcquisitionPolicyV35,
]:
    shared = dict(
        prior_v341_adapter_report_hash=prior_v341_adapter_report_hash,
        prior_v34_failure_report_hash=prior_v34_failure_report_hash,
        prior_v332_failure_report_hash=prior_v332_failure_report_hash,
        prior_epistemic_qualification_hash=prior_epistemic_qualification_hash,
        prior_active_design_qualification_hash=prior_active_design_qualification_hash,
        method_evidence_hash=method_evidence_hash,
    )
    return (
        GuardedAcquisitionPolicyV35.seal(
            policy_id="shared_random_baseline_v35",
            acquisition_role="shared_random_baseline",
            arm="unguarded_full_action",
            selection_rule="two_shared_anchors_then_prefrozen_random",
            execution_rule="execute_selected_action_without_online_interruption",
            **shared,
        ),
        GuardedAcquisitionPolicyV35.seal(
            policy_id="paired_advantage_unguarded_v35",
            acquisition_role="paired_advantage_unguarded",
            arm="unguarded_full_action",
            selection_rule="two_shared_anchors_then_v332_paired_advantage",
            execution_rule="execute_selected_action_without_online_interruption",
            **shared,
        ),
        GuardedAcquisitionPolicyV35.seal(
            policy_id="paired_advantage_persistent_guard_v35",
            acquisition_role="paired_advantage_persistent_guard",
            arm="interruptible_online_guard",
            selection_rule="two_shared_anchors_then_v332_paired_advantage",
            execution_rule="two_consecutive_mismatch_then_monotone_zero_fallback",
            **shared,
        ),
    )


def default_controlled_dynamics_exploratory_spec_v35(
    *,
    baseline_policy_hash: str,
    diagnostic_policy_hash: str,
    candidate_policy_hash: str,
    method_evidence_hash: str,
    prior_epistemic_qualification_hash: str,
    prior_active_design_qualification_hash: str,
    prior_v332_failure_report_hash: str,
    prior_v34_failure_report_hash: str,
    prior_v341_adapter_report_hash: str,
    frozen_at: datetime | None = None,
) -> ControlledDynamicsWorldPackSpecV35:
    return ControlledDynamicsWorldPackSpecV35.seal(
        experiment_id="controlled_dynamics_guarded_acquisition_factorial_v35",
        mechanisms=list(MECHANISMS_V31),
        seeds=list(EXPLORATORY_SEEDS_V35),
        baseline_policy_hash=baseline_policy_hash,
        diagnostic_policy_hash=diagnostic_policy_hash,
        candidate_policy_hash=candidate_policy_hash,
        method_evidence_hash=method_evidence_hash,
        prior_epistemic_qualification_hash=prior_epistemic_qualification_hash,
        prior_active_design_qualification_hash=prior_active_design_qualification_hash,
        prior_v332_failure_report_hash=prior_v332_failure_report_hash,
        prior_v34_failure_report_hash=prior_v34_failure_report_hash,
        prior_v341_adapter_report_hash=prior_v341_adapter_report_hash,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


def _abstained_case_v35(
    private_case: PrivateControlledDynamicsCaseV31,
    policy: GuardedAcquisitionPolicyV35,
    target,
    clarification_used: bool,
    *,
    data_quality_passed: bool,
    reason: str,
    executed_at: datetime,
) -> ControlledDynamicsCaseReceiptV34:
    public = private_case.public_case
    return ControlledDynamicsCaseReceiptV34.seal(
        receipt_id=f"receipt_{policy.acquisition_role}_{public.case_id}_v35",
        case_id=public.case_id,
        public_case_hash=public.public_hash,
        policy_hash=policy.policy_hash,
        arm=policy.arm,
        data_quality_passed=data_quality_passed,
        plan_admissible=False,
        abstention_reason=reason,
        clarification_used=clarification_used,
        decision_target=target,
        anchor_action_ids=[],
        anchor_action_hashes=[],
        anchor_observation_hashes=[],
        segment_receipts=[],
        performance_eligible=False,
        executed_at=executed_at,
    )


def _execute_case_v35(
    spec: ControlledDynamicsWorldPackSpecV35,
    private_case: PrivateControlledDynamicsCaseV31,
    policy: GuardedAcquisitionPolicyV35,
    *,
    executed_at: datetime,
) -> ControlledDynamicsCaseReceiptV34:
    public = private_case.public_case
    target = (
        private_case.true_decision_target
        if public.initial_contract.target_status == "default_unverified"
        else public.initial_contract.decision_target
    )
    clarification_used = public.initial_contract.target_status == "default_unverified"
    if public.pilot.quality_flags:
        return _abstained_case_v35(
            private_case,
            policy,
            target,
            clarification_used,
            data_quality_passed=False,
            reason="pilot_data_quality",
            executed_at=executed_at,
        )
    try:
        target, anchors, anchor_observations, active, fallback, trust = (
            _select_v332_plan_v34(spec, private_case)
        )
    except PlanAbstentionV34 as exc:
        return _abstained_case_v35(
            private_case,
            policy,
            target,
            clarification_used,
            data_quality_passed=True,
            reason=exc.reason,
            executed_at=executed_at,
        )
    calibration = _calibration_v34(
        spec, private_case, anchors, anchor_observations
    )
    if policy.acquisition_role == "shared_random_baseline":
        selected = fallback
        selected_mode: Literal["active", "prefrozen_fallback"] = (
            "prefrozen_fallback"
        )
    else:
        selected_mode = (
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
        receipt_id=f"receipt_{policy.acquisition_role}_{public.case_id}_v35",
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


def execute_controlled_dynamics_policy_v35(
    spec: ControlledDynamicsWorldPackSpecV35,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    policy: GuardedAcquisitionPolicyV35,
    *,
    executed_at: datetime | None = None,
) -> ControlledDynamicsSelectionBundleV34:
    spec.assert_sealed()
    private_pack.assert_sealed()
    policy.assert_sealed()
    if private_pack.spec_hash != spec.spec_hash:
        raise ValueError("V3.5 private pack belongs to another protocol")
    expected = {
        "shared_random_baseline": spec.baseline_policy_hash,
        "paired_advantage_unguarded": spec.diagnostic_policy_hash,
        "paired_advantage_persistent_guard": spec.candidate_policy_hash,
    }[policy.acquisition_role]
    if policy.policy_hash != expected:
        raise ValueError("V3.5 policy is not frozen in protocol")
    at = executed_at or datetime.now(timezone.utc)
    receipts = [
        _execute_case_v35(spec, private_case, policy, executed_at=at)
        for private_case in private_pack.cases
    ]
    return ControlledDynamicsSelectionBundleV34.seal(
        spec_hash=spec.spec_hash,
        private_pack_hash=private_pack.pack_hash,
        policy_hash=policy.policy_hash,
        arm=policy.arm,
        case_receipts=receipts,
    )


def _paired_summary_v35(
    values: np.ndarray,
    samples: np.ndarray,
) -> tuple[float, float, float]:
    bootstrap = np.mean(values[samples], axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return float(np.mean(values)), float(low), float(high)


def evaluate_controlled_dynamics_worldpack_v35(
    spec: ControlledDynamicsWorldPackSpecV35,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    baseline: ControlledDynamicsSelectionBundleV34,
    diagnostic: ControlledDynamicsSelectionBundleV34,
    candidate: ControlledDynamicsSelectionBundleV34,
    *,
    evaluated_at: datetime | None = None,
) -> GuardedAcquisitionEvolutionReportV35:
    spec.assert_sealed()
    private_pack.assert_sealed()
    for bundle in (baseline, diagnostic, candidate):
        bundle.assert_sealed()
        if bundle.spec_hash != spec.spec_hash:
            raise ValueError("V3.5 bundle belongs to another protocol")
        if bundle.private_pack_hash != private_pack.pack_hash:
            raise ValueError("V3.5 bundle belongs to another private pack")
    random_by = {item.case_id: item for item in baseline.case_receipts}
    active_by = {item.case_id: item for item in diagnostic.case_receipts}
    guarded_by = {item.case_id: item for item in candidate.case_receipts}
    private_by = {item.public_case.case_id: item for item in private_pack.cases}
    case_ids = list(random_by)

    def same_all(case_id: str, field: str) -> bool:
        values = [
            getattr(items[case_id], field)
            for items in (random_by, active_by, guarded_by)
        ]
        return values[0] == values[1] == values[2]

    shared_context_parity = all(
        same_all(case_id, "decision_target")
        and same_all(case_id, "anchor_action_hashes")
        and same_all(case_id, "anchor_observation_hashes")
        and same_all(case_id, "noise_schedule_hash")
        for case_id in case_ids
    )
    abstention_parity = all(
        same_all(case_id, "plan_admissible")
        and same_all(case_id, "abstention_reason")
        for case_id in case_ids
    )
    trust_parity = all(
        len({
            None if items[case_id].trust_decision is None
            else items[case_id].trust_decision.trust_hash
            for items in (random_by, active_by, guarded_by)
        }) == 1
        for case_id in case_ids
    )
    calibration_parity = all(
        len({
            None if items[case_id].calibration is None
            else items[case_id].calibration.calibration_hash
            for items in (random_by, active_by, guarded_by)
        }) == 1
        for case_id in case_ids
    )
    complete_ids = [
        case_id for case_id in case_ids if random_by[case_id].plan_admissible
    ]
    baseline_binding = all(
        random_by[case_id].trust_decision is not None
        and random_by[case_id].selected_action_hash
        == random_by[case_id].trust_decision.fallback_action_hash
        for case_id in complete_ids
    )
    candidate_binding = all(
        active_by[case_id].trust_decision is not None
        and guarded_by[case_id].trust_decision is not None
        and active_by[case_id].selected_action_hash
        == active_by[case_id].trust_decision.selected_action_hash
        and guarded_by[case_id].selected_action_hash
        == guarded_by[case_id].trust_decision.selected_action_hash
        and active_by[case_id].selected_action_hash
        == guarded_by[case_id].selected_action_hash
        for case_id in complete_ids
    )
    acquisition_changes = sum(
        active_by[case_id].selected_action_hash
        != random_by[case_id].selected_action_hash
        for case_id in complete_ids
    )
    receipts_complete = all(
        (not item.plan_admissible) or (
            item.calibration is not None
            and item.observation is not None
            and item.executed_intervention is not None
            and item.exposure_ledger is not None
            and len(item.segment_receipts)
            == item.exposure_ledger.executed_segment_count
        )
        for bundle in (baseline, diagnostic, candidate)
        for item in bundle.case_receipts
    )
    exposure_dominance = all(
        guarded_by[case_id].exposure_ledger is None or all(
            comparison[case_id].exposure_ledger is not None
            and guarded_by[case_id].exposure_ledger.used_duration
            <= comparison[case_id].exposure_ledger.used_duration + 1e-12
            and guarded_by[case_id].exposure_ledger.used_energy
            <= comparison[case_id].exposure_ledger.used_energy + 1e-12
            and guarded_by[case_id].exposure_ledger.used_peak_amplitude
            <= comparison[case_id].exposure_ledger.used_peak_amplitude + 1e-12
            and guarded_by[case_id].exposure_ledger.used_switch_count
            <= comparison[case_id].exposure_ledger.used_switch_count
            for comparison in (random_by, active_by)
        )
        for case_id in case_ids
    )
    state_violations = sum(
        item.exposure_ledger.state_envelope_violation_count
        for bundle in (baseline, diagnostic, candidate)
        for item in bundle.case_receipts
        if item.exposure_ledger is not None
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
        case_id for case_id in case_ids
        if all(
            items[case_id].target_loss is not None
            for items in (random_by, active_by, guarded_by)
        )
    ]
    if not eligible_ids:
        raise RuntimeError("V3.5 evaluation has no eligible cases")
    r = np.asarray([random_by[x].target_loss for x in eligible_ids], dtype=float)
    a = np.asarray([active_by[x].target_loss for x in eligible_ids], dtype=float)
    g = np.asarray([guarded_by[x].target_loss for x in eligible_ids], dtype=float)
    random = np.random.default_rng(spec.bootstrap_seed)
    samples = random.integers(
        0, len(eligible_ids), size=(spec.bootstrap_replicates, len(eligible_ids))
    )
    acq_mean, acq_low, acq_high = _paired_summary_v35(r - a, samples)
    guard_mean, guard_low, guard_high = _paired_summary_v35(a - g, samples)
    package = r - g
    package_mean, package_low, package_high = _paired_summary_v35(package, samples)
    mechanism_means = {
        mechanism: float(np.mean([
            package[index]
            for index, case_id in enumerate(eligible_ids)
            if private_by[case_id].mechanism == mechanism
        ]))
        for mechanism in MECHANISMS_V31
        if any(private_by[case_id].mechanism == mechanism for case_id in eligible_ids)
    }
    negative_count = int(
        np.sum(package < -spec.material_negative_transfer)
    )
    negative_rate = negative_count / len(package)
    gates = {
        "shared_target_anchor_noise_parity": shared_context_parity,
        "paired_abstention_parity": abstention_parity,
        "trust_decision_parity": trust_parity,
        "mismatch_calibration_parity": calibration_parity,
        "random_baseline_bound_to_prefrozen_fallback": baseline_binding,
        "acquisition_arms_bound_to_same_trust_selection": candidate_binding,
        "minimum_acquisition_change_exercised": acquisition_changes >= 1,
        "all_receipts_complete": receipts_complete,
        "guarded_exposure_dominated_by_other_arms": exposure_dominance,
        "zero_synthetic_state_envelope_violations": state_violations == 0,
        "package_macro_improvement_lower_bound": package_low >= 0.0,
        "package_mechanism_non_regression": min(mechanism_means.values())
        >= -spec.maximum_mechanism_regression,
        "package_negative_transfer_upper_bound": negative_rate
        <= spec.maximum_guard_negative_transfer_rate,
        "package_worst_case_loss_non_regression": float(np.max(g))
        <= float(np.max(r)) + 1e-12,
    }
    ready = all(gates.values())
    return GuardedAcquisitionEvolutionReportV35.seal(
        evolution_id="controlled_dynamics_guarded_acquisition_factorial_v35",
        spec_hash=spec.spec_hash,
        prior_v341_adapter_report_hash=spec.prior_v341_adapter_report_hash,
        eligible_case_count=len(eligible_ids),
        acquisition_change_count=acquisition_changes,
        guard_interruption_count=interruptions,
        guard_termination_count=terminations,
        random_vs_unguarded_macro_improvement=acq_mean,
        random_vs_unguarded_ci_low=acq_low,
        random_vs_unguarded_ci_high=acq_high,
        unguarded_vs_guarded_macro_improvement=guard_mean,
        unguarded_vs_guarded_ci_low=guard_low,
        unguarded_vs_guarded_ci_high=guard_high,
        package_macro_improvement=package_mean,
        package_ci_low=package_low,
        package_ci_high=package_high,
        package_mechanism_mean_improvements=mechanism_means,
        package_material_negative_transfer_count=negative_count,
        package_material_negative_transfer_rate=negative_rate,
        random_max_target_loss=float(np.max(r)),
        unguarded_max_target_loss=float(np.max(a)),
        guarded_max_target_loss=float(np.max(g)),
        gates=gates,
        guarded_acquisition_ready=ready,
        router_experiment_permitted=ready,
        status=(
            "guarded_acquisition_ready_for_router_experiment_v35"
            if ready else "guarded_acquisition_failed_v35"
        ),
        created_at=evaluated_at or datetime.now(timezone.utc),
    )


def run_controlled_dynamics_worldpack_v35(
    output_root: str | Path,
    *,
    spec: ControlledDynamicsWorldPackSpecV35,
    baseline_policy: GuardedAcquisitionPolicyV35,
    diagnostic_policy: GuardedAcquisitionPolicyV35,
    candidate_policy: GuardedAcquisitionPolicyV35,
    evaluated_at: datetime | None = None,
    run_id: str | None = None,
) -> ControlledDynamicsOutcomeV35:
    spec.assert_sealed()
    for policy in (baseline_policy, diagnostic_policy, candidate_policy):
        policy.assert_sealed()
    if (
        baseline_policy.policy_hash != spec.baseline_policy_hash
        or diagnostic_policy.policy_hash != spec.diagnostic_policy_hash
        or candidate_policy.policy_hash != spec.candidate_policy_hash
    ):
        raise ValueError("V3.5 policies are not frozen in protocol")
    at = evaluated_at or datetime.now(timezone.utc)
    store = RunStore(
        output_root,
        run_id=run_id or f"controlled-dynamics-v35-{uuid4().hex[:10]}",
    )
    refs = [
        store.put_artifact("controlled_dynamics_spec_v35", spec),
        store.put_artifact("controlled_dynamics_random_policy_v35", baseline_policy),
        store.put_artifact("controlled_dynamics_unguarded_policy_v35", diagnostic_policy),
        store.put_artifact("controlled_dynamics_guarded_policy_v35", candidate_policy),
    ]
    store.emit("controlled_dynamics_v35_protocol_frozen_before_private_pack", {
        "spec_hash": spec.spec_hash,
        "prior_v341_adapter_report_hash": spec.prior_v341_adapter_report_hash,
        "factorial_design": "R_A_AG",
    })
    private_pack = generate_private_controlled_dynamics_worldpack_v31(
        spec, generated_at=at
    )
    baseline = execute_controlled_dynamics_policy_v35(
        spec, private_pack, baseline_policy, executed_at=at
    )
    diagnostic = execute_controlled_dynamics_policy_v35(
        spec, private_pack, diagnostic_policy, executed_at=at
    )
    candidate = execute_controlled_dynamics_policy_v35(
        spec, private_pack, candidate_policy, executed_at=at
    )
    evolution = evaluate_controlled_dynamics_worldpack_v35(
        spec, private_pack, baseline, diagnostic, candidate, evaluated_at=at
    )
    refs.extend([
        store.put_artifact("private_controlled_dynamics_worldpack_v35", private_pack),
        store.put_artifact("controlled_dynamics_random_bundle_v35", baseline),
        store.put_artifact("controlled_dynamics_unguarded_bundle_v35", diagnostic),
        store.put_artifact("controlled_dynamics_guarded_bundle_v35", candidate),
        store.put_artifact("controlled_dynamics_evolution_report_v35", evolution),
    ])
    manifest = ControlledDynamicsManifestV35.seal(
        run_id=store.run_id,
        artifact_refs=refs,
        terminal_status=evolution.status,
        created_at=at,
    )
    manifest_ref = store.put_artifact("controlled_dynamics_manifest_v35", manifest)
    store.emit("controlled_dynamics_v35_worldpack_adjudicated", {
        "manifest_ref": manifest_ref.model_dump(mode="json")
    })
    if not verify_controlled_dynamics_run_v35(store.run_directory):
        raise RuntimeError("V3.5 run failed independent verification")
    return ControlledDynamicsOutcomeV35(
        store,
        spec,
        private_pack,
        baseline_policy,
        diagnostic_policy,
        candidate_policy,
        baseline,
        diagnostic,
        candidate,
        evolution,
        manifest,
    )


def verify_controlled_dynamics_run_v35(run_directory: str | Path) -> bool:
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
            if item.kind == "controlled_dynamics_manifest_v35"
        ]
        if len(manifest_refs) != 1:
            return False
        manifest = ControlledDynamicsManifestV35.model_validate(
            store.load_artifact(manifest_refs[0])
        )
        manifest.assert_sealed()

        def load_one(kind: str, model):
            references = [
                item for item in manifest.artifact_refs if item.kind == kind
            ]
            if len(references) != 1:
                raise RuntimeError(f"V3.5 manifest needs exactly one {kind}")
            return model.model_validate(store.load_artifact(references[0]))

        spec = load_one(
            "controlled_dynamics_spec_v35", ControlledDynamicsWorldPackSpecV35
        )
        baseline_policy = load_one(
            "controlled_dynamics_random_policy_v35", GuardedAcquisitionPolicyV35
        )
        diagnostic_policy = load_one(
            "controlled_dynamics_unguarded_policy_v35", GuardedAcquisitionPolicyV35
        )
        candidate_policy = load_one(
            "controlled_dynamics_guarded_policy_v35", GuardedAcquisitionPolicyV35
        )
        private_pack = load_one(
            "private_controlled_dynamics_worldpack_v35",
            PrivateControlledDynamicsWorldPackV31,
        )
        baseline = load_one(
            "controlled_dynamics_random_bundle_v35",
            ControlledDynamicsSelectionBundleV34,
        )
        diagnostic = load_one(
            "controlled_dynamics_unguarded_bundle_v35",
            ControlledDynamicsSelectionBundleV34,
        )
        candidate = load_one(
            "controlled_dynamics_guarded_bundle_v35",
            ControlledDynamicsSelectionBundleV34,
        )
        evolution = load_one(
            "controlled_dynamics_evolution_report_v35",
            GuardedAcquisitionEvolutionReportV35,
        )
        for artifact in (
            spec, baseline_policy, diagnostic_policy, candidate_policy,
            private_pack, baseline, diagnostic, candidate, evolution, manifest,
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
        replay_baseline = execute_controlled_dynamics_policy_v35(
            spec, private_pack, baseline_policy, executed_at=executed_at
        )
        replay_diagnostic = execute_controlled_dynamics_policy_v35(
            spec, private_pack, diagnostic_policy, executed_at=executed_at
        )
        replay_candidate = execute_controlled_dynamics_policy_v35(
            spec, private_pack, candidate_policy, executed_at=executed_at
        )
        if (
            replay_baseline.bundle_hash != baseline.bundle_hash
            or replay_diagnostic.bundle_hash != diagnostic.bundle_hash
            or replay_candidate.bundle_hash != candidate.bundle_hash
        ):
            return False
        recomputed = evaluate_controlled_dynamics_worldpack_v35(
            spec,
            private_pack,
            baseline,
            diagnostic,
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
        freezes = [
            event for event in events
            if event["event_type"]
            == "controlled_dynamics_v35_protocol_frozen_before_private_pack"
        ]
        private_events = [
            event for event in events
            if event["event_type"] == "artifact_committed"
            and event["payload"]["kind"] == "private_controlled_dynamics_worldpack_v35"
        ]
        return (
            len(freezes) == 1
            and len(private_events) == 1
            and freezes[0]["sequence"] < private_events[0]["sequence"]
            and store.verify_event_chain()
        )
    except (
        OSError, RuntimeError, ValueError, KeyError, TypeError,
        json.JSONDecodeError, FloatingPointError,
    ):
        return False
