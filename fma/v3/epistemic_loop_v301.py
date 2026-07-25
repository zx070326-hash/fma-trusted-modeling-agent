from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.stats import beta

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore
from fma.v2.schemas import Identifier, Sha256

from .epistemic_loop import (
    CONFIRMATION_SEEDS_V30,
    EXPLORATORY_SEEDS_V30,
    MECHANISMS_V30,
    ActionKindV30,
    ArmV30,
    EpisodeProblemContractV30,
    EpistemicActionProposalV30,
    EpistemicEvidenceReceiptV30,
    EpistemicPermissionDecisionV30,
    EpistemicStateV30,
    EpistemicToolResultV30,
    LossProfileIdV30,
    LossProfileV30,
    MechanismV30,
    MissionConstitutionV30,
    ProblemReformulationCaseResultV30,
    ProblemReformulationMechanismResultV30,
    ProblemReformulationPublicCaseV30,
    ShadowCapacityDecisionV30,
    _clopper_pearson_upper_v30,
    _decision_spread,
    _demand_draws,
    _mean_loss,
    _profile,
    _solve_capacity,
    default_loss_profiles_v30,
    decide_epistemic_permission_v30,
)


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


def _assert_timezone(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")


class SequentialEpistemicPolicyV301(StrictModel):
    schema_version: Literal["3.0.1"] = "3.0.1"
    policy_id: Identifier
    arm: ArmV30
    selection_rule: Literal[
        "collect_two_batches_under_frozen_contract",
        "clarify_if_decision_critical_then_collect",
    ]
    decision_spread_threshold: Annotated[int, Field(ge=0, le=20)]
    may_reformulate_problem: bool
    max_epistemic_actions: Literal[2] = 2
    prior_failure_report_hash: Sha256
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "SequentialEpistemicPolicyV301":
        expected = {
            "fixed_contract_more_data": (
                "collect_two_batches_under_frozen_contract",
                False,
            ),
            "reformulation_value_of_information": (
                "clarify_if_decision_critical_then_collect",
                True,
            ),
        }[self.arm]
        if (self.selection_rule, self.may_reformulate_problem) != expected:
            raise ValueError("V3.0.1 policy fields disagree with arm semantics")
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("policy_hash does not match V3.0.1 policy")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("V3.0.1 sequential policy is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "SequentialEpistemicPolicyV301":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"policy_hash"}),
            policy_hash=draft.content_hash(),
        )


class ProblemReformulationWorldPackSpecV301(StrictModel):
    schema_version: Literal["3.0.1"] = "3.0.1"
    experiment_id: Identifier
    phase: Literal["exploratory", "confirmation"]
    mission_constitution: MissionConstitutionV30
    mechanisms: list[MechanismV30] = Field(min_length=4, max_length=4)
    seeds: list[int] = Field(min_length=8)
    baseline_policy_hash: Sha256
    candidate_policy_hash: Sha256
    prior_failure_report_hash: Sha256
    evolved_component: Literal["epistemic_action_horizon_1_to_2"] = (
        "epistemic_action_horizon_1_to_2"
    )
    pilot_size: Annotated[int, Field(ge=8, le=64)] = 16
    batch_size: Annotated[int, Field(ge=4, le=64)] = 16
    evaluation_size: Annotated[int, Field(ge=256, le=4096)] = 1024
    known_semantics_modulus: Annotated[int, Field(ge=2, le=10)] = 4
    bootstrap_replicates: Annotated[int, Field(ge=500, le=10000)] = 2000
    bootstrap_seed: Annotated[int, Field(ge=0)] = 9301
    confidence: Annotated[float, Field(gt=0.5, lt=1, allow_inf_nan=False)] = 0.95
    min_macro_regret_improvement: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ] = 0.01
    per_mechanism_noninferiority_margin: Annotated[
        float, Field(ge=0, le=0.2, allow_inf_nan=False)
    ] = 0.005
    material_negative_transfer_threshold: Annotated[
        float, Field(gt=0, le=0.5, allow_inf_nan=False)
    ] = 0.05
    max_negative_transfer_rate: Annotated[
        float, Field(gt=0, le=0.5, allow_inf_nan=False)
    ] = 0.05
    literature_evidence_refs: list[Annotated[str, Field(min_length=3)]] = Field(
        min_length=2
    )
    frozen_at: datetime
    spec_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "ProblemReformulationWorldPackSpecV301":
        _assert_timezone(self.frozen_at, "frozen_at")
        self.mission_constitution.assert_sealed()
        if self.mechanisms != list(MECHANISMS_V30):
            raise ValueError("V3.0.1 mechanisms must be frozen and ordered")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("V3.0.1 seeds must be unique")
        if self.phase == "exploratory" and tuple(self.seeds) != EXPLORATORY_SEEDS_V30:
            raise ValueError("V3.0.1 exploratory seeds are not the frozen split")
        if self.phase == "confirmation" and tuple(self.seeds) != CONFIRMATION_SEEDS_V30:
            raise ValueError("V3.0.1 confirmation seeds are not the frozen split")
        if self.mission_constitution.epistemic_action_budget != 2:
            raise ValueError("V3.0.1 freezes a two-cost epistemic horizon")
        if set(self.mission_constitution.action_costs.values()) != {1}:
            raise ValueError("V3.0.1 actions must each cost one unit")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match V3.0.1 spec")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "spec_hash")

    def assert_sealed(self) -> None:
        if not self.spec_hash or self.spec_hash != self.content_hash():
            raise ValueError("V3.0.1 WorldPack spec is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ProblemReformulationWorldPackSpecV301":
        data.setdefault("frozen_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"spec_hash"}),
            spec_hash=draft.content_hash(),
        )


class PrivateProblemReformulationCaseV301(StrictModel):
    schema_version: Literal["3.0.1-private"] = "3.0.1-private"
    public_case: ProblemReformulationPublicCaseV30
    true_loss_profile: LossProfileV30
    demand_batches: list[list[Annotated[int, Field(ge=0, le=100)]]] = Field(
        min_length=2, max_length=2
    )
    evaluation_demand: list[Annotated[int, Field(ge=0, le=100)]] = Field(
        min_length=256
    )
    case_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_case(self) -> "PrivateProblemReformulationCaseV301":
        self.public_case.assert_sealed()
        if self.true_loss_profile.source_kind != "authoritative":
            raise ValueError("V3.0.1 private truth must be authoritative")
        if any(
            len(batch) != self.public_case.additional_batch_size
            for batch in self.demand_batches
        ):
            raise ValueError("V3.0.1 demand batches disagree with public size")
        if self.public_case.initial_contract.semantics_status == "authoritative":
            if (
                self.public_case.initial_contract.loss_profile.profile_id
                != self.true_loss_profile.profile_id
            ):
                raise ValueError("V3.0.1 known semantics disagree with private truth")
        if self.case_hash and self.case_hash != self.content_hash():
            raise ValueError("case_hash does not match V3.0.1 private case")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "case_hash")

    def assert_sealed(self) -> None:
        if not self.case_hash or self.case_hash != self.content_hash():
            raise ValueError("V3.0.1 private case is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PrivateProblemReformulationCaseV301":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"case_hash"}),
            case_hash=draft.content_hash(),
        )


class PrivateProblemReformulationWorldPackV301(StrictModel):
    schema_version: Literal["3.0.1-private"] = "3.0.1-private"
    spec_hash: Sha256
    cases: list[PrivateProblemReformulationCaseV301] = Field(min_length=32)
    generated_at: datetime
    pack_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_pack(self) -> "PrivateProblemReformulationWorldPackV301":
        _assert_timezone(self.generated_at, "generated_at")
        ids = [case.public_case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("V3.0.1 case ids must be unique")
        for case in self.cases:
            case.assert_sealed()
        if self.pack_hash and self.pack_hash != self.content_hash():
            raise ValueError("pack_hash does not match V3.0.1 private pack")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "pack_hash")

    def assert_sealed(self) -> None:
        if not self.pack_hash or self.pack_hash != self.content_hash():
            raise ValueError("V3.0.1 private pack is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PrivateProblemReformulationWorldPackV301":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"pack_hash"}),
            pack_hash=draft.content_hash(),
        )


def default_sequential_epistemic_policies_v301(
    *, prior_failure_report_hash: str
) -> tuple[SequentialEpistemicPolicyV301, SequentialEpistemicPolicyV301]:
    baseline = SequentialEpistemicPolicyV301.seal(
        policy_id="fixed_contract_two_batches_v301",
        arm="fixed_contract_more_data",
        selection_rule="collect_two_batches_under_frozen_contract",
        decision_spread_threshold=0,
        may_reformulate_problem=False,
        prior_failure_report_hash=prior_failure_report_hash,
    )
    candidate = SequentialEpistemicPolicyV301.seal(
        policy_id="clarify_then_collect_v301",
        arm="reformulation_value_of_information",
        selection_rule="clarify_if_decision_critical_then_collect",
        decision_spread_threshold=2,
        may_reformulate_problem=True,
        prior_failure_report_hash=prior_failure_report_hash,
    )
    return baseline, candidate


def _mission_v301(at: datetime) -> MissionConstitutionV30:
    return MissionConstitutionV30.seal(
        constitution_id="synthetic_capacity_epistemic_mission_v301",
        value_owner_ref="fixture:synthetic_value_owner",
        capacity_lower_bound=0,
        capacity_upper_bound=30,
        allowed_epistemic_actions=[
            "collect_demand_batch",
            "clarify_loss_semantics",
        ],
        action_costs={
            "collect_demand_batch": 1,
            "clarify_loss_semantics": 1,
        },
        epistemic_action_budget=2,
        frozen_at=at,
    )


def default_problem_reformulation_exploratory_spec_v301(
    *,
    prior_failure_report_hash: str,
    frozen_at: datetime | None = None,
) -> tuple[
    ProblemReformulationWorldPackSpecV301,
    SequentialEpistemicPolicyV301,
    SequentialEpistemicPolicyV301,
]:
    at = frozen_at or datetime.now(timezone.utc)
    baseline, candidate = default_sequential_epistemic_policies_v301(
        prior_failure_report_hash=prior_failure_report_hash
    )
    spec = ProblemReformulationWorldPackSpecV301.seal(
        experiment_id="problem_reformulation_exploratory_v301",
        phase="exploratory",
        mission_constitution=_mission_v301(at),
        mechanisms=list(MECHANISMS_V30),
        seeds=list(EXPLORATORY_SEEDS_V30),
        baseline_policy_hash=baseline.policy_hash,
        candidate_policy_hash=candidate.policy_hash,
        prior_failure_report_hash=prior_failure_report_hash,
        literature_evidence_refs=[
            "arxiv:2502.18864",
            "jmlr:23-32-20-807",
            "pmlr:144-ahmadi21a",
        ],
        frozen_at=at,
    )
    return spec, baseline, candidate


def default_problem_reformulation_confirmation_spec_v301(
    *,
    prior_failure_report_hash: str,
    frozen_at: datetime | None = None,
) -> tuple[
    ProblemReformulationWorldPackSpecV301,
    SequentialEpistemicPolicyV301,
    SequentialEpistemicPolicyV301,
]:
    at = frozen_at or datetime.now(timezone.utc)
    baseline, candidate = default_sequential_epistemic_policies_v301(
        prior_failure_report_hash=prior_failure_report_hash
    )
    spec = ProblemReformulationWorldPackSpecV301.seal(
        experiment_id="problem_reformulation_confirmation_v301",
        phase="confirmation",
        mission_constitution=_mission_v301(at),
        mechanisms=list(MECHANISMS_V30),
        seeds=list(CONFIRMATION_SEEDS_V30),
        baseline_policy_hash=baseline.policy_hash,
        candidate_policy_hash=candidate.policy_hash,
        prior_failure_report_hash=prior_failure_report_hash,
        literature_evidence_refs=[
            "arxiv:2502.18864",
            "jmlr:23-32-20-807",
            "pmlr:144-ahmadi21a",
        ],
        frozen_at=at,
    )
    return spec, baseline, candidate


def generate_private_problem_reformulation_worldpack_v301(
    spec: ProblemReformulationWorldPackSpecV301,
    *,
    generated_at: datetime | None = None,
) -> PrivateProblemReformulationWorldPackV301:
    spec.assert_sealed()
    mission = spec.mission_constitution
    profile_ids: tuple[LossProfileIdV30, ...] = (
        "balanced_absolute",
        "shortage_critical",
        "overage_critical",
    )
    cases: list[PrivateProblemReformulationCaseV301] = []
    for mechanism_index, mechanism in enumerate(spec.mechanisms):
        for seed in spec.seeds:
            case_id = f"epr301_{mechanism}_{seed}"
            total = spec.pilot_size + 2 * spec.batch_size + spec.evaluation_size
            demand = _demand_draws(mechanism, seed=seed, count=total)
            pilot = demand[: spec.pilot_size]
            first_end = spec.pilot_size + spec.batch_size
            second_end = first_end + spec.batch_size
            batches = [demand[spec.pilot_size:first_end], demand[first_end:second_end]]
            evaluation = demand[second_end:]
            true_id = profile_ids[(seed + mechanism_index) % len(profile_ids)]
            truth = _profile(
                true_id,
                source_kind="authoritative",
                source_ref=f"private_value_owner:{case_id}",
            )
            semantics_known = (seed + mechanism_index) % spec.known_semantics_modulus == 0
            if semantics_known:
                initial_profile = truth.model_copy(
                    update={"source_ref": f"public_value_owner:{case_id}"}
                )
                status = "authoritative"
                unresolved: list[Literal["loss_semantics"]] = []
            else:
                initial_profile = _profile(
                    "balanced_absolute",
                    source_kind="assumption",
                    source_ref="default_assumption:balanced_absolute",
                )
                status = "underspecified"
                unresolved = ["loss_semantics"]
            contract = EpisodeProblemContractV30.seal(
                contract_id=f"contract_{case_id}_v1",
                case_id=case_id,
                mission_constitution_hash=mission.constitution_hash,
                version=1,
                loss_profile=initial_profile,
                semantics_status=status,
                unresolved_fields=unresolved,
                frozen_at=spec.frozen_at,
            )
            public = ProblemReformulationPublicCaseV30.seal(
                case_id=case_id,
                mechanism=mechanism,
                pilot_demand=pilot,
                candidate_loss_profiles=default_loss_profiles_v30(),
                initial_contract=contract,
                additional_batch_size=spec.batch_size,
            )
            cases.append(
                PrivateProblemReformulationCaseV301.seal(
                    public_case=public,
                    true_loss_profile=truth,
                    demand_batches=batches,
                    evaluation_demand=evaluation,
                )
            )
    return PrivateProblemReformulationWorldPackV301.seal(
        spec_hash=spec.spec_hash,
        cases=cases,
        generated_at=generated_at or datetime.now(timezone.utc),
    )


class SequentialEpistemicStepReceiptV301(StrictModel):
    schema_version: Literal["3.0.1"] = "3.0.1"
    step_index: Annotated[int, Field(ge=1, le=2)]
    state_before_hash: Sha256
    proposal: EpistemicActionProposalV30
    permission: EpistemicPermissionDecisionV30
    tool_result: EpistemicToolResultV30
    state_after: EpistemicStateV30
    step_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_step(self) -> "SequentialEpistemicStepReceiptV301":
        self.proposal.assert_sealed()
        self.permission.assert_sealed()
        self.tool_result.assert_sealed()
        self.state_after.assert_sealed()
        if self.permission.proposal_hash != self.proposal.proposal_hash:
            raise ValueError("V3.0.1 permission belongs to another proposal")
        if self.tool_result.proposal_hash != self.proposal.proposal_hash:
            raise ValueError("V3.0.1 tool result belongs to another proposal")
        if (self.permission.decision == "allow") != (
            self.tool_result.status == "success"
        ):
            raise ValueError("V3.0.1 permission and result disagree")
        if self.step_hash and self.step_hash != self.content_hash():
            raise ValueError("step_hash does not match V3.0.1 step")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "step_hash")

    def assert_sealed(self) -> None:
        if not self.step_hash or self.step_hash != self.content_hash():
            raise ValueError("V3.0.1 step is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "SequentialEpistemicStepReceiptV301":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"step_hash"}),
            step_hash=draft.content_hash(),
        )


class SequentialEpistemicCaseReceiptV301(StrictModel):
    schema_version: Literal["3.0.1"] = "3.0.1"
    receipt_id: Identifier
    case_id: Identifier
    public_case_hash: Sha256
    policy_hash: Sha256
    arm: ArmV30
    initial_state_hash: Sha256
    steps: list[SequentialEpistemicStepReceiptV301] = Field(min_length=1, max_length=2)
    final_state: EpistemicStateV30
    shadow_decision: ShadowCapacityDecisionV30 | None
    action_cost_consumed: Annotated[int, Field(ge=0, le=2)]
    reformulation_count: Annotated[int, Field(ge=0, le=1)]
    permission_error_count: Annotated[int, Field(ge=0, le=2)]
    executed_at: datetime
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "SequentialEpistemicCaseReceiptV301":
        _assert_timezone(self.executed_at, "executed_at")
        self.final_state.assert_sealed()
        if [step.step_index for step in self.steps] != list(
            range(1, len(self.steps) + 1)
        ):
            raise ValueError("V3.0.1 steps must be contiguous")
        previous = self.initial_state_hash
        for step in self.steps:
            step.assert_sealed()
            if step.state_before_hash != previous:
                raise ValueError("V3.0.1 state transition chain is broken")
            previous = step.state_after.state_hash
        if previous != self.final_state.state_hash:
            raise ValueError("V3.0.1 final state is not the last step state")
        expected_cost = sum(
            step.proposal.proposed_cost
            for step in self.steps
            if step.permission.decision == "allow"
        )
        if self.action_cost_consumed != expected_cost:
            raise ValueError("V3.0.1 action cost is inconsistent")
        expected_errors = sum(step.tool_result.status != "success" for step in self.steps)
        if self.permission_error_count != expected_errors:
            raise ValueError("V3.0.1 permission error count is inconsistent")
        if self.reformulation_count != len(self.final_state.contract_history) - 1:
            raise ValueError("V3.0.1 reformulation count is inconsistent")
        if self.final_state.terminal_status == "shadow_decision_ready":
            if self.shadow_decision is None:
                raise ValueError("V3.0.1 decision-ready receipt needs a decision")
            self.shadow_decision.assert_sealed()
            if (
                self.shadow_decision.episode_contract_hash
                != self.final_state.current_contract_hash
            ):
                raise ValueError("V3.0.1 decision belongs to another contract")
        elif self.shadow_decision is not None:
            raise ValueError("V3.0.1 needs-evidence receipt cannot have a decision")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match V3.0.1 case receipt")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("V3.0.1 case receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "SequentialEpistemicCaseReceiptV301":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


class SequentialEpistemicBundleV301(StrictModel):
    schema_version: Literal["3.0.1"] = "3.0.1"
    spec_hash: Sha256
    private_pack_hash: Sha256
    policy_hash: Sha256
    arm: ArmV30
    case_receipts: list[SequentialEpistemicCaseReceiptV301] = Field(min_length=32)
    total_action_cost: Annotated[int, Field(ge=0)]
    total_reformulations: Annotated[int, Field(ge=0)]
    permission_error_count: Annotated[int, Field(ge=0)]
    bundle_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bundle(self) -> "SequentialEpistemicBundleV301":
        ids = []
        for receipt in self.case_receipts:
            receipt.assert_sealed()
            ids.append(receipt.case_id)
            if receipt.policy_hash != self.policy_hash or receipt.arm != self.arm:
                raise ValueError("V3.0.1 receipt belongs to another policy")
        if len(ids) != len(set(ids)):
            raise ValueError("V3.0.1 case ids must be unique")
        if self.total_action_cost != sum(
            item.action_cost_consumed for item in self.case_receipts
        ):
            raise ValueError("V3.0.1 total cost is inconsistent")
        if self.total_reformulations != sum(
            item.reformulation_count for item in self.case_receipts
        ):
            raise ValueError("V3.0.1 reformulations are inconsistent")
        if self.permission_error_count != sum(
            item.permission_error_count for item in self.case_receipts
        ):
            raise ValueError("V3.0.1 errors are inconsistent")
        if self.bundle_hash and self.bundle_hash != self.content_hash():
            raise ValueError("bundle_hash does not match V3.0.1 bundle")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "bundle_hash")

    def assert_sealed(self) -> None:
        if not self.bundle_hash or self.bundle_hash != self.content_hash():
            raise ValueError("V3.0.1 bundle is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "SequentialEpistemicBundleV301":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"bundle_hash"}),
            bundle_hash=draft.content_hash(),
        )


def _proposal_v301(
    public: ProblemReformulationPublicCaseV30,
    state: EpistemicStateV30,
    policy: SequentialEpistemicPolicyV301,
    mission: MissionConstitutionV30,
    *,
    step_index: int,
) -> EpistemicActionProposalV30:
    spread = _decision_spread(public, mission)
    if policy.arm == "fixed_contract_more_data":
        action: ActionKindV30 = "collect_demand_batch"
        rationale = "fixed_contract_collect_more"
    elif state.current_contract.semantics_status == "authoritative":
        action = "collect_demand_batch"
        rationale = "semantics_sufficient_collect_more"
    elif spread >= policy.decision_spread_threshold:
        action = "clarify_loss_semantics"
        rationale = "decision_critical_semantics_unresolved"
    else:
        action = "collect_demand_batch"
        rationale = "low_decision_spread_collect_more"
    return EpistemicActionProposalV30.seal(
        proposal_id=f"proposal_{policy.arm}_{public.case_id}_s{step_index}",
        case_id=public.case_id,
        policy_hash=policy.policy_hash,
        action_kind=action,
        proposed_cost=mission.action_costs[action],
        decision_spread=spread,
        rationale_code=rationale,
    )


def _tool_result_v301(
    private_case: PrivateProblemReformulationCaseV301,
    state: EpistemicStateV30,
    proposal: EpistemicActionProposalV30,
    permission: EpistemicPermissionDecisionV30,
    *,
    step_index: int,
    at: datetime,
) -> EpistemicToolResultV30:
    if permission.decision == "deny":
        return EpistemicToolResultV30.seal(
            proposal_hash=proposal.proposal_hash,
            status="denied",
            error_code="permission_denied",
            next_valid_actions=["stop_needs_evidence"],
        )
    if proposal.action_kind == "collect_demand_batch":
        prior_batches = sum(
            evidence.action_kind == "collect_demand_batch"
            for evidence in state.evidence_receipts
        )
        if prior_batches >= len(private_case.demand_batches):
            return EpistemicToolResultV30.seal(
                proposal_hash=proposal.proposal_hash,
                status="error",
                error_code="tool_failure",
                next_valid_actions=["stop_needs_evidence"],
            )
        evidence = EpistemicEvidenceReceiptV30.seal(
            evidence_id=f"evidence_demand_{proposal.case_id}_s{step_index}",
            case_id=proposal.case_id,
            action_kind="collect_demand_batch",
            payload_kind="demand_batch",
            demand_values=private_case.demand_batches[prior_batches],
            source_ref=f"synthetic_reality:demand:{proposal.case_id}:batch{prior_batches + 1}",
            observed_at=at,
        )
    else:
        evidence = EpistemicEvidenceReceiptV30.seal(
            evidence_id=f"evidence_semantics_{proposal.case_id}_s{step_index}",
            case_id=proposal.case_id,
            action_kind="clarify_loss_semantics",
            payload_kind="loss_semantics",
            loss_profile=private_case.true_loss_profile,
            source_ref=f"synthetic_value_owner:{proposal.case_id}",
            observed_at=at,
        )
    return EpistemicToolResultV30.seal(
        proposal_hash=proposal.proposal_hash,
        status="success",
        evidence=evidence,
    )


def _advance_state_v301(
    state: EpistemicStateV30,
    proposal: EpistemicActionProposalV30,
    permission: EpistemicPermissionDecisionV30,
    result: EpistemicToolResultV30,
    policy: SequentialEpistemicPolicyV301,
    *,
    at: datetime,
) -> EpistemicStateV30:
    if result.status != "success":
        return EpistemicStateV30.seal(
            case_id=state.case_id,
            mission_constitution_hash=state.mission_constitution_hash,
            arm=state.arm,
            policy_hash=state.policy_hash,
            contract_history=state.contract_history,
            current_contract_hash=state.current_contract_hash,
            evidence_receipts=state.evidence_receipts,
            remaining_action_budget=permission.budget_after,
            terminal_status="needs_evidence",
        )
    assert result.evidence is not None
    evidence = result.evidence
    history = list(state.contract_history)
    if proposal.action_kind == "clarify_loss_semantics":
        if not policy.may_reformulate_problem or evidence.loss_profile is None:
            raise RuntimeError("V3.0.1 clarification cannot be applied")
        parent = history[-1]
        history.append(
            EpisodeProblemContractV30.seal(
                contract_id=f"contract_{state.case_id}_v{parent.version + 1}",
                case_id=state.case_id,
                mission_constitution_hash=state.mission_constitution_hash,
                version=parent.version + 1,
                parent_contract_hash=parent.contract_hash,
                triggering_evidence_hash=evidence.evidence_hash,
                revision_reason="authoritative_loss_semantics_received",
                loss_profile=evidence.loss_profile,
                semantics_status="authoritative",
                unresolved_fields=[],
                frozen_at=at,
            )
        )
    terminal = (
        "shadow_decision_ready" if permission.budget_after == 0 else "running"
    )
    return EpistemicStateV30.seal(
        case_id=state.case_id,
        mission_constitution_hash=state.mission_constitution_hash,
        arm=state.arm,
        policy_hash=state.policy_hash,
        contract_history=history,
        current_contract_hash=history[-1].contract_hash,
        evidence_receipts=[*state.evidence_receipts, evidence],
        remaining_action_budget=permission.budget_after,
        terminal_status=terminal,
    )


def _shadow_decision_v301(
    public: ProblemReformulationPublicCaseV30,
    state: EpistemicStateV30,
    mission: MissionConstitutionV30,
    *,
    at: datetime,
) -> ShadowCapacityDecisionV30 | None:
    if state.terminal_status != "shadow_decision_ready":
        return None
    demand = list(public.pilot_demand)
    for evidence in state.evidence_receipts:
        demand.extend(evidence.demand_values)
    capacity, loss = _solve_capacity(demand, state.current_contract.loss_profile, mission)
    return ShadowCapacityDecisionV30.seal(
        decision_id=f"shadow_capacity_{state.arm}_{state.case_id}_v301",
        case_id=state.case_id,
        episode_contract_hash=state.current_contract_hash,
        selected_capacity=capacity,
        empirical_loss=loss,
        demand_observation_count=len(demand),
        semantics_status=state.current_contract.semantics_status,
        decided_at=at,
    )


def execute_sequential_epistemic_policy_v301(
    spec: ProblemReformulationWorldPackSpecV301,
    private_pack: PrivateProblemReformulationWorldPackV301,
    policy: SequentialEpistemicPolicyV301,
    *,
    executed_at: datetime,
) -> SequentialEpistemicBundleV301:
    spec.assert_sealed()
    private_pack.assert_sealed()
    policy.assert_sealed()
    expected_hash = (
        spec.baseline_policy_hash
        if policy.arm == "fixed_contract_more_data"
        else spec.candidate_policy_hash
    )
    if policy.policy_hash != expected_hash or private_pack.spec_hash != spec.spec_hash:
        raise ValueError("V3.0.1 execution inputs are cross-bound")
    mission = spec.mission_constitution
    receipts: list[SequentialEpistemicCaseReceiptV301] = []
    for private_case in private_pack.cases:
        public = private_case.public_case
        state = EpistemicStateV30.seal(
            case_id=public.case_id,
            mission_constitution_hash=mission.constitution_hash,
            arm=policy.arm,
            policy_hash=policy.policy_hash,
            contract_history=[public.initial_contract],
            current_contract_hash=public.initial_contract.contract_hash,
            evidence_receipts=[],
            remaining_action_budget=mission.epistemic_action_budget,
            terminal_status="running",
        )
        initial_hash = state.state_hash
        steps: list[SequentialEpistemicStepReceiptV301] = []
        for step_index in range(1, policy.max_epistemic_actions + 1):
            if state.terminal_status != "running":
                break
            before_hash = state.state_hash
            proposal = _proposal_v301(
                public, state, policy, mission, step_index=step_index
            )
            permission = decide_epistemic_permission_v30(
                proposal, state, mission, decided_at=executed_at
            )
            result = _tool_result_v301(
                private_case,
                state,
                proposal,
                permission,
                step_index=step_index,
                at=executed_at,
            )
            state = _advance_state_v301(
                state, proposal, permission, result, policy, at=executed_at
            )
            steps.append(
                SequentialEpistemicStepReceiptV301.seal(
                    step_index=step_index,
                    state_before_hash=before_hash,
                    proposal=proposal,
                    permission=permission,
                    tool_result=result,
                    state_after=state,
                )
            )
        decision = _shadow_decision_v301(public, state, mission, at=executed_at)
        receipts.append(
            SequentialEpistemicCaseReceiptV301.seal(
                receipt_id=f"receipt_{policy.arm}_{public.case_id}_v301",
                case_id=public.case_id,
                public_case_hash=public.public_hash,
                policy_hash=policy.policy_hash,
                arm=policy.arm,
                initial_state_hash=initial_hash,
                steps=steps,
                final_state=state,
                shadow_decision=decision,
                action_cost_consumed=sum(
                    step.proposal.proposed_cost
                    for step in steps
                    if step.permission.decision == "allow"
                ),
                reformulation_count=len(state.contract_history) - 1,
                permission_error_count=sum(
                    step.tool_result.status != "success" for step in steps
                ),
                executed_at=executed_at,
            )
        )
    return SequentialEpistemicBundleV301.seal(
        spec_hash=spec.spec_hash,
        private_pack_hash=private_pack.pack_hash,
        policy_hash=policy.policy_hash,
        arm=policy.arm,
        case_receipts=receipts,
        total_action_cost=sum(item.action_cost_consumed for item in receipts),
        total_reformulations=sum(item.reformulation_count for item in receipts),
        permission_error_count=sum(item.permission_error_count for item in receipts),
    )


class ProblemReformulationReportV301(StrictModel):
    schema_version: Literal["3.0.1"] = "3.0.1"
    spec_hash: Sha256
    private_pack_hash: Sha256
    baseline_bundle_hash: Sha256
    candidate_bundle_hash: Sha256
    prior_failure_report_hash: Sha256
    evolved_component: Literal["epistemic_action_horizon_1_to_2"]
    cases: list[ProblemReformulationCaseResultV30] = Field(min_length=32)
    mechanisms: list[ProblemReformulationMechanismResultV30] = Field(
        min_length=4, max_length=4
    )
    same_epistemic_action_cost: bool
    permission_error_count: Annotated[int, Field(ge=0)]
    underspecified_case_count: Annotated[int, Field(ge=1)]
    candidate_reformulation_count: Annotated[int, Field(ge=0)]
    spurious_reformulation_count: Annotated[int, Field(ge=0)]
    unresolved_candidate_semantics_count: Annotated[int, Field(ge=0)]
    material_negative_transfer_count: Annotated[int, Field(ge=0)]
    macro_regret_improvement: Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)]
    macro_regret_improvement_ci_lower: Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)]
    macro_regret_improvement_ci_upper: Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)]
    negative_transfer_rate_upper: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    gate_results: dict[str, bool]
    status: Literal[
        "exploratory_only",
        "candidate_rejected_epistemic_loop_v301",
        "promoted_for_synthetic_epistemic_loop_v301",
    ]
    reason_codes: list[str]
    evaluated_at: datetime
    report_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "ProblemReformulationReportV301":
        _assert_timezone(self.evaluated_at, "evaluated_at")
        if self.status == "promoted_for_synthetic_epistemic_loop_v301":
            if not self.gate_results or not all(self.gate_results.values()):
                raise ValueError("promoted V3.0.1 report must pass every gate")
            if self.reason_codes:
                raise ValueError("promoted V3.0.1 report cannot have reasons")
        elif self.status == "candidate_rejected_epistemic_loop_v301":
            if not self.reason_codes:
                raise ValueError("rejected V3.0.1 report needs reasons")
        if self.report_hash and self.report_hash != self.content_hash():
            raise ValueError("report_hash does not match V3.0.1 report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "report_hash")

    def assert_sealed(self) -> None:
        if not self.report_hash or self.report_hash != self.content_hash():
            raise ValueError("V3.0.1 report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ProblemReformulationReportV301":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"report_hash"}),
            report_hash=draft.content_hash(),
        )


def _normalized_regret_v301(decision_loss: float, oracle_loss: float) -> float:
    regret = max(decision_loss - oracle_loss, 0.0)
    return float(np.clip(regret / (decision_loss + oracle_loss + 1e-12), 0.0, 1.0))


def _bootstrap_v301(
    grouped: dict[str, list[float]], spec: ProblemReformulationWorldPackSpecV301
) -> np.ndarray:
    random = Random(spec.bootstrap_seed)
    draws = np.empty(spec.bootstrap_replicates, dtype=float)
    for draw in range(spec.bootstrap_replicates):
        means = []
        for mechanism in spec.mechanisms:
            values = grouped[mechanism]
            means.append(
                sum(values[random.randrange(len(values))] for _ in values)
                / len(values)
            )
        draws[draw] = sum(means) / len(means)
    return draws


def evaluate_problem_reformulation_v301(
    spec: ProblemReformulationWorldPackSpecV301,
    private_pack: PrivateProblemReformulationWorldPackV301,
    baseline: SequentialEpistemicBundleV301,
    candidate: SequentialEpistemicBundleV301,
    *,
    evaluated_at: datetime,
) -> ProblemReformulationReportV301:
    spec.assert_sealed()
    private_pack.assert_sealed()
    baseline.assert_sealed()
    candidate.assert_sealed()
    if baseline.arm != "fixed_contract_more_data" or candidate.arm != "reformulation_value_of_information":
        raise ValueError("V3.0.1 bundles are assigned to the wrong arms")
    if any(
        bundle.spec_hash != spec.spec_hash
        or bundle.private_pack_hash != private_pack.pack_hash
        for bundle in (baseline, candidate)
    ):
        raise ValueError("V3.0.1 evaluation inputs are cross-bound")
    baseline_by_id = {item.case_id: item for item in baseline.case_receipts}
    candidate_by_id = {item.case_id: item for item in candidate.case_receipts}
    private_by_id = {item.public_case.case_id: item for item in private_pack.cases}
    if set(baseline_by_id) != set(candidate_by_id) or set(baseline_by_id) != set(private_by_id):
        raise ValueError("V3.0.1 bundles do not cover the private pack")
    mission = spec.mission_constitution
    cases: list[ProblemReformulationCaseResultV30] = []
    grouped: dict[str, list[float]] = {mechanism: [] for mechanism in spec.mechanisms}
    unresolved = 0
    for private_case in private_pack.cases:
        public = private_case.public_case
        b = baseline_by_id[public.case_id]
        c = candidate_by_id[public.case_id]
        oracle_capacity, oracle_loss = _solve_capacity(
            private_case.evaluation_demand, private_case.true_loss_profile, mission
        )
        if b.shadow_decision is None:
            b_capacity = None
            b_regret = 1.0
        else:
            b_capacity = b.shadow_decision.selected_capacity
            b_loss = _mean_loss(
                b_capacity, private_case.evaluation_demand, private_case.true_loss_profile
            )
            b_regret = _normalized_regret_v301(b_loss, oracle_loss)
        if c.shadow_decision is None:
            c_capacity = None
            c_regret = 1.0
        else:
            c_capacity = c.shadow_decision.selected_capacity
            c_loss = _mean_loss(
                c_capacity, private_case.evaluation_demand, private_case.true_loss_profile
            )
            c_regret = _normalized_regret_v301(c_loss, oracle_loss)
        improvement = float(np.clip(b_regret - c_regret, -1.0, 1.0))
        grouped[public.mechanism].append(improvement)
        correct = (
            c.final_state.current_contract.loss_profile.profile_id
            == private_case.true_loss_profile.profile_id
        )
        if c.final_state.current_contract.semantics_status != "authoritative":
            unresolved += 1
        reformulated = c.reformulation_count == 1
        spurious = public.initial_contract.semantics_status == "authoritative" and reformulated
        cases.append(
            ProblemReformulationCaseResultV30(
                case_id=public.case_id,
                mechanism=public.mechanism,
                initial_semantics_status=public.initial_contract.semantics_status,
                true_loss_profile_id=private_case.true_loss_profile.profile_id,
                baseline_action=b.steps[0].proposal.action_kind,
                candidate_action=c.steps[0].proposal.action_kind,
                baseline_selected_capacity=b_capacity,
                candidate_selected_capacity=c_capacity,
                oracle_capacity=oracle_capacity,
                baseline_normalized_regret=b_regret,
                candidate_normalized_regret=c_regret,
                regret_improvement=improvement,
                candidate_semantics_correct=correct,
                candidate_reformulated=reformulated,
                spurious_reformulation=spurious,
                material_negative_transfer=(
                    c_regret - b_regret > spec.material_negative_transfer_threshold
                ),
            )
        )
    mechanism_results = []
    for mechanism in spec.mechanisms:
        selected = [case for case in cases if case.mechanism == mechanism]
        mechanism_results.append(
            ProblemReformulationMechanismResultV30(
                mechanism=mechanism,
                case_count=len(selected),
                mean_regret_improvement=float(np.mean([x.regret_improvement for x in selected])),
                mean_baseline_regret=float(np.mean([x.baseline_normalized_regret for x in selected])),
                mean_candidate_regret=float(np.mean([x.candidate_normalized_regret for x in selected])),
            )
        )
    macro = float(np.mean([item.mean_regret_improvement for item in mechanism_results]))
    draws = _bootstrap_v301(grouped, spec)
    alpha = 1.0 - spec.confidence
    ci_lower, ci_upper = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    negative_count = sum(item.material_negative_transfer for item in cases)
    negative_upper = _clopper_pearson_upper_v30(
        negative_count, len(cases), spec.confidence
    )
    underspecified = sum(item.initial_semantics_status == "underspecified" for item in cases)
    reformulations = sum(item.candidate_reformulated for item in cases)
    spurious_count = sum(item.spurious_reformulation for item in cases)
    errors = baseline.permission_error_count + candidate.permission_error_count
    gates = {
        "same_epistemic_action_cost": baseline.total_action_cost
        == candidate.total_action_cost
        == 2 * len(cases),
        "zero_permission_errors": errors == 0,
        "all_underspecified_semantics_resolved": unresolved == 0,
        "no_spurious_reformulation": spurious_count == 0,
        "reformulation_targets_only_missing_semantics": reformulations
        == underspecified,
        "macro_regret_ci_gate": float(ci_lower)
        >= spec.min_macro_regret_improvement,
        "per_mechanism_noninferiority": all(
            item.mean_regret_improvement >= -spec.per_mechanism_noninferiority_margin
            for item in mechanism_results
        ),
        "negative_transfer_rate_gate": negative_upper
        <= spec.max_negative_transfer_rate,
        "real_world_action_remains_forbidden": (
            not mission.real_world_action_authorized
            and all(
                receipt.shadow_decision is None
                or not receipt.shadow_decision.real_world_action_authorized
                for bundle in (baseline, candidate)
                for receipt in bundle.case_receipts
            )
        ),
    }
    reasons = [name for name, passed in gates.items() if not passed]
    if spec.phase == "exploratory":
        status = "exploratory_only"
    elif reasons:
        status = "candidate_rejected_epistemic_loop_v301"
    else:
        status = "promoted_for_synthetic_epistemic_loop_v301"
    return ProblemReformulationReportV301.seal(
        spec_hash=spec.spec_hash,
        private_pack_hash=private_pack.pack_hash,
        baseline_bundle_hash=baseline.bundle_hash,
        candidate_bundle_hash=candidate.bundle_hash,
        prior_failure_report_hash=spec.prior_failure_report_hash,
        evolved_component=spec.evolved_component,
        cases=cases,
        mechanisms=mechanism_results,
        same_epistemic_action_cost=gates["same_epistemic_action_cost"],
        permission_error_count=errors,
        underspecified_case_count=underspecified,
        candidate_reformulation_count=reformulations,
        spurious_reformulation_count=spurious_count,
        unresolved_candidate_semantics_count=unresolved,
        material_negative_transfer_count=negative_count,
        macro_regret_improvement=macro,
        macro_regret_improvement_ci_lower=float(ci_lower),
        macro_regret_improvement_ci_upper=float(ci_upper),
        negative_transfer_rate_upper=negative_upper,
        gate_results=gates,
        status=status,
        reason_codes=reasons,
        evaluated_at=evaluated_at,
    )


class EpistemicLoopQualificationV301(StrictModel):
    schema_version: Literal["3.0.1"] = "3.0.1"
    qualification_id: Identifier
    qualification_scope: Literal[
        "synthetic_sequential_problem_reformulation_capacity_worldpack_v301"
    ] = "synthetic_sequential_problem_reformulation_capacity_worldpack_v301"
    candidate_policy_hash: Sha256
    report_hash: Sha256
    prior_failure_report_hash: Sha256
    independent_problem_discovery_established: Literal[False] = False
    real_world_validity_established: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    broad_mathematical_modeling_established: Literal[False] = False
    qualified_at: datetime
    qualification_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_qualification(self) -> "EpistemicLoopQualificationV301":
        _assert_timezone(self.qualified_at, "qualified_at")
        if self.qualification_hash and self.qualification_hash != self.content_hash():
            raise ValueError("qualification_hash does not match V3.0.1 qualification")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "qualification_hash")

    def assert_sealed(self) -> None:
        if not self.qualification_hash or self.qualification_hash != self.content_hash():
            raise ValueError("V3.0.1 qualification is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EpistemicLoopQualificationV301":
        data.setdefault("qualified_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"qualification_hash"}),
            qualification_hash=draft.content_hash(),
        )


class EpistemicLoopManifestV301(StrictModel):
    schema_version: Literal["3.0.1"] = "3.0.1"
    run_id: Identifier
    artifact_refs: list[ArtifactRef] = Field(min_length=7)
    terminal_status: Literal[
        "exploratory_only",
        "candidate_rejected_epistemic_loop_v301",
        "promoted_for_synthetic_epistemic_loop_v301",
    ]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "EpistemicLoopManifestV301":
        _assert_timezone(self.created_at, "created_at")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match V3.0.1 manifest")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("V3.0.1 manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EpistemicLoopManifestV301":
        data.setdefault("created_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class SequentialEpistemicLoopOutcomeV301:
    store: RunStore
    spec: ProblemReformulationWorldPackSpecV301
    private_pack: PrivateProblemReformulationWorldPackV301
    baseline_policy: SequentialEpistemicPolicyV301
    candidate_policy: SequentialEpistemicPolicyV301
    baseline: SequentialEpistemicBundleV301
    candidate: SequentialEpistemicBundleV301
    report: ProblemReformulationReportV301
    qualification: EpistemicLoopQualificationV301 | None
    manifest: EpistemicLoopManifestV301


def run_problem_reformulation_worldpack_v301(
    output_root: str | Path,
    *,
    spec: ProblemReformulationWorldPackSpecV301,
    baseline_policy: SequentialEpistemicPolicyV301,
    candidate_policy: SequentialEpistemicPolicyV301,
    run_id: str | None = None,
    at: datetime | None = None,
) -> SequentialEpistemicLoopOutcomeV301:
    at = at or datetime.now(timezone.utc)
    spec.assert_sealed()
    baseline_policy.assert_sealed()
    candidate_policy.assert_sealed()
    if baseline_policy.policy_hash != spec.baseline_policy_hash:
        raise ValueError("V3.0.1 baseline was not frozen in the spec")
    if candidate_policy.policy_hash != spec.candidate_policy_hash:
        raise ValueError("V3.0.1 candidate was not frozen in the spec")
    store = RunStore(output_root, run_id=run_id)
    refs = [
        store.put_artifact("epistemic_loop_spec_v301", spec),
        store.put_artifact("epistemic_loop_baseline_policy_v301", baseline_policy),
        store.put_artifact("epistemic_loop_candidate_policy_v301", candidate_policy),
    ]
    store.emit(
        "epistemic_loop_v301_protocol_frozen_before_private_pack",
        {
            "spec_hash": spec.spec_hash,
            "prior_failure_report_hash": spec.prior_failure_report_hash,
            "evolved_component": spec.evolved_component,
        },
    )
    private_pack = generate_private_problem_reformulation_worldpack_v301(
        spec, generated_at=at
    )
    baseline = execute_sequential_epistemic_policy_v301(
        spec, private_pack, baseline_policy, executed_at=at
    )
    candidate = execute_sequential_epistemic_policy_v301(
        spec, private_pack, candidate_policy, executed_at=at
    )
    report = evaluate_problem_reformulation_v301(
        spec, private_pack, baseline, candidate, evaluated_at=at
    )
    refs.extend(
        [
            store.put_artifact("private_epistemic_worldpack_v301", private_pack),
            store.put_artifact("epistemic_loop_baseline_bundle_v301", baseline),
            store.put_artifact("epistemic_loop_candidate_bundle_v301", candidate),
            store.put_artifact("epistemic_loop_report_v301", report),
        ]
    )
    qualification = None
    if report.status == "promoted_for_synthetic_epistemic_loop_v301":
        qualification = EpistemicLoopQualificationV301.seal(
            qualification_id="sequential_problem_reformulation_capacity_v301",
            candidate_policy_hash=candidate_policy.policy_hash,
            report_hash=report.report_hash,
            prior_failure_report_hash=spec.prior_failure_report_hash,
            qualified_at=at,
        )
        refs.append(store.put_artifact("epistemic_loop_qualification_v301", qualification))
    manifest = EpistemicLoopManifestV301.seal(
        run_id=store.run_id,
        artifact_refs=refs,
        terminal_status=report.status,
        created_at=at,
    )
    manifest_ref = store.put_artifact("epistemic_loop_manifest_v301", manifest)
    store.emit(
        "epistemic_loop_v301_worldpack_adjudicated",
        {"manifest_ref": manifest_ref.model_dump(mode="json")},
    )
    if not verify_problem_reformulation_run_v301(store.run_directory):
        raise RuntimeError("V3.0.1 run failed independent verification")
    return SequentialEpistemicLoopOutcomeV301(
        store,
        spec,
        private_pack,
        baseline_policy,
        candidate_policy,
        baseline,
        candidate,
        report,
        qualification,
        manifest,
    )


def verify_problem_reformulation_run_v301(run_directory: str | Path) -> bool:
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
            ref for ref in committed if ref.kind == "epistemic_loop_manifest_v301"
        ]
        if len(manifest_refs) != 1:
            return False
        manifest = EpistemicLoopManifestV301.model_validate(
            store.load_artifact(manifest_refs[0])
        )
        manifest.assert_sealed()
        if manifest.run_id != store.run_id:
            return False

        def load_one(kind: str, model):
            refs = [ref for ref in manifest.artifact_refs if ref.kind == kind]
            if len(refs) != 1:
                raise RuntimeError(f"V3.0.1 manifest needs exactly one {kind}")
            return model.model_validate(store.load_artifact(refs[0]))

        spec = load_one(
            "epistemic_loop_spec_v301", ProblemReformulationWorldPackSpecV301
        )
        baseline_policy = load_one(
            "epistemic_loop_baseline_policy_v301", SequentialEpistemicPolicyV301
        )
        candidate_policy = load_one(
            "epistemic_loop_candidate_policy_v301", SequentialEpistemicPolicyV301
        )
        private_pack = load_one(
            "private_epistemic_worldpack_v301",
            PrivateProblemReformulationWorldPackV301,
        )
        baseline = load_one(
            "epistemic_loop_baseline_bundle_v301", SequentialEpistemicBundleV301
        )
        candidate = load_one(
            "epistemic_loop_candidate_bundle_v301", SequentialEpistemicBundleV301
        )
        report = load_one(
            "epistemic_loop_report_v301", ProblemReformulationReportV301
        )
        for item in (
            spec,
            baseline_policy,
            candidate_policy,
            private_pack,
            baseline,
            candidate,
            report,
        ):
            item.assert_sealed()
        regenerated = generate_private_problem_reformulation_worldpack_v301(
            spec, generated_at=private_pack.generated_at
        )
        if regenerated.pack_hash != private_pack.pack_hash:
            return False
        at = baseline.case_receipts[0].executed_at
        replay_baseline = execute_sequential_epistemic_policy_v301(
            spec, private_pack, baseline_policy, executed_at=at
        )
        replay_candidate = execute_sequential_epistemic_policy_v301(
            spec, private_pack, candidate_policy, executed_at=at
        )
        if replay_baseline.bundle_hash != baseline.bundle_hash:
            return False
        if replay_candidate.bundle_hash != candidate.bundle_hash:
            return False
        recomputed = evaluate_problem_reformulation_v301(
            spec,
            private_pack,
            baseline,
            candidate,
            evaluated_at=report.evaluated_at,
        )
        if recomputed.report_hash != report.report_hash:
            return False
        qualifications = [
            ref
            for ref in manifest.artifact_refs
            if ref.kind == "epistemic_loop_qualification_v301"
        ]
        if report.status == "promoted_for_synthetic_epistemic_loop_v301":
            if len(qualifications) != 1:
                return False
            qualification = EpistemicLoopQualificationV301.model_validate(
                store.load_artifact(qualifications[0])
            )
            qualification.assert_sealed()
            if (
                qualification.report_hash != report.report_hash
                or qualification.prior_failure_report_hash
                != spec.prior_failure_report_hash
            ):
                return False
        elif qualifications:
            return False
        freeze = [
            event
            for event in events
            if event["event_type"]
            == "epistemic_loop_v301_protocol_frozen_before_private_pack"
        ]
        adjudicated = [
            event
            for event in events
            if event["event_type"] == "epistemic_loop_v301_worldpack_adjudicated"
        ]
        return len(freeze) == 1 and len(adjudicated) == 1 and store.verify_event_chain()
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
