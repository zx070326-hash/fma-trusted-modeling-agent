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


ActionKindV30 = Literal["collect_demand_batch", "clarify_loss_semantics"]
ArmV30 = Literal[
    "fixed_contract_more_data",
    "reformulation_value_of_information",
]
MechanismV30 = Literal[
    "stable_poisson",
    "overdispersed_counts",
    "bimodal_regime",
    "right_skewed_demand",
]
LossProfileIdV30 = Literal[
    "balanced_absolute",
    "shortage_critical",
    "overage_critical",
]

MECHANISMS_V30: tuple[MechanismV30, ...] = (
    "stable_poisson",
    "overdispersed_counts",
    "bimodal_regime",
    "right_skewed_demand",
)
EXPLORATORY_SEEDS_V30: tuple[int, ...] = tuple(range(9101, 9109))
CONFIRMATION_SEEDS_V30: tuple[int, ...] = tuple(range(9201, 9221))


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


def _assert_timezone(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")


class LossProfileV30(StrictModel):
    profile_id: LossProfileIdV30
    underage_cost: Annotated[float, Field(gt=0, le=20, allow_inf_nan=False)]
    overage_cost: Annotated[float, Field(gt=0, le=20, allow_inf_nan=False)]
    source_kind: Literal["assumption", "candidate_catalog", "authoritative"]
    source_ref: Annotated[str, Field(min_length=3)]


def default_loss_profiles_v30(
    *, source_kind: Literal["assumption", "candidate_catalog", "authoritative"]
    = "candidate_catalog",
    source_ref_prefix: str = "catalog",
) -> list[LossProfileV30]:
    return [
        LossProfileV30(
            profile_id="balanced_absolute",
            underage_cost=1.0,
            overage_cost=1.0,
            source_kind=source_kind,
            source_ref=f"{source_ref_prefix}:balanced",
        ),
        LossProfileV30(
            profile_id="shortage_critical",
            underage_cost=5.0,
            overage_cost=1.0,
            source_kind=source_kind,
            source_ref=f"{source_ref_prefix}:shortage",
        ),
        LossProfileV30(
            profile_id="overage_critical",
            underage_cost=1.0,
            overage_cost=4.0,
            source_kind=source_kind,
            source_ref=f"{source_ref_prefix}:overage",
        ),
    ]


def _profile(
    profile_id: LossProfileIdV30,
    *,
    source_kind: Literal["assumption", "candidate_catalog", "authoritative"],
    source_ref: str,
) -> LossProfileV30:
    match = next(
        item for item in default_loss_profiles_v30() if item.profile_id == profile_id
    )
    return match.model_copy(
        update={"source_kind": source_kind, "source_ref": source_ref}
    )


class MissionConstitutionV30(StrictModel):
    schema_version: Literal["3.0"] = "3.0"
    constitution_id: Identifier
    value_owner_ref: Annotated[str, Field(min_length=3)]
    intended_decision: Literal["shadow_capacity_selection"] = (
        "shadow_capacity_selection"
    )
    decision_unit: Literal["capacity_units"] = "capacity_units"
    capacity_lower_bound: Annotated[int, Field(ge=0)]
    capacity_upper_bound: Annotated[int, Field(ge=1, le=100)]
    allowed_epistemic_actions: list[ActionKindV30] = Field(min_length=1)
    action_costs: dict[ActionKindV30, Annotated[int, Field(ge=1, le=10)]]
    epistemic_action_budget: Annotated[int, Field(ge=1, le=10)]
    forbidden_actions: list[Annotated[str, Field(min_length=3)]] = Field(
        default_factory=lambda: ["real_world_capacity_commit"]
    )
    real_world_action_authorized: Literal[False] = False
    stopping_policy: Literal[
        "one_epistemic_action_then_shadow_decision_or_needs_evidence"
    ] = "one_epistemic_action_then_shadow_decision_or_needs_evidence"
    frozen_at: datetime
    constitution_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_constitution(self) -> "MissionConstitutionV30":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.capacity_lower_bound >= self.capacity_upper_bound:
            raise ValueError("V3.0 capacity bounds are invalid")
        if len(set(self.allowed_epistemic_actions)) != len(
            self.allowed_epistemic_actions
        ):
            raise ValueError("V3.0 allowed epistemic actions must be unique")
        if set(self.action_costs) != set(self.allowed_epistemic_actions):
            raise ValueError("V3.0 action costs must cover exactly the allowed actions")
        if self.epistemic_action_budget < min(self.action_costs.values()):
            raise ValueError("V3.0 budget cannot execute any allowed action")
        if self.constitution_hash and self.constitution_hash != self.content_hash():
            raise ValueError("constitution_hash does not match V3.0 constitution")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "constitution_hash")

    def assert_sealed(self) -> None:
        if not self.constitution_hash or self.constitution_hash != self.content_hash():
            raise ValueError("V3.0 mission constitution is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "MissionConstitutionV30":
        data.setdefault("frozen_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"constitution_hash"}),
            constitution_hash=draft.content_hash(),
        )


class EpisodeProblemContractV30(StrictModel):
    schema_version: Literal["3.0"] = "3.0"
    contract_id: Identifier
    case_id: Identifier
    mission_constitution_hash: Sha256
    version: Annotated[int, Field(ge=1, le=8)]
    parent_contract_hash: Sha256 | None = None
    triggering_evidence_hash: Sha256 | None = None
    revision_reason: str | None = None
    question: Literal["select bounded capacity for the approved loss semantics"] = (
        "select bounded capacity for the approved loss semantics"
    )
    system_boundary: Literal["one-period synthetic demand shadow analysis"] = (
        "one-period synthetic demand shadow analysis"
    )
    loss_profile: LossProfileV30
    semantics_status: Literal["underspecified", "authoritative"]
    unresolved_fields: list[Literal["loss_semantics"]] = Field(default_factory=list)
    real_world_action_authorized: Literal[False] = False
    frozen_at: datetime
    contract_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "EpisodeProblemContractV30":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.version == 1:
            if any(
                value is not None
                for value in (
                    self.parent_contract_hash,
                    self.triggering_evidence_hash,
                    self.revision_reason,
                )
            ):
                raise ValueError("V3.0 root contract cannot claim revision lineage")
        else:
            if not all(
                value is not None
                for value in (
                    self.parent_contract_hash,
                    self.triggering_evidence_hash,
                    self.revision_reason,
                )
            ):
                raise ValueError("V3.0 child contract needs complete revision lineage")
        if self.semantics_status == "underspecified":
            if self.unresolved_fields != ["loss_semantics"]:
                raise ValueError("underspecified V3.0 contract must expose loss_semantics")
            if self.loss_profile.source_kind != "assumption":
                raise ValueError("underspecified V3.0 loss profile must be an assumption")
        else:
            if self.unresolved_fields:
                raise ValueError("authoritative V3.0 contract cannot keep unresolved fields")
            if self.loss_profile.source_kind != "authoritative":
                raise ValueError("authoritative V3.0 contract needs authoritative evidence")
        if self.contract_hash and self.contract_hash != self.content_hash():
            raise ValueError("contract_hash does not match V3.0 episode contract")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "contract_hash")

    def assert_sealed(self) -> None:
        if not self.contract_hash or self.contract_hash != self.content_hash():
            raise ValueError("V3.0 episode contract is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EpisodeProblemContractV30":
        data.setdefault("frozen_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"contract_hash"}),
            contract_hash=draft.content_hash(),
        )


class EpistemicActionPolicyV30(StrictModel):
    schema_version: Literal["3.0"] = "3.0"
    policy_id: Identifier
    arm: ArmV30
    selection_rule: Literal[
        "always_collect_more_under_frozen_contract",
        "clarify_when_contract_uncertainty_changes_decision_else_collect",
    ]
    decision_spread_threshold: Annotated[int, Field(ge=0, le=20)]
    may_reformulate_problem: bool
    max_epistemic_actions: Literal[1] = 1
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "EpistemicActionPolicyV30":
        expected = {
            "fixed_contract_more_data": (
                "always_collect_more_under_frozen_contract",
                False,
            ),
            "reformulation_value_of_information": (
                "clarify_when_contract_uncertainty_changes_decision_else_collect",
                True,
            ),
        }[self.arm]
        if (self.selection_rule, self.may_reformulate_problem) != expected:
            raise ValueError("V3.0 policy fields disagree with arm semantics")
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("policy_hash does not match V3.0 policy")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("V3.0 epistemic policy is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EpistemicActionPolicyV30":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"policy_hash"}),
            policy_hash=draft.content_hash(),
        )


class ProblemReformulationPublicCaseV30(StrictModel):
    schema_version: Literal["3.0-public"] = "3.0-public"
    case_id: Identifier
    mechanism: MechanismV30
    pilot_demand: list[Annotated[int, Field(ge=0, le=100)]] = Field(min_length=8)
    candidate_loss_profiles: list[LossProfileV30] = Field(min_length=3, max_length=3)
    initial_contract: EpisodeProblemContractV30
    additional_batch_size: Annotated[int, Field(ge=4, le=64)]
    public_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_public_case(self) -> "ProblemReformulationPublicCaseV30":
        self.initial_contract.assert_sealed()
        if self.initial_contract.case_id != self.case_id:
            raise ValueError("V3.0 public case and episode contract disagree")
        ids = [profile.profile_id for profile in self.candidate_loss_profiles]
        if sorted(ids) != sorted(
            ["balanced_absolute", "shortage_critical", "overage_critical"]
        ):
            raise ValueError("V3.0 public case needs the frozen loss-profile catalog")
        if any(profile.source_kind != "candidate_catalog" for profile in self.candidate_loss_profiles):
            raise ValueError("V3.0 candidate profiles must remain catalog hypotheses")
        if self.public_hash and self.public_hash != self.content_hash():
            raise ValueError("public_hash does not match V3.0 public case")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "public_hash")

    def assert_sealed(self) -> None:
        if not self.public_hash or self.public_hash != self.content_hash():
            raise ValueError("V3.0 public case is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ProblemReformulationPublicCaseV30":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"public_hash"}),
            public_hash=draft.content_hash(),
        )


class PrivateProblemReformulationCaseV30(StrictModel):
    schema_version: Literal["3.0-private"] = "3.0-private"
    public_case: ProblemReformulationPublicCaseV30
    true_loss_profile: LossProfileV30
    additional_demand_batch: list[Annotated[int, Field(ge=0, le=100)]]
    evaluation_demand: list[Annotated[int, Field(ge=0, le=100)]] = Field(
        min_length=256
    )
    case_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_private_case(self) -> "PrivateProblemReformulationCaseV30":
        self.public_case.assert_sealed()
        if self.true_loss_profile.source_kind != "authoritative":
            raise ValueError("V3.0 private truth must have authoritative semantics")
        if len(self.additional_demand_batch) != self.public_case.additional_batch_size:
            raise ValueError("V3.0 private demand batch size disagrees with public case")
        if self.public_case.initial_contract.semantics_status == "authoritative":
            if (
                self.public_case.initial_contract.loss_profile.profile_id
                != self.true_loss_profile.profile_id
            ):
                raise ValueError("V3.0 known public semantics disagree with private truth")
        if self.case_hash and self.case_hash != self.content_hash():
            raise ValueError("case_hash does not match V3.0 private case")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "case_hash")

    def assert_sealed(self) -> None:
        if not self.case_hash or self.case_hash != self.content_hash():
            raise ValueError("V3.0 private case is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PrivateProblemReformulationCaseV30":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"case_hash"}),
            case_hash=draft.content_hash(),
        )


class ProblemReformulationWorldPackSpecV30(StrictModel):
    schema_version: Literal["3.0"] = "3.0"
    experiment_id: Identifier
    phase: Literal["exploratory", "confirmation"]
    mission_constitution: MissionConstitutionV30
    mechanisms: list[MechanismV30] = Field(min_length=4, max_length=4)
    seeds: list[int] = Field(min_length=8)
    baseline_policy_hash: Sha256
    candidate_policy_hash: Sha256
    pilot_size: Annotated[int, Field(ge=8, le=64)] = 16
    additional_batch_size: Annotated[int, Field(ge=4, le=64)] = 16
    evaluation_size: Annotated[int, Field(ge=256, le=4096)] = 1024
    known_semantics_modulus: Annotated[int, Field(ge=2, le=10)] = 4
    bootstrap_replicates: Annotated[int, Field(ge=500, le=10000)] = 2000
    bootstrap_seed: Annotated[int, Field(ge=0)] = 9030
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
    def validate_spec(self) -> "ProblemReformulationWorldPackSpecV30":
        _assert_timezone(self.frozen_at, "frozen_at")
        self.mission_constitution.assert_sealed()
        if self.mechanisms != list(MECHANISMS_V30):
            raise ValueError("V3.0 WorldPack mechanisms must be frozen and ordered")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("V3.0 WorldPack seeds must be unique")
        if self.phase == "exploratory" and tuple(self.seeds) != EXPLORATORY_SEEDS_V30:
            raise ValueError("V3.0 exploratory seeds are not the frozen split")
        if self.phase == "confirmation" and tuple(self.seeds) != CONFIRMATION_SEEDS_V30:
            raise ValueError("V3.0 confirmation seeds are not the frozen split")
        if self.mission_constitution.epistemic_action_budget != 1:
            raise ValueError("V3.0 experiment freezes one epistemic action")
        if set(self.mission_constitution.action_costs.values()) != {1}:
            raise ValueError("V3.0 comparison requires unit-cost actions")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match V3.0 WorldPack spec")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "spec_hash")

    def assert_sealed(self) -> None:
        if not self.spec_hash or self.spec_hash != self.content_hash():
            raise ValueError("V3.0 WorldPack spec is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ProblemReformulationWorldPackSpecV30":
        data.setdefault("frozen_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"spec_hash"}),
            spec_hash=draft.content_hash(),
        )


class PrivateProblemReformulationWorldPackV30(StrictModel):
    schema_version: Literal["3.0-private"] = "3.0-private"
    spec_hash: Sha256
    cases: list[PrivateProblemReformulationCaseV30] = Field(min_length=32)
    generated_at: datetime
    pack_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_pack(self) -> "PrivateProblemReformulationWorldPackV30":
        _assert_timezone(self.generated_at, "generated_at")
        ids = [case.public_case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("V3.0 private case ids must be unique")
        for case in self.cases:
            case.assert_sealed()
        if self.pack_hash and self.pack_hash != self.content_hash():
            raise ValueError("pack_hash does not match V3.0 private pack")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "pack_hash")

    def assert_sealed(self) -> None:
        if not self.pack_hash or self.pack_hash != self.content_hash():
            raise ValueError("V3.0 private WorldPack is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PrivateProblemReformulationWorldPackV30":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"pack_hash"}),
            pack_hash=draft.content_hash(),
        )


class EpistemicEvidenceReceiptV30(StrictModel):
    schema_version: Literal["3.0"] = "3.0"
    evidence_id: Identifier
    case_id: Identifier
    action_kind: ActionKindV30
    payload_kind: Literal["demand_batch", "loss_semantics"]
    demand_values: list[Annotated[int, Field(ge=0, le=100)]] = Field(
        default_factory=list
    )
    loss_profile: LossProfileV30 | None = None
    source_ref: Annotated[str, Field(min_length=3)]
    action_cost: Literal[1] = 1
    observed_at: datetime
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> "EpistemicEvidenceReceiptV30":
        _assert_timezone(self.observed_at, "observed_at")
        if self.action_kind == "collect_demand_batch":
            if self.payload_kind != "demand_batch" or not self.demand_values:
                raise ValueError("V3.0 collection evidence needs a demand batch")
            if self.loss_profile is not None:
                raise ValueError("V3.0 collection evidence cannot contain loss semantics")
        else:
            if self.payload_kind != "loss_semantics" or self.loss_profile is None:
                raise ValueError("V3.0 clarification evidence needs loss semantics")
            if self.demand_values:
                raise ValueError("V3.0 clarification evidence cannot contain demand")
            if self.loss_profile.source_kind != "authoritative":
                raise ValueError("V3.0 clarification result must be authoritative")
        if self.evidence_hash and self.evidence_hash != self.content_hash():
            raise ValueError("evidence_hash does not match V3.0 evidence")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evidence_hash")

    def assert_sealed(self) -> None:
        if not self.evidence_hash or self.evidence_hash != self.content_hash():
            raise ValueError("V3.0 evidence receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EpistemicEvidenceReceiptV30":
        data.setdefault("observed_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evidence_hash"}),
            evidence_hash=draft.content_hash(),
        )


class EpistemicStateV30(StrictModel):
    schema_version: Literal["3.0"] = "3.0"
    case_id: Identifier
    mission_constitution_hash: Sha256
    arm: ArmV30
    policy_hash: Sha256
    contract_history: list[EpisodeProblemContractV30] = Field(min_length=1, max_length=8)
    current_contract_hash: Sha256
    evidence_receipts: list[EpistemicEvidenceReceiptV30] = Field(default_factory=list)
    remaining_action_budget: Annotated[int, Field(ge=0, le=10)]
    terminal_status: Literal["running", "shadow_decision_ready", "needs_evidence"]
    state_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_state(self) -> "EpistemicStateV30":
        contracts = self.contract_history
        for index, contract in enumerate(contracts):
            contract.assert_sealed()
            if contract.case_id != self.case_id:
                raise ValueError("V3.0 state contains a contract for another case")
            if contract.mission_constitution_hash != self.mission_constitution_hash:
                raise ValueError("V3.0 state contract belongs to another mission")
            if contract.version != index + 1:
                raise ValueError("V3.0 contract versions must be contiguous")
            if index:
                parent = contracts[index - 1]
                if contract.parent_contract_hash != parent.contract_hash:
                    raise ValueError("V3.0 contract lineage skips its parent")
                evidence_hashes = {
                    item.evidence_hash for item in self.evidence_receipts
                }
                if contract.triggering_evidence_hash not in evidence_hashes:
                    raise ValueError("V3.0 contract revision lacks triggering evidence")
        if self.current_contract_hash != contracts[-1].contract_hash:
            raise ValueError("V3.0 current contract must be the latest version")
        for item in self.evidence_receipts:
            item.assert_sealed()
            if item.case_id != self.case_id:
                raise ValueError("V3.0 state contains evidence for another case")
        if self.state_hash and self.state_hash != self.content_hash():
            raise ValueError("state_hash does not match V3.0 epistemic state")
        return self

    @property
    def current_contract(self) -> EpisodeProblemContractV30:
        return self.contract_history[-1]

    def content_hash(self) -> str:
        return _hash_without(self, "state_hash")

    def assert_sealed(self) -> None:
        if not self.state_hash or self.state_hash != self.content_hash():
            raise ValueError("V3.0 epistemic state is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EpistemicStateV30":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"state_hash"}),
            state_hash=draft.content_hash(),
        )


class EpistemicActionProposalV30(StrictModel):
    schema_version: Literal["3.0"] = "3.0"
    proposal_id: Identifier
    case_id: Identifier
    policy_hash: Sha256
    action_kind: ActionKindV30
    proposed_cost: Literal[1] = 1
    decision_spread: Annotated[int, Field(ge=0, le=100)]
    rationale_code: Literal[
        "fixed_contract_collect_more",
        "decision_critical_semantics_unresolved",
        "semantics_sufficient_collect_more",
        "low_decision_spread_collect_more",
    ]
    proposal_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_proposal(self) -> "EpistemicActionProposalV30":
        if self.proposal_hash and self.proposal_hash != self.content_hash():
            raise ValueError("proposal_hash does not match V3.0 action proposal")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "proposal_hash")

    def assert_sealed(self) -> None:
        if not self.proposal_hash or self.proposal_hash != self.content_hash():
            raise ValueError("V3.0 action proposal is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EpistemicActionProposalV30":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"proposal_hash"}),
            proposal_hash=draft.content_hash(),
        )


class EpistemicPermissionDecisionV30(StrictModel):
    schema_version: Literal["3.0"] = "3.0"
    proposal_hash: Sha256
    decision: Literal["allow", "deny"]
    policy_rule: Literal[
        "allow_scoped_read_only_epistemic_action",
        "deny_action_not_allowed",
        "deny_budget_exceeded",
        "deny_cost_mismatch",
    ]
    budget_before: Annotated[int, Field(ge=0, le=10)]
    budget_after: Annotated[int, Field(ge=0, le=10)]
    decided_at: datetime
    decision_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> "EpistemicPermissionDecisionV30":
        _assert_timezone(self.decided_at, "decided_at")
        if self.decision == "allow" and self.budget_after >= self.budget_before:
            raise ValueError("allowed V3.0 action must consume budget")
        if self.decision == "deny" and self.budget_after != self.budget_before:
            raise ValueError("denied V3.0 action cannot consume budget")
        if self.decision_hash and self.decision_hash != self.content_hash():
            raise ValueError("decision_hash does not match V3.0 permission decision")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "decision_hash")

    def assert_sealed(self) -> None:
        if not self.decision_hash or self.decision_hash != self.content_hash():
            raise ValueError("V3.0 permission decision is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EpistemicPermissionDecisionV30":
        data.setdefault("decided_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"decision_hash"}),
            decision_hash=draft.content_hash(),
        )


class EpistemicToolResultV30(StrictModel):
    schema_version: Literal["3.0"] = "3.0"
    proposal_hash: Sha256
    status: Literal["success", "denied", "error"]
    evidence: EpistemicEvidenceReceiptV30 | None = None
    error_code: Literal[
        "permission_denied", "tool_failure", "none"
    ] = "none"
    next_valid_actions: list[Literal["stop_needs_evidence"]] = Field(
        default_factory=list
    )
    result_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_result(self) -> "EpistemicToolResultV30":
        if self.status == "success":
            if self.evidence is None or self.error_code != "none":
                raise ValueError("successful V3.0 result needs evidence and no error")
            self.evidence.assert_sealed()
        else:
            if self.evidence is not None or self.error_code == "none":
                raise ValueError("failed V3.0 result needs a structured error")
            if self.next_valid_actions != ["stop_needs_evidence"]:
                raise ValueError("failed V3.0 result must expose the safe next action")
        if self.result_hash and self.result_hash != self.content_hash():
            raise ValueError("result_hash does not match V3.0 tool result")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "result_hash")

    def assert_sealed(self) -> None:
        if not self.result_hash or self.result_hash != self.content_hash():
            raise ValueError("V3.0 tool result is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EpistemicToolResultV30":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"result_hash"}),
            result_hash=draft.content_hash(),
        )


class EpistemicStepReceiptV30(StrictModel):
    schema_version: Literal["3.0"] = "3.0"
    step_index: Literal[1] = 1
    state_before_hash: Sha256
    proposal: EpistemicActionProposalV30
    permission: EpistemicPermissionDecisionV30
    tool_result: EpistemicToolResultV30
    state_after_hash: Sha256
    step_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_step(self) -> "EpistemicStepReceiptV30":
        self.proposal.assert_sealed()
        self.permission.assert_sealed()
        self.tool_result.assert_sealed()
        if self.permission.proposal_hash != self.proposal.proposal_hash:
            raise ValueError("V3.0 permission is bound to another proposal")
        if self.tool_result.proposal_hash != self.proposal.proposal_hash:
            raise ValueError("V3.0 tool result is bound to another proposal")
        if (self.permission.decision == "allow") != (
            self.tool_result.status == "success"
        ):
            raise ValueError("V3.0 permission and tool result disagree")
        if self.step_hash and self.step_hash != self.content_hash():
            raise ValueError("step_hash does not match V3.0 step receipt")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "step_hash")

    def assert_sealed(self) -> None:
        if not self.step_hash or self.step_hash != self.content_hash():
            raise ValueError("V3.0 epistemic step is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EpistemicStepReceiptV30":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"step_hash"}),
            step_hash=draft.content_hash(),
        )


class ShadowCapacityDecisionV30(StrictModel):
    schema_version: Literal["3.0"] = "3.0"
    decision_id: Identifier
    case_id: Identifier
    episode_contract_hash: Sha256
    selected_capacity: Annotated[int, Field(ge=0, le=100)]
    empirical_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    demand_observation_count: Annotated[int, Field(ge=8)]
    semantics_status: Literal["underspecified", "authoritative"]
    real_world_action_authorized: Literal[False] = False
    use_scope: Literal["synthetic_shadow_only"] = "synthetic_shadow_only"
    decided_at: datetime
    decision_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_shadow_decision(self) -> "ShadowCapacityDecisionV30":
        _assert_timezone(self.decided_at, "decided_at")
        if self.decision_hash and self.decision_hash != self.content_hash():
            raise ValueError("decision_hash does not match V3.0 shadow decision")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "decision_hash")

    def assert_sealed(self) -> None:
        if not self.decision_hash or self.decision_hash != self.content_hash():
            raise ValueError("V3.0 shadow capacity decision is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ShadowCapacityDecisionV30":
        data.setdefault("decided_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"decision_hash"}),
            decision_hash=draft.content_hash(),
        )


class EpistemicCaseReceiptV30(StrictModel):
    schema_version: Literal["3.0"] = "3.0"
    receipt_id: Identifier
    case_id: Identifier
    public_case_hash: Sha256
    policy_hash: Sha256
    arm: ArmV30
    initial_state_hash: Sha256
    steps: list[EpistemicStepReceiptV30] = Field(min_length=1, max_length=1)
    final_state: EpistemicStateV30
    shadow_decision: ShadowCapacityDecisionV30 | None
    action_budget_consumed: Annotated[int, Field(ge=0, le=1)]
    reformulation_count: Annotated[int, Field(ge=0, le=1)]
    permission_error_count: Annotated[int, Field(ge=0, le=1)]
    executed_at: datetime
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_case_receipt(self) -> "EpistemicCaseReceiptV30":
        _assert_timezone(self.executed_at, "executed_at")
        self.final_state.assert_sealed()
        for step in self.steps:
            step.assert_sealed()
        if self.final_state.case_id != self.case_id:
            raise ValueError("V3.0 receipt final state belongs to another case")
        if self.final_state.policy_hash != self.policy_hash:
            raise ValueError("V3.0 receipt final state belongs to another policy")
        if self.initial_state_hash != self.steps[0].state_before_hash:
            raise ValueError("V3.0 step does not start from the receipt state")
        if self.final_state.state_hash != self.steps[-1].state_after_hash:
            raise ValueError("V3.0 step does not end at the receipt state")
        expected_consumed = 1 if self.steps[0].permission.decision == "allow" else 0
        if self.action_budget_consumed != expected_consumed:
            raise ValueError("V3.0 receipt budget consumption is inconsistent")
        expected_errors = 0 if self.steps[0].tool_result.status == "success" else 1
        if self.permission_error_count != expected_errors:
            raise ValueError("V3.0 receipt permission error count is inconsistent")
        if self.reformulation_count != len(self.final_state.contract_history) - 1:
            raise ValueError("V3.0 receipt reformulation count is inconsistent")
        if self.final_state.terminal_status == "shadow_decision_ready":
            if self.shadow_decision is None:
                raise ValueError("decision-ready V3.0 state needs a shadow decision")
            self.shadow_decision.assert_sealed()
            if (
                self.shadow_decision.episode_contract_hash
                != self.final_state.current_contract_hash
            ):
                raise ValueError("V3.0 decision is bound to another episode contract")
        elif self.shadow_decision is not None:
            raise ValueError("needs-evidence V3.0 state cannot contain a decision")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match V3.0 case receipt")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("V3.0 case receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EpistemicCaseReceiptV30":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


class EpistemicSelectionBundleV30(StrictModel):
    schema_version: Literal["3.0"] = "3.0"
    spec_hash: Sha256
    private_pack_hash: Sha256
    policy_hash: Sha256
    arm: ArmV30
    case_receipts: list[EpistemicCaseReceiptV30] = Field(min_length=32)
    total_action_cost: Annotated[int, Field(ge=0)]
    total_reformulations: Annotated[int, Field(ge=0)]
    permission_error_count: Annotated[int, Field(ge=0)]
    bundle_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bundle(self) -> "EpistemicSelectionBundleV30":
        ids: list[str] = []
        for receipt in self.case_receipts:
            receipt.assert_sealed()
            ids.append(receipt.case_id)
            if receipt.policy_hash != self.policy_hash or receipt.arm != self.arm:
                raise ValueError("V3.0 receipt belongs to another arm")
        if len(ids) != len(set(ids)):
            raise ValueError("V3.0 bundle case ids must be unique")
        if self.total_action_cost != sum(
            receipt.action_budget_consumed for receipt in self.case_receipts
        ):
            raise ValueError("V3.0 bundle total cost is inconsistent")
        if self.total_reformulations != sum(
            receipt.reformulation_count for receipt in self.case_receipts
        ):
            raise ValueError("V3.0 bundle reformulations are inconsistent")
        if self.permission_error_count != sum(
            receipt.permission_error_count for receipt in self.case_receipts
        ):
            raise ValueError("V3.0 bundle permission errors are inconsistent")
        if self.bundle_hash and self.bundle_hash != self.content_hash():
            raise ValueError("bundle_hash does not match V3.0 selection bundle")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "bundle_hash")

    def assert_sealed(self) -> None:
        if not self.bundle_hash or self.bundle_hash != self.content_hash():
            raise ValueError("V3.0 selection bundle is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EpistemicSelectionBundleV30":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"bundle_hash"}),
            bundle_hash=draft.content_hash(),
        )


class ProblemReformulationCaseResultV30(StrictModel):
    case_id: Identifier
    mechanism: MechanismV30
    initial_semantics_status: Literal["underspecified", "authoritative"]
    true_loss_profile_id: LossProfileIdV30
    baseline_action: ActionKindV30
    candidate_action: ActionKindV30
    baseline_selected_capacity: Annotated[int, Field(ge=0, le=100)] | None
    candidate_selected_capacity: Annotated[int, Field(ge=0, le=100)] | None
    oracle_capacity: Annotated[int, Field(ge=0, le=100)]
    baseline_normalized_regret: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    candidate_normalized_regret: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    regret_improvement: Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)]
    candidate_semantics_correct: bool
    candidate_reformulated: bool
    spurious_reformulation: bool
    material_negative_transfer: bool


class ProblemReformulationMechanismResultV30(StrictModel):
    mechanism: MechanismV30
    case_count: Annotated[int, Field(ge=8)]
    mean_regret_improvement: Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)]
    mean_baseline_regret: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    mean_candidate_regret: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class ProblemReformulationReportV30(StrictModel):
    schema_version: Literal["3.0"] = "3.0"
    spec_hash: Sha256
    private_pack_hash: Sha256
    baseline_bundle_hash: Sha256
    candidate_bundle_hash: Sha256
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
        "candidate_rejected_epistemic_loop_v30",
        "promoted_for_synthetic_epistemic_loop_v30",
    ]
    reason_codes: list[str]
    evaluated_at: datetime
    report_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "ProblemReformulationReportV30":
        _assert_timezone(self.evaluated_at, "evaluated_at")
        if self.status == "promoted_for_synthetic_epistemic_loop_v30":
            if not self.gate_results or not all(self.gate_results.values()):
                raise ValueError("promoted V3.0 report must pass every gate")
            if self.reason_codes:
                raise ValueError("promoted V3.0 report cannot have failure reasons")
        elif self.status == "candidate_rejected_epistemic_loop_v30":
            if not self.reason_codes:
                raise ValueError("rejected V3.0 report needs failure reasons")
        if self.report_hash and self.report_hash != self.content_hash():
            raise ValueError("report_hash does not match V3.0 report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "report_hash")

    def assert_sealed(self) -> None:
        if not self.report_hash or self.report_hash != self.content_hash():
            raise ValueError("V3.0 problem-reformulation report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ProblemReformulationReportV30":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"report_hash"}),
            report_hash=draft.content_hash(),
        )


class EpistemicLoopQualificationV30(StrictModel):
    schema_version: Literal["3.0"] = "3.0"
    qualification_id: Identifier
    qualification_scope: Literal[
        "synthetic_problem_reformulation_capacity_worldpack_v30"
    ] = "synthetic_problem_reformulation_capacity_worldpack_v30"
    candidate_policy_hash: Sha256
    report_hash: Sha256
    independent_problem_discovery_established: Literal[False] = False
    real_world_validity_established: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    broad_mathematical_modeling_established: Literal[False] = False
    qualified_at: datetime
    qualification_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_qualification(self) -> "EpistemicLoopQualificationV30":
        _assert_timezone(self.qualified_at, "qualified_at")
        if self.qualification_hash and self.qualification_hash != self.content_hash():
            raise ValueError("qualification_hash does not match V3.0 qualification")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "qualification_hash")

    def assert_sealed(self) -> None:
        if not self.qualification_hash or self.qualification_hash != self.content_hash():
            raise ValueError("V3.0 epistemic-loop qualification is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EpistemicLoopQualificationV30":
        data.setdefault("qualified_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"qualification_hash"}),
            qualification_hash=draft.content_hash(),
        )


class EpistemicLoopManifestV30(StrictModel):
    schema_version: Literal["3.0"] = "3.0"
    run_id: Identifier
    artifact_refs: list[ArtifactRef] = Field(min_length=7)
    terminal_status: Literal[
        "exploratory_only",
        "candidate_rejected_epistemic_loop_v30",
        "promoted_for_synthetic_epistemic_loop_v30",
    ]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "EpistemicLoopManifestV30":
        _assert_timezone(self.created_at, "created_at")
        if len({(ref.kind, ref.sha256) for ref in self.artifact_refs}) != len(
            self.artifact_refs
        ):
            raise ValueError("V3.0 manifest artifact references must be unique")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match V3.0 manifest")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("V3.0 epistemic-loop manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EpistemicLoopManifestV30":
        data.setdefault("created_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class EpistemicLoopOutcomeV30:
    store: RunStore
    spec: ProblemReformulationWorldPackSpecV30
    private_pack: PrivateProblemReformulationWorldPackV30
    baseline_policy: EpistemicActionPolicyV30
    candidate_policy: EpistemicActionPolicyV30
    baseline: EpistemicSelectionBundleV30
    candidate: EpistemicSelectionBundleV30
    report: ProblemReformulationReportV30
    qualification: EpistemicLoopQualificationV30 | None
    manifest: EpistemicLoopManifestV30


def default_epistemic_policies_v30() -> tuple[
    EpistemicActionPolicyV30, EpistemicActionPolicyV30
]:
    baseline = EpistemicActionPolicyV30.seal(
        policy_id="fixed_contract_more_data_v30",
        arm="fixed_contract_more_data",
        selection_rule="always_collect_more_under_frozen_contract",
        decision_spread_threshold=0,
        may_reformulate_problem=False,
    )
    candidate = EpistemicActionPolicyV30.seal(
        policy_id="problem_reformulation_voi_v30",
        arm="reformulation_value_of_information",
        selection_rule="clarify_when_contract_uncertainty_changes_decision_else_collect",
        decision_spread_threshold=2,
        may_reformulate_problem=True,
    )
    return baseline, candidate


def _default_mission(at: datetime) -> MissionConstitutionV30:
    return MissionConstitutionV30.seal(
        constitution_id="synthetic_capacity_epistemic_mission_v30",
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
        epistemic_action_budget=1,
        frozen_at=at,
    )


def default_problem_reformulation_exploratory_spec_v30(
    *, frozen_at: datetime | None = None
) -> tuple[
    ProblemReformulationWorldPackSpecV30,
    EpistemicActionPolicyV30,
    EpistemicActionPolicyV30,
]:
    at = frozen_at or datetime.now(timezone.utc)
    baseline, candidate = default_epistemic_policies_v30()
    spec = ProblemReformulationWorldPackSpecV30.seal(
        experiment_id="problem_reformulation_exploratory_v30",
        phase="exploratory",
        mission_constitution=_default_mission(at),
        mechanisms=list(MECHANISMS_V30),
        seeds=list(EXPLORATORY_SEEDS_V30),
        baseline_policy_hash=baseline.policy_hash,
        candidate_policy_hash=candidate.policy_hash,
        literature_evidence_refs=[
            "arxiv:2502.18864",
            "jmlr:23-32-20-807",
            "pmlr:144-ahmadi21a",
        ],
        frozen_at=at,
    )
    return spec, baseline, candidate


def default_problem_reformulation_confirmation_spec_v30(
    *, frozen_at: datetime | None = None
) -> tuple[
    ProblemReformulationWorldPackSpecV30,
    EpistemicActionPolicyV30,
    EpistemicActionPolicyV30,
]:
    at = frozen_at or datetime.now(timezone.utc)
    baseline, candidate = default_epistemic_policies_v30()
    spec = ProblemReformulationWorldPackSpecV30.seal(
        experiment_id="problem_reformulation_confirmation_v30",
        phase="confirmation",
        mission_constitution=_default_mission(at),
        mechanisms=list(MECHANISMS_V30),
        seeds=list(CONFIRMATION_SEEDS_V30),
        baseline_policy_hash=baseline.policy_hash,
        candidate_policy_hash=candidate.policy_hash,
        literature_evidence_refs=[
            "arxiv:2502.18864",
            "jmlr:23-32-20-807",
            "pmlr:144-ahmadi21a",
        ],
        frozen_at=at,
    )
    return spec, baseline, candidate


def _demand_draws(
    mechanism: MechanismV30, *, seed: int, count: int
) -> list[int]:
    mechanism_index = MECHANISMS_V30.index(mechanism)
    random = np.random.default_rng(seed * 97 + mechanism_index * 100_003)
    if mechanism == "stable_poisson":
        values = random.poisson(lam=10.0 + (seed % 3), size=count)
    elif mechanism == "overdispersed_counts":
        n = 4.0
        mean = 10.0 + (seed % 4)
        p = n / (n + mean)
        values = random.negative_binomial(n, p, size=count)
    elif mechanism == "bimodal_regime":
        high = random.random(count) < 0.35
        values = np.where(
            high,
            random.poisson(17.0, size=count),
            random.poisson(6.0, size=count),
        )
    else:
        values = np.rint(random.lognormal(mean=2.15, sigma=0.48, size=count))
    return np.clip(values, 0, 30).astype(int).tolist()


def generate_private_problem_reformulation_worldpack_v30(
    spec: ProblemReformulationWorldPackSpecV30,
    *,
    generated_at: datetime | None = None,
) -> PrivateProblemReformulationWorldPackV30:
    spec.assert_sealed()
    mission = spec.mission_constitution
    mission.assert_sealed()
    cases: list[PrivateProblemReformulationCaseV30] = []
    profile_ids: tuple[LossProfileIdV30, ...] = (
        "balanced_absolute",
        "shortage_critical",
        "overage_critical",
    )
    for mechanism_index, mechanism in enumerate(spec.mechanisms):
        for seed in spec.seeds:
            case_id = f"epr_{mechanism}_{seed}"
            total = spec.pilot_size + spec.additional_batch_size + spec.evaluation_size
            demand = _demand_draws(mechanism, seed=seed, count=total)
            pilot = demand[: spec.pilot_size]
            batch = demand[
                spec.pilot_size : spec.pilot_size + spec.additional_batch_size
            ]
            evaluation = demand[spec.pilot_size + spec.additional_batch_size :]
            true_id = profile_ids[(seed + mechanism_index) % len(profile_ids)]
            true_profile = _profile(
                true_id,
                source_kind="authoritative",
                source_ref=f"private_value_owner:{case_id}",
            )
            semantics_known = (seed + mechanism_index) % spec.known_semantics_modulus == 0
            if semantics_known:
                initial_profile = true_profile.model_copy(
                    update={"source_ref": f"public_value_owner:{case_id}"}
                )
                semantics_status = "authoritative"
                unresolved: list[Literal["loss_semantics"]] = []
            else:
                initial_profile = _profile(
                    "balanced_absolute",
                    source_kind="assumption",
                    source_ref="default_assumption:balanced_absolute",
                )
                semantics_status = "underspecified"
                unresolved = ["loss_semantics"]
            initial_contract = EpisodeProblemContractV30.seal(
                contract_id=f"contract_{case_id}_v1",
                case_id=case_id,
                mission_constitution_hash=mission.constitution_hash,
                version=1,
                loss_profile=initial_profile,
                semantics_status=semantics_status,
                unresolved_fields=unresolved,
                frozen_at=spec.frozen_at,
            )
            public = ProblemReformulationPublicCaseV30.seal(
                case_id=case_id,
                mechanism=mechanism,
                pilot_demand=pilot,
                candidate_loss_profiles=default_loss_profiles_v30(),
                initial_contract=initial_contract,
                additional_batch_size=spec.additional_batch_size,
            )
            cases.append(
                PrivateProblemReformulationCaseV30.seal(
                    public_case=public,
                    true_loss_profile=true_profile,
                    additional_demand_batch=batch,
                    evaluation_demand=evaluation,
                )
            )
    return PrivateProblemReformulationWorldPackV30.seal(
        spec_hash=spec.spec_hash,
        cases=cases,
        generated_at=generated_at or datetime.now(timezone.utc),
    )


def _mean_loss(
    capacity: int, demand: list[int], profile: LossProfileV30
) -> float:
    values = np.asarray(demand, dtype=float)
    underage = np.maximum(values - capacity, 0.0)
    overage = np.maximum(capacity - values, 0.0)
    return float(
        np.mean(profile.underage_cost * underage + profile.overage_cost * overage)
    )


def _solve_capacity(
    demand: list[int],
    profile: LossProfileV30,
    mission: MissionConstitutionV30,
) -> tuple[int, float]:
    if not demand:
        raise ValueError("V3.0 capacity solve needs at least one observation")
    candidates = range(mission.capacity_lower_bound, mission.capacity_upper_bound + 1)
    scored = [(capacity, _mean_loss(capacity, demand, profile)) for capacity in candidates]
    return min(scored, key=lambda pair: (pair[1], pair[0]))


def _decision_spread(
    public: ProblemReformulationPublicCaseV30,
    mission: MissionConstitutionV30,
) -> int:
    capacities = [
        _solve_capacity(public.pilot_demand, profile, mission)[0]
        for profile in public.candidate_loss_profiles
    ]
    return max(capacities) - min(capacities)


def propose_epistemic_action_v30(
    public: ProblemReformulationPublicCaseV30,
    state: EpistemicStateV30,
    policy: EpistemicActionPolicyV30,
    mission: MissionConstitutionV30,
) -> EpistemicActionProposalV30:
    public.assert_sealed()
    state.assert_sealed()
    policy.assert_sealed()
    mission.assert_sealed()
    if state.case_id != public.case_id or state.policy_hash != policy.policy_hash:
        raise ValueError("V3.0 action context is cross-bound")
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
        proposal_id=f"proposal_{policy.arm}_{public.case_id}",
        case_id=public.case_id,
        policy_hash=policy.policy_hash,
        action_kind=action,
        proposed_cost=mission.action_costs[action],
        decision_spread=spread,
        rationale_code=rationale,
    )


def decide_epistemic_permission_v30(
    proposal: EpistemicActionProposalV30,
    state: EpistemicStateV30,
    mission: MissionConstitutionV30,
    *,
    decided_at: datetime,
) -> EpistemicPermissionDecisionV30:
    proposal.assert_sealed()
    state.assert_sealed()
    mission.assert_sealed()
    if state.mission_constitution_hash != mission.constitution_hash:
        raise ValueError("V3.0 permission state belongs to another mission")
    if proposal.case_id != state.case_id or proposal.policy_hash != state.policy_hash:
        raise ValueError("V3.0 permission proposal belongs to another state")
    before = state.remaining_action_budget
    if proposal.action_kind not in mission.allowed_epistemic_actions:
        decision = "deny"
        rule = "deny_action_not_allowed"
        after = before
    elif proposal.proposed_cost != mission.action_costs[proposal.action_kind]:
        decision = "deny"
        rule = "deny_cost_mismatch"
        after = before
    elif proposal.proposed_cost > before:
        decision = "deny"
        rule = "deny_budget_exceeded"
        after = before
    else:
        decision = "allow"
        rule = "allow_scoped_read_only_epistemic_action"
        after = before - proposal.proposed_cost
    return EpistemicPermissionDecisionV30.seal(
        proposal_hash=proposal.proposal_hash,
        decision=decision,
        policy_rule=rule,
        budget_before=before,
        budget_after=after,
        decided_at=decided_at,
    )


def execute_epistemic_tool_v30(
    private_case: PrivateProblemReformulationCaseV30,
    proposal: EpistemicActionProposalV30,
    permission: EpistemicPermissionDecisionV30,
    *,
    observed_at: datetime,
) -> EpistemicToolResultV30:
    private_case.assert_sealed()
    proposal.assert_sealed()
    permission.assert_sealed()
    if private_case.public_case.case_id != proposal.case_id:
        raise ValueError("V3.0 tool proposal belongs to another private case")
    if permission.proposal_hash != proposal.proposal_hash:
        raise ValueError("V3.0 tool permission belongs to another proposal")
    if permission.decision == "deny":
        return EpistemicToolResultV30.seal(
            proposal_hash=proposal.proposal_hash,
            status="denied",
            error_code="permission_denied",
            next_valid_actions=["stop_needs_evidence"],
        )
    if proposal.action_kind == "collect_demand_batch":
        evidence = EpistemicEvidenceReceiptV30.seal(
            evidence_id=f"evidence_demand_{proposal.case_id}",
            case_id=proposal.case_id,
            action_kind=proposal.action_kind,
            payload_kind="demand_batch",
            demand_values=private_case.additional_demand_batch,
            source_ref=f"synthetic_reality:demand:{proposal.case_id}",
            observed_at=observed_at,
        )
    else:
        evidence = EpistemicEvidenceReceiptV30.seal(
            evidence_id=f"evidence_semantics_{proposal.case_id}",
            case_id=proposal.case_id,
            action_kind=proposal.action_kind,
            payload_kind="loss_semantics",
            loss_profile=private_case.true_loss_profile,
            source_ref=f"synthetic_value_owner:{proposal.case_id}",
            observed_at=observed_at,
        )
    return EpistemicToolResultV30.seal(
        proposal_hash=proposal.proposal_hash,
        status="success",
        evidence=evidence,
    )


def _advance_epistemic_state(
    state: EpistemicStateV30,
    proposal: EpistemicActionProposalV30,
    permission: EpistemicPermissionDecisionV30,
    result: EpistemicToolResultV30,
    policy: EpistemicActionPolicyV30,
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
        if not policy.may_reformulate_problem:
            raise RuntimeError("V3.0 fixed policy cannot reformulate a problem")
        if evidence.loss_profile is None:
            raise RuntimeError("V3.0 semantic clarification returned no profile")
        parent = state.current_contract
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
    return EpistemicStateV30.seal(
        case_id=state.case_id,
        mission_constitution_hash=state.mission_constitution_hash,
        arm=state.arm,
        policy_hash=state.policy_hash,
        contract_history=history,
        current_contract_hash=history[-1].contract_hash,
        evidence_receipts=[*state.evidence_receipts, evidence],
        remaining_action_budget=permission.budget_after,
        terminal_status="shadow_decision_ready",
    )


def _make_shadow_decision(
    public: ProblemReformulationPublicCaseV30,
    state: EpistemicStateV30,
    mission: MissionConstitutionV30,
    *,
    decided_at: datetime,
) -> ShadowCapacityDecisionV30 | None:
    if state.terminal_status != "shadow_decision_ready":
        return None
    demand = list(public.pilot_demand)
    for evidence in state.evidence_receipts:
        demand.extend(evidence.demand_values)
    capacity, loss = _solve_capacity(demand, state.current_contract.loss_profile, mission)
    return ShadowCapacityDecisionV30.seal(
        decision_id=f"shadow_capacity_{state.arm}_{state.case_id}",
        case_id=state.case_id,
        episode_contract_hash=state.current_contract_hash,
        selected_capacity=capacity,
        empirical_loss=loss,
        demand_observation_count=len(demand),
        semantics_status=state.current_contract.semantics_status,
        decided_at=decided_at,
    )


def execute_epistemic_policy_v30(
    spec: ProblemReformulationWorldPackSpecV30,
    private_pack: PrivateProblemReformulationWorldPackV30,
    policy: EpistemicActionPolicyV30,
    *,
    executed_at: datetime,
) -> EpistemicSelectionBundleV30:
    spec.assert_sealed()
    private_pack.assert_sealed()
    policy.assert_sealed()
    if private_pack.spec_hash != spec.spec_hash:
        raise ValueError("V3.0 private pack belongs to another spec")
    expected_hash = (
        spec.baseline_policy_hash
        if policy.arm == "fixed_contract_more_data"
        else spec.candidate_policy_hash
    )
    if policy.policy_hash != expected_hash:
        raise ValueError("V3.0 policy is not the frozen arm")
    mission = spec.mission_constitution
    receipts: list[EpistemicCaseReceiptV30] = []
    for private_case in private_pack.cases:
        public = private_case.public_case
        initial_state = EpistemicStateV30.seal(
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
        proposal = propose_epistemic_action_v30(
            public, initial_state, policy, mission
        )
        permission = decide_epistemic_permission_v30(
            proposal, initial_state, mission, decided_at=executed_at
        )
        result = execute_epistemic_tool_v30(
            private_case, proposal, permission, observed_at=executed_at
        )
        final_state = _advance_epistemic_state(
            initial_state,
            proposal,
            permission,
            result,
            policy,
            at=executed_at,
        )
        step = EpistemicStepReceiptV30.seal(
            state_before_hash=initial_state.state_hash,
            proposal=proposal,
            permission=permission,
            tool_result=result,
            state_after_hash=final_state.state_hash,
        )
        decision = _make_shadow_decision(
            public, final_state, mission, decided_at=executed_at
        )
        receipts.append(
            EpistemicCaseReceiptV30.seal(
                receipt_id=f"receipt_{policy.arm}_{public.case_id}",
                case_id=public.case_id,
                public_case_hash=public.public_hash,
                policy_hash=policy.policy_hash,
                arm=policy.arm,
                initial_state_hash=initial_state.state_hash,
                steps=[step],
                final_state=final_state,
                shadow_decision=decision,
                action_budget_consumed=(1 if permission.decision == "allow" else 0),
                reformulation_count=len(final_state.contract_history) - 1,
                permission_error_count=(0 if result.status == "success" else 1),
                executed_at=executed_at,
            )
        )
    return EpistemicSelectionBundleV30.seal(
        spec_hash=spec.spec_hash,
        private_pack_hash=private_pack.pack_hash,
        policy_hash=policy.policy_hash,
        arm=policy.arm,
        case_receipts=receipts,
        total_action_cost=sum(item.action_budget_consumed for item in receipts),
        total_reformulations=sum(item.reformulation_count for item in receipts),
        permission_error_count=sum(item.permission_error_count for item in receipts),
    )


def _normalized_regret(decision_loss: float, oracle_loss: float) -> float:
    regret = max(decision_loss - oracle_loss, 0.0)
    denominator = decision_loss + oracle_loss + 1e-12
    return float(np.clip(regret / denominator, 0.0, 1.0))


def _stratified_macro_bootstrap_v30(
    grouped: dict[str, list[float]],
    spec: ProblemReformulationWorldPackSpecV30,
) -> np.ndarray:
    random = Random(spec.bootstrap_seed)
    draws = np.empty(spec.bootstrap_replicates, dtype=float)
    for draw in range(spec.bootstrap_replicates):
        mechanism_means: list[float] = []
        for mechanism in spec.mechanisms:
            values = grouped[mechanism]
            mechanism_means.append(
                sum(values[random.randrange(len(values))] for _ in values)
                / len(values)
            )
        draws[draw] = sum(mechanism_means) / len(mechanism_means)
    return draws


def _clopper_pearson_upper_v30(
    successes: int, trials: int, confidence: float
) -> float:
    if successes == trials:
        return 1.0
    return float(beta.ppf(confidence, successes + 1, trials - successes))


def evaluate_problem_reformulation_v30(
    spec: ProblemReformulationWorldPackSpecV30,
    private_pack: PrivateProblemReformulationWorldPackV30,
    baseline: EpistemicSelectionBundleV30,
    candidate: EpistemicSelectionBundleV30,
    *,
    evaluated_at: datetime,
) -> ProblemReformulationReportV30:
    spec.assert_sealed()
    private_pack.assert_sealed()
    baseline.assert_sealed()
    candidate.assert_sealed()
    if baseline.arm != "fixed_contract_more_data":
        raise ValueError("V3.0 baseline bundle has the wrong arm")
    if candidate.arm != "reformulation_value_of_information":
        raise ValueError("V3.0 candidate bundle has the wrong arm")
    if any(
        bundle.spec_hash != spec.spec_hash
        or bundle.private_pack_hash != private_pack.pack_hash
        for bundle in (baseline, candidate)
    ):
        raise ValueError("V3.0 evaluation inputs are cross-bound")
    baseline_by_id = {item.case_id: item for item in baseline.case_receipts}
    candidate_by_id = {item.case_id: item for item in candidate.case_receipts}
    private_by_id = {
        item.public_case.case_id: item for item in private_pack.cases
    }
    if set(baseline_by_id) != set(candidate_by_id) or set(baseline_by_id) != set(
        private_by_id
    ):
        raise ValueError("V3.0 bundles do not cover the private WorldPack")
    mission = spec.mission_constitution
    cases: list[ProblemReformulationCaseResultV30] = []
    grouped: dict[str, list[float]] = {mechanism: [] for mechanism in spec.mechanisms}
    unresolved = 0
    for private_case in private_pack.cases:
        public = private_case.public_case
        baseline_receipt = baseline_by_id[public.case_id]
        candidate_receipt = candidate_by_id[public.case_id]
        baseline_decision = baseline_receipt.shadow_decision
        candidate_decision = candidate_receipt.shadow_decision
        oracle_capacity, oracle_loss = _solve_capacity(
            private_case.evaluation_demand,
            private_case.true_loss_profile,
            mission,
        )
        if baseline_decision is None:
            baseline_capacity = None
            baseline_regret = 1.0
        else:
            baseline_capacity = baseline_decision.selected_capacity
            baseline_loss = _mean_loss(
                baseline_capacity,
                private_case.evaluation_demand,
                private_case.true_loss_profile,
            )
            baseline_regret = _normalized_regret(baseline_loss, oracle_loss)
        if candidate_decision is None:
            candidate_capacity = None
            candidate_regret = 1.0
        else:
            candidate_capacity = candidate_decision.selected_capacity
            candidate_loss = _mean_loss(
                candidate_capacity,
                private_case.evaluation_demand,
                private_case.true_loss_profile,
            )
            candidate_regret = _normalized_regret(candidate_loss, oracle_loss)
        improvement = float(np.clip(baseline_regret - candidate_regret, -1.0, 1.0))
        grouped[public.mechanism].append(improvement)
        candidate_semantics_correct = (
            candidate_receipt.final_state.current_contract.loss_profile.profile_id
            == private_case.true_loss_profile.profile_id
        )
        if candidate_receipt.final_state.current_contract.semantics_status != "authoritative":
            unresolved += 1
        reformulated = candidate_receipt.reformulation_count == 1
        spurious = (
            public.initial_contract.semantics_status == "authoritative" and reformulated
        )
        cases.append(
            ProblemReformulationCaseResultV30(
                case_id=public.case_id,
                mechanism=public.mechanism,
                initial_semantics_status=public.initial_contract.semantics_status,
                true_loss_profile_id=private_case.true_loss_profile.profile_id,
                baseline_action=baseline_receipt.steps[0].proposal.action_kind,
                candidate_action=candidate_receipt.steps[0].proposal.action_kind,
                baseline_selected_capacity=baseline_capacity,
                candidate_selected_capacity=candidate_capacity,
                oracle_capacity=oracle_capacity,
                baseline_normalized_regret=baseline_regret,
                candidate_normalized_regret=candidate_regret,
                regret_improvement=improvement,
                candidate_semantics_correct=candidate_semantics_correct,
                candidate_reformulated=reformulated,
                spurious_reformulation=spurious,
                material_negative_transfer=(
                    candidate_regret - baseline_regret
                    > spec.material_negative_transfer_threshold
                ),
            )
        )
    mechanism_results: list[ProblemReformulationMechanismResultV30] = []
    for mechanism in spec.mechanisms:
        mechanism_cases = [case for case in cases if case.mechanism == mechanism]
        mechanism_results.append(
            ProblemReformulationMechanismResultV30(
                mechanism=mechanism,
                case_count=len(mechanism_cases),
                mean_regret_improvement=float(
                    np.mean([case.regret_improvement for case in mechanism_cases])
                ),
                mean_baseline_regret=float(
                    np.mean(
                        [case.baseline_normalized_regret for case in mechanism_cases]
                    )
                ),
                mean_candidate_regret=float(
                    np.mean(
                        [case.candidate_normalized_regret for case in mechanism_cases]
                    )
                ),
            )
        )
    macro = float(np.mean([item.mean_regret_improvement for item in mechanism_results]))
    bootstrap = _stratified_macro_bootstrap_v30(grouped, spec)
    alpha = 1.0 - spec.confidence
    ci_lower, ci_upper = np.quantile(bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0])
    negative_count = sum(case.material_negative_transfer for case in cases)
    negative_upper = _clopper_pearson_upper_v30(
        negative_count, len(cases), spec.confidence
    )
    underspecified = sum(
        case.initial_semantics_status == "underspecified" for case in cases
    )
    spurious_count = sum(case.spurious_reformulation for case in cases)
    candidate_reformations = sum(case.candidate_reformulated for case in cases)
    permission_errors = baseline.permission_error_count + candidate.permission_error_count
    gates = {
        "same_epistemic_action_cost": baseline.total_action_cost
        == candidate.total_action_cost
        == len(cases),
        "zero_permission_errors": permission_errors == 0,
        "all_underspecified_semantics_resolved": unresolved == 0,
        "no_spurious_reformulation": spurious_count == 0,
        "reformulation_targets_only_missing_semantics": candidate_reformations
        == underspecified,
        "macro_regret_ci_gate": float(ci_lower)
        >= spec.min_macro_regret_improvement,
        "per_mechanism_noninferiority": all(
            item.mean_regret_improvement
            >= -spec.per_mechanism_noninferiority_margin
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
        status = "candidate_rejected_epistemic_loop_v30"
    else:
        status = "promoted_for_synthetic_epistemic_loop_v30"
    return ProblemReformulationReportV30.seal(
        spec_hash=spec.spec_hash,
        private_pack_hash=private_pack.pack_hash,
        baseline_bundle_hash=baseline.bundle_hash,
        candidate_bundle_hash=candidate.bundle_hash,
        cases=cases,
        mechanisms=mechanism_results,
        same_epistemic_action_cost=gates["same_epistemic_action_cost"],
        permission_error_count=permission_errors,
        underspecified_case_count=underspecified,
        candidate_reformulation_count=candidate_reformations,
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


def run_problem_reformulation_worldpack_v30(
    output_root: str | Path,
    *,
    spec: ProblemReformulationWorldPackSpecV30,
    baseline_policy: EpistemicActionPolicyV30,
    candidate_policy: EpistemicActionPolicyV30,
    run_id: str | None = None,
    at: datetime | None = None,
) -> EpistemicLoopOutcomeV30:
    at = at or datetime.now(timezone.utc)
    spec.assert_sealed()
    baseline_policy.assert_sealed()
    candidate_policy.assert_sealed()
    if baseline_policy.policy_hash != spec.baseline_policy_hash:
        raise ValueError("V3.0 baseline policy was not frozen in the spec")
    if candidate_policy.policy_hash != spec.candidate_policy_hash:
        raise ValueError("V3.0 candidate policy was not frozen in the spec")
    store = RunStore(output_root, run_id=run_id)
    refs: list[ArtifactRef] = [
        store.put_artifact("epistemic_loop_spec_v30", spec),
        store.put_artifact("epistemic_loop_baseline_policy_v30", baseline_policy),
        store.put_artifact("epistemic_loop_candidate_policy_v30", candidate_policy),
    ]
    store.emit(
        "epistemic_loop_protocol_frozen_before_private_pack",
        {
            "spec_hash": spec.spec_hash,
            "baseline_policy_hash": baseline_policy.policy_hash,
            "candidate_policy_hash": candidate_policy.policy_hash,
        },
    )
    private_pack = generate_private_problem_reformulation_worldpack_v30(
        spec, generated_at=at
    )
    baseline = execute_epistemic_policy_v30(
        spec,
        private_pack,
        baseline_policy,
        executed_at=at,
    )
    candidate = execute_epistemic_policy_v30(
        spec,
        private_pack,
        candidate_policy,
        executed_at=at,
    )
    report = evaluate_problem_reformulation_v30(
        spec,
        private_pack,
        baseline,
        candidate,
        evaluated_at=at,
    )
    refs.extend(
        [
            store.put_artifact("private_epistemic_worldpack_v30", private_pack),
            store.put_artifact("epistemic_loop_baseline_bundle_v30", baseline),
            store.put_artifact("epistemic_loop_candidate_bundle_v30", candidate),
            store.put_artifact("epistemic_loop_report_v30", report),
        ]
    )
    qualification: EpistemicLoopQualificationV30 | None = None
    if report.status == "promoted_for_synthetic_epistemic_loop_v30":
        qualification = EpistemicLoopQualificationV30.seal(
            qualification_id="problem_reformulation_capacity_v30",
            candidate_policy_hash=candidate_policy.policy_hash,
            report_hash=report.report_hash,
            qualified_at=at,
        )
        refs.append(store.put_artifact("epistemic_loop_qualification_v30", qualification))
    manifest = EpistemicLoopManifestV30.seal(
        run_id=store.run_id,
        artifact_refs=refs,
        terminal_status=report.status,
        created_at=at,
    )
    manifest_ref = store.put_artifact("epistemic_loop_manifest_v30", manifest)
    store.emit(
        "epistemic_loop_worldpack_adjudicated",
        {"manifest_ref": manifest_ref.model_dump(mode="json")},
    )
    if not verify_problem_reformulation_run_v30(store.run_directory):
        raise RuntimeError("V3.0 epistemic-loop run failed independent verification")
    return EpistemicLoopOutcomeV30(
        store=store,
        spec=spec,
        private_pack=private_pack,
        baseline_policy=baseline_policy,
        candidate_policy=candidate_policy,
        baseline=baseline,
        candidate=candidate,
        report=report,
        qualification=qualification,
        manifest=manifest,
    )


def verify_problem_reformulation_run_v30(run_directory: str | Path) -> bool:
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
            ref for ref in committed if ref.kind == "epistemic_loop_manifest_v30"
        ]
        if len(manifest_refs) != 1:
            return False
        manifest = EpistemicLoopManifestV30.model_validate(
            store.load_artifact(manifest_refs[0])
        )
        manifest.assert_sealed()
        if manifest.run_id != store.run_id:
            return False

        def load_one(kind: str, model):
            refs = [ref for ref in manifest.artifact_refs if ref.kind == kind]
            if len(refs) != 1:
                raise RuntimeError(f"V3.0 manifest needs exactly one {kind}")
            return model.model_validate(store.load_artifact(refs[0]))

        spec = load_one(
            "epistemic_loop_spec_v30", ProblemReformulationWorldPackSpecV30
        )
        baseline_policy = load_one(
            "epistemic_loop_baseline_policy_v30", EpistemicActionPolicyV30
        )
        candidate_policy = load_one(
            "epistemic_loop_candidate_policy_v30", EpistemicActionPolicyV30
        )
        private_pack = load_one(
            "private_epistemic_worldpack_v30",
            PrivateProblemReformulationWorldPackV30,
        )
        baseline = load_one(
            "epistemic_loop_baseline_bundle_v30", EpistemicSelectionBundleV30
        )
        candidate = load_one(
            "epistemic_loop_candidate_bundle_v30", EpistemicSelectionBundleV30
        )
        report = load_one(
            "epistemic_loop_report_v30", ProblemReformulationReportV30
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
        regenerated = generate_private_problem_reformulation_worldpack_v30(
            spec, generated_at=private_pack.generated_at
        )
        if regenerated.pack_hash != private_pack.pack_hash:
            return False
        executed_at = baseline.case_receipts[0].executed_at
        replay_baseline = execute_epistemic_policy_v30(
            spec, private_pack, baseline_policy, executed_at=executed_at
        )
        replay_candidate = execute_epistemic_policy_v30(
            spec, private_pack, candidate_policy, executed_at=executed_at
        )
        if replay_baseline.bundle_hash != baseline.bundle_hash:
            return False
        if replay_candidate.bundle_hash != candidate.bundle_hash:
            return False
        recomputed = evaluate_problem_reformulation_v30(
            spec,
            private_pack,
            baseline,
            candidate,
            evaluated_at=report.evaluated_at,
        )
        if recomputed.report_hash != report.report_hash:
            return False
        qualification_refs = [
            ref
            for ref in manifest.artifact_refs
            if ref.kind == "epistemic_loop_qualification_v30"
        ]
        if report.status == "promoted_for_synthetic_epistemic_loop_v30":
            if len(qualification_refs) != 1:
                return False
            qualification = EpistemicLoopQualificationV30.model_validate(
                store.load_artifact(qualification_refs[0])
            )
            qualification.assert_sealed()
            if (
                qualification.report_hash != report.report_hash
                or qualification.candidate_policy_hash != candidate_policy.policy_hash
            ):
                return False
        elif qualification_refs:
            return False
        freeze_events = [
            event
            for event in events
            if event["event_type"]
            == "epistemic_loop_protocol_frozen_before_private_pack"
        ]
        adjudication_events = [
            event
            for event in events
            if event["event_type"] == "epistemic_loop_worldpack_adjudicated"
        ]
        return (
            len(freeze_events) == 1
            and len(adjudication_events) == 1
            and store.verify_event_chain()
        )
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
