from __future__ import annotations

import json
import math
from collections import defaultdict
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

from .challenge_disposition_v371 import (
    ChallengeDispositionBundleV371,
    ChallengeDispositionEvolutionReportV371,
    ChallengeDispositionPolicyV371,
    ModelChallengeWorldPackSpecV371,
    TargetAwareApplicabilityStateV371,
    _nonlinear_residual_gain_v371,
    create_challenge_disposition_bundle_v371,
    evaluate_challenge_dispositions_v371,
    verify_challenge_disposition_run_v371,
)
from .controlled_dynamics_loop import (
    MECHANISMS_V31,
    ControlledDriftModelV31,
    ControlledDynamicsContractV31,
    MechanismV31,
    PrivateControlledDynamicsWorldPackV31,
    TargetClarificationEvidenceV31,
    _target_loss_v31,
    generate_private_controlled_dynamics_worldpack_v31,
)
from .model_challenge_v37 import (
    FAMILIES_V37,
    FamilyV37,
    ModelChallengeBundleV37,
    ModelChallengeEvolutionReportV37,
    ModelChallengeMethodEvidenceV37,
    ModelPortfolioPolicyV37,
    _fit_family_v37,
    _hash_without,
    _shared_observations_v37,
    evaluate_model_challenge_worldpack_v37,
    execute_model_challenge_policy_v37,
)


EXPLORATORY_SEEDS_V38 = (
    21001, 21059, 21107, 21157, 21211, 21269, 21317, 21377,
    21433, 21487, 21529, 21587, 21649, 21683, 21737, 21799,
)

NextActionV38 = Literal[
    "acquire_target_discriminating_evidence",
    "expand_non_nested_family",
    "proceed_private_validation",
]


class TargetClarificationPolicyV38(StrictModel):
    schema_version: Literal["3.8"] = "3.8"
    policy_id: Identifier
    source_disposition_evolution_hash: Sha256
    source_disposition_bundle_hash: Sha256
    source_status: Literal[
        "challenge_disposition_ready_for_synthetic_action_experiment_v371"
    ]
    source_clarification_count: Literal[39] = 39
    action_rule: Literal[
        "execute_only_source_clarify_decision_target_dispositions"
    ] = "execute_only_source_clarify_decision_target_dispositions"
    evidence_source_rule: Literal[
        "synthetic_value_owner_reveals_true_decision_target"
    ] = "synthetic_value_owner_reveals_true_decision_target"
    action_cost: Literal[1] = 1
    real_world_execution_permitted: Literal[False] = False
    task_router_permitted: Literal[False] = False
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "TargetClarificationPolicyV38":
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("policy_hash does not match V3.8 clarification policy")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("V3.8 clarification policy is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "TargetClarificationPolicyV38":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"policy_hash"}),
            policy_hash=draft.content_hash(),
        )


def _load_source_v371(
    run_directory: str | Path,
) -> tuple[ChallengeDispositionBundleV371, ChallengeDispositionEvolutionReportV371]:
    if not verify_challenge_disposition_run_v371(run_directory):
        raise ValueError("V3.8 source V3.7.1 run did not independently verify")
    store = RunStore.open_existing(run_directory)
    events = [
        json.loads(line)
        for line in store.event_path.read_text(encoding="utf-8").splitlines()
    ]
    refs = [
        ArtifactRef.model_validate(event["payload"])
        for event in events if event["event_type"] == "artifact_committed"
    ]

    def load(kind: str, model):
        matches = [item for item in refs if item.kind == kind]
        if len(matches) != 1:
            raise ValueError(f"V3.8 source run needs exactly one {kind}")
        return model.model_validate(store.load_artifact(matches[0]))

    return (
        load("challenge_disposition_bundle_v371", ChallengeDispositionBundleV371),
        load(
            "challenge_disposition_evolution_report_v371",
            ChallengeDispositionEvolutionReportV371,
        ),
    )


def build_target_clarification_policy_v38(
    source_v371_run_directory: str | Path,
) -> TargetClarificationPolicyV38:
    bundle, report = _load_source_v371(source_v371_run_directory)
    if report.status != "challenge_disposition_ready_for_synthetic_action_experiment_v371":
        raise ValueError("V3.8 clarification requires a ready disposition source")
    count = sum(
        item.proposed_action == "clarify_decision_target"
        for item in bundle.dispositions
    )
    if count != 39:
        raise ValueError("V3.8 source clarification count changed")
    return TargetClarificationPolicyV38.seal(
        policy_id="target_clarification_policy_v38",
        source_disposition_evolution_hash=report.evolution_hash,
        source_disposition_bundle_hash=bundle.bundle_hash,
        source_status=report.status,
    )


class TargetClarificationWorldPackSpecV38(ModelChallengeWorldPackSpecV371):
    schema_version: Literal["3.8"] = "3.8"
    seeds: list[int] = Field(min_length=16, max_length=16)
    bootstrap_seed: Literal[380722] = 380722
    clarification_policy_hash: Sha256
    source_disposition_evolution_hash: Sha256
    frozen_delta: Literal[
        "execute_target_clarification_and_condition_challenge_only"
    ] = "execute_target_clarification_and_condition_challenge_only"

    @model_validator(mode="after")
    def validate_spec(self) -> "TargetClarificationWorldPackSpecV38":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.mechanisms != list(MECHANISMS_V31):
            raise ValueError("V3.8 requires the frozen mechanism order")
        if self.seeds != list(EXPLORATORY_SEEDS_V38):
            raise ValueError("V3.8 seeds do not match the frozen exploratory set")
        if self.family_catalog != list(FAMILIES_V37):
            raise ValueError("V3.8 cannot change the family catalog")
        if self.observation_action_indices != [0, 7]:
            raise ValueError("V3.8 cannot change the observation set")
        if not math.isclose(
            (self.trajectory_points - 1) * self.time_step,
            self.segment_count * self.segment_duration,
            abs_tol=1e-12,
        ):
            raise ValueError("V3.8 segments do not cover the trajectory")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match V3.8 protocol")
        return self


def default_target_clarification_exploratory_spec_v38(
    *,
    method_evidence_hash: str,
    baseline_policy_hash: str,
    candidate_policy_hash: str,
    disposition_policy: ChallengeDispositionPolicyV371,
    clarification_policy: TargetClarificationPolicyV38,
    frozen_at: datetime | None = None,
) -> TargetClarificationWorldPackSpecV38:
    disposition_policy.assert_sealed()
    clarification_policy.assert_sealed()
    return TargetClarificationWorldPackSpecV38.seal(
        experiment_id="target_clarification_exploratory_v38",
        mechanisms=list(MECHANISMS_V31),
        seeds=list(EXPLORATORY_SEEDS_V38),
        family_catalog=list(FAMILIES_V37),
        observation_action_indices=[0, 7],
        baseline_policy_hash=baseline_policy_hash,
        candidate_policy_hash=candidate_policy_hash,
        method_evidence_hash=method_evidence_hash,
        disposition_policy_hash=disposition_policy.policy_hash,
        source_failure_evolution_hash=disposition_policy.source_failure_evolution_hash,
        source_failure_candidate_bundle_hash=(
            disposition_policy.source_failure_candidate_bundle_hash
        ),
        clarification_policy_hash=clarification_policy.policy_hash,
        source_disposition_evolution_hash=(
            clarification_policy.source_disposition_evolution_hash
        ),
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


class ClarificationActionReceiptV38(StrictModel):
    schema_version: Literal["3.8"] = "3.8"
    action_id: Identifier
    case_id: Identifier
    clarification_policy_hash: Sha256
    source_disposition_hash: Sha256
    before_contract_hash: Sha256
    status: Literal["executed_synthetic", "not_applicable"]
    clarification_evidence: TargetClarificationEvidenceV31 | None
    after_contract: ControlledDynamicsContractV31 | None
    budget_before: Literal[1] = 1
    budget_after: Annotated[int, Field(ge=0, le=1)]
    real_world_execution_permitted: Literal[False] = False
    executed_at: datetime
    action_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_action(self) -> "ClarificationActionReceiptV38":
        _assert_timezone(self.executed_at, "executed_at")
        if self.status == "executed_synthetic":
            if self.clarification_evidence is None or self.after_contract is None:
                raise ValueError("V3.8 executed clarification needs evidence and contract")
            self.clarification_evidence.assert_sealed()
            self.after_contract.assert_sealed()
            if self.budget_after != 0:
                raise ValueError("V3.8 executed clarification must consume one action")
            if (
                self.after_contract.parent_contract_hash != self.before_contract_hash
                or self.after_contract.triggering_evidence_hash
                != self.clarification_evidence.evidence_hash
                or self.after_contract.target_status != "authoritative"
                or self.after_contract.unresolved_fields
            ):
                raise ValueError("V3.8 clarification contract lineage differs")
        else:
            if self.clarification_evidence is not None or self.after_contract is not None:
                raise ValueError("V3.8 non-applicable action cannot contain outputs")
            if self.budget_after != 1:
                raise ValueError("V3.8 non-applicable action cannot consume budget")
        if self.action_hash and self.action_hash != self.content_hash():
            raise ValueError("action_hash does not match V3.8 clarification action")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "action_hash")

    def assert_sealed(self) -> None:
        if not self.action_hash or self.action_hash != self.content_hash():
            raise ValueError("V3.8 clarification action is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ClarificationActionReceiptV38":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"action_hash"}),
            action_hash=draft.content_hash(),
        )


class TargetConditionedChallengeV38(StrictModel):
    schema_version: Literal["3.8"] = "3.8"
    challenge_id: Identifier
    case_id: Identifier
    family: FamilyV37
    base_challenge_hash: Sha256
    authoritative_contract_hash: Sha256
    decision_target: Literal["free_run_prediction", "controlled_response_prediction"]
    relevant_fold_indices: list[Annotated[int, Field(ge=0, le=2)]] = Field(
        min_length=1, max_length=2
    )
    relevant_fold_losses: list[
        Annotated[float, Field(ge=0, allow_inf_nan=False)]
    ] = Field(min_length=1, max_length=2)
    target_cv_mean: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    target_cv_standard_error: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    relevant_simulation_failure_count: Annotated[int, Field(ge=0, le=2)]
    normalized_rank_ratio: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    normalized_condition_number: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    basis_term_count: Annotated[int, Field(ge=1)]
    minimum_rank_ratio: Literal[0.95] = 0.95
    maximum_condition_number: Literal[100000000.0] = 100000000.0
    maximum_cv_prediction_loss: Literal[0.35] = 0.35
    eligible: bool
    challenge_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_challenge(self) -> "TargetConditionedChallengeV38":
        expected_indices = [0] if self.decision_target == "free_run_prediction" else [1, 2]
        if self.relevant_fold_indices != expected_indices:
            raise ValueError("V3.8 target fold indices disagree")
        if len(self.relevant_fold_losses) != len(expected_indices):
            raise ValueError("V3.8 target fold loss count differs")
        values = np.asarray(self.relevant_fold_losses, dtype=float)
        mean = float(np.mean(values))
        se = (
            float(np.std(values, ddof=1) / math.sqrt(len(values)))
            if len(values) > 1 else 0.0
        )
        if not math.isclose(self.target_cv_mean, mean, abs_tol=1e-12):
            raise ValueError("V3.8 target CV mean does not recompute")
        if not math.isclose(self.target_cv_standard_error, se, abs_tol=1e-12):
            raise ValueError("V3.8 target CV standard error does not recompute")
        failures = sum(value >= 9.999 for value in values)
        if failures != self.relevant_simulation_failure_count:
            raise ValueError("V3.8 relevant failure count does not recompute")
        expected_eligible = (
            failures == 0
            and self.normalized_rank_ratio >= self.minimum_rank_ratio
            and self.normalized_condition_number <= self.maximum_condition_number
            and mean <= self.maximum_cv_prediction_loss
        )
        if self.eligible != expected_eligible:
            raise ValueError("V3.8 target eligibility does not recompute")
        if self.challenge_hash and self.challenge_hash != self.content_hash():
            raise ValueError("challenge_hash does not match V3.8 target challenge")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "challenge_hash")

    def assert_sealed(self) -> None:
        if not self.challenge_hash or self.challenge_hash != self.content_hash():
            raise ValueError("V3.8 target challenge is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "TargetConditionedChallengeV38":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"challenge_hash"}),
            challenge_hash=draft.content_hash(),
        )


def _condition_challenges_v38(
    base_challenges: list[object],
    contract: ControlledDynamicsContractV31,
) -> list[TargetConditionedChallengeV38]:
    contract.assert_sealed()
    indices = [0] if contract.decision_target == "free_run_prediction" else [1, 2]
    results: list[TargetConditionedChallengeV38] = []
    for base in base_challenges:
        losses = [base.fold_prediction_losses[index] for index in indices]
        values = np.asarray(losses, dtype=float)
        mean = float(np.mean(values))
        se = (
            float(np.std(values, ddof=1) / math.sqrt(len(values)))
            if len(values) > 1 else 0.0
        )
        failures = sum(value >= 9.999 for value in values)
        eligible = (
            failures == 0
            and base.normalized_rank_ratio >= 0.95
            and base.normalized_condition_number <= 100000000.0
            and mean <= 0.35
        )
        results.append(TargetConditionedChallengeV38.seal(
            challenge_id=f"target_challenge_{base.case_id}_{base.family}",
            case_id=base.case_id,
            family=base.family,
            base_challenge_hash=base.challenge_hash,
            authoritative_contract_hash=contract.contract_hash,
            decision_target=contract.decision_target,
            relevant_fold_indices=indices,
            relevant_fold_losses=losses,
            target_cv_mean=mean,
            target_cv_standard_error=se,
            relevant_simulation_failure_count=failures,
            normalized_rank_ratio=base.normalized_rank_ratio,
            normalized_condition_number=base.normalized_condition_number,
            basis_term_count=base.basis_term_count,
            eligible=eligible,
        ))
    return results


class TargetConditionedDecisionV38(StrictModel):
    schema_version: Literal["3.8"] = "3.8"
    decision_id: Identifier
    case_id: Identifier
    authoritative_contract_hash: Sha256
    challenge_hashes: list[Sha256] = Field(min_length=3, max_length=3)
    selected_family: FamilyV37 | None
    best_family: FamilyV37 | None
    one_standard_error_threshold: Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ] | None
    decision: Literal["select", "needs_evidence"]
    decision_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> "TargetConditionedDecisionV38":
        if self.decision == "select" and self.selected_family is None:
            raise ValueError("V3.8 selection needs a family")
        if self.decision == "needs_evidence" and self.selected_family is not None:
            raise ValueError("V3.8 needs-evidence cannot select a family")
        if self.decision_hash and self.decision_hash != self.content_hash():
            raise ValueError("decision_hash does not match V3.8 decision")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "decision_hash")

    def assert_sealed(self) -> None:
        if not self.decision_hash or self.decision_hash != self.content_hash():
            raise ValueError("V3.8 target decision is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "TargetConditionedDecisionV38":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"decision_hash"}),
            decision_hash=draft.content_hash(),
        )


def _target_decision_v38(
    case_id: str,
    contract: ControlledDynamicsContractV31,
    challenges: list[TargetConditionedChallengeV38],
) -> TargetConditionedDecisionV38:
    eligible = [item for item in challenges if item.eligible]
    if not eligible:
        return TargetConditionedDecisionV38.seal(
            decision_id=f"target_decision_{case_id}",
            case_id=case_id,
            authoritative_contract_hash=contract.contract_hash,
            challenge_hashes=[item.challenge_hash for item in challenges],
            selected_family=None,
            best_family=None,
            one_standard_error_threshold=None,
            decision="needs_evidence",
        )
    best = min(eligible, key=lambda item: (item.target_cv_mean, item.basis_term_count))
    threshold = best.target_cv_mean + best.target_cv_standard_error
    within = [item for item in eligible if item.target_cv_mean <= threshold]
    selected = min(within, key=lambda item: (item.basis_term_count, item.target_cv_mean))
    return TargetConditionedDecisionV38.seal(
        decision_id=f"target_decision_{case_id}",
        case_id=case_id,
        authoritative_contract_hash=contract.contract_hash,
        challenge_hashes=[item.challenge_hash for item in challenges],
        selected_family=selected.family,
        best_family=best.family,
        one_standard_error_threshold=threshold,
        decision="select",
    )


class TargetClarificationCaseReceiptV38(StrictModel):
    schema_version: Literal["3.8"] = "3.8"
    receipt_id: Identifier
    case_id: Identifier
    source_model_receipt_hash: Sha256
    source_disposition_hash: Sha256
    action: ClarificationActionReceiptV38
    authoritative_target_state: TargetAwareApplicabilityStateV371 | None
    conditioned_challenges: list[TargetConditionedChallengeV38] = Field(max_length=3)
    conditioned_decision: TargetConditionedDecisionV38 | None
    selected_model: ControlledDriftModelV31 | None
    next_action: NextActionV38 | None
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "TargetClarificationCaseReceiptV38":
        self.action.assert_sealed()
        if self.action.status == "executed_synthetic":
            if (
                self.authoritative_target_state is None
                or len(self.conditioned_challenges) != 3
                or self.conditioned_decision is None
                or self.next_action is None
            ):
                raise ValueError("V3.8 executed receipt is incomplete")
            self.authoritative_target_state.assert_sealed()
            self.conditioned_decision.assert_sealed()
            for challenge in self.conditioned_challenges:
                challenge.assert_sealed()
            if self.conditioned_decision.challenge_hashes != [
                item.challenge_hash for item in self.conditioned_challenges
            ]:
                raise ValueError("V3.8 conditioned decision binding differs")
            if self.conditioned_decision.decision == "select":
                if self.selected_model is None:
                    raise ValueError("V3.8 selected target decision needs model")
                self.selected_model.assert_sealed()
            elif self.selected_model is not None:
                raise ValueError("V3.8 needs-evidence cannot contain model")
        elif any((
            self.authoritative_target_state is not None,
            bool(self.conditioned_challenges),
            self.conditioned_decision is not None,
            self.selected_model is not None,
            self.next_action is not None,
        )):
            raise ValueError("V3.8 non-applicable receipt contains action outputs")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match V3.8 case receipt")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("V3.8 case receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "TargetClarificationCaseReceiptV38":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


class TargetClarificationBundleV38(StrictModel):
    schema_version: Literal["3.8"] = "3.8"
    bundle_id: Identifier
    spec_hash: Sha256
    source_candidate_bundle_hash: Sha256
    source_disposition_bundle_hash: Sha256
    clarification_policy_hash: Sha256
    case_receipts: list[TargetClarificationCaseReceiptV38] = Field(min_length=64, max_length=64)
    created_at: datetime
    bundle_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bundle(self) -> "TargetClarificationBundleV38":
        _assert_timezone(self.created_at, "created_at")
        case_ids = [item.case_id for item in self.case_receipts]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("V3.8 case ids must be unique")
        for item in self.case_receipts:
            item.assert_sealed()
        if self.bundle_hash and self.bundle_hash != self.content_hash():
            raise ValueError("bundle_hash does not match V3.8 bundle")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "bundle_hash")

    def assert_sealed(self) -> None:
        if not self.bundle_hash or self.bundle_hash != self.content_hash():
            raise ValueError("V3.8 bundle is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "TargetClarificationBundleV38":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"bundle_hash"}),
            bundle_hash=draft.content_hash(),
        )


def execute_target_clarifications_v38(
    spec: TargetClarificationWorldPackSpecV38,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    candidate: ModelChallengeBundleV37,
    dispositions: ChallengeDispositionBundleV371,
    disposition_policy: ChallengeDispositionPolicyV371,
    clarification_policy: TargetClarificationPolicyV38,
    *,
    executed_at: datetime,
) -> TargetClarificationBundleV38:
    for artifact in (
        spec, private_pack, candidate, dispositions,
        disposition_policy, clarification_policy,
    ):
        artifact.assert_sealed()
    if clarification_policy.policy_hash != spec.clarification_policy_hash:
        raise ValueError("V3.8 clarification policy is not frozen in protocol")
    private_by_id = {item.public_case.case_id: item for item in private_pack.cases}
    candidate_by_id = {item.case_id: item for item in candidate.case_receipts}
    disposition_by_id = {item.case_id: item for item in dispositions.dispositions}
    receipts: list[TargetClarificationCaseReceiptV38] = []
    for case_id, private_case in private_by_id.items():
        source = candidate_by_id[case_id]
        disposition = disposition_by_id[case_id]
        before = private_case.public_case.initial_contract
        should_execute = disposition.proposed_action == "clarify_decision_target"
        if not should_execute:
            action = ClarificationActionReceiptV38.seal(
                action_id=f"clarification_{case_id}",
                case_id=case_id,
                clarification_policy_hash=clarification_policy.policy_hash,
                source_disposition_hash=disposition.disposition_hash,
                before_contract_hash=before.contract_hash,
                status="not_applicable",
                clarification_evidence=None,
                after_contract=None,
                budget_after=1,
                executed_at=executed_at,
            )
            receipts.append(TargetClarificationCaseReceiptV38.seal(
                receipt_id=f"clarification_receipt_{case_id}",
                case_id=case_id,
                source_model_receipt_hash=source.receipt_hash,
                source_disposition_hash=disposition.disposition_hash,
                action=action,
                authoritative_target_state=None,
                conditioned_challenges=[],
                conditioned_decision=None,
                selected_model=None,
                next_action=None,
            ))
            continue
        evidence = TargetClarificationEvidenceV31.seal(
            evidence_id=f"target_evidence_v38_{case_id}",
            case_id=case_id,
            decision_target=private_case.true_decision_target,
            source_ref=f"synthetic_value_owner:{case_id}",
            observed_at=executed_at,
        )
        after = ControlledDynamicsContractV31.seal(
            contract_id=f"contract_{case_id}_v2",
            case_id=case_id,
            version=2,
            decision_target=evidence.decision_target,
            target_status="authoritative",
            unresolved_fields=[],
            parent_contract_hash=before.contract_hash,
            triggering_evidence_hash=evidence.evidence_hash,
            frozen_at=executed_at,
        )
        action = ClarificationActionReceiptV38.seal(
            action_id=f"clarification_{case_id}",
            case_id=case_id,
            clarification_policy_hash=clarification_policy.policy_hash,
            source_disposition_hash=disposition.disposition_hash,
            before_contract_hash=before.contract_hash,
            status="executed_synthetic",
            clarification_evidence=evidence,
            after_contract=after,
            budget_after=0,
            executed_at=executed_at,
        )
        target_state = TargetAwareApplicabilityStateV371.seal(
            target_state_id=f"target_state_v38_{case_id}",
            case_id=case_id,
            base_applicability_state_hash=source.applicability_state.state_hash,
            public_contract_hash=after.contract_hash,
            decision_target=after.decision_target,
            target_status=after.target_status,
            unresolved_fields=after.unresolved_fields,
            target_authority_evidence_hash=evidence.evidence_hash,
        )
        conditioned = _condition_challenges_v38(source.challenges, after)
        decision = _target_decision_v38(case_id, after, conditioned)
        selected_model = None
        if decision.selected_family is not None:
            selected_model = _fit_family_v37(
                private_case,
                _shared_observations_v37(private_case, spec),
                decision.selected_family,
                spec,
                model_suffix="target_conditioned_v38",
            )
        nonlinear_gain = _nonlinear_residual_gain_v371(source.challenges)
        if decision.selected_family is None:
            next_action: NextActionV38 = "acquire_target_discriminating_evidence"
        elif (
            decision.selected_family != "cubic_sparse_ode"
            and nonlinear_gain >= disposition_policy.nonlinear_residual_gain_trigger
        ):
            next_action = "expand_non_nested_family"
        else:
            next_action = "proceed_private_validation"
        receipts.append(TargetClarificationCaseReceiptV38.seal(
            receipt_id=f"clarification_receipt_{case_id}",
            case_id=case_id,
            source_model_receipt_hash=source.receipt_hash,
            source_disposition_hash=disposition.disposition_hash,
            action=action,
            authoritative_target_state=target_state,
            conditioned_challenges=conditioned,
            conditioned_decision=decision,
            selected_model=selected_model,
            next_action=next_action,
        ))
    return TargetClarificationBundleV38.seal(
        bundle_id=f"clarifications_{spec.experiment_id}",
        spec_hash=spec.spec_hash,
        source_candidate_bundle_hash=candidate.bundle_hash,
        source_disposition_bundle_hash=dispositions.bundle_hash,
        clarification_policy_hash=clarification_policy.policy_hash,
        case_receipts=receipts,
        created_at=executed_at,
    )


class PrivateClarificationCaseResultV38(StrictModel):
    schema_version: Literal["3.8"] = "3.8"
    case_id: Identifier
    mechanism: MechanismV31
    action_expected: bool
    action_executed: bool
    target_correct_before: bool | None
    target_correct_after: bool | None
    selected_family_after: FamilyV37 | None
    next_action: NextActionV38 | None
    selected_model_target_loss: Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ] | None


class TargetClarificationEvolutionReportV38(StrictModel):
    schema_version: Literal["3.8"] = "3.8"
    evolution_id: Identifier
    spec_hash: Sha256
    source_disposition_evolution_hash: Sha256
    clarification_bundle_hash: Sha256
    case_results: list[PrivateClarificationCaseResultV38] = Field(min_length=64, max_length=64)
    expected_action_count: Annotated[int, Field(ge=1)]
    executed_action_count: Annotated[int, Field(ge=0)]
    action_precision: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    action_recall: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    target_accuracy_before: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    target_accuracy_after: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    next_action_counts: dict[Identifier, Annotated[int, Field(ge=0)]]
    selected_model_counts_by_target: dict[Identifier, Annotated[int, Field(ge=0)]]
    selected_model_mean_loss_by_target: dict[
        Identifier, Annotated[float, Field(ge=0, allow_inf_nan=False)]
    ]
    gates: dict[Identifier, bool]
    ready_for_composed_synthetic_loop: bool
    status: Literal[
        "target_clarification_ready_for_composed_synthetic_loop_v38",
        "target_clarification_failed_v38",
    ]
    task_router_permitted: Literal[False] = False
    overall_qualification_permitted: Literal[False] = False
    confirmation_permitted: Literal[False] = False
    real_world_authorization_permitted: Literal[False] = False
    created_at: datetime
    evolution_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "TargetClarificationEvolutionReportV38":
        _assert_timezone(self.created_at, "created_at")
        ready = all(self.gates.values())
        if self.ready_for_composed_synthetic_loop != ready:
            raise ValueError("V3.8 readiness disagrees with gates")
        expected = (
            "target_clarification_ready_for_composed_synthetic_loop_v38"
            if ready else "target_clarification_failed_v38"
        )
        if self.status != expected:
            raise ValueError("V3.8 status disagrees with gates")
        if self.evolution_hash and self.evolution_hash != self.content_hash():
            raise ValueError("evolution_hash does not match V3.8 report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evolution_hash")

    def assert_sealed(self) -> None:
        if not self.evolution_hash or self.evolution_hash != self.content_hash():
            raise ValueError("V3.8 report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "TargetClarificationEvolutionReportV38":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evolution_hash"}),
            evolution_hash=draft.content_hash(),
        )


def evaluate_target_clarifications_v38(
    spec: TargetClarificationWorldPackSpecV38,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    candidate: ModelChallengeBundleV37,
    dispositions: ChallengeDispositionBundleV371,
    clarification_bundle: TargetClarificationBundleV38,
    *,
    evaluated_at: datetime,
) -> TargetClarificationEvolutionReportV38:
    for artifact in (
        spec, private_pack, candidate, dispositions, clarification_bundle
    ):
        artifact.assert_sealed()
    private_by_id = {item.public_case.case_id: item for item in private_pack.cases}
    candidate_by_id = {item.case_id: item for item in candidate.case_receipts}
    disposition_by_id = {item.case_id: item for item in dispositions.dispositions}
    clarification_by_id = {
        item.case_id: item for item in clarification_bundle.case_receipts
    }
    if not (
        set(private_by_id) == set(candidate_by_id)
        == set(disposition_by_id) == set(clarification_by_id)
    ):
        raise ValueError("V3.8 case coverage differs")
    results: list[PrivateClarificationCaseResultV38] = []
    expected_count = 0
    executed_count = 0
    true_positive = 0
    before_correct = 0
    after_correct = 0
    next_counts: dict[str, int] = defaultdict(int)
    selected_counts: dict[str, int] = defaultdict(int)
    selected_losses: dict[str, list[float]] = defaultdict(list)
    lineage_pass = True
    conditioned_pass = True
    for case_id, private_case in private_by_id.items():
        source = candidate_by_id[case_id]
        disposition = disposition_by_id[case_id]
        receipt = clarification_by_id[case_id]
        expected = disposition.proposed_action == "clarify_decision_target"
        executed = receipt.action.status == "executed_synthetic"
        expected_count += int(expected)
        executed_count += int(executed)
        true_positive += int(expected and executed)
        if not expected:
            results.append(PrivateClarificationCaseResultV38(
                case_id=case_id,
                mechanism=private_case.mechanism,
                action_expected=False,
                action_executed=executed,
                target_correct_before=None,
                target_correct_after=None,
                selected_family_after=None,
                next_action=None,
                selected_model_target_loss=None,
            ))
            continue
        before = private_case.public_case.initial_contract
        target_before = before.decision_target == private_case.true_decision_target
        after = receipt.action.after_contract
        target_after = after.decision_target == private_case.true_decision_target
        before_correct += int(target_before)
        after_correct += int(target_after)
        if (
            receipt.action.before_contract_hash != before.contract_hash
            or after.parent_contract_hash != before.contract_hash
            or after.triggering_evidence_hash
            != receipt.action.clarification_evidence.evidence_hash
            or receipt.action.clarification_evidence.decision_target
            != private_case.true_decision_target
        ):
            lineage_pass = False
        recomputed_challenges = _condition_challenges_v38(source.challenges, after)
        recomputed_decision = _target_decision_v38(
            case_id, after, recomputed_challenges
        )
        if (
            [item.challenge_hash for item in recomputed_challenges]
            != [item.challenge_hash for item in receipt.conditioned_challenges]
            or recomputed_decision.decision_hash
            != receipt.conditioned_decision.decision_hash
        ):
            conditioned_pass = False
        loss = None
        selected_family = receipt.conditioned_decision.selected_family
        if receipt.selected_model is not None:
            loss = _target_loss_v31(private_case, receipt.selected_model, spec)
            selected_counts[after.decision_target] += 1
            selected_losses[after.decision_target].append(loss)
        next_counts[receipt.next_action] += 1
        results.append(PrivateClarificationCaseResultV38(
            case_id=case_id,
            mechanism=private_case.mechanism,
            action_expected=True,
            action_executed=executed,
            target_correct_before=target_before,
            target_correct_after=target_after,
            selected_family_after=selected_family,
            next_action=receipt.next_action,
            selected_model_target_loss=loss,
        ))
    precision = true_positive / max(executed_count, 1)
    recall = true_positive / expected_count
    accuracy_before = before_correct / expected_count
    accuracy_after = after_correct / expected_count
    mean_losses = {
        target: float(np.mean(values))
        for target, values in selected_losses.items()
    }
    gates = {
        "action_precision": math.isclose(precision, 1.0, abs_tol=1e-12),
        "action_recall": math.isclose(recall, 1.0, abs_tol=1e-12),
        "target_accuracy_after": math.isclose(accuracy_after, 1.0, abs_tol=1e-12),
        "target_accuracy_strict_improvement": accuracy_after > accuracy_before,
        "contract_lineage_and_evidence": lineage_pass,
        "target_conditioned_challenge_recomputed": conditioned_pass,
        "needs_evidence_remains_visible": (
            next_counts.get("acquire_target_discriminating_evidence", 0) > 0
        ),
        "no_real_world_execution": all(
            not item.action.real_world_execution_permitted
            for item in clarification_bundle.case_receipts
        ),
    }
    ready = all(gates.values())
    return TargetClarificationEvolutionReportV38.seal(
        evolution_id=f"evolution_{spec.experiment_id}",
        spec_hash=spec.spec_hash,
        source_disposition_evolution_hash=spec.source_disposition_evolution_hash,
        clarification_bundle_hash=clarification_bundle.bundle_hash,
        case_results=results,
        expected_action_count=expected_count,
        executed_action_count=executed_count,
        action_precision=precision,
        action_recall=recall,
        target_accuracy_before=accuracy_before,
        target_accuracy_after=accuracy_after,
        next_action_counts=dict(next_counts),
        selected_model_counts_by_target=dict(selected_counts),
        selected_model_mean_loss_by_target=mean_losses,
        gates=gates,
        ready_for_composed_synthetic_loop=ready,
        status=(
            "target_clarification_ready_for_composed_synthetic_loop_v38"
            if ready else "target_clarification_failed_v38"
        ),
        created_at=evaluated_at,
    )


class TargetClarificationManifestV38(StrictModel):
    schema_version: Literal["3.8"] = "3.8"
    run_id: Identifier
    artifact_refs: list[ArtifactRef] = Field(min_length=13, max_length=13)
    terminal_status: Literal[
        "target_clarification_ready_for_composed_synthetic_loop_v38",
        "target_clarification_failed_v38",
    ]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "TargetClarificationManifestV38":
        _assert_timezone(self.created_at, "created_at")
        if len({item.kind for item in self.artifact_refs}) != len(self.artifact_refs):
            raise ValueError("V3.8 manifest artifact kinds must be unique")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match V3.8 manifest")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("V3.8 manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "TargetClarificationManifestV38":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class TargetClarificationOutcomeV38:
    store: RunStore
    method_evidence: ModelChallengeMethodEvidenceV37
    spec: TargetClarificationWorldPackSpecV38
    baseline_policy: ModelPortfolioPolicyV37
    candidate_policy: ModelPortfolioPolicyV37
    disposition_policy: ChallengeDispositionPolicyV371
    clarification_policy: TargetClarificationPolicyV38
    private_pack: PrivateControlledDynamicsWorldPackV31
    baseline_bundle: ModelChallengeBundleV37
    candidate_bundle: ModelChallengeBundleV37
    v37_report: ModelChallengeEvolutionReportV37
    disposition_bundle: ChallengeDispositionBundleV371
    clarification_bundle: TargetClarificationBundleV38
    evolution_report: TargetClarificationEvolutionReportV38
    manifest: TargetClarificationManifestV38


def run_target_clarification_worldpack_v38(
    output_root: str | Path,
    *,
    source_v371_run_directory: str | Path,
    method_evidence: ModelChallengeMethodEvidenceV37,
    spec: TargetClarificationWorldPackSpecV38,
    baseline_policy: ModelPortfolioPolicyV37,
    candidate_policy: ModelPortfolioPolicyV37,
    disposition_policy: ChallengeDispositionPolicyV371,
    clarification_policy: TargetClarificationPolicyV38,
    evaluated_at: datetime | None = None,
    run_id: str | None = None,
) -> TargetClarificationOutcomeV38:
    source_bundle, source_report = _load_source_v371(source_v371_run_directory)
    for artifact in (
        method_evidence, spec, baseline_policy, candidate_policy,
        disposition_policy, clarification_policy,
    ):
        artifact.assert_sealed()
    if (
        clarification_policy.source_disposition_bundle_hash != source_bundle.bundle_hash
        or clarification_policy.source_disposition_evolution_hash != source_report.evolution_hash
    ):
        raise ValueError("V3.8 clarification policy source binding differs")
    if (
        spec.method_evidence_hash != method_evidence.evidence_hash
        or spec.baseline_policy_hash != baseline_policy.policy_hash
        or spec.candidate_policy_hash != candidate_policy.policy_hash
        or spec.disposition_policy_hash != disposition_policy.policy_hash
        or spec.clarification_policy_hash != clarification_policy.policy_hash
    ):
        raise ValueError("V3.8 frozen artifact binding differs")
    at = evaluated_at or datetime.now(timezone.utc)
    store = RunStore(
        output_root,
        run_id=run_id or f"target-clarification-v38-{uuid4().hex[:10]}",
    )
    refs = [
        store.put_artifact("model_challenge_method_evidence_v38", method_evidence),
        store.put_artifact("model_challenge_baseline_policy_v38", baseline_policy),
        store.put_artifact("model_challenge_candidate_policy_v38", candidate_policy),
        store.put_artifact("challenge_disposition_policy_v38", disposition_policy),
        store.put_artifact("target_clarification_policy_v38", clarification_policy),
        store.put_artifact("target_clarification_spec_v38", spec),
    ]
    store.emit("target_clarification_v38_protocol_frozen_before_private_pack", {
        "spec_hash": spec.spec_hash,
        "source_disposition_evolution_hash": source_report.evolution_hash,
        "clarification_policy_hash": clarification_policy.policy_hash,
        "source_run_independently_verified": True,
    })
    private_pack = generate_private_controlled_dynamics_worldpack_v31(
        spec, generated_at=at
    )
    baseline = execute_model_challenge_policy_v37(
        spec, private_pack, baseline_policy, executed_at=at
    )
    candidate = execute_model_challenge_policy_v37(
        spec, private_pack, candidate_policy, executed_at=at
    )
    v37_report = evaluate_model_challenge_worldpack_v37(
        spec, private_pack, baseline, candidate, evaluated_at=at
    )
    dispositions = create_challenge_disposition_bundle_v371(
        spec, private_pack, candidate, disposition_policy, created_at=at
    )
    clarification_bundle = execute_target_clarifications_v38(
        spec,
        private_pack,
        candidate,
        dispositions,
        disposition_policy,
        clarification_policy,
        executed_at=at,
    )
    evolution = evaluate_target_clarifications_v38(
        spec,
        private_pack,
        candidate,
        dispositions,
        clarification_bundle,
        evaluated_at=at,
    )
    refs.extend([
        store.put_artifact("private_target_clarification_worldpack_v38", private_pack),
        store.put_artifact("model_challenge_baseline_bundle_v38", baseline),
        store.put_artifact("model_challenge_candidate_bundle_v38", candidate),
        store.put_artifact("model_challenge_report_v38", v37_report),
        store.put_artifact("challenge_disposition_bundle_v38", dispositions),
        store.put_artifact("target_clarification_bundle_v38", clarification_bundle),
        store.put_artifact("target_clarification_evolution_report_v38", evolution),
    ])
    manifest = TargetClarificationManifestV38.seal(
        run_id=store.run_id,
        artifact_refs=refs,
        terminal_status=evolution.status,
        created_at=at,
    )
    manifest_ref = store.put_artifact("target_clarification_manifest_v38", manifest)
    store.emit("target_clarification_v38_worldpack_adjudicated", {
        "manifest_ref": manifest_ref.model_dump(mode="json")
    })
    if not verify_target_clarification_run_v38(store.run_directory):
        raise RuntimeError("V3.8 run failed independent verification")
    return TargetClarificationOutcomeV38(
        store,
        method_evidence,
        spec,
        baseline_policy,
        candidate_policy,
        disposition_policy,
        clarification_policy,
        private_pack,
        baseline,
        candidate,
        v37_report,
        dispositions,
        clarification_bundle,
        evolution,
        manifest,
    )


def verify_target_clarification_run_v38(run_directory: str | Path) -> bool:
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
            item for item in committed if item.kind == "target_clarification_manifest_v38"
        ]
        if len(manifest_refs) != 1:
            return False
        manifest = TargetClarificationManifestV38.model_validate(
            store.load_artifact(manifest_refs[0])
        )
        manifest.assert_sealed()

        def load_one(kind: str, model):
            references = [item for item in manifest.artifact_refs if item.kind == kind]
            if len(references) != 1:
                raise RuntimeError(f"V3.8 manifest needs exactly one {kind}")
            return model.model_validate(store.load_artifact(references[0]))

        method = load_one(
            "model_challenge_method_evidence_v38", ModelChallengeMethodEvidenceV37
        )
        baseline_policy = load_one(
            "model_challenge_baseline_policy_v38", ModelPortfolioPolicyV37
        )
        candidate_policy = load_one(
            "model_challenge_candidate_policy_v38", ModelPortfolioPolicyV37
        )
        disposition_policy = load_one(
            "challenge_disposition_policy_v38", ChallengeDispositionPolicyV371
        )
        clarification_policy = load_one(
            "target_clarification_policy_v38", TargetClarificationPolicyV38
        )
        spec = load_one("target_clarification_spec_v38", TargetClarificationWorldPackSpecV38)
        private_pack = load_one(
            "private_target_clarification_worldpack_v38",
            PrivateControlledDynamicsWorldPackV31,
        )
        baseline = load_one(
            "model_challenge_baseline_bundle_v38", ModelChallengeBundleV37
        )
        candidate = load_one(
            "model_challenge_candidate_bundle_v38", ModelChallengeBundleV37
        )
        v37_report = load_one(
            "model_challenge_report_v38", ModelChallengeEvolutionReportV37
        )
        dispositions = load_one(
            "challenge_disposition_bundle_v38", ChallengeDispositionBundleV371
        )
        clarification_bundle = load_one(
            "target_clarification_bundle_v38", TargetClarificationBundleV38
        )
        evolution = load_one(
            "target_clarification_evolution_report_v38",
            TargetClarificationEvolutionReportV38,
        )
        for artifact in (
            method, baseline_policy, candidate_policy, disposition_policy,
            clarification_policy, spec, private_pack, baseline, candidate,
            v37_report, dispositions, clarification_bundle, evolution, manifest,
        ):
            artifact.assert_sealed()
        if manifest.run_id != store.run_id:
            return False
        if (
            spec.method_evidence_hash != method.evidence_hash
            or spec.baseline_policy_hash != baseline_policy.policy_hash
            or spec.candidate_policy_hash != candidate_policy.policy_hash
            or spec.disposition_policy_hash != disposition_policy.policy_hash
            or spec.clarification_policy_hash != clarification_policy.policy_hash
        ):
            return False
        regenerated = generate_private_controlled_dynamics_worldpack_v31(
            spec, generated_at=private_pack.generated_at
        )
        if regenerated.pack_hash != private_pack.pack_hash:
            return False
        executed_at = baseline.case_receipts[0].executed_at
        replay_baseline = execute_model_challenge_policy_v37(
            spec, private_pack, baseline_policy, executed_at=executed_at
        )
        replay_candidate = execute_model_challenge_policy_v37(
            spec, private_pack, candidate_policy, executed_at=executed_at
        )
        if (
            replay_baseline.bundle_hash != baseline.bundle_hash
            or replay_candidate.bundle_hash != candidate.bundle_hash
        ):
            return False
        replay_v37 = evaluate_model_challenge_worldpack_v37(
            spec,
            private_pack,
            baseline,
            candidate,
            evaluated_at=v37_report.created_at,
        )
        if replay_v37.evolution_hash != v37_report.evolution_hash:
            return False
        replay_dispositions = create_challenge_disposition_bundle_v371(
            spec,
            private_pack,
            candidate,
            disposition_policy,
            created_at=dispositions.created_at,
        )
        if replay_dispositions.bundle_hash != dispositions.bundle_hash:
            return False
        replay_clarifications = execute_target_clarifications_v38(
            spec,
            private_pack,
            candidate,
            dispositions,
            disposition_policy,
            clarification_policy,
            executed_at=clarification_bundle.created_at,
        )
        if replay_clarifications.bundle_hash != clarification_bundle.bundle_hash:
            return False
        recomputed = evaluate_target_clarifications_v38(
            spec,
            private_pack,
            candidate,
            dispositions,
            clarification_bundle,
            evaluated_at=evolution.created_at,
        )
        if recomputed.evolution_hash != evolution.evolution_hash:
            return False
        if any(
            any(word in item.kind for word in ("qualification", "confirmation", "task_router"))
            for item in manifest.artifact_refs
        ):
            return False
        freezes = [
            event for event in events
            if event["event_type"]
            == "target_clarification_v38_protocol_frozen_before_private_pack"
        ]
        private_events = [
            event for event in events
            if event["event_type"] == "artifact_committed"
            and event["payload"]["kind"]
            == "private_target_clarification_worldpack_v38"
        ]
        return (
            len(freezes) == 1
            and len(private_events) == 1
            and freezes[0]["sequence"] < private_events[0]["sequence"]
            and store.verify_event_chain()
        )
    except (
        OSError, RuntimeError, ValueError, KeyError, TypeError,
        json.JSONDecodeError, FloatingPointError, AttributeError,
    ):
        return False
