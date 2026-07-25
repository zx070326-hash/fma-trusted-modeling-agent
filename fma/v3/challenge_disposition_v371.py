from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import Field, model_validator

from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore
from fma.v2.schemas import Identifier, Sha256, _assert_timezone

from .controlled_dynamics_loop import (
    MECHANISMS_V31,
    MechanismV31,
    PrivateControlledDynamicsCaseV31,
    PrivateControlledDynamicsWorldPackV31,
    generate_private_controlled_dynamics_worldpack_v31,
)
from .model_challenge_v37 import (
    FAMILIES_V37,
    FamilyV37,
    ModelChallengeBundleV37,
    ModelChallengeEvolutionReportV37,
    ModelChallengeMethodEvidenceV37,
    ModelChallengeWorldPackSpecV37,
    ModelPortfolioPolicyV37,
    _hash_without,
    evaluate_model_challenge_worldpack_v37,
    execute_model_challenge_policy_v37,
    verify_model_challenge_run_v37,
)


EXPLORATORY_SEEDS_V371 = (
    19001, 19051, 19121, 19163, 19219, 19267, 19319, 19373,
    19423, 19471, 19531, 19577, 19603, 19661, 19717, 19763,
)

DispositionLayerV371 = Literal[
    "problem_layer",
    "data_layer",
    "model_layer",
    "evaluation_layer",
]
DispositionActionV371 = Literal[
    "repair_data_quality",
    "clarify_decision_target",
    "acquire_target_discriminating_evidence",
    "expand_non_nested_family",
    "proceed_private_validation",
]


class ChallengeDispositionPolicyV371(StrictModel):
    schema_version: Literal["3.7.1"] = "3.7.1"
    policy_id: Identifier
    source_failure_evolution_hash: Sha256
    source_failure_candidate_bundle_hash: Sha256
    source_failure_status: Literal["model_challenge_failed_v37"]
    training_nonlinear_residual_gain_by_mechanism: dict[
        Identifier, Annotated[float, Field(allow_inf_nan=False)]
    ]
    nonlinear_residual_gain_trigger: Literal[0.15] = 0.15
    priority_rule: Literal[
        "quality_then_target_then_eligibility_then_nonlinearity_then_private_validation"
    ] = "quality_then_target_then_eligibility_then_nonlinearity_then_private_validation"
    threshold_claim: Literal[
        "empirical_failure_signature_not_probability_or_identifiability_guarantee"
    ] = "empirical_failure_signature_not_probability_or_identifiability_guarantee"
    execution_permitted: Literal[False] = False
    task_router_permitted: Literal[False] = False
    real_world_action_permitted: Literal[False] = False
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "ChallengeDispositionPolicyV371":
        if set(self.training_nonlinear_residual_gain_by_mechanism) != set(MECHANISMS_V31):
            raise ValueError("V3.7.1 policy needs all source mechanisms")
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("policy_hash does not match V3.7.1 disposition policy")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("V3.7.1 disposition policy is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ChallengeDispositionPolicyV371":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"policy_hash"}),
            policy_hash=draft.content_hash(),
        )


def _load_source_v37(
    run_directory: str | Path,
) -> tuple[ModelChallengeBundleV37, ModelChallengeEvolutionReportV37]:
    if not verify_model_challenge_run_v37(run_directory):
        raise ValueError("V3.7.1 source V3.7 run did not independently verify")
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
            raise ValueError(f"V3.7.1 source run needs exactly one {kind}")
        return model.model_validate(store.load_artifact(matches[0]))

    return (
        load("model_challenge_candidate_bundle_v37", ModelChallengeBundleV37),
        load("model_challenge_evolution_report_v37", ModelChallengeEvolutionReportV37),
    )


def _nonlinear_residual_gain_v371(challenges: list[object]) -> float:
    by_family = {item.family: item for item in challenges}
    linear = by_family["linear_state_space"].normalized_derivative_residual
    nonlinear = min(
        by_family["quadratic_interaction_ode"].normalized_derivative_residual,
        by_family["cubic_sparse_ode"].normalized_derivative_residual,
    )
    return float((linear - nonlinear) / max(linear, 1e-12))


def build_challenge_disposition_policy_v371(
    source_v37_run_directory: str | Path,
) -> ChallengeDispositionPolicyV371:
    candidate, report = _load_source_v37(source_v37_run_directory)
    if report.status != "model_challenge_failed_v37":
        raise ValueError("V3.7.1 policy must be derived from a failed V3.7 run")
    mechanism_by_case = {item.case_id: item.mechanism for item in report.case_results}
    gains: dict[MechanismV31, list[float]] = defaultdict(list)
    for receipt in candidate.case_receipts:
        if receipt.applicability_state.quality_flags:
            continue
        gains[mechanism_by_case[receipt.case_id]].append(
            _nonlinear_residual_gain_v371(receipt.challenges)
        )
    return ChallengeDispositionPolicyV371.seal(
        policy_id="challenge_disposition_policy_v371",
        source_failure_evolution_hash=report.evolution_hash,
        source_failure_candidate_bundle_hash=candidate.bundle_hash,
        source_failure_status=report.status,
        training_nonlinear_residual_gain_by_mechanism={
            mechanism: sum(values) / len(values)
            for mechanism, values in gains.items()
        },
    )


class ModelChallengeWorldPackSpecV371(ModelChallengeWorldPackSpecV37):
    schema_version: Literal["3.7.1"] = "3.7.1"
    seeds: list[int] = Field(min_length=16, max_length=16)
    bootstrap_seed: Literal[371722] = 371722
    disposition_policy_hash: Sha256
    source_failure_evolution_hash: Sha256
    source_failure_candidate_bundle_hash: Sha256
    frozen_delta: Literal[
        "challenge_disposition_controller_only"
    ] = "challenge_disposition_controller_only"

    @model_validator(mode="after")
    def validate_spec(self) -> "ModelChallengeWorldPackSpecV371":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.mechanisms != list(MECHANISMS_V31):
            raise ValueError("V3.7.1 requires the frozen mechanism order")
        if self.seeds != list(EXPLORATORY_SEEDS_V371):
            raise ValueError("V3.7.1 seeds do not match the frozen set")
        if self.family_catalog != list(FAMILIES_V37):
            raise ValueError("V3.7.1 cannot change the V3.7 family catalog")
        if self.observation_action_indices != [0, 7]:
            raise ValueError("V3.7.1 cannot change V3.7 observations")
        if not math.isclose(
            (self.trajectory_points - 1) * self.time_step,
            self.segment_count * self.segment_duration,
            abs_tol=1e-12,
        ):
            raise ValueError("V3.7.1 segments do not cover the trajectory")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match V3.7.1 protocol")
        return self


def default_challenge_disposition_exploratory_spec_v371(
    *,
    method_evidence_hash: str,
    baseline_policy_hash: str,
    candidate_policy_hash: str,
    disposition_policy: ChallengeDispositionPolicyV371,
    frozen_at: datetime | None = None,
) -> ModelChallengeWorldPackSpecV371:
    disposition_policy.assert_sealed()
    return ModelChallengeWorldPackSpecV371.seal(
        experiment_id="challenge_disposition_exploratory_v371",
        mechanisms=list(MECHANISMS_V31),
        seeds=list(EXPLORATORY_SEEDS_V371),
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
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


class ChallengeDispositionV371(StrictModel):
    schema_version: Literal["3.7.1"] = "3.7.1"
    disposition_id: Identifier
    case_id: Identifier
    policy_hash: Sha256
    source_failure_evolution_hash: Sha256
    applicability_state_hash: Sha256
    target_state_hash: Sha256
    challenge_hashes: list[Sha256] = Field(min_length=3, max_length=3)
    model_decision_hash: Sha256
    observed_quality_flags: list[Identifier]
    observed_decision_target: Literal[
        "free_run_prediction", "controlled_response_prediction", "unspecified"
    ]
    observed_target_status: Literal["default_unverified", "authoritative"]
    observed_unresolved_fields: list[Literal["decision_target"]] = Field(max_length=1)
    observed_selected_family: FamilyV37 | None
    observed_eligible_family_count: Annotated[int, Field(ge=0, le=3)]
    nonlinear_residual_gain: Annotated[float, Field(allow_inf_nan=False)]
    issue_layer: DispositionLayerV371
    proposed_action: DispositionActionV371
    reason_code: Literal[
        "quality_failure_precedes_modeling",
        "decision_target_unresolved",
        "no_family_survived_predictive_challenge",
        "nonlinear_evidence_with_unstable_or_rejected_nested_family",
        "bounded_candidate_requires_private_validation",
    ]
    execution_permitted: Literal[False] = False
    private_mechanism_seen: Literal[False] = False
    private_probe_seen: Literal[False] = False
    private_target_loss_seen: Literal[False] = False
    disposition_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_disposition(self) -> "ChallengeDispositionV371":
        expected_layer = {
            "repair_data_quality": "data_layer",
            "clarify_decision_target": "problem_layer",
            "acquire_target_discriminating_evidence": "data_layer",
            "expand_non_nested_family": "model_layer",
            "proceed_private_validation": "evaluation_layer",
        }[self.proposed_action]
        if self.issue_layer != expected_layer:
            raise ValueError("V3.7.1 disposition layer disagrees with action")
        if self.disposition_hash and self.disposition_hash != self.content_hash():
            raise ValueError("disposition_hash does not match V3.7.1 disposition")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "disposition_hash")

    def assert_sealed(self) -> None:
        if not self.disposition_hash or self.disposition_hash != self.content_hash():
            raise ValueError("V3.7.1 disposition is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ChallengeDispositionV371":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"disposition_hash"}),
            disposition_hash=draft.content_hash(),
        )


class TargetAwareApplicabilityStateV371(StrictModel):
    schema_version: Literal["3.7.1"] = "3.7.1"
    target_state_id: Identifier
    case_id: Identifier
    base_applicability_state_hash: Sha256
    public_contract_hash: Sha256
    decision_target: Literal[
        "free_run_prediction", "controlled_response_prediction", "unspecified"
    ]
    target_status: Literal["default_unverified", "authoritative"]
    unresolved_fields: list[Literal["decision_target"]] = Field(max_length=1)
    target_authority_evidence_hash: Sha256 | None
    private_mechanism_seen: Literal[False] = False
    private_probe_seen: Literal[False] = False
    private_target_loss_seen: Literal[False] = False
    target_state_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_target_state(self) -> "TargetAwareApplicabilityStateV371":
        if self.target_status == "default_unverified":
            if self.unresolved_fields != ["decision_target"]:
                raise ValueError("V3.7.1 unverified target must remain unresolved")
            if self.target_authority_evidence_hash is not None:
                raise ValueError("V3.7.1 unverified target cannot claim authority evidence")
        elif self.unresolved_fields:
            raise ValueError("V3.7.1 authoritative target cannot remain unresolved")
        if self.target_state_hash and self.target_state_hash != self.content_hash():
            raise ValueError("target_state_hash does not match V3.7.1 target state")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "target_state_hash")

    def assert_sealed(self) -> None:
        if not self.target_state_hash or self.target_state_hash != self.content_hash():
            raise ValueError("V3.7.1 target-aware state is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "TargetAwareApplicabilityStateV371":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"target_state_hash"}),
            target_state_hash=draft.content_hash(),
        )


def _target_state_v371(
    private_case: PrivateControlledDynamicsCaseV31,
    candidate_receipt: object,
) -> TargetAwareApplicabilityStateV371:
    contract = private_case.public_case.initial_contract
    contract.assert_sealed()
    return TargetAwareApplicabilityStateV371.seal(
        target_state_id=f"target_state_{candidate_receipt.case_id}",
        case_id=candidate_receipt.case_id,
        base_applicability_state_hash=candidate_receipt.applicability_state.state_hash,
        public_contract_hash=contract.contract_hash,
        decision_target=contract.decision_target,
        target_status=contract.target_status,
        unresolved_fields=contract.unresolved_fields,
        target_authority_evidence_hash=contract.triggering_evidence_hash,
    )


def _disposition_v371(
    receipt: object,
    target_state: TargetAwareApplicabilityStateV371,
    policy: ChallengeDispositionPolicyV371,
) -> ChallengeDispositionV371:
    state = receipt.applicability_state
    decision = receipt.decision
    challenges = receipt.challenges
    gain = _nonlinear_residual_gain_v371(challenges)
    common = {
        "disposition_id": f"disposition_{receipt.case_id}",
        "case_id": receipt.case_id,
        "policy_hash": policy.policy_hash,
        "source_failure_evolution_hash": policy.source_failure_evolution_hash,
        "applicability_state_hash": state.state_hash,
        "target_state_hash": target_state.target_state_hash,
        "challenge_hashes": [item.challenge_hash for item in challenges],
        "model_decision_hash": decision.decision_hash,
        "observed_quality_flags": state.quality_flags,
        "observed_decision_target": state.decision_target,
        "observed_target_status": target_state.target_status,
        "observed_unresolved_fields": target_state.unresolved_fields,
        "observed_selected_family": decision.selected_family,
        "observed_eligible_family_count": sum(item.eligible for item in challenges),
        "nonlinear_residual_gain": gain,
    }
    if state.quality_flags:
        return ChallengeDispositionV371.seal(
            issue_layer="data_layer",
            proposed_action="repair_data_quality",
            reason_code="quality_failure_precedes_modeling",
            **common,
        )
    if (
        target_state.target_status == "default_unverified"
        or target_state.unresolved_fields
    ):
        return ChallengeDispositionV371.seal(
            issue_layer="problem_layer",
            proposed_action="clarify_decision_target",
            reason_code="decision_target_unresolved",
            **common,
        )
    if decision.selected_family is None:
        return ChallengeDispositionV371.seal(
            issue_layer="data_layer",
            proposed_action="acquire_target_discriminating_evidence",
            reason_code="no_family_survived_predictive_challenge",
            **common,
        )
    if (
        decision.selected_family != "cubic_sparse_ode"
        and gain >= policy.nonlinear_residual_gain_trigger
    ):
        return ChallengeDispositionV371.seal(
            issue_layer="model_layer",
            proposed_action="expand_non_nested_family",
            reason_code="nonlinear_evidence_with_unstable_or_rejected_nested_family",
            **common,
        )
    return ChallengeDispositionV371.seal(
        issue_layer="evaluation_layer",
        proposed_action="proceed_private_validation",
        reason_code="bounded_candidate_requires_private_validation",
        **common,
    )


class ChallengeDispositionBundleV371(StrictModel):
    schema_version: Literal["3.7.1"] = "3.7.1"
    bundle_id: Identifier
    spec_hash: Sha256
    source_candidate_bundle_hash: Sha256
    disposition_policy_hash: Sha256
    dispositions: list[ChallengeDispositionV371] = Field(min_length=64, max_length=64)
    created_at: datetime
    bundle_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bundle(self) -> "ChallengeDispositionBundleV371":
        _assert_timezone(self.created_at, "created_at")
        case_ids = [item.case_id for item in self.dispositions]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("V3.7.1 disposition case ids must be unique")
        for item in self.dispositions:
            item.assert_sealed()
            if item.policy_hash != self.disposition_policy_hash:
                raise ValueError("V3.7.1 disposition policy binding differs")
        if self.bundle_hash and self.bundle_hash != self.content_hash():
            raise ValueError("bundle_hash does not match V3.7.1 disposition bundle")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "bundle_hash")

    def assert_sealed(self) -> None:
        if not self.bundle_hash or self.bundle_hash != self.content_hash():
            raise ValueError("V3.7.1 disposition bundle is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ChallengeDispositionBundleV371":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"bundle_hash"}),
            bundle_hash=draft.content_hash(),
        )


def create_challenge_disposition_bundle_v371(
    spec: ModelChallengeWorldPackSpecV371,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    candidate: ModelChallengeBundleV37,
    policy: ChallengeDispositionPolicyV371,
    *,
    created_at: datetime,
) -> ChallengeDispositionBundleV371:
    for artifact in (spec, private_pack, candidate, policy):
        artifact.assert_sealed()
    if candidate.spec_hash != spec.spec_hash:
        raise ValueError("V3.7.1 candidate bundle belongs to another protocol")
    if policy.policy_hash != spec.disposition_policy_hash:
        raise ValueError("V3.7.1 disposition policy is not frozen in protocol")
    private_by_id = {
        item.public_case.case_id: item for item in private_pack.cases
    }
    if set(private_by_id) != {item.case_id for item in candidate.case_receipts}:
        raise ValueError("V3.7.1 target-state case coverage differs")
    return ChallengeDispositionBundleV371.seal(
        bundle_id=f"dispositions_{spec.experiment_id}",
        spec_hash=spec.spec_hash,
        source_candidate_bundle_hash=candidate.bundle_hash,
        disposition_policy_hash=policy.policy_hash,
        dispositions=[
            _disposition_v371(
                receipt,
                _target_state_v371(private_by_id[receipt.case_id], receipt),
                policy,
            )
            for receipt in candidate.case_receipts
        ],
        created_at=created_at,
    )


def _expected_action_v371(
    private_case: PrivateControlledDynamicsCaseV31,
    candidate_receipt: object,
) -> DispositionActionV371:
    state = candidate_receipt.applicability_state
    decision = candidate_receipt.decision
    if state.quality_flags:
        return "repair_data_quality"
    contract = private_case.public_case.initial_contract
    if contract.target_status == "default_unverified" or contract.unresolved_fields:
        return "clarify_decision_target"
    if decision.selected_family is None:
        return "acquire_target_discriminating_evidence"
    if (
        private_case.mechanism == "duffing_oscillator"
        and decision.selected_family != "cubic_sparse_ode"
    ):
        return "expand_non_nested_family"
    return "proceed_private_validation"


class PrivateDispositionCaseResultV371(StrictModel):
    schema_version: Literal["3.7.1"] = "3.7.1"
    case_id: Identifier
    mechanism: MechanismV31
    proposed_action: DispositionActionV371
    expected_action: DispositionActionV371
    route_correct: bool
    false_private_validation: bool


class ChallengeDispositionEvolutionReportV371(StrictModel):
    schema_version: Literal["3.7.1"] = "3.7.1"
    evolution_id: Identifier
    spec_hash: Sha256
    source_v37_evolution_hash: Sha256
    disposition_bundle_hash: Sha256
    case_results: list[PrivateDispositionCaseResultV371] = Field(min_length=64, max_length=64)
    route_accuracy: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    action_counts: dict[Identifier, Annotated[int, Field(ge=0)]]
    per_action_accuracy: dict[
        Identifier, Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    ]
    false_private_validation_count: Annotated[int, Field(ge=0)]
    gates: dict[Identifier, bool]
    ready_for_synthetic_action_experiment: bool
    status: Literal[
        "challenge_disposition_ready_for_synthetic_action_experiment_v371",
        "challenge_disposition_failed_v371",
    ]
    task_router_permitted: Literal[False] = False
    overall_qualification_permitted: Literal[False] = False
    confirmation_permitted: Literal[False] = False
    real_world_authorization_permitted: Literal[False] = False
    created_at: datetime
    evolution_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "ChallengeDispositionEvolutionReportV371":
        _assert_timezone(self.created_at, "created_at")
        ready = all(self.gates.values())
        if self.ready_for_synthetic_action_experiment != ready:
            raise ValueError("V3.7.1 readiness disagrees with gates")
        expected = (
            "challenge_disposition_ready_for_synthetic_action_experiment_v371"
            if ready else "challenge_disposition_failed_v371"
        )
        if self.status != expected:
            raise ValueError("V3.7.1 status disagrees with gates")
        if self.evolution_hash and self.evolution_hash != self.content_hash():
            raise ValueError("evolution_hash does not match V3.7.1 report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evolution_hash")

    def assert_sealed(self) -> None:
        if not self.evolution_hash or self.evolution_hash != self.content_hash():
            raise ValueError("V3.7.1 report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ChallengeDispositionEvolutionReportV371":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evolution_hash"}),
            evolution_hash=draft.content_hash(),
        )


def evaluate_challenge_dispositions_v371(
    spec: ModelChallengeWorldPackSpecV371,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    candidate: ModelChallengeBundleV37,
    source_v37_report: ModelChallengeEvolutionReportV37,
    dispositions: ChallengeDispositionBundleV371,
    *,
    evaluated_at: datetime,
) -> ChallengeDispositionEvolutionReportV371:
    for artifact in (spec, private_pack, candidate, source_v37_report, dispositions):
        artifact.assert_sealed()
    if dispositions.spec_hash != spec.spec_hash:
        raise ValueError("V3.7.1 disposition spec binding differs")
    if dispositions.source_candidate_bundle_hash != candidate.bundle_hash:
        raise ValueError("V3.7.1 disposition candidate binding differs")
    private_by_id = {item.public_case.case_id: item for item in private_pack.cases}
    candidate_by_id = {item.case_id: item for item in candidate.case_receipts}
    disposition_by_id = {item.case_id: item for item in dispositions.dispositions}
    if set(private_by_id) != set(candidate_by_id) or set(private_by_id) != set(disposition_by_id):
        raise ValueError("V3.7.1 case coverage differs")
    case_results: list[PrivateDispositionCaseResultV371] = []
    action_total: dict[str, int] = defaultdict(int)
    action_correct: dict[str, int] = defaultdict(int)
    false_private = 0
    for case_id, private_case in private_by_id.items():
        receipt = candidate_by_id[case_id]
        disposition = disposition_by_id[case_id]
        if (
            disposition.applicability_state_hash != receipt.applicability_state.state_hash
            or disposition.challenge_hashes
            != [item.challenge_hash for item in receipt.challenges]
            or disposition.model_decision_hash != receipt.decision.decision_hash
        ):
            raise ValueError("V3.7.1 disposition input binding differs")
        target_state = _target_state_v371(private_case, receipt)
        if (
            disposition.target_state_hash != target_state.target_state_hash
            or disposition.observed_target_status != target_state.target_status
            or disposition.observed_unresolved_fields != target_state.unresolved_fields
        ):
            raise ValueError("V3.7.1 disposition target-state binding differs")
        expected = _expected_action_v371(private_case, receipt)
        correct = disposition.proposed_action == expected
        bad_private = (
            disposition.proposed_action == "proceed_private_validation"
            and expected != "proceed_private_validation"
        )
        action_total[expected] += 1
        action_correct[expected] += int(correct)
        false_private += int(bad_private)
        case_results.append(PrivateDispositionCaseResultV371(
            case_id=case_id,
            mechanism=private_case.mechanism,
            proposed_action=disposition.proposed_action,
            expected_action=expected,
            route_correct=correct,
            false_private_validation=bad_private,
        ))
    route_accuracy = sum(item.route_correct for item in case_results) / len(case_results)
    per_action = {
        action: action_correct[action] / count
        for action, count in action_total.items()
    }
    gates = {
        "complete_case_coverage": len(case_results) == len(private_pack.cases) == 64,
        "overall_route_accuracy": route_accuracy >= 0.90,
        "per_action_accuracy": all(value >= 0.80 for value in per_action.values()),
        "problem_clarification_present": (
            action_total.get("clarify_decision_target", 0) > 0
        ),
        "no_false_private_validation": false_private == 0,
        "source_failure_hash_bound": (
            spec.source_failure_evolution_hash
            == dispositions.dispositions[0].source_failure_evolution_hash
        ),
        "no_private_disposition_inputs": all(
            not item.private_mechanism_seen
            and not item.private_probe_seen
            and not item.private_target_loss_seen
            and not item.execution_permitted
            for item in dispositions.dispositions
        ),
    }
    ready = all(gates.values())
    return ChallengeDispositionEvolutionReportV371.seal(
        evolution_id=f"evolution_{spec.experiment_id}",
        spec_hash=spec.spec_hash,
        source_v37_evolution_hash=source_v37_report.evolution_hash,
        disposition_bundle_hash=dispositions.bundle_hash,
        case_results=case_results,
        route_accuracy=route_accuracy,
        action_counts=dict(action_total),
        per_action_accuracy=per_action,
        false_private_validation_count=false_private,
        gates=gates,
        ready_for_synthetic_action_experiment=ready,
        status=(
            "challenge_disposition_ready_for_synthetic_action_experiment_v371"
            if ready else "challenge_disposition_failed_v371"
        ),
        created_at=evaluated_at,
    )


class ChallengeDispositionManifestV371(StrictModel):
    schema_version: Literal["3.7.1"] = "3.7.1"
    run_id: Identifier
    artifact_refs: list[ArtifactRef] = Field(min_length=11, max_length=11)
    terminal_status: Literal[
        "challenge_disposition_ready_for_synthetic_action_experiment_v371",
        "challenge_disposition_failed_v371",
    ]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "ChallengeDispositionManifestV371":
        _assert_timezone(self.created_at, "created_at")
        if len({item.kind for item in self.artifact_refs}) != len(self.artifact_refs):
            raise ValueError("V3.7.1 manifest artifact kinds must be unique")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match V3.7.1 manifest")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("V3.7.1 manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ChallengeDispositionManifestV371":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class ChallengeDispositionOutcomeV371:
    store: RunStore
    method_evidence: ModelChallengeMethodEvidenceV37
    spec: ModelChallengeWorldPackSpecV371
    baseline_policy: ModelPortfolioPolicyV37
    candidate_policy: ModelPortfolioPolicyV37
    disposition_policy: ChallengeDispositionPolicyV371
    private_pack: PrivateControlledDynamicsWorldPackV31
    baseline_bundle: ModelChallengeBundleV37
    candidate_bundle: ModelChallengeBundleV37
    source_v37_report: ModelChallengeEvolutionReportV37
    disposition_bundle: ChallengeDispositionBundleV371
    evolution_report: ChallengeDispositionEvolutionReportV371
    manifest: ChallengeDispositionManifestV371


def run_challenge_disposition_worldpack_v371(
    output_root: str | Path,
    *,
    source_v37_run_directory: str | Path,
    method_evidence: ModelChallengeMethodEvidenceV37,
    spec: ModelChallengeWorldPackSpecV371,
    baseline_policy: ModelPortfolioPolicyV37,
    candidate_policy: ModelPortfolioPolicyV37,
    disposition_policy: ChallengeDispositionPolicyV371,
    evaluated_at: datetime | None = None,
    run_id: str | None = None,
) -> ChallengeDispositionOutcomeV371:
    source_candidate, source_report = _load_source_v37(source_v37_run_directory)
    for artifact in (
        method_evidence, spec, baseline_policy, candidate_policy, disposition_policy
    ):
        artifact.assert_sealed()
    if (
        source_candidate.bundle_hash != disposition_policy.source_failure_candidate_bundle_hash
        or source_report.evolution_hash != disposition_policy.source_failure_evolution_hash
    ):
        raise ValueError("V3.7.1 disposition policy source binding differs")
    if (
        spec.method_evidence_hash != method_evidence.evidence_hash
        or spec.baseline_policy_hash != baseline_policy.policy_hash
        or spec.candidate_policy_hash != candidate_policy.policy_hash
        or spec.disposition_policy_hash != disposition_policy.policy_hash
    ):
        raise ValueError("V3.7.1 frozen artifact binding differs")
    at = evaluated_at or datetime.now(timezone.utc)
    store = RunStore(
        output_root,
        run_id=run_id or f"challenge-disposition-v371-{uuid4().hex[:10]}",
    )
    refs = [
        store.put_artifact("model_challenge_method_evidence_v371", method_evidence),
        store.put_artifact("model_challenge_spec_v371", spec),
        store.put_artifact("model_challenge_baseline_policy_v371", baseline_policy),
        store.put_artifact("model_challenge_candidate_policy_v371", candidate_policy),
        store.put_artifact("challenge_disposition_policy_v371", disposition_policy),
    ]
    store.emit("challenge_disposition_v371_protocol_frozen_before_private_pack", {
        "spec_hash": spec.spec_hash,
        "source_failure_evolution_hash": source_report.evolution_hash,
        "disposition_policy_hash": disposition_policy.policy_hash,
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
    evolution = evaluate_challenge_dispositions_v371(
        spec,
        private_pack,
        candidate,
        v37_report,
        dispositions,
        evaluated_at=at,
    )
    refs.extend([
        store.put_artifact("private_challenge_disposition_worldpack_v371", private_pack),
        store.put_artifact("model_challenge_baseline_bundle_v371", baseline),
        store.put_artifact("model_challenge_candidate_bundle_v371", candidate),
        store.put_artifact("source_model_challenge_report_v371", v37_report),
        store.put_artifact("challenge_disposition_bundle_v371", dispositions),
        store.put_artifact("challenge_disposition_evolution_report_v371", evolution),
    ])
    manifest = ChallengeDispositionManifestV371.seal(
        run_id=store.run_id,
        artifact_refs=refs,
        terminal_status=evolution.status,
        created_at=at,
    )
    manifest_ref = store.put_artifact("challenge_disposition_manifest_v371", manifest)
    store.emit("challenge_disposition_v371_worldpack_adjudicated", {
        "manifest_ref": manifest_ref.model_dump(mode="json")
    })
    if not verify_challenge_disposition_run_v371(store.run_directory):
        raise RuntimeError("V3.7.1 run failed independent verification")
    return ChallengeDispositionOutcomeV371(
        store,
        method_evidence,
        spec,
        baseline_policy,
        candidate_policy,
        disposition_policy,
        private_pack,
        baseline,
        candidate,
        v37_report,
        dispositions,
        evolution,
        manifest,
    )


def verify_challenge_disposition_run_v371(run_directory: str | Path) -> bool:
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
            if item.kind == "challenge_disposition_manifest_v371"
        ]
        if len(manifest_refs) != 1:
            return False
        manifest = ChallengeDispositionManifestV371.model_validate(
            store.load_artifact(manifest_refs[0])
        )
        manifest.assert_sealed()

        def load_one(kind: str, model):
            references = [item for item in manifest.artifact_refs if item.kind == kind]
            if len(references) != 1:
                raise RuntimeError(f"V3.7.1 manifest needs exactly one {kind}")
            return model.model_validate(store.load_artifact(references[0]))

        method = load_one(
            "model_challenge_method_evidence_v371", ModelChallengeMethodEvidenceV37
        )
        spec = load_one("model_challenge_spec_v371", ModelChallengeWorldPackSpecV371)
        baseline_policy = load_one(
            "model_challenge_baseline_policy_v371", ModelPortfolioPolicyV37
        )
        candidate_policy = load_one(
            "model_challenge_candidate_policy_v371", ModelPortfolioPolicyV37
        )
        disposition_policy = load_one(
            "challenge_disposition_policy_v371", ChallengeDispositionPolicyV371
        )
        private_pack = load_one(
            "private_challenge_disposition_worldpack_v371",
            PrivateControlledDynamicsWorldPackV31,
        )
        baseline = load_one(
            "model_challenge_baseline_bundle_v371", ModelChallengeBundleV37
        )
        candidate = load_one(
            "model_challenge_candidate_bundle_v371", ModelChallengeBundleV37
        )
        v37_report = load_one(
            "source_model_challenge_report_v371", ModelChallengeEvolutionReportV37
        )
        dispositions = load_one(
            "challenge_disposition_bundle_v371", ChallengeDispositionBundleV371
        )
        evolution = load_one(
            "challenge_disposition_evolution_report_v371",
            ChallengeDispositionEvolutionReportV371,
        )
        for artifact in (
            method, spec, baseline_policy, candidate_policy, disposition_policy,
            private_pack, baseline, candidate, v37_report, dispositions,
            evolution, manifest,
        ):
            artifact.assert_sealed()
        if manifest.run_id != store.run_id:
            return False
        if (
            spec.method_evidence_hash != method.evidence_hash
            or spec.baseline_policy_hash != baseline_policy.policy_hash
            or spec.candidate_policy_hash != candidate_policy.policy_hash
            or spec.disposition_policy_hash != disposition_policy.policy_hash
            or spec.source_failure_evolution_hash
            != disposition_policy.source_failure_evolution_hash
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
        recomputed_v37 = evaluate_model_challenge_worldpack_v37(
            spec,
            private_pack,
            baseline,
            candidate,
            evaluated_at=v37_report.created_at,
        )
        if recomputed_v37.evolution_hash != v37_report.evolution_hash:
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
        recomputed = evaluate_challenge_dispositions_v371(
            spec,
            private_pack,
            candidate,
            v37_report,
            dispositions,
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
            == "challenge_disposition_v371_protocol_frozen_before_private_pack"
        ]
        private_events = [
            event for event in events
            if event["event_type"] == "artifact_committed"
            and event["payload"]["kind"]
            == "private_challenge_disposition_worldpack_v371"
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
