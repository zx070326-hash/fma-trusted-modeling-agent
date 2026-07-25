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
from scipy.stats import beta

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore
from fma.v2.dynamics_ir import trajectory_nrmse
from fma.v2.schemas import Identifier, Sha256, _assert_timezone

from .controlled_dynamics_loop import (
    MECHANISMS_V31,
    ControlledDriftModelV31,
    ControlledObservationReceiptV31,
    MechanismV31,
    PrivateControlledDynamicsCaseV31,
    PrivateControlledDynamicsWorldPackV31,
    _simulate_model_v31,
    _target_loss_v31,
    generate_private_controlled_dynamics_worldpack_v31,
)
from .model_challenge_v37 import (
    FAMILY_DEGREES_V37,
    FAMILIES_V37,
    FamilyV37,
    ModelChallengeWorldPackSpecV37,
    _fit_family_v37,
    _hash_without,
    _shared_observations_v37,
)
from .target_discriminating_acquisition_v381 import (
    AcquisitionBundleV381,
    TargetDiscriminatingEvolutionReportV381,
    TargetDiscriminatingWorldPackSpecV381,
    verify_target_discriminating_run_v381,
)


EXPLORATORY_SEEDS_V39 = (
    23003, 23057, 23117, 23167, 23227, 23279, 23339, 23399,
    23447, 23497, 23549, 23599, 23663, 23719, 23773, 23827,
)

ValidatorArmV39 = Literal[
    "legacy_expanded_rows_as_segments",
    "action_hash_bound_segment_sequence",
]
InputBindingSourceV39 = Literal[
    "observation_expanded_rows_misread_as_segments",
    "pilot_zero_segment_contract",
    "public_catalog_action_hash",
]


def _committed_refs_v39(store: RunStore) -> list[ArtifactRef]:
    events = [
        json.loads(line)
        for line in store.event_path.read_text(encoding="utf-8").splitlines()
    ]
    return [
        ArtifactRef.model_validate(event["payload"])
        for event in events if event["event_type"] == "artifact_committed"
    ]


def _load_one_v39(store: RunStore, refs: list[ArtifactRef], kind: str, model):
    matches = [item for item in refs if item.kind == kind]
    if len(matches) != 1:
        raise ValueError(f"V3.9 requires exactly one {kind}")
    return model.model_validate(store.load_artifact(matches[0]))


def _load_source_v381_v39(source: str | Path) -> dict[str, object]:
    if not verify_target_discriminating_run_v381(source):
        raise ValueError("V3.9 source V3.8.1 run did not independently verify")
    store = RunStore.open_existing(source)
    refs = _committed_refs_v39(store)
    return {
        "spec": _load_one_v39(
            store, refs, "target_discriminating_spec_v381",
            TargetDiscriminatingWorldPackSpecV381,
        ),
        "private_pack": _load_one_v39(
            store, refs, "private_target_discriminating_worldpack_v381",
            PrivateControlledDynamicsWorldPackV31,
        ),
        "acquisition_bundle": _load_one_v39(
            store, refs, "candidate_acquisition_bundle_v381",
            AcquisitionBundleV381,
        ),
        "evolution": _load_one_v39(
            store, refs, "target_discriminating_evolution_report_v381",
            TargetDiscriminatingEvolutionReportV381,
        ),
    }


class ValidatorInputBugEvidenceV39(StrictModel):
    schema_version: Literal["3.9"] = "3.9"
    evidence_id: Identifier
    source_v381_evolution_hash: Sha256
    source_v381_candidate_acquisition_bundle_hash: Sha256
    diagnosed_case_count: Literal[22] = 22
    legacy_resolved_count: Literal[8] = 8
    recovered_resolved_count: Literal[22] = 22
    recovered_mean_public_cv_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    recovered_mean_private_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    recovered_duffing_mean_private_target_loss: Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ]
    root_cause: Literal[
        "expanded_point_input_rows_passed_to_segment_sequence_simulator"
    ] = "expanded_point_input_rows_passed_to_segment_sequence_simulator"
    historical_numeric_conclusions_superseded: Literal[True] = True
    model_qualification_permitted: Literal[False] = False
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> "ValidatorInputBugEvidenceV39":
        if self.recovered_duffing_mean_private_target_loss <= 0.2:
            raise ValueError("V3.9 training evidence must preserve skeleton gap")
        if self.evidence_hash and self.evidence_hash != self.content_hash():
            raise ValueError("evidence_hash does not match V3.9 bug evidence")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evidence_hash")

    def assert_sealed(self) -> None:
        if not self.evidence_hash or self.evidence_hash != self.content_hash():
            raise ValueError("V3.9 bug evidence is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ValidatorInputBugEvidenceV39":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evidence_hash"}),
            evidence_hash=draft.content_hash(),
        )


class ValidatorInputContractPolicyV39(StrictModel):
    schema_version: Literal["3.9"] = "3.9"
    policy_id: Identifier
    arm: ValidatorArmV39
    bug_evidence_hash: Sha256
    family_catalog: list[FamilyV37] = Field(min_length=3, max_length=3)
    observation_action_indices: list[int] = Field(min_length=2, max_length=2)
    input_binding_rule: Literal[
        "pass_expanded_observation_rows_to_segment_simulator_fault_injection",
        "bind_pilot_zero_or_unique_public_action_hash_to_six_segments",
    ]
    selection_rule: Literal[
        "simplest_eligible_within_best_mean_plus_best_standard_error"
    ] = "simplest_eligible_within_best_mean_plus_best_standard_error"
    private_mechanism_visible: Literal[False] = False
    private_probe_visible: Literal[False] = False
    private_target_loss_visible: Literal[False] = False
    real_world_execution_permitted: Literal[False] = False
    task_router_permitted: Literal[False] = False
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "ValidatorInputContractPolicyV39":
        if self.family_catalog != list(FAMILIES_V37):
            raise ValueError("V3.9 family catalog differs")
        if self.observation_action_indices != [0, 7]:
            raise ValueError("V3.9 observation actions differ")
        expected = (
            "pass_expanded_observation_rows_to_segment_simulator_fault_injection"
            if self.arm == "legacy_expanded_rows_as_segments"
            else "bind_pilot_zero_or_unique_public_action_hash_to_six_segments"
        )
        if self.input_binding_rule != expected:
            raise ValueError("V3.9 input binding rule disagrees with arm")
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("policy_hash does not match V3.9 policy")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("V3.9 policy is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ValidatorInputContractPolicyV39":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"policy_hash"}),
            policy_hash=draft.content_hash(),
        )


def default_validator_input_contract_policies_v39(
    evidence: ValidatorInputBugEvidenceV39,
) -> tuple[ValidatorInputContractPolicyV39, ValidatorInputContractPolicyV39]:
    evidence.assert_sealed()
    common = {
        "bug_evidence_hash": evidence.evidence_hash,
        "family_catalog": list(FAMILIES_V37),
        "observation_action_indices": [0, 7],
    }
    return (
        ValidatorInputContractPolicyV39.seal(
            policy_id="legacy_expanded_rows_fault_injection_v39",
            arm="legacy_expanded_rows_as_segments",
            input_binding_rule=(
                "pass_expanded_observation_rows_to_segment_simulator_fault_injection"
            ),
            **common,
        ),
        ValidatorInputContractPolicyV39.seal(
            policy_id="action_hash_bound_segment_sequence_v39",
            arm="action_hash_bound_segment_sequence",
            input_binding_rule=(
                "bind_pilot_zero_or_unique_public_action_hash_to_six_segments"
            ),
            **common,
        ),
    )


class ValidatorRecoveryWorldPackSpecV39(ModelChallengeWorldPackSpecV37):
    schema_version: Literal["3.9"] = "3.9"
    seeds: list[int] = Field(min_length=16, max_length=16)
    bootstrap_seed: Literal[390722] = 390722
    bug_evidence_hash: Sha256
    source_v381_evolution_hash: Sha256
    minimum_coverage_improvement: Literal[0.5] = 0.5
    unresolved_adjudicated_loss: Literal[10.0] = 10.0
    skeleton_gap_private_loss: Literal[0.2] = 0.2
    baseline_policy_hash: Sha256
    candidate_policy_hash: Sha256
    method_evidence_hash: Sha256
    frozen_delta: Literal[
        "heldout_simulation_input_binding_only"
    ] = "heldout_simulation_input_binding_only"

    @model_validator(mode="after")
    def validate_spec(self) -> "ValidatorRecoveryWorldPackSpecV39":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.mechanisms != list(MECHANISMS_V31):
            raise ValueError("V3.9 mechanism order differs")
        if self.seeds != list(EXPLORATORY_SEEDS_V39):
            raise ValueError("V3.9 seeds do not match frozen set")
        if self.family_catalog != list(FAMILIES_V37):
            raise ValueError("V3.9 family catalog differs")
        if self.observation_action_indices != [0, 7]:
            raise ValueError("V3.9 observation actions differ")
        if not math.isclose(
            (self.trajectory_points - 1) * self.time_step,
            self.segment_count * self.segment_duration,
            abs_tol=1e-12,
        ):
            raise ValueError("V3.9 segment coverage differs")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match V3.9 protocol")
        return self


def default_validator_recovery_spec_v39(
    *,
    evidence: ValidatorInputBugEvidenceV39,
    baseline_policy: ValidatorInputContractPolicyV39,
    candidate_policy: ValidatorInputContractPolicyV39,
    frozen_at: datetime | None = None,
) -> ValidatorRecoveryWorldPackSpecV39:
    for artifact in (evidence, baseline_policy, candidate_policy):
        artifact.assert_sealed()
    return ValidatorRecoveryWorldPackSpecV39.seal(
        experiment_id="validator_input_contract_recovery_v39",
        mechanisms=list(MECHANISMS_V31),
        seeds=list(EXPLORATORY_SEEDS_V39),
        family_catalog=list(FAMILIES_V37),
        observation_action_indices=[0, 7],
        baseline_policy_hash=baseline_policy.policy_hash,
        candidate_policy_hash=candidate_policy.policy_hash,
        method_evidence_hash=evidence.evidence_hash,
        bug_evidence_hash=evidence.evidence_hash,
        source_v381_evolution_hash=evidence.source_v381_evolution_hash,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


class FoldInputBindingReceiptV39(StrictModel):
    schema_version: Literal["3.9"] = "3.9"
    binding_id: Identifier
    case_id: Identifier
    observation_hash: Sha256
    observation_action_hash: Sha256 | None
    arm: ValidatorArmV39
    binding_source: InputBindingSourceV39
    observation_input_row_count: Literal[49] = 49
    simulator_input_value_count: Annotated[int, Field(ge=6, le=49)]
    required_segment_count: Literal[6] = 6
    simulator_input_hash: Sha256
    contract_valid: bool
    binding_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_binding(self) -> "FoldInputBindingReceiptV39":
        if self.arm == "legacy_expanded_rows_as_segments":
            expected_source = "observation_expanded_rows_misread_as_segments"
            expected_count = 49
            expected_valid = False
        else:
            expected_source = (
                "pilot_zero_segment_contract"
                if self.observation_action_hash is None
                else "public_catalog_action_hash"
            )
            expected_count = 6
            expected_valid = True
        if (
            self.binding_source != expected_source
            or self.simulator_input_value_count != expected_count
            or self.contract_valid != expected_valid
        ):
            raise ValueError("V3.9 input binding receipt semantics differ")
        if self.binding_hash and self.binding_hash != self.content_hash():
            raise ValueError("binding_hash does not match V3.9 binding")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "binding_hash")

    def assert_sealed(self) -> None:
        if not self.binding_hash or self.binding_hash != self.content_hash():
            raise ValueError("V3.9 input binding is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "FoldInputBindingReceiptV39":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"binding_hash"}),
            binding_hash=draft.content_hash(),
        )


def _input_binding_v39(
    private_case: PrivateControlledDynamicsCaseV31,
    observation: ControlledObservationReceiptV31,
    arm: ValidatorArmV39,
    spec,
) -> tuple[list[list[float]], FoldInputBindingReceiptV39]:
    action_hash = getattr(observation, "action_hash", None)
    if arm == "legacy_expanded_rows_as_segments":
        values = observation.inputs
        source: InputBindingSourceV39 = (
            "observation_expanded_rows_misread_as_segments"
        )
        valid = False
    elif action_hash is None:
        values = [[0.0] for _ in range(spec.segment_count)]
        source = "pilot_zero_segment_contract"
        valid = True
    else:
        matches = [
            action for action in private_case.public_case.action_catalog
            if action.action_hash == action_hash
        ]
        if len(matches) != 1:
            raise ValueError("V3.9 observation action hash is not unique in catalog")
        values = matches[0].input_values
        source = "public_catalog_action_hash"
        valid = True
    receipt = FoldInputBindingReceiptV39.seal(
        binding_id=f"binding_{arm}_{observation.observation_id}",
        case_id=private_case.public_case.case_id,
        observation_hash=observation.observation_hash,
        observation_action_hash=action_hash,
        arm=arm,
        binding_source=source,
        simulator_input_value_count=len(values),
        simulator_input_hash=sha256_value(values),
        contract_valid=valid,
    )
    return values, receipt


class ValidatorFamilyChallengeV39(StrictModel):
    schema_version: Literal["3.9"] = "3.9"
    challenge_id: Identifier
    case_id: Identifier
    arm: ValidatorArmV39
    family: FamilyV37
    source_observation_hashes: list[Sha256] = Field(min_length=3, max_length=3)
    fold_input_bindings: list[FoldInputBindingReceiptV39] = Field(min_length=3, max_length=3)
    fold_prediction_losses: list[
        Annotated[float, Field(ge=0, allow_inf_nan=False)]
    ] = Field(min_length=3, max_length=3)
    mean_cv_prediction_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    standard_error_cv_prediction_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    normalized_rank_ratio: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    normalized_condition_number: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    basis_term_count: Annotated[int, Field(ge=1)]
    final_model_hash: Sha256
    eligible: bool
    challenge_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_challenge(self) -> "ValidatorFamilyChallengeV39":
        for binding in self.fold_input_bindings:
            binding.assert_sealed()
            if binding.arm != self.arm:
                raise ValueError("V3.9 challenge/binding arm differs")
        values = np.asarray(self.fold_prediction_losses, dtype=float)
        mean = float(np.mean(values))
        se = float(np.std(values, ddof=1) / math.sqrt(len(values)))
        if not math.isclose(self.mean_cv_prediction_loss, mean, abs_tol=1e-12):
            raise ValueError("V3.9 CV mean does not recompute")
        if not math.isclose(
            self.standard_error_cv_prediction_loss, se, abs_tol=1e-12
        ):
            raise ValueError("V3.9 CV SE does not recompute")
        expected = (
            all(value < 9.999 for value in values)
            and self.normalized_rank_ratio >= 0.95
            and self.normalized_condition_number <= 100000000.0
            and mean <= 0.35
        )
        if self.eligible != expected:
            raise ValueError("V3.9 eligibility does not recompute")
        if self.challenge_hash and self.challenge_hash != self.content_hash():
            raise ValueError("challenge_hash does not match V3.9 challenge")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "challenge_hash")

    def assert_sealed(self) -> None:
        if not self.challenge_hash or self.challenge_hash != self.content_hash():
            raise ValueError("V3.9 family challenge is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ValidatorFamilyChallengeV39":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"challenge_hash"}),
            challenge_hash=draft.content_hash(),
        )


class ValidatorDecisionV39(StrictModel):
    schema_version: Literal["3.9"] = "3.9"
    decision_id: Identifier
    case_id: Identifier
    arm: ValidatorArmV39
    challenge_hashes: list[Sha256] = Field(min_length=3, max_length=3)
    decision: Literal["select", "abstain"]
    reason: Literal[
        "deny_data_quality",
        "needs_evidence_no_eligible_family",
        "one_standard_error_challenge",
    ]
    selected_family: FamilyV37 | None
    best_family: FamilyV37 | None
    one_standard_error_threshold: Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ] | None
    decision_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> "ValidatorDecisionV39":
        if (self.decision == "select") != (self.selected_family is not None):
            raise ValueError("V3.9 decision/family binding differs")
        if self.decision_hash and self.decision_hash != self.content_hash():
            raise ValueError("decision_hash does not match V3.9 decision")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "decision_hash")

    def assert_sealed(self) -> None:
        if not self.decision_hash or self.decision_hash != self.content_hash():
            raise ValueError("V3.9 decision is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ValidatorDecisionV39":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"decision_hash"}),
            decision_hash=draft.content_hash(),
        )


class ValidatorCaseReceiptV39(StrictModel):
    schema_version: Literal["3.9"] = "3.9"
    receipt_id: Identifier
    case_id: Identifier
    policy_hash: Sha256
    arm: ValidatorArmV39
    quality_flags: list[Identifier]
    challenges: list[ValidatorFamilyChallengeV39] = Field(min_length=3, max_length=3)
    decision: ValidatorDecisionV39
    selected_model: ControlledDriftModelV31 | None
    executed_at: datetime
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "ValidatorCaseReceiptV39":
        _assert_timezone(self.executed_at, "executed_at")
        self.decision.assert_sealed()
        for challenge in self.challenges:
            challenge.assert_sealed()
        if [item.family for item in self.challenges] != list(FAMILIES_V37):
            raise ValueError("V3.9 challenge family order differs")
        hashes = [item.challenge_hash for item in self.challenges]
        if self.decision.challenge_hashes != hashes:
            raise ValueError("V3.9 decision/challenge binding differs")
        if self.decision.decision == "select":
            if self.selected_model is None:
                raise ValueError("V3.9 selected decision needs model")
            self.selected_model.assert_sealed()
            degree = max(
                sum(term.exponents) for term in self.selected_model.basis_terms
            )
            if degree != FAMILY_DEGREES_V37[self.decision.selected_family]:
                raise ValueError("V3.9 selected model family differs")
        elif self.selected_model is not None:
            raise ValueError("V3.9 abstention cannot contain model")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match V3.9 receipt")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("V3.9 case receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ValidatorCaseReceiptV39":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


class ValidatorBundleV39(StrictModel):
    schema_version: Literal["3.9"] = "3.9"
    bundle_id: Identifier
    spec_hash: Sha256
    private_pack_hash: Sha256
    policy_hash: Sha256
    arm: ValidatorArmV39
    case_receipts: list[ValidatorCaseReceiptV39] = Field(min_length=64, max_length=64)
    created_at: datetime
    bundle_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bundle(self) -> "ValidatorBundleV39":
        _assert_timezone(self.created_at, "created_at")
        if len({item.case_id for item in self.case_receipts}) != 64:
            raise ValueError("V3.9 bundle case coverage differs")
        for item in self.case_receipts:
            item.assert_sealed()
            if item.arm != self.arm or item.policy_hash != self.policy_hash:
                raise ValueError("V3.9 receipt policy binding differs")
        if self.bundle_hash and self.bundle_hash != self.content_hash():
            raise ValueError("bundle_hash does not match V3.9 bundle")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "bundle_hash")

    def assert_sealed(self) -> None:
        if not self.bundle_hash or self.bundle_hash != self.content_hash():
            raise ValueError("V3.9 validator bundle is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ValidatorBundleV39":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"bundle_hash"}),
            bundle_hash=draft.content_hash(),
        )


def _challenge_family_v39(
    private_case: PrivateControlledDynamicsCaseV31,
    observations: list[ControlledObservationReceiptV31],
    family: FamilyV37,
    policy: ValidatorInputContractPolicyV39,
    spec: ValidatorRecoveryWorldPackSpecV39,
) -> tuple[ValidatorFamilyChallengeV39, ControlledDriftModelV31]:
    losses: list[float] = []
    bindings: list[FoldInputBindingReceiptV39] = []
    for holdout_index, holdout in enumerate(observations):
        training = [
            item for index, item in enumerate(observations)
            if index != holdout_index
        ]
        model = _fit_family_v37(
            private_case,
            training,
            family,
            spec,
            model_suffix=f"{policy.arm}_fold{holdout_index}_v39",
        )
        input_values, binding = _input_binding_v39(
            private_case, holdout, policy.arm, spec
        )
        bindings.append(binding)
        try:
            predicted = _simulate_model_v31(
                model,
                private_case.public_case.actuator,
                holdout.states[0],
                holdout.times,
                input_values,
                spec.segment_duration,
            )
            losses.append(trajectory_nrmse(holdout.states, predicted))
        except (RuntimeError, FloatingPointError, ValueError):
            losses.append(10.0)
    final_model = _fit_family_v37(
        private_case,
        observations,
        family,
        spec,
        model_suffix=f"{policy.arm}_all_v39",
    )
    basis_count = len(final_model.basis_terms)
    rank_ratio = final_model.normalized_design_rank / basis_count
    values = np.asarray(losses, dtype=float)
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    eligible = (
        all(value < 9.999 for value in values)
        and rank_ratio >= spec.minimum_rank_ratio
        and final_model.normalized_condition_number <= spec.maximum_condition_number
        and mean <= spec.maximum_cv_prediction_loss
    )
    return (
        ValidatorFamilyChallengeV39.seal(
            challenge_id=(
                f"validator_{policy.arm}_{private_case.public_case.case_id}_{family}"
            ),
            case_id=private_case.public_case.case_id,
            arm=policy.arm,
            family=family,
            source_observation_hashes=[
                item.observation_hash for item in observations
            ],
            fold_input_bindings=bindings,
            fold_prediction_losses=losses,
            mean_cv_prediction_loss=mean,
            standard_error_cv_prediction_loss=se,
            normalized_rank_ratio=rank_ratio,
            normalized_condition_number=final_model.normalized_condition_number,
            basis_term_count=basis_count,
            final_model_hash=final_model.model_hash,
            eligible=eligible,
        ),
        final_model,
    )


def _execute_case_v39(
    private_case: PrivateControlledDynamicsCaseV31,
    policy: ValidatorInputContractPolicyV39,
    spec: ValidatorRecoveryWorldPackSpecV39,
    executed_at: datetime,
) -> ValidatorCaseReceiptV39:
    observations = _shared_observations_v37(private_case, spec)
    challenges: list[ValidatorFamilyChallengeV39] = []
    models: dict[str, ControlledDriftModelV31] = {}
    for family in FAMILIES_V37:
        challenge, model = _challenge_family_v39(
            private_case, observations, family, policy, spec
        )
        challenges.append(challenge)
        models[family] = model
    common = {
        "decision_id": f"decision_{policy.arm}_{private_case.public_case.case_id}",
        "case_id": private_case.public_case.case_id,
        "arm": policy.arm,
        "challenge_hashes": [item.challenge_hash for item in challenges],
    }
    quality_flags = sorted({
        flag for item in observations for flag in item.quality_flags
    })
    selected_model = None
    if quality_flags:
        decision = ValidatorDecisionV39.seal(
            decision="abstain",
            reason="deny_data_quality",
            selected_family=None,
            best_family=None,
            one_standard_error_threshold=None,
            **common,
        )
    else:
        eligible = [item for item in challenges if item.eligible]
        if not eligible:
            decision = ValidatorDecisionV39.seal(
                decision="abstain",
                reason="needs_evidence_no_eligible_family",
                selected_family=None,
                best_family=None,
                one_standard_error_threshold=None,
                **common,
            )
        else:
            best = min(eligible, key=lambda item: (
                item.mean_cv_prediction_loss, item.basis_term_count
            ))
            threshold = (
                best.mean_cv_prediction_loss
                + best.standard_error_cv_prediction_loss
            )
            within = [
                item for item in eligible
                if item.mean_cv_prediction_loss <= threshold
            ]
            selected = min(within, key=lambda item: (
                item.basis_term_count, item.mean_cv_prediction_loss
            ))
            decision = ValidatorDecisionV39.seal(
                decision="select",
                reason="one_standard_error_challenge",
                selected_family=selected.family,
                best_family=best.family,
                one_standard_error_threshold=threshold,
                **common,
            )
            selected_model = models[selected.family]
    return ValidatorCaseReceiptV39.seal(
        receipt_id=f"receipt_{policy.arm}_{private_case.public_case.case_id}",
        case_id=private_case.public_case.case_id,
        policy_hash=policy.policy_hash,
        arm=policy.arm,
        quality_flags=quality_flags,
        challenges=challenges,
        decision=decision,
        selected_model=selected_model,
        executed_at=executed_at,
    )


def execute_validator_policy_v39(
    spec: ValidatorRecoveryWorldPackSpecV39,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    policy: ValidatorInputContractPolicyV39,
    *,
    executed_at: datetime,
) -> ValidatorBundleV39:
    for artifact in (spec, private_pack, policy):
        artifact.assert_sealed()
    expected = (
        spec.baseline_policy_hash
        if policy.arm == "legacy_expanded_rows_as_segments"
        else spec.candidate_policy_hash
    )
    if policy.policy_hash != expected:
        raise ValueError("V3.9 policy is not frozen in spec")
    return ValidatorBundleV39.seal(
        bundle_id=f"bundle_{policy.arm}_{spec.experiment_id}",
        spec_hash=spec.spec_hash,
        private_pack_hash=private_pack.pack_hash,
        policy_hash=policy.policy_hash,
        arm=policy.arm,
        case_receipts=[
            _execute_case_v39(private_case, policy, spec, executed_at)
            for private_case in private_pack.cases
        ],
        created_at=executed_at,
    )


def build_validator_input_bug_evidence_v39(
    source_v381_run_directory: str | Path,
) -> ValidatorInputBugEvidenceV39:
    source = _load_source_v381_v39(source_v381_run_directory)
    spec = source["spec"]
    private_pack = source["private_pack"]
    acquisition = source["acquisition_bundle"]
    target_ids = {
        item.case_id for item in acquisition.case_receipts
        if item.acquisition_expected
    }
    temporary_evidence_hash = "0" * 64
    legacy_policy = ValidatorInputContractPolicyV39.seal(
        policy_id="training_legacy_v39",
        arm="legacy_expanded_rows_as_segments",
        bug_evidence_hash=temporary_evidence_hash,
        family_catalog=list(FAMILIES_V37),
        observation_action_indices=[0, 7],
        input_binding_rule=(
            "pass_expanded_observation_rows_to_segment_simulator_fault_injection"
        ),
    )
    recovered_policy = ValidatorInputContractPolicyV39.seal(
        policy_id="training_recovered_v39",
        arm="action_hash_bound_segment_sequence",
        bug_evidence_hash=temporary_evidence_hash,
        family_catalog=list(FAMILIES_V37),
        observation_action_indices=[0, 7],
        input_binding_rule=(
            "bind_pilot_zero_or_unique_public_action_hash_to_six_segments"
        ),
    )
    legacy_count = 0
    recovered_count = 0
    recovered_cv: list[float] = []
    recovered_private: list[float] = []
    duffing_private: list[float] = []
    for private_case in private_pack.cases:
        case_id = private_case.public_case.case_id
        if case_id not in target_ids:
            continue
        legacy = _execute_case_v39(
            private_case, legacy_policy, spec, acquisition.created_at
        )
        recovered = _execute_case_v39(
            private_case, recovered_policy, spec, acquisition.created_at
        )
        legacy_count += int(legacy.selected_model is not None)
        recovered_count += int(recovered.selected_model is not None)
        if recovered.selected_model is not None:
            chosen = next(
                item for item in recovered.challenges
                if item.family == recovered.decision.selected_family
            )
            recovered_cv.append(chosen.mean_cv_prediction_loss)
            loss = _target_loss_v31(private_case, recovered.selected_model, spec)
            recovered_private.append(loss)
            if private_case.mechanism == "duffing_oscillator":
                duffing_private.append(loss)
    if (
        len(target_ids) != 22
        or legacy_count != 8
        or recovered_count != 22
    ):
        raise ValueError("V3.9 training validator failure signature changed")
    evolution = source["evolution"]
    return ValidatorInputBugEvidenceV39.seal(
        evidence_id="validator_input_contract_bug_v39",
        source_v381_evolution_hash=evolution.evolution_hash,
        source_v381_candidate_acquisition_bundle_hash=acquisition.bundle_hash,
        recovered_mean_public_cv_loss=float(np.mean(recovered_cv)),
        recovered_mean_private_target_loss=float(np.mean(recovered_private)),
        recovered_duffing_mean_private_target_loss=float(np.mean(duffing_private)),
    )


class PrivateValidatorCaseResultV39(StrictModel):
    schema_version: Literal["3.9"] = "3.9"
    case_id: Identifier
    mechanism: MechanismV31
    legacy_selected_family: FamilyV37 | None
    recovered_selected_family: FamilyV37 | None
    legacy_adjudicated_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    recovered_adjudicated_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    recovered_improvement: Annotated[float, Field(allow_inf_nan=False)]
    material_negative_transfer: bool


class ValidatorRecoveryEvolutionReportV39(StrictModel):
    schema_version: Literal["3.9"] = "3.9"
    evolution_id: Identifier
    spec_hash: Sha256
    bug_evidence_hash: Sha256
    baseline_bundle_hash: Sha256
    candidate_bundle_hash: Sha256
    case_results: list[PrivateValidatorCaseResultV39]
    performance_case_count: Annotated[int, Field(ge=1)]
    quality_case_count: Annotated[int, Field(ge=1)]
    legacy_quality_abstention_count: Annotated[int, Field(ge=0)]
    recovered_quality_abstention_count: Annotated[int, Field(ge=0)]
    legacy_coverage: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    recovered_coverage: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    legacy_mean_adjudicated_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    recovered_mean_adjudicated_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    paired_mean_target_loss_improvement: Annotated[float, Field(allow_inf_nan=False)]
    paired_improvement_ci_lower: Annotated[float, Field(allow_inf_nan=False)]
    paired_improvement_ci_upper: Annotated[float, Field(allow_inf_nan=False)]
    recovered_mean_target_loss_by_mechanism: dict[
        Identifier, Annotated[float, Field(ge=0, allow_inf_nan=False)]
    ]
    legacy_action_fold_misuse_count: Annotated[int, Field(ge=0)]
    recovered_action_fold_contract_count: Annotated[int, Field(ge=0)]
    material_negative_transfer_count: Annotated[int, Field(ge=0)]
    material_negative_transfer_upper_95: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    gates: dict[Identifier, bool]
    ready_for_skeleton_factorial: bool
    superseded_scientific_claims: list[Identifier]
    status: Literal[
        "validator_input_contract_recovered_ready_for_skeleton_factorial_v39",
        "validator_input_contract_recovery_failed_v39",
    ]
    task_router_permitted: Literal[False] = False
    qualification_permitted: Literal[False] = False
    confirmation_permitted: Literal[False] = False
    real_world_authorization_permitted: Literal[False] = False
    created_at: datetime
    evolution_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "ValidatorRecoveryEvolutionReportV39":
        _assert_timezone(self.created_at, "created_at")
        ready = all(self.gates.values())
        expected = (
            "validator_input_contract_recovered_ready_for_skeleton_factorial_v39"
            if ready else "validator_input_contract_recovery_failed_v39"
        )
        if self.ready_for_skeleton_factorial != ready or self.status != expected:
            raise ValueError("V3.9 report status disagrees with gates")
        if self.evolution_hash and self.evolution_hash != self.content_hash():
            raise ValueError("evolution_hash does not match V3.9 report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evolution_hash")

    def assert_sealed(self) -> None:
        if not self.evolution_hash or self.evolution_hash != self.content_hash():
            raise ValueError("V3.9 evolution report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ValidatorRecoveryEvolutionReportV39":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evolution_hash"}),
            evolution_hash=draft.content_hash(),
        )


def _bootstrap_ci_v39(values: np.ndarray, spec) -> tuple[float, float]:
    random = np.random.default_rng(spec.bootstrap_seed)
    means = np.asarray([
        float(np.mean(values[random.integers(0, len(values), len(values))]))
        for _ in range(spec.bootstrap_replicates)
    ])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate_validator_recovery_v39(
    spec: ValidatorRecoveryWorldPackSpecV39,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    baseline: ValidatorBundleV39,
    candidate: ValidatorBundleV39,
    *,
    evaluated_at: datetime,
) -> ValidatorRecoveryEvolutionReportV39:
    for artifact in (spec, private_pack, baseline, candidate):
        artifact.assert_sealed()
    if (
        baseline.arm != "legacy_expanded_rows_as_segments"
        or candidate.arm != "action_hash_bound_segment_sequence"
    ):
        raise ValueError("V3.9 evaluator arm order differs")
    private_by_id = {item.public_case.case_id: item for item in private_pack.cases}
    baseline_by_id = {item.case_id: item for item in baseline.case_receipts}
    candidate_by_id = {item.case_id: item for item in candidate.case_receipts}
    results: list[PrivateValidatorCaseResultV39] = []
    quality_count = 0
    legacy_quality = 0
    recovered_quality = 0
    legacy_selected = 0
    recovered_selected = 0
    mechanism_losses: dict[str, list[float]] = defaultdict(list)
    legacy_misuse = 0
    recovered_contracts = 0
    for case_id, private_case in private_by_id.items():
        legacy = baseline_by_id[case_id]
        recovered = candidate_by_id[case_id]
        for challenge in legacy.challenges:
            legacy_misuse += sum(
                binding.observation_action_hash is not None
                and not binding.contract_valid
                for binding in challenge.fold_input_bindings
            )
        for challenge in recovered.challenges:
            recovered_contracts += sum(
                binding.observation_action_hash is not None
                and binding.contract_valid
                and binding.simulator_input_value_count == spec.segment_count
                for binding in challenge.fold_input_bindings
            )
        if not private_case.performance_eligible:
            quality_count += 1
            legacy_quality += int(legacy.decision.reason == "deny_data_quality")
            recovered_quality += int(recovered.decision.reason == "deny_data_quality")
            continue
        legacy_selected += int(legacy.selected_model is not None)
        recovered_selected += int(recovered.selected_model is not None)
        legacy_loss = (
            _target_loss_v31(private_case, legacy.selected_model, spec)
            if legacy.selected_model is not None
            else spec.unresolved_adjudicated_loss
        )
        recovered_loss = (
            _target_loss_v31(private_case, recovered.selected_model, spec)
            if recovered.selected_model is not None
            else spec.unresolved_adjudicated_loss
        )
        mechanism_losses[private_case.mechanism].append(recovered_loss)
        results.append(PrivateValidatorCaseResultV39(
            case_id=case_id,
            mechanism=private_case.mechanism,
            legacy_selected_family=legacy.decision.selected_family,
            recovered_selected_family=recovered.decision.selected_family,
            legacy_adjudicated_target_loss=legacy_loss,
            recovered_adjudicated_target_loss=recovered_loss,
            recovered_improvement=legacy_loss - recovered_loss,
            material_negative_transfer=(
                recovered_loss - legacy_loss > spec.material_negative_transfer
            ),
        ))
    count = len(results)
    legacy_losses = np.asarray([
        item.legacy_adjudicated_target_loss for item in results
    ])
    recovered_losses = np.asarray([
        item.recovered_adjudicated_target_loss for item in results
    ])
    improvements = legacy_losses - recovered_losses
    ci_lower, ci_upper = _bootstrap_ci_v39(improvements, spec)
    negatives = sum(item.material_negative_transfer for item in results)
    negative_upper = (
        float(beta.ppf(0.95, negatives + 1, count - negatives))
        if count > negatives else 1.0
    )
    mechanism_means = {
        mechanism: float(np.mean(values))
        for mechanism, values in mechanism_losses.items()
    }
    legacy_coverage = legacy_selected / count
    recovered_coverage = recovered_selected / count
    expected_action_folds = len(private_pack.cases) * len(FAMILIES_V37) * 2
    gates = {
        "legacy_fault_reproduced": legacy_misuse == expected_action_folds,
        "recovered_action_contract_complete": (
            recovered_contracts == expected_action_folds
        ),
        "quality_abstention_preserved": (
            legacy_quality == recovered_quality == quality_count
        ),
        "coverage_improvement": (
            recovered_coverage - legacy_coverage
            >= spec.minimum_coverage_improvement
        ),
        "paired_improvement_ci_lower_positive": ci_lower > 0.0,
        "skeleton_gap_remains_visible": (
            max(mechanism_means.values()) > spec.skeleton_gap_private_loss
        ),
        "no_real_world_execution": True,
    }
    ready = all(gates.values())
    return ValidatorRecoveryEvolutionReportV39.seal(
        evolution_id=f"evolution_{spec.experiment_id}",
        spec_hash=spec.spec_hash,
        bug_evidence_hash=spec.bug_evidence_hash,
        baseline_bundle_hash=baseline.bundle_hash,
        candidate_bundle_hash=candidate.bundle_hash,
        case_results=results,
        performance_case_count=count,
        quality_case_count=quality_count,
        legacy_quality_abstention_count=legacy_quality,
        recovered_quality_abstention_count=recovered_quality,
        legacy_coverage=legacy_coverage,
        recovered_coverage=recovered_coverage,
        legacy_mean_adjudicated_target_loss=float(np.mean(legacy_losses)),
        recovered_mean_adjudicated_target_loss=float(np.mean(recovered_losses)),
        paired_mean_target_loss_improvement=float(np.mean(improvements)),
        paired_improvement_ci_lower=ci_lower,
        paired_improvement_ci_upper=ci_upper,
        recovered_mean_target_loss_by_mechanism=mechanism_means,
        legacy_action_fold_misuse_count=legacy_misuse,
        recovered_action_fold_contract_count=recovered_contracts,
        material_negative_transfer_count=negatives,
        material_negative_transfer_upper_95=negative_upper,
        gates=gates,
        ready_for_skeleton_factorial=ready,
        superseded_scientific_claims=[
            "v37_no_eligible_family_due_to_predictive_challenge",
            "v371_acquire_target_discriminating_evidence_route",
            "v381_zero_resolved_after_one_extra_action",
        ],
        status=(
            "validator_input_contract_recovered_ready_for_skeleton_factorial_v39"
            if ready else "validator_input_contract_recovery_failed_v39"
        ),
        created_at=evaluated_at,
    )


class ValidatorRecoveryManifestV39(StrictModel):
    schema_version: Literal["3.9"] = "3.9"
    run_id: Identifier
    artifact_refs: list[ArtifactRef] = Field(min_length=8, max_length=8)
    terminal_status: Literal[
        "validator_input_contract_recovered_ready_for_skeleton_factorial_v39",
        "validator_input_contract_recovery_failed_v39",
    ]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "ValidatorRecoveryManifestV39":
        _assert_timezone(self.created_at, "created_at")
        if len({item.kind for item in self.artifact_refs}) != 8:
            raise ValueError("V3.9 manifest kinds differ")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match V3.9 manifest")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("V3.9 manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ValidatorRecoveryManifestV39":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class ValidatorRecoveryOutcomeV39:
    store: RunStore
    spec: ValidatorRecoveryWorldPackSpecV39
    private_pack: PrivateControlledDynamicsWorldPackV31
    baseline_bundle: ValidatorBundleV39
    candidate_bundle: ValidatorBundleV39
    evolution_report: ValidatorRecoveryEvolutionReportV39
    manifest: ValidatorRecoveryManifestV39


def run_validator_recovery_worldpack_v39(
    output_root: str | Path,
    *,
    source_v381_run_directory: str | Path,
    evidence: ValidatorInputBugEvidenceV39,
    baseline_policy: ValidatorInputContractPolicyV39,
    candidate_policy: ValidatorInputContractPolicyV39,
    spec: ValidatorRecoveryWorldPackSpecV39,
    evaluated_at: datetime | None = None,
    run_id: str | None = None,
) -> ValidatorRecoveryOutcomeV39:
    source = _load_source_v381_v39(source_v381_run_directory)
    for artifact in (evidence, baseline_policy, candidate_policy, spec):
        artifact.assert_sealed()
    if (
        source["evolution"].evolution_hash != evidence.source_v381_evolution_hash
        or source["acquisition_bundle"].bundle_hash
        != evidence.source_v381_candidate_acquisition_bundle_hash
    ):
        raise ValueError("V3.9 source binding differs")
    if (
        spec.bug_evidence_hash != evidence.evidence_hash
        or spec.baseline_policy_hash != baseline_policy.policy_hash
        or spec.candidate_policy_hash != candidate_policy.policy_hash
    ):
        raise ValueError("V3.9 frozen artifact binding differs")
    at = evaluated_at or datetime.now(timezone.utc)
    store = RunStore(
        output_root,
        run_id=run_id or f"validator-recovery-v39-{uuid4().hex[:10]}",
    )
    refs = [
        store.put_artifact("validator_input_bug_evidence_v39", evidence),
        store.put_artifact("legacy_validator_policy_v39", baseline_policy),
        store.put_artifact("recovered_validator_policy_v39", candidate_policy),
        store.put_artifact("validator_recovery_spec_v39", spec),
    ]
    store.emit("validator_recovery_v39_protocol_frozen_before_private_pack", {
        "spec_hash": spec.spec_hash,
        "bug_evidence_hash": evidence.evidence_hash,
        "source_v381_independently_verified": True,
    })
    private_pack = generate_private_controlled_dynamics_worldpack_v31(
        spec, generated_at=at
    )
    baseline = execute_validator_policy_v39(
        spec, private_pack, baseline_policy, executed_at=at
    )
    candidate = execute_validator_policy_v39(
        spec, private_pack, candidate_policy, executed_at=at
    )
    evolution = evaluate_validator_recovery_v39(
        spec, private_pack, baseline, candidate, evaluated_at=at
    )
    refs.extend([
        store.put_artifact("private_validator_recovery_worldpack_v39", private_pack),
        store.put_artifact("legacy_validator_bundle_v39", baseline),
        store.put_artifact("recovered_validator_bundle_v39", candidate),
        store.put_artifact("validator_recovery_evolution_report_v39", evolution),
    ])
    manifest = ValidatorRecoveryManifestV39.seal(
        run_id=store.run_id,
        artifact_refs=refs,
        terminal_status=evolution.status,
        created_at=at,
    )
    manifest_ref = store.put_artifact("validator_recovery_manifest_v39", manifest)
    store.emit("validator_recovery_v39_worldpack_adjudicated", {
        "manifest_ref": manifest_ref.model_dump(mode="json")
    })
    if not verify_validator_recovery_run_v39(store.run_directory):
        raise RuntimeError("V3.9 run failed independent verification")
    return ValidatorRecoveryOutcomeV39(
        store, spec, private_pack, baseline, candidate, evolution, manifest
    )


def verify_validator_recovery_run_v39(run_directory: str | Path) -> bool:
    try:
        store = RunStore.open_existing(run_directory)
        if not store.verify_event_chain():
            return False
        events = [
            json.loads(line)
            for line in store.event_path.read_text(encoding="utf-8").splitlines()
        ]
        refs = _committed_refs_v39(store)
        if len(refs) != 9:
            return False
        for ref in refs:
            store.load_artifact(ref)
        evidence = _load_one_v39(
            store, refs, "validator_input_bug_evidence_v39",
            ValidatorInputBugEvidenceV39,
        )
        baseline_policy = _load_one_v39(
            store, refs, "legacy_validator_policy_v39",
            ValidatorInputContractPolicyV39,
        )
        candidate_policy = _load_one_v39(
            store, refs, "recovered_validator_policy_v39",
            ValidatorInputContractPolicyV39,
        )
        spec = _load_one_v39(
            store, refs, "validator_recovery_spec_v39",
            ValidatorRecoveryWorldPackSpecV39,
        )
        private_pack = _load_one_v39(
            store, refs, "private_validator_recovery_worldpack_v39",
            PrivateControlledDynamicsWorldPackV31,
        )
        baseline = _load_one_v39(
            store, refs, "legacy_validator_bundle_v39", ValidatorBundleV39
        )
        candidate = _load_one_v39(
            store, refs, "recovered_validator_bundle_v39", ValidatorBundleV39
        )
        evolution = _load_one_v39(
            store, refs, "validator_recovery_evolution_report_v39",
            ValidatorRecoveryEvolutionReportV39,
        )
        manifest = _load_one_v39(
            store, refs, "validator_recovery_manifest_v39",
            ValidatorRecoveryManifestV39,
        )
        for artifact in (
            evidence, baseline_policy, candidate_policy, spec, private_pack,
            baseline, candidate, evolution, manifest,
        ):
            artifact.assert_sealed()
        if {item.kind for item in manifest.artifact_refs} != {
            "validator_input_bug_evidence_v39",
            "legacy_validator_policy_v39",
            "recovered_validator_policy_v39",
            "validator_recovery_spec_v39",
            "private_validator_recovery_worldpack_v39",
            "legacy_validator_bundle_v39",
            "recovered_validator_bundle_v39",
            "validator_recovery_evolution_report_v39",
        }:
            return False
        regenerated = generate_private_controlled_dynamics_worldpack_v31(
            spec, generated_at=private_pack.generated_at
        )
        if regenerated.pack_hash != private_pack.pack_hash:
            return False
        replay_baseline = execute_validator_policy_v39(
            spec, private_pack, baseline_policy, executed_at=baseline.created_at
        )
        replay_candidate = execute_validator_policy_v39(
            spec, private_pack, candidate_policy, executed_at=candidate.created_at
        )
        if (
            replay_baseline.bundle_hash != baseline.bundle_hash
            or replay_candidate.bundle_hash != candidate.bundle_hash
        ):
            return False
        recomputed = evaluate_validator_recovery_v39(
            spec, private_pack, baseline, candidate,
            evaluated_at=evolution.created_at,
        )
        if (
            recomputed.evolution_hash != evolution.evolution_hash
            or manifest.terminal_status != evolution.status
        ):
            return False
        if any(
            any(word in item.kind for word in (
                "qualification", "confirmation", "task_router"
            ))
            for item in manifest.artifact_refs
        ):
            return False
        freezes = [
            event for event in events
            if event["event_type"]
            == "validator_recovery_v39_protocol_frozen_before_private_pack"
        ]
        private_events = [
            event for event in events
            if event["event_type"] == "artifact_committed"
            and event["payload"]["kind"]
            == "private_validator_recovery_worldpack_v39"
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


CONFIRMATION_SEEDS_V391 = (
    24001, 24049, 24107, 24169, 24223, 24281, 24337, 24391,
    24443, 24499, 24551, 24611, 24659, 24709, 24767, 24821,
)


class ValidatorRecoveryWorldPackSpecV391(ValidatorRecoveryWorldPackSpecV39):
    schema_version: Literal["3.9.1"] = "3.9.1"
    seeds: list[int] = Field(min_length=16, max_length=16)
    bootstrap_seed: Literal[391722] = 391722
    evaluator_case_partition_rule: Literal[
        "public_observation_quality_flags_only"
    ] = "public_observation_quality_flags_only"

    @model_validator(mode="after")
    def validate_spec(self) -> "ValidatorRecoveryWorldPackSpecV391":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.mechanisms != list(MECHANISMS_V31):
            raise ValueError("V3.9.1 mechanism order differs")
        if self.seeds != list(CONFIRMATION_SEEDS_V391):
            raise ValueError("V3.9.1 seeds do not match frozen set")
        if self.family_catalog != list(FAMILIES_V37):
            raise ValueError("V3.9.1 family catalog differs")
        if self.observation_action_indices != [0, 7]:
            raise ValueError("V3.9.1 observation actions differ")
        if not math.isclose(
            (self.trajectory_points - 1) * self.time_step,
            self.segment_count * self.segment_duration,
            abs_tol=1e-12,
        ):
            raise ValueError("V3.9.1 segment coverage differs")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match V3.9.1 protocol")
        return self


def default_validator_recovery_spec_v391(
    *,
    evidence: ValidatorInputBugEvidenceV39,
    baseline_policy: ValidatorInputContractPolicyV39,
    candidate_policy: ValidatorInputContractPolicyV39,
    frozen_at: datetime | None = None,
) -> ValidatorRecoveryWorldPackSpecV391:
    for artifact in (evidence, baseline_policy, candidate_policy):
        artifact.assert_sealed()
    return ValidatorRecoveryWorldPackSpecV391.seal(
        experiment_id="validator_input_contract_recovery_v391",
        mechanisms=list(MECHANISMS_V31),
        seeds=list(CONFIRMATION_SEEDS_V391),
        family_catalog=list(FAMILIES_V37),
        observation_action_indices=[0, 7],
        baseline_policy_hash=baseline_policy.policy_hash,
        candidate_policy_hash=candidate_policy.policy_hash,
        method_evidence_hash=evidence.evidence_hash,
        bug_evidence_hash=evidence.evidence_hash,
        source_v381_evolution_hash=evidence.source_v381_evolution_hash,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


class ValidatorRecoveryEvolutionReportV391(ValidatorRecoveryEvolutionReportV39):
    schema_version: Literal["3.9.1"] = "3.9.1"
    evaluator_case_partition_rule: Literal[
        "public_observation_quality_flags_only"
    ] = "public_observation_quality_flags_only"
    private_performance_eligible_used_for_partition: Literal[False] = False
    status: Literal[
        "validator_input_contract_recovered_ready_for_skeleton_factorial_v391",
        "validator_input_contract_recovery_failed_v391",
    ]

    @model_validator(mode="after")
    def validate_report(self) -> "ValidatorRecoveryEvolutionReportV391":
        _assert_timezone(self.created_at, "created_at")
        ready = all(self.gates.values())
        expected = (
            "validator_input_contract_recovered_ready_for_skeleton_factorial_v391"
            if ready else "validator_input_contract_recovery_failed_v391"
        )
        if self.ready_for_skeleton_factorial != ready or self.status != expected:
            raise ValueError("V3.9.1 report status disagrees with gates")
        if self.evolution_hash and self.evolution_hash != self.content_hash():
            raise ValueError("evolution_hash does not match V3.9.1 report")
        return self


def evaluate_validator_recovery_v391(
    spec: ValidatorRecoveryWorldPackSpecV391,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    baseline: ValidatorBundleV39,
    candidate: ValidatorBundleV39,
    *,
    evaluated_at: datetime,
) -> ValidatorRecoveryEvolutionReportV391:
    for artifact in (spec, private_pack, baseline, candidate):
        artifact.assert_sealed()
    if (
        baseline.arm != "legacy_expanded_rows_as_segments"
        or candidate.arm != "action_hash_bound_segment_sequence"
    ):
        raise ValueError("V3.9.1 evaluator arm order differs")
    private_by_id = {item.public_case.case_id: item for item in private_pack.cases}
    baseline_by_id = {item.case_id: item for item in baseline.case_receipts}
    candidate_by_id = {item.case_id: item for item in candidate.case_receipts}
    results: list[PrivateValidatorCaseResultV39] = []
    quality_count = 0
    legacy_quality = 0
    recovered_quality = 0
    legacy_selected = 0
    recovered_selected = 0
    mechanism_losses: dict[str, list[float]] = defaultdict(list)
    legacy_misuse = 0
    recovered_contracts = 0
    public_partition_complete = True
    for case_id, private_case in private_by_id.items():
        legacy = baseline_by_id[case_id]
        recovered = candidate_by_id[case_id]
        if legacy.quality_flags != recovered.quality_flags:
            public_partition_complete = False
        for challenge in legacy.challenges:
            legacy_misuse += sum(
                binding.observation_action_hash is not None
                and not binding.contract_valid
                for binding in challenge.fold_input_bindings
            )
        for challenge in recovered.challenges:
            recovered_contracts += sum(
                binding.observation_action_hash is not None
                and binding.contract_valid
                and binding.simulator_input_value_count == spec.segment_count
                for binding in challenge.fold_input_bindings
            )
        is_quality_case = bool(legacy.quality_flags)
        if is_quality_case:
            quality_count += 1
            legacy_quality += int(legacy.decision.reason == "deny_data_quality")
            recovered_quality += int(recovered.decision.reason == "deny_data_quality")
            continue
        legacy_selected += int(legacy.selected_model is not None)
        recovered_selected += int(recovered.selected_model is not None)
        legacy_loss = (
            _target_loss_v31(private_case, legacy.selected_model, spec)
            if legacy.selected_model is not None
            else spec.unresolved_adjudicated_loss
        )
        recovered_loss = (
            _target_loss_v31(private_case, recovered.selected_model, spec)
            if recovered.selected_model is not None
            else spec.unresolved_adjudicated_loss
        )
        mechanism_losses[private_case.mechanism].append(recovered_loss)
        results.append(PrivateValidatorCaseResultV39(
            case_id=case_id,
            mechanism=private_case.mechanism,
            legacy_selected_family=legacy.decision.selected_family,
            recovered_selected_family=recovered.decision.selected_family,
            legacy_adjudicated_target_loss=legacy_loss,
            recovered_adjudicated_target_loss=recovered_loss,
            recovered_improvement=legacy_loss - recovered_loss,
            material_negative_transfer=(
                recovered_loss - legacy_loss > spec.material_negative_transfer
            ),
        ))
    count = len(results)
    legacy_losses = np.asarray([
        item.legacy_adjudicated_target_loss for item in results
    ])
    recovered_losses = np.asarray([
        item.recovered_adjudicated_target_loss for item in results
    ])
    improvements = legacy_losses - recovered_losses
    ci_lower, ci_upper = _bootstrap_ci_v39(improvements, spec)
    negatives = sum(item.material_negative_transfer for item in results)
    negative_upper = (
        float(beta.ppf(0.95, negatives + 1, count - negatives))
        if count > negatives else 1.0
    )
    mechanism_means = {
        mechanism: float(np.mean(values))
        for mechanism, values in mechanism_losses.items()
    }
    legacy_coverage = legacy_selected / count
    recovered_coverage = recovered_selected / count
    expected_action_folds = len(private_pack.cases) * len(FAMILIES_V37) * 2
    gates = {
        "public_quality_partition_complete": (
            public_partition_complete and quality_count == spec.expected_quality_abstention_count
        ),
        "legacy_fault_reproduced": legacy_misuse == expected_action_folds,
        "recovered_action_contract_complete": (
            recovered_contracts == expected_action_folds
        ),
        "quality_abstention_preserved": (
            legacy_quality == recovered_quality == quality_count
        ),
        "coverage_improvement": (
            recovered_coverage - legacy_coverage
            >= spec.minimum_coverage_improvement
        ),
        "paired_improvement_ci_lower_positive": ci_lower > 0.0,
        "skeleton_gap_remains_visible": (
            max(mechanism_means.values()) > spec.skeleton_gap_private_loss
        ),
        "no_real_world_execution": True,
    }
    ready = all(gates.values())
    return ValidatorRecoveryEvolutionReportV391.seal(
        evolution_id=f"evolution_{spec.experiment_id}",
        spec_hash=spec.spec_hash,
        bug_evidence_hash=spec.bug_evidence_hash,
        baseline_bundle_hash=baseline.bundle_hash,
        candidate_bundle_hash=candidate.bundle_hash,
        case_results=results,
        performance_case_count=count,
        quality_case_count=quality_count,
        legacy_quality_abstention_count=legacy_quality,
        recovered_quality_abstention_count=recovered_quality,
        legacy_coverage=legacy_coverage,
        recovered_coverage=recovered_coverage,
        legacy_mean_adjudicated_target_loss=float(np.mean(legacy_losses)),
        recovered_mean_adjudicated_target_loss=float(np.mean(recovered_losses)),
        paired_mean_target_loss_improvement=float(np.mean(improvements)),
        paired_improvement_ci_lower=ci_lower,
        paired_improvement_ci_upper=ci_upper,
        recovered_mean_target_loss_by_mechanism=mechanism_means,
        legacy_action_fold_misuse_count=legacy_misuse,
        recovered_action_fold_contract_count=recovered_contracts,
        material_negative_transfer_count=negatives,
        material_negative_transfer_upper_95=negative_upper,
        gates=gates,
        ready_for_skeleton_factorial=ready,
        superseded_scientific_claims=[
            "v37_no_eligible_family_due_to_predictive_challenge",
            "v371_acquire_target_discriminating_evidence_route",
            "v381_zero_resolved_after_one_extra_action",
            "v39_private_performance_eligible_case_partition",
        ],
        status=(
            "validator_input_contract_recovered_ready_for_skeleton_factorial_v391"
            if ready else "validator_input_contract_recovery_failed_v391"
        ),
        created_at=evaluated_at,
    )


class ValidatorRecoveryManifestV391(ValidatorRecoveryManifestV39):
    schema_version: Literal["3.9.1"] = "3.9.1"
    terminal_status: Literal[
        "validator_input_contract_recovered_ready_for_skeleton_factorial_v391",
        "validator_input_contract_recovery_failed_v391",
    ]

    @model_validator(mode="after")
    def validate_manifest(self) -> "ValidatorRecoveryManifestV391":
        _assert_timezone(self.created_at, "created_at")
        if len({item.kind for item in self.artifact_refs}) != 8:
            raise ValueError("V3.9.1 manifest kinds differ")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match V3.9.1 manifest")
        return self


@dataclass(frozen=True)
class ValidatorRecoveryOutcomeV391:
    store: RunStore
    spec: ValidatorRecoveryWorldPackSpecV391
    private_pack: PrivateControlledDynamicsWorldPackV31
    baseline_bundle: ValidatorBundleV39
    candidate_bundle: ValidatorBundleV39
    evolution_report: ValidatorRecoveryEvolutionReportV391
    manifest: ValidatorRecoveryManifestV391


def run_validator_recovery_worldpack_v391(
    output_root: str | Path,
    *,
    source_v381_run_directory: str | Path,
    evidence: ValidatorInputBugEvidenceV39,
    baseline_policy: ValidatorInputContractPolicyV39,
    candidate_policy: ValidatorInputContractPolicyV39,
    spec: ValidatorRecoveryWorldPackSpecV391,
    evaluated_at: datetime | None = None,
    run_id: str | None = None,
) -> ValidatorRecoveryOutcomeV391:
    source = _load_source_v381_v39(source_v381_run_directory)
    for artifact in (evidence, baseline_policy, candidate_policy, spec):
        artifact.assert_sealed()
    if (
        source["evolution"].evolution_hash != evidence.source_v381_evolution_hash
        or source["acquisition_bundle"].bundle_hash
        != evidence.source_v381_candidate_acquisition_bundle_hash
    ):
        raise ValueError("V3.9.1 source binding differs")
    if (
        spec.bug_evidence_hash != evidence.evidence_hash
        or spec.baseline_policy_hash != baseline_policy.policy_hash
        or spec.candidate_policy_hash != candidate_policy.policy_hash
    ):
        raise ValueError("V3.9.1 frozen artifact binding differs")
    at = evaluated_at or datetime.now(timezone.utc)
    store = RunStore(
        output_root,
        run_id=run_id or f"validator-recovery-v391-{uuid4().hex[:10]}",
    )
    refs = [
        store.put_artifact("validator_input_bug_evidence_v391", evidence),
        store.put_artifact("legacy_validator_policy_v391", baseline_policy),
        store.put_artifact("recovered_validator_policy_v391", candidate_policy),
        store.put_artifact("validator_recovery_spec_v391", spec),
    ]
    store.emit("validator_recovery_v391_protocol_frozen_before_private_pack", {
        "spec_hash": spec.spec_hash,
        "bug_evidence_hash": evidence.evidence_hash,
        "case_partition_rule": spec.evaluator_case_partition_rule,
        "source_v381_independently_verified": True,
    })
    private_pack = generate_private_controlled_dynamics_worldpack_v31(
        spec, generated_at=at
    )
    baseline = execute_validator_policy_v39(
        spec, private_pack, baseline_policy, executed_at=at
    )
    candidate = execute_validator_policy_v39(
        spec, private_pack, candidate_policy, executed_at=at
    )
    evolution = evaluate_validator_recovery_v391(
        spec, private_pack, baseline, candidate, evaluated_at=at
    )
    refs.extend([
        store.put_artifact("private_validator_recovery_worldpack_v391", private_pack),
        store.put_artifact("legacy_validator_bundle_v391", baseline),
        store.put_artifact("recovered_validator_bundle_v391", candidate),
        store.put_artifact("validator_recovery_evolution_report_v391", evolution),
    ])
    manifest = ValidatorRecoveryManifestV391.seal(
        run_id=store.run_id,
        artifact_refs=refs,
        terminal_status=evolution.status,
        created_at=at,
    )
    manifest_ref = store.put_artifact("validator_recovery_manifest_v391", manifest)
    store.emit("validator_recovery_v391_worldpack_adjudicated", {
        "manifest_ref": manifest_ref.model_dump(mode="json")
    })
    if not verify_validator_recovery_run_v391(store.run_directory):
        raise RuntimeError("V3.9.1 run failed independent verification")
    return ValidatorRecoveryOutcomeV391(
        store, spec, private_pack, baseline, candidate, evolution, manifest
    )


def verify_validator_recovery_run_v391(run_directory: str | Path) -> bool:
    try:
        store = RunStore.open_existing(run_directory)
        if not store.verify_event_chain():
            return False
        events = [
            json.loads(line)
            for line in store.event_path.read_text(encoding="utf-8").splitlines()
        ]
        refs = _committed_refs_v39(store)
        if len(refs) != 9:
            return False
        for ref in refs:
            store.load_artifact(ref)
        evidence = _load_one_v39(
            store, refs, "validator_input_bug_evidence_v391",
            ValidatorInputBugEvidenceV39,
        )
        baseline_policy = _load_one_v39(
            store, refs, "legacy_validator_policy_v391",
            ValidatorInputContractPolicyV39,
        )
        candidate_policy = _load_one_v39(
            store, refs, "recovered_validator_policy_v391",
            ValidatorInputContractPolicyV39,
        )
        spec = _load_one_v39(
            store, refs, "validator_recovery_spec_v391",
            ValidatorRecoveryWorldPackSpecV391,
        )
        private_pack = _load_one_v39(
            store, refs, "private_validator_recovery_worldpack_v391",
            PrivateControlledDynamicsWorldPackV31,
        )
        baseline = _load_one_v39(
            store, refs, "legacy_validator_bundle_v391", ValidatorBundleV39
        )
        candidate = _load_one_v39(
            store, refs, "recovered_validator_bundle_v391", ValidatorBundleV39
        )
        evolution = _load_one_v39(
            store, refs, "validator_recovery_evolution_report_v391",
            ValidatorRecoveryEvolutionReportV391,
        )
        manifest = _load_one_v39(
            store, refs, "validator_recovery_manifest_v391",
            ValidatorRecoveryManifestV391,
        )
        for artifact in (
            evidence, baseline_policy, candidate_policy, spec, private_pack,
            baseline, candidate, evolution, manifest,
        ):
            artifact.assert_sealed()
        if {item.kind for item in manifest.artifact_refs} != {
            "validator_input_bug_evidence_v391",
            "legacy_validator_policy_v391",
            "recovered_validator_policy_v391",
            "validator_recovery_spec_v391",
            "private_validator_recovery_worldpack_v391",
            "legacy_validator_bundle_v391",
            "recovered_validator_bundle_v391",
            "validator_recovery_evolution_report_v391",
        }:
            return False
        regenerated = generate_private_controlled_dynamics_worldpack_v31(
            spec, generated_at=private_pack.generated_at
        )
        if regenerated.pack_hash != private_pack.pack_hash:
            return False
        replay_baseline = execute_validator_policy_v39(
            spec, private_pack, baseline_policy, executed_at=baseline.created_at
        )
        replay_candidate = execute_validator_policy_v39(
            spec, private_pack, candidate_policy, executed_at=candidate.created_at
        )
        if (
            replay_baseline.bundle_hash != baseline.bundle_hash
            or replay_candidate.bundle_hash != candidate.bundle_hash
        ):
            return False
        recomputed = evaluate_validator_recovery_v391(
            spec, private_pack, baseline, candidate,
            evaluated_at=evolution.created_at,
        )
        if (
            recomputed.evolution_hash != evolution.evolution_hash
            or manifest.terminal_status != evolution.status
        ):
            return False
        freezes = [
            event for event in events
            if event["event_type"]
            == "validator_recovery_v391_protocol_frozen_before_private_pack"
        ]
        private_events = [
            event for event in events
            if event["event_type"] == "artifact_committed"
            and event["payload"]["kind"]
            == "private_validator_recovery_worldpack_v391"
        ]
        return (
            len(freezes) == 1
            and len(private_events) == 1
            and freezes[0]["sequence"] < private_events[0]["sequence"]
            and not evolution.task_router_permitted
            and not evolution.qualification_permitted
            and not evolution.confirmation_permitted
            and not evolution.real_world_authorization_permitted
            and store.verify_event_chain()
        )
    except (
        OSError, RuntimeError, ValueError, KeyError, TypeError,
        json.JSONDecodeError, FloatingPointError, AttributeError,
    ):
        return False
