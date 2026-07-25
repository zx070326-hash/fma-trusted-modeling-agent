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
from scipy.stats import beta

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore
from fma.v2.dynamics_ir import trajectory_nrmse
from fma.v2.schemas import Identifier, Sha256, _assert_timezone

from .challenge_disposition_v371 import (
    ChallengeDispositionBundleV371,
    ChallengeDispositionPolicyV371,
    ModelChallengeWorldPackSpecV371,
    create_challenge_disposition_bundle_v371,
)
from .controlled_dynamics_loop import (
    MECHANISMS_V31,
    ControlledDriftModelV31,
    ControlledObservationReceiptV31,
    DecisionTargetV31,
    PiecewiseConstantInputActionV31,
    PrivateControlledDynamicsCaseV31,
    PrivateControlledDynamicsWorldPackV31,
    _input_at_time_v31,
    _simulate_model_v31,
    _target_loss_v31,
    generate_private_controlled_dynamics_worldpack_v31,
    validate_action_against_envelope_v31,
)
from .model_challenge_v37 import (
    FAMILY_DEGREES_V37,
    FAMILIES_V37,
    FamilyV37,
    ModelChallengeBundleV37,
    ModelChallengeMethodEvidenceV37,
    ModelPortfolioPolicyV37,
    _fit_family_v37,
    _hash_without,
    _shared_observations_v37,
    execute_model_challenge_policy_v37,
)
from .target_clarification_v38 import (
    TargetClarificationBundleV38,
    TargetClarificationEvolutionReportV38,
    TargetClarificationPolicyV38,
    TargetClarificationWorldPackSpecV38,
    execute_target_clarifications_v38,
    verify_target_clarification_run_v38,
)


CONFIRMATION_SEEDS_V381 = (
    22003, 22051, 22109, 22157, 22213, 22271, 22307, 22369,
    22409, 22469, 22531, 22571, 22639, 22691, 22751, 22787,
)

AcquisitionArmV381 = Literal[
    "deterministic_random_safe_action",
    "maximum_portfolio_trajectory_disagreement",
]
AcquisitionRecoveryV381 = Literal[
    "run_fresh_confirmation",
    "stop_repeat_acquisition_reclassify_estimator_or_family",
]


def _committed_refs(store: RunStore) -> list[ArtifactRef]:
    events = [
        json.loads(line)
        for line in store.event_path.read_text(encoding="utf-8").splitlines()
    ]
    return [
        ArtifactRef.model_validate(event["payload"])
        for event in events if event["event_type"] == "artifact_committed"
    ]


def _load_one(store: RunStore, refs: list[ArtifactRef], kind: str, model):
    matches = [item for item in refs if item.kind == kind]
    if len(matches) != 1:
        raise ValueError(f"V3.8.1 requires exactly one {kind}")
    return model.model_validate(store.load_artifact(matches[0]))


def _load_source_v38(source: str | Path) -> dict[str, object]:
    if not verify_target_clarification_run_v38(source):
        raise ValueError("V3.8.1 source V3.8 run did not independently verify")
    store = RunStore.open_existing(source)
    refs = _committed_refs(store)
    return {
        "method": _load_one(
            store, refs, "model_challenge_method_evidence_v38",
            ModelChallengeMethodEvidenceV37,
        ),
        "baseline_policy": _load_one(
            store, refs, "model_challenge_baseline_policy_v38",
            ModelPortfolioPolicyV37,
        ),
        "candidate_policy": _load_one(
            store, refs, "model_challenge_candidate_policy_v38",
            ModelPortfolioPolicyV37,
        ),
        "disposition_policy": _load_one(
            store, refs, "challenge_disposition_policy_v38",
            ChallengeDispositionPolicyV371,
        ),
        "clarification_policy": _load_one(
            store, refs, "target_clarification_policy_v38",
            TargetClarificationPolicyV38,
        ),
        "spec": _load_one(
            store, refs, "target_clarification_spec_v38",
            TargetClarificationWorldPackSpecV38,
        ),
        "private_pack": _load_one(
            store, refs, "private_target_clarification_worldpack_v38",
            PrivateControlledDynamicsWorldPackV31,
        ),
        "candidate_bundle": _load_one(
            store, refs, "model_challenge_candidate_bundle_v38",
            ModelChallengeBundleV37,
        ),
        "disposition_bundle": _load_one(
            store, refs, "challenge_disposition_bundle_v38",
            ChallengeDispositionBundleV371,
        ),
        "clarification_bundle": _load_one(
            store, refs, "target_clarification_bundle_v38",
            TargetClarificationBundleV38,
        ),
        "evolution": _load_one(
            store, refs, "target_clarification_evolution_report_v38",
            TargetClarificationEvolutionReportV38,
        ),
    }


class ActionTrajectoryScoreV381(StrictModel):
    schema_version: Literal["3.8.1"] = "3.8.1"
    action_id: Identifier
    action_hash: Sha256
    successful_model_simulations: Annotated[int, Field(ge=0, le=3)]
    normalized_trajectory_disagreement: Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ]
    empirical_prediction_risk: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    harness_validation_codes: list[Identifier]
    admissible: bool
    private_mechanism_seen: Literal[False] = False
    private_observation_seen: Literal[False] = False
    private_probe_seen: Literal[False] = False
    private_target_loss_seen: Literal[False] = False
    score_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_score(self) -> "ActionTrajectoryScoreV381":
        expected = (
            self.successful_model_simulations >= 2
            and self.empirical_prediction_risk <= 0.25
            and not self.harness_validation_codes
        )
        if self.admissible != expected:
            raise ValueError("V3.8.1 action admissibility does not recompute")
        if self.score_hash and self.score_hash != self.content_hash():
            raise ValueError("score_hash does not match V3.8.1 action score")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "score_hash")

    def assert_sealed(self) -> None:
        if not self.score_hash or self.score_hash != self.content_hash():
            raise ValueError("V3.8.1 action score is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ActionTrajectoryScoreV381":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"score_hash"}),
            score_hash=draft.content_hash(),
        )


def _available_actions_v381(
    private_case: PrivateControlledDynamicsCaseV31,
    spec,
) -> list[PiecewiseConstantInputActionV31]:
    return [
        action for index, action in enumerate(private_case.public_case.action_catalog)
        if index not in spec.observation_action_indices
    ]


def _action_input_rows_v381(action, times: list[float], spec) -> list[list[float]]:
    return [
        _input_at_time_v31(
            action.input_values, time, spec.segment_duration
        ).tolist()
        for time in times
    ]


def _portfolio_action_scores_v381(
    private_case: PrivateControlledDynamicsCaseV31,
    observations: list[object],
    spec,
) -> tuple[list[ControlledDriftModelV31], list[ActionTrajectoryScoreV381]]:
    models = [
        _fit_family_v37(
            private_case, observations, family, spec,
            model_suffix="pre_acquisition_v381",
        )
        for family in FAMILIES_V37
    ]
    public = private_case.public_case
    lower = np.asarray(public.envelope.state_lower_bounds, dtype=float)
    upper = np.asarray(public.envelope.state_upper_bounds, dtype=float)
    scale = np.maximum(upper - lower, 1e-6)
    times = public.pilot.times
    scores: list[ActionTrajectoryScoreV381] = []
    for action in _available_actions_v381(private_case, spec):
        failures = validate_action_against_envelope_v31(
            action, public.actuator, public.envelope
        )
        paths: list[np.ndarray] = []
        for model in models:
            try:
                paths.append(np.asarray(_simulate_model_v31(
                    model,
                    public.actuator,
                    public.initial_state,
                    times,
                    _action_input_rows_v381(action, times, spec),
                    spec.segment_duration,
                )))
            except (RuntimeError, FloatingPointError, ValueError):
                continue
        if paths:
            array = np.asarray(paths)
            risk = float(np.mean(np.any(
                (array < lower) | (array > upper), axis=(1, 2)
            )))
        else:
            risk = 1.0
        disagreement = (
            float(np.mean(np.var(np.asarray(paths) / scale, axis=0)))
            if len(paths) >= 2 else 0.0
        )
        scores.append(ActionTrajectoryScoreV381.seal(
            action_id=action.action_id,
            action_hash=action.action_hash,
            successful_model_simulations=len(paths),
            normalized_trajectory_disagreement=disagreement,
            empirical_prediction_risk=risk,
            harness_validation_codes=failures,
            admissible=(len(paths) >= 2 and risk <= 0.25 and not failures),
        ))
    return models, scores


def _choose_action_v381(
    private_case: PrivateControlledDynamicsCaseV31,
    observations: list[object],
    spec,
    arm: AcquisitionArmV381,
    random_binding: str,
) -> tuple[PiecewiseConstantInputActionV31 | None, list[ControlledDriftModelV31], list[ActionTrajectoryScoreV381]]:
    available = _available_actions_v381(private_case, spec)
    if arm == "deterministic_random_safe_action":
        valid = [
            item for item in available
            if not validate_action_against_envelope_v31(
                item, private_case.public_case.actuator,
                private_case.public_case.envelope,
            )
        ]
        if not valid:
            return None, [], []
        index = int(sha256_value([
            private_case.public_case.case_id, random_binding
        ])[:16], 16) % len(valid)
        return valid[index], [], []
    models, scores = _portfolio_action_scores_v381(
        private_case, observations, spec
    )
    eligible = [item for item in scores if item.admissible]
    if not eligible:
        return None, models, scores
    chosen_score = max(
        eligible,
        key=lambda item: (
            item.normalized_trajectory_disagreement,
            item.action_id,
        ),
    )
    chosen = next(
        item for item in available if item.action_hash == chosen_score.action_hash
    )
    return chosen, models, scores


class PostAcquisitionFamilyChallengeV381(StrictModel):
    schema_version: Literal["3.8.1"] = "3.8.1"
    challenge_id: Identifier
    case_id: Identifier
    family: FamilyV37
    decision_target: DecisionTargetV31
    source_observation_hashes: list[Sha256] = Field(min_length=4, max_length=4)
    relevant_holdout_hashes: list[Sha256] = Field(min_length=1, max_length=3)
    relevant_fold_losses: list[
        Annotated[float, Field(ge=0, allow_inf_nan=False)]
    ] = Field(min_length=1, max_length=3)
    target_cv_mean: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    target_cv_standard_error: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    normalized_rank_ratio: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    normalized_condition_number: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    basis_term_count: Annotated[int, Field(ge=1)]
    minimum_rank_ratio: Literal[0.95] = 0.95
    maximum_condition_number: Literal[100000000.0] = 100000000.0
    maximum_cv_prediction_loss: Literal[0.35] = 0.35
    eligible: bool
    final_model_hash: Sha256
    challenge_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_challenge(self) -> "PostAcquisitionFamilyChallengeV381":
        expected_count = 1 if self.decision_target == "free_run_prediction" else 3
        if (
            len(self.relevant_holdout_hashes) != expected_count
            or len(self.relevant_fold_losses) != expected_count
        ):
            raise ValueError("V3.8.1 target holdout count differs")
        values = np.asarray(self.relevant_fold_losses, dtype=float)
        mean = float(np.mean(values))
        se = (
            float(np.std(values, ddof=1) / math.sqrt(len(values)))
            if len(values) > 1 else 0.0
        )
        if not math.isclose(self.target_cv_mean, mean, abs_tol=1e-12):
            raise ValueError("V3.8.1 target CV mean does not recompute")
        if not math.isclose(self.target_cv_standard_error, se, abs_tol=1e-12):
            raise ValueError("V3.8.1 target CV SE does not recompute")
        expected = (
            all(value < 9.999 for value in values)
            and self.normalized_rank_ratio >= self.minimum_rank_ratio
            and self.normalized_condition_number <= self.maximum_condition_number
            and mean <= self.maximum_cv_prediction_loss
        )
        if self.eligible != expected:
            raise ValueError("V3.8.1 target eligibility does not recompute")
        if self.challenge_hash and self.challenge_hash != self.content_hash():
            raise ValueError("challenge_hash does not match V3.8.1 challenge")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "challenge_hash")

    def assert_sealed(self) -> None:
        if not self.challenge_hash or self.challenge_hash != self.content_hash():
            raise ValueError("V3.8.1 post-acquisition challenge is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PostAcquisitionFamilyChallengeV381":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"challenge_hash"}),
            challenge_hash=draft.content_hash(),
        )


class PostAcquisitionDecisionV381(StrictModel):
    schema_version: Literal["3.8.1"] = "3.8.1"
    decision_id: Identifier
    case_id: Identifier
    challenge_hashes: list[Sha256] = Field(min_length=3, max_length=3)
    decision: Literal["select", "needs_evidence"]
    selected_family: FamilyV37 | None
    best_family: FamilyV37 | None
    one_standard_error_threshold: Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ] | None
    decision_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> "PostAcquisitionDecisionV381":
        if (self.decision == "select") != (self.selected_family is not None):
            raise ValueError("V3.8.1 decision/family binding differs")
        if self.decision_hash and self.decision_hash != self.content_hash():
            raise ValueError("decision_hash does not match V3.8.1 decision")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "decision_hash")

    def assert_sealed(self) -> None:
        if not self.decision_hash or self.decision_hash != self.content_hash():
            raise ValueError("V3.8.1 post-acquisition decision is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PostAcquisitionDecisionV381":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"decision_hash"}),
            decision_hash=draft.content_hash(),
        )


def _post_acquisition_challenge_v381(
    private_case: PrivateControlledDynamicsCaseV31,
    observations: list[object],
    target: DecisionTargetV31,
    spec,
) -> tuple[
    list[PostAcquisitionFamilyChallengeV381],
    PostAcquisitionDecisionV381,
    ControlledDriftModelV31 | None,
]:
    if len(observations) != 4:
        raise ValueError("V3.8.1 post-acquisition challenge needs four observations")
    challenges: list[PostAcquisitionFamilyChallengeV381] = []
    models: dict[str, ControlledDriftModelV31] = {}
    for family in FAMILIES_V37:
        if target == "free_run_prediction":
            folds = [(observations[1:], observations[0])]
        else:
            action_observations = observations[1:]
            folds = [
                (
                    [observations[0]] + [
                        item for index, item in enumerate(action_observations)
                        if index != holdout_index
                    ],
                    action_observations[holdout_index],
                )
                for holdout_index in range(3)
            ]
        losses: list[float] = []
        holdout_hashes: list[str] = []
        for fold_index, (training, holdout) in enumerate(folds):
            model = _fit_family_v37(
                private_case,
                training,
                family,
                spec,
                model_suffix=f"post_acquisition_fold{fold_index}_v381",
            )
            holdout_hashes.append(holdout.observation_hash)
            try:
                predicted = _simulate_model_v31(
                    model,
                    private_case.public_case.actuator,
                    holdout.states[0],
                    holdout.times,
                    holdout.inputs,
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
            model_suffix="post_acquisition_all_v381",
        )
        models[family] = final_model
        basis_count = len(final_model.basis_terms)
        rank_ratio = final_model.normalized_design_rank / basis_count
        values = np.asarray(losses, dtype=float)
        mean = float(np.mean(values))
        se = (
            float(np.std(values, ddof=1) / math.sqrt(len(values)))
            if len(values) > 1 else 0.0
        )
        eligible = (
            all(value < 9.999 for value in values)
            and rank_ratio >= spec.minimum_rank_ratio
            and final_model.normalized_condition_number <= spec.maximum_condition_number
            and mean <= spec.maximum_cv_prediction_loss
        )
        challenges.append(PostAcquisitionFamilyChallengeV381.seal(
            challenge_id=(
                f"post_acquisition_{private_case.public_case.case_id}_{family}"
            ),
            case_id=private_case.public_case.case_id,
            family=family,
            decision_target=target,
            source_observation_hashes=[
                item.observation_hash for item in observations
            ],
            relevant_holdout_hashes=holdout_hashes,
            relevant_fold_losses=losses,
            target_cv_mean=mean,
            target_cv_standard_error=se,
            normalized_rank_ratio=rank_ratio,
            normalized_condition_number=final_model.normalized_condition_number,
            basis_term_count=basis_count,
            eligible=eligible,
            final_model_hash=final_model.model_hash,
        ))
    eligible = [item for item in challenges if item.eligible]
    common = {
        "decision_id": f"post_acquisition_decision_{private_case.public_case.case_id}",
        "case_id": private_case.public_case.case_id,
        "challenge_hashes": [item.challenge_hash for item in challenges],
    }
    if not eligible:
        decision = PostAcquisitionDecisionV381.seal(
            decision="needs_evidence",
            selected_family=None,
            best_family=None,
            one_standard_error_threshold=None,
            **common,
        )
        return challenges, decision, None
    best = min(eligible, key=lambda item: (
        item.target_cv_mean, item.basis_term_count
    ))
    threshold = best.target_cv_mean + best.target_cv_standard_error
    within = [item for item in eligible if item.target_cv_mean <= threshold]
    selected = min(within, key=lambda item: (
        item.basis_term_count, item.target_cv_mean
    ))
    decision = PostAcquisitionDecisionV381.seal(
        decision="select",
        selected_family=selected.family,
        best_family=best.family,
        one_standard_error_threshold=threshold,
        **common,
    )
    return challenges, decision, models[selected.family]


class AcquisitionTrainingEvidenceV381(StrictModel):
    schema_version: Literal["3.8.1"] = "3.8.1"
    evidence_id: Identifier
    source_v38_evolution_hash: Sha256
    source_v38_clarification_bundle_hash: Sha256
    source_acquisition_case_count: Literal[21] = 21
    random_one_action_resolved_count: Literal[0] = 0
    disagreement_one_action_resolved_count: Literal[0] = 0
    conclusion: Literal[
        "one_extra_action_did_not_resolve_training_cases"
    ] = "one_extra_action_did_not_resolve_training_cases"
    protocol_effect_guaranteed: Literal[False] = False
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> "AcquisitionTrainingEvidenceV381":
        if self.evidence_hash and self.evidence_hash != self.content_hash():
            raise ValueError("evidence_hash does not match V3.8.1 training evidence")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evidence_hash")

    def assert_sealed(self) -> None:
        if not self.evidence_hash or self.evidence_hash != self.content_hash():
            raise ValueError("V3.8.1 training evidence is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "AcquisitionTrainingEvidenceV381":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evidence_hash"}),
            evidence_hash=draft.content_hash(),
        )


def build_acquisition_training_evidence_v381(
    source_v38_run_directory: str | Path,
) -> AcquisitionTrainingEvidenceV381:
    source = _load_source_v38(source_v38_run_directory)
    spec = source["spec"]
    private_pack = source["private_pack"]
    clarifications = source["clarification_bundle"]
    clarification_by_id = {
        item.case_id: item for item in clarifications.case_receipts
    }
    count = 0
    resolved = {
        "deterministic_random_safe_action": 0,
        "maximum_portfolio_trajectory_disagreement": 0,
    }
    for private_case in private_pack.cases:
        clarification = clarification_by_id[private_case.public_case.case_id]
        if clarification.next_action != "acquire_target_discriminating_evidence":
            continue
        count += 1
        observations = _shared_observations_v37(private_case, spec)
        target = clarification.action.after_contract.decision_target
        for arm in resolved:
            action, _, _ = _choose_action_v381(
                private_case,
                observations,
                spec,
                arm,
                "training_random_v381",
            )
            if action is None:
                continue
            post = observations + [
                private_case.action_observations[action.action_id]
            ]
            _, decision, _ = _post_acquisition_challenge_v381(
                private_case, post, target, spec
            )
            resolved[arm] += int(decision.decision == "select")
    if count != 21 or any(resolved.values()):
        raise ValueError("V3.8.1 training failure signature changed")
    evolution = source["evolution"]
    return AcquisitionTrainingEvidenceV381.seal(
        evidence_id="acquisition_training_failure_v381",
        source_v38_evolution_hash=evolution.evolution_hash,
        source_v38_clarification_bundle_hash=clarifications.bundle_hash,
    )


class TargetDiscriminatingAcquisitionPolicyV381(StrictModel):
    schema_version: Literal["3.8.1"] = "3.8.1"
    policy_id: Identifier
    arm: AcquisitionArmV381
    training_evidence_hash: Sha256
    source_v38_evolution_hash: Sha256
    source_v38_clarification_bundle_hash: Sha256
    action_budget: Literal[1] = 1
    action_cost: Literal[1] = 1
    execute_only_next_action: Literal[
        "acquire_target_discriminating_evidence"
    ] = "acquire_target_discriminating_evidence"
    selection_rule: Literal[
        "sha256_case_policy_modulo_safe_catalog",
        "maximum_normalized_cross_family_trajectory_variance",
    ]
    private_mechanism_visible: Literal[False] = False
    private_observation_visible_before_permission: Literal[False] = False
    private_probe_visible: Literal[False] = False
    private_target_loss_visible: Literal[False] = False
    real_world_execution_permitted: Literal[False] = False
    task_router_permitted: Literal[False] = False
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "TargetDiscriminatingAcquisitionPolicyV381":
        expected = (
            "sha256_case_policy_modulo_safe_catalog"
            if self.arm == "deterministic_random_safe_action"
            else "maximum_normalized_cross_family_trajectory_variance"
        )
        if self.selection_rule != expected:
            raise ValueError("V3.8.1 acquisition selection rule differs")
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("policy_hash does not match V3.8.1 acquisition policy")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("V3.8.1 acquisition policy is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "TargetDiscriminatingAcquisitionPolicyV381":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"policy_hash"}),
            policy_hash=draft.content_hash(),
        )


def build_target_discriminating_policies_v381(
    training_evidence: AcquisitionTrainingEvidenceV381,
) -> tuple[
    TargetDiscriminatingAcquisitionPolicyV381,
    TargetDiscriminatingAcquisitionPolicyV381,
]:
    training_evidence.assert_sealed()
    common = {
        "training_evidence_hash": training_evidence.evidence_hash,
        "source_v38_evolution_hash": training_evidence.source_v38_evolution_hash,
        "source_v38_clarification_bundle_hash": (
            training_evidence.source_v38_clarification_bundle_hash
        ),
    }
    return (
        TargetDiscriminatingAcquisitionPolicyV381.seal(
            policy_id="deterministic_random_safe_action_v381",
            arm="deterministic_random_safe_action",
            selection_rule="sha256_case_policy_modulo_safe_catalog",
            **common,
        ),
        TargetDiscriminatingAcquisitionPolicyV381.seal(
            policy_id="maximum_portfolio_trajectory_disagreement_v381",
            arm="maximum_portfolio_trajectory_disagreement",
            selection_rule="maximum_normalized_cross_family_trajectory_variance",
            **common,
        ),
    )


class TargetDiscriminatingWorldPackSpecV381(ModelChallengeWorldPackSpecV371):
    schema_version: Literal["3.8.1"] = "3.8.1"
    seeds: list[int] = Field(min_length=16, max_length=16)
    bootstrap_seed: Literal[381722] = 381722
    clarification_policy_hash: Sha256
    training_evidence_hash: Sha256
    baseline_acquisition_policy_hash: Sha256
    candidate_acquisition_policy_hash: Sha256
    source_v38_evolution_hash: Sha256
    minimum_acquisition_case_count: Literal[8] = 8
    unresolved_adjudicated_loss: Literal[10.0] = 10.0
    material_acquisition_negative_transfer: Literal[0.1] = 0.1
    frozen_delta: Literal[
        "one_equal_cost_target_discriminating_action_only"
    ] = "one_equal_cost_target_discriminating_action_only"

    @model_validator(mode="after")
    def validate_spec(self) -> "TargetDiscriminatingWorldPackSpecV381":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.mechanisms != list(MECHANISMS_V31):
            raise ValueError("V3.8.1 requires the frozen mechanism order")
        if self.seeds != list(CONFIRMATION_SEEDS_V381):
            raise ValueError("V3.8.1 seeds do not match the frozen set")
        if self.family_catalog != list(FAMILIES_V37):
            raise ValueError("V3.8.1 cannot change the family catalog")
        if self.observation_action_indices != [0, 7]:
            raise ValueError("V3.8.1 cannot change initial observations")
        if not math.isclose(
            (self.trajectory_points - 1) * self.time_step,
            self.segment_count * self.segment_duration,
            abs_tol=1e-12,
        ):
            raise ValueError("V3.8.1 segments do not cover trajectory")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match V3.8.1 protocol")
        return self


def default_target_discriminating_spec_v381(
    *,
    method_evidence: ModelChallengeMethodEvidenceV37,
    baseline_model_policy: ModelPortfolioPolicyV37,
    candidate_model_policy: ModelPortfolioPolicyV37,
    disposition_policy: ChallengeDispositionPolicyV371,
    clarification_policy: TargetClarificationPolicyV38,
    training_evidence: AcquisitionTrainingEvidenceV381,
    baseline_acquisition_policy: TargetDiscriminatingAcquisitionPolicyV381,
    candidate_acquisition_policy: TargetDiscriminatingAcquisitionPolicyV381,
    frozen_at: datetime | None = None,
) -> TargetDiscriminatingWorldPackSpecV381:
    for artifact in (
        method_evidence, baseline_model_policy, candidate_model_policy,
        disposition_policy, clarification_policy, training_evidence,
        baseline_acquisition_policy, candidate_acquisition_policy,
    ):
        artifact.assert_sealed()
    return TargetDiscriminatingWorldPackSpecV381.seal(
        experiment_id="target_discriminating_acquisition_v381",
        mechanisms=list(MECHANISMS_V31),
        seeds=list(CONFIRMATION_SEEDS_V381),
        family_catalog=list(FAMILIES_V37),
        observation_action_indices=[0, 7],
        baseline_policy_hash=baseline_model_policy.policy_hash,
        candidate_policy_hash=candidate_model_policy.policy_hash,
        method_evidence_hash=method_evidence.evidence_hash,
        disposition_policy_hash=disposition_policy.policy_hash,
        source_failure_evolution_hash=disposition_policy.source_failure_evolution_hash,
        source_failure_candidate_bundle_hash=(
            disposition_policy.source_failure_candidate_bundle_hash
        ),
        clarification_policy_hash=clarification_policy.policy_hash,
        training_evidence_hash=training_evidence.evidence_hash,
        baseline_acquisition_policy_hash=baseline_acquisition_policy.policy_hash,
        candidate_acquisition_policy_hash=candidate_acquisition_policy.policy_hash,
        source_v38_evolution_hash=training_evidence.source_v38_evolution_hash,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


class AcquisitionProposalV381(StrictModel):
    schema_version: Literal["3.8.1"] = "3.8.1"
    proposal_id: Identifier
    case_id: Identifier
    policy_hash: Sha256
    arm: AcquisitionArmV381
    source_clarification_receipt_hash: Sha256
    candidate_action_hashes: list[Sha256] = Field(min_length=6, max_length=6)
    proposal_model_hashes: list[Sha256] = Field(max_length=3)
    action_scores: list[ActionTrajectoryScoreV381] = Field(max_length=6)
    selected_action_id: Identifier | None
    selected_action_hash: Sha256 | None
    decision: Literal["allow_synthetic", "abstain_no_admissible_action"]
    budget_before: Literal[1] = 1
    budget_after: Annotated[int, Field(ge=0, le=1)]
    private_mechanism_seen: Literal[False] = False
    private_observation_seen: Literal[False] = False
    private_probe_seen: Literal[False] = False
    private_target_loss_seen: Literal[False] = False
    real_world_execution_permitted: Literal[False] = False
    proposed_at: datetime
    proposal_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_proposal(self) -> "AcquisitionProposalV381":
        _assert_timezone(self.proposed_at, "proposed_at")
        if self.decision == "allow_synthetic":
            if self.selected_action_hash is None or self.selected_action_id is None:
                raise ValueError("V3.8.1 allowed proposal needs action")
            if self.selected_action_hash not in self.candidate_action_hashes:
                raise ValueError("V3.8.1 selected action is not a candidate")
            if self.budget_after != 0:
                raise ValueError("V3.8.1 allowed action must consume budget")
        else:
            if self.selected_action_hash is not None or self.selected_action_id is not None:
                raise ValueError("V3.8.1 abstention cannot select action")
            if self.budget_after != 1:
                raise ValueError("V3.8.1 abstention cannot consume budget")
        for score in self.action_scores:
            score.assert_sealed()
        if self.arm == "deterministic_random_safe_action":
            if self.proposal_model_hashes or self.action_scores:
                raise ValueError("V3.8.1 random arm cannot use model scores")
        elif len(self.proposal_model_hashes) != 3 or len(self.action_scores) != 6:
            raise ValueError("V3.8.1 disagreement arm needs three models and six scores")
        if self.proposal_hash and self.proposal_hash != self.content_hash():
            raise ValueError("proposal_hash does not match V3.8.1 proposal")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "proposal_hash")

    def assert_sealed(self) -> None:
        if not self.proposal_hash or self.proposal_hash != self.content_hash():
            raise ValueError("V3.8.1 acquisition proposal is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "AcquisitionProposalV381":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"proposal_hash"}),
            proposal_hash=draft.content_hash(),
        )


class SyntheticAcquisitionExecutionV381(StrictModel):
    schema_version: Literal["3.8.1"] = "3.8.1"
    execution_id: Identifier
    case_id: Identifier
    proposal_hash: Sha256
    action_hash: Sha256
    observation: ControlledObservationReceiptV31
    execution_scope: Literal["sealed_synthetic_reality_interface"] = (
        "sealed_synthetic_reality_interface"
    )
    real_world_execution: Literal[False] = False
    executed_at: datetime
    execution_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_execution(self) -> "SyntheticAcquisitionExecutionV381":
        _assert_timezone(self.executed_at, "executed_at")
        self.observation.assert_sealed()
        if self.observation.action_hash != self.action_hash:
            raise ValueError("V3.8.1 observation/action binding differs")
        if self.execution_hash and self.execution_hash != self.content_hash():
            raise ValueError("execution_hash does not match V3.8.1 execution")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "execution_hash")

    def assert_sealed(self) -> None:
        if not self.execution_hash or self.execution_hash != self.content_hash():
            raise ValueError("V3.8.1 acquisition execution is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "SyntheticAcquisitionExecutionV381":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"execution_hash"}),
            execution_hash=draft.content_hash(),
        )


class AcquisitionCaseReceiptV381(StrictModel):
    schema_version: Literal["3.8.1"] = "3.8.1"
    receipt_id: Identifier
    case_id: Identifier
    source_clarification_receipt_hash: Sha256
    acquisition_expected: bool
    proposal: AcquisitionProposalV381 | None
    execution: SyntheticAcquisitionExecutionV381 | None
    post_challenges: list[PostAcquisitionFamilyChallengeV381] = Field(max_length=3)
    post_decision: PostAcquisitionDecisionV381 | None
    selected_model: ControlledDriftModelV31 | None
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "AcquisitionCaseReceiptV381":
        if self.acquisition_expected:
            if self.proposal is None:
                raise ValueError("V3.8.1 expected acquisition needs proposal")
            self.proposal.assert_sealed()
            if self.proposal.decision == "allow_synthetic":
                if (
                    self.execution is None
                    or len(self.post_challenges) != 3
                    or self.post_decision is None
                ):
                    raise ValueError("V3.8.1 executed acquisition receipt incomplete")
                self.execution.assert_sealed()
                self.post_decision.assert_sealed()
                for challenge in self.post_challenges:
                    challenge.assert_sealed()
                if self.post_decision.challenge_hashes != [
                    item.challenge_hash for item in self.post_challenges
                ]:
                    raise ValueError("V3.8.1 decision/challenge binding differs")
                if self.post_decision.decision == "select":
                    if self.selected_model is None:
                        raise ValueError("V3.8.1 selected decision needs model")
                    self.selected_model.assert_sealed()
                elif self.selected_model is not None:
                    raise ValueError("V3.8.1 unresolved decision cannot contain model")
            elif any((
                self.execution is not None,
                bool(self.post_challenges),
                self.post_decision is not None,
                self.selected_model is not None,
            )):
                raise ValueError("V3.8.1 abstained proposal contains outputs")
        elif any((
            self.proposal is not None,
            self.execution is not None,
            bool(self.post_challenges),
            self.post_decision is not None,
            self.selected_model is not None,
        )):
            raise ValueError("V3.8.1 non-target case contains acquisition outputs")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match V3.8.1 case receipt")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("V3.8.1 acquisition case receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "AcquisitionCaseReceiptV381":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


class AcquisitionBundleV381(StrictModel):
    schema_version: Literal["3.8.1"] = "3.8.1"
    bundle_id: Identifier
    spec_hash: Sha256
    policy_hash: Sha256
    arm: AcquisitionArmV381
    source_clarification_bundle_hash: Sha256
    case_receipts: list[AcquisitionCaseReceiptV381] = Field(min_length=64, max_length=64)
    created_at: datetime
    bundle_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bundle(self) -> "AcquisitionBundleV381":
        _assert_timezone(self.created_at, "created_at")
        if len({item.case_id for item in self.case_receipts}) != 64:
            raise ValueError("V3.8.1 acquisition bundle case coverage differs")
        for item in self.case_receipts:
            item.assert_sealed()
        if self.bundle_hash and self.bundle_hash != self.content_hash():
            raise ValueError("bundle_hash does not match V3.8.1 acquisition bundle")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "bundle_hash")

    def assert_sealed(self) -> None:
        if not self.bundle_hash or self.bundle_hash != self.content_hash():
            raise ValueError("V3.8.1 acquisition bundle is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "AcquisitionBundleV381":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"bundle_hash"}),
            bundle_hash=draft.content_hash(),
        )


def execute_acquisition_policy_v381(
    spec: TargetDiscriminatingWorldPackSpecV381,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    clarification_bundle: TargetClarificationBundleV38,
    policy: TargetDiscriminatingAcquisitionPolicyV381,
    *,
    executed_at: datetime,
) -> AcquisitionBundleV381:
    for artifact in (spec, private_pack, clarification_bundle, policy):
        artifact.assert_sealed()
    expected_policy_hash = (
        spec.baseline_acquisition_policy_hash
        if policy.arm == "deterministic_random_safe_action"
        else spec.candidate_acquisition_policy_hash
    )
    if policy.policy_hash != expected_policy_hash:
        raise ValueError("V3.8.1 acquisition policy is not frozen in spec")
    private_by_id = {item.public_case.case_id: item for item in private_pack.cases}
    clarification_by_id = {
        item.case_id: item for item in clarification_bundle.case_receipts
    }
    receipts: list[AcquisitionCaseReceiptV381] = []
    for case_id, private_case in private_by_id.items():
        clarification = clarification_by_id[case_id]
        expected = (
            clarification.next_action == "acquire_target_discriminating_evidence"
        )
        common = {
            "receipt_id": f"acquisition_{policy.arm}_{case_id}",
            "case_id": case_id,
            "source_clarification_receipt_hash": clarification.receipt_hash,
            "acquisition_expected": expected,
        }
        if not expected:
            receipts.append(AcquisitionCaseReceiptV381.seal(
                proposal=None,
                execution=None,
                post_challenges=[],
                post_decision=None,
                selected_model=None,
                **common,
            ))
            continue
        observations = _shared_observations_v37(private_case, spec)
        action, models, scores = _choose_action_v381(
            private_case,
            observations,
            spec,
            policy.arm,
            policy.policy_hash,
        )
        candidates = _available_actions_v381(private_case, spec)
        allowed = action is not None
        proposal = AcquisitionProposalV381.seal(
            proposal_id=f"proposal_{policy.arm}_{case_id}",
            case_id=case_id,
            policy_hash=policy.policy_hash,
            arm=policy.arm,
            source_clarification_receipt_hash=clarification.receipt_hash,
            candidate_action_hashes=[item.action_hash for item in candidates],
            proposal_model_hashes=[item.model_hash for item in models],
            action_scores=scores,
            selected_action_id=action.action_id if action else None,
            selected_action_hash=action.action_hash if action else None,
            decision=(
                "allow_synthetic" if allowed else "abstain_no_admissible_action"
            ),
            budget_after=0 if allowed else 1,
            proposed_at=executed_at,
        )
        if not allowed:
            receipts.append(AcquisitionCaseReceiptV381.seal(
                proposal=proposal,
                execution=None,
                post_challenges=[],
                post_decision=None,
                selected_model=None,
                **common,
            ))
            continue
        if validate_action_against_envelope_v31(
            action,
            private_case.public_case.actuator,
            private_case.public_case.envelope,
        ):
            raise RuntimeError("V3.8.1 Harness refused selected catalog action")
        observation = private_case.action_observations[action.action_id]
        execution = SyntheticAcquisitionExecutionV381.seal(
            execution_id=f"execution_{policy.arm}_{case_id}",
            case_id=case_id,
            proposal_hash=proposal.proposal_hash,
            action_hash=action.action_hash,
            observation=observation,
            executed_at=executed_at,
        )
        target = clarification.action.after_contract.decision_target
        post_challenges, post_decision, selected_model = (
            _post_acquisition_challenge_v381(
                private_case,
                observations + [observation],
                target,
                spec,
            )
        )
        receipts.append(AcquisitionCaseReceiptV381.seal(
            proposal=proposal,
            execution=execution,
            post_challenges=post_challenges,
            post_decision=post_decision,
            selected_model=selected_model,
            **common,
        ))
    return AcquisitionBundleV381.seal(
        bundle_id=f"acquisition_{policy.arm}_{spec.experiment_id}",
        spec_hash=spec.spec_hash,
        policy_hash=policy.policy_hash,
        arm=policy.arm,
        source_clarification_bundle_hash=clarification_bundle.bundle_hash,
        case_receipts=receipts,
        created_at=executed_at,
    )


class PrivateAcquisitionCaseResultV381(StrictModel):
    schema_version: Literal["3.8.1"] = "3.8.1"
    case_id: Identifier
    mechanism: Identifier
    decision_target: DecisionTargetV31
    baseline_action_id: Identifier | None
    candidate_action_id: Identifier | None
    baseline_resolved: bool
    candidate_resolved: bool
    baseline_adjudicated_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    candidate_adjudicated_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    candidate_improvement: Annotated[float, Field(allow_inf_nan=False)]
    material_negative_transfer: bool


class TargetDiscriminatingEvolutionReportV381(StrictModel):
    schema_version: Literal["3.8.1"] = "3.8.1"
    evolution_id: Identifier
    spec_hash: Sha256
    training_evidence_hash: Sha256
    clarification_bundle_hash: Sha256
    baseline_bundle_hash: Sha256
    candidate_bundle_hash: Sha256
    case_results: list[PrivateAcquisitionCaseResultV381]
    acquisition_case_count: Annotated[int, Field(ge=0)]
    baseline_executed_count: Annotated[int, Field(ge=0)]
    candidate_executed_count: Annotated[int, Field(ge=0)]
    baseline_resolved_count: Annotated[int, Field(ge=0)]
    candidate_resolved_count: Annotated[int, Field(ge=0)]
    baseline_resolved_coverage: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    candidate_resolved_coverage: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    baseline_mean_adjudicated_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    candidate_mean_adjudicated_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    paired_mean_target_loss_improvement: Annotated[float, Field(allow_inf_nan=False)]
    paired_improvement_ci_lower: Annotated[float, Field(allow_inf_nan=False)]
    paired_improvement_ci_upper: Annotated[float, Field(allow_inf_nan=False)]
    material_negative_transfer_count: Annotated[int, Field(ge=0)]
    material_negative_transfer_upper_95: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    gates: dict[Identifier, bool]
    ready_for_confirmation: bool
    recovery_action: AcquisitionRecoveryV381
    status: Literal[
        "target_discriminating_acquisition_ready_for_confirmation_v381",
        "target_discriminating_acquisition_refuted_v381",
    ]
    task_router_permitted: Literal[False] = False
    qualification_permitted: Literal[False] = False
    confirmation_permitted: Literal[False] = False
    real_world_authorization_permitted: Literal[False] = False
    created_at: datetime
    evolution_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "TargetDiscriminatingEvolutionReportV381":
        _assert_timezone(self.created_at, "created_at")
        ready = all(self.gates.values())
        if self.ready_for_confirmation != ready:
            raise ValueError("V3.8.1 readiness disagrees with gates")
        expected_status = (
            "target_discriminating_acquisition_ready_for_confirmation_v381"
            if ready else "target_discriminating_acquisition_refuted_v381"
        )
        expected_recovery = (
            "run_fresh_confirmation"
            if ready else "stop_repeat_acquisition_reclassify_estimator_or_family"
        )
        if self.status != expected_status or self.recovery_action != expected_recovery:
            raise ValueError("V3.8.1 terminal action disagrees with gates")
        if self.evolution_hash and self.evolution_hash != self.content_hash():
            raise ValueError("evolution_hash does not match V3.8.1 report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evolution_hash")

    def assert_sealed(self) -> None:
        if not self.evolution_hash or self.evolution_hash != self.content_hash():
            raise ValueError("V3.8.1 evolution report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "TargetDiscriminatingEvolutionReportV381":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evolution_hash"}),
            evolution_hash=draft.content_hash(),
        )


def _bootstrap_ci_v381(values: np.ndarray, spec) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    random = np.random.default_rng(spec.bootstrap_seed)
    means = np.asarray([
        float(np.mean(values[random.integers(0, len(values), size=len(values))]))
        for _ in range(spec.bootstrap_replicates)
    ])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate_acquisition_policies_v381(
    spec: TargetDiscriminatingWorldPackSpecV381,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    clarification_bundle: TargetClarificationBundleV38,
    baseline: AcquisitionBundleV381,
    candidate: AcquisitionBundleV381,
    *,
    evaluated_at: datetime,
) -> TargetDiscriminatingEvolutionReportV381:
    for artifact in (
        spec, private_pack, clarification_bundle, baseline, candidate
    ):
        artifact.assert_sealed()
    if (
        baseline.arm != "deterministic_random_safe_action"
        or candidate.arm != "maximum_portfolio_trajectory_disagreement"
    ):
        raise ValueError("V3.8.1 adjudicator arm order differs")
    private_by_id = {item.public_case.case_id: item for item in private_pack.cases}
    clarification_by_id = {
        item.case_id: item for item in clarification_bundle.case_receipts
    }
    baseline_by_id = {item.case_id: item for item in baseline.case_receipts}
    candidate_by_id = {item.case_id: item for item in candidate.case_receipts}
    results: list[PrivateAcquisitionCaseResultV381] = []
    baseline_executed = 0
    candidate_executed = 0
    baseline_resolved = 0
    candidate_resolved = 0
    illegal = 0
    for case_id, private_case in private_by_id.items():
        clarification = clarification_by_id[case_id]
        if clarification.next_action != "acquire_target_discriminating_evidence":
            continue
        base = baseline_by_id[case_id]
        cand = candidate_by_id[case_id]
        baseline_executed += int(base.execution is not None)
        candidate_executed += int(cand.execution is not None)
        baseline_resolved_case = base.selected_model is not None
        candidate_resolved_case = cand.selected_model is not None
        baseline_resolved += int(baseline_resolved_case)
        candidate_resolved += int(candidate_resolved_case)
        base_loss = (
            _target_loss_v31(private_case, base.selected_model, spec)
            if base.selected_model is not None
            else spec.unresolved_adjudicated_loss
        )
        cand_loss = (
            _target_loss_v31(private_case, cand.selected_model, spec)
            if cand.selected_model is not None
            else spec.unresolved_adjudicated_loss
        )
        available_hashes = {
            item.action_hash for item in _available_actions_v381(private_case, spec)
        }
        for receipt in (base, cand):
            if (
                receipt.proposal is not None
                and receipt.proposal.selected_action_hash is not None
                and receipt.proposal.selected_action_hash not in available_hashes
            ):
                illegal += 1
        improvement = base_loss - cand_loss
        results.append(PrivateAcquisitionCaseResultV381(
            case_id=case_id,
            mechanism=private_case.mechanism,
            decision_target=clarification.action.after_contract.decision_target,
            baseline_action_id=(
                base.proposal.selected_action_id if base.proposal else None
            ),
            candidate_action_id=(
                cand.proposal.selected_action_id if cand.proposal else None
            ),
            baseline_resolved=baseline_resolved_case,
            candidate_resolved=candidate_resolved_case,
            baseline_adjudicated_target_loss=base_loss,
            candidate_adjudicated_target_loss=cand_loss,
            candidate_improvement=improvement,
            material_negative_transfer=(
                cand_loss - base_loss
                > spec.material_acquisition_negative_transfer
            ),
        ))
    count = len(results)
    base_losses = np.asarray([
        item.baseline_adjudicated_target_loss for item in results
    ])
    cand_losses = np.asarray([
        item.candidate_adjudicated_target_loss for item in results
    ])
    improvements = base_losses - cand_losses
    ci_lower, ci_upper = _bootstrap_ci_v381(improvements, spec)
    negatives = sum(item.material_negative_transfer for item in results)
    negative_upper = (
        float(beta.ppf(0.95, negatives + 1, count - negatives))
        if count > negatives else 1.0
    )
    base_coverage = baseline_resolved / max(count, 1)
    cand_coverage = candidate_resolved / max(count, 1)
    gates = {
        "minimum_acquisition_case_count": count >= spec.minimum_acquisition_case_count,
        "baseline_executes_all_targets": baseline_executed == count,
        "candidate_executes_all_targets": candidate_executed == count,
        "zero_illegal_or_duplicate_action": illegal == 0,
        "candidate_coverage_noninferior": cand_coverage >= base_coverage,
        "paired_improvement_ci_lower_positive": ci_lower > 0.0,
        "material_negative_transfer_upper": (
            negative_upper <= spec.maximum_negative_transfer_rate
        ),
        "no_real_world_execution": all(
            receipt.execution is None or not receipt.execution.real_world_execution
            for bundle in (baseline, candidate)
            for receipt in bundle.case_receipts
        ),
    }
    ready = all(gates.values())
    return TargetDiscriminatingEvolutionReportV381.seal(
        evolution_id=f"evolution_{spec.experiment_id}",
        spec_hash=spec.spec_hash,
        training_evidence_hash=spec.training_evidence_hash,
        clarification_bundle_hash=clarification_bundle.bundle_hash,
        baseline_bundle_hash=baseline.bundle_hash,
        candidate_bundle_hash=candidate.bundle_hash,
        case_results=results,
        acquisition_case_count=count,
        baseline_executed_count=baseline_executed,
        candidate_executed_count=candidate_executed,
        baseline_resolved_count=baseline_resolved,
        candidate_resolved_count=candidate_resolved,
        baseline_resolved_coverage=base_coverage,
        candidate_resolved_coverage=cand_coverage,
        baseline_mean_adjudicated_target_loss=(
            float(np.mean(base_losses)) if count else 0.0
        ),
        candidate_mean_adjudicated_target_loss=(
            float(np.mean(cand_losses)) if count else 0.0
        ),
        paired_mean_target_loss_improvement=(
            float(np.mean(improvements)) if count else 0.0
        ),
        paired_improvement_ci_lower=ci_lower,
        paired_improvement_ci_upper=ci_upper,
        material_negative_transfer_count=negatives,
        material_negative_transfer_upper_95=negative_upper,
        gates=gates,
        ready_for_confirmation=ready,
        recovery_action=(
            "run_fresh_confirmation"
            if ready else "stop_repeat_acquisition_reclassify_estimator_or_family"
        ),
        status=(
            "target_discriminating_acquisition_ready_for_confirmation_v381"
            if ready else "target_discriminating_acquisition_refuted_v381"
        ),
        created_at=evaluated_at,
    )


class TargetDiscriminatingManifestV381(StrictModel):
    schema_version: Literal["3.8.1"] = "3.8.1"
    run_id: Identifier
    artifact_refs: list[ArtifactRef] = Field(min_length=16, max_length=16)
    terminal_status: Literal[
        "target_discriminating_acquisition_ready_for_confirmation_v381",
        "target_discriminating_acquisition_refuted_v381",
    ]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "TargetDiscriminatingManifestV381":
        _assert_timezone(self.created_at, "created_at")
        if len({item.kind for item in self.artifact_refs}) != 16:
            raise ValueError("V3.8.1 manifest artifact kinds differ")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match V3.8.1 manifest")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("V3.8.1 manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "TargetDiscriminatingManifestV381":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class TargetDiscriminatingOutcomeV381:
    store: RunStore
    spec: TargetDiscriminatingWorldPackSpecV381
    private_pack: PrivateControlledDynamicsWorldPackV31
    model_candidate_bundle: ModelChallengeBundleV37
    disposition_bundle: ChallengeDispositionBundleV371
    clarification_bundle: TargetClarificationBundleV38
    baseline_bundle: AcquisitionBundleV381
    candidate_bundle: AcquisitionBundleV381
    evolution_report: TargetDiscriminatingEvolutionReportV381
    manifest: TargetDiscriminatingManifestV381


def run_target_discriminating_worldpack_v381(
    output_root: str | Path,
    *,
    source_v38_run_directory: str | Path,
    method_evidence: ModelChallengeMethodEvidenceV37,
    baseline_model_policy: ModelPortfolioPolicyV37,
    candidate_model_policy: ModelPortfolioPolicyV37,
    disposition_policy: ChallengeDispositionPolicyV371,
    clarification_policy: TargetClarificationPolicyV38,
    training_evidence: AcquisitionTrainingEvidenceV381,
    baseline_acquisition_policy: TargetDiscriminatingAcquisitionPolicyV381,
    candidate_acquisition_policy: TargetDiscriminatingAcquisitionPolicyV381,
    spec: TargetDiscriminatingWorldPackSpecV381,
    evaluated_at: datetime | None = None,
    run_id: str | None = None,
) -> TargetDiscriminatingOutcomeV381:
    source = _load_source_v38(source_v38_run_directory)
    for artifact in (
        method_evidence, baseline_model_policy, candidate_model_policy,
        disposition_policy, clarification_policy, training_evidence,
        baseline_acquisition_policy, candidate_acquisition_policy, spec,
    ):
        artifact.assert_sealed()
    if (
        source["evolution"].evolution_hash != training_evidence.source_v38_evolution_hash
        or source["clarification_bundle"].bundle_hash
        != training_evidence.source_v38_clarification_bundle_hash
    ):
        raise ValueError("V3.8.1 training source binding differs")
    bindings = {
        spec.method_evidence_hash: method_evidence.evidence_hash,
        spec.baseline_policy_hash: baseline_model_policy.policy_hash,
        spec.candidate_policy_hash: candidate_model_policy.policy_hash,
        spec.disposition_policy_hash: disposition_policy.policy_hash,
        spec.clarification_policy_hash: clarification_policy.policy_hash,
        spec.training_evidence_hash: training_evidence.evidence_hash,
        spec.baseline_acquisition_policy_hash: baseline_acquisition_policy.policy_hash,
        spec.candidate_acquisition_policy_hash: candidate_acquisition_policy.policy_hash,
    }
    if any(left != right for left, right in bindings.items()):
        raise ValueError("V3.8.1 frozen artifact binding differs")
    at = evaluated_at or datetime.now(timezone.utc)
    store = RunStore(
        output_root,
        run_id=run_id or f"target-discriminating-v381-{uuid4().hex[:10]}",
    )
    refs = [
        store.put_artifact("model_challenge_method_evidence_v381", method_evidence),
        store.put_artifact("model_challenge_baseline_policy_v381", baseline_model_policy),
        store.put_artifact("model_challenge_candidate_policy_v381", candidate_model_policy),
        store.put_artifact("challenge_disposition_policy_v381", disposition_policy),
        store.put_artifact("target_clarification_policy_v381", clarification_policy),
        store.put_artifact("acquisition_training_evidence_v381", training_evidence),
        store.put_artifact("baseline_acquisition_policy_v381", baseline_acquisition_policy),
        store.put_artifact("candidate_acquisition_policy_v381", candidate_acquisition_policy),
        store.put_artifact("target_discriminating_spec_v381", spec),
    ]
    store.emit("target_discriminating_v381_protocol_frozen_before_private_pack", {
        "spec_hash": spec.spec_hash,
        "training_evidence_hash": training_evidence.evidence_hash,
        "source_v38_independently_verified": True,
    })
    private_pack = generate_private_controlled_dynamics_worldpack_v31(
        spec, generated_at=at
    )
    candidate_model_bundle = execute_model_challenge_policy_v37(
        spec, private_pack, candidate_model_policy, executed_at=at
    )
    dispositions = create_challenge_disposition_bundle_v371(
        spec, private_pack, candidate_model_bundle, disposition_policy,
        created_at=at,
    )
    clarifications = execute_target_clarifications_v38(
        spec,
        private_pack,
        candidate_model_bundle,
        dispositions,
        disposition_policy,
        clarification_policy,
        executed_at=at,
    )
    baseline = execute_acquisition_policy_v381(
        spec, private_pack, clarifications, baseline_acquisition_policy,
        executed_at=at,
    )
    candidate = execute_acquisition_policy_v381(
        spec, private_pack, clarifications, candidate_acquisition_policy,
        executed_at=at,
    )
    evolution = evaluate_acquisition_policies_v381(
        spec, private_pack, clarifications, baseline, candidate,
        evaluated_at=at,
    )
    refs.extend([
        store.put_artifact("private_target_discriminating_worldpack_v381", private_pack),
        store.put_artifact("model_challenge_candidate_bundle_v381", candidate_model_bundle),
        store.put_artifact("challenge_disposition_bundle_v381", dispositions),
        store.put_artifact("target_clarification_bundle_v381", clarifications),
        store.put_artifact("baseline_acquisition_bundle_v381", baseline),
        store.put_artifact("candidate_acquisition_bundle_v381", candidate),
        store.put_artifact("target_discriminating_evolution_report_v381", evolution),
    ])
    manifest = TargetDiscriminatingManifestV381.seal(
        run_id=store.run_id,
        artifact_refs=refs,
        terminal_status=evolution.status,
        created_at=at,
    )
    manifest_ref = store.put_artifact("target_discriminating_manifest_v381", manifest)
    store.emit("target_discriminating_v381_worldpack_adjudicated", {
        "manifest_ref": manifest_ref.model_dump(mode="json")
    })
    if not verify_target_discriminating_run_v381(store.run_directory):
        raise RuntimeError("V3.8.1 run failed independent verification")
    return TargetDiscriminatingOutcomeV381(
        store,
        spec,
        private_pack,
        candidate_model_bundle,
        dispositions,
        clarifications,
        baseline,
        candidate,
        evolution,
        manifest,
    )


def verify_target_discriminating_run_v381(
    run_directory: str | Path,
) -> bool:
    try:
        store = RunStore.open_existing(run_directory)
        if not store.verify_event_chain():
            return False
        events = [
            json.loads(line)
            for line in store.event_path.read_text(encoding="utf-8").splitlines()
        ]
        refs = _committed_refs(store)
        if len(refs) != 17:
            return False
        for ref in refs:
            store.load_artifact(ref)
        manifest = _load_one(
            store, refs, "target_discriminating_manifest_v381",
            TargetDiscriminatingManifestV381,
        )
        manifest.assert_sealed()
        kinds = {item.kind for item in manifest.artifact_refs}
        expected_kinds = {
            "model_challenge_method_evidence_v381",
            "model_challenge_baseline_policy_v381",
            "model_challenge_candidate_policy_v381",
            "challenge_disposition_policy_v381",
            "target_clarification_policy_v381",
            "acquisition_training_evidence_v381",
            "baseline_acquisition_policy_v381",
            "candidate_acquisition_policy_v381",
            "target_discriminating_spec_v381",
            "private_target_discriminating_worldpack_v381",
            "model_challenge_candidate_bundle_v381",
            "challenge_disposition_bundle_v381",
            "target_clarification_bundle_v381",
            "baseline_acquisition_bundle_v381",
            "candidate_acquisition_bundle_v381",
            "target_discriminating_evolution_report_v381",
        }
        if kinds != expected_kinds:
            return False
        method = _load_one(
            store, refs, "model_challenge_method_evidence_v381",
            ModelChallengeMethodEvidenceV37,
        )
        baseline_model_policy = _load_one(
            store, refs, "model_challenge_baseline_policy_v381",
            ModelPortfolioPolicyV37,
        )
        candidate_model_policy = _load_one(
            store, refs, "model_challenge_candidate_policy_v381",
            ModelPortfolioPolicyV37,
        )
        disposition_policy = _load_one(
            store, refs, "challenge_disposition_policy_v381",
            ChallengeDispositionPolicyV371,
        )
        clarification_policy = _load_one(
            store, refs, "target_clarification_policy_v381",
            TargetClarificationPolicyV38,
        )
        training = _load_one(
            store, refs, "acquisition_training_evidence_v381",
            AcquisitionTrainingEvidenceV381,
        )
        baseline_policy = _load_one(
            store, refs, "baseline_acquisition_policy_v381",
            TargetDiscriminatingAcquisitionPolicyV381,
        )
        candidate_policy = _load_one(
            store, refs, "candidate_acquisition_policy_v381",
            TargetDiscriminatingAcquisitionPolicyV381,
        )
        spec = _load_one(
            store, refs, "target_discriminating_spec_v381",
            TargetDiscriminatingWorldPackSpecV381,
        )
        private_pack = _load_one(
            store, refs, "private_target_discriminating_worldpack_v381",
            PrivateControlledDynamicsWorldPackV31,
        )
        model_candidate = _load_one(
            store, refs, "model_challenge_candidate_bundle_v381",
            ModelChallengeBundleV37,
        )
        dispositions = _load_one(
            store, refs, "challenge_disposition_bundle_v381",
            ChallengeDispositionBundleV371,
        )
        clarifications = _load_one(
            store, refs, "target_clarification_bundle_v381",
            TargetClarificationBundleV38,
        )
        baseline = _load_one(
            store, refs, "baseline_acquisition_bundle_v381",
            AcquisitionBundleV381,
        )
        candidate = _load_one(
            store, refs, "candidate_acquisition_bundle_v381",
            AcquisitionBundleV381,
        )
        evolution = _load_one(
            store, refs, "target_discriminating_evolution_report_v381",
            TargetDiscriminatingEvolutionReportV381,
        )
        for artifact in (
            method, baseline_model_policy, candidate_model_policy,
            disposition_policy, clarification_policy, training,
            baseline_policy, candidate_policy, spec, private_pack,
            model_candidate, dispositions, clarifications,
            baseline, candidate, evolution,
        ):
            artifact.assert_sealed()
        regenerated = generate_private_controlled_dynamics_worldpack_v31(
            spec, generated_at=private_pack.generated_at
        )
        if regenerated.pack_hash != private_pack.pack_hash:
            return False
        replay_model_candidate = execute_model_challenge_policy_v37(
            spec,
            private_pack,
            candidate_model_policy,
            executed_at=model_candidate.created_at,
        )
        if replay_model_candidate.bundle_hash != model_candidate.bundle_hash:
            return False
        replay_dispositions = create_challenge_disposition_bundle_v371(
            spec,
            private_pack,
            model_candidate,
            disposition_policy,
            created_at=dispositions.created_at,
        )
        if replay_dispositions.bundle_hash != dispositions.bundle_hash:
            return False
        replay_clarifications = execute_target_clarifications_v38(
            spec,
            private_pack,
            model_candidate,
            dispositions,
            disposition_policy,
            clarification_policy,
            executed_at=clarifications.created_at,
        )
        if replay_clarifications.bundle_hash != clarifications.bundle_hash:
            return False
        replay_baseline = execute_acquisition_policy_v381(
            spec,
            private_pack,
            clarifications,
            baseline_policy,
            executed_at=baseline.created_at,
        )
        replay_candidate = execute_acquisition_policy_v381(
            spec,
            private_pack,
            clarifications,
            candidate_policy,
            executed_at=candidate.created_at,
        )
        if (
            replay_baseline.bundle_hash != baseline.bundle_hash
            or replay_candidate.bundle_hash != candidate.bundle_hash
        ):
            return False
        recomputed = evaluate_acquisition_policies_v381(
            spec,
            private_pack,
            clarifications,
            baseline,
            candidate,
            evaluated_at=evolution.created_at,
        )
        if recomputed.evolution_hash != evolution.evolution_hash:
            return False
        if manifest.terminal_status != evolution.status:
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
            == "target_discriminating_v381_protocol_frozen_before_private_pack"
        ]
        private_events = [
            event for event in events
            if event["event_type"] == "artifact_committed"
            and event["payload"]["kind"]
            == "private_target_discriminating_worldpack_v381"
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
