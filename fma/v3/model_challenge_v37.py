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
from scipy.signal import savgol_filter
from scipy.stats import beta

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore
from fma.v2.dynamics_ir import (
    evaluate_polynomial_library,
    polynomial_basis_terms,
    trajectory_nrmse,
)
from fma.v2.schemas import Identifier, Sha256, _assert_timezone

from .controlled_dynamics_loop import (
    MECHANISMS_V31,
    ControlledDriftModelV31,
    MechanismV31,
    PrivateControlledDynamicsCaseV31,
    PrivateControlledDynamicsWorldPackV31,
    _simulate_model_v31,
    _target_loss_v31,
    generate_private_controlled_dynamics_worldpack_v31,
)


EXPLORATORY_SEEDS_V37 = (
    18011, 18059, 18119, 18169, 18223, 18269, 18329, 18379,
    18433, 18481, 18539, 18587, 18637, 18691, 18743, 18797,
)

FamilyV37 = Literal[
    "linear_state_space",
    "quadratic_interaction_ode",
    "cubic_sparse_ode",
]
ArmV37 = Literal[
    "fixed_quadratic_baseline",
    "applicability_challenge_candidate",
]

FAMILIES_V37: tuple[FamilyV37, ...] = (
    "linear_state_space",
    "quadratic_interaction_ode",
    "cubic_sparse_ode",
)
FAMILY_DEGREES_V37: dict[FamilyV37, int] = {
    "linear_state_space": 1,
    "quadratic_interaction_ode": 2,
    "cubic_sparse_ode": 3,
}
EXPECTED_MINIMAL_FAMILY_V37: dict[MechanismV31, FamilyV37] = {
    "exponential_decay": "linear_state_space",
    "logistic_growth": "quadratic_interaction_ode",
    "damped_oscillator": "linear_state_space",
    "duffing_oscillator": "cubic_sparse_ode",
}


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class MethodSourceV37(StrictModel):
    source_id: Identifier
    title: Annotated[str, Field(min_length=8)]
    doi: Annotated[str, Field(pattern=r"^10\.[0-9]{4,9}/\S+$")]
    source_url: Annotated[str, Field(pattern=r"^https://")]
    accessed_on: Literal["2026-07-22"] = "2026-07-22"
    borrowed_principle: Annotated[str, Field(min_length=20)]
    guarantee_transferred: Literal[False] = False


class ModelChallengeMethodEvidenceV37(StrictModel):
    schema_version: Literal["3.7"] = "3.7"
    evidence_id: Identifier
    retrieval_scope: Literal["targeted_non_exhaustive_openalex_plus_primary_pages"] = (
        "targeted_non_exhaustive_openalex_plus_primary_pages"
    )
    sources: list[MethodSourceV37] = Field(min_length=4, max_length=4)
    external_content_treated_as_untrusted_data: Literal[True] = True
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> "ModelChallengeMethodEvidenceV37":
        if len({item.doi.lower() for item in self.sources}) != len(self.sources):
            raise ValueError("V3.7 method sources must have unique DOIs")
        if self.evidence_hash and self.evidence_hash != self.content_hash():
            raise ValueError("evidence_hash does not match V3.7 method evidence")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evidence_hash")

    def assert_sealed(self) -> None:
        if not self.evidence_hash or self.evidence_hash != self.content_hash():
            raise ValueError("V3.7 method evidence is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ModelChallengeMethodEvidenceV37":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evidence_hash"}),
            evidence_hash=draft.content_hash(),
        )


def default_model_challenge_method_evidence_v37() -> ModelChallengeMethodEvidenceV37:
    return ModelChallengeMethodEvidenceV37.seal(
        evidence_id="model_challenge_method_evidence_v37",
        sources=[
            MethodSourceV37(
                source_id="atkinson_fedorov_1975",
                title="The design of experiments for discriminating between two rival models",
                doi="10.1093/biomet/62.1.57",
                source_url="https://academic.oup.com/biomet/article-abstract/62/1/57/220443",
                borrowed_principle=(
                    "Treat discrimination among rival models as an explicit experimental objective."
                ),
            ),
            MethodSourceV37(
                source_id="kennedy_ohagan_2001",
                title="Bayesian Calibration of Computer Models",
                doi="10.1111/1467-9868.00294",
                source_url="https://academic.oup.com/jrsssb/article-abstract/63/3/425/7083367",
                borrowed_principle=(
                    "Keep parameter uncertainty distinct from inadequacy of the model family."
                ),
            ),
            MethodSourceV37(
                source_id="yao_etal_2018",
                title="Using Stacking to Average Bayesian Predictive Distributions",
                doi="10.1214/17-BA1091",
                source_url=(
                    "https://projecteuclid.org/journals/bayesian-analysis/volume-13/"
                    "issue-3/Using-Stacking-to-Average-Bayesian-Predictive-"
                    "Distributions-with-Discussion/10.1214/17-BA1091.full"
                ),
                borrowed_principle=(
                    "Use held-out predictive performance rather than training fit for model comparison."
                ),
            ),
            MethodSourceV37(
                source_id="plumlee_2019",
                title="Computer Model Calibration with Confidence and Consistency",
                doi="10.1111/rssb.12314",
                source_url="https://academic.oup.com/jrsssb/article/81/3/519/7048342",
                borrowed_principle=(
                    "Record input coverage and model discrepancy limits as part of applicability."
                ),
            ),
        ],
    )


class ModelPortfolioPolicyV37(StrictModel):
    schema_version: Literal["3.7"] = "3.7"
    policy_id: Identifier
    arm: ArmV37
    method_evidence_hash: Sha256
    family_catalog: list[FamilyV37] = Field(min_length=3, max_length=3)
    observation_action_indices: list[int] = Field(min_length=2, max_length=2)
    selection_rule: Literal[
        "fixed_quadratic",
        "simplest_eligible_within_best_mean_plus_best_standard_error",
    ]
    private_mechanism_visible: Literal[False] = False
    private_probe_visible: Literal[False] = False
    private_target_loss_visible: Literal[False] = False
    real_world_execution_permitted: Literal[False] = False
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "ModelPortfolioPolicyV37":
        if self.family_catalog != list(FAMILIES_V37):
            raise ValueError("V3.7 policy requires the frozen family order")
        if self.observation_action_indices != [0, 7]:
            raise ValueError("V3.7 policy requires frozen action indices [0, 7]")
        expected = (
            "fixed_quadratic"
            if self.arm == "fixed_quadratic_baseline"
            else "simplest_eligible_within_best_mean_plus_best_standard_error"
        )
        if self.selection_rule != expected:
            raise ValueError("V3.7 policy selection rule disagrees with arm")
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("policy_hash does not match V3.7 policy")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("V3.7 policy is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ModelPortfolioPolicyV37":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"policy_hash"}),
            policy_hash=draft.content_hash(),
        )


def default_model_portfolio_policies_v37(
    method_evidence_hash: str,
) -> tuple[ModelPortfolioPolicyV37, ModelPortfolioPolicyV37]:
    common = {
        "method_evidence_hash": method_evidence_hash,
        "family_catalog": list(FAMILIES_V37),
        "observation_action_indices": [0, 7],
    }
    return (
        ModelPortfolioPolicyV37.seal(
            policy_id="fixed_quadratic_baseline_v37",
            arm="fixed_quadratic_baseline",
            selection_rule="fixed_quadratic",
            **common,
        ),
        ModelPortfolioPolicyV37.seal(
            policy_id="applicability_challenge_candidate_v37",
            arm="applicability_challenge_candidate",
            selection_rule=(
                "simplest_eligible_within_best_mean_plus_best_standard_error"
            ),
            **common,
        ),
    )


class ModelChallengeWorldPackSpecV37(StrictModel):
    schema_version: Literal["3.7"] = "3.7"
    experiment_id: Identifier
    phase: Literal["exploratory"] = "exploratory"
    mechanisms: list[MechanismV31] = Field(min_length=4, max_length=4)
    seeds: list[int] = Field(min_length=16, max_length=16)
    trajectory_points: Literal[49] = 49
    time_step: Literal[0.04] = 0.04
    segment_count: Literal[6] = 6
    segment_duration: Literal[0.32] = 0.32
    input_amplitude: Literal[0.35] = 0.35
    observation_noise_fraction: Literal[0.01] = 0.01
    savgol_window: Literal[9] = 9
    savgol_order: Literal[3] = 3
    ridge_alpha: Literal[0.0001] = 0.0001
    sparsity_threshold: Literal[0.02] = 0.02
    maximum_empirical_prediction_risk: Literal[0.25] = 0.25
    model_mismatch_residual_threshold: Literal[0.24] = 0.24
    family_catalog: list[FamilyV37] = Field(min_length=3, max_length=3)
    observation_action_indices: list[int] = Field(min_length=2, max_length=2)
    minimum_rank_ratio: Literal[0.95] = 0.95
    maximum_condition_number: Literal[100000000.0] = 100000000.0
    maximum_cv_prediction_loss: Literal[0.35] = 0.35
    minimum_candidate_coverage: Literal[0.9] = 0.9
    minimum_routing_accuracy: Literal[0.75] = 0.75
    maximum_mechanism_regression: Literal[0.02] = 0.02
    material_negative_transfer: Literal[0.02] = 0.02
    maximum_negative_transfer_rate: Literal[0.1] = 0.1
    bootstrap_replicates: Literal[2000] = 2000
    bootstrap_seed: Literal[370722] = 370722
    expected_quality_abstention_count: Literal[12] = 12
    baseline_policy_hash: Sha256
    candidate_policy_hash: Sha256
    method_evidence_hash: Sha256
    frozen_delta: Literal[
        "fixed_quadratic_vs_applicability_challenge_only"
    ] = "fixed_quadratic_vs_applicability_challenge_only"
    frozen_at: datetime
    spec_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "ModelChallengeWorldPackSpecV37":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.mechanisms != list(MECHANISMS_V31):
            raise ValueError("V3.7 requires the frozen four-mechanism order")
        if self.seeds != list(EXPLORATORY_SEEDS_V37):
            raise ValueError("V3.7 seeds do not match the frozen exploratory set")
        if self.family_catalog != list(FAMILIES_V37):
            raise ValueError("V3.7 requires the frozen family catalog")
        if self.observation_action_indices != [0, 7]:
            raise ValueError("V3.7 requires action indices [0, 7]")
        if not math.isclose(
            (self.trajectory_points - 1) * self.time_step,
            self.segment_count * self.segment_duration,
            abs_tol=1e-12,
        ):
            raise ValueError("V3.7 segments do not cover the trajectory")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match V3.7 protocol")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "spec_hash")

    def assert_sealed(self) -> None:
        if not self.spec_hash or self.spec_hash != self.content_hash():
            raise ValueError("V3.7 protocol is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ModelChallengeWorldPackSpecV37":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"spec_hash"}),
            spec_hash=draft.content_hash(),
        )


def default_model_challenge_exploratory_spec_v37(
    *,
    method_evidence_hash: str,
    baseline_policy_hash: str,
    candidate_policy_hash: str,
    frozen_at: datetime | None = None,
) -> ModelChallengeWorldPackSpecV37:
    return ModelChallengeWorldPackSpecV37.seal(
        experiment_id="model_challenge_exploratory_v37",
        mechanisms=list(MECHANISMS_V31),
        seeds=list(EXPLORATORY_SEEDS_V37),
        family_catalog=list(FAMILIES_V37),
        observation_action_indices=[0, 7],
        baseline_policy_hash=baseline_policy_hash,
        candidate_policy_hash=candidate_policy_hash,
        method_evidence_hash=method_evidence_hash,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


class FamilyChallengeReceiptV37(StrictModel):
    schema_version: Literal["3.7"] = "3.7"
    challenge_id: Identifier
    case_id: Identifier
    family: FamilyV37
    polynomial_degree: Annotated[int, Field(ge=1, le=3)]
    basis_term_count: Annotated[int, Field(ge=1)]
    source_observation_hashes: list[Sha256] = Field(min_length=3, max_length=3)
    normalized_design_rank: Annotated[int, Field(ge=0)]
    normalized_rank_ratio: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    normalized_condition_number: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    normalized_derivative_residual: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    fold_prediction_losses: list[Annotated[float, Field(ge=0, allow_inf_nan=False)]] = Field(
        min_length=3, max_length=3
    )
    mean_cv_prediction_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    standard_error_cv_prediction_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    simulation_failure_count: Annotated[int, Field(ge=0, le=3)]
    minimum_rank_ratio: Literal[0.95] = 0.95
    maximum_condition_number: Literal[100000000.0] = 100000000.0
    maximum_cv_prediction_loss: Literal[0.35] = 0.35
    eligible: bool
    gate_codes: list[Identifier]
    challenge_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_challenge(self) -> "FamilyChallengeReceiptV37":
        if FAMILY_DEGREES_V37[self.family] != self.polynomial_degree:
            raise ValueError("V3.7 family degree disagrees")
        values = np.asarray(self.fold_prediction_losses, dtype=float)
        expected_mean = float(np.mean(values))
        expected_se = float(np.std(values, ddof=1) / math.sqrt(len(values)))
        if not math.isclose(self.mean_cv_prediction_loss, expected_mean, abs_tol=1e-12):
            raise ValueError("V3.7 challenge mean does not recompute")
        if not math.isclose(
            self.standard_error_cv_prediction_loss, expected_se, abs_tol=1e-12
        ):
            raise ValueError("V3.7 challenge standard error does not recompute")
        expected_eligible = (
            self.simulation_failure_count == 0
            and self.normalized_rank_ratio >= self.minimum_rank_ratio
            and self.normalized_condition_number <= self.maximum_condition_number
            and self.mean_cv_prediction_loss <= self.maximum_cv_prediction_loss
        )
        if self.eligible != expected_eligible:
            raise ValueError("V3.7 challenge eligibility does not recompute")
        if len(set(self.source_observation_hashes)) != 3:
            raise ValueError("V3.7 challenge needs three distinct observations")
        if self.challenge_hash and self.challenge_hash != self.content_hash():
            raise ValueError("challenge_hash does not match V3.7 challenge")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "challenge_hash")

    def assert_sealed(self) -> None:
        if not self.challenge_hash or self.challenge_hash != self.content_hash():
            raise ValueError("V3.7 challenge is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "FamilyChallengeReceiptV37":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"challenge_hash"}),
            challenge_hash=draft.content_hash(),
        )


class ApplicabilityStateV37(StrictModel):
    schema_version: Literal["3.7"] = "3.7"
    state_id: Identifier
    case_id: Identifier
    public_case_hash: Sha256
    contract_hash: Sha256
    actuator_hash: Sha256
    envelope_hash: Sha256
    observation_hashes: list[Sha256] = Field(min_length=3, max_length=3)
    state_dimension: Annotated[int, Field(ge=1)]
    decision_target: Literal["free_run_prediction", "controlled_response_prediction", "unspecified"]
    quality_flags: list[Identifier]
    trajectory_count: Literal[3] = 3
    points_per_trajectory: Literal[49] = 49
    state_coverage_ratio_by_dimension: list[
        Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    ]
    input_amplitude_coverage_ratio: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ]
    input_design_rank_ratio: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ]
    challenge_hashes: list[Sha256] = Field(min_length=3, max_length=3)
    heldout_loss_disagreement: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    private_mechanism_seen: Literal[False] = False
    private_probe_seen: Literal[False] = False
    private_target_loss_seen: Literal[False] = False
    state_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_state(self) -> "ApplicabilityStateV37":
        if len(self.observation_hashes) != len(set(self.observation_hashes)):
            raise ValueError("V3.7 state observation hashes must be distinct")
        if len(self.challenge_hashes) != len(set(self.challenge_hashes)):
            raise ValueError("V3.7 state challenge hashes must be distinct")
        if len(self.state_coverage_ratio_by_dimension) != self.state_dimension:
            raise ValueError("V3.7 state coverage dimension disagrees")
        if self.state_hash and self.state_hash != self.content_hash():
            raise ValueError("state_hash does not match V3.7 applicability state")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "state_hash")

    def assert_sealed(self) -> None:
        if not self.state_hash or self.state_hash != self.content_hash():
            raise ValueError("V3.7 applicability state is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ApplicabilityStateV37":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"state_hash"}),
            state_hash=draft.content_hash(),
        )


class ModelChallengeDecisionV37(StrictModel):
    schema_version: Literal["3.7"] = "3.7"
    decision_id: Identifier
    case_id: Identifier
    arm: ArmV37
    state_hash: Sha256
    challenge_hashes: list[Sha256] = Field(min_length=3, max_length=3)
    selected_family: FamilyV37 | None
    best_cv_family: FamilyV37 | None
    one_standard_error_threshold: Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ] | None
    decision: Literal["select", "abstain"]
    reason: Literal[
        "fixed_quadratic",
        "one_standard_error_challenge",
        "deny_data_quality",
        "needs_evidence_no_eligible_family",
    ]
    decision_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> "ModelChallengeDecisionV37":
        if self.decision == "select" and self.selected_family is None:
            raise ValueError("V3.7 selection needs a family")
        if self.decision == "abstain" and self.selected_family is not None:
            raise ValueError("V3.7 abstention cannot select a family")
        if self.arm == "fixed_quadratic_baseline" and self.decision == "select":
            if self.selected_family != "quadratic_interaction_ode":
                raise ValueError("V3.7 baseline must select quadratic family")
        if self.decision_hash and self.decision_hash != self.content_hash():
            raise ValueError("decision_hash does not match V3.7 decision")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "decision_hash")

    def assert_sealed(self) -> None:
        if not self.decision_hash or self.decision_hash != self.content_hash():
            raise ValueError("V3.7 decision is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ModelChallengeDecisionV37":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"decision_hash"}),
            decision_hash=draft.content_hash(),
        )


class ModelChallengeCaseReceiptV37(StrictModel):
    schema_version: Literal["3.7"] = "3.7"
    receipt_id: Identifier
    case_id: Identifier
    public_case_hash: Sha256
    policy_hash: Sha256
    arm: ArmV37
    observation_hashes: list[Sha256] = Field(min_length=3, max_length=3)
    applicability_state: ApplicabilityStateV37
    challenges: list[FamilyChallengeReceiptV37] = Field(min_length=3, max_length=3)
    decision: ModelChallengeDecisionV37
    selected_model: ControlledDriftModelV31 | None
    executed_at: datetime
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "ModelChallengeCaseReceiptV37":
        _assert_timezone(self.executed_at, "executed_at")
        self.applicability_state.assert_sealed()
        self.decision.assert_sealed()
        for challenge in self.challenges:
            challenge.assert_sealed()
        if [item.family for item in self.challenges] != list(FAMILIES_V37):
            raise ValueError("V3.7 receipt challenge order changed")
        challenge_hashes = [item.challenge_hash for item in self.challenges]
        if self.applicability_state.challenge_hashes != challenge_hashes:
            raise ValueError("V3.7 state does not bind challenge receipts")
        if self.decision.challenge_hashes != challenge_hashes:
            raise ValueError("V3.7 decision does not bind challenge receipts")
        if self.decision.state_hash != self.applicability_state.state_hash:
            raise ValueError("V3.7 decision does not bind applicability state")
        if self.observation_hashes != self.applicability_state.observation_hashes:
            raise ValueError("V3.7 receipt observation binding differs")
        if self.decision.decision == "select":
            if self.selected_model is None:
                raise ValueError("V3.7 selected receipt needs a fitted model")
            self.selected_model.assert_sealed()
            expected_degree = FAMILY_DEGREES_V37[self.decision.selected_family]
            actual_degree = max(sum(term.exponents) for term in self.selected_model.basis_terms)
            if actual_degree != expected_degree:
                raise ValueError("V3.7 selected model family does not match decision")
            if self.selected_model.source_observation_hashes != self.observation_hashes:
                raise ValueError("V3.7 selected model uses different observations")
        elif self.selected_model is not None:
            raise ValueError("V3.7 abstention cannot contain a fitted model")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match V3.7 receipt")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("V3.7 case receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ModelChallengeCaseReceiptV37":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


class ModelChallengeBundleV37(StrictModel):
    schema_version: Literal["3.7"] = "3.7"
    bundle_id: Identifier
    spec_hash: Sha256
    private_pack_hash: Sha256
    policy_hash: Sha256
    arm: ArmV37
    case_receipts: list[ModelChallengeCaseReceiptV37] = Field(min_length=64, max_length=64)
    created_at: datetime
    bundle_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bundle(self) -> "ModelChallengeBundleV37":
        _assert_timezone(self.created_at, "created_at")
        case_ids = [item.case_id for item in self.case_receipts]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("V3.7 bundle case ids must be unique")
        for receipt in self.case_receipts:
            receipt.assert_sealed()
            if receipt.policy_hash != self.policy_hash or receipt.arm != self.arm:
                raise ValueError("V3.7 receipt does not belong to bundle policy")
        if self.bundle_hash and self.bundle_hash != self.content_hash():
            raise ValueError("bundle_hash does not match V3.7 bundle")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "bundle_hash")

    def assert_sealed(self) -> None:
        if not self.bundle_hash or self.bundle_hash != self.content_hash():
            raise ValueError("V3.7 bundle is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ModelChallengeBundleV37":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"bundle_hash"}),
            bundle_hash=draft.content_hash(),
        )


def _shared_observations_v37(
    private_case: PrivateControlledDynamicsCaseV31,
    spec: ModelChallengeWorldPackSpecV37,
) -> list[object]:
    public = private_case.public_case
    observations: list[object] = [public.pilot]
    for index in spec.observation_action_indices:
        action = public.action_catalog[index]
        observations.append(private_case.action_observations[action.action_id])
    return observations


def _observation_arrays_v37(
    private_case: PrivateControlledDynamicsCaseV31,
    observations: list[object],
    degree: int,
    spec: ModelChallengeWorldPackSpecV37,
) -> tuple[list[object], np.ndarray, np.ndarray, list[str]]:
    public = private_case.public_case
    terms = polynomial_basis_terms(public.state_names, degree)
    libraries: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    hashes: list[str] = []
    actuator = np.asarray(public.actuator.matrix, dtype=float)
    trim = spec.savgol_window // 2
    for observation in observations:
        states = np.asarray(observation.states, dtype=float)
        inputs = np.asarray(observation.inputs, dtype=float)
        derivatives = np.column_stack([
            savgol_filter(
                states[:, equation],
                window_length=spec.savgol_window,
                polyorder=spec.savgol_order,
                deriv=1,
                delta=spec.time_step,
                mode="interp",
            )
            for equation in range(states.shape[1])
        ])
        drift_targets = derivatives - inputs @ actuator.T
        libraries.append(evaluate_polynomial_library(states[trim:-trim], terms))
        targets.append(drift_targets[trim:-trim])
        hashes.append(observation.observation_hash)
    return terms, np.vstack(libraries), np.vstack(targets), hashes


def _ridge_sparse_v37(
    library: np.ndarray,
    targets: np.ndarray,
    spec: ModelChallengeWorldPackSpecV37,
) -> np.ndarray:
    scales = np.sqrt(np.mean(library**2, axis=0))
    scales[scales < 1e-12] = 1.0
    normalized = library / scales
    coefficients = np.linalg.solve(
        normalized.T @ normalized + spec.ridge_alpha * np.eye(normalized.shape[1]),
        normalized.T @ targets,
    ).T / scales[np.newaxis, :]
    for equation in range(targets.shape[1]):
        active = np.abs(coefficients[equation]) >= spec.sparsity_threshold
        for _ in range(12):
            previous = active.copy()
            coefficients[equation, ~active] = 0.0
            if active.any():
                selected = normalized[:, active]
                solved = np.linalg.solve(
                    selected.T @ selected
                    + spec.ridge_alpha * np.eye(selected.shape[1]),
                    selected.T @ targets[:, equation],
                )
                coefficients[equation, active] = solved / scales[active]
            active = np.abs(coefficients[equation]) >= spec.sparsity_threshold
            if np.array_equal(active, previous):
                break
        coefficients[equation, ~active] = 0.0
    return coefficients


def _fit_family_v37(
    private_case: PrivateControlledDynamicsCaseV31,
    observations: list[object],
    family: FamilyV37,
    spec: ModelChallengeWorldPackSpecV37,
    *,
    model_suffix: str,
) -> ControlledDriftModelV31:
    degree = FAMILY_DEGREES_V37[family]
    terms, library, targets, hashes = _observation_arrays_v37(
        private_case, observations, degree, spec
    )
    coefficients = _ridge_sparse_v37(library, targets, spec)
    scales = np.sqrt(np.mean(library**2, axis=0))
    scales[scales < 1e-12] = 1.0
    normalized = library / scales
    fitted = library @ coefficients.T
    residual = float(
        np.sqrt(np.mean((fitted - targets) ** 2))
        / max(float(np.sqrt(np.mean(targets**2))), 0.1)
    )
    condition = float(np.linalg.cond(normalized))
    if not math.isfinite(condition):
        condition = 1e15
    return ControlledDriftModelV31.seal(
        model_id=f"model_{private_case.public_case.case_id}_{family}_{model_suffix}",
        case_id=private_case.public_case.case_id,
        state_names=private_case.public_case.state_names,
        actuator_hash=private_case.public_case.actuator.actuator_hash,
        basis_terms=terms,
        coefficient_matrix=coefficients.tolist(),
        source_observation_hashes=hashes,
        normalized_design_rank=int(np.linalg.matrix_rank(normalized, tol=1e-10)),
        normalized_condition_number=condition,
        normalized_derivative_residual=residual,
    )


def _challenge_family_v37(
    private_case: PrivateControlledDynamicsCaseV31,
    observations: list[object],
    family: FamilyV37,
    spec: ModelChallengeWorldPackSpecV37,
) -> tuple[FamilyChallengeReceiptV37, ControlledDriftModelV31]:
    losses: list[float] = []
    failures = 0
    for holdout_index, holdout in enumerate(observations):
        training = [
            item for index, item in enumerate(observations) if index != holdout_index
        ]
        model = _fit_family_v37(
            private_case,
            training,
            family,
            spec,
            model_suffix=f"fold{holdout_index}",
        )
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
        except RuntimeError:
            losses.append(10.0)
            failures += 1
    final_model = _fit_family_v37(
        private_case, observations, family, spec, model_suffix="all"
    )
    basis_count = len(final_model.basis_terms)
    rank_ratio = final_model.normalized_design_rank / max(basis_count, 1)
    values = np.asarray(losses, dtype=float)
    mean_loss = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    eligible = (
        failures == 0
        and rank_ratio >= spec.minimum_rank_ratio
        and final_model.normalized_condition_number <= spec.maximum_condition_number
        and mean_loss <= spec.maximum_cv_prediction_loss
    )
    challenge = FamilyChallengeReceiptV37.seal(
        challenge_id=f"challenge_{private_case.public_case.case_id}_{family}",
        case_id=private_case.public_case.case_id,
        family=family,
        polynomial_degree=FAMILY_DEGREES_V37[family],
        basis_term_count=basis_count,
        source_observation_hashes=final_model.source_observation_hashes,
        normalized_design_rank=final_model.normalized_design_rank,
        normalized_rank_ratio=rank_ratio,
        normalized_condition_number=final_model.normalized_condition_number,
        normalized_derivative_residual=final_model.normalized_derivative_residual,
        fold_prediction_losses=losses,
        mean_cv_prediction_loss=mean_loss,
        standard_error_cv_prediction_loss=standard_error,
        simulation_failure_count=failures,
        eligible=eligible,
        gate_codes=[
            "simulation_pass" if failures == 0 else "simulation_fail",
            "rank_pass" if rank_ratio >= spec.minimum_rank_ratio else "rank_fail",
            (
                "condition_pass"
                if final_model.normalized_condition_number <= spec.maximum_condition_number
                else "condition_fail"
            ),
            "cv_loss_pass" if mean_loss <= spec.maximum_cv_prediction_loss else "cv_loss_fail",
        ],
    )
    return challenge, final_model


def _applicability_state_v37(
    private_case: PrivateControlledDynamicsCaseV31,
    observations: list[object],
    challenges: list[FamilyChallengeReceiptV37],
) -> ApplicabilityStateV37:
    public = private_case.public_case
    states = np.vstack([np.asarray(item.states, dtype=float) for item in observations])
    inputs = np.vstack([np.asarray(item.inputs, dtype=float) for item in observations])
    lower = np.asarray(public.envelope.state_lower_bounds, dtype=float)
    upper = np.asarray(public.envelope.state_upper_bounds, dtype=float)
    coverage = np.clip(
        (np.max(states, axis=0) - np.min(states, axis=0)) / np.maximum(upper - lower, 1e-12),
        0.0,
        1.0,
    )
    maximum_input = max(
        float(np.max(np.abs([value for action in public.action_catalog for row in action.input_values for value in row]))),
        1e-12,
    )
    input_coverage = min(float(np.max(np.abs(inputs))) / maximum_input, 1.0)
    input_rank = int(np.linalg.matrix_rank(inputs, tol=1e-10))
    input_rank_ratio = input_rank / max(inputs.shape[1], 1)
    flags = sorted({flag for item in observations for flag in item.quality_flags})
    losses = np.asarray([item.mean_cv_prediction_loss for item in challenges], dtype=float)
    return ApplicabilityStateV37.seal(
        state_id=f"applicability_{public.case_id}",
        case_id=public.case_id,
        public_case_hash=public.public_hash,
        contract_hash=public.initial_contract.contract_hash,
        actuator_hash=public.actuator.actuator_hash,
        envelope_hash=public.envelope.envelope_hash,
        observation_hashes=[item.observation_hash for item in observations],
        state_dimension=len(public.state_names),
        decision_target=public.initial_contract.decision_target,
        quality_flags=flags,
        state_coverage_ratio_by_dimension=coverage.tolist(),
        input_amplitude_coverage_ratio=input_coverage,
        input_design_rank_ratio=input_rank_ratio,
        challenge_hashes=[item.challenge_hash for item in challenges],
        heldout_loss_disagreement=float(np.std(losses, ddof=0)),
    )


def _decision_v37(
    policy: ModelPortfolioPolicyV37,
    state: ApplicabilityStateV37,
    challenges: list[FamilyChallengeReceiptV37],
) -> ModelChallengeDecisionV37:
    common = {
        "decision_id": f"decision_{state.case_id}_{policy.arm}",
        "case_id": state.case_id,
        "arm": policy.arm,
        "state_hash": state.state_hash,
        "challenge_hashes": [item.challenge_hash for item in challenges],
    }
    if state.quality_flags:
        return ModelChallengeDecisionV37.seal(
            selected_family=None,
            best_cv_family=None,
            one_standard_error_threshold=None,
            decision="abstain",
            reason="deny_data_quality",
            **common,
        )
    if policy.arm == "fixed_quadratic_baseline":
        quadratic = next(
            item for item in challenges if item.family == "quadratic_interaction_ode"
        )
        return ModelChallengeDecisionV37.seal(
            selected_family="quadratic_interaction_ode",
            best_cv_family=quadratic.family,
            one_standard_error_threshold=quadratic.mean_cv_prediction_loss,
            decision="select",
            reason="fixed_quadratic",
            **common,
        )
    eligible = [item for item in challenges if item.eligible]
    if not eligible:
        return ModelChallengeDecisionV37.seal(
            selected_family=None,
            best_cv_family=None,
            one_standard_error_threshold=None,
            decision="abstain",
            reason="needs_evidence_no_eligible_family",
            **common,
        )
    best = min(eligible, key=lambda item: (item.mean_cv_prediction_loss, item.basis_term_count))
    threshold = best.mean_cv_prediction_loss + best.standard_error_cv_prediction_loss
    within = [item for item in eligible if item.mean_cv_prediction_loss <= threshold]
    selected = min(within, key=lambda item: (item.basis_term_count, item.mean_cv_prediction_loss))
    return ModelChallengeDecisionV37.seal(
        selected_family=selected.family,
        best_cv_family=best.family,
        one_standard_error_threshold=threshold,
        decision="select",
        reason="one_standard_error_challenge",
        **common,
    )


def execute_model_challenge_policy_v37(
    spec: ModelChallengeWorldPackSpecV37,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    policy: ModelPortfolioPolicyV37,
    *,
    executed_at: datetime,
) -> ModelChallengeBundleV37:
    spec.assert_sealed()
    private_pack.assert_sealed()
    policy.assert_sealed()
    if private_pack.spec_hash != spec.spec_hash:
        raise ValueError("V3.7 private pack belongs to another protocol")
    expected_policy_hash = (
        spec.baseline_policy_hash
        if policy.arm == "fixed_quadratic_baseline"
        else spec.candidate_policy_hash
    )
    if policy.policy_hash != expected_policy_hash:
        raise ValueError("V3.7 policy is not frozen in protocol")
    receipts: list[ModelChallengeCaseReceiptV37] = []
    for private_case in private_pack.cases:
        observations = _shared_observations_v37(private_case, spec)
        challenges: list[FamilyChallengeReceiptV37] = []
        models: dict[FamilyV37, ControlledDriftModelV31] = {}
        for family in FAMILIES_V37:
            challenge, model = _challenge_family_v37(
                private_case, observations, family, spec
            )
            challenges.append(challenge)
            models[family] = model
        state = _applicability_state_v37(private_case, observations, challenges)
        decision = _decision_v37(policy, state, challenges)
        selected_model = (
            models[decision.selected_family]
            if decision.selected_family is not None
            else None
        )
        receipts.append(ModelChallengeCaseReceiptV37.seal(
            receipt_id=f"receipt_{private_case.public_case.case_id}_{policy.arm}",
            case_id=private_case.public_case.case_id,
            public_case_hash=private_case.public_case.public_hash,
            policy_hash=policy.policy_hash,
            arm=policy.arm,
            observation_hashes=[item.observation_hash for item in observations],
            applicability_state=state,
            challenges=challenges,
            decision=decision,
            selected_model=selected_model,
            executed_at=executed_at,
        ))
    return ModelChallengeBundleV37.seal(
        bundle_id=f"bundle_{policy.arm}_{spec.experiment_id}",
        spec_hash=spec.spec_hash,
        private_pack_hash=private_pack.pack_hash,
        policy_hash=policy.policy_hash,
        arm=policy.arm,
        case_receipts=receipts,
        created_at=executed_at,
    )


class PrivateModelChallengeCaseResultV37(StrictModel):
    schema_version: Literal["3.7"] = "3.7"
    case_id: Identifier
    mechanism: MechanismV31
    performance_eligible: bool
    baseline_selected_family: FamilyV37 | None
    candidate_selected_family: FamilyV37 | None
    expected_minimal_family: FamilyV37
    candidate_routing_correct: bool
    baseline_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
    candidate_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
    target_loss_improvement: Annotated[float, Field(allow_inf_nan=False)] | None
    candidate_abstained_without_data_failure: bool


class ModelChallengeEvolutionReportV37(StrictModel):
    schema_version: Literal["3.7"] = "3.7"
    evolution_id: Identifier
    spec_hash: Sha256
    baseline_bundle_hash: Sha256
    candidate_bundle_hash: Sha256
    case_results: list[PrivateModelChallengeCaseResultV37] = Field(
        min_length=64, max_length=64
    )
    eligible_case_count: Annotated[int, Field(ge=1)]
    quality_abstention_count: Annotated[int, Field(ge=0)]
    candidate_model_count: Annotated[int, Field(ge=0)]
    candidate_coverage_rate: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    routing_accuracy: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    macro_target_loss_improvement: Annotated[float, Field(allow_inf_nan=False)]
    macro_ci_low: Annotated[float, Field(allow_inf_nan=False)]
    macro_ci_high: Annotated[float, Field(allow_inf_nan=False)]
    mechanism_mean_improvements: dict[Identifier, Annotated[float, Field(allow_inf_nan=False)]]
    material_negative_transfer_count: Annotated[int, Field(ge=0)]
    material_negative_transfer_rate_upper: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ]
    baseline_max_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    candidate_max_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    gates: dict[Identifier, bool]
    ready_for_non_nested_extension: bool
    status: Literal[
        "model_challenge_ready_for_non_nested_extension_v37",
        "model_challenge_failed_v37",
    ]
    router_experiment_permitted: Literal[False] = False
    overall_qualification_permitted: Literal[False] = False
    confirmation_permitted: Literal[False] = False
    real_world_authorization_permitted: Literal[False] = False
    created_at: datetime
    evolution_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "ModelChallengeEvolutionReportV37":
        _assert_timezone(self.created_at, "created_at")
        ready = all(self.gates.values())
        if self.ready_for_non_nested_extension != ready:
            raise ValueError("V3.7 readiness disagrees with gates")
        expected = (
            "model_challenge_ready_for_non_nested_extension_v37"
            if ready else "model_challenge_failed_v37"
        )
        if self.status != expected:
            raise ValueError("V3.7 status disagrees with gates")
        if self.evolution_hash and self.evolution_hash != self.content_hash():
            raise ValueError("evolution_hash does not match V3.7 report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evolution_hash")

    def assert_sealed(self) -> None:
        if not self.evolution_hash or self.evolution_hash != self.content_hash():
            raise ValueError("V3.7 report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ModelChallengeEvolutionReportV37":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evolution_hash"}),
            evolution_hash=draft.content_hash(),
        )


def _cluster_macro_bootstrap_v37(
    values: dict[tuple[int, MechanismV31], float],
    spec: ModelChallengeWorldPackSpecV37,
) -> tuple[float, float]:
    seeds = list(spec.seeds)
    random = np.random.default_rng(spec.bootstrap_seed)
    draws: list[float] = []
    for _ in range(spec.bootstrap_replicates):
        sampled = random.choice(seeds, size=len(seeds), replace=True)
        mechanism_means: list[float] = []
        for mechanism in spec.mechanisms:
            mechanism_values = [
                values[(int(seed), mechanism)]
                for seed in sampled
                if (int(seed), mechanism) in values
            ]
            if mechanism_values:
                mechanism_means.append(float(np.mean(mechanism_values)))
        draws.append(float(np.mean(mechanism_means)))
    low, high = np.quantile(np.asarray(draws, dtype=float), [0.025, 0.975])
    return float(low), float(high)


def evaluate_model_challenge_worldpack_v37(
    spec: ModelChallengeWorldPackSpecV37,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    baseline: ModelChallengeBundleV37,
    candidate: ModelChallengeBundleV37,
    *,
    evaluated_at: datetime,
) -> ModelChallengeEvolutionReportV37:
    for artifact in (spec, private_pack, baseline, candidate):
        artifact.assert_sealed()
    if baseline.arm != "fixed_quadratic_baseline":
        raise ValueError("V3.7 baseline arm changed")
    if candidate.arm != "applicability_challenge_candidate":
        raise ValueError("V3.7 candidate arm changed")
    if baseline.spec_hash != spec.spec_hash or candidate.spec_hash != spec.spec_hash:
        raise ValueError("V3.7 bundle spec binding differs")
    private_by_id = {item.public_case.case_id: item for item in private_pack.cases}
    baseline_by_id = {item.case_id: item for item in baseline.case_receipts}
    candidate_by_id = {item.case_id: item for item in candidate.case_receipts}
    if set(private_by_id) != set(baseline_by_id) or set(private_by_id) != set(candidate_by_id):
        raise ValueError("V3.7 bundle case coverage differs")
    public_context_identity = True
    case_results: list[PrivateModelChallengeCaseResultV37] = []
    improvements: dict[tuple[int, MechanismV31], float] = {}
    values_by_mechanism: dict[MechanismV31, list[float]] = {
        mechanism: [] for mechanism in spec.mechanisms
    }
    baseline_losses: list[float] = []
    candidate_losses: list[float] = []
    eligible_count = 0
    quality_abstentions = 0
    candidate_model_count = 0
    routing_correct_count = 0
    for case_id, private_case in private_by_id.items():
        base = baseline_by_id[case_id]
        cand = candidate_by_id[case_id]
        if (
            base.public_case_hash != cand.public_case_hash
            or base.observation_hashes != cand.observation_hashes
            or base.applicability_state.state_hash != cand.applicability_state.state_hash
            or [item.challenge_hash for item in base.challenges]
            != [item.challenge_hash for item in cand.challenges]
        ):
            public_context_identity = False
        data_passed = not base.applicability_state.quality_flags
        expected_family = EXPECTED_MINIMAL_FAMILY_V37[private_case.mechanism]
        if not data_passed:
            quality_abstentions += 1
            case_results.append(PrivateModelChallengeCaseResultV37(
                case_id=case_id,
                mechanism=private_case.mechanism,
                performance_eligible=False,
                baseline_selected_family=base.decision.selected_family,
                candidate_selected_family=cand.decision.selected_family,
                expected_minimal_family=expected_family,
                candidate_routing_correct=False,
                baseline_target_loss=None,
                candidate_target_loss=None,
                target_loss_improvement=None,
                candidate_abstained_without_data_failure=False,
            ))
            continue
        eligible_count += 1
        if base.selected_model is None:
            raise ValueError("V3.7 quality-passed baseline cannot abstain")
        baseline_loss = _target_loss_v31(private_case, base.selected_model, spec)
        baseline_losses.append(baseline_loss)
        candidate_abstained = cand.selected_model is None
        if candidate_abstained:
            candidate_loss = 10.0
        else:
            candidate_model_count += 1
            candidate_loss = _target_loss_v31(private_case, cand.selected_model, spec)
        candidate_losses.append(candidate_loss)
        improvement = baseline_loss - candidate_loss
        seed = int(case_id.rsplit("_", 1)[1])
        improvements[(seed, private_case.mechanism)] = improvement
        values_by_mechanism[private_case.mechanism].append(improvement)
        routing_correct = cand.decision.selected_family == expected_family
        routing_correct_count += int(routing_correct)
        case_results.append(PrivateModelChallengeCaseResultV37(
            case_id=case_id,
            mechanism=private_case.mechanism,
            performance_eligible=True,
            baseline_selected_family=base.decision.selected_family,
            candidate_selected_family=cand.decision.selected_family,
            expected_minimal_family=expected_family,
            candidate_routing_correct=routing_correct,
            baseline_target_loss=baseline_loss,
            candidate_target_loss=candidate_loss,
            target_loss_improvement=improvement,
            candidate_abstained_without_data_failure=candidate_abstained,
        ))
    mechanism_means = {
        mechanism: float(np.mean(values))
        for mechanism, values in values_by_mechanism.items()
    }
    macro = float(np.mean(list(mechanism_means.values())))
    ci_low, ci_high = _cluster_macro_bootstrap_v37(improvements, spec)
    flat = np.asarray(list(improvements.values()), dtype=float)
    negative_count = int(np.sum(flat < -spec.material_negative_transfer))
    if negative_count == len(flat):
        negative_upper = 1.0
    else:
        negative_upper = float(beta.ppf(
            0.95, negative_count + 1, len(flat) - negative_count
        ))
    candidate_coverage = candidate_model_count / eligible_count
    routing_accuracy = routing_correct_count / eligible_count
    gates = {
        "public_context_identity": public_context_identity,
        "quality_abstention_exact": (
            quality_abstentions == spec.expected_quality_abstention_count
        ),
        "candidate_coverage": candidate_coverage >= spec.minimum_candidate_coverage,
        "routing_accuracy": routing_accuracy >= spec.minimum_routing_accuracy,
        "macro_ci_lower_positive": ci_low > 0.0,
        "mechanism_non_regression": all(
            value >= -spec.maximum_mechanism_regression
            for value in mechanism_means.values()
        ),
        "negative_transfer_bound": (
            negative_upper <= spec.maximum_negative_transfer_rate
        ),
        "max_loss_non_regression": max(candidate_losses) <= max(baseline_losses) + 1e-12,
        "no_private_policy_inputs": all(
            not receipt.applicability_state.private_mechanism_seen
            and not receipt.applicability_state.private_probe_seen
            and not receipt.applicability_state.private_target_loss_seen
            for receipt in [*baseline.case_receipts, *candidate.case_receipts]
        ),
    }
    ready = all(gates.values())
    return ModelChallengeEvolutionReportV37.seal(
        evolution_id=f"evolution_{spec.experiment_id}",
        spec_hash=spec.spec_hash,
        baseline_bundle_hash=baseline.bundle_hash,
        candidate_bundle_hash=candidate.bundle_hash,
        case_results=case_results,
        eligible_case_count=eligible_count,
        quality_abstention_count=quality_abstentions,
        candidate_model_count=candidate_model_count,
        candidate_coverage_rate=candidate_coverage,
        routing_accuracy=routing_accuracy,
        macro_target_loss_improvement=macro,
        macro_ci_low=ci_low,
        macro_ci_high=ci_high,
        mechanism_mean_improvements=mechanism_means,
        material_negative_transfer_count=negative_count,
        material_negative_transfer_rate_upper=negative_upper,
        baseline_max_target_loss=max(baseline_losses),
        candidate_max_target_loss=max(candidate_losses),
        gates=gates,
        ready_for_non_nested_extension=ready,
        status=(
            "model_challenge_ready_for_non_nested_extension_v37"
            if ready else "model_challenge_failed_v37"
        ),
        created_at=evaluated_at,
    )


class ModelChallengeManifestV37(StrictModel):
    schema_version: Literal["3.7"] = "3.7"
    run_id: Identifier
    artifact_refs: list[ArtifactRef] = Field(min_length=8, max_length=8)
    terminal_status: Literal[
        "model_challenge_ready_for_non_nested_extension_v37",
        "model_challenge_failed_v37",
    ]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "ModelChallengeManifestV37":
        _assert_timezone(self.created_at, "created_at")
        if len({item.kind for item in self.artifact_refs}) != len(self.artifact_refs):
            raise ValueError("V3.7 manifest artifact kinds must be unique")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match V3.7 manifest")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("V3.7 manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ModelChallengeManifestV37":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class ModelChallengeOutcomeV37:
    store: RunStore
    method_evidence: ModelChallengeMethodEvidenceV37
    spec: ModelChallengeWorldPackSpecV37
    baseline_policy: ModelPortfolioPolicyV37
    candidate_policy: ModelPortfolioPolicyV37
    private_pack: PrivateControlledDynamicsWorldPackV31
    baseline_bundle: ModelChallengeBundleV37
    candidate_bundle: ModelChallengeBundleV37
    evolution_report: ModelChallengeEvolutionReportV37
    manifest: ModelChallengeManifestV37


def run_model_challenge_worldpack_v37(
    output_root: str | Path,
    *,
    method_evidence: ModelChallengeMethodEvidenceV37,
    spec: ModelChallengeWorldPackSpecV37,
    baseline_policy: ModelPortfolioPolicyV37,
    candidate_policy: ModelPortfolioPolicyV37,
    evaluated_at: datetime | None = None,
    run_id: str | None = None,
) -> ModelChallengeOutcomeV37:
    method_evidence.assert_sealed()
    spec.assert_sealed()
    baseline_policy.assert_sealed()
    candidate_policy.assert_sealed()
    if (
        spec.method_evidence_hash != method_evidence.evidence_hash
        or baseline_policy.method_evidence_hash != method_evidence.evidence_hash
        or candidate_policy.method_evidence_hash != method_evidence.evidence_hash
    ):
        raise ValueError("V3.7 method evidence binding differs")
    if (
        spec.baseline_policy_hash != baseline_policy.policy_hash
        or spec.candidate_policy_hash != candidate_policy.policy_hash
    ):
        raise ValueError("V3.7 policies are not frozen in protocol")
    if baseline_policy.arm != "fixed_quadratic_baseline":
        raise ValueError("V3.7 baseline policy arm changed")
    if candidate_policy.arm != "applicability_challenge_candidate":
        raise ValueError("V3.7 candidate policy arm changed")
    at = evaluated_at or datetime.now(timezone.utc)
    store = RunStore(
        output_root,
        run_id=run_id or f"model-challenge-v37-{uuid4().hex[:10]}",
    )
    refs = [
        store.put_artifact("model_challenge_method_evidence_v37", method_evidence),
        store.put_artifact("model_challenge_spec_v37", spec),
        store.put_artifact("model_challenge_baseline_policy_v37", baseline_policy),
        store.put_artifact("model_challenge_candidate_policy_v37", candidate_policy),
    ]
    store.emit("model_challenge_v37_protocol_frozen_before_private_pack", {
        "spec_hash": spec.spec_hash,
        "method_evidence_hash": method_evidence.evidence_hash,
        "baseline_policy_hash": baseline_policy.policy_hash,
        "candidate_policy_hash": candidate_policy.policy_hash,
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
    evolution = evaluate_model_challenge_worldpack_v37(
        spec, private_pack, baseline, candidate, evaluated_at=at
    )
    refs.extend([
        store.put_artifact("private_model_challenge_worldpack_v37", private_pack),
        store.put_artifact("model_challenge_baseline_bundle_v37", baseline),
        store.put_artifact("model_challenge_candidate_bundle_v37", candidate),
        store.put_artifact("model_challenge_evolution_report_v37", evolution),
    ])
    manifest = ModelChallengeManifestV37.seal(
        run_id=store.run_id,
        artifact_refs=refs,
        terminal_status=evolution.status,
        created_at=at,
    )
    manifest_ref = store.put_artifact("model_challenge_manifest_v37", manifest)
    store.emit("model_challenge_v37_worldpack_adjudicated", {
        "manifest_ref": manifest_ref.model_dump(mode="json")
    })
    if not verify_model_challenge_run_v37(store.run_directory):
        raise RuntimeError("V3.7 run failed independent verification")
    return ModelChallengeOutcomeV37(
        store,
        method_evidence,
        spec,
        baseline_policy,
        candidate_policy,
        private_pack,
        baseline,
        candidate,
        evolution,
        manifest,
    )


def verify_model_challenge_run_v37(run_directory: str | Path) -> bool:
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
            item for item in committed if item.kind == "model_challenge_manifest_v37"
        ]
        if len(manifest_refs) != 1:
            return False
        manifest = ModelChallengeManifestV37.model_validate(
            store.load_artifact(manifest_refs[0])
        )
        manifest.assert_sealed()

        def load_one(kind: str, model):
            references = [item for item in manifest.artifact_refs if item.kind == kind]
            if len(references) != 1:
                raise RuntimeError(f"V3.7 manifest needs exactly one {kind}")
            return model.model_validate(store.load_artifact(references[0]))

        method = load_one(
            "model_challenge_method_evidence_v37", ModelChallengeMethodEvidenceV37
        )
        spec = load_one("model_challenge_spec_v37", ModelChallengeWorldPackSpecV37)
        baseline_policy = load_one(
            "model_challenge_baseline_policy_v37", ModelPortfolioPolicyV37
        )
        candidate_policy = load_one(
            "model_challenge_candidate_policy_v37", ModelPortfolioPolicyV37
        )
        private_pack = load_one(
            "private_model_challenge_worldpack_v37", PrivateControlledDynamicsWorldPackV31
        )
        baseline = load_one(
            "model_challenge_baseline_bundle_v37", ModelChallengeBundleV37
        )
        candidate = load_one(
            "model_challenge_candidate_bundle_v37", ModelChallengeBundleV37
        )
        evolution = load_one(
            "model_challenge_evolution_report_v37", ModelChallengeEvolutionReportV37
        )
        for artifact in (
            method, spec, baseline_policy, candidate_policy, private_pack,
            baseline, candidate, evolution, manifest,
        ):
            artifact.assert_sealed()
        if manifest.run_id != store.run_id:
            return False
        if (
            spec.method_evidence_hash != method.evidence_hash
            or spec.baseline_policy_hash != baseline_policy.policy_hash
            or spec.candidate_policy_hash != candidate_policy.policy_hash
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
        recomputed = evaluate_model_challenge_worldpack_v37(
            spec,
            private_pack,
            baseline,
            candidate,
            evaluated_at=evolution.created_at,
        )
        if recomputed.evolution_hash != evolution.evolution_hash:
            return False
        if any(
            any(word in item.kind for word in ("qualification", "confirmation", "router"))
            for item in manifest.artifact_refs
        ):
            return False
        freezes = [
            event for event in events
            if event["event_type"] == "model_challenge_v37_protocol_frozen_before_private_pack"
        ]
        private_events = [
            event for event in events
            if event["event_type"] == "artifact_committed"
            and event["payload"]["kind"] == "private_model_challenge_worldpack_v37"
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
