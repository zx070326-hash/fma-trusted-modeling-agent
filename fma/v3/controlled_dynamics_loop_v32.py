from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Annotated, Literal
from uuid import uuid4

import numpy as np
from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore
from fma.v2.dynamics_ir import evaluate_polynomial_library
from fma.v2.schemas import Identifier, Sha256, _assert_timezone

from .controlled_dynamics_loop import (
    ArmV31,
    MECHANISMS_V31,
    ControlledDynamicsCaseReceiptV31,
    ControlledDynamicsContractV31,
    ControlledDynamicsReportV31,
    ControlledDynamicsSelectionBundleV31,
    ControlledDynamicsStepReceiptV31,
    PrivateControlledDynamicsCaseV31,
    PrivateControlledDynamicsWorldPackV31,
    RouteLayerV31,
    TargetClarificationEvidenceV31,
    _bootstrap_ensemble_v31,
    _d_opt_gain_v31,
    _ensemble_paths_v31,
    _fit_model_v31,
    _input_at_time_v31,
    _observation_arrays_v31,
    _permission_v31,
    _target_loss_v31,
    evaluate_controlled_dynamics_worldpack_v31,
    generate_private_controlled_dynamics_worldpack_v31,
)
from .experiment_ir import (
    ControlledObservationReceiptV31,
    DecisionTargetV31,
    ExperimentPermissionDecisionV31,
    PiecewiseConstantInputActionV31,
    validate_action_against_envelope_v31,
)


EXPLORATORY_SEEDS_V32 = (
    10007, 10061, 10103, 10151, 10211, 10259, 10301, 10357,
    10427, 10477, 10531, 10589, 10639, 10691, 10739, 10799,
)


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class GoalPosteriorRiskPolicyV32(StrictModel):
    schema_version: Literal["3.2"] = "3.2"
    policy_id: Identifier
    arm: ArmV31
    selection_rule: Literal[
        "prefrozen_random_without_replacement",
        "clarify_then_robust_goal_posterior_risk",
    ]
    may_reformulate_problem: bool
    maximum_actions: Literal[3] = 3
    known_actuator_required: Literal[True] = True
    prior_v311_evolution_report_hash: Sha256
    prior_epistemic_qualification_hash: Sha256
    prior_active_design_qualification_hash: Sha256
    method_evidence_hash: Sha256
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "GoalPosteriorRiskPolicyV32":
        expected = {
            "random_bounded_inputs": (
                "prefrozen_random_without_replacement", False,
            ),
            "goal_oriented_epistemic_control": (
                "clarify_then_robust_goal_posterior_risk", True,
            ),
        }[self.arm]
        if (self.selection_rule, self.may_reformulate_problem) != expected:
            raise ValueError("V3.2 arm and policy behavior disagree")
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("policy_hash does not match V3.2 policy")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("V3.2 policy is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "GoalPosteriorRiskPolicyV32":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"policy_hash"}),
            policy_hash=draft.content_hash(),
        )


class ControlledDynamicsWorldPackSpecV32(StrictModel):
    schema_version: Literal["3.2"] = "3.2"
    experiment_id: Identifier
    phase: Literal["exploratory"] = "exploratory"
    mechanisms: list[Literal[
        "exponential_decay", "logistic_growth",
        "damped_oscillator", "duffing_oscillator",
    ]] = Field(min_length=4, max_length=4)
    seeds: list[int] = Field(min_length=16, max_length=16)
    action_budget: Literal[3] = 3
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
    bootstrap_seed: Literal[320722] = 320722
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
    prior_v311_evolution_report_hash: Sha256
    frozen_delta: Literal[
        "heuristic_utility_to_goal_posterior_risk_only"
    ] = "heuristic_utility_to_goal_posterior_risk_only"
    frozen_at: datetime
    spec_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "ControlledDynamicsWorldPackSpecV32":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.mechanisms != list(MECHANISMS_V31):
            raise ValueError("V3.2 requires the frozen mechanism order")
        if tuple(self.seeds) != EXPLORATORY_SEEDS_V32:
            raise ValueError("V3.2 seeds do not match the frozen exploratory set")
        if self.goal_initial_state_scales != [0.75, 1.0, 1.25]:
            raise ValueError("V3.2 public goal initial-state scales changed")
        if not math.isclose(
            (self.trajectory_points - 1) * self.time_step,
            self.segment_count * self.segment_duration,
            abs_tol=1e-12,
        ):
            raise ValueError("V3.2 input segments do not cover the trajectory")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match V3.2 protocol")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "spec_hash")

    def assert_sealed(self) -> None:
        if not self.spec_hash or self.spec_hash != self.content_hash():
            raise ValueError("V3.2 protocol is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsWorldPackSpecV32":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"spec_hash"}),
            spec_hash=draft.content_hash(),
        )


class GoalPosteriorRiskAcquisitionReceiptV32(StrictModel):
    schema_version: Literal["3.2"] = "3.2"
    acquisition_id: Identifier
    case_id: Identifier
    action_hash: Sha256
    decision_target: DecisionTargetV31
    belief_precision_hash: Sha256
    goal_operator_hash: Sha256
    goal_feature_row_count: Annotated[int, Field(ge=1)]
    current_goal_posterior_risk: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    predicted_goal_posterior_risk: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    mean_fractional_goal_risk_reduction: Annotated[
        float, Field(ge=-1e-10, le=1.0, allow_inf_nan=False)
    ]
    robust_fractional_goal_risk_reduction: Annotated[
        float, Field(ge=-1e-10, le=1.0, allow_inf_nan=False)
    ]
    goal_risk_reduction_dispersion: Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ]
    d_optimal_gain_diagnostic: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    model_disagreement_diagnostic: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    empirical_prediction_risk: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ]
    ranking_score: Annotated[float, Field(allow_inf_nan=False)]
    action_cost: Literal[1] = 1
    admissible: bool
    gate_codes: list[Literal[
        "known_actuator", "peak_equal", "energy_equal", "switch_equal",
        "cost_equal", "empirical_risk_pass", "empirical_risk_fail",
    ]] = Field(min_length=6, max_length=6)
    covariance_proxy: Literal[
        "ridge_laplace_feature_covariance_not_calibrated_posterior"
    ] = "ridge_laplace_feature_covariance_not_calibrated_posterior"
    goal_operator_uses_private_probe: Literal[False] = False
    posterior_calibrated: Literal[False] = False
    formal_safety_proven: Literal[False] = False
    acquisition_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_acquisition(self) -> "GoalPosteriorRiskAcquisitionReceiptV32":
        risk_pass = "empirical_risk_pass" in self.gate_codes
        if self.admissible != risk_pass:
            raise ValueError("V3.2 acquisition admissibility must follow risk gate")
        if not math.isclose(
            self.ranking_score,
            self.robust_fractional_goal_risk_reduction,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("V3.2 ranking must use only robust goal-risk reduction")
        if self.predicted_goal_posterior_risk > self.current_goal_posterior_risk + 1e-10:
            raise ValueError("V3.2 covariance update increased mean posterior risk")
        if self.acquisition_hash and self.acquisition_hash != self.content_hash():
            raise ValueError("acquisition_hash does not match V3.2 receipt")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "acquisition_hash")

    def assert_sealed(self) -> None:
        if not self.acquisition_hash or self.acquisition_hash != self.content_hash():
            raise ValueError("V3.2 acquisition receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "GoalPosteriorRiskAcquisitionReceiptV32":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"acquisition_hash"}),
            acquisition_hash=draft.content_hash(),
        )


class ControlledDynamicsStepReceiptV32(ControlledDynamicsStepReceiptV31):
    schema_version: Literal["3.2"] = "3.2"
    acquisition_receipts: list[GoalPosteriorRiskAcquisitionReceiptV32] = Field(
        default_factory=list
    )


class ControlledDynamicsCaseReceiptV32(ControlledDynamicsCaseReceiptV31):
    schema_version: Literal["3.2"] = "3.2"
    steps: list[ControlledDynamicsStepReceiptV32] = Field(min_length=1, max_length=3)


class ControlledDynamicsSelectionBundleV32(ControlledDynamicsSelectionBundleV31):
    schema_version: Literal["3.2"] = "3.2"
    case_receipts: list[ControlledDynamicsCaseReceiptV32] = Field(
        min_length=64, max_length=64
    )


class GoalPosteriorRiskEvolutionReportV32(StrictModel):
    schema_version: Literal["3.2"] = "3.2"
    evolution_id: Identifier
    spec_hash: Sha256
    base_adjudication_report: ControlledDynamicsReportV31
    prior_v311_evolution_report_hash: Sha256
    single_component_delta: Literal[
        "heuristic_utility_to_goal_posterior_risk_only"
    ] = "heuristic_utility_to_goal_posterior_risk_only"
    acquisition_gates: dict[Identifier, bool]
    acquisition_candidate_ready: bool
    router_gate_passed: bool
    router_accuracy: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    status: Literal[
        "acquisition_candidate_ready_for_router_evolution_v32",
        "acquisition_candidate_failed_v32",
    ]
    estimator_changed: Literal[False] = False
    action_catalog_changed: Literal[False] = False
    action_horizon_changed: Literal[False] = False
    risk_gate_changed: Literal[False] = False
    statistical_gate_changed: Literal[False] = False
    model_router_changed: Literal[False] = False
    overall_qualification_permitted: Literal[False] = False
    confirmation_permitted: Literal[False] = False
    created_at: datetime
    evolution_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_evolution(self) -> "GoalPosteriorRiskEvolutionReportV32":
        _assert_timezone(self.created_at, "created_at")
        self.base_adjudication_report.assert_sealed()
        if self.base_adjudication_report.spec_hash != self.spec_hash:
            raise ValueError("V3.2 wrapper is bound to another adjudication")
        ready = all(self.acquisition_gates.values())
        if self.acquisition_candidate_ready != ready:
            raise ValueError("V3.2 acquisition readiness disagrees with gates")
        expected = (
            "acquisition_candidate_ready_for_router_evolution_v32"
            if ready else "acquisition_candidate_failed_v32"
        )
        if self.status != expected:
            raise ValueError("V3.2 status disagrees with acquisition gates")
        if self.router_gate_passed != self.base_adjudication_report.gates["routing_accuracy"]:
            raise ValueError("V3.2 router status was not inherited unchanged")
        if self.evolution_hash and self.evolution_hash != self.content_hash():
            raise ValueError("evolution_hash does not match V3.2 report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evolution_hash")

    def assert_sealed(self) -> None:
        if not self.evolution_hash or self.evolution_hash != self.content_hash():
            raise ValueError("V3.2 evolution report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "GoalPosteriorRiskEvolutionReportV32":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evolution_hash"}),
            evolution_hash=draft.content_hash(),
        )


class ControlledDynamicsManifestV32(StrictModel):
    schema_version: Literal["3.2"] = "3.2"
    run_id: Identifier
    artifact_refs: list[ArtifactRef] = Field(min_length=8)
    terminal_status: Literal[
        "acquisition_candidate_ready_for_router_evolution_v32",
        "acquisition_candidate_failed_v32",
    ]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "ControlledDynamicsManifestV32":
        _assert_timezone(self.created_at, "created_at")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match V3.2 manifest")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("V3.2 manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsManifestV32":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class ControlledDynamicsOutcomeV32:
    store: RunStore
    spec: ControlledDynamicsWorldPackSpecV32
    private_pack: PrivateControlledDynamicsWorldPackV31
    baseline_policy: GoalPosteriorRiskPolicyV32
    candidate_policy: GoalPosteriorRiskPolicyV32
    baseline_bundle: ControlledDynamicsSelectionBundleV32
    candidate_bundle: ControlledDynamicsSelectionBundleV32
    evolution_report: GoalPosteriorRiskEvolutionReportV32
    manifest: ControlledDynamicsManifestV32


def default_controlled_dynamics_policies_v32(
    *,
    prior_v311_evolution_report_hash: str,
    prior_epistemic_qualification_hash: str,
    prior_active_design_qualification_hash: str,
    method_evidence_hash: str,
) -> tuple[GoalPosteriorRiskPolicyV32, GoalPosteriorRiskPolicyV32]:
    shared = dict(
        prior_v311_evolution_report_hash=prior_v311_evolution_report_hash,
        prior_epistemic_qualification_hash=prior_epistemic_qualification_hash,
        prior_active_design_qualification_hash=prior_active_design_qualification_hash,
        method_evidence_hash=method_evidence_hash,
    )
    return (
        GoalPosteriorRiskPolicyV32.seal(
            policy_id="random_bounded_inputs_v32",
            arm="random_bounded_inputs",
            selection_rule="prefrozen_random_without_replacement",
            may_reformulate_problem=False,
            **shared,
        ),
        GoalPosteriorRiskPolicyV32.seal(
            policy_id="goal_posterior_risk_control_v32",
            arm="goal_oriented_epistemic_control",
            selection_rule="clarify_then_robust_goal_posterior_risk",
            may_reformulate_problem=True,
            **shared,
        ),
    )


def default_controlled_dynamics_exploratory_spec_v32(
    *,
    baseline_policy_hash: str,
    candidate_policy_hash: str,
    method_evidence_hash: str,
    prior_epistemic_qualification_hash: str,
    prior_active_design_qualification_hash: str,
    prior_v311_evolution_report_hash: str,
    frozen_at: datetime | None = None,
) -> ControlledDynamicsWorldPackSpecV32:
    return ControlledDynamicsWorldPackSpecV32.seal(
        experiment_id="controlled_dynamics_goal_risk_exploratory_v32",
        mechanisms=list(MECHANISMS_V31),
        seeds=list(EXPLORATORY_SEEDS_V32),
        baseline_policy_hash=baseline_policy_hash,
        candidate_policy_hash=candidate_policy_hash,
        method_evidence_hash=method_evidence_hash,
        prior_epistemic_qualification_hash=prior_epistemic_qualification_hash,
        prior_active_design_qualification_hash=prior_active_design_qualification_hash,
        prior_v311_evolution_report_hash=prior_v311_evolution_report_hash,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


def _ensemble_paths_from_initial_v32(
    public,
    initial_state: list[float],
    input_values: list[list[float]],
    terms,
    ensemble: np.ndarray,
    spec: ControlledDynamicsWorldPackSpecV32,
) -> np.ndarray:
    state = np.repeat(
        np.asarray(initial_state, dtype=float)[np.newaxis, :],
        ensemble.shape[0],
        axis=0,
    )
    actuator = np.asarray(public.actuator.matrix, dtype=float)
    paths = [state.copy()]
    for index in range(1, spec.trajectory_points):
        time = (index - 1) * spec.time_step
        control = _input_at_time_v31(input_values, time, spec.segment_duration)
        library = evaluate_polynomial_library(state, terms)
        drift = np.einsum("bt,bet->be", library, ensemble)
        state = state + spec.time_step * (
            drift + (actuator @ control)[np.newaxis, :]
        )
        state = np.clip(state, -1e4, 1e4)
        paths.append(state.copy())
    return np.stack(paths, axis=1)


def _public_goal_initials_v32(public, spec: ControlledDynamicsWorldPackSpecV32) -> list[list[float]]:
    initial = np.asarray(public.initial_state, dtype=float)
    lower = np.asarray(public.envelope.state_lower_bounds, dtype=float)
    upper = np.asarray(public.envelope.state_upper_bounds, dtype=float)
    width = upper - lower
    margin = spec.goal_envelope_margin_fraction * width
    initials: list[list[float]] = []
    for scale in spec.goal_initial_state_scales:
        state = initial * scale
        zero = np.abs(initial) < 1e-12
        state[zero] = initial[zero] + (
            (scale - 1.0) * spec.goal_zero_component_range_fraction * width[zero]
        )
        state = np.clip(state, lower + margin, upper - margin)
        initials.append(state.tolist())
    return initials


def _goal_feature_matrices_v32(
    public,
    target: DecisionTargetV31,
    terms,
    ensemble: np.ndarray,
    spec: ControlledDynamicsWorldPackSpecV32,
) -> list[np.ndarray]:
    trim = spec.savgol_window // 2
    path_sets: list[np.ndarray] = []
    if target == "controlled_response_prediction":
        path_sets = [
            _ensemble_paths_v31(public, action, terms, ensemble, spec)
            for action in public.action_catalog
        ]
    else:
        zero_inputs = [[0.0] for _ in range(spec.segment_count)]
        path_sets = [
            _ensemble_paths_from_initial_v32(
                public, initial, zero_inputs, terms, ensemble, spec
            )
            for initial in _public_goal_initials_v32(public, spec)
        ]
    return [
        np.vstack([
            evaluate_polynomial_library(paths[member, trim:-trim], terms)
            for paths in path_sets
        ])
        for member in range(ensemble.shape[0])
    ]


def _covariance_factor_v32(precision: np.ndarray, alpha: float) -> np.ndarray:
    """Return B with B B' = precision^-1 using a symmetric eigensolve.

    The feature dimension is small, but predicted nonlinear trajectories can
    make a direct inverse lose positive definiteness.  The eigenvalue floor is
    the already frozen ridge prior, not an additional tuning parameter.
    """
    symmetric = 0.5 * (precision + precision.T)
    values, vectors = np.linalg.eigh(symmetric)
    values = np.maximum(values, alpha)
    return vectors @ np.diag(1.0 / np.sqrt(values))


def _posterior_risk_after_features_v32(
    whitened_goal: np.ndarray,
    whitened_proposed: np.ndarray,
) -> float:
    """Evaluate tr(G C_post G')/n via an SVD in prior-whitened space."""
    _, singular_values, right = np.linalg.svd(
        whitened_proposed, full_matrices=True
    )
    weights = np.ones(right.shape[0], dtype=float)
    weights[:len(singular_values)] = 1.0 / (1.0 + singular_values**2)
    rotated_goal = whitened_goal @ right.T
    return float(np.sum(rotated_goal**2 * weights[np.newaxis, :]) / len(rotated_goal))


def _acquisition_receipts_v32(
    spec: ControlledDynamicsWorldPackSpecV32,
    public,
    observations: list[ControlledObservationReceiptV31],
    available: list[PiecewiseConstantInputActionV31],
    target: DecisionTargetV31,
    step_index: int,
) -> list[GoalPosteriorRiskAcquisitionReceiptV32]:
    terms, current_library, ensemble = _bootstrap_ensemble_v31(
        public,
        observations,
        spec,
        seed=int(sha256_value([spec.spec_hash, public.case_id, step_index])[:16], 16),
    )
    scales = np.sqrt(np.mean(current_library**2, axis=0))
    scales[scales < 1e-12] = 1.0
    normalized_current = current_library / scales
    precision = (
        spec.ridge_alpha * np.eye(normalized_current.shape[1])
        + normalized_current.T @ normalized_current
    )
    covariance_factor = _covariance_factor_v32(precision, spec.ridge_alpha)
    goal_matrices = _goal_feature_matrices_v32(
        public, target, terms, ensemble, spec
    )
    normalized_goals = [matrix / scales for matrix in goal_matrices]
    whitened_goals = [matrix @ covariance_factor for matrix in normalized_goals]
    current_risks = np.asarray([
        float(np.sum(matrix**2) / len(matrix))
        for matrix in whitened_goals
    ])
    precision_hash = sha256_value({
        "case_id": public.case_id,
        "step_index": step_index,
        "source_observation_hashes": [
            public.pilot.observation_hash,
            *[item.observation_hash for item in observations],
        ],
        "feature_scales": scales.tolist(),
        "precision": precision.tolist(),
    })
    goal_hash = sha256_value({
        "case_id": public.case_id,
        "step_index": step_index,
        "decision_target": target,
        "public_case_hash": public.public_hash,
        "initial_state_scales": spec.goal_initial_state_scales,
        "zero_component_range_fraction": spec.goal_zero_component_range_fraction,
        "envelope_margin_fraction": spec.goal_envelope_margin_fraction,
        "feature_matrices": [item.tolist() for item in goal_matrices],
    })
    lower = np.asarray(public.envelope.state_lower_bounds, dtype=float)
    upper = np.asarray(public.envelope.state_upper_bounds, dtype=float)
    state_scale = np.maximum(upper - lower, 1e-6)
    trim = spec.savgol_window // 2
    receipts: list[GoalPosteriorRiskAcquisitionReceiptV32] = []
    for action in available:
        failures = validate_action_against_envelope_v31(
            action, public.actuator, public.envelope
        )
        if failures:
            raise RuntimeError(f"V3.2 catalog action failed Harness validation: {failures}")
        paths = _ensemble_paths_v31(public, action, terms, ensemble, spec)
        mean_path = np.mean(paths, axis=0)
        outside = np.any((paths < lower) | (paths > upper), axis=(1, 2))
        empirical_risk = float(np.mean(outside))
        disagreement = float(np.mean(np.var(paths / state_scale, axis=0)))
        d_opt = _d_opt_gain_v31(current_library, mean_path, terms)
        posterior_risks: list[float] = []
        fractional_reductions: list[float] = []
        for member in range(ensemble.shape[0]):
            proposed = evaluate_polynomial_library(
                paths[member, trim:-trim], terms
            ) / scales
            post_risk = _posterior_risk_after_features_v32(
                whitened_goals[member], proposed @ covariance_factor
            )
            posterior_risks.append(post_risk)
            fractional_reductions.append(
                (current_risks[member] - post_risk)
                / max(current_risks[member], 1e-12)
            )
        fractions = np.asarray(fractional_reductions)
        robust_gain = float(np.quantile(
            fractions, spec.robust_goal_gain_quantile
        ))
        mean_current = float(np.mean(current_risks))
        mean_post = float(np.mean(posterior_risks))
        admissible = (
            empirical_risk
            <= public.envelope.maximum_empirical_prediction_risk
        )
        receipts.append(GoalPosteriorRiskAcquisitionReceiptV32.seal(
            acquisition_id=f"goal_acq_{public.case_id}_{step_index}_{action.action_id}",
            case_id=public.case_id,
            action_hash=action.action_hash,
            decision_target=target,
            belief_precision_hash=precision_hash,
            goal_operator_hash=goal_hash,
            goal_feature_row_count=sum(len(item) for item in goal_matrices),
            current_goal_posterior_risk=mean_current,
            predicted_goal_posterior_risk=mean_post,
            mean_fractional_goal_risk_reduction=float(np.mean(fractions)),
            robust_fractional_goal_risk_reduction=robust_gain,
            goal_risk_reduction_dispersion=float(np.std(fractions)),
            d_optimal_gain_diagnostic=d_opt,
            model_disagreement_diagnostic=disagreement,
            empirical_prediction_risk=empirical_risk,
            ranking_score=robust_gain,
            action_cost=action.action_cost,
            admissible=admissible,
            gate_codes=[
                "known_actuator", "peak_equal", "energy_equal", "switch_equal",
                "cost_equal",
                "empirical_risk_pass" if admissible else "empirical_risk_fail",
            ],
        ))
    return receipts


def _execute_case_v32(
    spec: ControlledDynamicsWorldPackSpecV32,
    private_case: PrivateControlledDynamicsCaseV31,
    policy: GoalPosteriorRiskPolicyV32,
    *,
    executed_at: datetime,
) -> ControlledDynamicsCaseReceiptV32:
    public = private_case.public_case
    public.assert_sealed()
    data_quality_passed = not public.pilot.quality_flags
    contract = public.initial_contract
    budget = spec.action_budget
    observations: list[ControlledObservationReceiptV31] = []
    selected_ids: list[str] = []
    steps: list[ControlledDynamicsStepReceiptV32] = []
    issue_routes: list[RouteLayerV31] = []
    abstention_count = 0

    for step_index in range(1, spec.action_budget + 1):
        if (
            policy.may_reformulate_problem
            and contract.target_status == "default_unverified"
        ):
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
            budget -= 1
            issue_routes.append("problem_layer")
            steps.append(ControlledDynamicsStepReceiptV32.seal(
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
                budget_before=budget,
                decided_at=executed_at,
            )
            if "data_layer" not in issue_routes:
                issue_routes.append("data_layer")
            abstention_count = 1
            steps.append(ControlledDynamicsStepReceiptV32.seal(
                step_index=step_index,
                action_kind="abstain",
                contract_before_hash=contract.contract_hash,
                contract_after_hash=contract.contract_hash,
                permission=permission,
            ))
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
                budget_before=budget,
                decided_at=executed_at,
            )
            abstention_count = 1
            steps.append(ControlledDynamicsStepReceiptV32.seal(
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
            budget_before=budget,
            decided_at=executed_at,
        )
        if permission.decision != "allow_synthetic":
            raise RuntimeError("V3.2 admissible action was not allowed")
        observation = private_case.action_observations[selected.action_id]
        observation.assert_sealed()
        if observation.empirical_peak_state_ratio > 1.0:
            raise RuntimeError("V3.2 hidden Reality Interface detected state-bound violation")
        observations.append(observation)
        selected_ids.append(selected.action_id)
        budget = permission.budget_after
        steps.append(ControlledDynamicsStepReceiptV32.seal(
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

    if contract.target_status == "default_unverified" and "problem_layer" not in issue_routes:
        issue_routes.append("problem_layer")
    model = None
    target_loss = None
    if data_quality_passed:
        model = _fit_model_v31(public, observations, spec)
        if model.normalized_derivative_residual > spec.model_mismatch_residual_threshold:
            issue_routes.append("model_layer")
        if private_case.performance_eligible:
            target_loss = _target_loss_v31(private_case, model, spec)
    return ControlledDynamicsCaseReceiptV32.seal(
        receipt_id=f"receipt_{policy.arm}_{public.case_id}_v32",
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
        action_budget_consumed=spec.action_budget - budget,
        abstention_count=abstention_count,
        target_loss=target_loss,
        executed_at=executed_at,
    )


def execute_controlled_dynamics_policy_v32(
    spec: ControlledDynamicsWorldPackSpecV32,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    policy: GoalPosteriorRiskPolicyV32,
    *,
    executed_at: datetime | None = None,
) -> ControlledDynamicsSelectionBundleV32:
    spec.assert_sealed()
    private_pack.assert_sealed()
    policy.assert_sealed()
    if private_pack.spec_hash != spec.spec_hash:
        raise ValueError("V3.2 private pack belongs to another protocol")
    expected = (
        spec.baseline_policy_hash
        if policy.arm == "random_bounded_inputs"
        else spec.candidate_policy_hash
    )
    if policy.policy_hash != expected:
        raise ValueError("V3.2 policy is not frozen in the protocol")
    at = executed_at or datetime.now(timezone.utc)
    receipts = [
        _execute_case_v32(spec, private_case, policy, executed_at=at)
        for private_case in private_pack.cases
    ]
    return ControlledDynamicsSelectionBundleV32.seal(
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


def evaluate_controlled_dynamics_worldpack_v32(
    spec: ControlledDynamicsWorldPackSpecV32,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    baseline: ControlledDynamicsSelectionBundleV32,
    candidate: ControlledDynamicsSelectionBundleV32,
    *,
    evaluated_at: datetime | None = None,
) -> GoalPosteriorRiskEvolutionReportV32:
    report = evaluate_controlled_dynamics_worldpack_v31(
        spec, private_pack, baseline, candidate, evaluated_at=evaluated_at
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
    ready = all(acquisition_gates.values())
    return GoalPosteriorRiskEvolutionReportV32.seal(
        evolution_id="controlled_dynamics_goal_risk_exploratory_v32",
        spec_hash=spec.spec_hash,
        base_adjudication_report=report,
        prior_v311_evolution_report_hash=spec.prior_v311_evolution_report_hash,
        acquisition_gates=acquisition_gates,
        acquisition_candidate_ready=ready,
        router_gate_passed=report.gates["routing_accuracy"],
        router_accuracy=report.routing_accuracy,
        status=(
            "acquisition_candidate_ready_for_router_evolution_v32"
            if ready else "acquisition_candidate_failed_v32"
        ),
        created_at=evaluated_at or datetime.now(timezone.utc),
    )


def run_controlled_dynamics_worldpack_v32(
    output_root: str | Path,
    *,
    spec: ControlledDynamicsWorldPackSpecV32,
    baseline_policy: GoalPosteriorRiskPolicyV32,
    candidate_policy: GoalPosteriorRiskPolicyV32,
    evaluated_at: datetime | None = None,
    run_id: str | None = None,
) -> ControlledDynamicsOutcomeV32:
    spec.assert_sealed()
    baseline_policy.assert_sealed()
    candidate_policy.assert_sealed()
    if baseline_policy.policy_hash != spec.baseline_policy_hash:
        raise ValueError("V3.2 baseline is not frozen in the protocol")
    if candidate_policy.policy_hash != spec.candidate_policy_hash:
        raise ValueError("V3.2 candidate is not frozen in the protocol")
    at = evaluated_at or datetime.now(timezone.utc)
    store = RunStore(
        output_root,
        run_id=run_id or f"controlled-dynamics-v32-{uuid4().hex[:10]}",
    )
    refs = [
        store.put_artifact("controlled_dynamics_spec_v32", spec),
        store.put_artifact("controlled_dynamics_baseline_policy_v32", baseline_policy),
        store.put_artifact("controlled_dynamics_candidate_policy_v32", candidate_policy),
    ]
    store.emit("controlled_dynamics_v32_protocol_frozen_before_private_pack", {
        "spec_hash": spec.spec_hash,
        "prior_v311_evolution_report_hash": spec.prior_v311_evolution_report_hash,
        "single_component_delta": spec.frozen_delta,
    })
    private_pack = generate_private_controlled_dynamics_worldpack_v31(
        spec, generated_at=at
    )
    baseline = execute_controlled_dynamics_policy_v32(
        spec, private_pack, baseline_policy, executed_at=at
    )
    candidate = execute_controlled_dynamics_policy_v32(
        spec, private_pack, candidate_policy, executed_at=at
    )
    evolution = evaluate_controlled_dynamics_worldpack_v32(
        spec, private_pack, baseline, candidate, evaluated_at=at
    )
    refs.extend([
        store.put_artifact("private_controlled_dynamics_worldpack_v32", private_pack),
        store.put_artifact("controlled_dynamics_baseline_bundle_v32", baseline),
        store.put_artifact("controlled_dynamics_candidate_bundle_v32", candidate),
        store.put_artifact(
            "controlled_dynamics_base_report_v32",
            evolution.base_adjudication_report,
        ),
        store.put_artifact("controlled_dynamics_evolution_report_v32", evolution),
    ])
    manifest = ControlledDynamicsManifestV32.seal(
        run_id=store.run_id,
        artifact_refs=refs,
        terminal_status=evolution.status,
        created_at=at,
    )
    manifest_ref = store.put_artifact("controlled_dynamics_manifest_v32", manifest)
    store.emit("controlled_dynamics_v32_worldpack_adjudicated", {
        "manifest_ref": manifest_ref.model_dump(mode="json")
    })
    if not verify_controlled_dynamics_run_v32(store.run_directory):
        raise RuntimeError("V3.2 controlled-dynamics run failed independent verification")
    return ControlledDynamicsOutcomeV32(
        store, spec, private_pack, baseline_policy, candidate_policy,
        baseline, candidate, evolution, manifest,
    )


def verify_controlled_dynamics_run_v32(run_directory: str | Path) -> bool:
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
            if item.kind == "controlled_dynamics_manifest_v32"
        ]
        if len(manifest_refs) != 1:
            return False
        manifest = ControlledDynamicsManifestV32.model_validate(
            store.load_artifact(manifest_refs[0])
        )
        manifest.assert_sealed()

        def load_one(kind: str, model):
            references = [item for item in manifest.artifact_refs if item.kind == kind]
            if len(references) != 1:
                raise RuntimeError(f"V3.2 manifest needs exactly one {kind}")
            return model.model_validate(store.load_artifact(references[0]))

        spec = load_one(
            "controlled_dynamics_spec_v32", ControlledDynamicsWorldPackSpecV32
        )
        baseline_policy = load_one(
            "controlled_dynamics_baseline_policy_v32", GoalPosteriorRiskPolicyV32
        )
        candidate_policy = load_one(
            "controlled_dynamics_candidate_policy_v32", GoalPosteriorRiskPolicyV32
        )
        private_pack = load_one(
            "private_controlled_dynamics_worldpack_v32",
            PrivateControlledDynamicsWorldPackV31,
        )
        baseline = load_one(
            "controlled_dynamics_baseline_bundle_v32",
            ControlledDynamicsSelectionBundleV32,
        )
        candidate = load_one(
            "controlled_dynamics_candidate_bundle_v32",
            ControlledDynamicsSelectionBundleV32,
        )
        base_report = load_one(
            "controlled_dynamics_base_report_v32", ControlledDynamicsReportV31
        )
        evolution = load_one(
            "controlled_dynamics_evolution_report_v32",
            GoalPosteriorRiskEvolutionReportV32,
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
        replay_baseline = execute_controlled_dynamics_policy_v32(
            spec, private_pack, baseline_policy, executed_at=executed_at
        )
        replay_candidate = execute_controlled_dynamics_policy_v32(
            spec, private_pack, candidate_policy, executed_at=executed_at
        )
        if (
            replay_baseline.bundle_hash != baseline.bundle_hash
            or replay_candidate.bundle_hash != candidate.bundle_hash
        ):
            return False
        recomputed = evaluate_controlled_dynamics_worldpack_v32(
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
            == "controlled_dynamics_v32_protocol_frozen_before_private_pack"
        ]
        return len(freeze_events) == 1 and store.verify_event_chain()
    except (
        OSError, RuntimeError, ValueError, KeyError, TypeError,
        json.JSONDecodeError, FloatingPointError, np.linalg.LinAlgError,
    ):
        return False
