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
from scipy.signal import savgol_filter
from scipy.stats import beta

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore
from fma.v2.dynamics_ir import (
    PolynomialBasisTermV24,
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
from .model_challenge_v37 import _hash_without
from .validator_input_contract_v39 import (
    ValidatorRecoveryEvolutionReportV391,
    verify_validator_recovery_run_v391,
)


EXPLORATORY_SEEDS_V310 = (
    25013, 25073, 25127, 25183, 25247, 25309,
    25367, 25423, 25481, 25537, 25589, 25643,
)

SkeletonV310 = Literal[
    "generic_polynomial_degree_1",
    "generic_polynomial_degree_2",
    "generic_polynomial_degree_3",
    "first_order_rate_law",
    "second_order_kinematic_force_law",
]
EstimatorV310 = Literal[
    "pointwise_savgol_stlsq",
    "integral_trapezoid_ridge",
]
ValidatorV310 = Literal[
    "trajectory_leave_one_experiment_out",
    "blocked_tail_forecast",
]
ArmV310 = Literal["nested_degree_baseline", "structured_factorial_candidate"]

SKELETONS_V310: tuple[SkeletonV310, ...] = (
    "generic_polynomial_degree_1",
    "generic_polynomial_degree_2",
    "generic_polynomial_degree_3",
    "first_order_rate_law",
    "second_order_kinematic_force_law",
)
ESTIMATORS_V310: tuple[EstimatorV310, ...] = (
    "pointwise_savgol_stlsq",
    "integral_trapezoid_ridge",
)
VALIDATORS_V310: tuple[ValidatorV310, ...] = (
    "trajectory_leave_one_experiment_out",
    "blocked_tail_forecast",
)
GENERIC_SKELETONS_V310 = SKELETONS_V310[:3]
STRUCTURED_SKELETONS_V310 = SKELETONS_V310[3:]


def _committed_refs_v310(store: RunStore) -> list[ArtifactRef]:
    events = [
        json.loads(line)
        for line in store.event_path.read_text(encoding="utf-8").splitlines()
    ]
    return [
        ArtifactRef.model_validate(event["payload"])
        for event in events if event["event_type"] == "artifact_committed"
    ]


def _load_one_v310(store: RunStore, refs: list[ArtifactRef], kind: str, model):
    matches = [item for item in refs if item.kind == kind]
    if len(matches) != 1:
        raise ValueError(f"V3.10 requires exactly one {kind}")
    return model.model_validate(store.load_artifact(matches[0]))


def _load_source_v391(run_directory: str | Path) -> ValidatorRecoveryEvolutionReportV391:
    if not verify_validator_recovery_run_v391(run_directory):
        raise ValueError("V3.10 source V3.9.1 run did not independently verify")
    store = RunStore.open_existing(run_directory)
    refs = _committed_refs_v310(store)
    return _load_one_v310(
        store,
        refs,
        "validator_recovery_evolution_report_v391",
        ValidatorRecoveryEvolutionReportV391,
    )


class MethodSourceV310(StrictModel):
    source_id: Identifier
    title: Annotated[str, Field(min_length=8)]
    source_url: Annotated[str, Field(pattern=r"^https://")]
    doi: str | None = None
    accessed_on: Literal["2026-07-22"] = "2026-07-22"
    borrowed_principle: Annotated[str, Field(min_length=20)]
    guarantee_transferred: Literal[False] = False


class SkeletonFactorialMethodEvidenceV310(StrictModel):
    schema_version: Literal["3.10"] = "3.10"
    evidence_id: Identifier
    retrieval_scope: Literal[
        "targeted_primary_sources_and_official_documentation"
    ] = "targeted_primary_sources_and_official_documentation"
    sources: list[MethodSourceV310] = Field(min_length=3, max_length=3)
    external_content_treated_as_untrusted_data: Literal[True] = True
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> "SkeletonFactorialMethodEvidenceV310":
        if len({item.source_url for item in self.sources}) != len(self.sources):
            raise ValueError("V3.10 method sources must be unique")
        if self.evidence_hash and self.evidence_hash != self.content_hash():
            raise ValueError("evidence_hash does not match V3.10 method evidence")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evidence_hash")

    def assert_sealed(self) -> None:
        if not self.evidence_hash or self.evidence_hash != self.content_hash():
            raise ValueError("V3.10 method evidence is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "SkeletonFactorialMethodEvidenceV310":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evidence_hash"}),
            evidence_hash=draft.content_hash(),
        )


def default_skeleton_factorial_method_evidence_v310() -> SkeletonFactorialMethodEvidenceV310:
    return SkeletonFactorialMethodEvidenceV310.seal(
        evidence_id="skeleton_factorial_method_evidence_v310",
        sources=[
            MethodSourceV310(
                source_id="brunton_proctor_kutz_2016",
                title="Discovering governing equations from data by sparse identification of nonlinear dynamical systems",
                source_url="https://doi.org/10.1073/pnas.1517384113",
                doi="10.1073/pnas.1517384113",
                borrowed_principle=(
                    "Compare parsimonious candidate dynamics using explicit libraries rather than one opaque universal fit."
                ),
            ),
            MethodSourceV310(
                source_id="messenger_bortz_2021",
                title="Weak SINDy Galerkin-Based Data-Driven Model Selection",
                source_url="https://doi.org/10.1137/20M1343166",
                doi="10.1137/20M1343166",
                borrowed_principle=(
                    "Use an integral weak form as an estimator arm that avoids direct numerical differentiation."
                ),
            ),
            MethodSourceV310(
                source_id="sklearn_time_series_split",
                title="TimeSeriesSplit official model selection documentation",
                source_url=(
                    "https://scikit-learn.org/stable/modules/generated/"
                    "sklearn.model_selection.TimeSeriesSplit.html"
                ),
                borrowed_principle=(
                    "Keep temporal order explicit and evaluate forecasting on future blocks rather than shuffled rows."
                ),
            ),
        ],
    )


class SkeletonFactorialPolicyV310(StrictModel):
    schema_version: Literal["3.10"] = "3.10"
    policy_id: Identifier
    method_evidence_hash: Sha256
    source_v391_evolution_hash: Sha256
    skeleton_catalog: list[SkeletonV310] = Field(min_length=5, max_length=5)
    estimator_catalog: list[EstimatorV310] = Field(min_length=2, max_length=2)
    validator_catalog: list[ValidatorV310] = Field(min_length=2, max_length=2)
    observation_action_indices: list[int] = Field(min_length=2, max_length=2)
    baseline_rule: Literal[
        "one_standard_error_over_nested_degree_savgol_loo"
    ] = "one_standard_error_over_nested_degree_savgol_loo"
    candidate_rule: Literal[
        "minimax_then_parsimony_with_public_loo_switch_guard"
    ] = "minimax_then_parsimony_with_public_loo_switch_guard"
    private_mechanism_visible: Literal[False] = False
    private_probe_visible: Literal[False] = False
    private_target_loss_visible: Literal[False] = False
    private_performance_eligible_used_for_partition: Literal[False] = False
    real_world_execution_permitted: Literal[False] = False
    task_router_permitted: Literal[False] = False
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "SkeletonFactorialPolicyV310":
        if self.skeleton_catalog != list(SKELETONS_V310):
            raise ValueError("V3.10 skeleton catalog differs")
        if self.estimator_catalog != list(ESTIMATORS_V310):
            raise ValueError("V3.10 estimator catalog differs")
        if self.validator_catalog != list(VALIDATORS_V310):
            raise ValueError("V3.10 validator catalog differs")
        if self.observation_action_indices != [0, 7]:
            raise ValueError("V3.10 observation actions differ")
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("policy_hash does not match V3.10 policy")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("V3.10 policy is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "SkeletonFactorialPolicyV310":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"policy_hash"}),
            policy_hash=draft.content_hash(),
        )


def default_skeleton_factorial_policy_v310(
    evidence: SkeletonFactorialMethodEvidenceV310,
    source_v391_evolution_hash: str,
) -> SkeletonFactorialPolicyV310:
    evidence.assert_sealed()
    return SkeletonFactorialPolicyV310.seal(
        policy_id="public_structured_skeleton_factorial_v310",
        method_evidence_hash=evidence.evidence_hash,
        source_v391_evolution_hash=source_v391_evolution_hash,
        skeleton_catalog=list(SKELETONS_V310),
        estimator_catalog=list(ESTIMATORS_V310),
        validator_catalog=list(VALIDATORS_V310),
        observation_action_indices=[0, 7],
    )


class SkeletonFactorialWorldPackSpecV310(StrictModel):
    schema_version: Literal["3.10"] = "3.10"
    experiment_id: Identifier
    phase: Literal["exploratory"] = "exploratory"
    mechanisms: list[MechanismV31] = Field(min_length=4, max_length=4)
    seeds: list[int] = Field(min_length=12, max_length=12)
    trajectory_points: Literal[49] = 49
    time_step: Literal[0.04] = 0.04
    segment_count: Literal[6] = 6
    segment_duration: Literal[0.32] = 0.32
    input_amplitude: Literal[0.35] = 0.35
    observation_noise_fraction: Literal[0.01] = 0.01
    maximum_empirical_prediction_risk: Literal[0.25] = 0.25
    savgol_window: Literal[9] = 9
    savgol_order: Literal[3] = 3
    ridge_alpha: Literal[0.0001] = 0.0001
    sparsity_threshold: Literal[0.02] = 0.02
    integral_window_intervals: Literal[4] = 4
    blocked_tail_start_index: Literal[30] = 30
    observation_action_indices: list[int] = Field(min_length=2, max_length=2)
    maximum_cv_prediction_loss: Literal[0.5] = 0.5
    minimum_rank_ratio: Literal[0.9] = 0.9
    maximum_condition_number: Literal[100000000.0] = 100000000.0
    expected_quality_abstention_count: Literal[9] = 9
    unresolved_adjudicated_loss: Literal[10.0] = 10.0
    material_negative_transfer: Literal[0.02] = 0.02
    maximum_mechanism_regression: Literal[0.02] = 0.02
    required_duffing_improvement: Literal[0.2] = 0.2
    maximum_duffing_candidate_loss: Literal[0.5] = 0.5
    bootstrap_replicates: Literal[2000] = 2000
    bootstrap_seed: Literal[3100722] = 3100722
    method_evidence_hash: Sha256
    policy_hash: Sha256
    source_v391_evolution_hash: Sha256
    frozen_delta: Literal[
        "state_topology_skeleton_by_estimator_by_temporal_validation_factorial_only"
    ] = "state_topology_skeleton_by_estimator_by_temporal_validation_factorial_only"
    frozen_at: datetime
    spec_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "SkeletonFactorialWorldPackSpecV310":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.mechanisms != list(MECHANISMS_V31):
            raise ValueError("V3.10 mechanism order differs")
        if self.seeds != list(EXPLORATORY_SEEDS_V310):
            raise ValueError("V3.10 seeds do not match frozen set")
        if self.observation_action_indices != [0, 7]:
            raise ValueError("V3.10 observation actions differ")
        if not math.isclose(
            (self.trajectory_points - 1) * self.time_step,
            self.segment_count * self.segment_duration,
            abs_tol=1e-12,
        ):
            raise ValueError("V3.10 segments do not cover trajectory")
        if self.blocked_tail_start_index <= self.savgol_window:
            raise ValueError("V3.10 blocked tail leaves too little training data")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match V3.10 protocol")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "spec_hash")

    def assert_sealed(self) -> None:
        if not self.spec_hash or self.spec_hash != self.content_hash():
            raise ValueError("V3.10 protocol is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "SkeletonFactorialWorldPackSpecV310":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"spec_hash"}),
            spec_hash=draft.content_hash(),
        )


def default_skeleton_factorial_spec_v310(
    evidence: SkeletonFactorialMethodEvidenceV310,
    policy: SkeletonFactorialPolicyV310,
    *,
    frozen_at: datetime | None = None,
) -> SkeletonFactorialWorldPackSpecV310:
    evidence.assert_sealed()
    policy.assert_sealed()
    return SkeletonFactorialWorldPackSpecV310.seal(
        experiment_id="skeleton_factorial_exploratory_v310",
        mechanisms=list(MECHANISMS_V31),
        seeds=list(EXPLORATORY_SEEDS_V310),
        observation_action_indices=[0, 7],
        method_evidence_hash=evidence.evidence_hash,
        policy_hash=policy.policy_hash,
        source_v391_evolution_hash=policy.source_v391_evolution_hash,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


@dataclass(frozen=True)
class _ObservationView:
    observation_id: str
    observation_hash: str
    action_hash: str | None
    times: list[float]
    states: list[list[float]]
    inputs: list[list[float]]


@dataclass(frozen=True)
class _FitOutcome:
    model: ControlledDriftModelV31
    rank_ratio: float
    condition_number: float
    normalized_residual: float
    active_coefficient_count: int


def _observations_v310(
    private_case: PrivateControlledDynamicsCaseV31,
    spec: SkeletonFactorialWorldPackSpecV310,
) -> list[object]:
    public = private_case.public_case
    return [
        public.pilot,
        *[
            private_case.action_observations[public.action_catalog[index].action_id]
            for index in spec.observation_action_indices
        ],
    ]


def _compatible_v310(state_names: list[str], skeleton: SkeletonV310) -> bool:
    if skeleton == "first_order_rate_law":
        return len(state_names) == 1
    if skeleton == "second_order_kinematic_force_law":
        return len(state_names) == 2 and state_names == ["position", "velocity"]
    return True


def _terms_v310(
    state_names: list[str], skeleton: SkeletonV310
) -> list[PolynomialBasisTermV24]:
    if skeleton.startswith("generic_polynomial_degree_"):
        return polynomial_basis_terms(state_names, int(skeleton.rsplit("_", 1)[1]))
    if skeleton == "first_order_rate_law":
        return [
            PolynomialBasisTermV24(term_id=state_names[0], exponents=[1]),
            PolynomialBasisTermV24(term_id=f"{state_names[0]}2", exponents=[2]),
        ]
    return [
        PolynomialBasisTermV24(term_id="position", exponents=[1, 0]),
        PolynomialBasisTermV24(term_id="velocity", exponents=[0, 1]),
        PolynomialBasisTermV24(term_id="position3", exponents=[3, 0]),
    ]


def _ridge_v310(library: np.ndarray, targets: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, float]:
    scales = np.sqrt(np.mean(library**2, axis=0))
    scales[scales < 1e-12] = 1.0
    normalized = library / scales
    solved = np.linalg.solve(
        normalized.T @ normalized + alpha * np.eye(normalized.shape[1]),
        normalized.T @ targets,
    ).T
    coefficients = solved / scales[np.newaxis, :]
    condition = float(np.linalg.cond(normalized))
    if not math.isfinite(condition):
        condition = 1e15
    return coefficients, normalized, condition


def _stlsq_v310(
    library: np.ndarray,
    targets: np.ndarray,
    alpha: float,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    coefficients, normalized, condition = _ridge_v310(library, targets, alpha)
    scales = np.sqrt(np.mean(library**2, axis=0))
    scales[scales < 1e-12] = 1.0
    for equation in range(targets.shape[1]):
        active = np.abs(coefficients[equation]) >= threshold
        for _ in range(12):
            previous = active.copy()
            coefficients[equation, ~active] = 0.0
            if active.any():
                selected = normalized[:, active]
                solved = np.linalg.solve(
                    selected.T @ selected + alpha * np.eye(selected.shape[1]),
                    selected.T @ targets[:, equation],
                )
                coefficients[equation, active] = solved / scales[active]
            active = np.abs(coefficients[equation]) >= threshold
            if np.array_equal(active, previous):
                break
        coefficients[equation, ~active] = 0.0
    return coefficients, normalized, condition


def _expanded_arrays_v310(
    observations: list[object],
    terms: list[PolynomialBasisTermV24],
    estimator: EstimatorV310,
    actuator: np.ndarray,
    spec: SkeletonFactorialWorldPackSpecV310,
) -> tuple[np.ndarray, np.ndarray]:
    libraries: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    if estimator == "pointwise_savgol_stlsq":
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
            libraries.append(evaluate_polynomial_library(states[trim:-trim], terms))
            targets.append((derivatives - inputs @ actuator.T)[trim:-trim])
        return np.vstack(libraries), np.vstack(targets)

    window = spec.integral_window_intervals
    for observation in observations:
        states = np.asarray(observation.states, dtype=float)
        inputs = np.asarray(observation.inputs, dtype=float)
        times = np.asarray(observation.times, dtype=float)
        library = evaluate_polynomial_library(states, terms)
        control = inputs @ actuator.T
        for start in range(0, len(states) - window):
            end = start + window
            libraries.append(
                np.trapezoid(
                    library[start:end + 1], times[start:end + 1], axis=0
                )[None, :]
            )
            delta = states[end] - states[start]
            integrated_control = np.trapezoid(
                control[start:end + 1], times[start:end + 1], axis=0
            )
            targets.append((delta - integrated_control)[None, :])
    return np.vstack(libraries), np.vstack(targets)


def _fit_v310(
    private_case: PrivateControlledDynamicsCaseV31,
    observations: list[object],
    skeleton: SkeletonV310,
    estimator: EstimatorV310,
    spec: SkeletonFactorialWorldPackSpecV310,
    *,
    suffix: str,
) -> _FitOutcome:
    state_names = private_case.public_case.state_names
    if not _compatible_v310(state_names, skeleton):
        raise ValueError("V3.10 attempted incompatible skeleton fit")
    terms = _terms_v310(state_names, skeleton)
    actuator = np.asarray(private_case.public_case.actuator.matrix, dtype=float)
    library, full_targets = _expanded_arrays_v310(
        observations, terms, estimator, actuator, spec
    )
    if skeleton == "second_order_kinematic_force_law":
        targets = full_targets[:, [1]]
    else:
        targets = full_targets
    if estimator == "pointwise_savgol_stlsq" and skeleton.startswith("generic_"):
        fitted_rows, normalized, condition = _stlsq_v310(
            library, targets, spec.ridge_alpha, spec.sparsity_threshold
        )
    else:
        fitted_rows, normalized, condition = _ridge_v310(
            library, targets, spec.ridge_alpha
        )
    if skeleton == "second_order_kinematic_force_law":
        coefficients = np.zeros((2, len(terms)), dtype=float)
        coefficients[0, 1] = 1.0
        coefficients[1] = fitted_rows[0]
        fitted = library @ fitted_rows.T
        comparison_targets = targets
    else:
        coefficients = fitted_rows
        fitted = library @ coefficients.T
        comparison_targets = targets
    residual = float(
        np.sqrt(np.mean((fitted - comparison_targets) ** 2))
        / max(float(np.sqrt(np.mean(comparison_targets**2))), 0.1)
    )
    rank = int(np.linalg.matrix_rank(normalized, tol=1e-10))
    rank_ratio = rank / max(normalized.shape[1], 1)
    hashes = [str(item.observation_hash) for item in observations]
    model = ControlledDriftModelV31.seal(
        model_id=f"model_{private_case.public_case.case_id}_{skeleton}_{estimator}_{suffix}",
        case_id=private_case.public_case.case_id,
        state_names=state_names,
        actuator_hash=private_case.public_case.actuator.actuator_hash,
        basis_terms=terms,
        coefficient_matrix=coefficients.tolist(),
        source_observation_hashes=hashes,
        normalized_design_rank=rank,
        normalized_condition_number=condition,
        normalized_derivative_residual=residual,
    )
    active = int(np.sum(np.abs(coefficients) > 1e-12))
    return _FitOutcome(model, rank_ratio, condition, residual, active)


def _segments_v310(
    private_case: PrivateControlledDynamicsCaseV31,
    observation: object,
    spec: SkeletonFactorialWorldPackSpecV310,
) -> tuple[list[list[float]], Literal["pilot_zero", "public_action_hash"]]:
    action_hash = getattr(observation, "action_hash", None)
    if action_hash is None:
        return [[0.0] for _ in range(spec.segment_count)], "pilot_zero"
    matches = [
        action for action in private_case.public_case.action_catalog
        if action.action_hash == action_hash
    ]
    if len(matches) != 1:
        raise ValueError("V3.10 action hash did not bind one public segment sequence")
    if len(matches[0].input_values) != spec.segment_count:
        raise ValueError("V3.10 public action has wrong segment count")
    return matches[0].input_values, "public_action_hash"


def _prefix_view_v310(observation: object, end: int) -> _ObservationView:
    content = {
        "source_observation_hash": observation.observation_hash,
        "exclusive_end": end,
        "times": observation.times[:end],
        "states": observation.states[:end],
        "inputs": observation.inputs[:end],
    }
    return _ObservationView(
        observation_id=f"prefix_{observation.observation_id}_{end}",
        observation_hash=sha256_value(content),
        action_hash=getattr(observation, "action_hash", None),
        times=observation.times[:end],
        states=observation.states[:end],
        inputs=observation.inputs[:end],
    )


class InputBindingReceiptV310(StrictModel):
    schema_version: Literal["3.10"] = "3.10"
    case_id: Identifier
    observation_hash: Sha256
    action_hash: Sha256 | None
    binding_source: Literal["pilot_zero", "public_action_hash"]
    segment_count: Literal[6] = 6
    segment_hash: Sha256
    contract_valid: Literal[True] = True
    binding_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_binding(self) -> "InputBindingReceiptV310":
        expected = "pilot_zero" if self.action_hash is None else "public_action_hash"
        if self.binding_source != expected:
            raise ValueError("V3.10 input binding source differs")
        if self.binding_hash and self.binding_hash != self.content_hash():
            raise ValueError("binding_hash does not match V3.10 input binding")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "binding_hash")

    def assert_sealed(self) -> None:
        if not self.binding_hash or self.binding_hash != self.content_hash():
            raise ValueError("V3.10 input binding is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "InputBindingReceiptV310":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"binding_hash"}),
            binding_hash=draft.content_hash(),
        )


class FactorCellReceiptV310(StrictModel):
    schema_version: Literal["3.10"] = "3.10"
    cell_id: Identifier
    case_id: Identifier
    skeleton: SkeletonV310
    estimator: EstimatorV310
    validator: ValidatorV310
    fold_losses: list[Annotated[float, Field(ge=0, allow_inf_nan=False)]] = Field(min_length=3, max_length=3)
    fold_model_hashes: list[Sha256] = Field(min_length=3, max_length=3)
    input_bindings: list[InputBindingReceiptV310] = Field(min_length=3, max_length=3)
    mean_validation_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    standard_error: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    simulation_failure_count: Annotated[int, Field(ge=0, le=3)]
    final_model_hash: Sha256
    normalized_rank_ratio: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    normalized_condition_number: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    normalized_fit_residual: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    active_coefficient_count: Annotated[int, Field(ge=1)]
    eligible: bool
    private_values_used: Literal[False] = False
    cell_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_cell(self) -> "FactorCellReceiptV310":
        for binding in self.input_bindings:
            binding.assert_sealed()
        values = np.asarray(self.fold_losses, dtype=float)
        if not math.isclose(self.mean_validation_loss, float(np.mean(values)), abs_tol=1e-12):
            raise ValueError("V3.10 cell mean does not recompute")
        expected_se = float(np.std(values, ddof=1) / math.sqrt(len(values)))
        if not math.isclose(self.standard_error, expected_se, abs_tol=1e-12):
            raise ValueError("V3.10 cell SE does not recompute")
        if self.cell_hash and self.cell_hash != self.content_hash():
            raise ValueError("cell_hash does not match V3.10 cell")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "cell_hash")

    def assert_sealed(self) -> None:
        if not self.cell_hash or self.cell_hash != self.content_hash():
            raise ValueError("V3.10 cell is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "FactorCellReceiptV310":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"cell_hash"}),
            cell_hash=draft.content_hash(),
        )


class FactorPairReceiptV310(StrictModel):
    schema_version: Literal["3.10"] = "3.10"
    pair_id: Identifier
    case_id: Identifier
    skeleton: SkeletonV310
    estimator: EstimatorV310
    cell_hashes: list[Sha256] = Field(min_length=2, max_length=2)
    robust_validation_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    both_validators_eligible: bool
    final_model: ControlledDriftModelV31
    pair_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_pair(self) -> "FactorPairReceiptV310":
        self.final_model.assert_sealed()
        if self.final_model.case_id != self.case_id:
            raise ValueError("V3.10 pair model case differs")
        if self.pair_hash and self.pair_hash != self.content_hash():
            raise ValueError("pair_hash does not match V3.10 pair")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "pair_hash")

    def assert_sealed(self) -> None:
        if not self.pair_hash or self.pair_hash != self.content_hash():
            raise ValueError("V3.10 pair is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "FactorPairReceiptV310":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"pair_hash"}),
            pair_hash=draft.content_hash(),
        )


class FactorialDecisionV310(StrictModel):
    schema_version: Literal["3.10"] = "3.10"
    decision_id: Identifier
    case_id: Identifier
    arm: ArmV310
    candidate_pair_hashes: list[Sha256]
    decision: Literal["select", "abstain"]
    reason: Literal["public_quality_failure", "no_eligible_pair", "public_factorial_selection"]
    selected_pair_hash: Sha256 | None
    selected_skeleton: SkeletonV310 | None
    selected_estimator: EstimatorV310 | None
    selected_robust_validation_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
    private_values_used: Literal[False] = False
    decision_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> "FactorialDecisionV310":
        selected = self.decision == "select"
        fields = (
            self.selected_pair_hash,
            self.selected_skeleton,
            self.selected_estimator,
            self.selected_robust_validation_loss,
        )
        if selected != all(item is not None for item in fields):
            raise ValueError("V3.10 decision selection fields differ")
        if self.decision_hash and self.decision_hash != self.content_hash():
            raise ValueError("decision_hash does not match V3.10 decision")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "decision_hash")

    def assert_sealed(self) -> None:
        if not self.decision_hash or self.decision_hash != self.content_hash():
            raise ValueError("V3.10 decision is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "FactorialDecisionV310":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"decision_hash"}),
            decision_hash=draft.content_hash(),
        )


class SkeletonFactorialCaseReceiptV310(StrictModel):
    schema_version: Literal["3.10"] = "3.10"
    receipt_id: Identifier
    case_id: Identifier
    public_case_hash: Sha256
    policy_hash: Sha256
    public_quality_flags: list[Identifier]
    cells: list[FactorCellReceiptV310]
    pairs: list[FactorPairReceiptV310]
    baseline_decision: FactorialDecisionV310
    candidate_decision: FactorialDecisionV310
    executed_at: datetime
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "SkeletonFactorialCaseReceiptV310":
        _assert_timezone(self.executed_at, "executed_at")
        self.baseline_decision.assert_sealed()
        self.candidate_decision.assert_sealed()
        for cell in self.cells:
            cell.assert_sealed()
        for pair in self.pairs:
            pair.assert_sealed()
        pair_hashes = {item.pair_hash for item in self.pairs}
        for decision in (self.baseline_decision, self.candidate_decision):
            if not set(decision.candidate_pair_hashes).issubset(pair_hashes):
                raise ValueError("V3.10 decision references unknown pair")
            if decision.selected_pair_hash and decision.selected_pair_hash not in pair_hashes:
                raise ValueError("V3.10 selected pair is missing")
        if self.public_quality_flags:
            if self.cells or self.pairs:
                raise ValueError("V3.10 quality abstention cannot run factorial")
            if self.baseline_decision.reason != "public_quality_failure" or self.candidate_decision.reason != "public_quality_failure":
                raise ValueError("V3.10 quality case was not publicly abstained")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match V3.10 case receipt")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("V3.10 case receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "SkeletonFactorialCaseReceiptV310":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


class SkeletonFactorialBundleV310(StrictModel):
    schema_version: Literal["3.10"] = "3.10"
    bundle_id: Identifier
    spec_hash: Sha256
    private_pack_hash: Sha256
    policy_hash: Sha256
    case_receipts: list[SkeletonFactorialCaseReceiptV310] = Field(min_length=48, max_length=48)
    created_at: datetime
    bundle_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bundle(self) -> "SkeletonFactorialBundleV310":
        _assert_timezone(self.created_at, "created_at")
        ids = [item.case_id for item in self.case_receipts]
        if len(ids) != len(set(ids)):
            raise ValueError("V3.10 bundle case ids differ")
        for receipt in self.case_receipts:
            receipt.assert_sealed()
            if receipt.policy_hash != self.policy_hash:
                raise ValueError("V3.10 case policy differs")
        if self.bundle_hash and self.bundle_hash != self.content_hash():
            raise ValueError("bundle_hash does not match V3.10 bundle")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "bundle_hash")

    def assert_sealed(self) -> None:
        if not self.bundle_hash or self.bundle_hash != self.content_hash():
            raise ValueError("V3.10 bundle is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "SkeletonFactorialBundleV310":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"bundle_hash"}),
            bundle_hash=draft.content_hash(),
        )


def _binding_receipt_v310(
    private_case: PrivateControlledDynamicsCaseV31,
    observation: object,
    segments: list[list[float]],
    source: Literal["pilot_zero", "public_action_hash"],
) -> InputBindingReceiptV310:
    return InputBindingReceiptV310.seal(
        case_id=private_case.public_case.case_id,
        observation_hash=observation.observation_hash,
        action_hash=getattr(observation, "action_hash", None),
        binding_source=source,
        segment_hash=sha256_value(segments),
    )


def _cell_v310(
    private_case: PrivateControlledDynamicsCaseV31,
    observations: list[object],
    skeleton: SkeletonV310,
    estimator: EstimatorV310,
    validator: ValidatorV310,
    final_fit: _FitOutcome,
    spec: SkeletonFactorialWorldPackSpecV310,
) -> FactorCellReceiptV310:
    losses: list[float] = []
    model_hashes: list[str] = []
    bindings: list[InputBindingReceiptV310] = []
    failures = 0
    if validator == "trajectory_leave_one_experiment_out":
        for holdout_index, holdout in enumerate(observations):
            training = [
                item for index, item in enumerate(observations)
                if index != holdout_index
            ]
            fit = _fit_v310(
                private_case,
                training,
                skeleton,
                estimator,
                spec,
                suffix=f"loo{holdout_index}",
            )
            segments, source = _segments_v310(private_case, holdout, spec)
            bindings.append(_binding_receipt_v310(private_case, holdout, segments, source))
            model_hashes.append(fit.model.model_hash)
            try:
                predicted = _simulate_model_v31(
                    fit.model,
                    private_case.public_case.actuator,
                    holdout.states[0],
                    holdout.times,
                    segments,
                    spec.segment_duration,
                )
                losses.append(trajectory_nrmse(holdout.states, predicted))
            except RuntimeError:
                losses.append(spec.unresolved_adjudicated_loss)
                failures += 1
    else:
        prefix_end = spec.blocked_tail_start_index + 1
        prefixes = [_prefix_view_v310(item, prefix_end) for item in observations]
        fit = _fit_v310(
            private_case,
            prefixes,
            skeleton,
            estimator,
            spec,
            suffix="blocked_prefixes",
        )
        for holdout in observations:
            segments, source = _segments_v310(private_case, holdout, spec)
            bindings.append(_binding_receipt_v310(private_case, holdout, segments, source))
            model_hashes.append(fit.model.model_hash)
            tail_times = holdout.times[spec.blocked_tail_start_index:]
            tail_states = holdout.states[spec.blocked_tail_start_index:]
            try:
                predicted = _simulate_model_v31(
                    fit.model,
                    private_case.public_case.actuator,
                    tail_states[0],
                    tail_times,
                    segments,
                    spec.segment_duration,
                )
                losses.append(trajectory_nrmse(tail_states, predicted))
            except RuntimeError:
                losses.append(spec.unresolved_adjudicated_loss)
                failures += 1
    values = np.asarray(losses, dtype=float)
    mean = float(np.mean(values))
    eligible = (
        failures == 0
        and mean <= spec.maximum_cv_prediction_loss
        and final_fit.rank_ratio >= spec.minimum_rank_ratio
        and final_fit.condition_number <= spec.maximum_condition_number
    )
    return FactorCellReceiptV310.seal(
        cell_id=(
            f"cell_{private_case.public_case.case_id}_{skeleton}_{estimator}_{validator}"
        ),
        case_id=private_case.public_case.case_id,
        skeleton=skeleton,
        estimator=estimator,
        validator=validator,
        fold_losses=losses,
        fold_model_hashes=model_hashes,
        input_bindings=bindings,
        mean_validation_loss=mean,
        standard_error=float(np.std(values, ddof=1) / math.sqrt(len(values))),
        simulation_failure_count=failures,
        final_model_hash=final_fit.model.model_hash,
        normalized_rank_ratio=final_fit.rank_ratio,
        normalized_condition_number=final_fit.condition_number,
        normalized_fit_residual=final_fit.normalized_residual,
        active_coefficient_count=final_fit.active_coefficient_count,
        eligible=eligible,
    )


def _abstain_decision_v310(
    case_id: str,
    arm: ArmV310,
    reason: Literal["public_quality_failure", "no_eligible_pair"],
    pair_hashes: list[str],
) -> FactorialDecisionV310:
    return FactorialDecisionV310.seal(
        decision_id=f"decision_{case_id}_{arm}",
        case_id=case_id,
        arm=arm,
        candidate_pair_hashes=pair_hashes,
        decision="abstain",
        reason=reason,
        selected_pair_hash=None,
        selected_skeleton=None,
        selected_estimator=None,
        selected_robust_validation_loss=None,
    )


def _select_decisions_v310(
    case_id: str,
    cells: list[FactorCellReceiptV310],
    pairs: list[FactorPairReceiptV310],
) -> tuple[FactorialDecisionV310, FactorialDecisionV310]:
    cell_by_key = {
        (item.skeleton, item.estimator, item.validator): item for item in cells
    }
    pair_by_key = {(item.skeleton, item.estimator): item for item in pairs}

    baseline_options: list[tuple[FactorPairReceiptV310, FactorCellReceiptV310]] = []
    for skeleton in GENERIC_SKELETONS_V310:
        cell = cell_by_key[(
            skeleton,
            "pointwise_savgol_stlsq",
            "trajectory_leave_one_experiment_out",
        )]
        if cell.eligible:
            baseline_options.append((
                pair_by_key[(skeleton, "pointwise_savgol_stlsq")], cell
            ))
    if not baseline_options:
        baseline = _abstain_decision_v310(
            case_id, "nested_degree_baseline", "no_eligible_pair", []
        )
    else:
        best_pair, best_cell = min(
            baseline_options, key=lambda item: item[1].mean_validation_loss
        )
        threshold = best_cell.mean_validation_loss + best_cell.standard_error
        eligible_within = [
            item for item in baseline_options
            if item[1].mean_validation_loss <= threshold
        ]
        selected_pair, selected_cell = min(
            eligible_within,
            key=lambda item: GENERIC_SKELETONS_V310.index(item[0].skeleton),
        )
        baseline = FactorialDecisionV310.seal(
            decision_id=f"decision_{case_id}_nested_degree_baseline",
            case_id=case_id,
            arm="nested_degree_baseline",
            candidate_pair_hashes=[item[0].pair_hash for item in baseline_options],
            decision="select",
            reason="public_factorial_selection",
            selected_pair_hash=selected_pair.pair_hash,
            selected_skeleton=selected_pair.skeleton,
            selected_estimator=selected_pair.estimator,
            selected_robust_validation_loss=selected_cell.mean_validation_loss,
        )

    candidate_options = [item for item in pairs if item.both_validators_eligible]
    if not candidate_options:
        candidate = _abstain_decision_v310(
            case_id, "structured_factorial_candidate", "no_eligible_pair", []
        )
    else:
        best = min(candidate_options, key=lambda item: item.robust_validation_loss)
        best_cells = [
            cell_by_key[(best.skeleton, best.estimator, validator)]
            for validator in VALIDATORS_V310
        ]
        uncertainty = max(item.standard_error for item in best_cells)
        near_best = [
            item for item in candidate_options
            if item.robust_validation_loss <= best.robust_validation_loss + uncertainty
        ]

        def parsimony(item: FactorPairReceiptV310) -> tuple[int, int, float]:
            if item.skeleton in STRUCTURED_SKELETONS_V310:
                structural_rank = 0
            else:
                structural_rank = 1 + GENERIC_SKELETONS_V310.index(item.skeleton)
            estimator_rank = 0 if item.estimator == "integral_trapezoid_ridge" else 1
            return structural_rank, estimator_rank, item.robust_validation_loss

        selected = min(near_best, key=parsimony)
        # Parsimony may break a statistically supported tie, but it may not create
        # evidence for a model switch.  Retain the nested baseline unless the
        # candidate's public LOO upper one-SE bound beats the baseline LOO mean.
        if baseline.selected_pair_hash is not None:
            baseline_pair = next(
                item for item in pairs if item.pair_hash == baseline.selected_pair_hash
            )
            baseline_loo = cell_by_key[(
                baseline_pair.skeleton,
                baseline_pair.estimator,
                "trajectory_leave_one_experiment_out",
            )]
            selected_loo = cell_by_key[(
                selected.skeleton,
                selected.estimator,
                "trajectory_leave_one_experiment_out",
            )]
            if (
                selected_loo.mean_validation_loss + selected_loo.standard_error
                >= baseline_loo.mean_validation_loss
                and baseline_pair.both_validators_eligible
            ):
                selected = baseline_pair
        candidate = FactorialDecisionV310.seal(
            decision_id=f"decision_{case_id}_structured_factorial_candidate",
            case_id=case_id,
            arm="structured_factorial_candidate",
            candidate_pair_hashes=[item.pair_hash for item in candidate_options],
            decision="select",
            reason="public_factorial_selection",
            selected_pair_hash=selected.pair_hash,
            selected_skeleton=selected.skeleton,
            selected_estimator=selected.estimator,
            selected_robust_validation_loss=selected.robust_validation_loss,
        )
    return baseline, candidate


def _execute_case_v310(
    private_case: PrivateControlledDynamicsCaseV31,
    policy: SkeletonFactorialPolicyV310,
    spec: SkeletonFactorialWorldPackSpecV310,
    executed_at: datetime,
) -> SkeletonFactorialCaseReceiptV310:
    public = private_case.public_case
    quality_flags = list(public.pilot.quality_flags)
    if quality_flags:
        baseline = _abstain_decision_v310(
            public.case_id, "nested_degree_baseline", "public_quality_failure", []
        )
        candidate = _abstain_decision_v310(
            public.case_id,
            "structured_factorial_candidate",
            "public_quality_failure",
            [],
        )
        return SkeletonFactorialCaseReceiptV310.seal(
            receipt_id=f"factorial_{public.case_id}",
            case_id=public.case_id,
            public_case_hash=public.public_hash,
            policy_hash=policy.policy_hash,
            public_quality_flags=quality_flags,
            cells=[],
            pairs=[],
            baseline_decision=baseline,
            candidate_decision=candidate,
            executed_at=executed_at,
        )

    observations = _observations_v310(private_case, spec)
    cells: list[FactorCellReceiptV310] = []
    pairs: list[FactorPairReceiptV310] = []
    for skeleton in SKELETONS_V310:
        if not _compatible_v310(public.state_names, skeleton):
            continue
        for estimator in ESTIMATORS_V310:
            final_fit = _fit_v310(
                private_case,
                observations,
                skeleton,
                estimator,
                spec,
                suffix="all_public_observations",
            )
            pair_cells = [
                _cell_v310(
                    private_case,
                    observations,
                    skeleton,
                    estimator,
                    validator,
                    final_fit,
                    spec,
                )
                for validator in VALIDATORS_V310
            ]
            cells.extend(pair_cells)
            pairs.append(FactorPairReceiptV310.seal(
                pair_id=f"pair_{public.case_id}_{skeleton}_{estimator}",
                case_id=public.case_id,
                skeleton=skeleton,
                estimator=estimator,
                cell_hashes=[item.cell_hash for item in pair_cells],
                robust_validation_loss=max(
                    item.mean_validation_loss for item in pair_cells
                ),
                both_validators_eligible=all(item.eligible for item in pair_cells),
                final_model=final_fit.model,
            ))
    baseline, candidate = _select_decisions_v310(public.case_id, cells, pairs)
    return SkeletonFactorialCaseReceiptV310.seal(
        receipt_id=f"factorial_{public.case_id}",
        case_id=public.case_id,
        public_case_hash=public.public_hash,
        policy_hash=policy.policy_hash,
        public_quality_flags=quality_flags,
        cells=cells,
        pairs=pairs,
        baseline_decision=baseline,
        candidate_decision=candidate,
        executed_at=executed_at,
    )


def execute_skeleton_factorial_v310(
    spec: SkeletonFactorialWorldPackSpecV310,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    policy: SkeletonFactorialPolicyV310,
    *,
    executed_at: datetime,
) -> SkeletonFactorialBundleV310:
    for artifact in (spec, private_pack, policy):
        artifact.assert_sealed()
    if private_pack.spec_hash != spec.spec_hash or policy.policy_hash != spec.policy_hash:
        raise ValueError("V3.10 execution artifact binding differs")
    return SkeletonFactorialBundleV310.seal(
        bundle_id=f"bundle_{spec.experiment_id}",
        spec_hash=spec.spec_hash,
        private_pack_hash=private_pack.pack_hash,
        policy_hash=policy.policy_hash,
        case_receipts=[
            _execute_case_v310(case, policy, spec, executed_at)
            for case in private_pack.cases
        ],
        created_at=executed_at,
    )


class PrivateFactorialCaseResultV310(StrictModel):
    schema_version: Literal["3.10"] = "3.10"
    case_id: Identifier
    mechanism: MechanismV31
    baseline_skeleton: SkeletonV310 | None
    baseline_estimator: EstimatorV310 | None
    candidate_skeleton: SkeletonV310 | None
    candidate_estimator: EstimatorV310 | None
    baseline_adjudicated_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    candidate_adjudicated_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    candidate_improvement: Annotated[float, Field(allow_inf_nan=False)]
    material_negative_transfer: bool
    pair_private_target_losses: dict[str, Annotated[float, Field(ge=0, allow_inf_nan=False)]]
    private_values_visible_to_generator: Literal[False] = False

    @model_validator(mode="after")
    def validate_result(self) -> "PrivateFactorialCaseResultV310":
        expected = (
            self.baseline_adjudicated_target_loss
            - self.candidate_adjudicated_target_loss
        )
        if not math.isclose(self.candidate_improvement, expected, abs_tol=1e-12):
            raise ValueError("V3.10 case improvement does not recompute")
        return self


class SkeletonFactorialEvolutionReportV310(StrictModel):
    schema_version: Literal["3.10"] = "3.10"
    evolution_id: Identifier
    spec_hash: Sha256
    source_v391_evolution_hash: Sha256
    bundle_hash: Sha256
    case_results: list[PrivateFactorialCaseResultV310] = Field(min_length=39, max_length=39)
    performance_case_count: Literal[39] = 39
    quality_case_count: Literal[9] = 9
    baseline_coverage: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    candidate_coverage: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    baseline_mean_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    candidate_mean_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    paired_mean_target_loss_improvement: Annotated[float, Field(allow_inf_nan=False)]
    paired_improvement_ci_lower: Annotated[float, Field(allow_inf_nan=False)]
    paired_improvement_ci_upper: Annotated[float, Field(allow_inf_nan=False)]
    baseline_mean_loss_by_mechanism: dict[MechanismV31, Annotated[float, Field(ge=0, allow_inf_nan=False)]]
    candidate_mean_loss_by_mechanism: dict[MechanismV31, Annotated[float, Field(ge=0, allow_inf_nan=False)]]
    mean_private_loss_by_factor_pair: dict[str, Annotated[float, Field(ge=0, allow_inf_nan=False)]]
    candidate_selection_counts: dict[str, Annotated[int, Field(ge=0)]]
    valid_input_binding_count: Annotated[int, Field(ge=0)]
    expected_input_binding_count: Literal[1872] = 1872
    material_negative_transfer_count: Annotated[int, Field(ge=0, le=39)]
    material_negative_transfer_upper_95: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    public_quality_partition_only: Literal[True] = True
    private_performance_eligible_used_for_partition: Literal[False] = False
    gates: dict[Identifier, bool]
    ready_for_cross_domain_skeleton_confirmation: bool
    task_router_permitted: Literal[False] = False
    model_qualification_permitted: Literal[False] = False
    real_world_execution_permitted: Literal[False] = False
    status: Literal[
        "skeleton_factorial_ready_for_cross_domain_v310",
        "skeleton_factorial_refuted_v310",
    ]
    created_at: datetime
    evolution_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "SkeletonFactorialEvolutionReportV310":
        _assert_timezone(self.created_at, "created_at")
        ready = all(self.gates.values())
        expected = (
            "skeleton_factorial_ready_for_cross_domain_v310"
            if ready else "skeleton_factorial_refuted_v310"
        )
        if self.ready_for_cross_domain_skeleton_confirmation != ready or self.status != expected:
            raise ValueError("V3.10 report status disagrees with gates")
        if set(self.baseline_mean_loss_by_mechanism) != set(MECHANISMS_V31):
            raise ValueError("V3.10 baseline mechanism report incomplete")
        if set(self.candidate_mean_loss_by_mechanism) != set(MECHANISMS_V31):
            raise ValueError("V3.10 candidate mechanism report incomplete")
        if self.evolution_hash and self.evolution_hash != self.content_hash():
            raise ValueError("evolution_hash does not match V3.10 report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evolution_hash")

    def assert_sealed(self) -> None:
        if not self.evolution_hash or self.evolution_hash != self.content_hash():
            raise ValueError("V3.10 report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "SkeletonFactorialEvolutionReportV310":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evolution_hash"}),
            evolution_hash=draft.content_hash(),
        )


def _selected_pair_v310(
    receipt: SkeletonFactorialCaseReceiptV310,
    decision: FactorialDecisionV310,
) -> FactorPairReceiptV310 | None:
    if decision.selected_pair_hash is None:
        return None
    matches = [item for item in receipt.pairs if item.pair_hash == decision.selected_pair_hash]
    if len(matches) != 1:
        raise ValueError("V3.10 decision did not resolve one selected pair")
    return matches[0]


def _bootstrap_ci_v310(
    values: np.ndarray,
    spec: SkeletonFactorialWorldPackSpecV310,
) -> tuple[float, float]:
    random = np.random.default_rng(spec.bootstrap_seed)
    indices = random.integers(
        0, len(values), size=(spec.bootstrap_replicates, len(values))
    )
    means = np.mean(values[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate_skeleton_factorial_v310(
    spec: SkeletonFactorialWorldPackSpecV310,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    bundle: SkeletonFactorialBundleV310,
    *,
    evaluated_at: datetime,
) -> SkeletonFactorialEvolutionReportV310:
    for artifact in (spec, private_pack, bundle):
        artifact.assert_sealed()
    if (
        private_pack.spec_hash != spec.spec_hash
        or bundle.spec_hash != spec.spec_hash
        or bundle.private_pack_hash != private_pack.pack_hash
    ):
        raise ValueError("V3.10 evaluator artifact binding differs")
    private_by_id = {item.public_case.case_id: item for item in private_pack.cases}
    receipt_by_id = {item.case_id: item for item in bundle.case_receipts}
    results: list[PrivateFactorialCaseResultV310] = []
    quality_count = 0
    baseline_selected = 0
    candidate_selected = 0
    valid_bindings = 0
    factorial_complete = True
    public_partition_only = True
    private_separation = True
    baseline_by_mechanism: dict[str, list[float]] = defaultdict(list)
    candidate_by_mechanism: dict[str, list[float]] = defaultdict(list)
    factor_losses: dict[str, list[float]] = defaultdict(list)
    selection_counts: dict[str, int] = defaultdict(int)
    for case_id, private_case in private_by_id.items():
        receipt = receipt_by_id[case_id]
        public_flags = list(private_case.public_case.pilot.quality_flags)
        if receipt.public_quality_flags != public_flags:
            public_partition_only = False
        if public_flags:
            quality_count += 1
            if (
                receipt.baseline_decision.decision != "abstain"
                or receipt.candidate_decision.decision != "abstain"
            ):
                public_partition_only = False
            continue
        if len(receipt.cells) != 16 or len(receipt.pairs) != 8:
            factorial_complete = False
        valid_bindings += sum(
            binding.contract_valid
            for cell in receipt.cells
            for binding in cell.input_bindings
        )
        private_separation = private_separation and all(
            not cell.private_values_used for cell in receipt.cells
        ) and not receipt.baseline_decision.private_values_used and not receipt.candidate_decision.private_values_used
        baseline_pair = _selected_pair_v310(receipt, receipt.baseline_decision)
        candidate_pair = _selected_pair_v310(receipt, receipt.candidate_decision)
        baseline_selected += int(baseline_pair is not None)
        candidate_selected += int(candidate_pair is not None)
        baseline_loss = (
            _target_loss_v31(private_case, baseline_pair.final_model, spec)
            if baseline_pair is not None else spec.unresolved_adjudicated_loss
        )
        candidate_loss = (
            _target_loss_v31(private_case, candidate_pair.final_model, spec)
            if candidate_pair is not None else spec.unresolved_adjudicated_loss
        )
        pair_private: dict[str, float] = {}
        for pair in receipt.pairs:
            key = f"{pair.skeleton}|{pair.estimator}"
            loss = _target_loss_v31(private_case, pair.final_model, spec)
            pair_private[key] = loss
            factor_losses[key].append(loss)
        if candidate_pair is not None:
            selection_counts[
                f"{candidate_pair.skeleton}|{candidate_pair.estimator}"
            ] += 1
        baseline_by_mechanism[private_case.mechanism].append(baseline_loss)
        candidate_by_mechanism[private_case.mechanism].append(candidate_loss)
        improvement = baseline_loss - candidate_loss
        results.append(PrivateFactorialCaseResultV310(
            case_id=case_id,
            mechanism=private_case.mechanism,
            baseline_skeleton=(baseline_pair.skeleton if baseline_pair else None),
            baseline_estimator=(baseline_pair.estimator if baseline_pair else None),
            candidate_skeleton=(candidate_pair.skeleton if candidate_pair else None),
            candidate_estimator=(candidate_pair.estimator if candidate_pair else None),
            baseline_adjudicated_target_loss=baseline_loss,
            candidate_adjudicated_target_loss=candidate_loss,
            candidate_improvement=improvement,
            material_negative_transfer=(
                candidate_loss - baseline_loss > spec.material_negative_transfer
            ),
            pair_private_target_losses=pair_private,
        ))
    baseline_losses = np.asarray([
        item.baseline_adjudicated_target_loss for item in results
    ])
    candidate_losses = np.asarray([
        item.candidate_adjudicated_target_loss for item in results
    ])
    improvements = baseline_losses - candidate_losses
    ci_lower, ci_upper = _bootstrap_ci_v310(improvements, spec)
    negatives = sum(item.material_negative_transfer for item in results)
    negative_upper = (
        float(beta.ppf(0.95, negatives + 1, len(results) - negatives))
        if len(results) > negatives else 1.0
    )
    baseline_means = {
        mechanism: float(np.mean(baseline_by_mechanism[mechanism]))
        for mechanism in MECHANISMS_V31
    }
    candidate_means = {
        mechanism: float(np.mean(candidate_by_mechanism[mechanism]))
        for mechanism in MECHANISMS_V31
    }
    all_selection_keys = [
        f"{skeleton}|{estimator}"
        for skeleton in SKELETONS_V310
        for estimator in ESTIMATORS_V310
    ]
    complete_selection_counts = {
        key: selection_counts.get(key, 0) for key in all_selection_keys
    }
    structured_first = sum(
        value for key, value in complete_selection_counts.items()
        if key.startswith("first_order_rate_law|")
    )
    structured_second = sum(
        value for key, value in complete_selection_counts.items()
        if key.startswith("second_order_kinematic_force_law|")
    )
    mechanism_nonregression = all(
        candidate_means[mechanism]
        <= baseline_means[mechanism] + spec.maximum_mechanism_regression
        for mechanism in MECHANISMS_V31
        if mechanism != "duffing_oscillator"
    )
    duffing_improvement = (
        baseline_means["duffing_oscillator"]
        - candidate_means["duffing_oscillator"]
    )
    gates = {
        "public_quality_partition_complete": (
            public_partition_only
            and quality_count == spec.expected_quality_abstention_count
        ),
        "factorial_matrix_complete": factorial_complete,
        "semantic_input_contract_complete": (
            valid_bindings == 1872
        ),
        "private_generator_separation": private_separation,
        "candidate_coverage": candidate_selected / len(results) >= 0.9,
        "both_structured_topologies_exercised": (
            structured_first > 0 and structured_second > 0
        ),
        "paired_improvement_ci_lower_positive": ci_lower > 0.0,
        "duffing_gap_reduced": (
            duffing_improvement >= spec.required_duffing_improvement
            and candidate_means["duffing_oscillator"]
            <= spec.maximum_duffing_candidate_loss
        ),
        "other_mechanisms_non_regressing": mechanism_nonregression,
        "material_negative_transfer_controlled": negative_upper <= 0.1,
        "no_task_router_or_real_world_execution": True,
    }
    ready = all(gates.values())
    return SkeletonFactorialEvolutionReportV310.seal(
        evolution_id=f"evolution_{spec.experiment_id}",
        spec_hash=spec.spec_hash,
        source_v391_evolution_hash=spec.source_v391_evolution_hash,
        bundle_hash=bundle.bundle_hash,
        case_results=results,
        quality_case_count=quality_count,
        baseline_coverage=baseline_selected / len(results),
        candidate_coverage=candidate_selected / len(results),
        baseline_mean_target_loss=float(np.mean(baseline_losses)),
        candidate_mean_target_loss=float(np.mean(candidate_losses)),
        paired_mean_target_loss_improvement=float(np.mean(improvements)),
        paired_improvement_ci_lower=ci_lower,
        paired_improvement_ci_upper=ci_upper,
        baseline_mean_loss_by_mechanism=baseline_means,
        candidate_mean_loss_by_mechanism=candidate_means,
        mean_private_loss_by_factor_pair={
            key: float(np.mean(values)) for key, values in factor_losses.items()
        },
        candidate_selection_counts=complete_selection_counts,
        valid_input_binding_count=valid_bindings,
        material_negative_transfer_count=negatives,
        material_negative_transfer_upper_95=negative_upper,
        gates=gates,
        ready_for_cross_domain_skeleton_confirmation=ready,
        status=(
            "skeleton_factorial_ready_for_cross_domain_v310"
            if ready else "skeleton_factorial_refuted_v310"
        ),
        created_at=evaluated_at,
    )


class SkeletonFactorialManifestV310(StrictModel):
    schema_version: Literal["3.10"] = "3.10"
    run_id: Identifier
    artifact_refs: list[ArtifactRef] = Field(min_length=6, max_length=6)
    terminal_status: Literal[
        "skeleton_factorial_ready_for_cross_domain_v310",
        "skeleton_factorial_refuted_v310",
    ]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "SkeletonFactorialManifestV310":
        _assert_timezone(self.created_at, "created_at")
        if len({item.kind for item in self.artifact_refs}) != 6:
            raise ValueError("V3.10 manifest kinds differ")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match V3.10 manifest")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("V3.10 manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "SkeletonFactorialManifestV310":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class SkeletonFactorialOutcomeV310:
    store: RunStore
    spec: SkeletonFactorialWorldPackSpecV310
    private_pack: PrivateControlledDynamicsWorldPackV31
    bundle: SkeletonFactorialBundleV310
    evolution_report: SkeletonFactorialEvolutionReportV310
    manifest: SkeletonFactorialManifestV310


def run_skeleton_factorial_worldpack_v310(
    output_root: str | Path,
    *,
    source_v391_run_directory: str | Path,
    evidence: SkeletonFactorialMethodEvidenceV310,
    policy: SkeletonFactorialPolicyV310,
    spec: SkeletonFactorialWorldPackSpecV310,
    evaluated_at: datetime | None = None,
    run_id: str | None = None,
) -> SkeletonFactorialOutcomeV310:
    source = _load_source_v391(source_v391_run_directory)
    for artifact in (evidence, policy, spec):
        artifact.assert_sealed()
    if (
        source.evolution_hash != policy.source_v391_evolution_hash
        or source.evolution_hash != spec.source_v391_evolution_hash
        or evidence.evidence_hash != policy.method_evidence_hash
        or evidence.evidence_hash != spec.method_evidence_hash
        or policy.policy_hash != spec.policy_hash
    ):
        raise ValueError("V3.10 frozen lineage binding differs")
    at = evaluated_at or datetime.now(timezone.utc)
    store = RunStore(
        output_root,
        run_id=run_id or f"skeleton-factorial-v310-{uuid4().hex[:10]}",
    )
    refs = [
        store.put_artifact("skeleton_factorial_method_evidence_v310", evidence),
        store.put_artifact("skeleton_factorial_policy_v310", policy),
        store.put_artifact("skeleton_factorial_spec_v310", spec),
    ]
    store.emit("skeleton_factorial_v310_protocol_frozen_before_private_pack", {
        "spec_hash": spec.spec_hash,
        "source_v391_evolution_hash": source.evolution_hash,
        "frozen_delta": spec.frozen_delta,
        "private_generator_separation": True,
    })
    private_pack = generate_private_controlled_dynamics_worldpack_v31(
        spec, generated_at=at
    )
    bundle = execute_skeleton_factorial_v310(
        spec, private_pack, policy, executed_at=at
    )
    evolution = evaluate_skeleton_factorial_v310(
        spec, private_pack, bundle, evaluated_at=at
    )
    refs.extend([
        store.put_artifact("private_skeleton_factorial_worldpack_v310", private_pack),
        store.put_artifact("skeleton_factorial_bundle_v310", bundle),
        store.put_artifact("skeleton_factorial_evolution_report_v310", evolution),
    ])
    manifest = SkeletonFactorialManifestV310.seal(
        run_id=store.run_id,
        artifact_refs=refs,
        terminal_status=evolution.status,
        created_at=at,
    )
    manifest_ref = store.put_artifact("skeleton_factorial_manifest_v310", manifest)
    store.emit("skeleton_factorial_v310_worldpack_adjudicated", {
        "manifest_ref": manifest_ref.model_dump(mode="json")
    })
    if not verify_skeleton_factorial_run_v310(
        store.run_directory,
        source_v391_run_directory=source_v391_run_directory,
    ):
        raise RuntimeError("V3.10 run failed independent verification")
    return SkeletonFactorialOutcomeV310(
        store, spec, private_pack, bundle, evolution, manifest
    )


def verify_skeleton_factorial_run_v310(
    run_directory: str | Path,
    *,
    source_v391_run_directory: str | Path,
) -> bool:
    try:
        source = _load_source_v391(source_v391_run_directory)
        store = RunStore.open_existing(run_directory)
        if not store.verify_event_chain():
            return False
        events = [
            json.loads(line)
            for line in store.event_path.read_text(encoding="utf-8").splitlines()
        ]
        refs = _committed_refs_v310(store)
        if len(refs) != 7:
            return False
        for ref in refs:
            store.load_artifact(ref)
        evidence = _load_one_v310(
            store, refs, "skeleton_factorial_method_evidence_v310",
            SkeletonFactorialMethodEvidenceV310,
        )
        policy = _load_one_v310(
            store, refs, "skeleton_factorial_policy_v310",
            SkeletonFactorialPolicyV310,
        )
        spec = _load_one_v310(
            store, refs, "skeleton_factorial_spec_v310",
            SkeletonFactorialWorldPackSpecV310,
        )
        private_pack = _load_one_v310(
            store, refs, "private_skeleton_factorial_worldpack_v310",
            PrivateControlledDynamicsWorldPackV31,
        )
        bundle = _load_one_v310(
            store, refs, "skeleton_factorial_bundle_v310",
            SkeletonFactorialBundleV310,
        )
        evolution = _load_one_v310(
            store, refs, "skeleton_factorial_evolution_report_v310",
            SkeletonFactorialEvolutionReportV310,
        )
        manifest = _load_one_v310(
            store, refs, "skeleton_factorial_manifest_v310",
            SkeletonFactorialManifestV310,
        )
        for artifact in (
            evidence, policy, spec, private_pack, bundle, evolution, manifest
        ):
            artifact.assert_sealed()
        if (
            source.evolution_hash != spec.source_v391_evolution_hash
            or source.evolution_hash != policy.source_v391_evolution_hash
            or evidence.evidence_hash != spec.method_evidence_hash
            or evidence.evidence_hash != policy.method_evidence_hash
            or policy.policy_hash != spec.policy_hash
        ):
            return False
        regenerated = generate_private_controlled_dynamics_worldpack_v31(
            spec, generated_at=private_pack.generated_at
        )
        if regenerated.pack_hash != private_pack.pack_hash:
            return False
        recomputed_bundle = execute_skeleton_factorial_v310(
            spec, private_pack, policy, executed_at=bundle.created_at
        )
        if recomputed_bundle.bundle_hash != bundle.bundle_hash:
            return False
        recomputed_report = evaluate_skeleton_factorial_v310(
            spec, private_pack, bundle, evaluated_at=evolution.created_at
        )
        if recomputed_report.evolution_hash != evolution.evolution_hash:
            return False
        if (
            manifest.terminal_status != evolution.status
            or [item.model_dump(mode="json") for item in manifest.artifact_refs]
            != [item.model_dump(mode="json") for item in refs[:6]]
        ):
            return False
        event_types = [event["event_type"] for event in events]
        freeze_index = event_types.index(
            "skeleton_factorial_v310_protocol_frozen_before_private_pack"
        )
        private_commit_index = next(
            index for index, event in enumerate(events)
            if event["event_type"] == "artifact_committed"
            and event["payload"]["kind"]
            == "private_skeleton_factorial_worldpack_v310"
        )
        return freeze_index < private_commit_index
    except (
        OSError, RuntimeError, ValueError, KeyError, TypeError,
        json.JSONDecodeError, FloatingPointError, AttributeError,
        np.linalg.LinAlgError,
    ):
        return False
