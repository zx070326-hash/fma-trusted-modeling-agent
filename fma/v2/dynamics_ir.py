from __future__ import annotations

import math
from datetime import datetime, timezone
from itertools import combinations_with_replacement
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

from fma.hashing import sha256_value
from fma.schemas import StrictModel

from .schemas import Identifier, Sha256, _assert_timezone


Arm = Literal["direct_generation", "retrieval_evolution_memory"]
FitReason = Literal[
    "partial_state_observation",
    "candidate_library_rank_deficient",
    "candidate_library_ill_conditioned",
    "nonfinite_fit",
    "simulation_failed",
]


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class DynamicsDataSnapshotV24(StrictModel):
    """Public, uniformly sampled state trajectory available to a generator."""

    schema_version: Literal["2.4"] = "2.4"
    snapshot_id: Identifier
    declared_state_names: list[Identifier] = Field(min_length=1, max_length=4)
    observed_state_names: list[Identifier] = Field(min_length=1, max_length=4)
    times: list[Annotated[float, Field(allow_inf_nan=False)]] = Field(min_length=21)
    values: list[list[Annotated[float, Field(allow_inf_nan=False)]]] = Field(
        min_length=21
    )
    trust_class: Literal["untrusted_observation_data"] = "untrusted_observation_data"
    snapshot_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_snapshot(self) -> "DynamicsDataSnapshotV24":
        if len(self.declared_state_names) != len(set(self.declared_state_names)):
            raise ValueError("declared dynamics state names must be unique")
        if len(self.observed_state_names) != len(set(self.observed_state_names)):
            raise ValueError("observed dynamics state names must be unique")
        if not set(self.observed_state_names).issubset(self.declared_state_names):
            raise ValueError("observed states must be declared states")
        if len(self.times) != len(self.values):
            raise ValueError("dynamics times and values have different lengths")
        if any(len(row) != len(self.observed_state_names) for row in self.values):
            raise ValueError("dynamics value rows do not match observed states")
        deltas = np.diff(np.asarray(self.times, dtype=float))
        if (deltas <= 0).any():
            raise ValueError("dynamics times must be strictly increasing")
        if not np.allclose(deltas, deltas[0], rtol=1e-9, atol=1e-12):
            raise ValueError("V2.4 dynamics adapter requires uniform sampling")
        if self.snapshot_hash and self.snapshot_hash != self.content_hash():
            raise ValueError("snapshot_hash does not match dynamics data")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "snapshot_hash")

    def assert_sealed(self) -> None:
        if not self.snapshot_hash or self.snapshot_hash != self.content_hash():
            raise ValueError("dynamics data snapshot is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DynamicsDataSnapshotV24":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"snapshot_hash"}),
            snapshot_hash=draft.content_hash(),
        )


class DynamicsCandidateDefinitionV24(StrictModel):
    schema_version: Literal["2.4"] = "2.4"
    candidate_id: Identifier
    polynomial_degree: Annotated[int, Field(ge=1, le=3)]
    derivative_estimator: Literal["savgol"] = "savgol"
    savgol_window: Annotated[int, Field(ge=5, le=51)] = 11
    savgol_polynomial_order: Annotated[int, Field(ge=2, le=5)] = 3
    regression_kind: Literal["ridge", "stlsq"]
    ridge_alpha: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
    sparsity_threshold: Annotated[float, Field(ge=0, le=10, allow_inf_nan=False)]
    maximum_iterations: Annotated[int, Field(ge=1, le=50)] = 12
    definition_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_definition(self) -> "DynamicsCandidateDefinitionV24":
        if self.savgol_window % 2 == 0:
            raise ValueError("Savitzky-Golay window must be odd")
        if self.savgol_polynomial_order >= self.savgol_window:
            raise ValueError("Savitzky-Golay order must be smaller than its window")
        if self.regression_kind == "ridge" and self.sparsity_threshold != 0:
            raise ValueError("dense ridge candidate cannot set a sparsity threshold")
        if self.regression_kind == "stlsq" and self.sparsity_threshold <= 0:
            raise ValueError("STLSQ candidate requires a positive threshold")
        if self.definition_hash and self.definition_hash != self.content_hash():
            raise ValueError("definition_hash does not match dynamics candidate")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "definition_hash")

    def assert_sealed(self) -> None:
        if not self.definition_hash or self.definition_hash != self.content_hash():
            raise ValueError("dynamics candidate definition is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DynamicsCandidateDefinitionV24":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"definition_hash"}),
            definition_hash=draft.content_hash(),
        )


class DynamicsArmPolicyV24(StrictModel):
    schema_version: Literal["2.4"] = "2.4"
    policy_id: Identifier
    arm: Arm
    knowledge_bundle_hash: Sha256 | None = None
    evolution_evidence_hash: Sha256 | None = None
    candidates: list[DynamicsCandidateDefinitionV24] = Field(min_length=4, max_length=4)
    selection_rule: Literal[
        "inner_trajectory_nrmse_plus_complexity",
        "safety_guarded_sparse_vs_dense",
    ] = (
        "inner_trajectory_nrmse_plus_complexity"
    )
    safety_improvement_margin: Annotated[
        float, Field(ge=0, lt=0.5, allow_inf_nan=False)
    ] = 0.0
    complexity_penalty_per_active_coefficient: Annotated[
        float, Field(ge=0, le=0.1, allow_inf_nan=False)
    ]
    maximum_normalized_condition_number: Annotated[
        float, Field(gt=1, le=1e12, allow_inf_nan=False)
    ]
    candidate_budget: Literal[4] = 4
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "DynamicsArmPolicyV24":
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("dynamics policy candidate ids must be unique")
        for candidate in self.candidates:
            candidate.assert_sealed()
        if self.arm == "retrieval_evolution_memory" and not self.knowledge_bundle_hash:
            raise ValueError("memory policy must bind an exact knowledge bundle")
        if self.arm == "direct_generation" and self.knowledge_bundle_hash is not None:
            raise ValueError("direct policy cannot claim a knowledge-bundle dependency")
        if self.arm == "direct_generation" and self.evolution_evidence_hash is not None:
            raise ValueError("direct policy cannot claim failure-evolution evidence")
        if self.selection_rule == "safety_guarded_sparse_vs_dense":
            kinds = {candidate.regression_kind for candidate in self.candidates}
            if self.arm != "retrieval_evolution_memory" or kinds != {"ridge", "stlsq"}:
                raise ValueError("guarded selection requires a memory policy with dense and sparse candidates")
            if self.safety_improvement_margin <= 0:
                raise ValueError("guarded selection requires a positive safety margin")
        elif self.safety_improvement_margin != 0:
            raise ValueError("ordinary Dynamics selection cannot set a safety margin")
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("policy_hash does not match dynamics arm policy")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("dynamics arm policy is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DynamicsArmPolicyV24":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"policy_hash"}),
            policy_hash=draft.content_hash(),
        )


class PolynomialBasisTermV24(StrictModel):
    term_id: Identifier
    exponents: list[Annotated[int, Field(ge=0, le=3)]] = Field(min_length=1, max_length=4)


class TrajectoryIdentifiabilityDiagnosticV24(StrictModel):
    diagnostic_scope: Literal["empirical_library_rank_on_frozen_trajectory"] = (
        "empirical_library_rank_on_frozen_trajectory"
    )
    observed_state_count: Annotated[int, Field(ge=1, le=4)]
    declared_state_count: Annotated[int, Field(ge=1, le=4)]
    library_term_count: Annotated[int, Field(ge=0, le=100)]
    normalized_design_rank: Annotated[int, Field(ge=0, le=100)]
    normalized_condition_number: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
    status: Literal[
        "trajectory_identifiable",
        "partial_observation",
        "rank_deficient",
        "ill_conditioned",
    ]
    structural_identifiability_proven: Literal[False] = False
    limitations: list[Annotated[str, Field(min_length=12)]] = Field(min_length=2)


class DynamicsModelIRV24(StrictModel):
    schema_version: Literal["2.4"] = "2.4"
    model_id: Identifier
    source_data_hash: Sha256
    source_policy_hash: Sha256
    candidate_definition_hash: Sha256
    state_names: list[Identifier] = Field(min_length=1, max_length=4)
    basis_terms: list[PolynomialBasisTermV24] = Field(min_length=2, max_length=100)
    coefficient_matrix: list[list[Annotated[float, Field(allow_inf_nan=False)]]]
    derivative_estimator: Literal["savgol"]
    regression_kind: Literal["ridge", "stlsq"]
    active_coefficient_count: Annotated[int, Field(ge=0, le=400)]
    fitted_at: datetime
    model_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_model(self) -> "DynamicsModelIRV24":
        _assert_timezone(self.fitted_at, "fitted_at")
        if len(self.state_names) != len(set(self.state_names)):
            raise ValueError("dynamics IR state names must be unique")
        if any(len(term.exponents) != len(self.state_names) for term in self.basis_terms):
            raise ValueError("dynamics basis exponent dimensions do not match states")
        if len({term.term_id for term in self.basis_terms}) != len(self.basis_terms):
            raise ValueError("dynamics basis term ids must be unique")
        if len(self.coefficient_matrix) != len(self.state_names):
            raise ValueError("dynamics coefficient rows do not match state equations")
        if any(len(row) != len(self.basis_terms) for row in self.coefficient_matrix):
            raise ValueError("dynamics coefficient columns do not match basis terms")
        actual_active = sum(abs(value) > 1e-12 for row in self.coefficient_matrix for value in row)
        if actual_active != self.active_coefficient_count:
            raise ValueError("active coefficient count does not match dynamics IR")
        if self.model_hash and self.model_hash != self.content_hash():
            raise ValueError("model_hash does not match dynamics IR")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "model_hash")

    def assert_sealed(self) -> None:
        if not self.model_hash or self.model_hash != self.content_hash():
            raise ValueError("dynamics model IR is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DynamicsModelIRV24":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"model_hash"}),
            model_hash=draft.content_hash(),
        )


class DynamicsFitResultV24(StrictModel):
    candidate_id: Identifier
    candidate_definition_hash: Sha256
    data_snapshot_hash: Sha256
    status: Literal["fit_succeeded", "needs_evidence"]
    reason_codes: list[FitReason]
    model: DynamicsModelIRV24 | None = None
    identifiability: TrajectoryIdentifiabilityDiagnosticV24
    derivative_rmse: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    fit_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_fit(self) -> "DynamicsFitResultV24":
        if (self.status == "fit_succeeded") != (self.model is not None):
            raise ValueError("successful dynamics fit requires a model and only it may contain one")
        if (self.status == "fit_succeeded") == bool(self.reason_codes):
            raise ValueError("successful fit needs no reasons; abstention needs reasons")
        if self.model is not None:
            self.model.assert_sealed()
        if self.fit_hash and self.fit_hash != self.content_hash():
            raise ValueError("fit_hash does not match dynamics fit result")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "fit_hash")

    def assert_sealed(self) -> None:
        if not self.fit_hash or self.fit_hash != self.content_hash():
            raise ValueError("dynamics fit result is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DynamicsFitResultV24":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"fit_hash"}),
            fit_hash=draft.content_hash(),
        )


class DynamicsCandidateInnerScoreV24(StrictModel):
    candidate_id: Identifier
    fit_hash: Sha256
    status: Literal["scored", "needs_evidence"]
    trajectory_nrmse: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    selection_score: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    active_coefficient_count: Annotated[int, Field(ge=0)] | None = None
    reason_codes: list[FitReason]

    @model_validator(mode="after")
    def validate_score(self) -> "DynamicsCandidateInnerScoreV24":
        values = (self.trajectory_nrmse, self.selection_score, self.active_coefficient_count)
        if self.status == "scored" and any(value is None for value in values):
            raise ValueError("scored dynamics candidate needs complete metrics")
        if self.status == "needs_evidence" and any(value is not None for value in values):
            raise ValueError("abstained dynamics candidate cannot contain score metrics")
        return self


class DynamicsSelectionReceiptV24(StrictModel):
    schema_version: Literal["2.4"] = "2.4"
    receipt_id: Identifier
    arm: Arm
    arm_policy_hash: Sha256
    public_data_hash: Sha256
    training_points: Annotated[int, Field(ge=21)]
    inner_validation_points: Annotated[int, Field(ge=5)]
    scores: list[DynamicsCandidateInnerScoreV24] = Field(min_length=4, max_length=4)
    status: Literal["selected", "needs_evidence"]
    selected_candidate_id: Identifier | None = None
    selected_model_hash: Sha256 | None = None
    candidate_fit_count: Literal[4] = 4
    inner_simulation_count: Annotated[int, Field(ge=0, le=4)]
    selected_at: datetime
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "DynamicsSelectionReceiptV24":
        _assert_timezone(self.selected_at, "selected_at")
        if len({score.candidate_id for score in self.scores}) != 4:
            raise ValueError("selection receipt needs four unique candidate scores")
        selected_fields = (self.selected_candidate_id, self.selected_model_hash)
        if self.status == "selected" and any(value is None for value in selected_fields):
            raise ValueError("selected receipt needs candidate and model hashes")
        if self.status == "needs_evidence" and any(value is not None for value in selected_fields):
            raise ValueError("abstention receipt cannot name a selected model")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match dynamics selection")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("dynamics selection receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DynamicsSelectionReceiptV24":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


def default_dynamics_arm_policy(
    arm: Arm,
    *,
    knowledge_bundle_hash: str | None = None,
) -> DynamicsArmPolicyV24:
    common = {
        "derivative_estimator": "savgol",
        "savgol_window": 11,
        "savgol_polynomial_order": 3,
        "ridge_alpha": 1e-8,
        "maximum_iterations": 12,
    }
    if arm == "direct_generation":
        definitions = [
            ("dense_linear", 1, "ridge", 0.0, 1e-8),
            ("dense_quadratic", 2, "ridge", 0.0, 1e-8),
            ("dense_cubic", 3, "ridge", 0.0, 1e-8),
            ("dense_quadratic_regularized", 2, "ridge", 0.0, 1e-3),
        ]
    else:
        if knowledge_bundle_hash is None:
            raise ValueError("memory dynamics policy requires knowledge_bundle_hash")
        definitions = [
            ("sparse_linear", 1, "stlsq", 0.01, 1e-8),
            ("sparse_quadratic", 2, "stlsq", 0.01, 1e-8),
            ("sparse_cubic", 3, "stlsq", 0.01, 1e-8),
            ("sparse_quadratic_conservative", 2, "stlsq", 0.05, 1e-8),
        ]
    candidates = [
        DynamicsCandidateDefinitionV24.seal(
            candidate_id=candidate_id,
            polynomial_degree=degree,
            regression_kind=regression,
            sparsity_threshold=threshold,
            **{**common, "ridge_alpha": alpha},
        )
        for candidate_id, degree, regression, threshold, alpha in definitions
    ]
    return DynamicsArmPolicyV24.seal(
        policy_id=f"dynamics_{arm}_policy_v24",
        arm=arm,
        knowledge_bundle_hash=knowledge_bundle_hash,
        evolution_evidence_hash=None,
        candidates=candidates,
        complexity_penalty_per_active_coefficient=0.002,
        maximum_normalized_condition_number=1e8,
    )


def safe_dynamics_arm_policy_v24(
    *,
    knowledge_bundle_hash: str,
    exploratory_report_hash: str,
) -> DynamicsArmPolicyV24:
    """Failure-evolved policy: remove unsafe cubic extrapolation, retain a dense fallback."""

    common = {
        "derivative_estimator": "savgol",
        "savgol_window": 11,
        "savgol_polynomial_order": 3,
        "ridge_alpha": 1e-8,
        "maximum_iterations": 12,
    }
    definitions = [
        ("safe_sparse_linear", 1, "stlsq", 0.01, 1e-8),
        ("safe_sparse_quadratic", 2, "stlsq", 0.01, 1e-8),
        ("safe_sparse_quadratic_conservative", 2, "stlsq", 0.05, 1e-8),
        ("safe_dense_quadratic_fallback", 2, "ridge", 0.0, 1e-3),
    ]
    candidates = [
        DynamicsCandidateDefinitionV24.seal(
            candidate_id=candidate_id,
            polynomial_degree=degree,
            regression_kind=regression,
            sparsity_threshold=threshold,
            **{**common, "ridge_alpha": alpha},
        )
        for candidate_id, degree, regression, threshold, alpha in definitions
    ]
    return DynamicsArmPolicyV24.seal(
        policy_id="dynamics_retrieval_evolution_safe_policy_v24",
        arm="retrieval_evolution_memory",
        knowledge_bundle_hash=knowledge_bundle_hash,
        evolution_evidence_hash=exploratory_report_hash,
        candidates=candidates,
        complexity_penalty_per_active_coefficient=0.002,
        maximum_normalized_condition_number=1e8,
    )


def guarded_dynamics_arm_policy_v24(
    *,
    knowledge_bundle_hash: str,
    failed_confirmation_report_hash: str,
    safety_improvement_margin: float = 0.10,
) -> DynamicsArmPolicyV24:
    """Second failure evolution: sparse candidates must beat a matched dense guard."""

    common = {
        "derivative_estimator": "savgol",
        "savgol_window": 11,
        "savgol_polynomial_order": 3,
        "maximum_iterations": 12,
    }
    definitions = [
        ("guard_dense_linear", 1, "ridge", 0.0, 1e-8),
        ("guard_dense_quadratic", 2, "ridge", 0.0, 1e-3),
        ("guard_sparse_linear", 1, "stlsq", 0.005, 1e-8),
        ("guard_sparse_quadratic", 2, "stlsq", 0.01, 1e-8),
    ]
    candidates = [
        DynamicsCandidateDefinitionV24.seal(
            candidate_id=candidate_id,
            polynomial_degree=degree,
            regression_kind=regression,
            ridge_alpha=alpha,
            sparsity_threshold=threshold,
            **common,
        )
        for candidate_id, degree, regression, threshold, alpha in definitions
    ]
    return DynamicsArmPolicyV24.seal(
        policy_id="dynamics_guarded_sparse_policy_v24",
        arm="retrieval_evolution_memory",
        knowledge_bundle_hash=knowledge_bundle_hash,
        evolution_evidence_hash=failed_confirmation_report_hash,
        candidates=candidates,
        selection_rule="safety_guarded_sparse_vs_dense",
        safety_improvement_margin=safety_improvement_margin,
        complexity_penalty_per_active_coefficient=0.0,
        maximum_normalized_condition_number=1e8,
    )


def polynomial_basis_terms(
    state_names: list[str], degree: int
) -> list[PolynomialBasisTermV24]:
    count = len(state_names)
    terms = [PolynomialBasisTermV24(term_id="one", exponents=[0] * count)]
    for total_degree in range(1, degree + 1):
        for indices in combinations_with_replacement(range(count), total_degree):
            exponents = [0] * count
            for index in indices:
                exponents[index] += 1
            pieces = [
                name if exponent == 1 else f"{name}{exponent}"
                for name, exponent in zip(state_names, exponents, strict=True)
                if exponent
            ]
            terms.append(
                PolynomialBasisTermV24(
                    term_id="_".join(pieces),
                    exponents=exponents,
                )
            )
    return terms


def evaluate_polynomial_library(
    values: np.ndarray, terms: list[PolynomialBasisTermV24]
) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("dynamics library expects a two-dimensional state matrix")
    columns = []
    for term in terms:
        column = np.ones(values.shape[0], dtype=float)
        for index, exponent in enumerate(term.exponents):
            if exponent:
                column *= values[:, index] ** exponent
        columns.append(column)
    result = np.column_stack(columns)
    if not np.isfinite(result).all():
        raise ValueError("dynamics candidate library contains nonfinite values")
    return result


def fit_dynamics_candidate(
    snapshot: DynamicsDataSnapshotV24,
    definition: DynamicsCandidateDefinitionV24,
    policy: DynamicsArmPolicyV24,
    *,
    fitted_at: datetime | None = None,
) -> DynamicsFitResultV24:
    snapshot.assert_sealed()
    definition.assert_sealed()
    policy.assert_sealed()
    if definition.definition_hash not in {
        candidate.definition_hash for candidate in policy.candidates
    }:
        raise ValueError("dynamics candidate is absent from the frozen arm policy")
    assert snapshot.snapshot_hash is not None
    assert definition.definition_hash is not None
    assert policy.policy_hash is not None
    at = fitted_at or datetime.now(timezone.utc)
    if snapshot.observed_state_names != snapshot.declared_state_names:
        diagnostic = TrajectoryIdentifiabilityDiagnosticV24(
            observed_state_count=len(snapshot.observed_state_names),
            declared_state_count=len(snapshot.declared_state_names),
            library_term_count=0,
            normalized_design_rank=0,
            normalized_condition_number=None,
            status="partial_observation",
            limitations=_identifiability_limitations(),
        )
        return DynamicsFitResultV24.seal(
            candidate_id=definition.candidate_id,
            candidate_definition_hash=definition.definition_hash,
            data_snapshot_hash=snapshot.snapshot_hash,
            status="needs_evidence",
            reason_codes=["partial_state_observation"],
            identifiability=diagnostic,
        )

    values = np.asarray(snapshot.values, dtype=float)
    times = np.asarray(snapshot.times, dtype=float)
    if len(values) < definition.savgol_window + 4:
        raise ValueError("dynamics snapshot is too short for the frozen derivative estimator")
    dt = float(times[1] - times[0])
    derivatives = np.column_stack(
        [
            savgol_filter(
                values[:, index],
                window_length=definition.savgol_window,
                polyorder=definition.savgol_polynomial_order,
                deriv=1,
                delta=dt,
                mode="interp",
            )
            for index in range(values.shape[1])
        ]
    )
    trim = definition.savgol_window // 2
    fitted_values = values[trim:-trim]
    targets = derivatives[trim:-trim]
    terms = polynomial_basis_terms(
        snapshot.declared_state_names, definition.polynomial_degree
    )
    library = evaluate_polynomial_library(fitted_values, terms)
    scales = np.sqrt(np.mean(library**2, axis=0))
    scales[scales < 1e-12] = 1.0
    normalized = library / scales
    rank = int(np.linalg.matrix_rank(normalized, tol=1e-10))
    condition = float(np.linalg.cond(normalized))
    if not math.isfinite(condition):
        condition = policy.maximum_normalized_condition_number * 10.0
    if rank < normalized.shape[1]:
        identifiability_status = "rank_deficient"
        reason: FitReason = "candidate_library_rank_deficient"
    elif condition > policy.maximum_normalized_condition_number:
        identifiability_status = "ill_conditioned"
        reason = "candidate_library_ill_conditioned"
    else:
        identifiability_status = "trajectory_identifiable"
        reason = "nonfinite_fit"  # replaced below on the success path
    diagnostic = TrajectoryIdentifiabilityDiagnosticV24(
        observed_state_count=len(snapshot.observed_state_names),
        declared_state_count=len(snapshot.declared_state_names),
        library_term_count=normalized.shape[1],
        normalized_design_rank=rank,
        normalized_condition_number=condition,
        status=identifiability_status,
        limitations=_identifiability_limitations(),
    )
    if identifiability_status != "trajectory_identifiable":
        return DynamicsFitResultV24.seal(
            candidate_id=definition.candidate_id,
            candidate_definition_hash=definition.definition_hash,
            data_snapshot_hash=snapshot.snapshot_hash,
            status="needs_evidence",
            reason_codes=[reason],
            identifiability=diagnostic,
        )

    coefficients = _fit_coefficients(library, targets, definition)
    predictions = library @ coefficients.T
    derivative_rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))
    if not np.isfinite(coefficients).all() or not math.isfinite(derivative_rmse):
        return DynamicsFitResultV24.seal(
            candidate_id=definition.candidate_id,
            candidate_definition_hash=definition.definition_hash,
            data_snapshot_hash=snapshot.snapshot_hash,
            status="needs_evidence",
            reason_codes=["nonfinite_fit"],
            identifiability=diagnostic,
        )
    coefficients[np.abs(coefficients) < 1e-12] = 0.0
    model = DynamicsModelIRV24.seal(
        model_id=f"{definition.candidate_id}_{snapshot.snapshot_id}",
        source_data_hash=snapshot.snapshot_hash,
        source_policy_hash=policy.policy_hash,
        candidate_definition_hash=definition.definition_hash,
        state_names=snapshot.declared_state_names,
        basis_terms=terms,
        coefficient_matrix=coefficients.tolist(),
        derivative_estimator=definition.derivative_estimator,
        regression_kind=definition.regression_kind,
        active_coefficient_count=int(np.count_nonzero(coefficients)),
        fitted_at=at,
    )
    return DynamicsFitResultV24.seal(
        candidate_id=definition.candidate_id,
        candidate_definition_hash=definition.definition_hash,
        data_snapshot_hash=snapshot.snapshot_hash,
        status="fit_succeeded",
        reason_codes=[],
        model=model,
        identifiability=diagnostic,
        derivative_rmse=derivative_rmse,
    )


def select_dynamics_candidate(
    public_snapshot: DynamicsDataSnapshotV24,
    policy: DynamicsArmPolicyV24,
    *,
    training_points: int,
    selected_at: datetime | None = None,
) -> DynamicsSelectionReceiptV24:
    public_snapshot.assert_sealed()
    policy.assert_sealed()
    if training_points < 21 or training_points >= len(public_snapshot.times) - 4:
        raise ValueError("invalid dynamics training/inner split")
    at = selected_at or datetime.now(timezone.utc)
    training = DynamicsDataSnapshotV24.seal(
        snapshot_id=f"{public_snapshot.snapshot_id}_training",
        declared_state_names=public_snapshot.declared_state_names,
        observed_state_names=public_snapshot.observed_state_names,
        times=public_snapshot.times[:training_points],
        values=public_snapshot.values[:training_points],
    )
    scores: list[DynamicsCandidateInnerScoreV24] = []
    successful: list[
        tuple[float, float, DynamicsFitResultV24, DynamicsCandidateDefinitionV24]
    ] = []
    simulation_count = 0
    for definition in policy.candidates:
        fit = fit_dynamics_candidate(training, definition, policy, fitted_at=at)
        assert fit.fit_hash is not None
        if fit.status != "fit_succeeded":
            scores.append(
                DynamicsCandidateInnerScoreV24(
                    candidate_id=definition.candidate_id,
                    fit_hash=fit.fit_hash,
                    status="needs_evidence",
                    reason_codes=fit.reason_codes,
                )
            )
            continue
        assert fit.model is not None
        try:
            validation_times = public_snapshot.times[training_points - 1 :]
            predicted = simulate_dynamics_model(
                fit.model,
                public_snapshot.values[training_points - 1],
                validation_times,
            )[1:]
            observed = public_snapshot.values[training_points:]
            nrmse = trajectory_nrmse(observed, predicted)
        except (RuntimeError, ValueError, FloatingPointError):
            scores.append(
                DynamicsCandidateInnerScoreV24(
                    candidate_id=definition.candidate_id,
                    fit_hash=fit.fit_hash,
                    status="needs_evidence",
                    reason_codes=["simulation_failed"],
                )
            )
            continue
        simulation_count += 1
        score = nrmse + (
            policy.complexity_penalty_per_active_coefficient
            * fit.model.active_coefficient_count
        )
        scores.append(
            DynamicsCandidateInnerScoreV24(
                candidate_id=definition.candidate_id,
                fit_hash=fit.fit_hash,
                status="scored",
                trajectory_nrmse=nrmse,
                selection_score=score,
                active_coefficient_count=fit.model.active_coefficient_count,
                reason_codes=[],
            )
        )
        successful.append((score, nrmse, fit, definition))
    assert public_snapshot.snapshot_hash is not None
    assert policy.policy_hash is not None
    if not successful:
        return DynamicsSelectionReceiptV24.seal(
            receipt_id=f"{public_snapshot.snapshot_id}_{policy.arm}_selection",
            arm=policy.arm,
            arm_policy_hash=policy.policy_hash,
            public_data_hash=public_snapshot.snapshot_hash,
            training_points=training_points,
            inner_validation_points=len(public_snapshot.times) - training_points,
            scores=scores,
            status="needs_evidence",
            inner_simulation_count=simulation_count,
            selected_at=at,
        )
    if policy.selection_rule == "safety_guarded_sparse_vs_dense":
        dense = [item for item in successful if item[3].regression_kind == "ridge"]
        sparse = [item for item in successful if item[3].regression_kind == "stlsq"]
        if dense and sparse:
            best_dense = min(dense, key=lambda item: (item[1], item[2].candidate_id))
            best_sparse = min(sparse, key=lambda item: (item[1], item[2].candidate_id))
            selected_item = (
                best_sparse
                if best_sparse[1]
                <= best_dense[1] * (1.0 - policy.safety_improvement_margin)
                else best_dense
            )
        else:
            selected_item = min(successful, key=lambda item: (item[0], item[2].candidate_id))
    else:
        selected_item = min(successful, key=lambda item: (item[0], item[2].candidate_id))
    _, _, selected, _ = selected_item
    assert selected.model is not None and selected.model.model_hash is not None
    return DynamicsSelectionReceiptV24.seal(
        receipt_id=f"{public_snapshot.snapshot_id}_{policy.arm}_selection",
        arm=policy.arm,
        arm_policy_hash=policy.policy_hash,
        public_data_hash=public_snapshot.snapshot_hash,
        training_points=training_points,
        inner_validation_points=len(public_snapshot.times) - training_points,
        scores=scores,
        status="selected",
        selected_candidate_id=selected.candidate_id,
        selected_model_hash=selected.model.model_hash,
        inner_simulation_count=simulation_count,
        selected_at=at,
    )


def simulate_dynamics_model(
    model: DynamicsModelIRV24,
    initial_state: list[float],
    times: list[float],
) -> list[list[float]]:
    model.assert_sealed()
    if len(initial_state) != len(model.state_names):
        raise ValueError("initial state does not match dynamics model")
    if len(times) < 2 or any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("simulation times must be strictly increasing")
    coefficients = np.asarray(model.coefficient_matrix, dtype=float)

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        row = evaluate_polynomial_library(state.reshape(1, -1), model.basis_terms)[0]
        derivative = coefficients @ row
        if not np.isfinite(derivative).all() or np.max(np.abs(state)) > 1e6:
            raise FloatingPointError("dynamics simulation diverged")
        return derivative

    try:
        solution = solve_ivp(
            rhs,
            (float(times[0]), float(times[-1])),
            np.asarray(initial_state, dtype=float),
            t_eval=np.asarray(times, dtype=float),
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )
    except (FloatingPointError, ValueError) as exc:
        raise RuntimeError("dynamics simulation failed") from exc
    if not solution.success or solution.y.shape[1] != len(times):
        raise RuntimeError("dynamics integrator did not cover the requested horizon")
    values = solution.y.T
    if not np.isfinite(values).all() or np.max(np.abs(values)) > 1e6:
        raise RuntimeError("dynamics simulation produced nonfinite or divergent states")
    return values.tolist()


def trajectory_nrmse(
    truth: list[list[float]], prediction: list[list[float]]
) -> float:
    actual = np.asarray(truth, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    if actual.shape != predicted.shape or actual.ndim != 2 or actual.size == 0:
        raise ValueError("trajectory metric arrays must be nonempty and shape-compatible")
    scales = np.std(actual, axis=0)
    fallback = np.maximum(np.mean(np.abs(actual), axis=0), 1.0)
    scales = np.where(scales > 1e-8, scales, fallback)
    per_state = np.sqrt(np.mean((predicted - actual) ** 2, axis=0)) / scales
    value = float(np.mean(per_state))
    if not math.isfinite(value):
        raise ValueError("trajectory NRMSE is not finite")
    return value


def support_f1(
    model: DynamicsModelIRV24,
    truth_coefficients: dict[str, dict[str, float]],
    *,
    nonzero_tolerance: float = 1e-6,
) -> float:
    model.assert_sealed()
    terms = [term.term_id for term in model.basis_terms]
    predicted_support = {
        (state_name, term_id)
        for state_name, row in zip(model.state_names, model.coefficient_matrix, strict=True)
        for term_id, value in zip(terms, row, strict=True)
        if abs(value) > nonzero_tolerance
    }
    truth_support = {
        (state_name, term_id)
        for state_name, coefficients in truth_coefficients.items()
        for term_id, value in coefficients.items()
        if abs(value) > nonzero_tolerance
    }
    true_positive = len(predicted_support & truth_support)
    if not predicted_support and not truth_support:
        return 1.0
    precision = true_positive / len(predicted_support) if predicted_support else 0.0
    recall = true_positive / len(truth_support) if truth_support else 0.0
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


def _fit_coefficients(
    library: np.ndarray,
    targets: np.ndarray,
    definition: DynamicsCandidateDefinitionV24,
) -> np.ndarray:
    scales = np.sqrt(np.mean(library**2, axis=0))
    scales[scales < 1e-12] = 1.0
    normalized = library / scales
    gram = normalized.T @ normalized
    regularizer = definition.ridge_alpha * np.eye(normalized.shape[1])
    coefficients_normalized = np.linalg.solve(
        gram + regularizer,
        normalized.T @ targets,
    ).T
    coefficients = coefficients_normalized / scales[np.newaxis, :]
    if definition.regression_kind == "ridge":
        return coefficients
    for equation in range(targets.shape[1]):
        active = np.abs(coefficients[equation]) >= definition.sparsity_threshold
        for _ in range(definition.maximum_iterations):
            previous = active.copy()
            coefficients[equation, ~active] = 0.0
            if active.any():
                selected = normalized[:, active]
                selected_gram = selected.T @ selected
                selected_coefficients = np.linalg.solve(
                    selected_gram
                    + definition.ridge_alpha * np.eye(selected.shape[1]),
                    selected.T @ targets[:, equation],
                )
                coefficients[equation, active] = selected_coefficients / scales[active]
            active = np.abs(coefficients[equation]) >= definition.sparsity_threshold
            if np.array_equal(active, previous):
                break
        coefficients[equation, ~active] = 0.0
    return coefficients


def _identifiability_limitations() -> list[str]:
    return [
        "rank is conditional on the frozen candidate library and observed trajectory",
        "this diagnostic is neither a symbolic structural-identifiability nor observability proof",
    ]
