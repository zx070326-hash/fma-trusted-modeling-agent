from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

from fma.hashing import sha256_value
from fma.schemas import StrictModel

from .dynamics_ir import (
    DynamicsDataSnapshotV24,
    PolynomialBasisTermV24,
    evaluate_polynomial_library,
    polynomial_basis_terms,
    trajectory_nrmse,
)
from .schemas import Identifier, Sha256, _assert_timezone


EstimatorArm = Literal["point_savgol", "window_integral_matching"]
RegressionKind = Literal["ridge", "stlsq"]
FitReasonV25 = Literal[
    "partial_state_observation",
    "candidate_library_rank_deficient",
    "candidate_library_ill_conditioned",
    "insufficient_estimation_rows",
    "nonfinite_fit",
    "simulation_failed",
]


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class DynamicsCandidateDefinitionV25(StrictModel):
    """Additive estimator definition; V2.4 schemas and hashes remain untouched."""

    schema_version: Literal["2.5"] = "2.5"
    candidate_id: Identifier
    estimator_arm: EstimatorArm
    polynomial_degree: Annotated[int, Field(ge=1, le=2)]
    savgol_window: Annotated[int, Field(ge=5, le=51)] | None = None
    savgol_polynomial_order: Annotated[int, Field(ge=2, le=5)] | None = None
    integral_window_points: Annotated[int, Field(ge=5, le=101)] | None = None
    integral_window_step: Annotated[int, Field(ge=1, le=20)] | None = None
    quadrature_rule: Literal["trapezoidal"] | None = None
    regression_kind: RegressionKind
    ridge_alpha: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
    sparsity_threshold: Annotated[float, Field(ge=0, le=10, allow_inf_nan=False)]
    maximum_iterations: Annotated[int, Field(ge=1, le=50)] = 12
    definition_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_definition(self) -> "DynamicsCandidateDefinitionV25":
        if self.estimator_arm == "point_savgol":
            if self.savgol_window is None or self.savgol_polynomial_order is None:
                raise ValueError("point estimator requires a frozen Savitzky-Golay configuration")
            if self.savgol_window % 2 == 0:
                raise ValueError("Savitzky-Golay window must be odd")
            if self.savgol_polynomial_order >= self.savgol_window:
                raise ValueError("Savitzky-Golay order must be smaller than its window")
            if any(
                value is not None
                for value in (
                    self.integral_window_points,
                    self.integral_window_step,
                    self.quadrature_rule,
                )
            ):
                raise ValueError("point estimator cannot claim an integral configuration")
        else:
            if (
                self.integral_window_points is None
                or self.integral_window_step is None
                or self.quadrature_rule != "trapezoidal"
            ):
                raise ValueError("integral estimator requires a frozen window and quadrature")
            if self.savgol_window is not None or self.savgol_polynomial_order is not None:
                raise ValueError("integral estimator cannot claim a point-derivative configuration")
        if self.regression_kind == "ridge" and self.sparsity_threshold != 0:
            raise ValueError("dense ridge candidate cannot set a sparsity threshold")
        if self.regression_kind == "stlsq" and self.sparsity_threshold <= 0:
            raise ValueError("STLSQ candidate requires a positive threshold")
        if self.definition_hash and self.definition_hash != self.content_hash():
            raise ValueError("definition_hash does not match V2.5 dynamics candidate")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "definition_hash")

    def assert_sealed(self) -> None:
        if not self.definition_hash or self.definition_hash != self.content_hash():
            raise ValueError("V2.5 dynamics candidate definition is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DynamicsCandidateDefinitionV25":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"definition_hash"}),
            definition_hash=draft.content_hash(),
        )


class DynamicsEstimatorPolicyV25(StrictModel):
    schema_version: Literal["2.5"] = "2.5"
    policy_id: Identifier
    estimator_arm: EstimatorArm
    knowledge_bundle_hash: Sha256
    failure_evidence_hash: Sha256
    candidates: list[DynamicsCandidateDefinitionV25] = Field(min_length=4, max_length=4)
    selection_rule: Literal["safety_guarded_sparse_vs_dense"] = (
        "safety_guarded_sparse_vs_dense"
    )
    safety_improvement_margin: Annotated[
        float, Field(gt=0, lt=0.5, allow_inf_nan=False)
    ]
    complexity_penalty_per_active_coefficient: Literal[0.0] = 0.0
    maximum_normalized_condition_number: Annotated[
        float, Field(gt=1, le=1e12, allow_inf_nan=False)
    ]
    candidate_budget: Literal[4] = 4
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "DynamicsEstimatorPolicyV25":
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("V2.5 dynamics policy candidate ids must be unique")
        if {candidate.estimator_arm for candidate in self.candidates} != {
            self.estimator_arm
        }:
            raise ValueError("all V2.5 candidates must use the policy estimator arm")
        if {candidate.regression_kind for candidate in self.candidates} != {
            "ridge",
            "stlsq",
        }:
            raise ValueError("guarded V2.5 policy requires dense and sparse candidates")
        for candidate in self.candidates:
            candidate.assert_sealed()
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("policy_hash does not match V2.5 dynamics policy")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("V2.5 dynamics estimator policy is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DynamicsEstimatorPolicyV25":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"policy_hash"}),
            policy_hash=draft.content_hash(),
        )


class EstimationDesignDiagnosticV25(StrictModel):
    diagnostic_scope: Literal[
        "point_derivative_library_on_frozen_trajectory",
        "window_integral_library_on_frozen_trajectory",
    ]
    observed_state_count: Annotated[int, Field(ge=1, le=4)]
    declared_state_count: Annotated[int, Field(ge=1, le=4)]
    estimation_row_count: Annotated[int, Field(ge=0, le=2_000)]
    library_term_count: Annotated[int, Field(ge=0, le=100)]
    normalized_design_rank: Annotated[int, Field(ge=0, le=100)]
    normalized_condition_number: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
    status: Literal[
        "trajectory_identifiable",
        "partial_observation",
        "rank_deficient",
        "ill_conditioned",
        "insufficient_rows",
    ]
    structural_identifiability_proven: Literal[False] = False
    row_independence_proven: Literal[False] = False
    limitations: list[Annotated[str, Field(min_length=12)]] = Field(min_length=3)


class DynamicsModelIRV25(StrictModel):
    schema_version: Literal["2.5"] = "2.5"
    model_id: Identifier
    source_data_hash: Sha256
    source_policy_hash: Sha256
    candidate_definition_hash: Sha256
    estimator_arm: EstimatorArm
    state_names: list[Identifier] = Field(min_length=1, max_length=4)
    basis_terms: list[PolynomialBasisTermV24] = Field(min_length=2, max_length=100)
    coefficient_matrix: list[list[Annotated[float, Field(allow_inf_nan=False)]]]
    regression_kind: RegressionKind
    active_coefficient_count: Annotated[int, Field(ge=0, le=400)]
    fitted_at: datetime
    model_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_model(self) -> "DynamicsModelIRV25":
        _assert_timezone(self.fitted_at, "fitted_at")
        if len(self.state_names) != len(set(self.state_names)):
            raise ValueError("V2.5 dynamics IR state names must be unique")
        if any(len(term.exponents) != len(self.state_names) for term in self.basis_terms):
            raise ValueError("V2.5 basis exponent dimensions do not match states")
        if len({term.term_id for term in self.basis_terms}) != len(self.basis_terms):
            raise ValueError("V2.5 dynamics basis term ids must be unique")
        if len(self.coefficient_matrix) != len(self.state_names):
            raise ValueError("V2.5 coefficient rows do not match state equations")
        if any(len(row) != len(self.basis_terms) for row in self.coefficient_matrix):
            raise ValueError("V2.5 coefficient columns do not match basis terms")
        actual = sum(
            abs(value) > 1e-12
            for row in self.coefficient_matrix
            for value in row
        )
        if actual != self.active_coefficient_count:
            raise ValueError("active coefficient count does not match V2.5 dynamics IR")
        if self.model_hash and self.model_hash != self.content_hash():
            raise ValueError("model_hash does not match V2.5 dynamics IR")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "model_hash")

    def assert_sealed(self) -> None:
        if not self.model_hash or self.model_hash != self.content_hash():
            raise ValueError("V2.5 dynamics model IR is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DynamicsModelIRV25":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"model_hash"}),
            model_hash=draft.content_hash(),
        )


class DynamicsFitResultV25(StrictModel):
    schema_version: Literal["2.5"] = "2.5"
    candidate_id: Identifier
    estimator_arm: EstimatorArm
    candidate_definition_hash: Sha256
    data_snapshot_hash: Sha256
    status: Literal["fit_succeeded", "needs_evidence"]
    reason_codes: list[FitReasonV25]
    model: DynamicsModelIRV25 | None = None
    design_diagnostic: EstimationDesignDiagnosticV25
    equation_residual_rmse: Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ] | None = None
    fit_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_fit(self) -> "DynamicsFitResultV25":
        if (self.status == "fit_succeeded") != (self.model is not None):
            raise ValueError("successful V2.5 fit requires a model")
        if (self.status == "fit_succeeded") == bool(self.reason_codes):
            raise ValueError("successful V2.5 fit needs no reasons; abstention needs reasons")
        if self.status == "fit_succeeded" and self.equation_residual_rmse is None:
            raise ValueError("successful V2.5 fit needs an equation residual")
        if self.status == "needs_evidence" and self.equation_residual_rmse is not None:
            raise ValueError("abstained V2.5 fit cannot claim an equation residual")
        if self.model is not None:
            self.model.assert_sealed()
        if self.fit_hash and self.fit_hash != self.content_hash():
            raise ValueError("fit_hash does not match V2.5 dynamics fit")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "fit_hash")

    def assert_sealed(self) -> None:
        if not self.fit_hash or self.fit_hash != self.content_hash():
            raise ValueError("V2.5 dynamics fit result is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DynamicsFitResultV25":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"fit_hash"}),
            fit_hash=draft.content_hash(),
        )


class DynamicsCandidateInnerScoreV25(StrictModel):
    candidate_id: Identifier
    fit_hash: Sha256
    status: Literal["scored", "needs_evidence"]
    trajectory_nrmse: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    selection_score: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    active_coefficient_count: Annotated[int, Field(ge=0)] | None = None
    reason_codes: list[FitReasonV25]

    @model_validator(mode="after")
    def validate_score(self) -> "DynamicsCandidateInnerScoreV25":
        values = (self.trajectory_nrmse, self.selection_score, self.active_coefficient_count)
        if self.status == "scored" and any(value is None for value in values):
            raise ValueError("scored V2.5 candidate needs complete metrics")
        if self.status == "needs_evidence" and any(value is not None for value in values):
            raise ValueError("abstained V2.5 candidate cannot contain score metrics")
        return self


class DynamicsSelectionReceiptV25(StrictModel):
    schema_version: Literal["2.5"] = "2.5"
    receipt_id: Identifier
    estimator_arm: EstimatorArm
    policy_hash: Sha256
    public_data_hash: Sha256
    training_points: Annotated[int, Field(ge=21)]
    inner_validation_points: Annotated[int, Field(ge=5)]
    scores: list[DynamicsCandidateInnerScoreV25] = Field(min_length=4, max_length=4)
    status: Literal["selected", "needs_evidence"]
    selected_candidate_id: Identifier | None = None
    selected_model_hash: Sha256 | None = None
    candidate_fit_count: Literal[4] = 4
    inner_simulation_count: Annotated[int, Field(ge=0, le=4)]
    selected_at: datetime
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "DynamicsSelectionReceiptV25":
        _assert_timezone(self.selected_at, "selected_at")
        if len({score.candidate_id for score in self.scores}) != 4:
            raise ValueError("V2.5 selection receipt needs four candidate scores")
        selected = (self.selected_candidate_id, self.selected_model_hash)
        if self.status == "selected" and any(value is None for value in selected):
            raise ValueError("selected V2.5 receipt needs candidate and model hashes")
        if self.status == "needs_evidence" and any(value is not None for value in selected):
            raise ValueError("abstained V2.5 receipt cannot name a model")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match V2.5 selection")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("V2.5 dynamics selection receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DynamicsSelectionReceiptV25":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


def default_estimator_policy_v25(
    estimator_arm: EstimatorArm,
    *,
    knowledge_bundle_hash: str,
    failure_evidence_hash: str,
    integral_window_points: int = 15,
    integral_window_step: int = 3,
) -> DynamicsEstimatorPolicyV25:
    common = {
        "ridge_alpha": 1e-8,
        "maximum_iterations": 12,
    }
    forms: dict[str, object]
    if estimator_arm == "point_savgol":
        forms = {
            "savgol_window": 11,
            "savgol_polynomial_order": 3,
        }
    else:
        forms = {
            "integral_window_points": integral_window_points,
            "integral_window_step": integral_window_step,
            "quadrature_rule": "trapezoidal",
        }
    definitions = [
        ("guard_dense_linear", 1, "ridge", 0.0, 1e-8),
        ("guard_dense_quadratic", 2, "ridge", 0.0, 1e-3),
        ("guard_sparse_linear", 1, "stlsq", 0.005, 1e-8),
        ("guard_sparse_quadratic", 2, "stlsq", 0.01, 1e-8),
    ]
    candidates = [
        DynamicsCandidateDefinitionV25.seal(
            candidate_id=candidate_id,
            estimator_arm=estimator_arm,
            polynomial_degree=degree,
            regression_kind=regression,
            sparsity_threshold=threshold,
            **{**common, **forms, "ridge_alpha": alpha},
        )
        for candidate_id, degree, regression, threshold, alpha in definitions
    ]
    return DynamicsEstimatorPolicyV25.seal(
        policy_id=f"dynamics_{estimator_arm}_policy_v25",
        estimator_arm=estimator_arm,
        knowledge_bundle_hash=knowledge_bundle_hash,
        failure_evidence_hash=failure_evidence_hash,
        candidates=candidates,
        safety_improvement_margin=0.10,
        maximum_normalized_condition_number=1e8,
    )


def assert_single_component_estimator_ablation_v25(
    point_policy: DynamicsEstimatorPolicyV25,
    integral_policy: DynamicsEstimatorPolicyV25,
) -> None:
    point_policy.assert_sealed()
    integral_policy.assert_sealed()
    if point_policy.estimator_arm != "point_savgol":
        raise ValueError("first V2.5 ablation policy must be point_savgol")
    if integral_policy.estimator_arm != "window_integral_matching":
        raise ValueError("second V2.5 ablation policy must be window_integral_matching")
    policy_fields = (
        "knowledge_bundle_hash",
        "failure_evidence_hash",
        "selection_rule",
        "safety_improvement_margin",
        "complexity_penalty_per_active_coefficient",
        "maximum_normalized_condition_number",
        "candidate_budget",
    )
    if any(
        getattr(point_policy, field) != getattr(integral_policy, field)
        for field in policy_fields
    ):
        raise ValueError("V2.5 estimator policies differ outside the estimator component")
    candidate_fields = (
        "candidate_id",
        "polynomial_degree",
        "regression_kind",
        "ridge_alpha",
        "sparsity_threshold",
        "maximum_iterations",
    )
    for point, integral in zip(
        point_policy.candidates, integral_policy.candidates, strict=True
    ):
        if any(getattr(point, field) != getattr(integral, field) for field in candidate_fields):
            raise ValueError("V2.5 estimator candidates differ outside equation construction")


def build_estimation_system_v25(
    snapshot: DynamicsDataSnapshotV24,
    definition: DynamicsCandidateDefinitionV25,
) -> tuple[np.ndarray, np.ndarray, list[PolynomialBasisTermV24]]:
    """Build the exact frozen linear system used by fitting and stability diagnostics."""

    snapshot.assert_sealed()
    definition.assert_sealed()
    values = np.asarray(snapshot.values, dtype=float)
    times = np.asarray(snapshot.times, dtype=float)
    terms = polynomial_basis_terms(
        snapshot.declared_state_names, definition.polynomial_degree
    )
    if definition.estimator_arm == "point_savgol":
        assert definition.savgol_window is not None
        assert definition.savgol_polynomial_order is not None
        if len(values) < definition.savgol_window + 4:
            raise ValueError("snapshot is too short for the point estimator")
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
        return (
            evaluate_polynomial_library(values[trim:-trim], terms),
            derivatives[trim:-trim],
            terms,
        )

    assert definition.integral_window_points is not None
    assert definition.integral_window_step is not None
    width = definition.integral_window_points
    if len(values) < width + 2:
        raise ValueError("snapshot is too short for the integral estimator")
    point_library = evaluate_polynomial_library(values, terms)
    design_rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for start in range(0, len(values) - width + 1, definition.integral_window_step):
        stop = start + width
        segment = point_library[start:stop]
        segment_times = times[start:stop]
        deltas = np.diff(segment_times)
        integral = np.sum(
            0.5 * (segment[:-1] + segment[1:]) * deltas[:, np.newaxis],
            axis=0,
        )
        design_rows.append(integral)
        targets.append(values[stop - 1] - values[start])
    return np.vstack(design_rows), np.vstack(targets), terms


def fit_dynamics_candidate_v25(
    snapshot: DynamicsDataSnapshotV24,
    definition: DynamicsCandidateDefinitionV25,
    policy: DynamicsEstimatorPolicyV25,
    *,
    fitted_at: datetime | None = None,
) -> DynamicsFitResultV25:
    snapshot.assert_sealed()
    definition.assert_sealed()
    policy.assert_sealed()
    if definition.definition_hash not in {
        candidate.definition_hash for candidate in policy.candidates
    }:
        raise ValueError("V2.5 candidate is absent from the frozen policy")
    assert snapshot.snapshot_hash is not None
    assert definition.definition_hash is not None
    assert policy.policy_hash is not None
    scope = (
        "point_derivative_library_on_frozen_trajectory"
        if definition.estimator_arm == "point_savgol"
        else "window_integral_library_on_frozen_trajectory"
    )
    if snapshot.observed_state_names != snapshot.declared_state_names:
        diagnostic = EstimationDesignDiagnosticV25(
            diagnostic_scope=scope,
            observed_state_count=len(snapshot.observed_state_names),
            declared_state_count=len(snapshot.declared_state_names),
            estimation_row_count=0,
            library_term_count=0,
            normalized_design_rank=0,
            normalized_condition_number=None,
            status="partial_observation",
            limitations=_design_limitations(definition.estimator_arm),
        )
        return DynamicsFitResultV25.seal(
            candidate_id=definition.candidate_id,
            estimator_arm=definition.estimator_arm,
            candidate_definition_hash=definition.definition_hash,
            data_snapshot_hash=snapshot.snapshot_hash,
            status="needs_evidence",
            reason_codes=["partial_state_observation"],
            design_diagnostic=diagnostic,
        )
    try:
        design, targets, terms = build_estimation_system_v25(snapshot, definition)
    except ValueError:
        diagnostic = EstimationDesignDiagnosticV25(
            diagnostic_scope=scope,
            observed_state_count=len(snapshot.observed_state_names),
            declared_state_count=len(snapshot.declared_state_names),
            estimation_row_count=0,
            library_term_count=0,
            normalized_design_rank=0,
            normalized_condition_number=None,
            status="insufficient_rows",
            limitations=_design_limitations(definition.estimator_arm),
        )
        return DynamicsFitResultV25.seal(
            candidate_id=definition.candidate_id,
            estimator_arm=definition.estimator_arm,
            candidate_definition_hash=definition.definition_hash,
            data_snapshot_hash=snapshot.snapshot_hash,
            status="needs_evidence",
            reason_codes=["insufficient_estimation_rows"],
            design_diagnostic=diagnostic,
        )
    scales = np.sqrt(np.mean(design**2, axis=0))
    scales[scales < 1e-12] = 1.0
    normalized = design / scales
    rank = int(np.linalg.matrix_rank(normalized, tol=1e-10))
    condition = float(np.linalg.cond(normalized))
    if not math.isfinite(condition):
        condition = policy.maximum_normalized_condition_number * 10.0
    if design.shape[0] < design.shape[1]:
        status = "insufficient_rows"
        reason: FitReasonV25 = "insufficient_estimation_rows"
    elif rank < normalized.shape[1]:
        status = "rank_deficient"
        reason = "candidate_library_rank_deficient"
    elif condition > policy.maximum_normalized_condition_number:
        status = "ill_conditioned"
        reason = "candidate_library_ill_conditioned"
    else:
        status = "trajectory_identifiable"
        reason = "nonfinite_fit"
    diagnostic = EstimationDesignDiagnosticV25(
        diagnostic_scope=scope,
        observed_state_count=len(snapshot.observed_state_names),
        declared_state_count=len(snapshot.declared_state_names),
        estimation_row_count=design.shape[0],
        library_term_count=design.shape[1],
        normalized_design_rank=rank,
        normalized_condition_number=condition,
        status=status,
        limitations=_design_limitations(definition.estimator_arm),
    )
    if status != "trajectory_identifiable":
        return DynamicsFitResultV25.seal(
            candidate_id=definition.candidate_id,
            estimator_arm=definition.estimator_arm,
            candidate_definition_hash=definition.definition_hash,
            data_snapshot_hash=snapshot.snapshot_hash,
            status="needs_evidence",
            reason_codes=[reason],
            design_diagnostic=diagnostic,
        )
    coefficients = fit_coefficients_v25(design, targets, definition)
    residual = float(np.sqrt(np.mean((design @ coefficients.T - targets) ** 2)))
    if not np.isfinite(coefficients).all() or not math.isfinite(residual):
        return DynamicsFitResultV25.seal(
            candidate_id=definition.candidate_id,
            estimator_arm=definition.estimator_arm,
            candidate_definition_hash=definition.definition_hash,
            data_snapshot_hash=snapshot.snapshot_hash,
            status="needs_evidence",
            reason_codes=["nonfinite_fit"],
            design_diagnostic=diagnostic,
        )
    coefficients[np.abs(coefficients) < 1e-12] = 0.0
    model = DynamicsModelIRV25.seal(
        model_id=f"{definition.candidate_id}_{snapshot.snapshot_id}_{definition.estimator_arm}",
        source_data_hash=snapshot.snapshot_hash,
        source_policy_hash=policy.policy_hash,
        candidate_definition_hash=definition.definition_hash,
        estimator_arm=definition.estimator_arm,
        state_names=snapshot.declared_state_names,
        basis_terms=terms,
        coefficient_matrix=coefficients.tolist(),
        regression_kind=definition.regression_kind,
        active_coefficient_count=int(np.count_nonzero(coefficients)),
        fitted_at=fitted_at or datetime.now(timezone.utc),
    )
    return DynamicsFitResultV25.seal(
        candidate_id=definition.candidate_id,
        estimator_arm=definition.estimator_arm,
        candidate_definition_hash=definition.definition_hash,
        data_snapshot_hash=snapshot.snapshot_hash,
        status="fit_succeeded",
        reason_codes=[],
        model=model,
        design_diagnostic=diagnostic,
        equation_residual_rmse=residual,
    )


def select_dynamics_candidate_v25(
    public_snapshot: DynamicsDataSnapshotV24,
    policy: DynamicsEstimatorPolicyV25,
    *,
    training_points: int,
    selected_at: datetime | None = None,
) -> DynamicsSelectionReceiptV25:
    public_snapshot.assert_sealed()
    policy.assert_sealed()
    if training_points < 21 or training_points >= len(public_snapshot.times) - 4:
        raise ValueError("invalid V2.5 dynamics training/inner split")
    at = selected_at or datetime.now(timezone.utc)
    training = DynamicsDataSnapshotV24.seal(
        snapshot_id=f"{public_snapshot.snapshot_id}_training_v25",
        declared_state_names=public_snapshot.declared_state_names,
        observed_state_names=public_snapshot.observed_state_names,
        times=public_snapshot.times[:training_points],
        values=public_snapshot.values[:training_points],
    )
    scores: list[DynamicsCandidateInnerScoreV25] = []
    successful: list[
        tuple[float, DynamicsFitResultV25, DynamicsCandidateDefinitionV25]
    ] = []
    simulation_count = 0
    for definition in policy.candidates:
        fit = fit_dynamics_candidate_v25(
            training, definition, policy, fitted_at=at
        )
        assert fit.fit_hash is not None
        if fit.status != "fit_succeeded":
            scores.append(
                DynamicsCandidateInnerScoreV25(
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
            predicted = simulate_dynamics_model_v25(
                fit.model,
                public_snapshot.values[training_points - 1],
                validation_times,
            )[1:]
            nrmse = trajectory_nrmse(
                public_snapshot.values[training_points:], predicted
            )
        except (RuntimeError, ValueError, FloatingPointError):
            scores.append(
                DynamicsCandidateInnerScoreV25(
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
            DynamicsCandidateInnerScoreV25(
                candidate_id=definition.candidate_id,
                fit_hash=fit.fit_hash,
                status="scored",
                trajectory_nrmse=nrmse,
                selection_score=score,
                active_coefficient_count=fit.model.active_coefficient_count,
                reason_codes=[],
            )
        )
        successful.append((nrmse, fit, definition))
    assert public_snapshot.snapshot_hash is not None
    assert policy.policy_hash is not None
    common = {
        "receipt_id": f"{public_snapshot.snapshot_id}_{policy.estimator_arm}_selection_v25",
        "estimator_arm": policy.estimator_arm,
        "policy_hash": policy.policy_hash,
        "public_data_hash": public_snapshot.snapshot_hash,
        "training_points": training_points,
        "inner_validation_points": len(public_snapshot.times) - training_points,
        "scores": scores,
        "inner_simulation_count": simulation_count,
        "selected_at": at,
    }
    if not successful:
        return DynamicsSelectionReceiptV25.seal(
            **common,
            status="needs_evidence",
        )
    dense = [item for item in successful if item[2].regression_kind == "ridge"]
    sparse = [item for item in successful if item[2].regression_kind == "stlsq"]
    if dense and sparse:
        best_dense = min(dense, key=lambda item: (item[0], item[1].candidate_id))
        best_sparse = min(sparse, key=lambda item: (item[0], item[1].candidate_id))
        selected = (
            best_sparse
            if best_sparse[0]
            <= best_dense[0] * (1.0 - policy.safety_improvement_margin)
            else best_dense
        )
    else:
        selected = min(successful, key=lambda item: (item[0], item[1].candidate_id))
    fit = selected[1]
    assert fit.model is not None and fit.model.model_hash is not None
    return DynamicsSelectionReceiptV25.seal(
        **common,
        status="selected",
        selected_candidate_id=fit.candidate_id,
        selected_model_hash=fit.model.model_hash,
    )


def simulate_dynamics_model_v25(
    model: DynamicsModelIRV25,
    initial_state: list[float],
    times: list[float],
) -> list[list[float]]:
    model.assert_sealed()
    if len(initial_state) != len(model.state_names):
        raise ValueError("initial state does not match V2.5 dynamics model")
    if len(times) < 2 or any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("simulation times must be strictly increasing")
    coefficients = np.asarray(model.coefficient_matrix, dtype=float)

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        row = evaluate_polynomial_library(state.reshape(1, -1), model.basis_terms)[0]
        derivative = coefficients @ row
        if not np.isfinite(derivative).all() or np.max(np.abs(state)) > 1e6:
            raise FloatingPointError("V2.5 dynamics simulation diverged")
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
        raise RuntimeError("V2.5 dynamics simulation failed") from exc
    if not solution.success or solution.y.shape[1] != len(times):
        raise RuntimeError("V2.5 dynamics integrator did not cover the requested horizon")
    values = solution.y.T
    if not np.isfinite(values).all() or np.max(np.abs(values)) > 1e6:
        raise RuntimeError("V2.5 dynamics simulation produced divergent states")
    return values.tolist()


def fit_coefficients_v25(
    design: np.ndarray,
    targets: np.ndarray,
    definition: DynamicsCandidateDefinitionV25,
) -> np.ndarray:
    scales = np.sqrt(np.mean(design**2, axis=0))
    scales[scales < 1e-12] = 1.0
    normalized = design / scales
    gram = normalized.T @ normalized
    coefficients_normalized = np.linalg.solve(
        gram + definition.ridge_alpha * np.eye(normalized.shape[1]),
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
                selected_coefficients = np.linalg.solve(
                    selected.T @ selected
                    + definition.ridge_alpha * np.eye(selected.shape[1]),
                    selected.T @ targets[:, equation],
                )
                coefficients[equation, active] = selected_coefficients / scales[active]
            active = np.abs(coefficients[equation]) >= definition.sparsity_threshold
            if np.array_equal(active, previous):
                break
        coefficients[equation, ~active] = 0.0
    return coefficients


def _design_limitations(estimator_arm: EstimatorArm) -> list[str]:
    limitations = [
        "rank is conditional on the frozen candidate library and observed trajectory",
        "this is neither symbolic structural-identifiability nor observability proof",
        "serial dependence and effective independent information are not established",
    ]
    if estimator_arm == "window_integral_matching":
        limitations.append(
            "overlapping integral windows share observations and are not independent samples"
        )
    return limitations
