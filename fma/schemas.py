from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .hashing import sha256_value


Identifier = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")]
FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClauseKind(str, Enum):
    OBJECTIVE = "objective"
    HARD_CONSTRAINT = "hard_constraint"
    ASSUMPTION = "assumption"


class ContractClause(StrictModel):
    clause_id: Identifier
    kind: ClauseKind
    statement: Annotated[str, Field(min_length=3)]
    unit: Annotated[str, Field(min_length=1)] = "unitless"
    source_ref: Annotated[str, Field(min_length=1)]
    acceptance_criterion: Annotated[str, Field(min_length=3)]


class ContractAcceptanceTest(StrictModel):
    test_id: Identifier
    kind: Literal["known_optimum", "assignment_case"]
    assignment: dict[Identifier, FiniteNumber] = Field(default_factory=dict)
    expected_feasible: bool | None = None
    expected_objective: FiniteNumber | None = None
    tolerance: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1e-7
    source_ref: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def validate_test(self) -> "ContractAcceptanceTest":
        if self.kind == "known_optimum" and self.expected_objective is None:
            raise ValueError("known_optimum requires expected_objective")
        if self.kind == "assignment_case":
            if not self.assignment or self.expected_feasible is None:
                raise ValueError(
                    "assignment_case requires an assignment and expected_feasible"
                )
        return self


class ContractFact(StrictModel):
    """A public, source-backed numeric fact available to model generators."""

    fact_id: Identifier
    statement: Annotated[str, Field(min_length=3)]
    value: FiniteNumber
    unit: Annotated[str, Field(min_length=1)] = "unitless"
    source_ref: Annotated[str, Field(min_length=1)]


class ContractDecision(StrictModel):
    decision_id: Identifier
    statement: Annotated[str, Field(min_length=3)]
    kind: Literal["continuous", "integer", "binary"]
    unit: Annotated[str, Field(min_length=1)] = "unitless"
    lower_bound: FiniteNumber | None = None
    upper_bound: FiniteNumber | None = None
    source_ref: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def validate_declared_bounds(self) -> "ContractDecision":
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError(f"invalid declared bounds for {self.decision_id}")
        return self


class ProblemContract(StrictModel):
    schema_version: Literal["1.0", "1.1"] = "1.1"
    contract_id: Identifier
    version: Annotated[int, Field(ge=1)] = 1
    question: Annotated[str, Field(min_length=5)]
    system_boundary: Annotated[str, Field(min_length=3)]
    decision_horizon: Annotated[str, Field(min_length=3)]
    decisions: list[ContractDecision] = Field(default_factory=list)
    clauses: Annotated[list[ContractClause], Field(min_length=1)]
    public_facts: list[ContractFact] = Field(default_factory=list)
    acceptance_tests: Annotated[list[ContractAcceptanceTest], Field(min_length=1)]
    permitted_actions: list[str] = Field(default_factory=lambda: ["local_compute"])
    forbidden_actions: list[str] = Field(default_factory=lambda: ["external_action"])
    risk_level: Literal["A0", "A1", "A2", "A3", "A4"] = "A1"
    frozen_at: datetime
    frozen_hash: str | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "ProblemContract":
        ids = [clause.clause_id for clause in self.clauses]
        if len(ids) != len(set(ids)):
            raise ValueError("contract clause_id values must be unique")
        decision_ids = [decision.decision_id for decision in self.decisions]
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("contract decision_id values must be unique")
        fact_ids = [fact.fact_id for fact in self.public_facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("contract fact_id values must be unique")
        test_ids = [test.test_id for test in self.acceptance_tests]
        if len(test_ids) != len(set(test_ids)):
            raise ValueError("contract acceptance test_id values must be unique")
        kinds = {clause.kind for clause in self.clauses}
        if ClauseKind.OBJECTIVE not in kinds:
            raise ValueError("contract needs at least one objective clause")
        if ClauseKind.HARD_CONSTRAINT not in kinds:
            raise ValueError("contract needs at least one hard-constraint clause")
        if self.frozen_hash is not None and self.frozen_hash != self.content_hash():
            raise ValueError("frozen_hash does not match contract content")
        return self

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"frozen_hash"})
        # v1.0 artifacts created before public_facts existed omitted the field.
        # Preserve their sealed hash while including non-empty facts in every
        # new contract hash.
        if self.schema_version == "1.0":
            for backward_compatible_field in ("decisions", "public_facts"):
                if not payload[backward_compatible_field]:
                    payload.pop(backward_compatible_field)
        return sha256_value(payload)

    def assert_frozen(self) -> None:
        if not self.frozen_hash:
            raise ValueError("problem contract must be frozen before execution")
        if self.frozen_hash != self.content_hash():
            raise ValueError("problem contract changed after it was frozen")

    @classmethod
    def freeze(cls, **data: object) -> "ProblemContract":
        data.setdefault("frozen_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(**draft.model_dump(exclude={"frozen_hash"}), frozen_hash=draft.content_hash())


class VariableKind(str, Enum):
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    BINARY = "binary"


class VariableSpec(StrictModel):
    name: Identifier
    kind: VariableKind
    lower_bound: FiniteNumber
    upper_bound: FiniteNumber
    unit: Annotated[str, Field(min_length=1)] = "unitless"

    @model_validator(mode="after")
    def validate_bounds(self) -> "VariableSpec":
        if self.lower_bound > self.upper_bound:
            raise ValueError(f"invalid bounds for {self.name}")
        if self.kind == VariableKind.BINARY and (
            self.lower_bound != 0 or self.upper_bound != 1
        ):
            raise ValueError("binary variables must use bounds [0, 1]")
        return self


class ObjectiveSpec(StrictModel):
    sense: Literal["minimize", "maximize"]
    coefficients: dict[Identifier, FiniteNumber]
    constant: FiniteNumber = 0.0
    unit: Annotated[str, Field(min_length=1)] = "unitless"
    contract_clause_ids: Annotated[list[Identifier], Field(min_length=1)]


class ConstraintSpec(StrictModel):
    constraint_id: Identifier
    coefficients: dict[Identifier, FiniteNumber]
    sense: Literal["<=", ">=", "=="]
    rhs: FiniteNumber
    lhs_unit: Annotated[str, Field(min_length=1)] = "unitless"
    rhs_unit: Annotated[str, Field(min_length=1)] = "unitless"
    contract_clause_ids: Annotated[list[Identifier], Field(min_length=1)]


class CandidateLineage(StrictModel):
    root_kind: Literal["retrieved", "fresh", "human", "fixture"] = "fresh"
    parent_candidate_ids: list[Identifier] = Field(default_factory=list)
    evolution_operator: str | None = None
    rationale: Annotated[str, Field(min_length=3)]


class OptimizationModelIR(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    candidate_id: Identifier
    contract_hash: Annotated[str, Field(min_length=64, max_length=64)]
    skeleton_id: Identifier
    lineage: CandidateLineage
    variables: Annotated[list[VariableSpec], Field(min_length=1)]
    objective: ObjectiveSpec
    constraints: Annotated[list[ConstraintSpec], Field(min_length=1)]
    validation_obligations: Annotated[list[str], Field(min_length=1)]
    compiler_target: Literal["scipy_milp_v1"] = "scipy_milp_v1"
    ir_hash: str | None = None

    @model_validator(mode="after")
    def validate_ir(self) -> "OptimizationModelIR":
        variable_names = [variable.name for variable in self.variables]
        if len(variable_names) != len(set(variable_names)):
            raise ValueError("variable names must be unique")
        constraint_ids = [constraint.constraint_id for constraint in self.constraints]
        if len(constraint_ids) != len(set(constraint_ids)):
            raise ValueError("constraint_id values must be unique")
        known = set(variable_names)
        referenced = set(self.objective.coefficients)
        for constraint in self.constraints:
            referenced.update(constraint.coefficients)
        unknown = referenced - known
        if unknown:
            raise ValueError(f"coefficients reference unknown variables: {sorted(unknown)}")
        if self.ir_hash is not None and self.ir_hash != self.content_hash():
            raise ValueError("ir_hash does not match model content")
        return self

    def content_hash(self) -> str:
        return sha256_value(self.model_dump(mode="json", exclude={"ir_hash"}))

    def assert_sealed(self) -> None:
        if not self.ir_hash:
            raise ValueError("model IR must be sealed before compilation")
        if self.ir_hash != self.content_hash():
            raise ValueError("model IR changed after it was sealed")

    @classmethod
    def seal(cls, **data: object) -> "OptimizationModelIR":
        draft = cls(**data)
        return cls(**draft.model_dump(exclude={"ir_hash"}), ir_hash=draft.content_hash())


class CompilerCertificate(StrictModel):
    compiler_version: Literal["scipy_milp_v1"] = "scipy_milp_v1"
    source_ir_hash: str
    variable_order: list[str]
    objective_vector: list[float]
    integrality: list[int]
    variable_bounds: list[tuple[float, float]]
    constraint_rows: list[dict[str, object]]
    matrix_hash: str
    execution_array_hash: str


class SolutionArtifact(StrictModel):
    solver: str
    solver_status: Literal["optimal", "infeasible", "unbounded", "failed"]
    source_ir_hash: str
    compiled_matrix_hash: str
    execution_array_hash: str
    values: dict[str, FiniteNumber]
    objective_value: FiniteNumber | None = None
    message: str


CheckStatus = Literal["pass", "fail", "warning", "not_run"]


class CheckRecord(StrictModel):
    status: CheckStatus
    detail: str


class ValidationVector(StrictModel):
    verifier_version: Literal["optimization_verifier_v1"] = "optimization_verifier_v1"
    candidate_id: str
    checks: dict[str, CheckRecord]
    violations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)

    def hard_gates_pass(self, required_checks: set[str]) -> bool:
        return all(
            name in self.checks and self.checks[name].status == "pass"
            for name in required_checks
        )


class ReproductionReport(StrictModel):
    replay_version: Literal["deterministic_replay_v1"] = "deterministic_replay_v1"
    status: Literal["pass", "fail"]
    source_ir_hash: str
    replay_matrix_hash: str | None = None
    replay_execution_hash: str | None = None
    reference_objective: float | None = None
    replay_objective: float | None = None
    value_delta_max: float | None = None
    checks: dict[str, bool]
    detail: str


class ArtifactRef(StrictModel):
    kind: str
    sha256: str
    relative_path: str


class PromotionDecision(StrictModel):
    policy_version: Literal["optimization_promotion_v1"] = "optimization_promotion_v1"
    candidate_id: str
    claim_node_id: str
    status: Literal["validated", "run_invalid", "needs_evidence"]
    validation_scope: Literal["synthetic_oracle"]
    gate_results: dict[str, bool]
    reasons: list[str]
    evidence_snapshot_hash: str
    evidence_snapshot_artifact: ArtifactRef


class CandidateRunOutcome(StrictModel):
    run_id: str
    candidate_id: str
    run_directory: str
    claim_node_id: str
    claim_status: str
    decision: PromotionDecision
    solution: SolutionArtifact
    validation: ValidationVector
    reproduction: ReproductionReport
    evidence_node_ids: dict[str, str]
    dossier_path: str
