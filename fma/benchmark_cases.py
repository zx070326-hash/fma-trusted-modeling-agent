from __future__ import annotations

import itertools
import math
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import Field, model_validator

from .hashing import sha256_value
from .schemas import Identifier, OptimizationModelIR, ProblemContract, StrictModel


SUITE_VERSION = "fma_bench_v0"
FROZEN_AT = datetime(2026, 7, 20, tzinfo=timezone.utc)
Family = Literal[
    "resource_allocation",
    "knapsack",
    "assignment",
    "transportation",
    "facility_location",
    "set_covering",
]
TaskKind = Literal["build", "revise", "explain", "no_result"]
ExpectedStatus = Literal["validated", "no_result", "needs_evidence"]
MechanicalScope = Literal["full", "final_model", "ir_only", "status_only"]


class BenchmarkCase(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    suite_version: Literal[SUITE_VERSION] = SUITE_VERSION
    case_id: Identifier
    family: Family
    task_kind: TaskKind
    mechanical_scope: MechanicalScope
    expected_status: ExpectedStatus
    expected_skeleton_id: Identifier | None = None
    expected_evolution_operator: str | None = None
    revision_of: Identifier | None = None
    reason_class: str | None = None
    oracle_assignment_count: Annotated[int | None, Field(ge=1)] = None
    privacy_canary: Annotated[str, Field(min_length=12)]
    contract: ProblemContract
    reference_ir: OptimizationModelIR | None = None
    sealed_hash: str | None = None

    @model_validator(mode="after")
    def validate_case(self) -> "BenchmarkCase":
        self.contract.assert_frozen()
        if self.expected_status == "validated":
            if self.reference_ir is None:
                raise ValueError("validated benchmark cases require a reference IR")
            self.reference_ir.assert_sealed()
            if self.reference_ir.contract_hash != self.contract.frozen_hash:
                raise ValueError("reference IR is not bound to the benchmark contract")
            if self.oracle_assignment_count is None:
                raise ValueError("validated benchmark cases require an oracle size")
        elif self.reference_ir is not None:
            raise ValueError("non-validated benchmark cases cannot carry a reference IR")
        if self.task_kind == "no_result" and self.expected_status != "no_result":
            raise ValueError("no_result tasks must expect no_result")
        if self.expected_status == "no_result" and not self.reason_class:
            raise ValueError("no_result cases require a private reason class")
        if self.sealed_hash is not None and self.sealed_hash != self.content_hash():
            raise ValueError("benchmark case sealed_hash mismatch")
        return self

    def content_hash(self) -> str:
        return sha256_value(self.model_dump(mode="json", exclude={"sealed_hash"}))

    @classmethod
    def seal(cls, **data: object) -> "BenchmarkCase":
        draft = cls(**data)
        return cls(**draft.model_dump(exclude={"sealed_hash"}), sealed_hash=draft.content_hash())

    def public_hash(self) -> str:
        from .codex_driver import ExplorerProblemView

        return sha256_value(ExplorerProblemView.from_contract(self.contract))

    def expected_hash(self) -> str:
        return sha256_value(
            {
                "case_id": self.case_id,
                "expected_status": self.expected_status,
                "reference_ir_hash": self.reference_ir.ir_hash if self.reference_ir else None,
                "reason_class": self.reason_class,
                "expected_skeleton_id": self.expected_skeleton_id,
                "expected_evolution_operator": self.expected_evolution_operator,
            }
        )


class BenchmarkSuite(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    suite_version: Literal[SUITE_VERSION] = SUITE_VERSION
    cases: Annotated[list[BenchmarkCase], Field(min_length=1)]
    suite_hash: str | None = None

    @model_validator(mode="after")
    def validate_suite(self) -> "BenchmarkSuite":
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("benchmark case IDs must be unique")
        hashes = [case.sealed_hash for case in self.cases]
        if any(value is None for value in hashes) or len(hashes) != len(set(hashes)):
            raise ValueError("benchmark case hashes must be present and unique")
        if self.suite_hash is not None and self.suite_hash != self.content_hash():
            raise ValueError("benchmark suite hash mismatch")
        return self

    def content_hash(self) -> str:
        return sha256_value(
            {
                "schema_version": self.schema_version,
                "suite_version": self.suite_version,
                "cases": [
                    {"case_id": case.case_id, "sealed_hash": case.sealed_hash}
                    for case in sorted(self.cases, key=lambda item: item.case_id)
                ],
            }
        )

    @classmethod
    def seal(cls, cases: list[BenchmarkCase]) -> "BenchmarkSuite":
        ordered = sorted(cases, key=lambda case: case.case_id)
        draft = cls(cases=ordered)
        return cls(cases=ordered, suite_hash=draft.content_hash())

    def public_manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "suite_version": self.suite_version,
            "suite_commitment": self.suite_hash,
            "case_count": len(self.cases),
            "cases": [
                {
                    "case_id": case.case_id,
                    "family": case.family,
                    "task_kind": case.task_kind,
                    "mechanical_scope": case.mechanical_scope,
                    "public_hash": case.public_hash(),
                }
                for case in self.cases
            ],
        }

    def sealed_commitment(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "suite_version": self.suite_version,
            "suite_hash": self.suite_hash,
            "cases": [
                {
                    "case_id": case.case_id,
                    "public_hash": case.public_hash(),
                    "expected_hash": case.expected_hash(),
                    "sealed_hash": case.sealed_hash,
                }
                for case in self.cases
            ],
        }


def _variable(
    name: str,
    *,
    kind: str = "binary",
    lower: float = 0,
    upper: float = 1,
    unit: str = "decision_unit",
) -> dict[str, object]:
    return {
        "name": name,
        "kind": kind,
        "lower_bound": lower,
        "upper_bound": upper,
        "unit": unit,
    }


def _constraint(
    clause_id: str,
    coefficients: dict[str, float],
    sense: str,
    rhs: float,
    statement: str,
    *,
    unit: str = "count_unit",
) -> dict[str, object]:
    return {
        "clause_id": clause_id,
        "constraint_id": clause_id,
        "coefficients": coefficients,
        "sense": sense,
        "rhs": rhs,
        "statement": statement,
        "unit": unit,
    }


def _linear_expression(coefficients: dict[str, float]) -> str:
    parts = [f"{value:g}*{name}" for name, value in coefficients.items() if value != 0]
    return " + ".join(parts) if parts else "0"


def _feasible(constraints: list[dict[str, object]], assignment: dict[str, float]) -> bool:
    for constraint in constraints:
        coefficients = constraint["coefficients"]
        assert isinstance(coefficients, dict)
        lhs = sum(float(value) * assignment[name] for name, value in coefficients.items())
        rhs = float(constraint["rhs"])
        sense = str(constraint["sense"])
        if sense == "<=" and lhs > rhs + 1e-9:
            return False
        if sense == ">=" and lhs < rhs - 1e-9:
            return False
        if sense == "==" and abs(lhs - rhs) > 1e-9:
            return False
    return True


def _objective(
    coefficients: dict[str, float], constant: float, assignment: dict[str, float]
) -> float:
    return constant + sum(value * assignment[name] for name, value in coefficients.items())


def _enumerate_optimum(
    variables: list[dict[str, object]],
    coefficients: dict[str, float],
    constant: float,
    sense: str,
    constraints: list[dict[str, object]],
) -> tuple[float | None, int]:
    domains: list[range] = []
    for variable in variables:
        if variable["kind"] not in {"integer", "binary"}:
            raise ValueError("the exact benchmark oracle accepts integer domains only")
        domains.append(
            range(math.ceil(float(variable["lower_bound"])), math.floor(float(variable["upper_bound"])) + 1)
        )
    assignment_count = math.prod(len(domain) for domain in domains)
    if assignment_count > 100_000:
        raise ValueError("positive benchmark case exceeds the exact oracle budget")
    names = [str(variable["name"]) for variable in variables]
    best: float | None = None
    for values in itertools.product(*domains):
        assignment = dict(zip(names, map(float, values), strict=True))
        if not _feasible(constraints, assignment):
            continue
        value = _objective(coefficients, constant, assignment)
        if best is None or (sense == "maximize" and value > best) or (sense == "minimize" and value < best):
            best = value
    return best, assignment_count


def _public_facts(
    case_id: str,
    variables: list[dict[str, object]],
    objective_coefficients: dict[str, float],
    objective_unit: str,
    constraints: list[dict[str, object]],
    canary: str,
) -> list[dict[str, object]]:
    facts: list[dict[str, object]] = []
    units = {str(variable["name"]): str(variable["unit"]) for variable in variables}
    for name, value in objective_coefficients.items():
        facts.append(
            {
                "fact_id": f"{case_id}_objective_{name}",
                "statement": f"The linear objective coefficient of {name} is {value:g}",
                "value": value,
                "unit": f"{objective_unit}_per_{units[name]}",
                "source_ref": f"sealed:{canary}:objective",
            }
        )
    for index, constraint in enumerate(constraints, start=1):
        unit = str(constraint["unit"])
        coefficients = constraint["coefficients"]
        assert isinstance(coefficients, dict)
        for name, value in coefficients.items():
            facts.append(
                {
                    "fact_id": f"{case_id}_c{index}_{name}",
                    "statement": f"In requirement {constraint['clause_id']}, the coefficient of {name} is {float(value):g}",
                    "value": value,
                    "unit": f"{unit}_per_{units[name]}",
                    "source_ref": f"sealed:{canary}:constraint",
                }
            )
        facts.append(
            {
                "fact_id": f"{case_id}_c{index}_rhs",
                "statement": f"The right-hand side of requirement {constraint['clause_id']} is {float(constraint['rhs']):g}",
                "value": constraint["rhs"],
                "unit": unit,
                "source_ref": f"sealed:{canary}:rhs",
            }
        )
    return facts


def _acceptance_tests(
    case_id: str,
    variables: list[dict[str, object]],
    objective_coefficients: dict[str, float],
    objective_constant: float,
    constraints: list[dict[str, object]],
    expected_objective: float,
    optimal_assignment: dict[str, float],
    canary: str,
) -> list[dict[str, object]]:
    names = [str(variable["name"]) for variable in variables]
    if set(optimal_assignment) != set(names):
        raise ValueError(f"{case_id}: optimal assignment keys are incomplete")
    if not _feasible(constraints, optimal_assignment):
        raise ValueError(f"{case_id}: registered optimal assignment is infeasible")
    observed = _objective(objective_coefficients, objective_constant, optimal_assignment)
    if abs(observed - expected_objective) > 1e-9:
        raise ValueError(f"{case_id}: registered objective does not match the optimal assignment")

    probes: list[tuple[str, dict[str, float]]] = [
        ("zero", {name: 0.0 for name in names}),
        ("upper", {str(v["name"]): float(v["upper_bound"]) for v in variables}),
    ]
    for variable in variables:
        assignment = {name: 0.0 for name in names}
        assignment[str(variable["name"])] = min(1.0, float(variable["upper_bound"]))
        probes.append((f"basis_{variable['name']}", assignment))

    tests: list[dict[str, object]] = [
        {
            "test_id": f"{case_id}_known_optimum",
            "kind": "known_optimum",
            "expected_objective": expected_objective,
            "source_ref": f"sealed:{canary}:independent_optimum",
        },
        {
            "test_id": f"{case_id}_optimal_assignment",
            "kind": "assignment_case",
            "assignment": optimal_assignment,
            "expected_feasible": True,
            "expected_objective": expected_objective,
            "source_ref": f"sealed:{canary}:optimal_assignment",
        },
    ]
    for label, assignment in probes:
        tests.append(
            {
                "test_id": f"{case_id}_{label}",
                "kind": "assignment_case",
                "assignment": assignment,
                "expected_feasible": _feasible(constraints, assignment),
                "expected_objective": _objective(
                    objective_coefficients, objective_constant, assignment
                ),
                "source_ref": f"sealed:{canary}:probe",
            }
        )
    return tests


def _positive_case(
    *,
    case_id: str,
    family: Family,
    task_kind: Literal["build", "revise", "explain"],
    question: str,
    variables: list[dict[str, object]],
    objective_sense: str,
    objective_coefficients: dict[str, float],
    constraints: list[dict[str, object]],
    expected_objective: float,
    optimal_assignment: dict[str, float],
    skeleton_id: str,
    objective_unit: str = "objective_unit",
    objective_constant: float = 0,
    evolution_operator: str = "formulate",
    revision_of: str | None = None,
    contract_version: int = 1,
) -> BenchmarkCase:
    canary = f"sealed_canary_{case_id}_v0"
    optimum, assignment_count = _enumerate_optimum(
        variables,
        objective_coefficients,
        objective_constant,
        objective_sense,
        constraints,
    )
    if optimum is None or abs(optimum - expected_objective) > 1e-9:
        raise ValueError(
            f"{case_id}: manual optimum {expected_objective} disagrees with enumeration {optimum}"
        )
    decisions = [
        {
            "decision_id": variable["name"],
            "statement": f"Decision variable {variable['name']} for the stated system",
            "kind": variable["kind"],
            "unit": variable["unit"],
            "lower_bound": variable["lower_bound"],
            "upper_bound": variable["upper_bound"],
            "source_ref": f"sealed:{canary}:decision",
        }
        for variable in variables
    ]
    clauses: list[dict[str, object]] = [
        {
            "clause_id": "primary_objective",
            "kind": "objective",
            "statement": f"{objective_sense.title()} the registered linear objective",
            "unit": objective_unit,
            "source_ref": f"sealed:{canary}:objective_clause",
            "acceptance_criterion": (
                f"Use the linear objective {_linear_expression(objective_coefficients)} "
                f"with constant {objective_constant:g}"
            ),
        }
    ]
    for constraint in constraints:
        clauses.append(
            {
                "clause_id": constraint["clause_id"],
                "kind": "hard_constraint",
                "statement": constraint["statement"],
                "unit": constraint["unit"],
                "source_ref": f"sealed:{canary}:hard_clause",
                "acceptance_criterion": (
                    f"Require {_linear_expression(constraint['coefficients'])} "
                    f"{constraint['sense']} {float(constraint['rhs']):g}"
                ),
            }
        )
    contract = ProblemContract.freeze(
        schema_version="1.1",
        contract_id=f"bench_{family}",
        version=contract_version,
        question=question,
        system_boundary=f"Synthetic finite {family.replace('_', ' ')} benchmark",
        decision_horizon="One finite planning instance",
        decisions=decisions,
        clauses=clauses,
        public_facts=_public_facts(
            case_id,
            variables,
            objective_coefficients,
            objective_unit,
            constraints,
            canary,
        ),
        acceptance_tests=_acceptance_tests(
            case_id,
            variables,
            objective_coefficients,
            objective_constant,
            constraints,
            expected_objective,
            optimal_assignment,
            canary,
        ),
        permitted_actions=[
            "codex_cli_inference",
            "local_compute",
            "write_local_run_artifacts",
        ],
        forbidden_actions=["model_generated_network_access", "external_action"],
        risk_level="A1",
        frozen_at=FROZEN_AT,
    )
    ir = OptimizationModelIR.seal(
        candidate_id=f"{case_id}_reference",
        contract_hash=contract.frozen_hash,
        skeleton_id=skeleton_id,
        lineage={
            "root_kind": "fixture",
            "parent_candidate_ids": [],
            "evolution_operator": evolution_operator,
            "rationale": "Sealed benchmark reference used only by the evaluator",
        },
        variables=variables,
        objective={
            "sense": objective_sense,
            "coefficients": objective_coefficients,
            "constant": objective_constant,
            "unit": objective_unit,
            "contract_clause_ids": ["primary_objective"],
        },
        constraints=[
            {
                "constraint_id": constraint["constraint_id"],
                "coefficients": constraint["coefficients"],
                "sense": constraint["sense"],
                "rhs": constraint["rhs"],
                "lhs_unit": constraint["unit"],
                "rhs_unit": constraint["unit"],
                "contract_clause_ids": [constraint["clause_id"]],
            }
            for constraint in constraints
        ],
        validation_obligations=[
            "Recompute the full finite feasible set",
            "Compare the objective on every bounded assignment",
            "Replay the sealed candidate in a fresh process",
        ],
    )
    return BenchmarkCase.seal(
        case_id=case_id,
        family=family,
        task_kind=task_kind,
        mechanical_scope={"build": "full", "revise": "final_model", "explain": "ir_only"}[
            task_kind
        ],
        expected_status="validated",
        expected_skeleton_id=skeleton_id,
        expected_evolution_operator=evolution_operator,
        revision_of=revision_of,
        oracle_assignment_count=assignment_count,
        privacy_canary=canary,
        contract=contract,
        reference_ir=ir,
    )


def _no_result_case(
    *,
    case_id: str,
    family: Family,
    question: str,
    variables: list[dict[str, object]],
    clauses: list[dict[str, object]],
    public_facts: list[dict[str, object]],
    acceptance_tests: list[dict[str, object]],
    reason_class: str,
    skeleton_id: str,
) -> BenchmarkCase:
    canary = f"sealed_canary_{case_id}_v0"
    decisions = [
        {
            "decision_id": variable["name"],
            "statement": f"Decision variable {variable['name']} for the stated system",
            "kind": variable["kind"],
            "unit": variable["unit"],
            "lower_bound": variable.get("lower_bound"),
            "upper_bound": variable.get("upper_bound"),
            "source_ref": f"sealed:{canary}:decision",
        }
        for variable in variables
    ]
    private_tests = []
    for test in acceptance_tests:
        private_tests.append({**test, "source_ref": f"sealed:{canary}:private_test"})
    sourced_facts = []
    for fact in public_facts:
        sourced_facts.append({**fact, "source_ref": f"sealed:{canary}:public_fact"})
    sourced_clauses = []
    for clause in clauses:
        sourced_clauses.append({**clause, "source_ref": f"sealed:{canary}:clause"})
    contract = ProblemContract.freeze(
        schema_version="1.1",
        contract_id=f"bench_{family}",
        version=99,
        question=question,
        system_boundary=f"Synthetic {family.replace('_', ' ')} abstention benchmark",
        decision_horizon="One finite planning instance",
        decisions=decisions,
        clauses=sourced_clauses,
        public_facts=sourced_facts,
        acceptance_tests=private_tests,
        permitted_actions=[
            "codex_cli_inference",
            "local_compute",
            "write_local_run_artifacts",
        ],
        forbidden_actions=["model_generated_network_access", "external_action"],
        risk_level="A1",
        frozen_at=FROZEN_AT,
    )
    return BenchmarkCase.seal(
        case_id=case_id,
        family=family,
        task_kind="no_result",
        mechanical_scope="status_only",
        expected_status="no_result",
        expected_skeleton_id=skeleton_id,
        reason_class=reason_class,
        privacy_canary=canary,
        contract=contract,
    )


def _binary_names(prefix: str, count: int, unit: str) -> list[dict[str, object]]:
    return [_variable(f"{prefix}{index}", unit=unit) for index in range(1, count + 1)]


def build_fma_bench_v0() -> BenchmarkSuite:
    cases: list[BenchmarkCase] = []

    ra_variables = [
        _variable("x", kind="integer", upper=10, unit="product_unit"),
        _variable("y", kind="integer", upper=10, unit="product_unit"),
    ]
    ra_base_constraints = [
        _constraint("resource_a", {"x": 2, "y": 1}, "<=", 8, "Resource A use is capped at eight", unit="resource_a_unit"),
        _constraint("resource_b", {"x": 1, "y": 3}, "<=", 9, "Resource B use is capped at nine", unit="resource_b_unit"),
    ]
    cases.extend(
        [
            _positive_case(
                case_id="ra_b1", family="resource_allocation", task_kind="build",
                question="Choose integer production quantities x and y to maximize benefit under two resource limits.",
                variables=ra_variables, objective_sense="maximize", objective_coefficients={"x": 3, "y": 5},
                constraints=ra_base_constraints, expected_objective=19, optimal_assignment={"x": 3, "y": 2},
                skeleton_id="resource_allocation", evolution_operator="formulate",
            ),
            _positive_case(
                case_id="ra_r1", family="resource_allocation", task_kind="revise",
                question="Revise the resource-allocation model because Resource B capacity is tightened from nine to six.",
                variables=ra_variables, objective_sense="maximize", objective_coefficients={"x": 3, "y": 5},
                constraints=[ra_base_constraints[0], _constraint("resource_b", {"x": 1, "y": 3}, "<=", 6, "Resource B use is capped at six", unit="resource_b_unit")],
                expected_objective=14, optimal_assignment={"x": 3, "y": 1}, skeleton_id="resource_allocation",
                evolution_operator="tighten_capacity", revision_of="ra_b1", contract_version=2,
            ),
            _positive_case(
                case_id="ra_e1", family="resource_allocation", task_kind="explain",
                question="Formulate the model and explain in the rationale how the two resource limits determine the optimum.",
                variables=ra_variables, objective_sense="maximize", objective_coefficients={"x": 3, "y": 5},
                constraints=ra_base_constraints, expected_objective=19, optimal_assignment={"x": 3, "y": 2},
                skeleton_id="resource_allocation", evolution_operator="explain_binding_constraints",
            ),
        ]
    )
    cases.append(
        _no_result_case(
            case_id="ra_n1", family="resource_allocation",
            question="Maximize 3*x + 5*y + 4*x*y for bounded integer x and y under x+y<=3. The product term must be represented exactly without auxiliary decisions.",
            variables=[_variable("x", kind="integer", upper=2, unit="product_unit"), _variable("y", kind="integer", upper=2, unit="product_unit")],
            clauses=[
                {"clause_id": "primary_objective", "kind": "objective", "statement": "Maximize a nonlinear benefit containing x times y", "unit": "benefit_unit", "acceptance_criterion": "The objective is exactly 3*x + 5*y + 4*x*y"},
                {"clause_id": "capacity", "kind": "hard_constraint", "statement": "Total production cannot exceed three", "unit": "product_unit", "acceptance_criterion": "Require x+y<=3"},
            ],
            public_facts=[],
            acceptance_tests=[
                {"test_id": "ra_n1_p00", "kind": "assignment_case", "assignment": {"x": 0, "y": 0}, "expected_feasible": True, "expected_objective": 0},
                {"test_id": "ra_n1_p10", "kind": "assignment_case", "assignment": {"x": 1, "y": 0}, "expected_feasible": True, "expected_objective": 3},
                {"test_id": "ra_n1_p01", "kind": "assignment_case", "assignment": {"x": 0, "y": 1}, "expected_feasible": True, "expected_objective": 5},
                {"test_id": "ra_n1_p11", "kind": "assignment_case", "assignment": {"x": 1, "y": 1}, "expected_feasible": True, "expected_objective": 12},
            ],
            reason_class="nonlinear", skeleton_id="resource_allocation",
        )
    )

    kp_variables = _binary_names("i", 4, "item_choice")
    kp_capacity = _constraint("weight_capacity", {"i1": 2, "i2": 3, "i3": 4, "i4": 5}, "<=", 7, "Selected item weight cannot exceed seven", unit="weight_unit")
    cases.extend(
        [
            _positive_case(
                case_id="kp_b1", family="knapsack", task_kind="build",
                question="Select a subset of four indivisible items to maximize value without exceeding capacity seven.",
                variables=kp_variables, objective_sense="maximize", objective_coefficients={"i1": 3, "i2": 4, "i3": 5, "i4": 8},
                constraints=[kp_capacity], expected_objective=11, optimal_assignment={"i1": 1, "i2": 0, "i3": 0, "i4": 1},
                skeleton_id="knapsack", evolution_operator="formulate",
            ),
            _positive_case(
                case_id="kp_r1", family="knapsack", task_kind="revise",
                question="Revise the four-item knapsack: select at least two items, and items one and four are mutually exclusive.",
                variables=kp_variables, objective_sense="maximize", objective_coefficients={"i1": 3, "i2": 4, "i3": 5, "i4": 8},
                constraints=[kp_capacity, _constraint("minimum_items", {"i1": 1, "i2": 1, "i3": 1, "i4": 1}, ">=", 2, "At least two items must be selected", unit="item_count"), _constraint("mutual_exclusion", {"i1": 1, "i4": 1}, "<=", 1, "Items one and four cannot both be selected", unit="item_count")],
                expected_objective=9, optimal_assignment={"i1": 0, "i2": 1, "i3": 1, "i4": 0}, skeleton_id="knapsack",
                evolution_operator="add_selection_rules", revision_of="kp_b1", contract_version=2,
            ),
            _positive_case(
                case_id="kp_e1", family="knapsack", task_kind="explain",
                question="Add a fifth item of weight four and value four, formulate the knapsack, and explain its dominance relation in the rationale.",
                variables=_binary_names("i", 5, "item_choice"), objective_sense="maximize",
                objective_coefficients={"i1": 3, "i2": 4, "i3": 5, "i4": 8, "i5": 4},
                constraints=[_constraint("weight_capacity", {"i1": 2, "i2": 3, "i3": 4, "i4": 5, "i5": 4}, "<=", 7, "Selected item weight cannot exceed seven", unit="weight_unit")],
                expected_objective=11, optimal_assignment={"i1": 1, "i2": 0, "i3": 0, "i4": 1, "i5": 0},
                skeleton_id="knapsack", evolution_operator="add_dominated_item",
            ),
        ]
    )
    kp20 = _binary_names("i", 20, "item_choice")
    kp20_zero = {f"i{index}": 0 for index in range(1, 21)}
    cases.append(
        _no_result_case(
            case_id="kp_n1", family="knapsack",
            question="Formulate a 20-item binary knapsack with unit values and weights and capacity ten under the mandatory exact-enumeration budget.",
            variables=kp20,
            clauses=[
                {"clause_id": "primary_objective", "kind": "objective", "statement": "Maximize the number of selected items", "unit": "value_unit", "acceptance_criterion": "Use unit objective coefficients"},
                {"clause_id": "capacity", "kind": "hard_constraint", "statement": "At most ten items may be selected", "unit": "item_count", "acceptance_criterion": "The sum of all decisions is at most ten"},
            ],
            public_facts=[],
            acceptance_tests=[{"test_id": "kp_n1_zero", "kind": "assignment_case", "assignment": kp20_zero, "expected_feasible": True, "expected_objective": 0}],
            reason_class="oracle_budget", skeleton_id="knapsack",
        )
    )

    assignment_names = [f"x{worker}{job}" for worker in range(1, 4) for job in range(1, 4)]
    as_variables = [_variable(name, unit="assignment_choice") for name in assignment_names]
    as_costs = {"x11": 4, "x12": 1, "x13": 3, "x21": 2, "x22": 0, "x23": 5, "x31": 3, "x32": 2, "x33": 2}
    as_constraints = [
        *[_constraint(f"worker_{w}", {f"x{w}{j}": 1 for j in range(1, 4)}, "==", 1, f"Worker {w} receives exactly one job", unit="assignment_count") for w in range(1, 4)],
        *[_constraint(f"job_{j}", {f"x{w}{j}": 1 for w in range(1, 4)}, "==", 1, f"Job {j} is assigned exactly once", unit="assignment_count") for j in range(1, 4)],
    ]
    as_opt = {name: float(name in {"x12", "x21", "x33"}) for name in assignment_names}
    as_rev_opt = {name: float(name in {"x11", "x22", "x33"}) for name in assignment_names}
    cases.extend(
        [
            _positive_case(case_id="as_b1", family="assignment", task_kind="build", question="Assign three workers to three jobs one-to-one while minimizing the registered cost matrix.", variables=as_variables, objective_sense="minimize", objective_coefficients=as_costs, constraints=as_constraints, expected_objective=5, optimal_assignment=as_opt, skeleton_id="assignment", objective_unit="cost_unit", evolution_operator="formulate"),
            _positive_case(case_id="as_r1", family="assignment", task_kind="revise", question="Revise the three-by-three assignment because worker two is forbidden from job one.", variables=as_variables, objective_sense="minimize", objective_coefficients=as_costs, constraints=[*as_constraints, _constraint("forbid_x21", {"x21": 1}, "==", 0, "Worker two cannot perform job one", unit="assignment_count")], expected_objective=6, optimal_assignment=as_rev_opt, skeleton_id="assignment", objective_unit="cost_unit", evolution_operator="add_forbidden_arc", revision_of="as_b1", contract_version=2),
            _positive_case(case_id="as_e1", family="assignment", task_kind="explain", question="Formulate the three-by-three assignment and explain the distinct role of worker and job equalities in the rationale.", variables=as_variables, objective_sense="minimize", objective_coefficients=as_costs, constraints=as_constraints, expected_objective=5, optimal_assignment=as_opt, skeleton_id="assignment", objective_unit="cost_unit", evolution_operator="explain_bipartite_balance"),
        ]
    )
    as3_zero = {name: 0 for name in assignment_names}
    as3_permutations = (
        ("identity", {"x11", "x22", "x33"}, -3),
        ("swap_23", {"x11", "x23", "x32"}, 0),
        ("swap_12", {"x12", "x21", "x33"}, 0),
        ("cycle_123", {"x12", "x23", "x31"}, 0),
        ("cycle_132", {"x13", "x21", "x32"}, 0),
        ("reverse", {"x13", "x22", "x31"}, 0),
    )
    cases.append(
        _no_result_case(
            case_id="as_n1", family="assignment",
            question="Solve a three-by-three assignment whose objective is exactly the discount term -3*x11*x22 and must be represented on every feasible assignment without auxiliary decisions.",
            variables=as_variables,
            clauses=[
                {"clause_id": "primary_objective", "kind": "objective", "statement": "Minimize the joint-selection discount", "unit": "cost_unit", "acceptance_criterion": "Represent exactly -3*x11*x22 on every feasible assignment"},
                {"clause_id": "one_to_one", "kind": "hard_constraint", "statement": "Use a complete one-to-one assignment", "unit": "assignment_count", "acceptance_criterion": "Every row and column sum must equal one"},
            ], public_facts=[],
            acceptance_tests=[
                {"test_id": "as_n1_zero", "kind": "assignment_case", "assignment": as3_zero, "expected_feasible": False, "expected_objective": 0},
                *[
                    {
                        "test_id": f"as_n1_{label}",
                        "kind": "assignment_case",
                        "assignment": {
                            name: float(name in selected) for name in assignment_names
                        },
                        "expected_feasible": True,
                        "expected_objective": objective,
                    }
                    for label, selected, objective in as3_permutations
                ],
            ],
            reason_class="nonlinear", skeleton_id="assignment",
        )
    )

    tr_names = ["x_ac", "x_ad", "x_bc", "x_bd"]
    tr_variables = [_variable(name, kind="integer", upper=5, unit="shipment_unit") for name in tr_names]
    tr_costs = {"x_ac": 1, "x_ad": 4, "x_bc": 3, "x_bd": 1}
    tr_constraints = [
        _constraint("supply_a", {"x_ac": 1, "x_ad": 1}, "==", 4, "Origin A ships exactly four units", unit="shipment_unit"),
        _constraint("supply_b", {"x_bc": 1, "x_bd": 1}, "==", 3, "Origin B ships exactly three units", unit="shipment_unit"),
        _constraint("demand_c", {"x_ac": 1, "x_bc": 1}, "==", 2, "Destination C receives exactly two units", unit="shipment_unit"),
        _constraint("demand_d", {"x_ad": 1, "x_bd": 1}, "==", 5, "Destination D receives exactly five units", unit="shipment_unit"),
    ]
    cases.extend(
        [
            _positive_case(case_id="tr_b1", family="transportation", task_kind="build", question="Route integer shipments from two origins to two destinations to meet all balances at minimum cost.", variables=tr_variables, objective_sense="minimize", objective_coefficients=tr_costs, constraints=tr_constraints, expected_objective=13, optimal_assignment={"x_ac": 2, "x_ad": 2, "x_bc": 0, "x_bd": 3}, skeleton_id="transportation", objective_unit="cost_unit", evolution_operator="formulate"),
            _positive_case(case_id="tr_r1", family="transportation", task_kind="revise", question="Revise the balanced transport model because arc A-to-C can carry at most one unit.", variables=tr_variables, objective_sense="minimize", objective_coefficients=tr_costs, constraints=[*tr_constraints, _constraint("arc_ac_capacity", {"x_ac": 1}, "<=", 1, "Arc A to C carries at most one unit", unit="shipment_unit")], expected_objective=18, optimal_assignment={"x_ac": 1, "x_ad": 3, "x_bc": 1, "x_bd": 2}, skeleton_id="transportation", objective_unit="cost_unit", evolution_operator="add_arc_capacity", revision_of="tr_b1", contract_version=2),
            _positive_case(case_id="tr_e1", family="transportation", task_kind="explain", question="Formulate the balanced two-by-two transport model and explain why one balance equality is algebraically redundant.", variables=tr_variables, objective_sense="minimize", objective_coefficients=tr_costs, constraints=tr_constraints, expected_objective=13, optimal_assignment={"x_ac": 2, "x_ad": 2, "x_bc": 0, "x_bd": 3}, skeleton_id="transportation", objective_unit="cost_unit", evolution_operator="explain_balance_redundancy"),
        ]
    )
    cases.append(
        _no_result_case(
            case_id="tr_n1", family="transportation", question="Meet hard destination demands of three and five units using origins with hard supplies of four and three; shortages and external supply are forbidden.",
            variables=tr_variables,
            clauses=[
                {"clause_id": "primary_objective", "kind": "objective", "statement": "Minimize transport cost", "unit": "cost_unit", "acceptance_criterion": "Use the registered arc costs"},
                {"clause_id": "supply", "kind": "hard_constraint", "statement": "Total hard supply is seven units", "unit": "shipment_unit", "acceptance_criterion": "Ship no more than seven units"},
                {"clause_id": "demand", "kind": "hard_constraint", "statement": "Hard destination demand totals eight units", "unit": "shipment_unit", "acceptance_criterion": "Deliver exactly eight units"},
            ], public_facts=[],
            acceptance_tests=[{"test_id": "tr_n1_zero", "kind": "assignment_case", "assignment": {name: 0 for name in tr_names}, "expected_feasible": False}],
            reason_class="infeasible", skeleton_id="transportation",
        )
    )

    fl_names = ["y1", "y2", "x11", "x12", "x21", "x22"]
    fl_variables = [_variable(name, unit="binary_action") for name in fl_names]
    fl_costs = {"y1": 2, "y2": 2, "x11": 1, "x12": 6, "x21": 5, "x22": 1}
    fl_constraints = [
        _constraint("customer_1", {"x11": 1, "x12": 1}, "==", 1, "Customer one is assigned exactly once", unit="assignment_count"),
        _constraint("customer_2", {"x21": 1, "x22": 1}, "==", 1, "Customer two is assigned exactly once", unit="assignment_count"),
        _constraint("link_x11", {"x11": 1, "y1": -1}, "<=", 0, "Customer one uses facility one only if it opens", unit="binary_count"),
        _constraint("link_x21", {"x21": 1, "y1": -1}, "<=", 0, "Customer two uses facility one only if it opens", unit="binary_count"),
        _constraint("link_x12", {"x12": 1, "y2": -1}, "<=", 0, "Customer one uses facility two only if it opens", unit="binary_count"),
        _constraint("link_x22", {"x22": 1, "y2": -1}, "<=", 0, "Customer two uses facility two only if it opens", unit="binary_count"),
    ]
    fl_opt = {"y1": 1, "y2": 1, "x11": 1, "x12": 0, "x21": 0, "x22": 1}
    fl_rev = {"y1": 1, "y2": 0, "x11": 1, "x12": 0, "x21": 1, "x22": 0}
    cases.extend(
        [
            _positive_case(case_id="fl_b1", family="facility_location", task_kind="build", question="Open facilities and assign two customers exactly once while minimizing opening and assignment cost.", variables=fl_variables, objective_sense="minimize", objective_coefficients=fl_costs, constraints=fl_constraints, expected_objective=6, optimal_assignment=fl_opt, skeleton_id="facility_location", objective_unit="cost_unit", evolution_operator="formulate"),
            _positive_case(case_id="fl_r1", family="facility_location", task_kind="revise", question="Revise the facility model so that at most one facility may open.", variables=fl_variables, objective_sense="minimize", objective_coefficients=fl_costs, constraints=[*fl_constraints, _constraint("one_facility", {"y1": 1, "y2": 1}, "<=", 1, "At most one facility may open", unit="binary_count")], expected_objective=8, optimal_assignment=fl_rev, skeleton_id="facility_location", objective_unit="cost_unit", evolution_operator="add_open_limit", revision_of="fl_b1", contract_version=2),
            _positive_case(case_id="fl_e1", family="facility_location", task_kind="explain", question="Formulate the facility model and explain why each assignment-to-opening linking constraint is necessary.", variables=fl_variables, objective_sense="minimize", objective_coefficients=fl_costs, constraints=fl_constraints, expected_objective=6, optimal_assignment=fl_opt, skeleton_id="facility_location", objective_unit="cost_unit", evolution_operator="explain_linking_constraints"),
        ]
    )
    cases.append(
        _no_result_case(
            case_id="fl_n1", family="facility_location", question="Choose between two facilities, but the opening cost of facility two is unknown and may change the optimal choice. Do not invent it.",
            variables=fl_variables,
            clauses=[
                {"clause_id": "primary_objective", "kind": "objective", "statement": "Minimize opening and assignment costs including the unknown facility-two opening cost", "unit": "cost_unit", "acceptance_criterion": "Do not assign an unstated coefficient to y2"},
                {"clause_id": "assign_once", "kind": "hard_constraint", "statement": "Each customer must be assigned exactly once", "unit": "assignment_count", "acceptance_criterion": "Each customer assignment row sums to one"},
                {"clause_id": "link_open", "kind": "hard_constraint", "statement": "Assignments require an open facility", "unit": "binary_count", "acceptance_criterion": "Every assignment variable is bounded by its facility opening variable"},
            ], public_facts=[],
            acceptance_tests=[{"test_id": "fl_n1_probe", "kind": "assignment_case", "assignment": fl_opt, "expected_feasible": True}],
            reason_class="missing_fact", skeleton_id="facility_location",
        )
    )

    sc_variables = _binary_names("s", 4, "set_choice")
    sc_costs = {"s1": 3, "s2": 2, "s3": 3, "s4": 2}
    sc_constraints = [
        _constraint("cover_1", {"s1": 1, "s4": 1}, ">=", 1, "Element one must be covered", unit="coverage_count"),
        _constraint("cover_2", {"s1": 1, "s2": 1}, ">=", 1, "Element two must be covered", unit="coverage_count"),
        _constraint("cover_3", {"s2": 1, "s3": 1}, ">=", 1, "Element three must be covered", unit="coverage_count"),
        _constraint("cover_4", {"s3": 1, "s4": 1}, ">=", 1, "Element four must be covered", unit="coverage_count"),
    ]
    cases.extend(
        [
            _positive_case(case_id="sc_b1", family="set_covering", task_kind="build", question="Choose minimum-cost sets so that all four elements are covered.", variables=sc_variables, objective_sense="minimize", objective_coefficients=sc_costs, constraints=sc_constraints, expected_objective=4, optimal_assignment={"s1": 0, "s2": 1, "s3": 0, "s4": 1}, skeleton_id="set_covering", objective_unit="cost_unit", evolution_operator="formulate"),
            _positive_case(case_id="sc_r1", family="set_covering", task_kind="revise", question="Revise the set-cover model because set four is unavailable.", variables=sc_variables, objective_sense="minimize", objective_coefficients=sc_costs, constraints=[*sc_constraints, _constraint("disable_s4", {"s4": 1}, "==", 0, "Set four is unavailable", unit="set_count")], expected_objective=6, optimal_assignment={"s1": 1, "s2": 0, "s3": 1, "s4": 0}, skeleton_id="set_covering", objective_unit="cost_unit", evolution_operator="disable_set", revision_of="sc_b1", contract_version=2),
            _positive_case(case_id="sc_e1", family="set_covering", task_kind="explain", question="Add a fifth set covering all elements at cost seven, formulate the model, and explain why the cheaper combination remains preferable.", variables=_binary_names("s", 5, "set_choice"), objective_sense="minimize", objective_coefficients={**sc_costs, "s5": 7}, constraints=[_constraint(str(c["clause_id"]), {**c["coefficients"], "s5": 1}, str(c["sense"]), float(c["rhs"]), str(c["statement"]), unit=str(c["unit"])) for c in sc_constraints], expected_objective=4, optimal_assignment={"s1": 0, "s2": 1, "s3": 0, "s4": 1, "s5": 0}, skeleton_id="set_covering", objective_unit="cost_unit", evolution_operator="add_dominated_set"),
        ]
    )
    sc_continuous = [_variable(f"s{index}", kind="continuous", upper=1, unit="set_fraction") for index in range(1, 5)]
    cases.append(
        _no_result_case(
            case_id="sc_n1", family="set_covering", question="Solve the fractional set-cover relaxation with continuous decisions between zero and one.",
            variables=sc_continuous,
            clauses=[
                {"clause_id": "primary_objective", "kind": "objective", "statement": "Minimize fractional set cost", "unit": "cost_unit", "acceptance_criterion": "Use continuous decisions exactly as declared"},
                {"clause_id": "cover_all", "kind": "hard_constraint", "statement": "Every element requires fractional coverage of at least one", "unit": "coverage_count", "acceptance_criterion": "Each element coverage sum is at least one"},
            ], public_facts=[],
            acceptance_tests=[{"test_id": "sc_n1_all", "kind": "assignment_case", "assignment": {f"s{i}": 1 for i in range(1, 5)}, "expected_feasible": True, "expected_objective": 10}],
            reason_class="unsupported_domain", skeleton_id="set_covering",
        )
    )

    return BenchmarkSuite.seal(cases)
