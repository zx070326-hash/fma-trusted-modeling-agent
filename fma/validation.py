from __future__ import annotations

import itertools
import math

from .hashing import sha256_value
from .optimization import evaluate_objective
from .schemas import (
    CheckRecord,
    ClauseKind,
    CompilerCertificate,
    ConstraintSpec,
    OptimizationModelIR,
    ProblemContract,
    SolutionArtifact,
    ValidationVector,
    VariableKind,
)


TOLERANCE = 1e-7
MAX_ORACLE_ASSIGNMENTS = 100_000
MAX_PREFLIGHT_VARIABLES = 1_000
MAX_PREFLIGHT_CONSTRAINTS = 5_000

REQUIRED_HARD_CHECKS = {
    "schema_integrity",
    "contract_binding",
    "contract_coverage",
    "unit_consistency",
    "compiler_fidelity",
    "solution_provenance",
    "solver_status",
    "feasibility",
    "objective_consistency",
    "optimality_oracle",
    "contract_acceptance",
    "reproducibility",
}


def _record(ok: bool, passed: str, failed: str) -> CheckRecord:
    return CheckRecord(status="pass" if ok else "fail", detail=passed if ok else failed)


def _satisfies(constraint: ConstraintSpec, lhs: float, tolerance: float) -> bool:
    if constraint.sense == "<=":
        return lhs <= constraint.rhs + tolerance
    if constraint.sense == ">=":
        return lhs >= constraint.rhs - tolerance
    return abs(lhs - constraint.rhs) <= tolerance


def _compiler_fidelity(ir: OptimizationModelIR, certificate: CompilerCertificate) -> tuple[bool, str]:
    """Reconstruct compiler mappings directly from IR, without solver arrays."""
    variable_order = [variable.name for variable in ir.variables]
    sign = 1.0 if ir.objective.sense == "minimize" else -1.0
    objective_vector = [
        sign * ir.objective.coefficients.get(name, 0.0) for name in variable_order
    ]
    integrality = [
        0 if variable.kind == VariableKind.CONTINUOUS else 1
        for variable in ir.variables
    ]
    variable_bounds = [
        (variable.lower_bound, variable.upper_bound) for variable in ir.variables
    ]
    rows: list[dict[str, object]] = []
    for constraint in ir.constraints:
        rows.append(
            {
                "constraint_id": constraint.constraint_id,
                "coefficients": [
                    constraint.coefficients.get(name, 0.0) for name in variable_order
                ],
                "sense": constraint.sense,
                "rhs": constraint.rhs,
            }
        )
    payload = {
        "variable_order": variable_order,
        "objective_vector": objective_vector,
        "integrality": integrality,
        "variable_bounds": variable_bounds,
        "constraint_rows": rows,
    }
    expected_hash = sha256_value(payload)
    constraint_lower: list[float | str] = []
    constraint_upper: list[float | str] = []
    for constraint in ir.constraints:
        if constraint.sense == "<=":
            constraint_lower.append("-inf")
            constraint_upper.append(constraint.rhs)
        elif constraint.sense == ">=":
            constraint_lower.append(constraint.rhs)
            constraint_upper.append("+inf")
        else:
            constraint_lower.append(constraint.rhs)
            constraint_upper.append(constraint.rhs)
    execution_payload = {
        "c": objective_vector,
        "integrality": integrality,
        "lower_bounds": [bound[0] for bound in variable_bounds],
        "upper_bounds": [bound[1] for bound in variable_bounds],
        "matrix": [row["coefficients"] for row in rows],
        "constraint_lower": constraint_lower,
        "constraint_upper": constraint_upper,
    }
    expected_execution_hash = sha256_value(execution_payload)
    fields_match = (
        certificate.source_ir_hash == ir.ir_hash
        and certificate.variable_order == variable_order
        and certificate.objective_vector == objective_vector
        and certificate.integrality == integrality
        and certificate.variable_bounds == variable_bounds
        and certificate.constraint_rows == rows
        and certificate.matrix_hash == expected_hash
        and certificate.execution_array_hash == expected_execution_hash
    )
    if fields_match:
        return True, "compiler certificate is a complete deterministic mapping of the IR"
    return False, "compiler certificate does not faithfully map the source IR"


def _coverage(contract: ProblemContract, ir: OptimizationModelIR) -> tuple[bool, list[str]]:
    clauses = {clause.clause_id: clause for clause in contract.clauses}
    failures: list[str] = []
    objective_refs = set(ir.objective.contract_clause_ids)
    constraint_refs = {
        clause_id
        for constraint in ir.constraints
        for clause_id in constraint.contract_clause_ids
    }
    for clause_id in objective_refs | constraint_refs:
        if clause_id not in clauses:
            failures.append(f"unknown contract clause: {clause_id}")
    for clause_id in objective_refs:
        if clause_id in clauses and clauses[clause_id].kind != ClauseKind.OBJECTIVE:
            failures.append(f"objective references non-objective clause: {clause_id}")
    for clause_id in constraint_refs:
        if clause_id in clauses and clauses[clause_id].kind != ClauseKind.HARD_CONSTRAINT:
            failures.append(f"constraint references non-hard clause: {clause_id}")
    required_objectives = {
        clause.clause_id for clause in contract.clauses if clause.kind == ClauseKind.OBJECTIVE
    }
    required_constraints = {
        clause.clause_id
        for clause in contract.clauses
        if clause.kind == ClauseKind.HARD_CONSTRAINT
    }
    for missing in sorted(required_objectives - objective_refs):
        failures.append(f"unmapped objective clause: {missing}")
    for missing in sorted(required_constraints - constraint_refs):
        failures.append(f"unmapped hard-constraint clause: {missing}")
    return not failures, failures


def _units(contract: ProblemContract, ir: OptimizationModelIR) -> tuple[bool, list[str]]:
    clauses = {clause.clause_id: clause for clause in contract.clauses}
    failures: list[str] = []
    if ir.objective.unit == "":
        failures.append("objective unit is empty")
    for clause_id in ir.objective.contract_clause_ids:
        if clause_id in clauses and clauses[clause_id].unit != ir.objective.unit:
            failures.append(f"objective unit differs from clause {clause_id}")
    for constraint in ir.constraints:
        if constraint.lhs_unit != constraint.rhs_unit:
            failures.append(f"unit mismatch in {constraint.constraint_id}")
        for clause_id in constraint.contract_clause_ids:
            if clause_id in clauses and clauses[clause_id].unit != constraint.rhs_unit:
                failures.append(
                    f"constraint unit differs from clause {clause_id} in {constraint.constraint_id}"
                )
    return not failures, failures


def _decision_alignment(
    contract: ProblemContract, ir: OptimizationModelIR
) -> tuple[bool, list[str]]:
    if not contract.decisions:
        return True, []
    declared = {decision.decision_id: decision for decision in contract.decisions}
    submitted = {variable.name: variable for variable in ir.variables}
    failures: list[str] = []
    for missing in sorted(set(declared) - set(submitted)):
        failures.append(f"missing declared decision variable: {missing}")
    for unknown in sorted(set(submitted) - set(declared)):
        failures.append(f"undeclared decision variable: {unknown}")
    for name in sorted(set(declared) & set(submitted)):
        decision = declared[name]
        variable = submitted[name]
        if variable.kind.value != decision.kind:
            failures.append(f"decision kind differs for {name}")
        if variable.unit != decision.unit:
            failures.append(f"decision unit differs for {name}")
        if (
            decision.lower_bound is not None
            and variable.lower_bound != decision.lower_bound
        ):
            failures.append(f"declared lower bound differs for {name}")
        if (
            decision.upper_bound is not None
            and variable.upper_bound != decision.upper_bound
        ):
            failures.append(f"declared upper bound differs for {name}")
    return not failures, failures


def public_preflight_candidate(
    contract: ProblemContract, ir: OptimizationModelIR
) -> list[str]:
    """Checks derived only from the public contract view and submitted IR.

    These diagnostics are safe to return to an Explorer.  Executable acceptance
    tests deliberately live in :func:`preflight_candidate` so a repair loop
    cannot turn the hidden evaluator into an oracle.
    """
    failures: list[str] = []
    try:
        contract.assert_frozen()
        ir.assert_sealed()
    except ValueError as exc:
        failures.append(str(exc))
    if ir.contract_hash != contract.frozen_hash:
        failures.append("IR contract_hash differs from the frozen contract")
    _, coverage_failures = _coverage(contract, ir)
    failures.extend(coverage_failures)
    _, unit_failures = _units(contract, ir)
    failures.extend(unit_failures)
    _, decision_failures = _decision_alignment(contract, ir)
    failures.extend(decision_failures)
    if len(ir.variables) > MAX_PREFLIGHT_VARIABLES:
        failures.append("candidate exceeds the variable-count preflight budget")
    if len(ir.constraints) > MAX_PREFLIGHT_CONSTRAINTS:
        failures.append("candidate exceeds the constraint-count preflight budget")

    return failures


def preflight_candidate(contract: ProblemContract, ir: OptimizationModelIR) -> list[str]:
    """All cheap checks required before compilation or solver resource use."""
    failures = public_preflight_candidate(contract, ir)

    expected_variables = {variable.name for variable in ir.variables}
    for test in contract.acceptance_tests:
        if test.kind != "assignment_case":
            continue
        if set(test.assignment) != expected_variables:
            failures.append(f"{test.test_id}: assignment keys do not match IR variables")
            continue
        feasible, _ = _feasibility(ir, test.assignment, test.tolerance)
        if feasible != test.expected_feasible:
            failures.append(
                f"{test.test_id}: feasibility {feasible} != registered "
                f"{test.expected_feasible}"
            )
        if test.expected_objective is not None:
            observed = ir.objective.constant
            for name, coefficient in ir.objective.coefficients.items():
                observed += coefficient * test.assignment[name]
            if abs(observed - test.expected_objective) > test.tolerance:
                failures.append(
                    f"{test.test_id}: assignment objective {observed} != "
                    f"registered {test.expected_objective}"
                )
    return failures


def _feasibility(
    ir: OptimizationModelIR,
    values: dict[str, float],
    tolerance: float,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    expected = {variable.name for variable in ir.variables}
    missing = expected - set(values)
    unknown = set(values) - expected
    if missing:
        failures.append(f"missing variables: {sorted(missing)}")
    if unknown:
        failures.append(f"unknown variables: {sorted(unknown)}")
    if failures:
        return False, failures
    for variable in ir.variables:
        value = values[variable.name]
        if value < variable.lower_bound - tolerance or value > variable.upper_bound + tolerance:
            failures.append(f"bound violation: {variable.name}={value}")
        if variable.kind in {VariableKind.INTEGER, VariableKind.BINARY} and abs(value - round(value)) > tolerance:
            failures.append(f"integrality violation: {variable.name}={value}")
    for constraint in ir.constraints:
        lhs = sum(
            coefficient * values[name]
            for name, coefficient in constraint.coefficients.items()
        )
        if not _satisfies(constraint, lhs, tolerance):
            failures.append(
                f"constraint violation: {constraint.constraint_id} has lhs={lhs}, "
                f"expected {constraint.sense} {constraint.rhs}"
            )
    return not failures, failures


def _oracle(
    ir: OptimizationModelIR,
    submitted_objective: float | None,
    tolerance: float,
    max_assignments: int,
) -> tuple[CheckRecord, dict[str, float | int | str | bool | None], list[str]]:
    if any(variable.kind == VariableKind.CONTINUOUS for variable in ir.variables):
        return (
            CheckRecord(status="not_run", detail="exact oracle supports bounded integer models only"),
            {"oracle_assignments": None, "oracle_objective": None},
            ["exact optimality oracle is unavailable for continuous variables"],
        )
    bounds: list[tuple[int, int]] = []
    assignment_count = 1
    for variable in ir.variables:
        lower = math.ceil(variable.lower_bound)
        upper = math.floor(variable.upper_bound)
        width = max(0, upper - lower + 1)
        bounds.append((lower, upper))
        assignment_count *= width
        if assignment_count > max_assignments:
            break
    if assignment_count > max_assignments:
        return (
            CheckRecord(
                status="not_run",
                detail=f"oracle space {assignment_count} exceeds limit {max_assignments}",
            ),
            {"oracle_assignments": assignment_count, "oracle_objective": None},
            ["exact optimality oracle budget exceeded"],
        )
    domains = [range(lower, upper + 1) for lower, upper in bounds]
    names = [variable.name for variable in ir.variables]
    best: float | None = None
    feasible_count = 0
    for assignment in itertools.product(*domains):
        values = dict(zip(names, map(float, assignment), strict=True))
        # Deliberately do not call the normal feasibility checker here: the
        # exact oracle is a second implementation over the source IR.
        oracle_feasible = True
        for constraint in ir.constraints:
            lhs = sum(
                coefficient * values[name]
                for name, coefficient in constraint.coefficients.items()
            )
            if constraint.sense == "<=" and lhs > constraint.rhs + tolerance:
                oracle_feasible = False
                break
            if constraint.sense == ">=" and lhs < constraint.rhs - tolerance:
                oracle_feasible = False
                break
            if constraint.sense == "==" and abs(lhs - constraint.rhs) > tolerance:
                oracle_feasible = False
                break
        if not oracle_feasible:
            continue
        feasible_count += 1
        objective = ir.objective.constant
        for variable_name, coefficient in ir.objective.coefficients.items():
            objective += coefficient * values[variable_name]
        if best is None:
            best = objective
        elif ir.objective.sense == "maximize" and objective > best:
            best = objective
        elif ir.objective.sense == "minimize" and objective < best:
            best = objective
    metrics: dict[str, float | int | str | bool | None] = {
        "oracle_assignments": assignment_count,
        "oracle_feasible_assignments": feasible_count,
        "oracle_objective": best,
    }
    if best is None:
        return (
            CheckRecord(status="fail", detail="original IR has no feasible integer assignment"),
            metrics,
            ["oracle found no feasible assignment"],
        )
    matches = submitted_objective is not None and abs(submitted_objective - best) <= tolerance
    detail = (
        f"independent enumeration confirms optimum {best}"
        if matches
        else f"submitted objective {submitted_objective} differs from oracle optimum {best}"
    )
    return CheckRecord(status="pass" if matches else "fail", detail=detail), metrics, ([] if matches else [detail])


def _contract_acceptance(
    contract: ProblemContract,
    ir: OptimizationModelIR,
    solution: SolutionArtifact,
    tolerance: float,
) -> tuple[bool, list[str]]:
    """Run frozen executable microcase tests; this is not NL semantic equivalence."""
    failures: list[str] = []
    expected_variables = {variable.name for variable in ir.variables}
    for test in contract.acceptance_tests:
        if test.kind == "known_optimum":
            if set(solution.values) != expected_variables:
                failures.append(f"{test.test_id}: submitted assignment is incomplete")
                continue
            observed = ir.objective.constant
            for name, coefficient in ir.objective.coefficients.items():
                observed += coefficient * solution.values[name]
            assert test.expected_objective is not None
            if abs(observed - test.expected_objective) > test.tolerance:
                failures.append(
                    f"{test.test_id}: observed objective {observed} != "
                    f"registered {test.expected_objective}"
                )
            continue

        if set(test.assignment) != expected_variables:
            failures.append(f"{test.test_id}: assignment keys do not match IR variables")
            continue
        feasible, _ = _feasibility(ir, test.assignment, test.tolerance)
        if feasible != test.expected_feasible:
            failures.append(
                f"{test.test_id}: feasibility {feasible} != registered "
                f"{test.expected_feasible}"
            )
        if test.expected_objective is not None:
            observed = ir.objective.constant
            for name, coefficient in ir.objective.coefficients.items():
                observed += coefficient * test.assignment[name]
            if abs(observed - test.expected_objective) > test.tolerance:
                failures.append(
                    f"{test.test_id}: assignment objective {observed} != "
                    f"registered {test.expected_objective}"
                )
    return not failures, failures


def validate_candidate(
    contract: ProblemContract,
    ir: OptimizationModelIR,
    certificate: CompilerCertificate,
    solution: SolutionArtifact,
    *,
    tolerance: float = TOLERANCE,
    max_oracle_assignments: int = MAX_ORACLE_ASSIGNMENTS,
) -> ValidationVector:
    checks: dict[str, CheckRecord] = {}
    violations: list[str] = []
    warnings: list[str] = []

    try:
        contract.assert_frozen()
        ir.assert_sealed()
        schema_ok = True
        schema_detail = "contract and IR hashes match their canonical content"
    except ValueError as exc:
        schema_ok = False
        schema_detail = str(exc)
        violations.append(str(exc))
    checks["schema_integrity"] = _record(schema_ok, schema_detail, schema_detail)

    binding_ok = ir.contract_hash == contract.frozen_hash
    checks["contract_binding"] = _record(
        binding_ok,
        "IR is bound to the frozen contract hash",
        "IR contract_hash differs from the frozen contract",
    )
    if not binding_ok:
        violations.append("IR is bound to a different contract")

    coverage_ok, coverage_failures = _coverage(contract, ir)
    checks["contract_coverage"] = _record(
        coverage_ok,
        "all objective and hard-constraint clauses are mapped",
        "; ".join(coverage_failures),
    )
    violations.extend(coverage_failures)

    units_ok, unit_failures = _units(contract, ir)
    checks["unit_consistency"] = _record(
        units_ok,
        "unit labels agree across mapped clauses and equations",
        "; ".join(unit_failures),
    )
    violations.extend(unit_failures)

    compiler_ok, compiler_detail = _compiler_fidelity(ir, certificate)
    checks["compiler_fidelity"] = _record(
        compiler_ok,
        compiler_detail,
        compiler_detail,
    )
    if not compiler_ok:
        violations.append(compiler_detail)

    provenance_ok = (
        solution.source_ir_hash == ir.ir_hash
        and solution.compiled_matrix_hash == certificate.matrix_hash
        and solution.execution_array_hash == certificate.execution_array_hash
    )
    checks["solution_provenance"] = _record(
        provenance_ok,
        "solution is bound to this IR and compiler certificate",
        "solution provenance hashes do not match this candidate",
    )
    if not provenance_ok:
        violations.append("solution provenance mismatch")

    solver_ok = solution.solver_status == "optimal"
    checks["solver_status"] = _record(
        solver_ok,
        "solver reported an optimum",
        f"solver status is {solution.solver_status}",
    )
    if not solver_ok:
        violations.append(f"solver status is {solution.solver_status}")

    feasible, feasibility_failures = _feasibility(ir, solution.values, tolerance)
    checks["feasibility"] = _record(
        feasible,
        "independent IR interpreter confirms bounds, integrality, and constraints",
        "; ".join(feasibility_failures),
    )
    violations.extend(feasibility_failures)

    if set(solution.values) == {variable.name for variable in ir.variables}:
        recomputed_objective = evaluate_objective(ir, solution.values)
        objective_ok = (
            solution.objective_value is not None
            and abs(recomputed_objective - solution.objective_value) <= tolerance
        )
        objective_detail = (
            f"reported objective matches independent recomputation {recomputed_objective}"
            if objective_ok
            else f"reported objective {solution.objective_value} differs from recomputed {recomputed_objective}"
        )
    else:
        recomputed_objective = None
        objective_ok = False
        objective_detail = "objective cannot be recomputed from an incomplete assignment"
    checks["objective_consistency"] = _record(
        objective_ok,
        objective_detail,
        objective_detail,
    )
    if not objective_ok:
        violations.append(objective_detail)

    oracle_check, oracle_metrics, oracle_violations = _oracle(
        ir,
        recomputed_objective,
        tolerance,
        max_oracle_assignments,
    )
    checks["optimality_oracle"] = oracle_check
    if oracle_check.status == "not_run":
        warnings.extend(oracle_violations)
    else:
        violations.extend(oracle_violations)

    acceptance_ok, acceptance_failures = _contract_acceptance(
        contract, ir, solution, tolerance
    )
    checks["contract_acceptance"] = _record(
        acceptance_ok,
        "all frozen executable microcase acceptance tests passed",
        "; ".join(acceptance_failures),
    )
    violations.extend(acceptance_failures)

    checks["leakage_firewall"] = CheckRecord(
        status="not_run",
        detail="no instrumented leakage audit exists in this no-retrieval vertical slice",
    )
    warnings.append("leakage was not instrumented; no hidden dataset was used by design")
    checks["reproducibility"] = CheckRecord(
        status="not_run",
        detail="fresh-process replay has not yet been attached",
    )
    metrics = {
        "recomputed_objective": recomputed_objective,
        "tolerance": tolerance,
        **oracle_metrics,
    }
    return ValidationVector(
        candidate_id=ir.candidate_id,
        checks=checks,
        violations=violations,
        warnings=warnings,
        metrics=metrics,
    )


def attach_reproduction(
    validation: ValidationVector,
    *,
    passed: bool,
    detail: str,
) -> ValidationVector:
    updated = validation.model_copy(deep=True)
    updated.checks["reproducibility"] = CheckRecord(
        status="pass" if passed else "fail",
        detail=detail,
    )
    if not passed:
        updated.violations.append(detail)
    return updated
