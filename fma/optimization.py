from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp

from .hashing import sha256_value
from .schemas import (
    CompilerCertificate,
    OptimizationModelIR,
    SolutionArtifact,
    VariableKind,
)


@dataclass(frozen=True)
class CompiledOptimization:
    ir: OptimizationModelIR
    c: np.ndarray
    integrality: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    matrix: np.ndarray
    constraint_lower: np.ndarray
    constraint_upper: np.ndarray
    certificate: CompilerCertificate


def _row_bounds(sense: str, rhs: float) -> tuple[float, float]:
    if sense == "<=":
        return -np.inf, rhs
    if sense == ">=":
        return rhs, np.inf
    return rhs, rhs


def _certificate_payload(
    variable_order: list[str],
    objective_vector: list[float],
    integrality: list[int],
    variable_bounds: list[tuple[float, float]],
    constraint_rows: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "variable_order": variable_order,
        "objective_vector": objective_vector,
        "integrality": integrality,
        "variable_bounds": variable_bounds,
        "constraint_rows": constraint_rows,
    }


def _bound_token(value: float) -> float | str:
    if np.isneginf(value):
        return "-inf"
    if np.isposinf(value):
        return "+inf"
    return float(value)


def _execution_payload_from_arrays(
    c: np.ndarray,
    integrality: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    matrix: np.ndarray,
    constraint_lower: np.ndarray,
    constraint_upper: np.ndarray,
) -> dict[str, object]:
    return {
        "c": c.tolist(),
        "integrality": integrality.tolist(),
        "lower_bounds": lower_bounds.tolist(),
        "upper_bounds": upper_bounds.tolist(),
        "matrix": matrix.tolist(),
        "constraint_lower": [_bound_token(value) for value in constraint_lower],
        "constraint_upper": [_bound_token(value) for value in constraint_upper],
    }


def execution_array_payload(compiled: "CompiledOptimization") -> dict[str, object]:
    """Canonical representation of the numeric arrays actually sent to SciPy."""
    return _execution_payload_from_arrays(
        compiled.c,
        compiled.integrality,
        compiled.lower_bounds,
        compiled.upper_bounds,
        compiled.matrix,
        compiled.constraint_lower,
        compiled.constraint_upper,
    )


def compile_optimization(ir: OptimizationModelIR) -> CompiledOptimization:
    """Deterministically compile sealed linear IR into SciPy MILP arrays."""
    ir.assert_sealed()
    variable_order = [variable.name for variable in ir.variables]
    objective_sign = 1.0 if ir.objective.sense == "minimize" else -1.0
    objective_vector = [
        objective_sign * ir.objective.coefficients.get(name, 0.0)
        for name in variable_order
    ]
    integrality = [
        0 if variable.kind == VariableKind.CONTINUOUS else 1
        for variable in ir.variables
    ]
    variable_bounds = [
        (variable.lower_bound, variable.upper_bound) for variable in ir.variables
    ]

    constraint_rows: list[dict[str, object]] = []
    numeric_rows: list[list[float]] = []
    row_lowers: list[float] = []
    row_uppers: list[float] = []
    for constraint in ir.constraints:
        coefficients = [constraint.coefficients.get(name, 0.0) for name in variable_order]
        lower, upper = _row_bounds(constraint.sense, constraint.rhs)
        numeric_rows.append(coefficients)
        row_lowers.append(lower)
        row_uppers.append(upper)
        constraint_rows.append(
            {
                "constraint_id": constraint.constraint_id,
                "coefficients": coefficients,
                "sense": constraint.sense,
                "rhs": constraint.rhs,
            }
        )

    payload = _certificate_payload(
        variable_order,
        objective_vector,
        integrality,
        variable_bounds,
        constraint_rows,
    )
    c_array = np.asarray(objective_vector, dtype=float)
    integrality_array = np.asarray(integrality, dtype=int)
    lower_array = np.asarray([bound[0] for bound in variable_bounds], dtype=float)
    upper_array = np.asarray([bound[1] for bound in variable_bounds], dtype=float)
    matrix_array = np.asarray(numeric_rows, dtype=float)
    constraint_lower_array = np.asarray(row_lowers, dtype=float)
    constraint_upper_array = np.asarray(row_uppers, dtype=float)
    execution_hash = sha256_value(
        _execution_payload_from_arrays(
            c_array,
            integrality_array,
            lower_array,
            upper_array,
            matrix_array,
            constraint_lower_array,
            constraint_upper_array,
        )
    )
    certificate = CompilerCertificate(
        source_ir_hash=ir.ir_hash,
        variable_order=variable_order,
        objective_vector=objective_vector,
        integrality=integrality,
        variable_bounds=variable_bounds,
        constraint_rows=constraint_rows,
        matrix_hash=sha256_value(payload),
        execution_array_hash=execution_hash,
    )
    return CompiledOptimization(
        ir=ir,
        c=c_array,
        integrality=integrality_array,
        lower_bounds=lower_array,
        upper_bounds=upper_array,
        matrix=matrix_array,
        constraint_lower=constraint_lower_array,
        constraint_upper=constraint_upper_array,
        certificate=certificate,
    )


def evaluate_objective(ir: OptimizationModelIR, values: dict[str, float]) -> float:
    return ir.objective.constant + sum(
        coefficient * values[name]
        for name, coefficient in ir.objective.coefficients.items()
    )


def solve_compiled(
    compiled: CompiledOptimization,
    *,
    time_limit_seconds: float = 10.0,
) -> SolutionArtifact:
    actual_execution_hash = sha256_value(execution_array_payload(compiled))
    if actual_execution_hash != compiled.certificate.execution_array_hash:
        raise RuntimeError("compiled numeric arrays changed after certification")
    constraints = LinearConstraint(
        compiled.matrix,
        compiled.constraint_lower,
        compiled.constraint_upper,
    )
    result = milp(
        compiled.c,
        integrality=compiled.integrality,
        bounds=Bounds(compiled.lower_bounds, compiled.upper_bounds),
        constraints=constraints,
        options={"disp": False, "time_limit": time_limit_seconds},
    )
    if result.status == 0 and result.x is not None:
        status = "optimal"
        values = {
            name: float(result.x[index])
            for index, name in enumerate(compiled.certificate.variable_order)
        }
        objective_value = evaluate_objective(compiled.ir, values)
    else:
        status = {2: "infeasible", 3: "unbounded"}.get(result.status, "failed")
        values = {}
        objective_value = None
    return SolutionArtifact(
        solver=f"scipy.optimize.milp/{scipy.__version__}",
        solver_status=status,
        source_ir_hash=compiled.ir.ir_hash,
        compiled_matrix_hash=compiled.certificate.matrix_hash,
        execution_array_hash=actual_execution_hash,
        values=values,
        objective_value=objective_value,
        message=str(result.message),
    )
