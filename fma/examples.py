from __future__ import annotations

from .hashing import sha256_value
from .optimization import compile_optimization, execution_array_payload
from .schemas import (
    CompilerCertificate,
    OptimizationModelIR,
    ProblemContract,
    SolutionArtifact,
)


def resource_allocation_contract() -> ProblemContract:
    return ProblemContract.freeze(
        contract_id="resource_allocation_demo",
        version=1,
        question="How many units of products x and y maximize benefit under two resources?",
        system_boundary="One synthetic factory with two products and two finite resources",
        decision_horizon="One planning period",
        decisions=[
            {
                "decision_id": "x",
                "statement": "Number of units of product x to make",
                "kind": "integer",
                "unit": "product_unit",
                "lower_bound": 0,
                "source_ref": "synthetic_demo:decision_x",
            },
            {
                "decision_id": "y",
                "statement": "Number of units of product y to make",
                "kind": "integer",
                "unit": "product_unit",
                "lower_bound": 0,
                "source_ref": "synthetic_demo:decision_y",
            },
        ],
        clauses=[
            {
                "clause_id": "maximize_benefit",
                "kind": "objective",
                "statement": "Maximize total benefit points",
                "unit": "benefit_point",
                "source_ref": "synthetic_demo:objective",
                "acceptance_criterion": "Return the globally maximal integer objective",
            },
            {
                "clause_id": "resource_a_limit",
                "kind": "hard_constraint",
                "statement": "Resource A use cannot exceed eight units",
                "unit": "resource_a_unit",
                "source_ref": "synthetic_demo:resource_a",
                "acceptance_criterion": "Two x plus y is at most eight",
            },
            {
                "clause_id": "resource_b_limit",
                "kind": "hard_constraint",
                "statement": "Resource B use cannot exceed nine units",
                "unit": "resource_b_unit",
                "source_ref": "synthetic_demo:resource_b",
                "acceptance_criterion": "x plus three y is at most nine",
            },
            {
                "clause_id": "integer_decisions",
                "kind": "assumption",
                "statement": "Production quantities are nonnegative integers",
                "unit": "unitless",
                "source_ref": "synthetic_demo:assumption",
                "acceptance_criterion": "Both decision values are integral and nonnegative",
            },
        ],
        public_facts=[
            {
                "fact_id": "benefit_per_x",
                "statement": "Each unit of x contributes three benefit points",
                "value": 3,
                "unit": "benefit_point_per_product_unit",
                "source_ref": "synthetic_demo:public_data",
            },
            {
                "fact_id": "benefit_per_y",
                "statement": "Each unit of y contributes five benefit points",
                "value": 5,
                "unit": "benefit_point_per_product_unit",
                "source_ref": "synthetic_demo:public_data",
            },
            {
                "fact_id": "resource_a_capacity",
                "statement": "Eight units of resource A are available",
                "value": 8,
                "unit": "resource_a_unit",
                "source_ref": "synthetic_demo:public_data",
            },
            {
                "fact_id": "resource_b_capacity",
                "statement": "Nine units of resource B are available",
                "value": 9,
                "unit": "resource_b_unit",
                "source_ref": "synthetic_demo:public_data",
            },
            {
                "fact_id": "resource_a_per_x",
                "statement": "Each unit of x consumes two units of resource A",
                "value": 2,
                "unit": "resource_a_unit_per_product_unit",
                "source_ref": "synthetic_demo:public_data",
            },
            {
                "fact_id": "resource_a_per_y",
                "statement": "Each unit of y consumes one unit of resource A",
                "value": 1,
                "unit": "resource_a_unit_per_product_unit",
                "source_ref": "synthetic_demo:public_data",
            },
            {
                "fact_id": "resource_b_per_x",
                "statement": "Each unit of x consumes one unit of resource B",
                "value": 1,
                "unit": "resource_b_unit_per_product_unit",
                "source_ref": "synthetic_demo:public_data",
            },
            {
                "fact_id": "resource_b_per_y",
                "statement": "Each unit of y consumes three units of resource B",
                "value": 3,
                "unit": "resource_b_unit_per_product_unit",
                "source_ref": "synthetic_demo:public_data",
            },
        ],
        acceptance_tests=[
            {
                "test_id": "known_global_optimum",
                "kind": "known_optimum",
                "expected_objective": 19,
                "source_ref": "synthetic_demo:independent_enumeration",
            },
            {
                "test_id": "known_feasible_optimum",
                "kind": "assignment_case",
                "assignment": {"x": 3, "y": 2},
                "expected_feasible": True,
                "expected_objective": 19,
                "source_ref": "synthetic_demo:hand_check",
            },
            {
                "test_id": "objective_x_unit_case",
                "kind": "assignment_case",
                "assignment": {"x": 1, "y": 0},
                "expected_feasible": True,
                "expected_objective": 3,
                "source_ref": "synthetic_demo:coefficient_identification",
            },
            {
                "test_id": "objective_y_unit_case",
                "kind": "assignment_case",
                "assignment": {"x": 0, "y": 1},
                "expected_feasible": True,
                "expected_objective": 5,
                "source_ref": "synthetic_demo:coefficient_identification",
            },
            {
                "test_id": "resource_b_counterexample",
                "kind": "assignment_case",
                "assignment": {"x": 0, "y": 8},
                "expected_feasible": False,
                "source_ref": "synthetic_demo:hand_check",
            },
            {
                "test_id": "gross_infeasibility_counterexample",
                "kind": "assignment_case",
                "assignment": {"x": 10, "y": 10},
                "expected_feasible": False,
                "source_ref": "synthetic_demo:hand_check",
            },
        ],
        permitted_actions=[
            "codex_cli_inference",
            "local_compute",
            "write_local_run_artifacts",
        ],
        forbidden_actions=[
            "model_generated_network_access",
            "external_action",
        ],
        risk_level="A1",
    )


def resource_allocation_ir(contract: ProblemContract) -> OptimizationModelIR:
    contract.assert_frozen()
    return OptimizationModelIR.seal(
        candidate_id="resource_allocation_milp",
        contract_hash=contract.frozen_hash,
        skeleton_id="bounded_integer_linear_program",
        lineage={
            "root_kind": "fixture",
            "parent_candidate_ids": [],
            "rationale": "Hand-authored fixture isolates harness trust from model generation",
        },
        variables=[
            {
                "name": "x",
                "kind": "integer",
                "lower_bound": 0,
                "upper_bound": 10,
                "unit": "product_unit",
            },
            {
                "name": "y",
                "kind": "integer",
                "lower_bound": 0,
                "upper_bound": 10,
                "unit": "product_unit",
            },
        ],
        objective={
            "sense": "maximize",
            "coefficients": {"x": 3, "y": 5},
            "constant": 0,
            "unit": "benefit_point",
            "contract_clause_ids": ["maximize_benefit"],
        },
        constraints=[
            {
                "constraint_id": "resource_a",
                "coefficients": {"x": 2, "y": 1},
                "sense": "<=",
                "rhs": 8,
                "lhs_unit": "resource_a_unit",
                "rhs_unit": "resource_a_unit",
                "contract_clause_ids": ["resource_a_limit"],
            },
            {
                "constraint_id": "resource_b",
                "coefficients": {"x": 1, "y": 3},
                "sense": "<=",
                "rhs": 9,
                "lhs_unit": "resource_b_unit",
                "rhs_unit": "resource_b_unit",
                "contract_clause_ids": ["resource_b_limit"],
            },
        ],
        validation_obligations=[
            "Recompute every constraint from source IR",
            "Confirm global optimum by bounded integer enumeration",
            "Replay from serialized contract and IR in a fresh process",
        ],
    )


def submitted_solution(
    ir: OptimizationModelIR,
    *,
    values: dict[str, float],
    objective_value: float,
    matrix_hash: str | None = None,
    execution_hash: str | None = None,
    message: str = "fault-injection submission",
) -> SolutionArtifact:
    compiled = compile_optimization(ir)
    return SolutionArtifact(
        solver="external_submission/fixture",
        solver_status="optimal",
        source_ir_hash=ir.ir_hash,
        compiled_matrix_hash=matrix_hash or compiled.certificate.matrix_hash,
        execution_array_hash=execution_hash or compiled.certificate.execution_array_hash,
        values=values,
        objective_value=objective_value,
        message=message,
    )


def dropped_constraint_certificate(ir: OptimizationModelIR) -> CompilerCertificate:
    compiled = compile_optimization(ir)
    correct = compiled.certificate
    rows = correct.constraint_rows[:-1]
    payload = {
        "variable_order": correct.variable_order,
        "objective_vector": correct.objective_vector,
        "integrality": correct.integrality,
        "variable_bounds": correct.variable_bounds,
        "constraint_rows": rows,
    }
    execution_payload = execution_array_payload(compiled)
    execution_payload["matrix"] = execution_payload["matrix"][:-1]
    execution_payload["constraint_lower"] = execution_payload["constraint_lower"][:-1]
    execution_payload["constraint_upper"] = execution_payload["constraint_upper"][:-1]
    return CompilerCertificate(
        source_ir_hash=correct.source_ir_hash,
        variable_order=correct.variable_order,
        objective_vector=correct.objective_vector,
        integrality=correct.integrality,
        variable_bounds=correct.variable_bounds,
        constraint_rows=rows,
        matrix_hash=sha256_value(payload),
        execution_array_hash=sha256_value(execution_payload),
    )
