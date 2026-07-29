from __future__ import annotations

import math

import pytest

from fma.hashing import sha256_value
from fma.v6.capability_catalog_v68 import (
    default_development_capability_registry_v68,
    positive_log_increment_manifest_v68,
    scalar_autonomous_ode_manifest_v68,
)
from fma.v6.capability_sdk_v68 import (
    CapabilityQueryV68,
    MeasurementSignatureV68,
)
from fma.v6.portfolio_protocol_v68 import (
    BranchBudgetV68,
    BranchOuterEvaluationV68,
    CommonLossContractV68,
    OuterSelectionPolicyV68,
    PortfolioBranchRequestV68,
    PortfolioBudgetV68,
    compile_modeling_portfolio_protocol_v68,
    execute_outer_selection_v68,
    one_step_rmse_semantic_hash_v68,
    one_step_rmse_v68,
    outer_selector_semantic_hash_v68,
    verify_outer_selection_v68,
)
from fma.v6.recovery_kernel import ProblemSignatureV60


def _protocol():
    registry = default_development_capability_registry_v68()
    log_pack = positive_log_increment_manifest_v68()
    ode_pack = scalar_autonomous_ode_manifest_v68()
    query = CapabilityQueryV68.seal(
        workspace_spec_hash="a" * 64,
        s0_gate_hash="b" * 64,
        problem_signature=ProblemSignatureV60(
            state_kind="scalar",
            time_kind="continuous",
            dynamics_kind="autonomous",
            observation_kind="complete",
            task_kind="prediction",
            observation_count=40,
            positive_observations=True,
            strictly_increasing_time=True,
        ),
        claim_kind="predictive",
        measurement=MeasurementSignatureV68(
            measurement_contract_hash="c" * 64,
            scale_type="ratio",
            study_design_type="time_series",
            missingness_policy="reject_incomplete_series",
            measurement_unit="registered_positive_state",
            time_basis="day",
            minimum_planned_observations=40,
        ),
    )
    branch_budget = BranchBudgetV68.seal(
        max_wall_seconds=60,
        max_cpu_seconds=60,
        max_memory_megabytes=512,
        max_artifact_bytes=2_000_000,
        max_model_calls=0,
        max_tool_calls=2,
    )
    return compile_modeling_portfolio_protocol_v68(
        query=query,
        registry=registry,
        branch_requests=[
            PortfolioBranchRequestV68(
                branch_id="branch_log",
                manifest_id=log_pack.manifest_id,
                manifest_hash=str(log_pack.manifest_hash),
                budget=branch_budget,
            ),
            PortfolioBranchRequestV68(
                branch_id="branch_ode",
                manifest_id=ode_pack.manifest_id,
                manifest_hash=str(ode_pack.manifest_hash),
                budget=branch_budget,
            ),
        ],
        budget=PortfolioBudgetV68.seal(
            max_parallel_branches=2,
            total_wall_seconds=120,
            total_cpu_seconds=120,
            total_memory_megabytes=1024,
            total_artifact_bytes=4_000_000,
            total_model_calls=0,
            total_tool_calls=4,
        ),
        common_loss=CommonLossContractV68.seal(
            loss_id="one_step_rmse",
            loss_implementation_ref="fma.v68.common_loss.one_step_rmse",
            loss_semantic_hash=one_step_rmse_semantic_hash_v68(),
            direction="minimize",
            loss_unit="registered_positive_state",
            common_data_view_rule=(
                "Every branch receives the same content-addressed ordered series."
            ),
            common_evaluation_origin_rule=(
                "Every branch is scored on the same frozen outer rolling origins."
            ),
        ),
        outer_selection=OuterSelectionPolicyV68.seal(
            policy_id="nested_one_step_rmse_v68",
            implementation_ref="fma.v68.selection.nested_one_step_rmse",
            implementation_semantic_hash=outer_selector_semantic_hash_v68(),
            tie_tolerance=1e-12,
            minimum_completed_branches=2,
        ),
    )


def _evaluation(
    protocol,
    branch_id: str,
    *,
    predictions: list[float] | None,
    parameter_count: int,
    status: str = "PASS",
    origins: list[str] | None = None,
) -> BranchOuterEvaluationV68:
    branch = next(item for item in protocol.branches if item.branch_id == branch_id)
    return BranchOuterEvaluationV68.seal(
        protocol_hash=str(protocol.protocol_hash),
        branch_id=branch_id,
        manifest_id=branch.manifest_id,
        manifest_hash=branch.manifest_hash,
        execution_evidence_hash=sha256_value(
            {"branch_id": branch_id, "status": status}
        ),
        outer_origin_ids=origins or ["origin_1", "origin_2", "origin_3"],
        observations=[10.0, 12.0, 11.0],
        predictions=predictions or [],
        parameter_count=parameter_count,
        execution_status=status,
        failure_reason=None if status == "PASS" else "branch_not_eligible",
    )


def test_code_owned_rmse_and_unique_outer_winner_replay() -> None:
    protocol = _protocol()
    evaluations = [
        _evaluation(
            protocol,
            "branch_log",
            predictions=[10.1, 12.2, 10.9],
            parameter_count=2,
        ),
        _evaluation(
            protocol,
            "branch_ode",
            predictions=[9.0, 13.0, 10.0],
            parameter_count=1,
        ),
    ]
    decision = execute_outer_selection_v68(
        protocol=protocol,
        evaluations=evaluations,
    )

    assert math.isclose(
        one_step_rmse_v68([10.0, 12.0], [11.0, 11.0]),
        1.0,
    )
    assert decision.decision == "SELECT"
    assert decision.selected_branch_id == "branch_log"
    assert decision.scientific_qualification_granted is False
    assert decision.real_world_action_authorized is False
    assert verify_outer_selection_v68(
        protocol=protocol,
        evaluations=evaluations,
        decision=decision,
    )


def test_outer_selector_uses_parsimony_and_abstains_on_unresolved_tie() -> None:
    protocol = _protocol()
    common_predictions = [10.1, 12.1, 10.9]
    parsimony = execute_outer_selection_v68(
        protocol=protocol,
        evaluations=[
            _evaluation(
                protocol,
                "branch_log",
                predictions=common_predictions,
                parameter_count=2,
            ),
            _evaluation(
                protocol,
                "branch_ode",
                predictions=common_predictions,
                parameter_count=1,
            ),
        ],
    )
    assert parsimony.selected_branch_id == "branch_ode"

    tied = execute_outer_selection_v68(
        protocol=protocol,
        evaluations=[
            _evaluation(
                protocol,
                "branch_log",
                predictions=common_predictions,
                parameter_count=2,
            ),
            _evaluation(
                protocol,
                "branch_ode",
                predictions=common_predictions,
                parameter_count=2,
            ),
        ],
    )
    assert tied.decision == "ABSTAIN"
    assert tied.selected_branch_id is None
    assert tied.reason_code == "unresolved_loss_parsimony_tie"


def test_outer_selector_retains_failure_and_rejects_mismatched_views() -> None:
    protocol = _protocol()
    partial = execute_outer_selection_v68(
        protocol=protocol,
        evaluations=[
            _evaluation(
                protocol,
                "branch_log",
                predictions=[10.1, 12.2, 10.9],
                parameter_count=2,
            ),
            _evaluation(
                protocol,
                "branch_ode",
                predictions=None,
                parameter_count=1,
                status="FAIL",
            ),
        ],
    )
    assert partial.decision == "ABSTAIN"
    assert partial.reason_code == "insufficient_completed_branches"
    assert partial.rejected_branches == {
        "branch_ode": "branch_not_eligible"
    }

    with pytest.raises(ValueError, match="different outer data views"):
        execute_outer_selection_v68(
            protocol=protocol,
            evaluations=[
                _evaluation(
                    protocol,
                    "branch_log",
                    predictions=[10.1, 12.2, 10.9],
                    parameter_count=2,
                ),
                _evaluation(
                    protocol,
                    "branch_ode",
                    predictions=[9.0, 13.0, 10.0],
                    parameter_count=1,
                    origins=["origin_1", "origin_2", "origin_other"],
                ),
            ],
        )
