from __future__ import annotations

import pytest

from fma.v6.capability_catalog_v68 import (
    POSITIVE_LOG_INCREMENT_MANIFEST_ID_V68,
    SCALAR_ODE_MANIFEST_ID_V68,
    conform_capability_manifest_v68,
    default_development_capability_registry_v68,
    positive_log_increment_manifest_v68,
    scalar_autonomous_ode_manifest_v68,
)
from fma.v6.capability_sdk_v68 import (
    CapabilityQueryV68,
    CapabilityRegistryV68,
    MeasurementSignatureV68,
)
from fma.v6.portfolio_protocol_v68 import (
    BranchBudgetV68,
    CommonLossContractV68,
    OuterSelectionPolicyV68,
    PortfolioBranchRequestV68,
    PortfolioBudgetV68,
    compile_modeling_portfolio_protocol_v68,
    one_step_rmse_semantic_hash_v68,
    outer_selector_semantic_hash_v68,
    verify_modeling_portfolio_protocol_v68,
)
from fma.v6.recovery_kernel import ProblemSignatureV60


def _query() -> CapabilityQueryV68:
    return CapabilityQueryV68.seal(
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
            measurement_unit="registered positive state",
            time_basis="calendar year",
            minimum_planned_observations=40,
        ),
    )


def _branch_budget() -> BranchBudgetV68:
    return BranchBudgetV68.seal(
        max_wall_seconds=120,
        max_cpu_seconds=120,
        max_memory_megabytes=512,
        max_artifact_bytes=2_000_000,
        max_model_calls=0,
        max_tool_calls=2,
    )


def _portfolio_budget() -> PortfolioBudgetV68:
    return PortfolioBudgetV68.seal(
        max_parallel_branches=2,
        total_wall_seconds=240,
        total_cpu_seconds=240,
        total_memory_megabytes=1024,
        total_artifact_bytes=4_000_000,
        total_model_calls=0,
        total_tool_calls=4,
    )


def _common_loss() -> CommonLossContractV68:
    return CommonLossContractV68.seal(
        loss_id="one_step_rmse",
        loss_implementation_ref="fma.v68.common_loss.one_step_rmse",
        loss_semantic_hash=one_step_rmse_semantic_hash_v68(),
        direction="minimize",
        loss_unit="registered positive state",
        common_data_view_rule=(
            "Every branch receives the same content-addressed ordered series."
        ),
        common_evaluation_origin_rule=(
            "Every branch is scored on the same frozen outer rolling origins."
        ),
    )


def _outer_selection() -> OuterSelectionPolicyV68:
    return OuterSelectionPolicyV68.seal(
        policy_id="nested_one_step_rmse_v68",
        implementation_ref="fma.v68.selection.nested_one_step_rmse",
        implementation_semantic_hash=outer_selector_semantic_hash_v68(),
        tie_tolerance=1e-12,
        minimum_completed_branches=2,
    )


def test_real_catalog_manifests_are_conformant_but_development_only() -> None:
    ode = scalar_autonomous_ode_manifest_v68()
    log_increment = positive_log_increment_manifest_v68()

    for manifest in (ode, log_increment):
        manifest.assert_sealed()
        report = conform_capability_manifest_v68(manifest)
        report.assert_sealed()
        assert report.status == "PASS"
        assert report.maturity_after_report == "development_sandbox"
        assert report.maturity_promotion_granted is False
        assert report.report_is_scientific_evidence is False

    assert ode.capability_pack.pack_id == "scalar_autonomous_ode_v52"
    assert log_increment.capability_pack.pack_id == "positive_log_increment_v68"
    assert log_increment.claim_ceilings == {
        "predictive": "local_predictive_evidence_only"
    }
    assert log_increment.capability_pack.minimum_observations == 34
    assert log_increment.skeleton_atoms == [
        "log_growth_ar1",
        "log_random_walk_drift",
        "stochastic_log_increment",
    ]
    assert not {
        "constant",
        "exponential",
        "gompertz",
        "logistic",
    } & set(log_increment.skeleton_atoms)

    altered_typed_ir = log_increment.typed_ir.model_copy(
        update={"compiler_input_schema_hash": "f" * 64}
    )
    payload = log_increment.model_dump(
        mode="json",
        exclude={"typed_ir", "manifest_hash"},
    )
    altered_manifest = type(log_increment).seal(
        **payload,
        typed_ir=altered_typed_ir,
    )
    with pytest.raises(
        KeyError,
        match="runtime definition manifest hash mismatch",
    ):
        conform_capability_manifest_v68(altered_manifest)


def test_real_catalog_routes_two_non_nested_packs_and_blocks_stage_use() -> None:
    registry = default_development_capability_registry_v68()
    decision = registry.route(_query())
    assert decision.status == "ROUTABLE"
    assert decision.compatible_manifest_ids == [
        POSITIVE_LOG_INCREMENT_MANIFEST_ID_V68,
        SCALAR_ODE_MANIFEST_ID_V68,
    ]

    log_increment = positive_log_increment_manifest_v68()
    with pytest.raises(PermissionError, match="registry is disabled"):
        CapabilityRegistryV68(
            runtime_mode="stage_workflow",
            admitted_stage_manifest_hashes={
                log_increment.manifest_id: str(log_increment.manifest_hash)
            },
        )


def test_real_catalog_compiles_replayable_observation_free_portfolio() -> None:
    registry = default_development_capability_registry_v68()
    query = _query()
    ode = scalar_autonomous_ode_manifest_v68()
    log_increment = positive_log_increment_manifest_v68()
    requests = [
        PortfolioBranchRequestV68(
            branch_id="branch_log_increment",
            manifest_id=log_increment.manifest_id,
            manifest_hash=str(log_increment.manifest_hash),
            budget=_branch_budget(),
        ),
        PortfolioBranchRequestV68(
            branch_id="branch_ode",
            manifest_id=ode.manifest_id,
            manifest_hash=str(ode.manifest_hash),
            budget=_branch_budget(),
        ),
    ]
    budget = _portfolio_budget()
    common_loss = _common_loss()
    outer = _outer_selection()

    protocol = compile_modeling_portfolio_protocol_v68(
        query=query,
        registry=registry,
        branch_requests=list(reversed(requests)),
        budget=budget,
        common_loss=common_loss,
        outer_selection=outer,
    )
    repeated = compile_modeling_portfolio_protocol_v68(
        query=query,
        registry=registry,
        branch_requests=requests,
        budget=budget,
        common_loss=common_loss,
        outer_selection=outer,
    )

    protocol.assert_sealed()
    assert protocol == repeated
    assert [item.branch_id for item in protocol.branches] == [
        "branch_log_increment",
        "branch_ode",
    ]
    assert protocol.observation_values_accessed_during_compilation is False
    assert protocol.observed_statistics_accessed_during_compilation is False
    assert protocol.protocol_is_scientific_evidence is False
    assert protocol.scientific_qualification_granted is False
    assert protocol.real_world_action_authorized is False
    assert verify_modeling_portfolio_protocol_v68(
        query=query,
        registry=registry,
        branch_requests=requests,
        budget=budget,
        common_loss=common_loss,
        outer_selection=outer,
        protocol=protocol,
    )
