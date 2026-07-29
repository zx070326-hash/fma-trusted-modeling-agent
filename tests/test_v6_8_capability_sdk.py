from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from fma.hashing import sha256_value
from fma.v6.capability_sdk_v68 import (
    BaselineContractV68,
    BenchmarkContractV68,
    CapabilityManifestV68,
    CapabilityQueryV68,
    CapabilityRegistryV68,
    LevelObligationV68,
    MeasurementApplicabilityV68,
    MeasurementSignatureV68,
    RecoveryContractV68,
    ResourceEnvelopeV68,
    SemanticImplementationBindingV68,
    TypedModelIRContractV68,
    evaluate_capability_conformance_v68,
    skeleton_subsumption_hash_v68,
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
from fma.v6.recovery_kernel import CapabilityPackV60, ProblemSignatureV60


def _digest(label: str) -> str:
    return sha256_value({"test_identity": label})


def _binding(label: str) -> SemanticImplementationBindingV68:
    return SemanticImplementationBindingV68(
        implementation_id=label,
        implementation_version="v1",
        entrypoint_ref=f"tests.capabilities.{label}",
        semantic_hash=_digest(f"implementation:{label}"),
    )


def _legacy_pack(
    label: str,
    *,
    minimum_observations: int = 12,
) -> CapabilityPackV60:
    return CapabilityPackV60.seal(
        pack_id=f"pack_{label}",
        pack_version="v1",
        state_kinds=["scalar"],
        time_kinds=["continuous"],
        dynamics_kinds=["autonomous"],
        observation_kinds=["complete"],
        minimum_observations=minimum_observations,
        requires_positive_observations=True,
        requires_strictly_increasing_time=True,
        executor_id=f"executor_{label}",
        scientific_adapter_ids=[f"scientific_adapter_{label}"],
        baseline_ids=["persistence"],
        supported_levels=["L0", "L1", "L2", "L3", "L4"],
    )


def _manifest(
    label: str,
    *,
    skeleton_atoms: list[str],
    maturity: str = "development_sandbox",
    minimum_observations: int = 12,
) -> CapabilityManifestV68:
    atoms = sorted(skeleton_atoms)
    return CapabilityManifestV68.seal(
        manifest_id=f"manifest_{label}",
        capability_pack=_legacy_pack(
            label,
            minimum_observations=minimum_observations,
        ),
        supported_claim_kinds=["predictive"],
        claim_ceilings={"predictive": "local_predictive_evidence_only"},
        supported_task_kinds=["prediction"],
        measurement_applicability=MeasurementApplicabilityV68(
            scale_types=["ratio"],
            study_design_types=["time_series"],
            missingness_policies=["reject_incomplete_series"],
            minimum_planned_observations=minimum_observations,
        ),
        maturity=maturity,
        stage_workflow_promotion_receipt_hash=(
            _digest(f"promotion:{label}")
            if maturity == "stage_workflow"
            else None
        ),
        typed_ir=TypedModelIRContractV68(
            ir_kind=f"ir_{label}",
            ir_schema_version="v1",
            ir_schema_hash=_digest(f"ir:{label}"),
            compiler_input_schema_hash=_digest(f"compiler-input:{label}"),
            compiler_output_schema_hash=_digest(f"compiler-output:{label}"),
        ),
        compiler=_binding(f"compiler_{label}"),
        executor=_binding(f"executor_{label}"),
        level_obligations=[
            LevelObligationV68(
                level=level,
                obligation_id=f"{label}_{level.lower()}",
                verifier=_binding(f"verifier_{label}_{level.lower()}"),
                evidence_kind=f"evidence_{label}_{level.lower()}",
            )
            for level in ("L0", "L1", "L2", "L3", "L4")
        ],
        resources=ResourceEnvelopeV68(
            max_wall_seconds=120,
            max_cpu_seconds=120,
            max_memory_megabytes=512,
            max_artifact_bytes=100_000,
            max_model_calls=4,
            max_tool_calls=12,
        ),
        baselines=BaselineContractV68(
            baseline_ids=["persistence"],
            supported_common_loss_ids=["one_step_rmse"],
        ),
        recovery=RecoveryContractV68(
            allowed_actions=["ABSTAIN", "BRANCH", "PATCH", "RETRY"],
            max_graph_attempts=3,
            max_same_attempt_retries=1,
        ),
        benchmark=BenchmarkContractV68(
            benchmark_suite_id=f"benchmark_{label}",
            benchmark_suite_version="v1",
            benchmark_suite_hash=_digest(f"benchmark:{label}"),
            minimum_public_cases=10,
            minimum_adversarial_cases=3,
            external_private_evaluation_required_for_stage_workflow=True,
        ),
        skeleton_atoms=atoms,
        skeleton_subsumption_hash=skeleton_subsumption_hash_v68(atoms),
    )


def _query(
    *,
    claim_kind: str = "predictive",
    task_kind: str = "prediction",
    scale_type: str = "ratio",
    observations: int = 40,
) -> CapabilityQueryV68:
    return CapabilityQueryV68.seal(
        workspace_spec_hash="a" * 64,
        s0_gate_hash="b" * 64,
        problem_signature=ProblemSignatureV60(
            state_kind="scalar",
            time_kind="continuous",
            dynamics_kind="autonomous",
            observation_kind="complete",
            task_kind=task_kind,
            observation_count=observations,
            positive_observations=True,
            strictly_increasing_time=True,
        ),
        claim_kind=claim_kind,
        measurement=MeasurementSignatureV68(
            measurement_contract_hash="c" * 64,
            scale_type=scale_type,
            study_design_type="time_series",
            missingness_policy="reject_incomplete_series",
            measurement_unit="registered unit",
            time_basis="calendar year",
            minimum_planned_observations=observations,
        ),
    )


def _branch_budget(
    *,
    cpu_seconds: int = 30,
    memory_megabytes: int = 128,
) -> BranchBudgetV68:
    return BranchBudgetV68.seal(
        max_wall_seconds=30,
        max_cpu_seconds=cpu_seconds,
        max_memory_megabytes=memory_megabytes,
        max_artifact_bytes=10_000,
        max_model_calls=1,
        max_tool_calls=2,
    )


def _portfolio_budget(
    *,
    total_cpu_seconds: int = 100,
) -> PortfolioBudgetV68:
    return PortfolioBudgetV68.seal(
        max_parallel_branches=2,
        total_wall_seconds=60,
        total_cpu_seconds=total_cpu_seconds,
        total_memory_megabytes=512,
        total_artifact_bytes=30_000,
        total_model_calls=4,
        total_tool_calls=8,
    )


def _common_loss() -> CommonLossContractV68:
    return CommonLossContractV68.seal(
        loss_id="one_step_rmse",
        loss_implementation_ref="fma.v68.common_loss.one_step_rmse",
        loss_semantic_hash=one_step_rmse_semantic_hash_v68(),
        direction="minimize",
        loss_unit="registered unit",
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
        tie_tolerance=1e-9,
        minimum_completed_branches=2,
    )


def _request(
    branch_id: str,
    manifest: CapabilityManifestV68,
    *,
    budget: BranchBudgetV68 | None = None,
) -> PortfolioBranchRequestV68:
    assert manifest.manifest_hash is not None
    return PortfolioBranchRequestV68(
        branch_id=branch_id,
        manifest_id=manifest.manifest_id,
        manifest_hash=manifest.manifest_hash,
        budget=budget or _branch_budget(),
    )


def test_manifest_composes_sealed_v60_pack_and_conformance_cannot_promote() -> None:
    manifest = _manifest(
        "ode",
        skeleton_atoms=["deterministic_flow", "observed_state"],
    )
    manifest.assert_sealed()
    manifest.capability_pack.assert_sealed()
    assert manifest.capability_pack.schema_version == "6.0"
    assert manifest.schema_version == "6.8-capability-manifest"
    assert manifest.claim_ceilings == {
        "predictive": "local_predictive_evidence_only"
    }
    assert [item.level for item in manifest.level_obligations] == [
        "L0",
        "L1",
        "L2",
        "L3",
        "L4",
    ]
    assert manifest.manifest_is_scientific_evidence is False
    assert manifest.scientific_qualification_granted is False
    assert manifest.real_world_action_authorized is False

    report = evaluate_capability_conformance_v68(
        manifest,
        observed_semantic_hashes=manifest.expected_semantic_hashes(),
    )
    report.assert_sealed()
    assert report.status == "PASS"
    assert len(report.checks) == 11
    assert report.expected_semantic_hashes_hash == sha256_value(
        manifest.expected_semantic_hashes()
    )
    assert report.declared_maturity == "development_sandbox"
    assert report.maturity_after_report == "development_sandbox"
    assert report.maturity_promotion_granted is False
    assert report.report_is_scientific_evidence is False

    mismatched = dict(manifest.expected_semantic_hashes())
    mismatched["compiler"] = "f" * 64
    failed = evaluate_capability_conformance_v68(
        manifest,
        observed_semantic_hashes=mismatched,
    )
    assert failed.status == "FAIL"
    assert failed.maturity_after_report == "development_sandbox"
    assert failed.maturity_promotion_granted is False

    duplicate_payload = report.model_dump(
        mode="json",
        exclude={"checks", "report_hash"},
    )
    duplicate_checks = list(report.checks)
    duplicate_checks[-1] = duplicate_checks[-2]
    with pytest.raises(ValueError, match="sorted and unique"):
        type(report).seal(
            **duplicate_payload,
            checks=sorted(duplicate_checks, key=lambda item: item.check_id),
        )

    forged_payload = report.model_dump(
        mode="json",
        exclude={"expected_semantic_hashes_hash", "report_hash"},
    )
    forged_report = type(report).seal(
        **forged_payload,
        expected_semantic_hashes_hash="e" * 64,
    )
    registry = CapabilityRegistryV68(runtime_mode="development_sandbox")
    with pytest.raises(ValueError, match="bound to another manifest"):
        registry.register(manifest, conformance_report=forged_report)

    unsealed_pack = manifest.capability_pack.model_copy(
        update={"pack_hash": None}
    )
    payload = manifest.model_dump(
        mode="json",
        exclude={"capability_pack", "manifest_hash"},
    )
    with pytest.raises(ValueError, match="capability pack is not sealed"):
        CapabilityManifestV68.seal(
            **payload,
            capability_pack=unsealed_pack,
        )


def test_registry_exact_route_and_stage_admission_are_separate() -> None:
    manifest = _manifest(
        "ode",
        skeleton_atoms=["deterministic_flow", "observed_state"],
    )
    development = CapabilityRegistryV68(
        runtime_mode="development_sandbox"
    )
    development.register(manifest)
    assert development.lookup_exact(
        manifest.manifest_id,
        str(manifest.manifest_hash),
    ) == manifest
    with pytest.raises(KeyError, match="hash mismatch"):
        development.lookup_exact(manifest.manifest_id, "0" * 64)

    decision = development.route(_query())
    assert decision.status == "ROUTABLE"
    assert decision.compatible_manifest_ids == [manifest.manifest_id]
    assert decision.decision_is_scientific_evidence is False
    assert decision.scientific_qualification_granted is False
    assert development.snapshot().scientific_qualification_granted is False

    incompatible = development.route(
        _query(
            claim_kind="mechanistic",
            task_kind="control",
            scale_type="ordinal",
        )
    )
    assert incompatible.status == "CAPABILITY_GAP"
    reasons = incompatible.incompatibilities[manifest.manifest_id]
    assert "claim_kind:mechanistic" in reasons
    assert "task_kind:control" in reasons
    assert "measurement.scale_type:ordinal" in reasons

    with pytest.raises(PermissionError, match="registry is disabled"):
        CapabilityRegistryV68(
            runtime_mode="stage_workflow",
            admitted_stage_manifest_hashes={
                manifest.manifest_id: str(manifest.manifest_hash)
            },
        )

    with pytest.raises(ValueError, match="promotion authority is NOT_RUN"):
        _manifest(
            "qualified_ode",
            skeleton_atoms=["deterministic_flow", "observed_state"],
            maturity="stage_workflow",
        )


def test_registry_snapshot_and_route_are_registration_order_independent() -> None:
    first = _manifest(
        "ode",
        skeleton_atoms=["deterministic_flow", "observed_state"],
    )
    second = _manifest(
        "state_space",
        skeleton_atoms=["latent_state", "linear_gaussian"],
    )
    left = CapabilityRegistryV68(runtime_mode="development_sandbox")
    right = CapabilityRegistryV68(runtime_mode="development_sandbox")
    for registry, manifests in (
        (left, [first, second]),
        (right, [second, first]),
    ):
        for manifest in manifests:
            registry.register(manifest)
    assert left.snapshot() == right.snapshot()
    assert left.route(_query()) == right.route(_query())


def test_portfolio_is_deterministic_observation_free_and_replayable() -> None:
    ode = _manifest(
        "ode",
        skeleton_atoms=["deterministic_flow", "observed_state"],
    )
    state_space = _manifest(
        "state_space",
        skeleton_atoms=["latent_state", "linear_gaussian"],
    )
    registry = CapabilityRegistryV68(runtime_mode="development_sandbox")
    registry.register(state_space)
    registry.register(ode)
    query = _query()
    budget = _portfolio_budget()
    common_loss = _common_loss()
    outer = _outer_selection()
    requests = [
        _request("branch_state_space", state_space),
        _request("branch_ode", ode),
    ]

    first = compile_modeling_portfolio_protocol_v68(
        query=query,
        registry=registry,
        branch_requests=requests,
        budget=budget,
        common_loss=common_loss,
        outer_selection=outer,
    )
    second = compile_modeling_portfolio_protocol_v68(
        query=query,
        registry=registry,
        branch_requests=list(reversed(requests)),
        budget=budget,
        common_loss=common_loss,
        outer_selection=outer,
    )
    first.assert_sealed()
    assert first == second
    assert [item.branch_id for item in first.branches] == [
        "branch_ode",
        "branch_state_space",
    ]
    assert first.observation_values_accessed_during_compilation is False
    assert first.observed_statistics_accessed_during_compilation is False
    assert first.private_acceptance_data_accessed is False
    assert first.protocol_is_scientific_evidence is False
    assert first.scientific_qualification_granted is False
    assert first.real_world_action_authorized is False
    assert all(
        item.scientific_qualification_granted is False
        and item.real_world_action_authorized is False
        for item in first.branches
    )
    assert verify_modeling_portfolio_protocol_v68(
        query=query,
        registry=registry,
        branch_requests=requests,
        budget=budget,
        common_loss=common_loss,
        outer_selection=outer,
        protocol=first,
    )
    compile_parameters = set(
        inspect.signature(
            compile_modeling_portfolio_protocol_v68
        ).parameters
    )
    assert compile_parameters == {
        "query",
        "registry",
        "branch_requests",
        "budget",
        "common_loss",
        "outer_selection",
    }


@pytest.mark.parametrize(
    ("left_atoms", "right_atoms", "message"),
    [
        (
            ["deterministic_flow", "observed_state"],
            ["deterministic_flow", "observed_state"],
            "duplicate mathematical skeletons",
        ),
        (
            ["deterministic_flow"],
            ["deterministic_flow", "observed_state"],
            "subsumed by another branch",
        ),
    ],
)
def test_portfolio_rejects_duplicate_or_subsuming_skeletons(
    left_atoms: list[str],
    right_atoms: list[str],
    message: str,
) -> None:
    left = _manifest("left", skeleton_atoms=left_atoms)
    right = _manifest("right", skeleton_atoms=right_atoms)
    registry = CapabilityRegistryV68(runtime_mode="development_sandbox")
    registry.register(left)
    registry.register(right)
    with pytest.raises(ValueError, match=message):
        compile_modeling_portfolio_protocol_v68(
            query=_query(),
            registry=registry,
            branch_requests=[
                _request("branch_left", left),
                _request("branch_right", right),
            ],
            budget=_portfolio_budget(),
            common_loss=_common_loss(),
            outer_selection=_outer_selection(),
        )


def test_portfolio_fails_closed_on_count_resource_and_total_budget() -> None:
    left = _manifest("left", skeleton_atoms=["atom_left"])
    right = _manifest("right", skeleton_atoms=["atom_right"])
    registry = CapabilityRegistryV68(runtime_mode="development_sandbox")
    registry.register(left)
    registry.register(right)

    with pytest.raises(ValueError, match="between two and eight"):
        compile_modeling_portfolio_protocol_v68(
            query=_query(),
            registry=registry,
            branch_requests=[_request("branch_left", left)],
            budget=_portfolio_budget(),
            common_loss=_common_loss(),
            outer_selection=_outer_selection(),
        )

    with pytest.raises(ValueError, match="capability max_cpu_seconds"):
        compile_modeling_portfolio_protocol_v68(
            query=_query(),
            registry=registry,
            branch_requests=[
                _request(
                    "branch_left",
                    left,
                    budget=_branch_budget(cpu_seconds=121),
                ),
                _request("branch_right", right),
            ],
            budget=_portfolio_budget(total_cpu_seconds=200),
            common_loss=_common_loss(),
            outer_selection=_outer_selection(),
        )

    with pytest.raises(ValueError, match="total_cpu_seconds"):
        compile_modeling_portfolio_protocol_v68(
            query=_query(),
            registry=registry,
            branch_requests=[
                _request("branch_left", left),
                _request("branch_right", right),
            ],
            budget=_portfolio_budget(total_cpu_seconds=59),
            common_loss=_common_loss(),
            outer_selection=_outer_selection(),
        )

    serial_budget = PortfolioBudgetV68.seal(
        max_parallel_branches=1,
        total_wall_seconds=30,
        total_cpu_seconds=100,
        total_memory_megabytes=512,
        total_artifact_bytes=30_000,
        total_model_calls=4,
        total_tool_calls=8,
    )
    with pytest.raises(ValueError, match="schedule can exceed"):
        compile_modeling_portfolio_protocol_v68(
            query=_query(),
            registry=registry,
            branch_requests=[
                _request("branch_left", left),
                _request("branch_right", right),
            ],
            budget=serial_budget,
            common_loss=_common_loss(),
            outer_selection=_outer_selection(),
        )


def test_portfolio_loss_and_selector_are_code_owned() -> None:
    loss_payload = _common_loss().model_dump(
        mode="json",
        exclude={"loss_semantic_hash", "contract_hash"},
    )
    with pytest.raises(ValueError, match="code-owned registry"):
        CommonLossContractV68.seal(
            **loss_payload,
            loss_semantic_hash="f" * 64,
        )

    selection_payload = _outer_selection().model_dump(
        mode="json",
        exclude={"implementation_semantic_hash", "policy_hash"},
    )
    with pytest.raises(ValueError, match="code-owned registry"):
        OuterSelectionPolicyV68.seal(
            **selection_payload,
            implementation_semantic_hash="e" * 64,
        )


def test_portfolio_rejects_incompatible_pack_and_hash_tampering() -> None:
    left = _manifest("left", skeleton_atoms=["atom_left"])
    right = _manifest(
        "right",
        skeleton_atoms=["atom_right"],
        minimum_observations=50,
    )
    registry = CapabilityRegistryV68(runtime_mode="development_sandbox")
    registry.register(left)
    registry.register(right)
    with pytest.raises(ValueError, match="incompatible"):
        compile_modeling_portfolio_protocol_v68(
            query=_query(observations=40),
            registry=registry,
            branch_requests=[
                _request("branch_left", left),
                _request("branch_right", right),
            ],
            budget=_portfolio_budget(),
            common_loss=_common_loss(),
            outer_selection=_outer_selection(),
        )

    usable_right = _manifest("usable_right", skeleton_atoms=["atom_right"])
    usable_registry = CapabilityRegistryV68(
        runtime_mode="development_sandbox"
    )
    usable_registry.register(left)
    usable_registry.register(usable_right)
    protocol = compile_modeling_portfolio_protocol_v68(
        query=_query(),
        registry=usable_registry,
        branch_requests=[
            _request("branch_left", left),
            _request("branch_right", usable_right),
        ],
        budget=_portfolio_budget(),
        common_loss=_common_loss(),
        outer_selection=_outer_selection(),
    )
    tampered = protocol.model_dump(mode="json")
    tampered["protocol_id"] = "portfolio-tampered"
    with pytest.raises(ValidationError, match="protocol hash differs"):
        type(protocol).model_validate(tampered)
