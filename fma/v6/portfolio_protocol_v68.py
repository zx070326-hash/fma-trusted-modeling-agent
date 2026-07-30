"""Observation-free V6.8 multi-capability portfolio protocol.

This module freezes *which* capability manifests may run, their per-branch and
shared budgets, one common loss, and the outer selection rule.  It does not
execute a model, inspect observations, sign a V5 gate, or grant scientific
qualification.  V6.7 single-adapter protocols remain unchanged.
"""

from __future__ import annotations

import math
import marshal
from collections.abc import Callable, Sequence
from typing import Annotated, Literal

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256

from .capability_sdk_v68 import (
    CapabilityManifestV68,
    CapabilityQueryV68,
    CapabilityRegistryV68,
    CapabilityRuntimeModeV68,
    ResourceEnvelopeV68,
    skeleton_subsumption_hash_v68,
)


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _sealed(model_type, data: dict[str, object], hash_field: str):
    draft = model_type(**data)
    payload = draft.model_dump(mode="json", exclude={hash_field})
    payload[hash_field] = draft.content_hash()
    return model_type(**payload)


def _callable_semantic_hash_v68(
    callable_object: Callable[..., object],
    *,
    semantic_contract: dict[str, object],
) -> str:
    code = getattr(callable_object, "__code__", None)
    if code is None:
        raise ValueError("V6.8 portfolio callable has no code identity")
    return sha256_value(
        {
            "module": callable_object.__module__,
            "qualname": callable_object.__qualname__,
            "marshalled_code": marshal.dumps(code).hex(),
            "semantic_contract": semantic_contract,
        }
    )


def one_step_rmse_v68(
    observations: Sequence[float],
    predictions: Sequence[float],
) -> float:
    """Compute the code-owned common one-step RMSE."""

    if len(observations) == 0 or len(observations) != len(predictions):
        raise ValueError("one-step RMSE requires equal non-empty sequences")
    pairs = [(float(actual), float(predicted)) for actual, predicted in zip(
        observations,
        predictions,
        strict=True,
    )]
    if any(
        not math.isfinite(actual) or not math.isfinite(predicted)
        for actual, predicted in pairs
    ):
        raise ValueError("one-step RMSE requires finite values")
    return math.sqrt(
        sum((predicted - actual) ** 2 for actual, predicted in pairs)
        / len(pairs)
    )


def one_step_rmse_semantic_hash_v68() -> str:
    return _callable_semantic_hash_v68(
        one_step_rmse_v68,
        semantic_contract={
            "loss": "root_mean_squared_error",
            "horizon_steps": 1,
            "direction": "minimize",
            "version": "v6.8",
        },
    )


def outer_selector_semantic_hash_v68() -> str:
    return _callable_semantic_hash_v68(
        execute_outer_selection_v68,
        semantic_contract={
            "rule": "best_common_loss_then_parsimony_else_abstain",
            "failed_branch": "ineligible_and_retained",
            "version": "v6.8",
        },
    )


class BranchBudgetV68(StrictModel):
    schema_version: Literal["6.8-branch-budget"] = "6.8-branch-budget"
    max_wall_seconds: Annotated[int, Field(ge=1)]
    max_cpu_seconds: Annotated[int, Field(ge=1)]
    max_memory_megabytes: Annotated[int, Field(ge=64)]
    max_artifact_bytes: Annotated[int, Field(ge=1)]
    max_model_calls: Annotated[int, Field(ge=0)]
    max_tool_calls: Annotated[int, Field(ge=0)]
    budget_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_budget(self) -> "BranchBudgetV68":
        if self.budget_hash and self.budget_hash != self.content_hash():
            raise ValueError("V6.8 branch budget hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "budget_hash")

    @classmethod
    def seal(cls, **data: object) -> "BranchBudgetV68":
        return _sealed(cls, data, "budget_hash")

    def assert_sealed(self) -> None:
        if not self.budget_hash or self.budget_hash != self.content_hash():
            raise ValueError("V6.8 branch budget is not sealed")

    def assert_within(self, envelope: ResourceEnvelopeV68) -> None:
        for field_name in (
            "max_wall_seconds",
            "max_cpu_seconds",
            "max_memory_megabytes",
            "max_artifact_bytes",
            "max_model_calls",
            "max_tool_calls",
        ):
            if getattr(self, field_name) > getattr(envelope, field_name):
                raise ValueError(
                    f"branch budget exceeds capability {field_name}"
                )


class PortfolioBudgetV68(StrictModel):
    schema_version: Literal["6.8-portfolio-budget"] = "6.8-portfolio-budget"
    max_parallel_branches: Annotated[int, Field(ge=1, le=8)]
    total_wall_seconds: Annotated[int, Field(ge=1)]
    total_cpu_seconds: Annotated[int, Field(ge=1)]
    total_memory_megabytes: Annotated[int, Field(ge=128)]
    total_artifact_bytes: Annotated[int, Field(ge=1)]
    total_model_calls: Annotated[int, Field(ge=0)]
    total_tool_calls: Annotated[int, Field(ge=0)]
    budget_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_budget(self) -> "PortfolioBudgetV68":
        if self.budget_hash and self.budget_hash != self.content_hash():
            raise ValueError("V6.8 portfolio budget hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "budget_hash")

    @classmethod
    def seal(cls, **data: object) -> "PortfolioBudgetV68":
        return _sealed(cls, data, "budget_hash")

    def assert_sealed(self) -> None:
        if not self.budget_hash or self.budget_hash != self.content_hash():
            raise ValueError("V6.8 portfolio budget is not sealed")


class CommonLossContractV68(StrictModel):
    schema_version: Literal["6.8-common-loss-contract"] = (
        "6.8-common-loss-contract"
    )
    loss_id: Literal["one_step_rmse"] = "one_step_rmse"
    loss_implementation_ref: Annotated[str, Field(min_length=5, max_length=500)]
    loss_semantic_hash: Sha256
    direction: Literal["minimize"] = "minimize"
    loss_unit: Annotated[str, Field(min_length=1, max_length=120)]
    common_data_view_rule: Annotated[str, Field(min_length=10, max_length=1000)]
    common_evaluation_origin_rule: Annotated[
        str,
        Field(min_length=10, max_length=1000),
    ]
    failed_or_not_run_branch_is_ineligible: Literal[True] = True
    private_acceptance_feedback_permitted: Literal[False] = False
    contract_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "CommonLossContractV68":
        expected_semantic_hash = one_step_rmse_semantic_hash_v68()
        if (
            self.loss_implementation_ref
            != "fma.v68.common_loss.one_step_rmse"
            or self.loss_semantic_hash != expected_semantic_hash
            or self.common_data_view_rule
            != "Every branch receives the same content-addressed ordered series."
            or self.common_evaluation_origin_rule
            != "Every branch is scored on the same frozen outer rolling origins."
        ):
            raise ValueError(
                "V6.8 common loss differs from the code-owned registry"
            )
        if self.contract_hash and self.contract_hash != self.content_hash():
            raise ValueError("V6.8 common-loss contract hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "contract_hash")

    @classmethod
    def seal(cls, **data: object) -> "CommonLossContractV68":
        return _sealed(cls, data, "contract_hash")

    def assert_sealed(self) -> None:
        if not self.contract_hash or self.contract_hash != self.content_hash():
            raise ValueError("V6.8 common-loss contract is not sealed")


class OuterSelectionPolicyV68(StrictModel):
    schema_version: Literal["6.8-outer-selection-policy"] = (
        "6.8-outer-selection-policy"
    )
    policy_id: Literal["nested_one_step_rmse_v68"] = (
        "nested_one_step_rmse_v68"
    )
    implementation_ref: Annotated[str, Field(min_length=5, max_length=500)]
    implementation_semantic_hash: Sha256
    selection_rule: Literal[
        "best_common_loss_then_parsimony_else_abstain"
    ] = "best_common_loss_then_parsimony_else_abstain"
    tie_tolerance: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    minimum_completed_branches: Annotated[int, Field(ge=2, le=8)]
    all_branches_fail_action: Literal["ABSTAIN"] = "ABSTAIN"
    branch_failure_is_retained: Literal[True] = True
    inner_selection_isolated_by_branch: Literal[True] = True
    outcome_conditioned_branch_addition_permitted: Literal[False] = False
    private_acceptance_feedback_permitted: Literal[False] = False
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "OuterSelectionPolicyV68":
        expected_semantic_hash = outer_selector_semantic_hash_v68()
        if (
            self.implementation_ref
            != "fma.v68.selection.nested_one_step_rmse"
            or self.implementation_semantic_hash != expected_semantic_hash
        ):
            raise ValueError(
                "V6.8 outer selection differs from the code-owned registry"
            )
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("V6.8 outer-selection policy hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    @classmethod
    def seal(cls, **data: object) -> "OuterSelectionPolicyV68":
        return _sealed(cls, data, "policy_hash")

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("V6.8 outer-selection policy is not sealed")


class PortfolioBranchRequestV68(StrictModel):
    """One exact manifest request supplied to the pre-data compiler."""

    branch_id: Identifier
    manifest_id: Identifier
    manifest_hash: Sha256
    budget: BranchBudgetV68

    @model_validator(mode="after")
    def validate_request(self) -> "PortfolioBranchRequestV68":
        self.budget.assert_sealed()
        return self


class PortfolioBranchV68(StrictModel):
    """Frozen branch identity copied from one exact registry manifest."""

    schema_version: Literal["6.8-portfolio-branch"] = "6.8-portfolio-branch"
    branch_id: Identifier
    manifest_id: Identifier
    manifest_hash: Sha256
    capability_pack_id: Identifier
    capability_pack_hash: Sha256
    capability_pack_version: Identifier
    skeleton_atoms: Annotated[list[Identifier], Field(min_length=1)]
    skeleton_subsumption_hash: Sha256
    typed_ir_schema_hash: Sha256
    compiler_semantic_hash: Sha256
    executor_semantic_hash: Sha256
    verifier_semantic_hashes: dict[Identifier, Sha256]
    supported_common_loss_ids: Annotated[list[Identifier], Field(min_length=1)]
    baseline_ids: Annotated[list[Identifier], Field(min_length=1)]
    capability_resource_envelope: ResourceEnvelopeV68
    budget: BranchBudgetV68
    branch_hash: Sha256 | None = None
    branch_protocol_is_scientific_evidence: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_branch(self) -> "PortfolioBranchV68":
        self.budget.assert_sealed()
        self.budget.assert_within(self.capability_resource_envelope)
        if self.skeleton_atoms != sorted(set(self.skeleton_atoms)):
            raise ValueError("V6.8 branch skeleton atoms must be sorted and unique")
        if self.skeleton_subsumption_hash != skeleton_subsumption_hash_v68(
            self.skeleton_atoms
        ):
            raise ValueError("V6.8 branch skeleton subsumption hash differs")
        if list(self.verifier_semantic_hashes) != [
            "L0",
            "L1",
            "L2",
            "L3",
            "L4",
        ]:
            raise ValueError("V6.8 branch must bind ordered L0-L4 verifiers")
        for field_name in ("supported_common_loss_ids", "baseline_ids"):
            values = getattr(self, field_name)
            if values != sorted(set(values)):
                raise ValueError(f"V6.8 branch {field_name} must be sorted")
        if self.branch_hash and self.branch_hash != self.content_hash():
            raise ValueError("V6.8 portfolio branch hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "branch_hash")

    @classmethod
    def seal(cls, **data: object) -> "PortfolioBranchV68":
        return _sealed(cls, data, "branch_hash")

    def assert_sealed(self) -> None:
        if not self.branch_hash or self.branch_hash != self.content_hash():
            raise ValueError("V6.8 portfolio branch is not sealed")


def _assert_non_subsuming_skeletons(branches: list[PortfolioBranchV68]) -> None:
    for index, left in enumerate(branches):
        left_atoms = set(left.skeleton_atoms)
        for right in branches[index + 1 :]:
            right_atoms = set(right.skeleton_atoms)
            if left_atoms == right_atoms:
                raise ValueError(
                    "portfolio contains duplicate mathematical skeletons"
                )
            if left_atoms < right_atoms or right_atoms < left_atoms:
                raise ValueError(
                    "portfolio contains a skeleton subsumed by another branch"
                )


class ModelingPortfolioProtocolV68(StrictModel):
    """Sealed portfolio control plane compiled before observation access."""

    schema_version: Literal["6.8-modeling-portfolio-protocol"] = (
        "6.8-modeling-portfolio-protocol"
    )
    protocol_id: Identifier
    compiler_id: Literal["modeling-portfolio-compiler-v68"] = (
        "modeling-portfolio-compiler-v68"
    )
    runtime_mode: CapabilityRuntimeModeV68
    workspace_spec_hash: Sha256
    s0_gate_hash: Sha256
    measurement_contract_hash: Sha256
    capability_query_hash: Sha256
    registry_snapshot_hash: Sha256
    route_decision_hash: Sha256
    branches: Annotated[
        list[PortfolioBranchV68],
        Field(min_length=2, max_length=8),
    ]
    budget: PortfolioBudgetV68
    common_loss: CommonLossContractV68
    outer_selection: OuterSelectionPolicyV68
    observation_values_accessed_during_compilation: Literal[False] = False
    observed_statistics_accessed_during_compilation: Literal[False] = False
    private_acceptance_data_accessed: Literal[False] = False
    model_text_executable: Literal[False] = False
    protocol_is_scientific_evidence: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    protocol_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_protocol(self) -> "ModelingPortfolioProtocolV68":
        self.budget.assert_sealed()
        self.common_loss.assert_sealed()
        self.outer_selection.assert_sealed()
        for branch in self.branches:
            branch.assert_sealed()
        branch_ids = [item.branch_id for item in self.branches]
        if branch_ids != sorted(set(branch_ids)):
            raise ValueError("V6.8 portfolio branches must be sorted and unique")
        manifest_ids = [item.manifest_id for item in self.branches]
        if len(manifest_ids) != len(set(manifest_ids)):
            raise ValueError("V6.8 portfolio repeats a capability manifest")
        pack_ids = [item.capability_pack_id for item in self.branches]
        if len(pack_ids) != len(set(pack_ids)):
            raise ValueError("V6.8 portfolio repeats a capability pack")
        _assert_non_subsuming_skeletons(self.branches)
        if self.budget.max_parallel_branches > len(self.branches):
            raise ValueError("parallel branch budget exceeds branch count")
        if self.outer_selection.minimum_completed_branches > len(
            self.branches
        ):
            raise ValueError("outer selection requires too many completed branches")
        if any(
            self.common_loss.loss_id not in item.supported_common_loss_ids
            for item in self.branches
        ):
            raise ValueError("portfolio common loss is unsupported by a branch")
        sums = {
            "total_cpu_seconds": sum(
                item.budget.max_cpu_seconds for item in self.branches
            ),
            "total_memory_megabytes": sum(
                item.budget.max_memory_megabytes for item in self.branches
            ),
            "total_artifact_bytes": sum(
                item.budget.max_artifact_bytes for item in self.branches
            ),
            "total_model_calls": sum(
                item.budget.max_model_calls for item in self.branches
            ),
            "total_tool_calls": sum(
                item.budget.max_tool_calls for item in self.branches
            ),
        }
        for total_field, allocated in sums.items():
            if allocated > getattr(self.budget, total_field):
                raise ValueError(
                    f"portfolio branch allocation exceeds {total_field}"
                )
        if any(
            item.budget.max_wall_seconds > self.budget.total_wall_seconds
            for item in self.branches
        ):
            raise ValueError(
                "a branch wall budget exceeds the portfolio wall budget"
            )
        batches = math.ceil(
            len(self.branches) / self.budget.max_parallel_branches
        )
        conservative_wall_seconds = batches * max(
            item.budget.max_wall_seconds for item in self.branches
        )
        if conservative_wall_seconds > self.budget.total_wall_seconds:
            raise ValueError(
                "portfolio branch schedule can exceed total_wall_seconds"
            )
        if self.protocol_hash and self.protocol_hash != self.content_hash():
            raise ValueError("V6.8 modeling portfolio protocol hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "protocol_hash")

    @classmethod
    def seal(cls, **data: object) -> "ModelingPortfolioProtocolV68":
        return _sealed(cls, data, "protocol_hash")

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if not self.protocol_hash or self.protocol_hash != self.content_hash():
            raise ValueError("V6.8 modeling portfolio protocol is not sealed")


def _branch_from_manifest(
    request: PortfolioBranchRequestV68,
    manifest: CapabilityManifestV68,
) -> PortfolioBranchV68:
    request.budget.assert_within(manifest.resources)
    return PortfolioBranchV68.seal(
        branch_id=request.branch_id,
        manifest_id=manifest.manifest_id,
        manifest_hash=manifest.manifest_hash,
        capability_pack_id=manifest.capability_pack.pack_id,
        capability_pack_hash=manifest.capability_pack.pack_hash,
        capability_pack_version=manifest.capability_pack.pack_version,
        skeleton_atoms=manifest.skeleton_atoms,
        skeleton_subsumption_hash=manifest.skeleton_subsumption_hash,
        typed_ir_schema_hash=manifest.typed_ir.ir_schema_hash,
        compiler_semantic_hash=manifest.compiler.semantic_hash,
        executor_semantic_hash=manifest.executor.semantic_hash,
        verifier_semantic_hashes={
            item.level: item.verifier.semantic_hash
            for item in manifest.level_obligations
        },
        supported_common_loss_ids=(
            manifest.baselines.supported_common_loss_ids
        ),
        baseline_ids=manifest.baselines.baseline_ids,
        capability_resource_envelope=manifest.resources,
        budget=request.budget,
    )


def compile_modeling_portfolio_protocol_v68(
    *,
    query: CapabilityQueryV68,
    registry: CapabilityRegistryV68,
    branch_requests: list[PortfolioBranchRequestV68],
    budget: PortfolioBudgetV68,
    common_loss: CommonLossContractV68,
    outer_selection: OuterSelectionPolicyV68,
) -> ModelingPortfolioProtocolV68:
    """Compile an exact 2--8 branch portfolio without accepting observations."""

    query.assert_sealed()
    budget.assert_sealed()
    common_loss.assert_sealed()
    outer_selection.assert_sealed()
    if common_loss.loss_unit != query.measurement.measurement_unit:
        raise ValueError(
            "portfolio common-loss unit differs from the measurement contract"
        )
    if not 2 <= len(branch_requests) <= 8:
        raise ValueError("V6.8 portfolio requires between two and eight branches")
    request_ids = [item.branch_id for item in branch_requests]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("V6.8 portfolio branch request IDs must be unique")

    snapshot = registry.snapshot()
    route = registry.route(query)
    compatible = set(route.compatible_manifest_ids)
    branches: list[PortfolioBranchV68] = []
    for request in sorted(branch_requests, key=lambda item: item.branch_id):
        manifest = registry.lookup_exact(
            request.manifest_id,
            request.manifest_hash,
        )
        if manifest.manifest_id not in compatible:
            raise ValueError(
                "portfolio requested a capability incompatible with the query"
            )
        if common_loss.loss_id not in (
            manifest.baselines.supported_common_loss_ids
        ):
            raise ValueError(
                "portfolio requested a pack without the common loss"
            )
        branches.append(_branch_from_manifest(request, manifest))
    _assert_non_subsuming_skeletons(branches)

    protocol_id = "portfolio-" + sha256_value(
        {
            "runtime_mode": registry.runtime_mode,
            "query_hash": query.query_hash,
            "registry_snapshot_hash": snapshot.snapshot_hash,
            "route_decision_hash": route.decision_hash,
            "branch_hashes": [item.branch_hash for item in branches],
            "budget_hash": budget.budget_hash,
            "common_loss_hash": common_loss.contract_hash,
            "outer_selection_hash": outer_selection.policy_hash,
        }
    )[:24]
    return ModelingPortfolioProtocolV68.seal(
        protocol_id=protocol_id,
        runtime_mode=registry.runtime_mode,
        workspace_spec_hash=query.workspace_spec_hash,
        s0_gate_hash=query.s0_gate_hash,
        measurement_contract_hash=query.measurement.measurement_contract_hash,
        capability_query_hash=query.query_hash,
        registry_snapshot_hash=snapshot.snapshot_hash,
        route_decision_hash=route.decision_hash,
        branches=branches,
        budget=budget,
        common_loss=common_loss,
        outer_selection=outer_selection,
    )


def verify_modeling_portfolio_protocol_v68(
    *,
    query: CapabilityQueryV68,
    registry: CapabilityRegistryV68,
    branch_requests: list[PortfolioBranchRequestV68],
    budget: PortfolioBudgetV68,
    common_loss: CommonLossContractV68,
    outer_selection: OuterSelectionPolicyV68,
    protocol: ModelingPortfolioProtocolV68,
) -> bool:
    """Replay the observation-free compiler and compare the complete artifact."""

    try:
        protocol.assert_sealed()
        expected = compile_modeling_portfolio_protocol_v68(
            query=query,
            registry=registry,
            branch_requests=branch_requests,
            budget=budget,
            common_loss=common_loss,
            outer_selection=outer_selection,
        )
    except (KeyError, PermissionError, TypeError, ValueError):
        return False
    return expected == protocol


FiniteNumberV68 = Annotated[float, Field(allow_inf_nan=False)]
BranchExecutionStatusV68 = Literal["PASS", "FAIL", "NOT_RUN"]
PortfolioSelectionKindV68 = Literal["SELECT", "ABSTAIN"]


class BranchOuterEvaluationV68(StrictModel):
    """One branch projected onto the exact common outer origins."""

    schema_version: Literal["6.8-branch-outer-evaluation"] = (
        "6.8-branch-outer-evaluation"
    )
    protocol_hash: Sha256
    branch_id: Identifier
    manifest_id: Identifier
    manifest_hash: Sha256
    execution_evidence_hash: Sha256
    outer_origin_ids: Annotated[list[Identifier], Field(min_length=1)]
    observations: Annotated[list[FiniteNumberV68], Field(min_length=1)]
    predictions: list[FiniteNumberV68]
    parameter_count: Annotated[int, Field(ge=0)]
    execution_status: BranchExecutionStatusV68
    failure_reason: Identifier | None = None
    evaluation_is_scientific_qualification: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    evaluation_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_evaluation(self) -> "BranchOuterEvaluationV68":
        if self.outer_origin_ids != list(dict.fromkeys(self.outer_origin_ids)):
            raise ValueError("V6.8 outer origins must be ordered and unique")
        if len(self.outer_origin_ids) != len(self.observations):
            raise ValueError("V6.8 origins and observations differ in length")
        if self.execution_status == "PASS":
            if (
                len(self.predictions) != len(self.observations)
                or self.failure_reason is not None
            ):
                raise ValueError(
                    "passing V6.8 branch lacks a complete prediction view"
                )
        elif self.predictions or self.failure_reason is None:
            raise ValueError(
                "failed or NOT_RUN V6.8 branch must retain one reason"
            )
        if self.evaluation_hash and (
            self.evaluation_hash != self.content_hash()
        ):
            raise ValueError("V6.8 branch outer evaluation hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evaluation_hash")

    @classmethod
    def seal(cls, **data: object) -> "BranchOuterEvaluationV68":
        return _sealed(cls, data, "evaluation_hash")

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if (
            not self.evaluation_hash
            or self.evaluation_hash != self.content_hash()
        ):
            raise ValueError("V6.8 branch outer evaluation is not sealed")


class PortfolioSelectionDecisionV68(StrictModel):
    """Recomputed local portfolio choice; never a scientific or action gate."""

    schema_version: Literal["6.8-portfolio-selection-decision"] = (
        "6.8-portfolio-selection-decision"
    )
    protocol_hash: Sha256
    common_loss_contract_hash: Sha256
    outer_selection_policy_hash: Sha256
    data_view_hash: Sha256
    evaluation_hashes: dict[Identifier, Sha256]
    branch_statuses: dict[Identifier, BranchExecutionStatusV68]
    eligible_branch_ids: list[Identifier]
    rejected_branches: dict[Identifier, Identifier]
    common_loss_by_branch: dict[Identifier, FiniteNumberV68]
    parameter_count_by_branch: dict[Identifier, int]
    loss_tie_candidates: list[Identifier]
    parsimony_finalists: list[Identifier]
    decision: PortfolioSelectionKindV68
    selected_branch_id: Identifier | None
    reason_code: Identifier
    selection_protocol_executed: Literal[True] = True
    decision_is_scientific_evidence: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    decision_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> "PortfolioSelectionDecisionV68":
        for values, field_name in (
            (self.eligible_branch_ids, "eligible branches"),
            (self.loss_tie_candidates, "loss-tie candidates"),
            (self.parsimony_finalists, "parsimony finalists"),
        ):
            if values != sorted(set(values)):
                raise ValueError(f"V6.8 {field_name} must be sorted and unique")
        for mapping, field_name in (
            (self.evaluation_hashes, "evaluation hashes"),
            (self.branch_statuses, "branch statuses"),
            (self.rejected_branches, "rejected branches"),
            (self.common_loss_by_branch, "common losses"),
            (self.parameter_count_by_branch, "parameter counts"),
        ):
            if list(mapping) != sorted(mapping):
                raise ValueError(f"V6.8 {field_name} must be sorted")
        eligible = set(self.eligible_branch_ids)
        if (
            set(self.common_loss_by_branch) != eligible
            or set(self.parameter_count_by_branch) != eligible
            or set(self.rejected_branches) & eligible
        ):
            raise ValueError("V6.8 selection branch partitions differ")
        if self.decision == "SELECT":
            if (
                self.selected_branch_id is None
                or self.parsimony_finalists != [self.selected_branch_id]
                or self.reason_code != "unique_loss_parsimony_winner"
            ):
                raise ValueError("V6.8 SELECT decision is inconsistent")
        elif (
            self.selected_branch_id is not None
            or self.reason_code
            not in {
                "insufficient_completed_branches",
                "unresolved_loss_parsimony_tie",
            }
        ):
            raise ValueError("V6.8 ABSTAIN decision is inconsistent")
        if self.decision_hash and self.decision_hash != self.content_hash():
            raise ValueError("V6.8 portfolio selection hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "decision_hash")

    @classmethod
    def seal(cls, **data: object) -> "PortfolioSelectionDecisionV68":
        return _sealed(cls, data, "decision_hash")

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if not self.decision_hash or self.decision_hash != self.content_hash():
            raise ValueError("V6.8 portfolio selection is not sealed")


def execute_outer_selection_v68(
    *,
    protocol: ModelingPortfolioProtocolV68,
    evaluations: list[BranchOuterEvaluationV68],
) -> PortfolioSelectionDecisionV68:
    """Score exact common origins and select uniquely, otherwise abstain."""

    protocol.assert_sealed()
    expected_branches = {item.branch_id: item for item in protocol.branches}
    supplied_ids = [item.branch_id for item in evaluations]
    if supplied_ids != sorted(set(supplied_ids)):
        raise ValueError("V6.8 outer evaluations must be sorted and unique")
    if set(supplied_ids) != set(expected_branches):
        raise ValueError(
            "V6.8 outer evaluations must retain every portfolio branch"
        )
    data_view_hashes: set[str] = set()
    statuses: dict[str, BranchExecutionStatusV68] = {}
    evaluation_hashes: dict[str, str] = {}
    rejected: dict[str, str] = {}
    losses: dict[str, float] = {}
    parameter_counts: dict[str, int] = {}
    for evaluation in evaluations:
        evaluation.assert_sealed()
        branch = expected_branches[evaluation.branch_id]
        if (
            evaluation.protocol_hash != protocol.protocol_hash
            or evaluation.manifest_id != branch.manifest_id
            or evaluation.manifest_hash != branch.manifest_hash
        ):
            raise ValueError("V6.8 outer evaluation branch binding differs")
        data_view_hashes.add(
            sha256_value(
                {
                    "outer_origin_ids": evaluation.outer_origin_ids,
                    "observations": evaluation.observations,
                }
            )
        )
        statuses[evaluation.branch_id] = evaluation.execution_status
        evaluation_hashes[evaluation.branch_id] = str(
            evaluation.evaluation_hash
        )
        if evaluation.execution_status == "PASS":
            losses[evaluation.branch_id] = one_step_rmse_v68(
                evaluation.observations,
                evaluation.predictions,
            )
            parameter_counts[evaluation.branch_id] = (
                evaluation.parameter_count
            )
        else:
            rejected[evaluation.branch_id] = str(evaluation.failure_reason)
    if len(data_view_hashes) != 1:
        raise ValueError("V6.8 branches use different outer data views")
    data_view_hash = next(iter(data_view_hashes))
    eligible = sorted(losses)
    loss_ties: list[str] = []
    finalists: list[str] = []
    selected: str | None = None
    if len(eligible) < protocol.outer_selection.minimum_completed_branches:
        decision: PortfolioSelectionKindV68 = "ABSTAIN"
        reason = "insufficient_completed_branches"
    else:
        best_loss = min(losses.values())
        tolerance = protocol.outer_selection.tie_tolerance
        loss_ties = sorted(
            branch_id
            for branch_id, loss in losses.items()
            if loss <= best_loss + tolerance
        )
        fewest_parameters = min(
            parameter_counts[branch_id] for branch_id in loss_ties
        )
        finalists = sorted(
            branch_id
            for branch_id in loss_ties
            if parameter_counts[branch_id] == fewest_parameters
        )
        if len(finalists) == 1:
            decision = "SELECT"
            selected = finalists[0]
            reason = "unique_loss_parsimony_winner"
        else:
            decision = "ABSTAIN"
            reason = "unresolved_loss_parsimony_tie"
    return PortfolioSelectionDecisionV68.seal(
        protocol_hash=str(protocol.protocol_hash),
        common_loss_contract_hash=str(protocol.common_loss.contract_hash),
        outer_selection_policy_hash=str(
            protocol.outer_selection.policy_hash
        ),
        data_view_hash=data_view_hash,
        evaluation_hashes={
            key: evaluation_hashes[key] for key in sorted(evaluation_hashes)
        },
        branch_statuses={key: statuses[key] for key in sorted(statuses)},
        eligible_branch_ids=eligible,
        rejected_branches={key: rejected[key] for key in sorted(rejected)},
        common_loss_by_branch={key: losses[key] for key in sorted(losses)},
        parameter_count_by_branch={
            key: parameter_counts[key] for key in sorted(parameter_counts)
        },
        loss_tie_candidates=loss_ties,
        parsimony_finalists=finalists,
        decision=decision,
        selected_branch_id=selected,
        reason_code=reason,
    )


def verify_outer_selection_v68(
    *,
    protocol: ModelingPortfolioProtocolV68,
    evaluations: list[BranchOuterEvaluationV68],
    decision: PortfolioSelectionDecisionV68,
) -> bool:
    """Replay the complete score/selection function without repair."""

    try:
        decision.assert_sealed()
        expected = execute_outer_selection_v68(
            protocol=protocol,
            evaluations=evaluations,
        )
    except (KeyError, TypeError, ValueError):
        return False
    return decision == expected


__all__ = [
    "BranchBudgetV68",
    "BranchOuterEvaluationV68",
    "CommonLossContractV68",
    "ModelingPortfolioProtocolV68",
    "OuterSelectionPolicyV68",
    "PortfolioBranchRequestV68",
    "PortfolioBranchV68",
    "PortfolioBudgetV68",
    "PortfolioSelectionDecisionV68",
    "compile_modeling_portfolio_protocol_v68",
    "execute_outer_selection_v68",
    "one_step_rmse_semantic_hash_v68",
    "one_step_rmse_v68",
    "outer_selector_semantic_hash_v68",
    "verify_modeling_portfolio_protocol_v68",
    "verify_outer_selection_v68",
]
