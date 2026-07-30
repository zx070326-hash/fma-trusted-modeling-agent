"""Additive V6.8 capability-pack contracts and deterministic routing.

V6.8 deliberately wraps, rather than reinterprets, the sealed V6.0
``CapabilityPackV60`` contract.  A V6.8 manifest adds the information needed
to use a pack in a general modelling portfolio: claim and measurement
applicability, typed-IR and implementation identities, L0--L4 obligations,
resource and recovery envelopes, benchmark bindings, and a mechanical
skeleton-subsumption identity.

The registry is a code-owned in-memory routing table.  Artifacts may name a
pack, but they cannot cause Python imports or promote a development pack into
the stage workflow.  Stage-workflow admission requires an exact manifest hash
supplied to the registry by the surrounding authority boundary.
"""

from __future__ import annotations

from typing import Annotated, Literal, Mapping, Protocol, runtime_checkable

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256

from .measurement_study_design import ClaimKindV67
from .recovery_kernel import CapabilityPackV60, ProblemSignatureV60


CapabilityRuntimeModeV68 = Literal["development_sandbox", "stage_workflow"]
CapabilityMaturityV68 = Literal["development_sandbox", "stage_workflow"]
ClaimCeilingV68 = Literal[
    "local_descriptive_evidence_only",
    "local_predictive_evidence_only",
    "mechanistic_hypothesis_only",
    "prescriptive_simulation_only",
    "local_generalization_diagnostic_only",
]
CapabilityRouteStatusV68 = Literal["ROUTABLE", "CAPABILITY_GAP"]
ConformanceStatusV68 = Literal["PASS", "FAIL"]
LevelV68 = Literal["L0", "L1", "L2", "L3", "L4"]
TaskKindV68 = Literal[
    "explanation",
    "prediction",
    "control",
    "optimization",
    "design",
    "mixed",
]
ScaleTypeV68 = Literal[
    "nominal",
    "ordinal",
    "interval",
    "ratio",
    "count",
    "event_time",
]
StudyDesignTypeV68 = Literal[
    "observational_longitudinal",
    "observational_cross_sectional",
    "time_series",
    "panel",
    "randomized_experiment",
    "quasi_experiment",
    "simulation",
]
MissingnessPolicyV68 = Literal[
    "reject_incomplete_series",
    "complete_case",
    "multiple_imputation",
    "model_based",
    "sensitivity_analysis_required",
    "not_applicable_by_design",
]

_LEVELS: tuple[LevelV68, ...] = ("L0", "L1", "L2", "L3", "L4")
_CLAIM_CEILINGS: dict[str, ClaimCeilingV68] = {
    "descriptive": "local_descriptive_evidence_only",
    "predictive": "local_predictive_evidence_only",
    "mechanistic": "mechanistic_hypothesis_only",
    "prescriptive": "prescriptive_simulation_only",
    "generalization": "local_generalization_diagnostic_only",
}


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _require_sorted_unique(values: list[str], field_name: str) -> None:
    if values != sorted(set(values)):
        raise ValueError(f"{field_name} must be sorted and unique")


def skeleton_subsumption_hash_v68(skeleton_atoms: list[str]) -> str:
    """Return the exact identity used for duplicate/subsumption checks."""

    _require_sorted_unique(skeleton_atoms, "skeleton atoms")
    if not skeleton_atoms:
        raise ValueError("skeleton atoms cannot be empty")
    return sha256_value(
        {
            "schema_version": "6.8-skeleton-subsumption",
            "skeleton_atoms": skeleton_atoms,
            "subsumption_rule": "set_inclusion",
        }
    )


class MeasurementSignatureV68(StrictModel):
    """Pre-data measurement attributes used by capability routing."""

    schema_version: Literal["6.8-measurement-signature"] = (
        "6.8-measurement-signature"
    )
    measurement_contract_hash: Sha256
    scale_type: ScaleTypeV68
    study_design_type: StudyDesignTypeV68
    missingness_policy: MissingnessPolicyV68
    measurement_unit: Annotated[str, Field(min_length=1, max_length=120)]
    time_basis: Annotated[str, Field(min_length=3, max_length=300)]
    minimum_planned_observations: Annotated[int, Field(ge=1)]
    observation_values_included: Literal[False] = False
    observed_statistics_included: Literal[False] = False


class CapabilityQueryV68(StrictModel):
    """A sealed, observation-free routing request."""

    schema_version: Literal["6.8-capability-query"] = "6.8-capability-query"
    workspace_spec_hash: Sha256
    s0_gate_hash: Sha256
    problem_signature: ProblemSignatureV60
    claim_kind: ClaimKindV67
    measurement: MeasurementSignatureV68
    private_acceptance_data_accessed: Literal[False] = False
    query_hash: Sha256 | None = None
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_query(self) -> "CapabilityQueryV68":
        if (
            self.problem_signature.observation_count
            != self.measurement.minimum_planned_observations
        ):
            raise ValueError(
                "problem and measurement planned observation counts differ"
            )
        if self.query_hash and self.query_hash != self.content_hash():
            raise ValueError("V6.8 capability query hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "query_hash")

    @classmethod
    def seal(cls, **data: object) -> "CapabilityQueryV68":
        draft = cls(**data)
        return cls(
            **draft.model_dump(mode="json", exclude={"query_hash"}),
            query_hash=draft.content_hash(),
        )

    def assert_sealed(self) -> None:
        if not self.query_hash or self.query_hash != self.content_hash():
            raise ValueError("V6.8 capability query is not sealed")


class MeasurementApplicabilityV68(StrictModel):
    """Declarative measurement boundary for one capability manifest."""

    scale_types: Annotated[list[ScaleTypeV68], Field(min_length=1)]
    study_design_types: Annotated[
        list[StudyDesignTypeV68],
        Field(min_length=1),
    ]
    missingness_policies: Annotated[
        list[MissingnessPolicyV68],
        Field(min_length=1),
    ]
    accepted_measurement_units: list[str] = Field(default_factory=list)
    accepted_time_bases: list[str] = Field(default_factory=list)
    minimum_planned_observations: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def validate_applicability(self) -> "MeasurementApplicabilityV68":
        for field_name in (
            "scale_types",
            "study_design_types",
            "missingness_policies",
            "accepted_measurement_units",
            "accepted_time_bases",
        ):
            _require_sorted_unique(
                list(getattr(self, field_name)),
                f"measurement applicability {field_name}",
            )
        return self

    def incompatibilities(self, value: MeasurementSignatureV68) -> list[str]:
        reasons: list[str] = []
        if value.scale_type not in self.scale_types:
            reasons.append(f"measurement.scale_type:{value.scale_type}")
        if value.study_design_type not in self.study_design_types:
            reasons.append(
                f"measurement.study_design_type:{value.study_design_type}"
            )
        if value.missingness_policy not in self.missingness_policies:
            reasons.append(
                f"measurement.missingness_policy:{value.missingness_policy}"
            )
        if (
            self.accepted_measurement_units
            and value.measurement_unit not in self.accepted_measurement_units
        ):
            reasons.append(
                f"measurement.measurement_unit:{value.measurement_unit}"
            )
        if (
            self.accepted_time_bases
            and value.time_basis not in self.accepted_time_bases
        ):
            reasons.append(f"measurement.time_basis:{value.time_basis}")
        if (
            value.minimum_planned_observations
            < self.minimum_planned_observations
        ):
            reasons.append(
                "measurement.minimum_planned_observations:"
                f"{value.minimum_planned_observations}"
                f"<{self.minimum_planned_observations}"
            )
        return sorted(reasons)


class TypedModelIRContractV68(StrictModel):
    """Hash-bound schema identity for the pack's non-executable model IR."""

    schema_version: Literal["6.8-typed-model-ir-contract"] = (
        "6.8-typed-model-ir-contract"
    )
    ir_kind: Identifier
    ir_schema_version: Identifier
    ir_schema_hash: Sha256
    compiler_input_schema_hash: Sha256
    compiler_output_schema_hash: Sha256
    model_text_is_executable: Literal[False] = False
    arbitrary_code_is_valid_ir: Literal[False] = False


class SemanticImplementationBindingV68(StrictModel):
    """Auditable identity for a code-owned implementation entry point."""

    implementation_id: Identifier
    implementation_version: Identifier
    entrypoint_ref: Annotated[str, Field(min_length=5, max_length=500)]
    semantic_hash: Sha256
    dynamic_import_from_artifact_permitted: Literal[False] = False
    model_supplied_implementation_permitted: Literal[False] = False


class LevelObligationV68(StrictModel):
    """One mandatory L0--L4 verifier obligation."""

    level: LevelV68
    obligation_id: Identifier
    verifier: SemanticImplementationBindingV68
    evidence_kind: Identifier
    not_run_counts_as_pass: Literal[False] = False
    verifier_is_independent_of_generator: Literal[True] = True


class ResourceEnvelopeV68(StrictModel):
    """Maximum resources a single branch may request."""

    max_wall_seconds: Annotated[int, Field(ge=1)]
    max_cpu_seconds: Annotated[int, Field(ge=1)]
    max_memory_megabytes: Annotated[int, Field(ge=64)]
    max_artifact_bytes: Annotated[int, Field(ge=1)]
    max_model_calls: Annotated[int, Field(ge=0)]
    max_tool_calls: Annotated[int, Field(ge=0)]
    network_access_permitted: Literal[False] = False


class BaselineContractV68(StrictModel):
    baseline_ids: Annotated[list[Identifier], Field(min_length=1)]
    supported_common_loss_ids: Annotated[
        list[Identifier],
        Field(min_length=1),
    ]
    identical_evaluation_window_required: Literal[True] = True
    failed_baseline_counts_as_pass: Literal[False] = False

    @model_validator(mode="after")
    def validate_baselines(self) -> "BaselineContractV68":
        _require_sorted_unique(self.baseline_ids, "baseline IDs")
        _require_sorted_unique(
            self.supported_common_loss_ids,
            "supported common loss IDs",
        )
        return self


RecoveryActionV68 = Literal[
    "RETRY",
    "PATCH",
    "BRANCH",
    "ACQUIRE_DATA",
    "ABSTAIN",
    "HUMAN",
]


class RecoveryContractV68(StrictModel):
    allowed_actions: Annotated[list[RecoveryActionV68], Field(min_length=1)]
    max_graph_attempts: Annotated[int, Field(ge=1)]
    max_same_attempt_retries: Annotated[int, Field(ge=0)]
    adapter_change_requires_successor_program: Literal[True] = True
    threshold_change_requires_successor_program: Literal[True] = True
    private_feedback_may_drive_recovery: Literal[False] = False

    @model_validator(mode="after")
    def validate_recovery(self) -> "RecoveryContractV68":
        _require_sorted_unique(self.allowed_actions, "recovery actions")
        return self


class BenchmarkContractV68(StrictModel):
    benchmark_suite_id: Identifier
    benchmark_suite_version: Identifier
    benchmark_suite_hash: Sha256
    minimum_public_cases: Annotated[int, Field(ge=1)]
    minimum_adversarial_cases: Annotated[int, Field(ge=1)]
    external_private_evaluation_required_for_stage_workflow: bool
    private_outcomes_returned_to_generator: Literal[False] = False
    benchmark_contract_is_promotion_authority: Literal[False] = False


class CapabilityManifestV68(StrictModel):
    """Complete additive manifest for one code-owned capability pack."""

    schema_version: Literal["6.8-capability-manifest"] = (
        "6.8-capability-manifest"
    )
    manifest_id: Identifier
    capability_pack: CapabilityPackV60
    supported_claim_kinds: Annotated[list[ClaimKindV67], Field(min_length=1)]
    claim_ceilings: dict[ClaimKindV67, ClaimCeilingV68]
    supported_task_kinds: Annotated[list[TaskKindV68], Field(min_length=1)]
    measurement_applicability: MeasurementApplicabilityV68
    maturity: CapabilityMaturityV68
    stage_workflow_promotion_receipt_hash: Sha256 | None = None
    typed_ir: TypedModelIRContractV68
    compiler: SemanticImplementationBindingV68
    executor: SemanticImplementationBindingV68
    level_obligations: Annotated[
        list[LevelObligationV68],
        Field(min_length=5, max_length=5),
    ]
    resources: ResourceEnvelopeV68
    baselines: BaselineContractV68
    recovery: RecoveryContractV68
    benchmark: BenchmarkContractV68
    skeleton_atoms: Annotated[list[Identifier], Field(min_length=1)]
    skeleton_subsumption_hash: Sha256
    manifest_hash: Sha256 | None = None
    manifest_is_scientific_evidence: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_manifest(self) -> "CapabilityManifestV68":
        self.capability_pack.assert_sealed()
        _require_sorted_unique(
            list(self.supported_claim_kinds),
            "supported claim kinds",
        )
        if list(self.claim_ceilings) != list(self.supported_claim_kinds):
            raise ValueError(
                "V6.8 claim ceilings must exactly match supported claims"
            )
        if any(
            ceiling != _CLAIM_CEILINGS[claim]
            for claim, ceiling in self.claim_ceilings.items()
        ):
            raise ValueError(
                "V6.8 claim ceiling differs from the code-owned ceiling"
            )
        _require_sorted_unique(
            list(self.supported_task_kinds),
            "supported task kinds",
        )
        _require_sorted_unique(self.skeleton_atoms, "skeleton atoms")
        if [item.level for item in self.level_obligations] != list(_LEVELS):
            raise ValueError("V6.8 capability manifest requires ordered L0-L4")
        obligation_ids = [item.obligation_id for item in self.level_obligations]
        if len(obligation_ids) != len(set(obligation_ids)):
            raise ValueError("V6.8 verifier obligation IDs must be unique")
        if self.executor.implementation_id != self.capability_pack.executor_id:
            raise ValueError(
                "V6.8 manifest executor differs from its V6.0 routing pack"
            )
        if self.baselines.baseline_ids != self.capability_pack.baseline_ids:
            raise ValueError(
                "V6.8 manifest baselines differ from its V6.0 routing pack"
            )
        if (
            self.measurement_applicability.minimum_planned_observations
            < self.capability_pack.minimum_observations
        ):
            raise ValueError(
                "V6.8 measurement minimum is below the routing-pack minimum"
            )
        expected_subsumption = skeleton_subsumption_hash_v68(
            self.skeleton_atoms
        )
        if self.skeleton_subsumption_hash != expected_subsumption:
            raise ValueError("V6.8 skeleton subsumption hash differs")
        if self.maturity == "stage_workflow":
            # V6.8 deliberately ships no promotion authority.  A bare digest is
            # not a verifiable receipt and accepting one would let a caller
            # promote its own pack.  A successor schema must bind a committed,
            # independently signed promotion artifact before this maturity can
            # be represented.
            raise ValueError(
                "V6.8 stage-workflow promotion authority is NOT_RUN"
            )
        elif self.stage_workflow_promotion_receipt_hash is not None:
            raise ValueError(
                "development maturity cannot carry a promotion receipt"
            )
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("V6.8 capability manifest hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    @classmethod
    def seal(cls, **data: object) -> "CapabilityManifestV68":
        draft = cls(**data)
        return cls(
            **draft.model_dump(mode="json", exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("V6.8 capability manifest is not sealed")

    def expected_semantic_hashes(self) -> dict[str, str]:
        expected = {
            "benchmark": self.benchmark.benchmark_suite_hash,
            "capability_pack": str(self.capability_pack.pack_hash),
            "compiler": self.compiler.semantic_hash,
            "executor": self.executor.semantic_hash,
            "skeleton_subsumption": self.skeleton_subsumption_hash,
            "typed_ir": sha256_value(
                self.typed_ir.model_dump(mode="json")
            ),
        }
        expected.update(
            {
                f"verifier.{item.level}": item.verifier.semantic_hash
                for item in self.level_obligations
            }
        )
        return {key: expected[key] for key in sorted(expected)}

    def incompatibilities(self, query: CapabilityQueryV68) -> list[str]:
        reasons = self.capability_pack.incompatibilities(
            query.problem_signature
        )
        if query.claim_kind not in self.supported_claim_kinds:
            reasons.append(f"claim_kind:{query.claim_kind}")
        if query.problem_signature.task_kind not in self.supported_task_kinds:
            reasons.append(
                f"task_kind:{query.problem_signature.task_kind}"
            )
        reasons.extend(
            self.measurement_applicability.incompatibilities(query.measurement)
        )
        return sorted(set(reasons))


@runtime_checkable
class CapabilityPackRuntimeV68(Protocol):
    """Minimal runtime boundary; the harness owns invocation and receipts."""

    manifest: CapabilityManifestV68

    def compile_typed_ir(self, request: StrictModel) -> StrictModel:
        ...

    def execute_typed_ir(self, model_ir: StrictModel) -> StrictModel:
        ...


class ConformanceCheckV68(StrictModel):
    check_id: Identifier
    expected_hash: Sha256
    observed_hash: Sha256 | None
    status: ConformanceStatusV68
    reason_code: Identifier

    @model_validator(mode="after")
    def validate_check(self) -> "ConformanceCheckV68":
        expected_status = (
            "PASS"
            if self.observed_hash == self.expected_hash
            else "FAIL"
        )
        if self.status != expected_status:
            raise ValueError("conformance check status differs from hashes")
        expected_reason = (
            "semantic_hash_match"
            if self.status == "PASS"
            else (
                "semantic_hash_missing"
                if self.observed_hash is None
                else "semantic_hash_mismatch"
            )
        )
        if self.reason_code != expected_reason:
            raise ValueError("conformance reason differs from hashes")
        return self


class CapabilityConformanceReportV68(StrictModel):
    """Mechanical identity report; it can never promote capability maturity."""

    schema_version: Literal["6.8-capability-conformance-report"] = (
        "6.8-capability-conformance-report"
    )
    manifest_id: Identifier
    manifest_hash: Sha256
    expected_semantic_hashes_hash: Sha256
    declared_maturity: CapabilityMaturityV68
    maturity_after_report: CapabilityMaturityV68
    checks: Annotated[list[ConformanceCheckV68], Field(min_length=11)]
    status: ConformanceStatusV68
    maturity_promotion_granted: Literal[False] = False
    report_is_scientific_evidence: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    report_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "CapabilityConformanceReportV68":
        if self.maturity_after_report != self.declared_maturity:
            raise ValueError("mechanical conformance cannot change maturity")
        check_ids = [item.check_id for item in self.checks]
        if check_ids != sorted(set(check_ids)):
            raise ValueError(
                "conformance checks must be sorted and unique"
            )
        expected = (
            "PASS"
            if all(item.status == "PASS" for item in self.checks)
            else "FAIL"
        )
        if self.status != expected:
            raise ValueError("conformance report status differs from checks")
        if self.report_hash and self.report_hash != self.content_hash():
            raise ValueError("V6.8 conformance report hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "report_hash")

    @classmethod
    def seal(cls, **data: object) -> "CapabilityConformanceReportV68":
        draft = cls(**data)
        return cls(
            **draft.model_dump(mode="json", exclude={"report_hash"}),
            report_hash=draft.content_hash(),
        )

    def assert_sealed(self) -> None:
        if not self.report_hash or self.report_hash != self.content_hash():
            raise ValueError("V6.8 conformance report is not sealed")


def evaluate_capability_conformance_v68(
    manifest: CapabilityManifestV68,
    *,
    observed_semantic_hashes: Mapping[str, str],
) -> CapabilityConformanceReportV68:
    """Compare independently observed semantic identities with the manifest."""

    manifest.assert_sealed()
    expected = manifest.expected_semantic_hashes()
    checks: list[ConformanceCheckV68] = []
    for check_id, expected_hash in expected.items():
        observed = observed_semantic_hashes.get(check_id)
        status: ConformanceStatusV68 = (
            "PASS" if observed == expected_hash else "FAIL"
        )
        reason = (
            "semantic_hash_match"
            if status == "PASS"
            else (
                "semantic_hash_missing"
                if observed is None
                else "semantic_hash_mismatch"
            )
        )
        checks.append(
            ConformanceCheckV68(
                check_id=check_id,
                expected_hash=expected_hash,
                observed_hash=observed,
                status=status,
                reason_code=reason,
            )
        )
    # Unexpected runtime identities are also a fail-closed conformance error.
    for check_id in sorted(set(observed_semantic_hashes) - set(expected)):
        observed = observed_semantic_hashes[check_id]
        checks.append(
            ConformanceCheckV68(
                check_id=f"unexpected.{check_id}",
                expected_hash="0" * 64,
                observed_hash=observed,
                status="FAIL",
                reason_code="semantic_hash_mismatch",
            )
        )
    checks.sort(key=lambda item: item.check_id)
    return CapabilityConformanceReportV68.seal(
        manifest_id=manifest.manifest_id,
        manifest_hash=manifest.manifest_hash,
        expected_semantic_hashes_hash=sha256_value(expected),
        declared_maturity=manifest.maturity,
        maturity_after_report=manifest.maturity,
        checks=checks,
        status=(
            "PASS"
            if all(item.status == "PASS" for item in checks)
            else "FAIL"
        ),
    )


class CapabilityRegistryEntryV68(StrictModel):
    manifest_id: Identifier
    manifest_hash: Sha256
    capability_pack_id: Identifier
    capability_pack_hash: Sha256
    maturity: CapabilityMaturityV68
    conformance_report_hash: Sha256 | None


class CapabilityRegistrySnapshotV68(StrictModel):
    schema_version: Literal["6.8-capability-registry-snapshot"] = (
        "6.8-capability-registry-snapshot"
    )
    runtime_mode: CapabilityRuntimeModeV68
    entries: list[CapabilityRegistryEntryV68]
    snapshot_hash: Sha256 | None = None
    snapshot_is_scientific_evidence: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_snapshot(self) -> "CapabilityRegistrySnapshotV68":
        identifiers = [item.manifest_id for item in self.entries]
        if identifiers != sorted(set(identifiers)):
            raise ValueError("registry entries must be sorted and unique")
        if self.snapshot_hash and self.snapshot_hash != self.content_hash():
            raise ValueError("V6.8 registry snapshot hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "snapshot_hash")

    @classmethod
    def seal(cls, **data: object) -> "CapabilityRegistrySnapshotV68":
        draft = cls(**data)
        return cls(
            **draft.model_dump(mode="json", exclude={"snapshot_hash"}),
            snapshot_hash=draft.content_hash(),
        )


class CapabilityRouteDecisionV68(StrictModel):
    schema_version: Literal["6.8-capability-route-decision"] = (
        "6.8-capability-route-decision"
    )
    runtime_mode: CapabilityRuntimeModeV68
    query_hash: Sha256
    registry_snapshot_hash: Sha256
    compatible_manifest_ids: list[Identifier]
    compatible_manifest_hashes: dict[Identifier, Sha256]
    incompatibilities: dict[Identifier, list[str]]
    status: CapabilityRouteStatusV68
    decision_hash: Sha256 | None = None
    decision_is_scientific_evidence: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_decision(self) -> "CapabilityRouteDecisionV68":
        if self.compatible_manifest_ids != sorted(
            set(self.compatible_manifest_ids)
        ):
            raise ValueError("compatible manifest IDs must be sorted and unique")
        if list(self.compatible_manifest_hashes) != (
            self.compatible_manifest_ids
        ):
            raise ValueError("compatible manifest hashes differ from IDs")
        if list(self.incompatibilities) != sorted(self.incompatibilities):
            raise ValueError("route incompatibilities must be key-sorted")
        if set(self.compatible_manifest_ids) & set(self.incompatibilities):
            raise ValueError("a manifest cannot be compatible and incompatible")
        expected = (
            "ROUTABLE"
            if self.compatible_manifest_ids
            else "CAPABILITY_GAP"
        )
        if self.status != expected:
            raise ValueError("route status differs from compatible manifests")
        if self.decision_hash and self.decision_hash != self.content_hash():
            raise ValueError("V6.8 route decision hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "decision_hash")

    @classmethod
    def seal(cls, **data: object) -> "CapabilityRouteDecisionV68":
        draft = cls(**data)
        return cls(
            **draft.model_dump(mode="json", exclude={"decision_hash"}),
            decision_hash=draft.content_hash(),
        )


class CapabilityRegistryV68:
    """Deterministic exact registry with a separate stage-admission boundary."""

    def __init__(
        self,
        *,
        runtime_mode: CapabilityRuntimeModeV68,
        admitted_stage_manifest_hashes: Mapping[str, str] | None = None,
    ) -> None:
        self.runtime_mode = runtime_mode
        if runtime_mode == "stage_workflow":
            raise PermissionError(
                "V6.8 stage-workflow registry is disabled until an external "
                "promotion authority is implemented"
            )
        self._admitted_stage_manifest_hashes = dict(
            admitted_stage_manifest_hashes or {}
        )
        if (
            runtime_mode == "development_sandbox"
            and self._admitted_stage_manifest_hashes
        ):
            raise ValueError(
                "development registry cannot carry stage admission authority"
            )
        self._manifests: dict[str, CapabilityManifestV68] = {}
        self._reports: dict[str, CapabilityConformanceReportV68 | None] = {}
        self._pack_ids: set[str] = set()

    def register(
        self,
        manifest: CapabilityManifestV68,
        *,
        conformance_report: CapabilityConformanceReportV68 | None = None,
    ) -> None:
        manifest.assert_sealed()
        if manifest.manifest_id in self._manifests:
            raise ValueError(f"duplicate capability manifest: {manifest.manifest_id}")
        if manifest.capability_pack.pack_id in self._pack_ids:
            raise ValueError(
                "duplicate capability pack ID: "
                f"{manifest.capability_pack.pack_id}"
            )
        if conformance_report is not None:
            conformance_report.assert_sealed()
            if (
                conformance_report.manifest_id != manifest.manifest_id
                or conformance_report.manifest_hash != manifest.manifest_hash
                or conformance_report.declared_maturity != manifest.maturity
                or conformance_report.expected_semantic_hashes_hash
                != sha256_value(manifest.expected_semantic_hashes())
            ):
                raise ValueError(
                    "conformance report is bound to another manifest"
                )
            expected_semantics = manifest.expected_semantic_hashes()
            report_checks = {
                item.check_id: item for item in conformance_report.checks
            }
            if set(report_checks) != set(expected_semantics) or any(
                report_checks[check_id].expected_hash != expected_hash
                for check_id, expected_hash in expected_semantics.items()
            ):
                raise ValueError(
                    "conformance report check set differs from the manifest"
                )
        if self.runtime_mode == "stage_workflow":
            admitted_hash = self._admitted_stage_manifest_hashes.get(
                manifest.manifest_id
            )
            if (
                manifest.maturity != "stage_workflow"
                or admitted_hash != manifest.manifest_hash
            ):
                raise PermissionError(
                    "manifest lacks exact stage-workflow admission"
                )
            if (
                conformance_report is None
                or conformance_report.status != "PASS"
            ):
                raise PermissionError(
                    "stage-workflow registration requires passing conformance"
                )
        self._manifests[manifest.manifest_id] = manifest.model_copy(deep=True)
        self._reports[manifest.manifest_id] = (
            conformance_report.model_copy(deep=True)
            if conformance_report is not None
            else None
        )
        self._pack_ids.add(manifest.capability_pack.pack_id)

    def lookup_exact(
        self,
        manifest_id: str,
        manifest_hash: str,
    ) -> CapabilityManifestV68:
        manifest = self._manifests.get(manifest_id)
        if manifest is None:
            raise KeyError(f"unregistered capability manifest: {manifest_id}")
        if manifest.manifest_hash != manifest_hash:
            raise KeyError(
                f"capability manifest hash mismatch: {manifest_id}"
            )
        manifest.assert_sealed()
        return manifest.model_copy(deep=True)

    def snapshot(self) -> CapabilityRegistrySnapshotV68:
        entries: list[CapabilityRegistryEntryV68] = []
        for manifest_id, manifest in sorted(self._manifests.items()):
            report = self._reports[manifest_id]
            entries.append(
                CapabilityRegistryEntryV68(
                    manifest_id=manifest_id,
                    manifest_hash=manifest.manifest_hash,
                    capability_pack_id=manifest.capability_pack.pack_id,
                    capability_pack_hash=manifest.capability_pack.pack_hash,
                    maturity=manifest.maturity,
                    conformance_report_hash=(
                        report.report_hash if report is not None else None
                    ),
                )
            )
        return CapabilityRegistrySnapshotV68.seal(
            runtime_mode=self.runtime_mode,
            entries=entries,
        )

    def route(self, query: CapabilityQueryV68) -> CapabilityRouteDecisionV68:
        query.assert_sealed()
        snapshot = self.snapshot()
        compatible: list[str] = []
        compatible_hashes: dict[str, str] = {}
        incompatibilities: dict[str, list[str]] = {}
        for manifest_id, manifest in sorted(self._manifests.items()):
            reasons = manifest.incompatibilities(query)
            if (
                self.runtime_mode == "stage_workflow"
                and manifest.maturity != "stage_workflow"
            ):
                reasons.append(
                    f"maturity:{manifest.maturity}<stage_workflow"
                )
            if reasons:
                incompatibilities[manifest_id] = sorted(set(reasons))
            else:
                compatible.append(manifest_id)
                compatible_hashes[manifest_id] = str(manifest.manifest_hash)
        return CapabilityRouteDecisionV68.seal(
            runtime_mode=self.runtime_mode,
            query_hash=query.query_hash,
            registry_snapshot_hash=snapshot.snapshot_hash,
            compatible_manifest_ids=compatible,
            compatible_manifest_hashes={
                key: compatible_hashes[key] for key in compatible
            },
            incompatibilities={
                key: incompatibilities[key]
                for key in sorted(incompatibilities)
            },
            status="ROUTABLE" if compatible else "CAPABILITY_GAP",
        )


__all__ = [
    "BaselineContractV68",
    "BenchmarkContractV68",
    "CapabilityConformanceReportV68",
    "CapabilityManifestV68",
    "CapabilityMaturityV68",
    "CapabilityPackRuntimeV68",
    "CapabilityQueryV68",
    "CapabilityRegistryEntryV68",
    "CapabilityRegistrySnapshotV68",
    "CapabilityRegistryV68",
    "CapabilityRouteDecisionV68",
    "CapabilityRuntimeModeV68",
    "ConformanceCheckV68",
    "LevelObligationV68",
    "MeasurementApplicabilityV68",
    "MeasurementSignatureV68",
    "RecoveryContractV68",
    "ResourceEnvelopeV68",
    "SemanticImplementationBindingV68",
    "TypedModelIRContractV68",
    "evaluate_capability_conformance_v68",
    "skeleton_subsumption_hash_v68",
]
