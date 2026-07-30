"""Typed local service that connects the web studio to the FMA V5 kernel.

The browser never receives the V5 authority key and cannot write graph state
directly.  It may request a task, bounded S0/S1 discovery, or the registered
positive-scalar-ODE S2--S6 path.  This service validates the request, invokes
isolated Codex role processes, and asks the existing harness to authenticate
checks, reviews, and graph transitions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, Protocol, cast
from uuid import uuid4

from pydantic import Field, ValidationError, model_validator

from fma._file_lock import exclusive_file_lock
from fma.codex_driver import CliLocator, CodexCLIConfig, ProcessRunner
from fma.hashing import canonical_json, sha256_value
from fma.operator_v70 import (
    IntakeManifestV70,
    OperatorAuthorityBindingV70,
    OperatorConflictError,
    OperatorLeaseError,
    OperatorLeaseV70,
    OperatorPacketV70,
    OperatorPlaneError,
    OperatorStoreV70,
    OperatorSubmissionV70,
    capture_file_manifest,
    changed_manifest_paths,
)
from fma.schemas import ArtifactRef, StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5.scaffold import scaffold_task_workspace
from fma.v5.stage_workspace import (
    POLICIES,
    StageWorkspaceError,
    StageWorkspaceV50,
    _evaluate_arithmetic,
)
from fma.v5.workspace_schemas import (
    IndependentReviewReceiptV50,
    RoleExecutionReceiptV50,
    StageId,
    TaskWorkspaceSpecV50,
    WorkflowProfileV50,
)
from fma.v5_1.codex_stage_driver import (
    RoleDraftV51,
    RoleProcessOutcomeV51,
    StageRoleDriverV51,
    StageRoleTransportV51,
    commit_generator_outcome_v51,
)
from fma.v5_8.epistemic import EpistemicGraphStoreV58
from fma.v6.capability_catalog_v68 import (
    default_development_capability_registry_v68,
)
from fma.v6.capability_sdk_v68 import (
    CapabilityQueryV68,
    MeasurementSignatureV68,
)
from fma.v6.decision_value import (
    DECISION_INTENT_PATH,
    DecisionValueIntentV62,
)
from fma.v6.executable_candidate import (
    EXECUTABLE_CANDIDATE_INTENT_PATH,
    EXECUTABLE_CANDIDATE_IR_PATH,
)
from fma.v6.measurement_study_design import (
    MEASUREMENT_STUDY_DESIGN_PATH_V67,
    ApplicabilityBoundaryV67,
    BiasPlanV67,
    ConfoundingPlanV67,
    ConstructDefinitionV67,
    EthicsBoundaryV67,
    MeasurementDefinitionV67,
    MeasurementErrorPlanV67,
    MeasurementStudyDesignContractV67,
    MissingnessPlanV67,
    PopulationDefinitionV67,
    SamplingPlanV67,
    StudyDesignV67,
)
from fma.v6.predata_protocol import (
    CANDIDATE_EXECUTION_BINDING_PATH_V67,
    PREDATA_EXECUTION_PROTOCOL_PATH_V67,
    PreDataExecutionProtocolV67,
    compile_predata_execution_protocol_v67,
    registered_positive_series_capability_pack_v67,
    verify_predata_execution_protocol_v67,
)
from fma.v6.predata_transaction import (
    PREDATA_PREPARATION_COMPLETION_KIND_V67,
    PREDATA_PREPARATION_EVIDENCE_KIND_V67,
    PREDATA_PREPARATION_INTENT_KIND_V67,
    PREDATA_TRANSACTION_POLICY_KIND_V67,
    PreDataPreparationCompletionV67,
    PreDataPreparationIntentV67,
    PreDataTransactionPolicyV67,
    predata_contract_file_bytes_v67,
    predata_preparation_payload_v67,
)
from fma.v6.portfolio_protocol_v68 import BranchBudgetV68, PortfolioBudgetV68
from fma.v6.portfolio_runtime_v69 import (
    MAXIMUM_ROLLING_ORIGINS_V69,
    PersistenceBaselinePolicyV69,
    PositiveSeriesSnapshotV69,
)
from fma.v6.portfolio_transaction_v69 import (
    PORTFOLIO_TRANSACTION_INTENT_KIND_V69,
    DevelopmentPortfolioTransactionV69,
    PortfolioTransactionStateV69,
)
from fma.v6.provenance import (
    MEASUREMENT_SCHEMA_PATH,
    PROVENANCE_BINDING_PATH,
    MeasurementSchemaV62,
)
from fma.v6.public_source import (
    SOURCE_CONTRACT_PATH,
    SOURCE_RAW_PATH,
    SOURCE_RECEIPT_PATH,
    SOURCE_VERIFICATION_PATH,
    SourceFetcherV62,
    WorldBankSourceContractV62,
    materialize_world_bank_series_v62,
    verify_world_bank_source_v62,
)
from fma.v6.recovery_kernel import (
    FailureDiagnosisV60,
    FailureCategoryV60,
    ProblemSignatureV60,
    RecoveryKernelV60,
    RecoveryPlanV60,
    RecoveryTransitionReceiptV60,
)
from fma.v6.s1_review_recovery import (
    S1_FORMALIZATION_FAILURE_CODE_V67,
    S1BoundedRepairContextV67,
    S1FormalizationRejectionHandoffV67,
    build_s1_bounded_repair_context_v67,
    build_s1_formalization_rejection_evidence_v67,
    build_s1_formalization_rejection_handoff_v67,
    s1_recovery_evidence_refs_v67,
)
from fma.v6.stage_driver import CodexStageRoleTransportV66
from fma.v6.stage_gate_outcome import (
    latest_stage_gate_outcome_v66,
    record_blocked_stage_gate_v66,
)
from fma.v6.stage_review_recovery import (
    S0_EVALUATION_PROFILE_PATH_V66,
    DecisionFunctionDraftV66,
    RegimeDiagnosisDraftV66,
    S0RepairContextV66,
    S0ReviewerFindingCodesV66,
    S0ReviewFindingSetV66,
    authorize_s0_semantic_repair_v66,
    build_s0_repair_context_v66,
    frozen_s0_evaluation_profile_v66,
    materialize_decision_function_v66,
    materialize_regime_diagnosis_v66,
    seal_s0_review_findings_v66,
)
from fma.v6.scientific_closure import (
    materialize_scientific_closure_v62,
    scientific_closure_summary_v62,
)
from fma.v6.scientific_success import (
    materialize_scientific_success_v61,
    scientific_success_summary_v61,
)
from fma.v6.source_auth import (
    S2_SOURCE_REVERIFICATION_PATH,
    SOURCE_ACQUISITION_AUTH_PATH,
    SourceTransportAuthorityV62,
)

from .backhalf_runtime import (
    ADAPTIVE_ADAPTER_ID,
    ODE_ADAPTER_ID,
    RAW_RELATIVE_PATH,
    BackhalfRuntimeError,
    StudioBackhalfOrchestratorV59,
    StudioODEDataRequestV59,
    backhalf_summary_v59,
    ingest_ode_data_v59,
    validate_v67_data_compatibility_v67,
    validate_v67_pre_acquisition_v67,
)
from .s1_runtime import (
    S1FormalizationRejectedV67,
    S1RuntimeError,
    StudioS1OrchestratorV58,
)


_TASK_ID_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,58}[A-Za-z0-9])?")
_S0_PATHS = (
    "problem/contract.json",
    "problem/decision_function.json",
    "docs/regime.json",
)
_S0_OWNED_PATHS = (*_S0_PATHS, S0_EVALUATION_PROFILE_PATH_V66)
_S1_PATHS = (
    "docs/candidates.json",
    "docs/assumptions.json",
    "docs/symbols.json",
    "docs/model_spec.json",
    "docs/validation_plan.json",
    EXECUTABLE_CANDIDATE_INTENT_PATH,
    EXECUTABLE_CANDIDATE_IR_PATH,
    CANDIDATE_EXECUTION_BINDING_PATH_V67,
)
_OPERATOR_POLICY_V70 = {
    "schema_version": "7.0-studio-operator-policy",
    "authority_boundary": (
        "operator work state never opens a stage gate or grants scientific authority"
    ),
    "lease_fencing_required": True,
    "worker_submission_requires_authority_projection": True,
    "max_attempts": 3,
    "default_lease_seconds": 300,
    "actions": {
        "run_s0": {
            "purpose": "Materialize and independently review the S0 problem contract.",
            "write_paths": ["."],
            "expected_outputs": ["current authenticated S0 gate or typed failure"],
            "tool_profile": "codex_stage_roles_local",
        },
        "prepare_predata_v67": {
            "purpose": "Freeze typed source and measurement contracts before observation access.",
            "write_paths": ["docs", "data", ".fma"],
            "expected_outputs": ["authenticated pre-data transaction receipt"],
            "tool_profile": "typed_local_input",
        },
        "reconcile_predata_v67": {
            "purpose": "Replay the exact interrupted pre-data publication transaction.",
            "write_paths": ["docs", "data", ".fma"],
            "expected_outputs": ["reconciled pre-data transaction state"],
            "tool_profile": "deterministic_local_reconcile",
        },
        "run_s1": {
            "purpose": "Run isolated candidate branches, synthesis, and independent S1 review.",
            "write_paths": ["."],
            "expected_outputs": ["current authenticated S1 gate or typed failure"],
            "tool_profile": "codex_parallel_roles_local",
        },
        "ingest_ode_data": {
            "purpose": "Freeze user-supplied observations against the current S1 authority.",
            "write_paths": ["data", ".fma"],
            "expected_outputs": ["hash-bound raw-data baseline"],
            "tool_profile": "typed_local_input",
        },
        "ingest_world_bank_data": {
            "purpose": "Acquire and freeze the registered public source against current authority.",
            "write_paths": ["data", ".fma"],
            "expected_outputs": ["source receipt and authenticated raw-data baseline"],
            "tool_profile": "allowlisted_public_source",
        },
        "run_backhalf": {
            "purpose": "Execute and independently verify the registered S2-S6 capability path.",
            "write_paths": ["."],
            "expected_outputs": ["current authenticated S6 gate or typed failure"],
            "tool_profile": "codex_and_typed_adapters_local",
        },
        "prepare_portfolio_v69": {
            "purpose": "Freeze the development-only parallel portfolio protocol.",
            "write_paths": [".fma", "results"],
            "expected_outputs": ["portfolio protocol and intent receipt"],
            "tool_profile": "typed_local_input",
        },
        "ingest_portfolio_v69": {
            "purpose": "Bind a public series to the frozen development portfolio.",
            "write_paths": [".fma", "data", "results"],
            "expected_outputs": ["portfolio data snapshot receipt"],
            "tool_profile": "typed_local_input",
        },
        "run_portfolio_v69": {
            "purpose": "Evaluate isolated development branches under one code-owned selector.",
            "write_paths": ["."],
            "expected_outputs": ["SELECT or ABSTAIN development decision"],
            "tool_profile": "typed_development_portfolio",
        },
        "reconcile_portfolio_v69": {
            "purpose": "Replay an interrupted portfolio transaction without changing policy.",
            "write_paths": ["."],
            "expected_outputs": ["reconciled portfolio transaction state"],
            "tool_profile": "deterministic_local_reconcile",
        },
        "inspect_s0": {
            "purpose": "Inspect S0 evidence, reviews, and gate status without mutation.",
            "write_paths": [],
            "expected_outputs": ["read-only S0 status"],
            "tool_profile": "read_only",
        },
        "inspect_s1": {
            "purpose": "Inspect candidate provenance and the S1 gate without mutation.",
            "write_paths": [],
            "expected_outputs": ["read-only S1 status"],
            "tool_profile": "read_only",
        },
        "inspect_s6": {
            "purpose": "Inspect workflow evidence and claim ceilings without mutation.",
            "write_paths": [],
            "expected_outputs": ["read-only S6 delivery status"],
            "tool_profile": "read_only",
        },
        "recover": {
            "purpose": "Apply one code-owned recovery transition to the modeling graph.",
            "write_paths": ["."],
            "expected_outputs": ["typed recovery transition receipt"],
            "tool_profile": "deterministic_graph_recovery",
        },
    },
}
_OPERATOR_POLICY_HASH_V70 = sha256_value(_OPERATOR_POLICY_V70)
_OPERATOR_MUTATING_ACTIONS_V70 = frozenset(
    action
    for action, policy in _OPERATOR_POLICY_V70["actions"].items()
    if policy["write_paths"]
)


class StudioBridgeError(RuntimeError):
    """Base error returned by the local bridge."""

    error_type = "internal_error"
    http_status = 500


class StudioValidationError(StudioBridgeError):
    error_type = "invalid_arguments"
    http_status = 400


class StudioConflictError(StudioBridgeError):
    error_type = "conflict"
    http_status = 409


class StudioNotFoundError(StudioBridgeError):
    error_type = "not_found"
    http_status = 404


def _safe_validation_message(exc: ValidationError) -> str:
    """Describe invalid fields without echoing caller-supplied values."""

    issues = [
        {
            "field": ".".join(str(part) for part in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        }
        for error in exc.errors(include_input=False)
    ]
    return json.dumps(issues, ensure_ascii=False, sort_keys=True)


class DecisionUseRequestV62(StrictModel):
    """User/value-owner supplied loss policy, frozen before S0."""

    schema_version: Literal["6.2"] = "6.2"
    decision_id: Identifier
    value_owner_ref: Identifier
    action_unit: Identifier
    underage_unit_cost: float = Field(gt=0, allow_inf_nan=False)
    overage_unit_cost: float = Field(gt=0, allow_inf_nan=False)
    minimum_relative_loss_improvement: float = Field(
        default=0.05,
        ge=0,
        le=1,
        allow_inf_nan=False,
    )
    maximum_mean_normalized_regret: float = Field(
        default=0.20,
        gt=0,
        le=1,
        allow_inf_nan=False,
    )


class CreateTaskRequest(StrictModel):
    objective: str = Field(min_length=12, max_length=4000)
    workspace_id: str | None = Field(default=None, max_length=60)
    evidence_scope: Literal["development", "public_data"] = "development"
    workflow_mode: Literal["legacy", "v67"] = "legacy"
    decision_use: DecisionUseRequestV62 | None = None
    intake_id: str | None = Field(default=None, max_length=80)
    intake_manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_intake_binding(self) -> "CreateTaskRequest":
        if (self.intake_id is None) != (self.intake_manifest_hash is None):
            raise ValueError(
                "intake_id and intake_manifest_hash must be supplied together"
            )
        return self


class StudioWorkflowModeContractV67(StrictModel):
    """Code-owned task mode frozen before S0 work begins."""

    schema_version: Literal["6.7-studio-workflow-mode"] = "6.7-studio-workflow-mode"
    workspace_spec_hash: Sha256
    workflow_mode: Literal["v67"] = "v67"
    evidence_scope: Literal["development", "public_data"]
    predata_required_before_s1: Literal[True] = True
    observation_access_before_predata_permitted: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    contract_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "StudioWorkflowModeContractV67":
        if self.contract_hash and self.contract_hash != self.content_hash():
            raise ValueError("V6.7 Studio workflow-mode contract hash differs")
        return self

    def content_hash(self) -> str:
        return sha256_value(self.model_dump(mode="json", exclude={"contract_hash"}))

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if not self.contract_hash or self.contract_hash != self.content_hash():
            raise ValueError("V6.7 Studio workflow-mode contract is not sealed")

    @classmethod
    def seal(
        cls,
        *,
        workspace_spec_hash: str,
        evidence_scope: Literal["development", "public_data"],
    ) -> "StudioWorkflowModeContractV67":
        draft = cls(
            workspace_spec_hash=workspace_spec_hash,
            evidence_scope=evidence_scope,
        )
        return cls(
            **draft.model_dump(exclude={"contract_hash"}),
            contract_hash=draft.content_hash(),
        )


class StudioWorldBankDataRequestV62(StrictModel):
    """Official-source intake without caller-supplied values or source IDs."""

    schema_version: Literal["6.2"] = "6.2"
    adapter_id: Literal[
        "scalar_autonomous_ode_v52",
        "adaptive_positive_series_v57",
    ] = ODE_ADAPTER_ID
    contract_id: Identifier
    country_code: str
    indicator_id: str
    start_year: int
    end_year: int
    minimum_observations: int = 23
    state_unit: Identifier
    attribution: str = Field(min_length=10, max_length=500)
    semantic_name: str = Field(min_length=3, max_length=300)
    operational_definition: str = Field(min_length=10, max_length=2000)
    observation_time_basis: str = Field(min_length=3, max_length=300)
    aggregation_level: str = Field(min_length=3, max_length=300)
    fixture_only: bool = False


class StudioPortfolioPrepareRequestV69(StrictModel):
    """Observation-free controls for the development-only V6.9 portfolio."""

    schema_version: Literal["6.9-studio-portfolio-prepare"] = (
        "6.9-studio-portfolio-prepare"
    )
    planned_observation_count: int = Field(ge=35, le=10_000)
    state_unit: Identifier
    time_unit: Identifier
    initial_training_count: int = Field(default=34, ge=34)
    max_origins: int | None = Field(
        default=None,
        ge=1,
        le=MAXIMUM_ROLLING_ORIGINS_V69,
    )
    min_relative_improvement: float = Field(
        default=0.01,
        gt=0,
        le=1,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def validate_prepare_request(self) -> "StudioPortfolioPrepareRequestV69":
        available_origins = (
            self.planned_observation_count - self.initial_training_count
        )
        if available_origins < 1:
            raise ValueError(
                "planned_observation_count must exceed initial_training_count"
            )
        if (
            self.max_origins is not None
            and self.max_origins > available_origins
        ):
            raise ValueError(
                "max_origins exceeds the frozen planned observation window"
            )
        if len(self.time_unit) < 3:
            raise ValueError("time_unit must contain at least three characters")
        return self


class StudioPortfolioSeriesRequestV69(StrictModel):
    """Public positive scalar series for the frozen V6.9 portfolio."""

    schema_version: Literal["6.9-studio-portfolio-series"] = (
        "6.9-studio-portfolio-series"
    )
    times: list[float] = Field(min_length=35, max_length=10_000)
    observations: list[float] = Field(min_length=35, max_length=10_000)
    source_id: str = Field(min_length=3, max_length=500)
    public_data_only: Literal[True] = True

    @model_validator(mode="after")
    def validate_series_request(self) -> "StudioPortfolioSeriesRequestV69":
        if len(self.times) != len(self.observations):
            raise ValueError("portfolio times and observations must have equal length")
        if any(not math.isfinite(value) for value in self.times):
            raise ValueError("portfolio times must be finite")
        if any(
            right <= left for left, right in zip(self.times, self.times[1:])
        ):
            raise ValueError("portfolio times must be strictly increasing")
        if any(
            not math.isfinite(value) or value <= 0
            for value in self.observations
        ):
            raise ValueError("portfolio observations must be finite and positive")
        return self


class StudioRecoveryRequestV60(StrictModel):
    """A bounded failure observation; code owns rollback and action mapping."""

    schema_version: Literal["6.0"] = "6.0"
    failed_stage: StageId
    category: FailureCategoryV60
    failure_code: Identifier
    expected_information_gain: float = Field(
        default=0.5,
        ge=0,
        le=1,
        allow_inf_nan=False,
    )
    holdout_exposed: bool = False
    private_evidence_used: bool = False


class DecisionFunctionCanaryDraftV58(StrictModel):
    canary_id: Identifier
    input_values: list[float] = Field(min_length=1, max_length=8)
    expected: float = Field(allow_inf_nan=False)
    tolerance: float = Field(default=1e-9, gt=0, allow_inf_nan=False)


class DecisionFunctionDraftV58(StrictModel):
    """Structured-output-safe core; the harness restores named canary inputs."""

    schema_version: Literal["5.8"] = "5.8"
    function_id: Identifier
    input_names: list[Identifier] = Field(min_length=1, max_length=8)
    expression: str = Field(min_length=1, max_length=1000)
    sense: Literal["minimize", "maximize", "report_only"]
    output_unit: str = Field(min_length=1, max_length=200)
    canaries: list[DecisionFunctionCanaryDraftV58] = Field(
        min_length=1,
        max_length=8,
    )

    @model_validator(mode="after")
    def validate_draft(self) -> "DecisionFunctionDraftV58":
        if len(self.input_names) != len(set(self.input_names)):
            raise ValueError("input_names must be unique")
        canary_ids = [item.canary_id for item in self.canaries]
        if len(canary_ids) != len(set(canary_ids)):
            raise ValueError("canary IDs must be unique")
        if any(
            len(item.input_values) != len(self.input_names) for item in self.canaries
        ):
            raise ValueError("canary input_values must align with input_names")
        return self


class StudioEvent(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    sequence: int = Field(ge=1)
    event_type: Identifier
    status: Literal["accepted", "running", "succeeded", "failed", "blocked"]
    message: str = Field(min_length=1, max_length=500)
    details: dict[str, Any] = Field(default_factory=dict)
    recorded_at: datetime
    previous_event_hash: str | None = None
    event_hash: str


@dataclass(frozen=True)
class _OperatorRunClaimV70:
    packet: OperatorPacketV70
    lease: OperatorLeaseV70
    before_manifest: dict[str, str]


class RoleTransportFactory(Protocol):
    def __call__(self, output_root: Path) -> StageRoleTransportV51: ...


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json_document_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_temporary_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def _install_bytes_new(
    path: Path,
    payload: bytes,
    *,
    accept_identical_race: bool,
) -> None:
    if path.exists():
        if accept_identical_race and path.is_file() and path.read_bytes() == payload:
            return
        raise StudioConflictError(
            f"refusing to overwrite existing artifact: {path.name}"
        )
    temporary = _write_temporary_bytes(path, payload)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            if (
                accept_identical_race
                and path.is_file()
                and path.read_bytes() == payload
            ):
                return
            raise StudioConflictError(
                f"refusing to overwrite concurrently created artifact: {path.name}"
            ) from exc
        except OSError as exc:
            raise StudioConflictError(
                f"atomic no-replace installation failed: {path.name}"
            ) from exc
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_bytes_atomic(path: Path, payload: bytes) -> None:
    temporary = _write_temporary_bytes(path, payload)
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_new(path: Path, payload: object) -> None:
    _install_bytes_new(
        path,
        _json_document_bytes(payload),
        accept_identical_race=False,
    )


def _write_bytes_new(path: Path, payload: bytes) -> None:
    _install_bytes_new(path, payload, accept_identical_race=True)


def _safe_json(text: str, *, artifact_type: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StudioValidationError(
            f"Codex returned invalid JSON for {artifact_type}"
        ) from exc
    if not isinstance(value, dict):
        raise StudioValidationError(f"{artifact_type} must be a JSON object")
    return value


def _decision_intent_from_request(
    request: DecisionUseRequestV62 | None,
) -> DecisionValueIntentV62 | None:
    if request is None:
        return None
    return DecisionValueIntentV62.seal(
        decision_id=request.decision_id,
        value_owner_ref=request.value_owner_ref,
        action_unit=request.action_unit,
        underage_unit_cost=request.underage_unit_cost,
        overage_unit_cost=request.overage_unit_cost,
        minimum_relative_loss_improvement=(request.minimum_relative_loss_improvement),
        maximum_mean_normalized_regret=(request.maximum_mean_normalized_regret),
    )


def _load_decision_intent(
    workspace: StageWorkspaceV50,
) -> DecisionValueIntentV62 | None:
    path = workspace.root / DECISION_INTENT_PATH
    if not path.is_file():
        return None
    intent = DecisionValueIntentV62.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    intent.assert_sealed()
    return intent


PreDataBundleV67 = tuple[
    WorldBankSourceContractV62,
    MeasurementStudyDesignContractV67,
    PreDataExecutionProtocolV67,
]


@dataclass(frozen=True)
class _PreDataTransactionStateV67:
    status: Literal[
        "NOT_STARTED",
        "RECOVERY_PENDING",
        "STALE_PENDING",
        "COMPLETED",
        "LEGACY_COMPLETED",
    ]
    bundle: PreDataBundleV67 | None = None
    intent_ref: ArtifactRef | None = None
    intent: PreDataPreparationIntentV67 | None = None
    preparation_ref: ArtifactRef | None = None
    preparation_payload: dict[str, Any] | None = None
    completion_ref: ArtifactRef | None = None
    completion: PreDataPreparationCompletionV67 | None = None


def world_bank_source_contract_from_studio_request_v67(
    request: StudioWorldBankDataRequestV62,
) -> WorldBankSourceContractV62:
    """Project the exact public source semantics without opening the source."""

    return WorldBankSourceContractV62.seal(
        contract_id=request.contract_id,
        country_code=request.country_code,
        indicator_id=request.indicator_id,
        start_year=request.start_year,
        end_year=request.end_year,
        minimum_observations=request.minimum_observations,
        state_unit=request.state_unit,
        attribution=request.attribution,
        fixture_only=request.fixture_only,
    )


def build_world_bank_predata_bundle_v67(
    *,
    request: StudioWorldBankDataRequestV62,
    workspace_spec_hash: Sha256,
    s0_gate_hash: Sha256,
) -> PreDataBundleV67:
    """Compile the code-owned source, study, and execution contracts pre-data."""

    source = world_bank_source_contract_from_studio_request_v67(request)
    source.assert_sealed()
    if source.contract_hash is None:
        raise ValueError("sealed source contract has no hash")

    construct_id = f"{request.contract_id}.construct"
    measurement = MeasurementStudyDesignContractV67.seal(
        contract_id=f"{request.contract_id}.measurement-study-v67",
        workspace_spec_hash=workspace_spec_hash,
        s0_gate_hash=s0_gate_hash,
        source_contract_id=source.contract_id,
        source_contract_hash=source.contract_hash,
        claim_kind="predictive",
        claim_scope=(
            f"Predict future values of the registered {request.semantic_name} "
            f"series for {request.country_code} under indicator "
            f"{request.indicator_id}; do not infer causes, mechanisms, policy "
            "effects, individual outcomes, or real-world actions."
        ),
        construct_definition=ConstructDefinitionV67(
            construct_id=construct_id,
            name=request.semantic_name,
            conceptual_definition=(
                "The target construct is intentionally bounded to the exact "
                f"registered official indicator: {request.operational_definition}"
            ),
            role="outcome",
            representation="direct",
            representation_rationale=(
                "The modelling target is the published indicator itself, not "
                "a broader latent construct or an unregistered substitute."
            ),
        ),
        measurement=MeasurementDefinitionV67(
            measurement_id=request.contract_id,
            construct_id=construct_id,
            operational_definition=request.operational_definition,
            unit=request.state_unit,
            time_basis=request.observation_time_basis,
            aggregation_basis=request.aggregation_level,
            scale_type="ratio",
            source_definition=(
                "World Bank World Development Indicators series "
                f"{request.indicator_id} for {request.country_code}, bound to "
                f"source contract {source.contract_id}."
            ),
            directionality="higher_is_more",
        ),
        population=PopulationDefinitionV67(
            population_id=f"{request.contract_id}.population",
            target_population=(
                "All eligible country-year records for "
                f"{request.country_code} under the unchanged registered "
                f"indicator definition from {request.start_year} through "
                f"{request.end_year}."
            ),
            unit_of_analysis="registered country-year",
            spatial_scope=f"country aggregate for {request.country_code}",
            temporal_scope=(
                f"calendar years {request.start_year} through {request.end_year}"
            ),
            inclusion_criteria=[
                "Record uses the exact registered country and indicator",
                "Record year lies inside the frozen source interval",
                "Record contains one finite positive annual value",
            ],
            exclusion_criteria=[
                "Duplicate country-year record",
                "Missing, non-finite, or non-positive value",
                "Record outside the frozen indicator definition",
            ],
        ),
        sampling=SamplingPlanV67(
            sampling_frame=(
                "Every eligible annual record returned by the exact frozen "
                "World Bank source contract and interval."
            ),
            sampling_method="administrative_complete_series",
            selection_rule=(
                "Use every eligible year in the frozen interval in ascending "
                "order; no post-result deletion, extension, or cherry-picking."
            ),
            minimum_sample_size=request.minimum_observations,
            stopping_rule=(
                f"Stop at the predeclared end year {request.end_year}; do not "
                "extend the series after model or verifier results are seen."
            ),
            representativeness_limitations=(
                "Country-level administrative aggregates do not represent "
                "individuals, subnational units, or another indicator regime."
            ),
        ),
        missingness=MissingnessPlanV67(
            anticipated_sources=[
                "The official series may omit an eligible country-year",
                "A provider revision may remove a previously published value",
            ],
            mechanism_assumptions=[
                "Missingness may depend on source production processes",
                "Missingness is not assumed random without separate evidence",
            ],
            handling_policy="reject_incomplete_series",
            sensitivity_analysis_plan=(
                "Reject the frozen adapter before fitting and require a new "
                "graph attempt and successor protocol before any imputation "
                "or alternative missing-data handling."
            ),
        ),
        measurement_error=MeasurementErrorPlanV67(
            anticipated_error_sources=[
                "Provider revisions to historical values",
                "Changes in the registered indicator methodology",
                "Rounding or aggregation in the official series",
            ],
            error_structure_assumption=(
                "Magnitude, direction, and temporal dependence of measurement "
                "error are unknown before independent calibration evidence."
            ),
            calibration_or_reference_plan=(
                "Bind the provider metadata and compare authenticated vintages "
                "only if a separate registered calibration adapter is added."
            ),
            propagation_or_sensitivity_plan=(
                "Treat uncalibrated error as a claim limitation and execute "
                "only predeclared perturbation checks after source binding."
            ),
        ),
        bias=BiasPlanV67(
            anticipated_biases=[
                "Post-selection of the modelling interval",
                "Publication and revision bias in the available history",
                "Country aggregation masking within-country variation",
            ],
            mitigation_plan=(
                "Freeze source, interval, exclusions, validation design, and "
                "thresholds before observation access; preserve gate failures."
            ),
            residual_bias_policy=(
                "Report unresolved bias and prevent elevation to causal, "
                "individual, policy, welfare, or intervention claims."
            ),
        ),
        confounding=ConfoundingPlanV67(
            relevance="not_applicable_to_noncausal_claim",
            identification_or_control_strategy=(
                "This task identifies no causal effect and controls no "
                "confounder; it is a predictive time-series study only."
            ),
            unmeasured_confounding_policy=(
                "Predictive association cannot be interpreted as a mechanism "
                "or intervention effect."
            ),
        ),
        study_design=StudyDesignV67(
            design_type="time_series",
            target_quantity=(
                "A next-step predictive distribution for the exact registered "
                "annual indicator within the declared applicability boundary."
            ),
            temporal_ordering=(
                "Training observations must precede every held-out forecast "
                "origin in calendar time."
            ),
            comparison_strategy=(
                "Compare registered candidates with frozen naive baselines at "
                "the identical rolling origins."
            ),
            validation_design=(
                "Use leakage-safe rolling-origin confirmation with adapter, "
                "thresholds, folds, and stopping rules frozen before data."
            ),
            leakage_prevention_plan=(
                "Fit and select on each training prefix only; do not expose "
                "future folds or private acceptance outcomes to generators."
            ),
        ),
        applicability=ApplicabilityBoundaryV67(
            intended_use=(
                "Local retrospective evaluation and bounded next-step "
                "forecasting of the same registered annual indicator."
            ),
            in_scope_conditions=[
                "Same indicator definition and state unit",
                "Same country aggregation and annual time basis",
                "Finite positive series satisfying the frozen adapter",
            ],
            out_of_scope_conditions=[
                "Causal attribution or mechanism discovery",
                "Individual or subnational inference",
                "Policy recommendation or automated action",
            ],
            transport_assumptions=[
                "Provider metadata remain definitionally compatible",
                "Forecast regime is represented by the frozen history",
            ],
            abstention_conditions=[
                "Indicator definition, unit, or aggregation changes",
                "Required history or rolling folds are unavailable",
                "Any registered compatibility or validation gate fails",
            ],
        ),
        ethics=EthicsBoundaryV67(
            risk_level="minimal",
            human_participant_data_expected=False,
            sensitive_data_expected=False,
            consent_or_legal_basis_plan=(
                "Use only the declared aggregate public indicator under its "
                "recorded license and attribution terms."
            ),
            prohibited_uses=[
                "Individual eligibility or benefit decision",
                "Automated policy or operational action",
                "Causal, clinical, or welfare claim",
            ],
            ethics_review_required=False,
        ),
    )
    measurement.assert_sealed()
    capability_pack = registered_positive_series_capability_pack_v67(request.adapter_id)
    protocol = compile_predata_execution_protocol_v67(
        measurement_contract=measurement,
        capability_pack=capability_pack,
    )
    protocol.assert_sealed()
    return source, measurement, protocol


class StudioTaskService:
    """Single-process bridge service with fail-closed task-level concurrency."""

    def __init__(
        self,
        task_root: str | Path,
        *,
        authority_key: bytes,
        authority_key_id: str,
        codex_config: CodexCLIConfig | None = None,
        codex_process_runner: ProcessRunner | None = None,
        codex_cli_locator: CliLocator | None = None,
        role_transport_factory: RoleTransportFactory | None = None,
        world_bank_fetcher: SourceFetcherV62 | None = None,
    ) -> None:
        if len(authority_key) < 32:
            raise ValueError("authority_key must contain at least 32 bytes")
        self.task_root = Path(task_root).resolve()
        self.task_root.mkdir(parents=True, exist_ok=True)
        self.authority_key = bytes(authority_key)
        self.authority_key_id = authority_key_id
        self.codex_config = codex_config or CodexCLIConfig()
        self.codex_process_runner = codex_process_runner
        self.codex_cli_locator = codex_cli_locator
        self.role_transport_factory = role_transport_factory
        self.world_bank_fetcher = world_bank_fetcher
        self.operator_store = OperatorStoreV70(self.task_root)
        self._lock = threading.RLock()
        self._active_tasks: set[str] = set()

    def _task_path(self, task_id: str) -> Path:
        if not _TASK_ID_PATTERN.fullmatch(task_id) or task_id in {".", ".."}:
            raise StudioValidationError("task_id is not a safe identifier")
        path = (self.task_root / task_id).resolve(strict=False)
        try:
            path.relative_to(self.task_root)
        except ValueError as exc:
            raise StudioValidationError("task path escapes configured root") from exc
        return path

    def _workspace(self, task_id: str) -> StageWorkspaceV50:
        root = self._task_path(task_id)
        if not root.is_dir():
            raise StudioNotFoundError(f"task not found: {task_id}")
        return StageWorkspaceV50.open_existing(
            root,
            authority_key=self.authority_key,
            authority_key_id=self.authority_key_id,
        )

    def _operator_authority_binding_v70(
        self,
        workspace: StageWorkspaceV50,
    ) -> OperatorAuthorityBindingV70:
        if workspace.spec.spec_hash is None:
            raise StudioConflictError("operator projection requires a sealed workspace")
        if not workspace.verify():
            raise StudioConflictError(
                "operator projection refuses an invalid authority workspace"
            )
        intake_records = workspace._artifacts_of_kind(
            "studio_intake_manifest_v70"
        )
        installed_intake = workspace.root / "problem" / "intake"
        if len(intake_records) > 1:
            raise StudioConflictError(
                "operator projection found multiple committed intake manifests"
            )
        if intake_records:
            try:
                intake = IntakeManifestV70.model_validate(
                    intake_records[0][1]
                )
                self.operator_store.verify_materialized_intake(
                    intake.intake_id,
                    workspace.root,
                )
            except (ValueError, OperatorPlaneError) as exc:
                raise StudioConflictError(
                    "operator projection refuses a changed workspace intake"
                ) from exc
        elif installed_intake.exists():
            raise StudioConflictError(
                "operator projection refuses an unbound workspace intake"
            )
        state = workspace.graph.project_state()
        status = workspace.status()
        return OperatorAuthorityBindingV70.seal(
            workspace_id=workspace.spec.workspace_id,
            graph_id=workspace.spec.graph_id,
            workspace_spec_hash=workspace.spec.spec_hash,
            graph_snapshot_hash=state.snapshot.snapshot_hash,
            frontier_node_hashes=tuple(
                sorted(state.snapshot.frontier_node_hashes)
            ),
            stage_statuses=dict(status.stage_statuses),
            current_gate_hashes=dict(status.current_gate_hashes),
            frontier_stages=tuple(status.frontier_stages),
            operator_policy_hash=_OPERATOR_POLICY_HASH_V70,
        )

    def _operator_packet_v70(
        self,
        workspace: StageWorkspaceV50,
        action: str,
    ) -> OperatorPacketV70:
        try:
            policy = cast(dict[str, Any], _OPERATOR_POLICY_V70["actions"][action])
        except KeyError as exc:
            raise StudioValidationError(
                f"operator action is not registered: {action}"
            ) from exc
        binding = self._operator_authority_binding_v70(workspace)
        idempotency_key = "studio-v70:" + sha256_value(
            {
                "workspace_id": workspace.spec.workspace_id,
                "action": action,
                "authority_binding_hash": binding.binding_hash,
                "operator_policy_hash": _OPERATOR_POLICY_HASH_V70,
            }
        )
        return OperatorPacketV70.seal(
            workspace_id=workspace.spec.workspace_id,
            action=action,
            purpose=policy["purpose"],
            authority_binding=binding,
            read_paths=(".",),
            write_paths=tuple(policy["write_paths"]),
            allowed_tool_profile=policy["tool_profile"],
            expected_outputs=tuple(policy["expected_outputs"]),
            max_attempts=cast(int, _OPERATOR_POLICY_V70["max_attempts"]),
            lease_seconds=cast(
                int, _OPERATOR_POLICY_V70["default_lease_seconds"]
            ),
            max_wall_seconds=1800,
            idempotency_key=idempotency_key,
        )

    @staticmethod
    def _preferred_operator_action_v70(actions: list[str]) -> str | None:
        for action in actions:
            if action in _OPERATOR_MUTATING_ACTIONS_V70:
                return action
        return actions[0] if actions else None

    def project_next_packet_v70(
        self,
        task_id: str,
        *,
        next_valid_actions: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Pure read-only projection of the next bounded work packet."""

        workspace = self._workspace(task_id)
        actions = next_valid_actions
        if actions is None:
            actions = cast(
                list[str],
                self.snapshot(
                    task_id,
                    _include_operator_packet=False,
                )["next_valid_actions"],
            )
        action = self._preferred_operator_action_v70(actions)
        if action is None:
            return None
        packet = self._operator_packet_v70(workspace, action)
        return packet.model_dump(mode="json")

    def _prepare_operator_run_v70(
        self,
        task_id: str,
        action: str,
    ) -> _OperatorRunClaimV70:
        workspace = self._workspace(task_id)
        packet = self._operator_packet_v70(workspace, action)
        work = self.operator_store.ensure_work(packet)
        current = self._operator_authority_binding_v70(self._workspace(task_id))
        if current.binding_hash != packet.authority_binding.binding_hash:
            raise StudioConflictError(
                "authority graph changed before operator work could be claimed"
            )
        worker_id = (
            f"studio-{os.getpid()}-{threading.get_ident()}-{uuid4().hex[:12]}"
        )
        try:
            lease = self.operator_store.claim(
                cast(str, work["work_id"]),
                worker_id=worker_id,
                lease_seconds=packet.lease_seconds,
            )
        except OperatorConflictError as exc:
            raise StudioConflictError(str(exc)) from exc
        except OperatorPlaneError as exc:
            raise StudioBridgeError(str(exc)) from exc
        try:
            before = capture_file_manifest(self._task_path(task_id))
        except Exception as exc:
            try:
                self.operator_store.fail(
                    lease,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            except OperatorPlaneError:
                pass
            raise StudioBridgeError(
                "operator could not capture the pre-run workspace manifest"
            ) from exc
        return _OperatorRunClaimV70(
            packet=packet,
            lease=lease,
            before_manifest=before,
        )

    @staticmethod
    def _operator_authority_receipt_v70(
        *,
        action: str,
        workspace: StageWorkspaceV50,
        result: dict[str, Any],
        output_binding: OperatorAuthorityBindingV70,
    ) -> tuple[bool, str, tuple[str, ...]]:
        stage_for_action = {
            "run_s0": "S0",
            "run_s1": "S1",
            "run_backhalf": "S6",
        }
        stage = stage_for_action.get(action)
        if stage is not None:
            gate_hash = workspace.current_gate(cast(StageId, stage))
            if gate_hash is None:
                return (
                    False,
                    cast(str, output_binding.binding_hash),
                    (f"{stage.lower()}_gate_not_open",),
                )
            return True, gate_hash, ()
        if action == "run_portfolio_v69":
            portfolio = cast(dict[str, Any], result.get("portfolio_v69") or {})
            run_hash = portfolio.get("run_hash")
            completed = (
                portfolio.get("transaction_status") == "COMPLETED"
                and isinstance(run_hash, str)
                and len(run_hash) == 64
            )
            return (
                completed,
                run_hash if completed else cast(str, output_binding.binding_hash),
                () if completed else ("portfolio_not_completed",),
            )
        return (
            workspace.verify(),
            cast(str, output_binding.binding_hash),
            () if workspace.verify() else ("authority_workspace_invalid",),
        )

    def _execute_operator_run_v70(
        self,
        claim: _OperatorRunClaimV70,
        callback: Any,
    ) -> dict[str, Any]:
        stop_heartbeat = threading.Event()
        heartbeat_errors: list[Exception] = []

        def heartbeat_worker() -> None:
            interval = max(10, claim.packet.lease_seconds // 3)
            while not stop_heartbeat.wait(interval):
                try:
                    self.operator_store.heartbeat(
                        claim.lease,
                        lease_seconds=claim.packet.lease_seconds,
                    )
                except Exception as exc:
                    heartbeat_errors.append(exc)
                    return

        heartbeat_thread = threading.Thread(
            target=heartbeat_worker,
            name=f"fma-operator-heartbeat-{claim.lease.work_id}",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            result = cast(dict[str, Any], callback())
        except Exception as exc:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=1)
            try:
                self.operator_store.fail(
                    claim.lease,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            except OperatorLeaseError:
                pass
            raise
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=1)
        workspace = self._workspace(claim.packet.workspace_id)
        output_binding = self._operator_authority_binding_v70(workspace)
        after = capture_file_manifest(workspace.root)
        changed_paths = changed_manifest_paths(claim.before_manifest, after)
        submission = OperatorSubmissionV70.seal(
            work_id=claim.lease.work_id,
            packet_hash=cast(str, claim.packet.packet_hash),
            input_binding_hash=cast(
                str, claim.packet.authority_binding.binding_hash
            ),
            output_binding=output_binding,
            before_manifest_hash=sha256_value(claim.before_manifest),
            after_manifest_hash=sha256_value(after),
            changed_paths=changed_paths,
            result_summary={
                "action": claim.packet.action,
                "workflow_stage_statuses": result.get("workflow", {}).get(
                    "stage_statuses", {}
                ),
                "authority_workspace_verified": workspace.verify(),
                "heartbeat_error_types": [
                    type(error).__name__ for error in heartbeat_errors
                ],
            },
            submitted_at=_utc_now().isoformat(),
        )
        self.operator_store.submit(claim.lease, submission)
        accepted, receipt_hash, reasons = self._operator_authority_receipt_v70(
            action=claim.packet.action,
            workspace=workspace,
            result=result,
            output_binding=output_binding,
        )
        self.operator_store.project_authority_decision(
            claim.lease.work_id,
            accepted=accepted,
            authority_receipt_hash=receipt_hash,
            reason_codes=reasons,
        )
        return result

    def _fail_operator_claim_v70(
        self,
        claim: _OperatorRunClaimV70,
        exc: BaseException,
    ) -> None:
        try:
            self.operator_store.fail(
                claim.lease,
                error_type=type(exc).__name__,
                message=str(exc),
            )
        except OperatorPlaneError:
            return

    def reconcile_operator_v70(self) -> dict[str, Any]:
        """Repair only exact submitted effects from authenticated graph facts."""

        expired = self.operator_store.reconcile_expired()
        reconciled: list[str] = []
        work_items = self.operator_store.list_work()
        ambiguous: list[str] = [
            cast(str, item["work_id"])
            for item in work_items
            if item["status"] == "RECOVERY_PENDING"
        ]
        for item in work_items:
            if item["status"] != "SUBMITTED" or item["submission"] is None:
                continue
            packet = OperatorPacketV70.model_validate(item["packet"])
            submission = OperatorSubmissionV70.model_validate(
                item["submission"]
            )
            try:
                workspace = self._workspace(packet.workspace_id)
                current_binding = self._operator_authority_binding_v70(workspace)
                current_manifest_hash = sha256_value(
                    capture_file_manifest(workspace.root)
                )
            except Exception:
                ambiguous.append(cast(str, item["work_id"]))
                continue
            if (
                submission.output_binding.binding_hash
                != current_binding.binding_hash
                or submission.after_manifest_hash != current_manifest_hash
            ):
                ambiguous.append(cast(str, item["work_id"]))
                continue
            output_binding = submission.output_binding
            accepted, receipt_hash, reasons = (
                self._operator_authority_receipt_v70(
                    action=packet.action,
                    workspace=workspace,
                    result=self.snapshot(
                        packet.workspace_id,
                        _include_operator_packet=False,
                    ),
                    output_binding=output_binding,
                )
            )
            self.operator_store.reconcile_authority_effect(
                cast(str, item["work_id"]),
                output_binding=output_binding,
                authority_receipt_hash=receipt_hash,
                reason_code=(
                    "exact_submitted_authority_projection_recovered"
                    if accepted
                    else (
                        reasons[0]
                        if reasons
                        else "exact_submitted_authority_rejection_recovered"
                    )
                ),
                accepted=accepted,
            )
            reconciled.append(cast(str, item["work_id"]))
        return {
            "status": "success",
            "schema_version": "7.0-operator-reconcile",
            "expired_work_ids": expired,
            "authority_reconciled_work_ids": reconciled,
            "authority_ambiguous_work_ids": sorted(set(ambiguous)),
            "scientific_qualification_granted": False,
            "real_world_action_authorized": False,
        }

    def operator_doctor_v70(self) -> dict[str, Any]:
        operational = self.operator_store.doctor()
        authority_errors: dict[str, str] = {}
        checked = 0
        for child in sorted(self.task_root.iterdir()):
            if not child.is_dir() or child.name == self.operator_store.root.name:
                continue
            workspace_spec = child / ".fma" / "workspace_spec.json"
            if not workspace_spec.exists():
                if (child / ".fma").exists():
                    authority_errors[child.name] = "workspace_spec_missing"
                continue
            checked += 1
            try:
                workspace = self._workspace(child.name)
                self._operator_authority_binding_v70(workspace)
            except Exception as exc:
                authority_errors[child.name] = type(exc).__name__
        status = (
            "FAIL"
            if operational["status"] == "FAIL" or authority_errors
            else operational["status"]
        )
        return {
            "status": status,
            "schema_version": "7.0-studio-doctor",
            "operational": operational,
            "authority": {
                "status": "FAIL" if authority_errors else "PASS",
                "workspace_count": checked,
                "errors": authority_errors,
            },
            "claim_scope": "workflow_control_only",
            "scientific_qualification_granted": False,
            "real_world_action_authorized": False,
        }

    def _event_path(self, task_id: str) -> Path:
        return self._task_path(task_id) / ".fma" / "studio_events.jsonl"

    def _event_lock_path(self, task_id: str) -> Path:
        return self._task_path(task_id) / ".fma" / ".studio_events.writer.lock"

    def _events_unlocked(self, task_id: str) -> list[StudioEvent]:
        path = self._event_path(task_id)
        if not path.is_file():
            return []
        events: list[StudioEvent] = []
        previous: str | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            event = StudioEvent.model_validate_json(line)
            payload = event.model_dump(mode="json", exclude={"event_hash"})
            expected = sha256_value(payload)
            if event.previous_event_hash != previous or event.event_hash != expected:
                raise StudioBridgeError("studio event chain verification failed")
            events.append(event)
            previous = event.event_hash
        return events

    def _events(self, task_id: str) -> list[StudioEvent]:
        with exclusive_file_lock(self._event_lock_path(task_id)):
            return self._events_unlocked(task_id)

    def _append_event(
        self,
        task_id: str,
        *,
        event_type: str,
        status: Literal["accepted", "running", "succeeded", "failed", "blocked"],
        message: str,
        details: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> StudioEvent:
        with self._lock:
            with exclusive_file_lock(self._event_lock_path(task_id)):
                events = self._events_unlocked(task_id)
                event_details = dict(details or {})
                if idempotency_key is not None:
                    event_details["idempotency_key"] = idempotency_key
                if idempotency_key is not None:
                    matches = [
                        event
                        for event in events
                        if event.details.get("idempotency_key") == idempotency_key
                    ]
                    if len(matches) > 1:
                        raise StudioBridgeError(
                            "studio event idempotency chain contains duplicates"
                        )
                    if matches:
                        existing = matches[0]
                        if (
                            existing.event_type != event_type
                            or existing.status != status
                            or existing.message != message
                            or existing.details != event_details
                        ):
                            raise StudioBridgeError(
                                "studio event idempotency key was reused with "
                                "different content"
                            )
                        return existing
                payload = {
                    "schema_version": "1.0",
                    "sequence": len(events) + 1,
                    "event_type": event_type,
                    "status": status,
                    "message": message,
                    "details": event_details,
                    "recorded_at": _utc_now(),
                    "previous_event_hash": (events[-1].event_hash if events else None),
                }
                unsigned = StudioEvent(**payload, event_hash="0" * 64)
                event = StudioEvent(
                    **payload,
                    event_hash=sha256_value(
                        unsigned.model_dump(
                            mode="json",
                            exclude={"event_hash"},
                        )
                    ),
                )
                path = self._event_path(task_id)
                serialized = "".join(
                    canonical_json(item.model_dump(mode="json")) + "\n"
                    for item in (*events, event)
                ).encode("utf-8")
                _replace_bytes_atomic(path, serialized)
                return event

    @staticmethod
    def _load_predata_bundle_v67(
        workspace: StageWorkspaceV50,
    ) -> PreDataBundleV67 | None:
        paths = (
            workspace.root / SOURCE_CONTRACT_PATH,
            workspace.root / MEASUREMENT_STUDY_DESIGN_PATH_V67,
            workspace.root / PREDATA_EXECUTION_PROTOCOL_PATH_V67,
        )
        present = [path.is_file() for path in paths]
        if not any(present):
            return None
        if present == [True, False, False] and (
            workspace.current_gate("S1") is not None
            or any(
                (workspace.root / relative).exists()
                for relative in (
                    RAW_RELATIVE_PATH,
                    SOURCE_RAW_PATH,
                    SOURCE_RECEIPT_PATH,
                )
            )
        ):
            # A V6.2 source contract created after legacy S1 is not a partial
            # V6.7 bundle.  Preserve that additive path.
            return None
        if not all(present):
            raise StudioConflictError(
                "V6.7 pre-data contract bundle is only partially materialized"
            )
        try:
            source = WorldBankSourceContractV62.model_validate_json(
                paths[0].read_text(encoding="utf-8")
            )
            measurement = MeasurementStudyDesignContractV67.model_validate_json(
                paths[1].read_text(encoding="utf-8")
            )
            protocol = PreDataExecutionProtocolV67.model_validate_json(
                paths[2].read_text(encoding="utf-8")
            )
            source.assert_sealed()
            measurement.assert_sealed()
            protocol.assert_sealed()
        except (OSError, TypeError, ValueError) as exc:
            raise StudioConflictError(
                "V6.7 pre-data contract bundle cannot be replayed"
            ) from exc
        workspace_hash = workspace.spec.spec_hash
        s0_gate_hash = workspace.current_gate("S0")
        if workspace_hash is None or s0_gate_hash is None:
            raise StudioConflictError(
                "V6.7 pre-data contracts require a current S0 authority"
            )
        if (
            measurement.workspace_spec_hash != workspace_hash
            or protocol.workspace_spec_hash != workspace_hash
            or measurement.s0_gate_hash != s0_gate_hash
            or protocol.s0_gate_hash != s0_gate_hash
        ):
            raise StudioConflictError(
                "V6.7 pre-data contracts reference stale workspace authority"
            )
        if (
            measurement.source_contract_id != source.contract_id
            or measurement.source_contract_hash != source.contract_hash
            or protocol.source_contract_id != source.contract_id
            or protocol.source_contract_hash != source.contract_hash
            or protocol.measurement_contract_id != measurement.contract_id
            or protocol.measurement_contract_hash != measurement.contract_hash
        ):
            raise StudioConflictError(
                "V6.7 source, measurement, and execution contracts differ"
            )
        try:
            capability_pack = registered_positive_series_capability_pack_v67(
                protocol.adapter_binding.adapter_id
            )
        except ValueError as exc:
            raise StudioConflictError(
                "V6.7 protocol capability pack is unavailable"
            ) from exc
        if not verify_predata_execution_protocol_v67(
            measurement_contract=measurement,
            capability_pack=capability_pack,
            protocol=protocol,
        ):
            raise StudioConflictError(
                "V6.7 pre-data protocol failed deterministic replay"
            )
        return source, measurement, protocol

    @staticmethod
    def _workflow_mode_contract_v67(
        workspace: StageWorkspaceV50,
    ) -> tuple[ArtifactRef, StudioWorkflowModeContractV67] | None:
        try:
            records = workspace._artifacts_of_kind(
                "studio_workflow_mode_v67",
                StudioWorkflowModeContractV67,
            )
            for _, contract in records:
                contract.assert_sealed()
        except (TypeError, ValidationError, ValueError) as exc:
            raise StudioConflictError(
                "V6.7 Studio workflow-mode contract is invalid"
            ) from exc
        if len(records) > 1:
            raise StudioConflictError(
                "multiple V6.7 Studio workflow-mode contracts are committed"
            )
        if not records:
            return None
        reference, contract = records[0]
        if (
            contract.workspace_spec_hash != workspace.spec.spec_hash
            or contract.evidence_scope != workspace.spec.evidence_scope
        ):
            raise StudioConflictError("V6.7 Studio workflow-mode contract is stale")
        return reference, contract

    @classmethod
    def _predata_transaction_policy_v67(
        cls,
        workspace: StageWorkspaceV50,
    ) -> tuple[ArtifactRef, PreDataTransactionPolicyV67] | None:
        try:
            records = workspace._artifacts_of_kind(
                PREDATA_TRANSACTION_POLICY_KIND_V67,
                PreDataTransactionPolicyV67,
            )
            for _, policy in records:
                policy.assert_sealed()
        except (TypeError, ValidationError, ValueError) as exc:
            raise StudioConflictError(
                "V6.7 pre-data transaction policy is invalid"
            ) from exc
        if len(records) > 1:
            raise StudioConflictError(
                "multiple V6.7 pre-data transaction policies are committed"
            )
        if not records:
            return None
        workflow_mode = cls._workflow_mode_contract_v67(workspace)
        if workflow_mode is None:
            raise StudioConflictError(
                "V6.7 pre-data transaction policy lacks workflow authority"
            )
        workflow_ref, workflow_contract = workflow_mode
        reference, policy = records[0]
        if (
            policy.workspace_spec_hash != workspace.spec.spec_hash
            or policy.workflow_mode_contract_hash != workflow_contract.contract_hash
            or policy.workflow_mode_artifact_hash != workflow_ref.sha256
            or policy.evidence_scope != workspace.spec.evidence_scope
            or policy.authority_key_id != workspace.authority_key_id
            or not workspace._verify_mac(
                PREDATA_TRANSACTION_POLICY_KIND_V67,
                policy.unsigned_hash(),
                policy.authority_auth_tag,
            )
        ):
            raise StudioConflictError(
                "V6.7 pre-data transaction policy failed authority verification"
            )
        return reference, policy

    @staticmethod
    def _predata_policy_installation_is_pristine_v67(
        workspace: StageWorkspaceV50,
    ) -> bool:
        """Limit crash recovery to a task that has not started S0 or pre-data work."""

        if workspace.current_gate("S0") is not None:
            return False
        if workspace.status().stage_statuses.get("S0") != "frontier":
            return False
        if any(
            (workspace.root / relative_path).exists()
            for relative_path in (
                SOURCE_CONTRACT_PATH,
                MEASUREMENT_STUDY_DESIGN_PATH_V67,
                PREDATA_EXECUTION_PROTOCOL_PATH_V67,
            )
        ):
            return False
        return not any(
            workspace._artifacts_of_kind(kind)
            for kind in (
                PREDATA_PREPARATION_INTENT_KIND_V67,
                PREDATA_PREPARATION_EVIDENCE_KIND_V67,
                PREDATA_PREPARATION_COMPLETION_KIND_V67,
            )
        )

    @classmethod
    def _commit_predata_transaction_policy_v67(
        cls,
        workspace: StageWorkspaceV50,
    ) -> tuple[ArtifactRef, PreDataTransactionPolicyV67]:
        existing = cls._predata_transaction_policy_v67(workspace)
        if existing is not None:
            return existing
        workflow_mode = cls._workflow_mode_contract_v67(workspace)
        if workflow_mode is None or workspace.spec.spec_hash is None:
            raise StudioConflictError(
                "V6.7 pre-data transaction policy requires workflow authority"
            )
        workflow_ref, workflow_contract = workflow_mode
        policy_draft = PreDataTransactionPolicyV67(
            workspace_spec_hash=workspace.spec.spec_hash,
            workflow_mode_contract_hash=str(workflow_contract.contract_hash),
            workflow_mode_artifact_hash=workflow_ref.sha256,
            evidence_scope=workspace.spec.evidence_scope,
            authority_key_id=workspace.authority_key_id,
        )
        policy = policy_draft.authenticate(
            workspace._mac(
                PREDATA_TRANSACTION_POLICY_KIND_V67,
                policy_draft.unsigned_hash(),
            )
        )
        policy.assert_sealed()
        reference = cls._commit_evidence_once(
            workspace,
            PREDATA_TRANSACTION_POLICY_KIND_V67,
            policy.model_dump(mode="json"),
        )
        return reference, policy

    @classmethod
    def _effective_workflow_mode(
        cls,
        workspace: StageWorkspaceV50,
    ) -> Literal["legacy", "v67"]:
        if cls._workflow_mode_contract_v67(workspace) is not None:
            return "v67"
        legacy_v67_paths = (
            workspace.root / SOURCE_CONTRACT_PATH,
            workspace.root / MEASUREMENT_STUDY_DESIGN_PATH_V67,
            workspace.root / PREDATA_EXECUTION_PROTOCOL_PATH_V67,
        )
        if all(path.is_file() for path in legacy_v67_paths) and (
            workspace._artifacts_of_kind(PREDATA_PREPARATION_EVIDENCE_KIND_V67)
        ):
            # Additive migration for V6.7 bundles created before task mode was
            # made explicit.  Raw file presence alone never selects V6.7.
            return "v67"
        return "legacy"

    @staticmethod
    def _predata_preparation_record_v67(
        workspace: StageWorkspaceV50,
        bundle: PreDataBundleV67,
    ) -> tuple[ArtifactRef, dict[str, Any]]:
        source, measurement, protocol = bundle
        workspace_spec_hash = workspace.spec.spec_hash
        s0_gate_hash = workspace.current_gate("S0")
        if workspace_spec_hash is None or s0_gate_hash is None:
            raise StudioConflictError(
                "V6.7 pre-data preparation lacks current workspace authority"
            )
        expected = predata_preparation_payload_v67(
            workspace_spec_hash=workspace_spec_hash,
            s0_gate_hash=s0_gate_hash,
            source_contract=source,
            measurement_contract=measurement,
            predata_protocol=protocol,
        )
        matches: list[tuple[ArtifactRef, dict[str, Any]]] = []
        for reference, payload in workspace._artifacts_of_kind(
            PREDATA_PREPARATION_EVIDENCE_KIND_V67
        ):
            if not isinstance(payload, dict):
                raise StudioConflictError(
                    "V6.7 pre-data preparation evidence is invalid"
                )
            if (
                payload.get("workspace_spec_hash") != workspace_spec_hash
                or payload.get("s0_gate_hash") != s0_gate_hash
                or payload.get("source_contract_hash") != source.contract_hash
                or payload.get("measurement_contract_hash") != measurement.contract_hash
                or payload.get("protocol_hash") != protocol.protocol_hash
            ):
                continue
            matches.append((reference, payload))
        if len(matches) != 1:
            raise StudioConflictError(
                "V6.7 pre-data bundle lacks one authoritative preparation record"
            )
        reference, payload = matches[0]
        if payload != expected:
            raise StudioConflictError(
                "V6.7 pre-data preparation evidence differs from its bundle"
            )
        return reference, payload

    @staticmethod
    def _verified_predata_intents_v67(
        workspace: StageWorkspaceV50,
    ) -> list[tuple[ArtifactRef, PreDataPreparationIntentV67]]:
        try:
            records = workspace._artifacts_of_kind(
                PREDATA_PREPARATION_INTENT_KIND_V67,
                PreDataPreparationIntentV67,
            )
            workflow_mode = StudioTaskService._workflow_mode_contract_v67(workspace)
            for _, intent in records:
                intent.assert_sealed()
        except (OSError, TypeError, ValidationError, ValueError) as exc:
            raise StudioConflictError(
                "V6.7 pre-data preparation intent is invalid"
            ) from exc
        if records and workflow_mode is None:
            raise StudioConflictError(
                "V6.7 pre-data intent lacks a workflow-mode authority"
            )
        if workflow_mode is None:
            return records
        workflow_ref, workflow_contract = workflow_mode
        for _, intent in records:
            if (
                intent.workspace_spec_hash != workspace.spec.spec_hash
                or intent.workflow_mode_contract_hash != workflow_contract.contract_hash
                or intent.workflow_mode_artifact_hash != workflow_ref.sha256
                or intent.evidence_scope != workspace.spec.evidence_scope
                or intent.authority_key_id != workspace.authority_key_id
                or not workspace._verify_mac(
                    PREDATA_PREPARATION_INTENT_KIND_V67,
                    intent.unsigned_hash(),
                    intent.authority_auth_tag,
                )
            ):
                raise StudioConflictError(
                    "V6.7 pre-data intent failed authority verification"
                )
        return records

    @staticmethod
    def _verified_predata_completions_v67(
        workspace: StageWorkspaceV50,
        intents: list[tuple[ArtifactRef, PreDataPreparationIntentV67]],
    ) -> list[tuple[ArtifactRef, PreDataPreparationCompletionV67]]:
        try:
            records = workspace._artifacts_of_kind(
                PREDATA_PREPARATION_COMPLETION_KIND_V67,
                PreDataPreparationCompletionV67,
            )
            for _, completion in records:
                completion.assert_sealed()
        except (OSError, TypeError, ValidationError, ValueError) as exc:
            raise StudioConflictError(
                "V6.7 pre-data preparation completion is invalid"
            ) from exc
        intent_by_artifact_hash = {
            reference.sha256: intent for reference, intent in intents
        }
        for _, completion in records:
            intent = intent_by_artifact_hash.get(completion.intent_artifact_hash)
            if intent is None:
                raise StudioConflictError(
                    "V6.7 pre-data completion references no committed intent"
                )
            if (
                completion.workspace_spec_hash != intent.workspace_spec_hash
                or completion.s0_gate_hash != intent.s0_gate_hash
                or completion.workflow_mode_contract_hash
                != intent.workflow_mode_contract_hash
                or completion.workflow_mode_artifact_hash
                != intent.workflow_mode_artifact_hash
                or completion.evidence_scope != intent.evidence_scope
                or completion.intent_hash != intent.intent_hash
                or completion.preparation_evidence_payload_hash
                != intent.preparation_evidence_payload_hash
                or completion.source_contract_hash
                != intent.source_contract.contract_hash
                or completion.measurement_contract_hash
                != intent.measurement_contract.contract_hash
                or completion.protocol_hash != intent.predata_protocol.protocol_hash
                or completion.artifact_file_hashes != intent.artifact_file_hashes
                or completion.authority_key_id != workspace.authority_key_id
                or not workspace._verify_mac(
                    PREDATA_PREPARATION_COMPLETION_KIND_V67,
                    completion.unsigned_hash(),
                    completion.authority_auth_tag,
                )
            ):
                raise StudioConflictError(
                    "V6.7 pre-data completion failed authority verification"
                )
        return records

    @staticmethod
    def _intent_projection_files_complete_v67(
        workspace: StageWorkspaceV50,
        intent: PreDataPreparationIntentV67,
    ) -> bool:
        expected_files = predata_contract_file_bytes_v67(
            intent.source_contract,
            intent.measurement_contract,
            intent.predata_protocol,
        )
        present = 0
        for relative_path, expected_bytes in expected_files.items():
            path = workspace.root / relative_path
            if not path.exists():
                continue
            try:
                actual = path.read_bytes()
            except OSError as exc:
                raise StudioConflictError(
                    "V6.7 pre-data projection cannot be read"
                ) from exc
            if actual != expected_bytes:
                raise StudioConflictError(
                    "V6.7 pre-data projection differs from its committed intent"
                )
            present += 1
        return present == len(expected_files)

    @staticmethod
    def _matching_predata_preparation_v67(
        workspace: StageWorkspaceV50,
        intent: PreDataPreparationIntentV67,
    ) -> tuple[ArtifactRef, dict[str, Any]] | None:
        expected = predata_preparation_payload_v67(
            workspace_spec_hash=intent.workspace_spec_hash,
            s0_gate_hash=intent.s0_gate_hash,
            source_contract=intent.source_contract,
            measurement_contract=intent.measurement_contract,
            predata_protocol=intent.predata_protocol,
            artifact_file_hashes=intent.artifact_file_hashes,
        )
        matches: list[tuple[ArtifactRef, dict[str, Any]]] = []
        for reference, payload in workspace._artifacts_of_kind(
            PREDATA_PREPARATION_EVIDENCE_KIND_V67
        ):
            if not isinstance(payload, dict):
                raise StudioConflictError(
                    "V6.7 pre-data preparation evidence is invalid"
                )
            if (
                payload.get("workspace_spec_hash") != intent.workspace_spec_hash
                or payload.get("s0_gate_hash") != intent.s0_gate_hash
            ):
                continue
            if payload != expected:
                raise StudioConflictError(
                    "V6.7 pre-data preparation evidence conflicts with its "
                    "committed intent"
                )
            matches.append((reference, payload))
        if len(matches) > 1:
            raise StudioConflictError(
                "multiple V6.7 pre-data preparation records are committed"
            )
        return matches[0] if matches else None

    @classmethod
    def _predata_transaction_state_v67(
        cls,
        workspace: StageWorkspaceV50,
    ) -> _PreDataTransactionStateV67:
        transaction_policy = cls._predata_transaction_policy_v67(workspace)
        intents = cls._verified_predata_intents_v67(workspace)
        completions = cls._verified_predata_completions_v67(
            workspace,
            intents,
        )
        current_s0 = workspace.current_gate("S0")
        current = [
            (reference, intent)
            for reference, intent in intents
            if intent.s0_gate_hash == current_s0
        ]
        if len(current) > 1:
            raise StudioConflictError(
                "multiple current V6.7 pre-data intents are committed"
            )
        if not current:
            if intents:
                if len(intents) > 1:
                    return _PreDataTransactionStateV67(
                        status="STALE_PENDING"
                    )
                intent_ref, intent = intents[0]
                cls._intent_projection_files_complete_v67(
                    workspace,
                    intent,
                )
                preparation = cls._matching_predata_preparation_v67(
                    workspace,
                    intent,
                )
                stale_completions = [
                    (reference, completion)
                    for reference, completion in completions
                    if completion.intent_artifact_hash == intent_ref.sha256
                    and completion.intent_hash == intent.intent_hash
                ]
                if len(stale_completions) > 1:
                    raise StudioConflictError(
                        "multiple stale V6.7 pre-data completions are committed"
                    )
                completion_record = (
                    stale_completions[0] if stale_completions else None
                )
                return _PreDataTransactionStateV67(
                    status="STALE_PENDING",
                    bundle=(
                        intent.source_contract,
                        intent.measurement_contract,
                        intent.predata_protocol,
                    ),
                    intent_ref=intent_ref,
                    intent=intent,
                    preparation_ref=(
                        preparation[0] if preparation is not None else None
                    ),
                    preparation_payload=(
                        preparation[1] if preparation is not None else None
                    ),
                    completion_ref=(
                        completion_record[0]
                        if completion_record is not None
                        else None
                    ),
                    completion=(
                        completion_record[1]
                        if completion_record is not None
                        else None
                    ),
                )
            bundle = cls._load_predata_bundle_v67(workspace)
            if bundle is None:
                return _PreDataTransactionStateV67(status="NOT_STARTED")
            if transaction_policy is not None:
                raise StudioConflictError(
                    "governed V6.7 workspace cannot accept legacy pre-data "
                    "completion without intent and completion authority"
                )
            preparation_ref, preparation_payload = cls._predata_preparation_record_v67(
                workspace, bundle
            )
            return _PreDataTransactionStateV67(
                status="LEGACY_COMPLETED",
                bundle=bundle,
                preparation_ref=preparation_ref,
                preparation_payload=preparation_payload,
            )

        intent_ref, intent = current[0]
        bundle: PreDataBundleV67 = (
            intent.source_contract,
            intent.measurement_contract,
            intent.predata_protocol,
        )
        projections_complete = cls._intent_projection_files_complete_v67(
            workspace,
            intent,
        )
        preparation = cls._matching_predata_preparation_v67(
            workspace,
            intent,
        )
        current_completions = [
            (reference, completion)
            for reference, completion in completions
            if completion.intent_artifact_hash == intent_ref.sha256
            and completion.intent_hash == intent.intent_hash
        ]
        if len(current_completions) > 1:
            raise StudioConflictError(
                "multiple current V6.7 pre-data completions are committed"
            )
        if not current_completions:
            return _PreDataTransactionStateV67(
                status="RECOVERY_PENDING",
                bundle=bundle,
                intent_ref=intent_ref,
                intent=intent,
                preparation_ref=(preparation[0] if preparation else None),
                preparation_payload=(preparation[1] if preparation else None),
            )
        completion_ref, completion = current_completions[0]
        if preparation is None:
            raise StudioConflictError(
                "completed V6.7 pre-data transaction lacks preparation evidence"
            )
        preparation_ref, preparation_payload = preparation
        if (
            completion.preparation_evidence_artifact_hash != preparation_ref.sha256
            or completion.preparation_evidence_payload_hash
            != sha256_value(preparation_payload)
        ):
            raise StudioConflictError(
                "V6.7 pre-data completion differs from preparation evidence"
            )
        if not projections_complete:
            return _PreDataTransactionStateV67(
                status="RECOVERY_PENDING",
                bundle=bundle,
                intent_ref=intent_ref,
                intent=intent,
                preparation_ref=preparation_ref,
                preparation_payload=preparation_payload,
                completion_ref=completion_ref,
                completion=completion,
            )
        persisted = cls._load_predata_bundle_v67(workspace)
        if persisted != bundle:
            raise StudioConflictError(
                "completed V6.7 pre-data projections differ from their intent"
            )
        return _PreDataTransactionStateV67(
            status="COMPLETED",
            bundle=bundle,
            intent_ref=intent_ref,
            intent=intent,
            preparation_ref=preparation_ref,
            preparation_payload=preparation_payload,
            completion_ref=completion_ref,
            completion=completion,
        )

    @classmethod
    def _authoritative_predata_bundle_v67(
        cls,
        workspace: StageWorkspaceV50,
    ) -> PreDataBundleV67 | None:
        state = cls._predata_transaction_state_v67(workspace)
        if state.status in {"COMPLETED", "LEGACY_COMPLETED"}:
            return state.bundle
        return None

    @classmethod
    def _predata_request_summary_v67(
        cls,
        state: _PreDataTransactionStateV67,
    ) -> dict[str, Any]:
        if state.bundle is None:
            raise StudioConflictError("V6.7 pre-data summary requires a frozen bundle")
        bundle = state.bundle
        source, measurement, protocol = bundle
        return {
            "schema_version": "6.2",
            "adapter_id": protocol.adapter_binding.adapter_id,
            "contract_id": source.contract_id,
            "country_code": source.country_code,
            "indicator_id": source.indicator_id,
            "start_year": source.start_year,
            "end_year": source.end_year,
            "minimum_observations": source.minimum_observations,
            "state_unit": source.state_unit,
            "attribution": source.attribution,
            "semantic_name": measurement.construct_definition.name,
            "operational_definition": (measurement.measurement.operational_definition),
            "observation_time_basis": measurement.measurement.time_basis,
            "aggregation_level": measurement.measurement.aggregation_basis,
            "fixture_only": source.fixture_only,
            "source_contract_hash": source.contract_hash,
            "measurement_contract_hash": measurement.contract_hash,
            "protocol_hash": protocol.protocol_hash,
            "preparation_evidence_hash": (
                state.preparation_ref.sha256
                if state.preparation_ref is not None
                else None
            ),
            "intent_hash": (
                state.intent.intent_hash if state.intent is not None else None
            ),
            "completion_hash": (
                state.completion.completion_hash
                if state.completion is not None
                else None
            ),
            "capability_pack_hash": (protocol.adapter_binding.capability_pack_hash),
        }

    def _predata_projection_v67(
        self,
        workspace: StageWorkspaceV50,
        *,
        active: bool,
    ) -> dict[str, Any]:
        workflow_mode = self._effective_workflow_mode(workspace)
        state = self._predata_transaction_state_v67(workspace)
        bundle = state.bundle
        prepared = state.status in {"COMPLETED", "LEGACY_COMPLETED"}
        s0_open = workspace.current_gate("S0") is not None
        s1_open = workspace.current_gate("S1") is not None
        available = bool(
            workflow_mode == "v67"
            and s0_open
            and not s1_open
            and not active
            and state.status == "NOT_STARTED"
            and workspace.status().stage_statuses["S1"] == "frontier"
            and not any(
                (workspace.root / relative).exists()
                for relative in (
                    *_S1_PATHS,
                    RAW_RELATIVE_PATH,
                    SOURCE_RAW_PATH,
                    SOURCE_RECEIPT_PATH,
                    SOURCE_VERIFICATION_PATH,
                    SOURCE_ACQUISITION_AUTH_PATH,
                    S2_SOURCE_REVERIFICATION_PATH,
                    MEASUREMENT_SCHEMA_PATH,
                )
            )
        )
        request_summary = (
            self._predata_request_summary_v67(state) if bundle is not None else None
        )
        return {
            "schema_version": "6.7",
            "workflow_mode": workflow_mode,
            "available": available,
            "prepared": prepared,
            "transaction_status": state.status,
            "recovery_available": bool(
                state.status == "RECOVERY_PENDING"
                and s0_open
                and not s1_open
                and not active
            ),
            "intent_hash": (
                state.intent.intent_hash if state.intent is not None else None
            ),
            "completion_hash": (
                state.completion.completion_hash
                if state.completion is not None
                else None
            ),
            "required_before_v67_s1": workflow_mode == "v67",
            "request_summary": request_summary,
            "source_contract_hash": (
                bundle[0].contract_hash if bundle is not None else None
            ),
            "measurement_contract_hash": (
                bundle[1].contract_hash if bundle is not None else None
            ),
            "protocol_hash": (bundle[2].protocol_hash if bundle is not None else None),
            "observation_values_included": False,
            "private_acceptance_data_included": False,
            "scientific_qualification_granted": False,
            "real_world_action_authorized": False,
        }

    @staticmethod
    def _portfolio_lane_eligible_v69(
        workspace: StageWorkspaceV50,
    ) -> bool:
        return (
            workspace.spec.evidence_scope == "development"
            and StudioTaskService._effective_workflow_mode(workspace) == "legacy"
        )

    def _portfolio_transaction_v69(
        self,
        workspace: StageWorkspaceV50,
    ) -> DevelopmentPortfolioTransactionV69:
        return DevelopmentPortfolioTransactionV69(
            workspace.root,
            authority_key=self.authority_key,
            authority_key_id=self.authority_key_id,
        )

    def _portfolio_lane_lock_path_v69(self, task_id: str) -> Path:
        return self._task_path(task_id) / ".fma" / ".v69-s1-portfolio.lock"

    @staticmethod
    def _project_portfolio_transaction_v69(
        transaction: DevelopmentPortfolioTransactionV69,
    ) -> PortfolioTransactionStateV69:
        try:
            return transaction.project()
        except PermissionError as exc:
            raise StudioConflictError(str(exc)) from exc
        except (StageWorkspaceError, ValidationError, ValueError) as exc:
            raise StudioBridgeError(
                "V6.9 portfolio integrity verification failed closed"
            ) from exc

    @contextmanager
    def _portfolio_mutation_claim_v69(
        self,
        task_id: str,
    ) -> Iterator[None]:
        with self._lock:
            if task_id in self._active_tasks:
                raise StudioConflictError("task already has an active stage run")
            self._active_tasks.add(task_id)
        try:
            yield
        finally:
            with self._lock:
                self._active_tasks.discard(task_id)

    @contextmanager
    def _portfolio_persistent_mutation_claim_v69(
        self,
        task_id: str,
    ) -> Iterator[None]:
        with self._portfolio_mutation_claim_v69(task_id):
            with exclusive_file_lock(
                self._portfolio_lane_lock_path_v69(task_id)
            ):
                yield

    @staticmethod
    def _empty_portfolio_projection_v69(
        *,
        available: bool,
        transaction_status: str = "NOT_STARTED",
    ) -> dict[str, Any]:
        return {
            "schema_version": "6.9",
            "development_only": True,
            "available": available,
            "transaction_status": transaction_status,
            "recovery_available": False,
            "protocol_hash": None,
            "snapshot_hash": None,
            "outer_origin_plan_hash": None,
            "branch_statuses": {},
            "evaluation_hashes": {},
            "decision": None,
            "selected_branch_id": None,
            "decision_hash": None,
            "run_hash": None,
            "baseline_guard_status": "NOT_RUN",
            "persistence_relative_improvement": None,
            "engineering_status": transaction_status,
            "scientific_evidence_status": "NOT_RUN",
            "claim_ceiling": "development_protocol_only",
            "problem_signature_source": "caller_selected_v69_narrow_lane",
            "derived_from_s0_typed_problem_signature": False,
            "s1_s6_gates_touched": False,
            "scientific_qualification_granted": False,
            "real_world_action_authorized": False,
        }

    def _portfolio_projection_v69(
        self,
        workspace: StageWorkspaceV50,
        *,
        active: bool,
    ) -> dict[str, Any]:
        eligible = self._portfolio_lane_eligible_v69(workspace)
        intent_records = workspace._artifacts_of_kind(
            PORTFOLIO_TRANSACTION_INTENT_KIND_V69
        )
        downstream_gate_open = any(
            workspace.current_gate(stage) is not None
            for stage in ("S1", "S2", "S3", "S4", "S5", "S6")
        )
        if downstream_gate_open and intent_records:
            raise StudioBridgeError(
                "V6.9 portfolio and S1-S6 gate authority coexist; "
                "the task failed closed"
            )
        if active and not intent_records:
            return self._empty_portfolio_projection_v69(available=False)
        if not eligible or downstream_gate_open:
            return self._empty_portfolio_projection_v69(
                available=False,
                transaction_status="NOT_STARTED",
            )

        state = self._project_portfolio_transaction_v69(
            self._portfolio_transaction_v69(workspace)
        )
        status_map = {
            "NOT_STARTED": "NOT_STARTED",
            "FROZEN": "PREPARED",
            "DATA_STAGED": "DATA_READY",
            "RECOVERY_PENDING": "RUN_PENDING",
            "COMPLETED": "COMPLETED",
            "STALE_PENDING": "STALE_PENDING",
        }
        projected_status = status_map[state.status]
        s0_open = workspace.current_gate("S0") is not None
        available = bool(
            state.status == "NOT_STARTED"
            and s0_open
            and workspace.status().stage_statuses["S1"] == "frontier"
            and not active
        )
        run = state.run
        baseline_status = "NOT_RUN"
        if run is not None:
            if run.final_decision == "SELECT":
                baseline_status = "PASS"
            elif run.reason_code == "persistence-baseline-not-beaten":
                baseline_status = "FAIL"
        return {
            "schema_version": "6.9",
            "development_only": True,
            "available": available,
            "transaction_status": projected_status,
            "recovery_available": bool(
                state.status == "RECOVERY_PENDING" and not active
            ),
            "protocol_hash": state.protocol_hash,
            "snapshot_hash": state.snapshot_hash,
            "outer_origin_plan_hash": state.plan_hash,
            "branch_statuses": state.branch_statuses,
            "evaluation_hashes": state.evaluation_hashes,
            "decision": state.final_decision,
            "selected_branch_id": (
                run.selected_branch_id if run is not None else None
            ),
            "decision_hash": state.run_hash,
            "run_hash": state.run_hash,
            "baseline_guard_status": baseline_status,
            "persistence_relative_improvement": (
                run.persistence_relative_improvement
                if run is not None
                else None
            ),
            "engineering_status": projected_status,
            "scientific_evidence_status": "NOT_RUN",
            "claim_ceiling": "development_protocol_only",
            "problem_signature_source": "caller_selected_v69_narrow_lane",
            "derived_from_s0_typed_problem_signature": False,
            "s1_s6_gates_touched": False,
            "scientific_qualification_granted": False,
            "real_world_action_authorized": False,
        }

    @staticmethod
    def _assert_no_portfolio_lane_v69(
        workspace: StageWorkspaceV50,
    ) -> None:
        if workspace._artifacts_of_kind(
            PORTFOLIO_TRANSACTION_INTENT_KIND_V69
        ):
            raise StudioConflictError(
                "S1 is blocked after a V6.9 development portfolio freeze; "
                "complete or inspect that isolated side lane in this task"
            )

    def publish_intake_v70(
        self,
        *,
        idempotency_key: str,
        objective: str,
        attachment_paths: list[str | Path] | tuple[str | Path, ...] = (),
        workspace_id: str | None = None,
        evidence_scope: Literal["development", "public_data"] = "development",
        workflow_mode: Literal["legacy", "v67"] = "legacy",
    ) -> dict[str, Any]:
        """Publish an immutable untrusted intake without creating a task."""

        try:
            manifest = self.operator_store.publish_intake(
                idempotency_key=idempotency_key,
                objective=objective,
                attachment_paths=attachment_paths,
                requested_workspace_id=workspace_id,
                evidence_scope=evidence_scope,
                workflow_mode=workflow_mode,
            )
        except (ValueError, OperatorPlaneError) as exc:
            raise StudioValidationError(str(exc)) from exc
        return {
            "status": "success",
            "intake": manifest.model_dump(mode="json"),
            "next_action": "create_task_from_intake",
            "claim_scope": "workflow_control_only",
            "scientific_qualification_granted": False,
            "real_world_action_authorized": False,
        }

    def create_task_from_intake_v70(self, intake_id: str) -> dict[str, Any]:
        try:
            manifest = self.operator_store.get_intake(intake_id)
        except OperatorPlaneError as exc:
            raise StudioValidationError(str(exc)) from exc
        return self.create_task(
            {
                "objective": manifest.objective,
                "workspace_id": manifest.requested_workspace_id,
                "evidence_scope": manifest.evidence_scope,
                "workflow_mode": manifest.workflow_mode,
                "intake_id": manifest.intake_id,
                "intake_manifest_hash": manifest.manifest_hash,
            }
        )

    def create_task(
        self, request: CreateTaskRequest | dict[str, Any]
    ) -> dict[str, Any]:
        try:
            validated = (
                request
                if isinstance(request, CreateTaskRequest)
                else CreateTaskRequest.model_validate(request)
            )
        except ValueError as exc:
            raise StudioValidationError(str(exc)) from exc
        objective = validated.objective.strip()
        decision_intent = _decision_intent_from_request(validated.decision_use)
        intake_manifest: IntakeManifestV70 | None = None
        if validated.intake_id is not None:
            try:
                intake_manifest = self.operator_store.get_intake(
                    validated.intake_id
                )
            except OperatorPlaneError as exc:
                raise StudioValidationError(str(exc)) from exc
            if (
                intake_manifest.manifest_hash != validated.intake_manifest_hash
                or intake_manifest.objective != objective
                or intake_manifest.evidence_scope != validated.evidence_scope
                or intake_manifest.workflow_mode != validated.workflow_mode
                or (
                    intake_manifest.requested_workspace_id is not None
                    and validated.workspace_id
                    != intake_manifest.requested_workspace_id
                )
            ):
                raise StudioConflictError(
                    "task request differs from the published intake manifest"
                )
        if intake_manifest is not None:
            mission_payload = {
                "schema_version": "studio-mission-4-intake-bound",
                "objective": objective,
                "value_owner": "user",
                "workflow_mode": validated.workflow_mode,
                "evidence_scope": validated.evidence_scope,
                "decision_intent_hash": (
                    decision_intent.intent_hash
                    if decision_intent is not None
                    else None
                ),
                "intake_id": intake_manifest.intake_id,
                "intake_manifest_hash": intake_manifest.manifest_hash,
            }
        elif validated.workflow_mode == "v67":
            mission_payload = {
                "schema_version": "studio-mission-3",
                "objective": objective,
                "value_owner": "user",
                "workflow_mode": "v67",
                "decision_intent_hash": (
                    decision_intent.intent_hash if decision_intent is not None else None
                ),
            }
        else:
            mission_payload = (
                {
                    "schema_version": "studio-mission-2",
                    "objective": objective,
                    "value_owner": "user",
                    "decision_intent_hash": decision_intent.intent_hash,
                }
                if decision_intent is not None
                else {
                    "schema_version": "studio-mission-1",
                    "objective": objective,
                    "value_owner": "user",
                }
            )
        mission_hash = sha256_value(mission_payload)
        if intake_manifest is not None:
            derived = mission_hash[:12]
        elif validated.workflow_mode == "v67":
            derived = sha256_value(
                {
                    "objective": objective,
                    "workflow_mode": "v67",
                    "decision_intent_hash": (
                        decision_intent.intent_hash
                        if decision_intent is not None
                        else None
                    ),
                }
            )[:12]
        else:
            derived = (
                hashlib.sha256(objective.encode("utf-8")).hexdigest()[:12]
                if decision_intent is None
                else sha256_value(
                    {
                        "objective": objective,
                        "decision_intent_hash": decision_intent.intent_hash,
                    }
                )[:12]
            )
        task_id = validated.workspace_id or f"task-{derived}"
        if not _TASK_ID_PATTERN.fullmatch(task_id) or task_id in {".", ".."}:
            raise StudioValidationError("workspace_id is not a safe identifier")
        root = self._task_path(task_id)

        with self._lock:
            if root.exists():
                workspace = self._workspace(task_id)
                if (
                    workspace.spec.objective != objective
                    or workspace.spec.mission_hash != mission_hash
                ):
                    raise StudioConflictError(
                        "workspace_id already exists with another mission or "
                        "decision-use contract"
                    )
                if validated.workflow_mode == "v67":
                    workflow_mode = self._workflow_mode_contract_v67(workspace)
                    transaction_policy = (
                        self._predata_transaction_policy_v67(workspace)
                        if workflow_mode is not None
                        else None
                    )
                    if workflow_mode is None or transaction_policy is None:
                        if not self._predata_policy_installation_is_pristine_v67(
                            workspace
                        ):
                            raise StudioConflictError(
                                "V6.7 task-creation replay cannot install missing "
                                "workflow or pre-data policy authority after S0 or "
                                "pre-data work"
                            )
                        if workflow_mode is None:
                            if workspace.spec.spec_hash is None:
                                raise StudioConflictError(
                                    "V6.7 workflow-mode replay requires a sealed "
                                    "workspace"
                                )
                            contract = StudioWorkflowModeContractV67.seal(
                                workspace_spec_hash=workspace.spec.spec_hash,
                                evidence_scope=validated.evidence_scope,
                            )
                            workspace.commit_evidence(
                                "studio_workflow_mode_v67",
                                contract.model_dump(mode="json"),
                            )
                        self._commit_predata_transaction_policy_v67(workspace)
                if intake_manifest is not None:
                    installed_path = (
                        root / "problem" / "intake" / "manifest.json"
                    )
                    if not installed_path.is_file():
                        raise StudioConflictError(
                            "intake-bound workspace is missing its installed manifest"
                        )
                    installed = IntakeManifestV70.model_validate_json(
                        installed_path.read_text(encoding="utf-8")
                    )
                    if (
                        installed.manifest_hash
                        != intake_manifest.manifest_hash
                    ):
                        raise StudioConflictError(
                            "installed intake manifest differs on task replay"
                        )
                    self.operator_store.verify_materialized_intake(
                        intake_manifest.intake_id,
                        root,
                    )
                    prior_binding = self.operator_store.get_intake_binding(
                        intake_manifest.intake_id,
                        workspace_id=task_id,
                    )
                    if prior_binding is None:
                        binding = self._operator_authority_binding_v70(workspace)
                        if (
                            binding.current_gate_hashes
                            or binding.frontier_stages != ("S0",)
                            or binding.stage_statuses.get("S0") != "frontier"
                        ):
                            raise StudioConflictError(
                                "advanced intake workspace is missing its "
                                "original operator binding"
                            )
                        self.operator_store.bind_intake(
                            intake_manifest.intake_id,
                            workspace_id=task_id,
                            authority_binding_hash=cast(
                                str,
                                binding.binding_hash,
                            ),
                        )
                return self.snapshot(task_id)

            evidence_items = (
                [
                    {
                        "kind": "studio_intake_manifest_v70",
                        "sha256": intake_manifest.manifest_hash,
                        "trust": "user_supplied_untrusted",
                    }
                ]
                if intake_manifest is not None
                else []
            )
            evidence_snapshot_hash = sha256_value(
                {
                    "schema_version": (
                        "studio-evidence-2-intake-bound"
                        if intake_manifest is not None
                        else "studio-evidence-1"
                    ),
                    "objective_hash": hashlib.sha256(
                        objective.encode("utf-8")
                    ).hexdigest(),
                    "items": evidence_items,
                    "data_ingested": False,
                }
            )
            creation_root = (
                self.operator_store.staging_root
                / f"w-{uuid4().hex[:12]}"
                if intake_manifest is not None
                else root
            )
            scaffold_task_workspace(creation_root, task_id, objective)
            if intake_manifest is not None:
                self.operator_store.materialize_intake(
                    intake_manifest.intake_id,
                    creation_root,
                )
            spec = TaskWorkspaceSpecV50.seal(
                workspace_id=task_id,
                graph_id=f"v5-{task_id}",
                objective=objective,
                mission_hash=mission_hash,
                evidence_snapshot_hash=evidence_snapshot_hash,
                evaluator_epoch="studio-v1",
                profile=WorkflowProfileV50.seal(),
                evidence_scope=validated.evidence_scope,
            )
            workspace = StageWorkspaceV50.create(
                creation_root,
                spec,
                authority_key=self.authority_key,
                authority_key_id=self.authority_key_id,
            )
            workflow_mode_contract: StudioWorkflowModeContractV67 | None = None
            predata_transaction_policy: PreDataTransactionPolicyV67 | None = None
            if validated.workflow_mode == "v67":
                if spec.spec_hash is None:
                    raise StudioConflictError(
                        "V6.7 workflow mode requires a sealed workspace"
                    )
                workflow_mode_contract = StudioWorkflowModeContractV67.seal(
                    workspace_spec_hash=spec.spec_hash,
                    evidence_scope=validated.evidence_scope,
                )
                workspace.commit_evidence(
                    "studio_workflow_mode_v67",
                    workflow_mode_contract.model_dump(mode="json"),
                )
                _, predata_transaction_policy = (
                    self._commit_predata_transaction_policy_v67(
                        workspace,
                    )
                )
            if decision_intent is not None:
                _write_json_new(
                    creation_root / DECISION_INTENT_PATH,
                    decision_intent.model_dump(mode="json"),
                )
            if intake_manifest is not None:
                workspace.commit_evidence(
                    "studio_intake_manifest_v70",
                    intake_manifest.model_dump(mode="json"),
                )
                if not workspace.verify():
                    raise StudioConflictError(
                        "intake-bound staging workspace failed verification"
                    )
                try:
                    os.replace(creation_root, root)
                except OSError as exc:
                    raise StudioConflictError(
                        "task workspace was concurrently created"
                    ) from exc
                workspace = self._workspace(task_id)
                binding = self._operator_authority_binding_v70(workspace)
                self.operator_store.bind_intake(
                    intake_manifest.intake_id,
                    workspace_id=task_id,
                    authority_binding_hash=cast(str, binding.binding_hash),
                )
            self._append_event(
                task_id,
                event_type="task_created",
                status="succeeded",
                message="FMA task workspace and S0 frontier created",
                details={
                    "evidence_scope": validated.evidence_scope,
                    "workflow_mode": validated.workflow_mode,
                    "workflow_mode_contract_hash": (
                        workflow_mode_contract.contract_hash
                        if workflow_mode_contract is not None
                        else None
                    ),
                    "predata_transaction_policy_hash": (
                        predata_transaction_policy.policy_hash
                        if predata_transaction_policy is not None
                        else None
                    ),
                    "decision_intent_hash": (
                        decision_intent.intent_hash
                        if decision_intent is not None
                        else None
                    ),
                    "intake_id": (
                        intake_manifest.intake_id
                        if intake_manifest is not None
                        else None
                    ),
                    "intake_manifest_hash": (
                        intake_manifest.manifest_hash
                        if intake_manifest is not None
                        else None
                    ),
                    "scientific_qualification_granted": False,
                    "real_world_action_authorized": False,
                },
            )
            return self.snapshot(task_id)

    def snapshot(
        self,
        task_id: str,
        *,
        _include_operator_packet: bool = True,
    ) -> dict[str, Any]:
        workspace = self._workspace(task_id)
        status = workspace.status()
        events = self._events(task_id)
        s0_open = workspace.current_gate("S0") is not None
        s1_open = workspace.current_gate("S1") is not None
        s6_open = workspace.current_gate("S6") is not None
        backhalf = backhalf_summary_v59(workspace)
        recovery = RecoveryKernelV60(workspace).summary()
        scientific_success = scientific_success_summary_v61(workspace)
        scientific_closure = scientific_closure_summary_v62(workspace)
        with self._lock:
            active = task_id in self._active_tasks
        active = active or self.operator_store.has_live_lease(task_id)
        predata_v67 = self._predata_projection_v67(
            workspace,
            active=active,
        )
        portfolio_v69 = self._portfolio_projection_v69(
            workspace,
            active=active,
        )
        if active:
            next_valid_actions: list[str] = []
        elif s6_open:
            next_valid_actions = ["inspect_s6"]
        elif s1_open:
            next_valid_actions = ["inspect_s1"]
            next_valid_actions.append(
                "run_backhalf"
                if backhalf["data_received"]
                else (
                    "ingest_world_bank_data"
                    if predata_v67["workflow_mode"] == "v67" and predata_v67["prepared"]
                    else "ingest_ode_data"
                )
            )
        elif s0_open and (
            (
                recovery["human_required"]
                and recovery["human_reason"] == S1_FORMALIZATION_FAILURE_CODE_V67
            )
            or (
                recovery["stopped"]
                and recovery["stop_reason"] == S1_FORMALIZATION_FAILURE_CODE_V67
            )
        ):
            next_valid_actions = ["inspect_s1"]
        elif s0_open:
            next_valid_actions = ["inspect_s0"]
            portfolio_status = portfolio_v69["transaction_status"]
            if portfolio_status != "NOT_STARTED":
                portfolio_action = {
                    "PREPARED": "ingest_portfolio_v69",
                    "DATA_READY": "run_portfolio_v69",
                    "RUN_PENDING": "reconcile_portfolio_v69",
                    "COMPLETED": "inspect_portfolio_v69",
                    "STALE_PENDING": "inspect_portfolio_stale_v69",
                }.get(portfolio_status)
                if portfolio_action is not None:
                    next_valid_actions.append(portfolio_action)
            elif predata_v67["transaction_status"] == "STALE_PENDING":
                next_valid_actions.append("inspect_predata_stale_v67")
            elif (
                predata_v67["workflow_mode"] == "v67"
                and predata_v67["transaction_status"] == "RECOVERY_PENDING"
            ):
                next_valid_actions.append("reconcile_predata_v67")
            elif predata_v67["workflow_mode"] == "v67" and not predata_v67["prepared"]:
                next_valid_actions.append("prepare_predata_v67")
            else:
                next_valid_actions.append("run_s1")
                if portfolio_v69["available"]:
                    next_valid_actions.append("prepare_portfolio_v69")
        elif status.stage_statuses["S0"] in {
            "awaiting_gate_evidence",
            "blocked",
            "failed",
        }:
            next_valid_actions = ["inspect_s0"]
        else:
            next_valid_actions = ["run_s0"]
        payload = {
            "status": "success",
            "task_id": task_id,
            "objective": workspace.spec.objective,
            "evidence_scope": workspace.spec.evidence_scope,
            "workflow": status.model_dump(mode="json"),
            "activity": "running"
            if active
            else (events[-1].status if events else "idle"),
            "events": [event.model_dump(mode="json") for event in events[-30:]],
            "epistemic": EpistemicGraphStoreV58(workspace.root).summary(),
            "backhalf": backhalf,
            "recovery": recovery,
            "scientific_success": scientific_success,
            "scientific_closure": scientific_closure,
            "predata_v67": predata_v67,
            "portfolio_v69": portfolio_v69,
            "operator_v70": self.operator_store.operational_summary(task_id),
            "next_valid_actions": next_valid_actions,
            "scientific_qualification_granted": False,
            "real_world_action_authorized": False,
        }
        payload["next_packet_v70"] = (
            self.project_next_packet_v70(
                task_id,
                next_valid_actions=next_valid_actions,
            )
            if _include_operator_packet
            else None
        )
        return payload

    def list_tasks(self) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        corrupt_items: list[dict[str, str]] = []
        for child in sorted(self.task_root.iterdir()):
            if not child.is_dir() or child.name == self.operator_store.root.name:
                continue
            if not (child / ".fma" / "workspace_spec.json").is_file():
                if (child / ".fma").exists():
                    corrupt_items.append(
                        {
                            "task_id": child.name,
                            "health": "corrupt",
                            "reason_code": "workspace_spec_missing",
                        }
                    )
                continue
            try:
                snapshot = self.snapshot(child.name)
            except (OSError, ValueError, RuntimeError) as exc:
                corrupt_items.append(
                    {
                        "task_id": child.name,
                        "health": "corrupt",
                        "reason_code": type(exc).__name__,
                    }
                )
                continue
            items.append(
                {
                    "task_id": snapshot["task_id"],
                    "objective": snapshot["objective"],
                    "activity": snapshot["activity"],
                    "stage_statuses": snapshot["workflow"]["stage_statuses"],
                    "operator_status": snapshot["operator_v70"],
                }
            )
        return {
            "status": "success",
            "items": items,
            "corrupt_items": corrupt_items,
            "health": "degraded" if corrupt_items else "healthy",
        }

    def _transport(self, task_id: str) -> StageRoleTransportV51:
        output_root = self._task_path(task_id) / ".fma" / "roles"
        if self.role_transport_factory is not None:
            return self.role_transport_factory(output_root)
        return CodexStageRoleTransportV66(
            output_root,
            self.codex_config,
            process_runner=self.codex_process_runner,
            cli_locator=self.codex_cli_locator,
        )

    def _materialize_s0(
        self,
        workspace: StageWorkspaceV50,
        outcome: RoleProcessOutcomeV51,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        profile = frozen_s0_evaluation_profile_v66()
        profile.assert_sealed()
        artifacts = {
            artifact.artifact_type: artifact.content
            for artifact in outcome.draft.proposed_artifacts
        }
        if set(artifacts) != {"decision_function", "regime_diagnosis"}:
            raise StudioValidationError(
                "Codex must return exactly decision_function and regime_diagnosis"
            )
        decision_payload = _safe_json(
            artifacts["decision_function"],
            artifact_type="decision_function",
        )
        if decision_payload.get("schema_version") != "6.6-s0-decision-draft":
            raise StudioValidationError(
                "new S0 work requires the V6.6 decision draft schema"
            )
        decision = materialize_decision_function_v66(
            DecisionFunctionDraftV66.model_validate(decision_payload)
        )
        regime_payload = _safe_json(
            artifacts["regime_diagnosis"],
            artifact_type="regime_diagnosis",
        )
        if regime_payload.get("schema_version") != "6.6-s0-regime-draft":
            raise StudioValidationError(
                "new S0 work requires the V6.6 regime draft schema"
            )
        regime = materialize_regime_diagnosis_v66(
            RegimeDiagnosisDraftV66.model_validate(regime_payload)
        )
        try:
            for canary in decision.canaries:
                actual = _evaluate_arithmetic(decision.expression, canary.inputs)
                if abs(actual - canary.expected) > canary.tolerance:
                    raise StudioValidationError(
                        f"decision function canary failed: {canary.canary_id}"
                    )
        except (ArithmeticError, SyntaxError, TypeError, ValueError) as exc:
            raise StudioValidationError(
                "decision function expression is not executable by the safe "
                f"arithmetic evaluator: {exc}"
            ) from exc
        decision_intent = _load_decision_intent(workspace)
        if decision_intent is not None:
            if decision.input_names != ["prediction", "target"]:
                raise StudioValidationError(
                    "decision-value tasks require canonical inputs prediction,target"
                )
            if decision.sense != "minimize":
                raise StudioValidationError(
                    "decision-value tasks require a minimized loss function"
                )
            probes = (
                (
                    {"prediction": 9.0, "target": 10.0},
                    decision_intent.underage_unit_cost,
                ),
                (
                    {"prediction": 11.0, "target": 10.0},
                    decision_intent.overage_unit_cost,
                ),
                ({"prediction": 10.0, "target": 10.0}, 0.0),
            )
            for inputs, expected in probes:
                actual = _evaluate_arithmetic(decision.expression, inputs)
                if not math.isclose(
                    actual,
                    expected,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    raise StudioValidationError(
                        "decision function does not implement the frozen "
                        "underage/overage loss"
                    )
        if regime.decision_function_id != decision.function_id:
            raise StudioValidationError(
                "regime decision_function_id does not match decision function"
            )
        if workspace.spec.evidence_snapshot_hash not in regime.evidence_hashes:
            raise StudioValidationError(
                "regime diagnosis is not bound to the frozen evidence snapshot"
            )
        root = workspace.root
        if any((root / relative).exists() for relative in _S0_OWNED_PATHS):
            raise StudioConflictError(
                "S0 artifacts already exist; automatic re-execution is blocked"
            )
        _write_json_new(
            root / "problem" / "contract.json",
            {
                "schema_version": "5.0",
                "mission_hash": workspace.spec.mission_hash,
                "evidence_snapshot_hash": workspace.spec.evidence_snapshot_hash,
                "evaluation_profile_hash": profile.profile_hash,
                "question": workspace.spec.objective,
            },
        )
        _write_json_new(
            root / S0_EVALUATION_PROFILE_PATH_V66,
            profile.model_dump(mode="json"),
        )
        _write_json_new(
            root / "problem" / "decision_function.json",
            decision.model_dump(mode="json"),
        )
        _write_json_new(
            root / "docs" / "regime.json",
            regime.model_dump(mode="json"),
        )
        return (
            decision_payload,
            regime.model_dump(mode="json"),
            profile.model_dump(mode="json"),
        )

    def _commit_review(
        self,
        workspace: StageWorkspaceV50,
        *,
        producer: RoleProcessOutcomeV51,
        reviewer: RoleProcessOutcomeV51,
        decision_payload: dict[str, Any],
        regime_payload: dict[str, Any],
        evaluation_profile_payload: dict[str, Any],
    ) -> dict[str, Any]:
        manifest = workspace._manifest_for_stage("S0")
        checks = workspace._latest_checks("S0", str(manifest.manifest_hash))
        allowed_inputs = sorted(
            {item.sha256 for item in manifest.files}
            | {
                str(result.result_hash)
                for result in checks.values()
                if result.result_hash is not None
            }
        )
        if reviewer.draft.verdict == "APPROVE" and reviewer.draft.findings:
            raise StudioValidationError(
                "an approving S0 review cannot contain blocking findings"
            )
        finding_set: S0ReviewFindingSetV66 | None = None
        if reviewer.draft.verdict in {"REJECT", "HUMAN"}:
            provisional = seal_s0_review_findings_v66(
                task_id=workspace.spec.workspace_id,
                attempt_id=f"s0-a{manifest.attempt}",
                reviewer_receipt_hash="0" * 64,
                reviewer_codes=reviewer.draft.findings,
                regime_payload=regime_payload,
                decision_payload=decision_payload,
                evaluation_profile_payload=evaluation_profile_payload,
            )
            finding_ids = sorted(item.finding_id for item in provisional.findings)
        else:
            finding_ids = []
        trace = workspace.commit_evidence(
            "codex_review_transport_trace_v51",
            {
                "stage": "S0",
                "role": "referee",
                "producer_run_id": producer.request.run_id,
                "reviewer_run_id": reviewer.request.run_id,
                "producer_context_id": producer.request.context_id,
                "reviewer_context_id": reviewer.request.context_id,
                "context_isolation_attested": True,
                "allowed_input_hashes": allowed_inputs,
                "process_receipt": reviewer.receipt.model_dump(mode="json"),
            },
        )
        output = workspace.commit_evidence(
            "codex_review_output_v51",
            {
                "stage": "S0",
                "role": "referee",
                "verdict": reviewer.draft.verdict,
                "finding_ids": finding_ids,
                "draft": reviewer.draft.model_dump(mode="json"),
            },
        )
        receipt = workspace.issue_review(
            stage="S0",
            review_id=f"review-{reviewer.request.run_id}",
            role="referee",
            producer_run_id=producer.request.run_id,
            reviewer_run_id=reviewer.request.run_id,
            producer_context_id=producer.request.context_id,
            reviewer_context_id=reviewer.request.context_id,
            prompt_hash=reviewer.receipt.prompt_hash,
            output_schema_hash=reviewer.receipt.output_schema_hash,
            allowed_input_hashes=allowed_inputs,
            transport_trace_hash=trace.sha256,
            output_artifact_hash=output.sha256,
            verdict=reviewer.draft.verdict,
            finding_ids=finding_ids,
            issued_by="verifier",
        )
        finding_set_artifact_hash: str | None = None
        if reviewer.draft.verdict in {"REJECT", "HUMAN"}:
            assert receipt.receipt_hash is not None
            finding_set = seal_s0_review_findings_v66(
                task_id=workspace.spec.workspace_id,
                attempt_id=f"s0-a{manifest.attempt}",
                reviewer_receipt_hash=receipt.receipt_hash,
                reviewer_codes=reviewer.draft.findings,
                regime_payload=regime_payload,
                decision_payload=decision_payload,
                evaluation_profile_payload=evaluation_profile_payload,
            )
            if sorted(item.finding_id for item in finding_set.findings) != finding_ids:
                raise StudioValidationError(
                    "S0 finding IDs changed after review authentication"
                )
            finding_ref = workspace.commit_evidence(
                "s0_review_finding_set_v66",
                finding_set.model_dump(mode="json"),
            )
            finding_set_artifact_hash = finding_ref.sha256
        return {
            "receipt": receipt,
            "finding_set": finding_set,
            "finding_set_artifact_hash": finding_set_artifact_hash,
        }

    @staticmethod
    def _authenticated_s0_review_receipt(
        workspace: StageWorkspaceV50,
    ) -> IndependentReviewReceiptV50 | None:
        """Return the sole authenticated referee receipt for this S0 attempt."""

        manifest = workspace._manifest_for_stage("S0")
        receipts = [
            item
            for _, item in workspace._artifacts_of_kind(
                "independent_review_receipt_v50",
                IndependentReviewReceiptV50,
            )
            if item.stage == "S0"
            and item.role == "referee"
            and item.input_manifest_hash == manifest.manifest_hash
            and workspace.verify_review(item)
        ]
        if len(receipts) > 1:
            raise StudioConflictError(
                "current S0 attempt has multiple authenticated referee receipts"
            )
        return receipts[0] if receipts else None

    @staticmethod
    def _s0_review_replay_payloads(
        workspace: StageWorkspaceV50,
        review_receipt: IndependentReviewReceiptV50,
    ) -> tuple[RoleDraftV51, dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Recover only content already bound by authenticated receipts/files."""

        executions = [
            item
            for _, item in workspace._artifacts_of_kind(
                "role_execution_receipt_v50",
                RoleExecutionReceiptV50,
            )
            if item.stage == "S0"
            and item.role == "modeler"
            and item.run_id == review_receipt.producer_run_id
            and workspace.verify_role_execution(item)
        ]
        if len(executions) != 1:
            raise StudioConflictError(
                "authenticated S0 review lacks one verified producer execution"
            )
        producer_output = workspace._artifact_payload_by_hash(
            executions[0].output_artifact_hash
        )
        review_output = workspace._artifact_payload_by_hash(
            review_receipt.output_artifact_hash
        )
        if (
            not isinstance(producer_output, dict)
            or producer_output.get("stage") != "S0"
            or producer_output.get("role") != "modeler"
            or not isinstance(review_output, dict)
            or review_output.get("stage") != "S0"
            or review_output.get("role") != "referee"
        ):
            raise StudioConflictError(
                "authenticated S0 role outputs have invalid stage bindings"
            )
        try:
            producer_draft = RoleDraftV51.model_validate(producer_output["draft"])
            review_draft = RoleDraftV51.model_validate(review_output["draft"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StudioConflictError(
                "authenticated S0 role output cannot be replayed"
            ) from exc
        if (
            producer_output.get("request_hash") != producer_draft.request_hash
            or review_draft.verdict != review_receipt.verdict
            or sorted(review_output.get("finding_ids", []))
            != review_receipt.finding_ids
        ):
            raise StudioConflictError(
                "authenticated S0 role output differs from its receipt"
            )
        proposed = {
            item.artifact_type: item.content
            for item in producer_draft.proposed_artifacts
        }
        if set(proposed) != {"decision_function", "regime_diagnosis"}:
            raise StudioConflictError(
                "authenticated S0 producer output lacks required artifacts"
            )
        try:
            decision_payload = _safe_json(
                proposed["decision_function"],
                artifact_type="decision_function",
            )
            regime_payload = json.loads(
                (workspace.root / "docs" / "regime.json").read_text(encoding="utf-8")
            )
            profile_payload = json.loads(
                (workspace.root / S0_EVALUATION_PROFILE_PATH_V66).read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, TypeError, ValueError) as exc:
            raise StudioConflictError(
                "current S0 materialized evidence cannot be replayed"
            ) from exc
        if not all(
            isinstance(payload, dict)
            for payload in (
                decision_payload,
                regime_payload,
                profile_payload,
            )
        ):
            raise StudioConflictError(
                "current S0 materialized evidence must be JSON objects"
            )
        return (
            review_draft,
            decision_payload,
            regime_payload,
            profile_payload,
        )

    def _apply_s0_review_gate_transition(
        self,
        workspace: StageWorkspaceV50,
        *,
        task_id: str,
        graph_attempt: int,
        review_verdict: str,
        review_receipt: IndependentReviewReceiptV50,
        finding_set: S0ReviewFindingSetV66 | None,
        finding_set_artifact_hash: str | None,
        resumed_authenticated_review: bool,
    ) -> S0RepairContextV66 | None:
        """Apply one authenticated review without invoking another model."""

        manifest = workspace._manifest_for_stage("S0")
        gate = workspace.evaluate_gate("S0")
        blocked_outcome_hash: str | None = None
        repair_authorization_hash: str | None = None
        recovery_receipt_hash: str | None = None
        recovery_status: str | None = None
        next_repair_context: S0RepairContextV66 | None = None

        if review_verdict == "REJECT":
            if (
                gate.decision != "BLOCKED"
                or finding_set is None
                or finding_set_artifact_hash is None
                or review_receipt.receipt_hash is None
            ):
                raise StudioValidationError("S0 rejection lacks typed gate evidence")
            checks = workspace._latest_checks(
                "S0",
                str(manifest.manifest_hash),
            )
            blocked_outcome = record_blocked_stage_gate_v66(
                workspace,
                stage="S0",
                manifest_hash=str(manifest.manifest_hash),
                policy_hash=POLICIES["S0"].policy_hash,
                check_result_hashes=[
                    str(item.result_hash)
                    for item in checks.values()
                    if item.result_hash is not None
                ],
                review_receipt_hashes=[str(review_receipt.receipt_hash)],
                finding_set_hash=finding_set_artifact_hash,
                reason_codes=["independent_review_rejected"],
            )
            blocked_outcome_hash = blocked_outcome.outcome_hash
            recovery_result = self._execute_s0_rejection_recovery(
                workspace,
                finding_set=finding_set,
                finding_set_artifact_hash=finding_set_artifact_hash,
                review_receipt=review_receipt,
                graph_attempt=graph_attempt,
            )
            repair_authorization_hash = recovery_result["authorization_artifact_hash"]
            recovery_receipt = recovery_result["recovery_receipt"]
            recovery_receipt_hash = recovery_receipt.receipt_hash
            recovery_status = recovery_receipt.status
            next_repair_context = cast(
                S0RepairContextV66 | None,
                recovery_result["repair_context"],
            )
        elif review_verdict == "HUMAN":
            if finding_set is None or finding_set_artifact_hash is None:
                raise StudioValidationError("S0 HUMAN verdict lacks typed findings")
            kernel = RecoveryKernelV60(workspace)
            _, _, recovery_receipt = kernel.recover(
                failed_stage="S0",
                category="capability_gap",
                failure_code=(f"s0human_{finding_set.failure_signature[:20]}"),
                evidence_refs=sorted(
                    set(kernel.evidence_refs_for_stage("S0"))
                    | {
                        finding_set_artifact_hash,
                        review_receipt.transport_trace_hash,
                        review_receipt.output_artifact_hash,
                    }
                ),
                expected_information_gain=0.0,
            )
            recovery_receipt_hash = recovery_receipt.receipt_hash
            recovery_status = recovery_receipt.status

        final_status = "succeeded" if gate.decision == "OPEN" else "blocked"
        self._append_event(
            task_id,
            event_type="s0_gate_evaluated",
            status=final_status,
            message=(
                "S0 gate opened; S1 candidate formalization is available"
                if gate.decision == "OPEN"
                else f"S0 gate did not open: {gate.decision}"
            ),
            details={
                "graph_attempt": graph_attempt,
                "decision": gate.decision,
                "reasons": gate.reasons,
                "review_verdict": review_verdict,
                "finding_signature": (
                    finding_set.failure_signature if finding_set is not None else None
                ),
                "finding_set_artifact_hash": finding_set_artifact_hash,
                "gate_outcome_hash": blocked_outcome_hash,
                "repair_authorization_hash": repair_authorization_hash,
                "recovery_receipt_hash": recovery_receipt_hash,
                "recovery_status": recovery_status,
                "resumed_authenticated_review": (resumed_authenticated_review),
                "new_reviewer_invoked": False,
                "scientific_qualification_granted": False,
                "real_world_action_authorized": False,
            },
        )
        if next_repair_context is not None:
            self._append_event(
                task_id,
                event_type="s0_semantic_recovery_created",
                status="succeeded",
                message=(
                    "One typed pre-data S0 repair attempt was created through the graph"
                ),
                details={
                    "prior_graph_attempt": graph_attempt,
                    "new_graph_attempt": workspace._latest_attempt("S0"),
                    "failure_signature": (next_repair_context.failure_signature),
                    "repair_context_hash": (next_repair_context.context_hash),
                    "reviewer_rationale_disclosed": False,
                    "private_evidence_disclosed": False,
                },
            )
        return next_repair_context

    def _resume_authenticated_s0_review(
        self,
        workspace: StageWorkspaceV50,
        *,
        task_id: str,
    ) -> tuple[bool, S0RepairContextV66 | None]:
        """Finish a committed S0 review; never sample another reviewer."""

        recovery_state = RecoveryKernelV60(workspace).load_state()
        if recovery_state.stopped or recovery_state.human_required:
            return True, None
        receipt = self._authenticated_s0_review_receipt(workspace)
        if receipt is None:
            return False, None
        (
            review_draft,
            decision_payload,
            regime_payload,
            profile_payload,
        ) = self._s0_review_replay_payloads(workspace, receipt)
        attempt_id = f"s0-a{workspace._latest_attempt('S0')}"
        matches = [
            (reference, item)
            for reference, item in workspace._artifacts_of_kind(
                "s0_review_finding_set_v66",
                S0ReviewFindingSetV66,
            )
            if item.task_id == workspace.spec.workspace_id
            and item.attempt_id == attempt_id
            and item.reviewer_receipt_hash == receipt.receipt_hash
        ]
        if len(matches) > 1:
            raise StudioConflictError(
                "current S0 review has multiple typed finding sets"
            )
        finding_set: S0ReviewFindingSetV66 | None = None
        finding_set_artifact_hash: str | None = None
        if review_draft.verdict in {"REJECT", "HUMAN"}:
            if matches:
                finding_ref, finding_set = matches[0]
                finding_set.assert_sealed()
                finding_set_artifact_hash = finding_ref.sha256
            else:
                if receipt.receipt_hash is None:
                    raise StudioConflictError(
                        "authenticated S0 review receipt is unsealed"
                    )
                finding_set = seal_s0_review_findings_v66(
                    task_id=workspace.spec.workspace_id,
                    attempt_id=attempt_id,
                    reviewer_receipt_hash=receipt.receipt_hash,
                    reviewer_codes=review_draft.findings,
                    regime_payload=regime_payload,
                    decision_payload=decision_payload,
                    evaluation_profile_payload=profile_payload,
                )
                if (
                    sorted(item.finding_id for item in finding_set.findings)
                    != receipt.finding_ids
                ):
                    raise StudioConflictError(
                        "replayed S0 findings differ from authenticated receipt"
                    )
                finding_ref = workspace.commit_evidence(
                    "s0_review_finding_set_v66",
                    finding_set.model_dump(mode="json"),
                )
                finding_set_artifact_hash = finding_ref.sha256
        elif matches or receipt.finding_ids:
            raise StudioConflictError(
                "approving S0 review cannot bind blocking findings"
            )
        next_context = self._apply_s0_review_gate_transition(
            workspace,
            task_id=task_id,
            graph_attempt=workspace._latest_attempt("S0"),
            review_verdict=review_draft.verdict,
            review_receipt=receipt,
            finding_set=finding_set,
            finding_set_artifact_hash=finding_set_artifact_hash,
            resumed_authenticated_review=True,
        )
        return True, next_context

    @staticmethod
    def _current_s0_repair_context(
        workspace: StageWorkspaceV50,
    ) -> S0RepairContextV66 | None:
        attempt_id = f"s0-a{workspace._latest_attempt('S0')}"
        contexts = [
            item
            for _, item in workspace._artifacts_of_kind(
                "s0_repair_context_v66",
                S0RepairContextV66,
            )
            if item.task_id == workspace.spec.workspace_id
            and item.new_attempt_id == attempt_id
        ]
        for context in contexts:
            context.assert_sealed()
        if len(contexts) > 1:
            raise StudioConflictError("current S0 attempt has multiple repair contexts")
        return contexts[0] if contexts else None

    def _execute_s0_rejection_recovery(
        self,
        workspace: StageWorkspaceV50,
        *,
        finding_set: S0ReviewFindingSetV66,
        finding_set_artifact_hash: str,
        review_receipt: Any,
        graph_attempt: int,
    ) -> dict[str, Any]:
        prior_contexts = [
            item
            for _, item in workspace._artifacts_of_kind(
                "s0_repair_context_v66",
                S0RepairContextV66,
            )
            if item.task_id == workspace.spec.workspace_id
        ]
        for context in prior_contexts:
            context.assert_sealed()
        authorization = authorize_s0_semantic_repair_v66(
            finding_set,
            repair_attempts_used=graph_attempt - 1,
            previous_failure_signatures=sorted(
                {item.failure_signature for item in prior_contexts}
            ),
            holdout_exposed=False,
            private_evidence_used=False,
        )
        authorization_ref = workspace.commit_evidence(
            "s0_semantic_repair_decision_v66",
            authorization.model_dump(mode="json"),
        )
        kernel = RecoveryKernelV60(workspace)
        evidence_refs = sorted(
            set(kernel.evidence_refs_for_stage("S0"))
            | {
                finding_set_artifact_hash,
                review_receipt.transport_trace_hash,
                review_receipt.output_artifact_hash,
            }
        )
        _, _, recovery_receipt = kernel.recover(
            failed_stage="S0",
            category="contract_semantics",
            failure_code=f"s0review_{finding_set.failure_signature[:20]}",
            evidence_refs=evidence_refs,
            expected_information_gain=(1.0 if authorization.authorized else 0.0),
        )
        next_context: S0RepairContextV66 | None = None
        if authorization.authorized:
            if recovery_receipt.status != "ATTEMPT_CREATED":
                raise StudioConflictError(
                    "authorized S0 repair did not create a graph attempt"
                )
            next_context = build_s0_repair_context_v66(
                finding_set=finding_set,
                authorization=authorization,
                new_attempt_id=f"s0-a{workspace._latest_attempt('S0')}",
            )
            workspace.commit_evidence(
                "s0_repair_context_v66",
                next_context.model_dump(mode="json"),
            )
        elif recovery_receipt.status not in {
            "ABSTAINED",
            "HUMAN_REQUIRED",
        }:
            raise StudioConflictError("non-authorized S0 repair mutated graph state")
        return {
            "authorization_artifact_hash": authorization_ref.sha256,
            "recovery_receipt": recovery_receipt,
            "repair_context": next_context,
        }

    def _resume_blocked_s0_recovery(
        self,
        workspace: StageWorkspaceV50,
        *,
        task_id: str,
    ) -> S0RepairContextV66 | None:
        """Resume only a fully authenticated interrupted reject transition."""

        kernel = RecoveryKernelV60(workspace)
        recovery_state = kernel.load_state()
        if recovery_state.stopped or recovery_state.human_required:
            return None
        outcome = latest_stage_gate_outcome_v66(workspace, "S0")
        if outcome is None:
            return None
        finding_set = S0ReviewFindingSetV66.model_validate(
            workspace._artifact_payload_by_hash(outcome.finding_set_hash)
        )
        finding_set.assert_sealed()
        receipts = [
            item
            for _, item in workspace._artifacts_of_kind(
                "independent_review_receipt_v50"
            )
            if isinstance(item, dict)
            and item.get("receipt_hash") == finding_set.reviewer_receipt_hash
        ]
        if len(receipts) != 1:
            raise StudioConflictError("blocked S0 outcome lacks one review receipt")
        # Re-parse through the workspace's typed index so its HMAC and bound
        # transport/output evidence are verified before any graph mutation.
        typed_receipts = workspace._latest_reviews(
            "S0",
            outcome.manifest_hash,
        )
        review_receipt = next(
            (
                item
                for item in typed_receipts.values()
                if item.receipt_hash == finding_set.reviewer_receipt_hash
            ),
            None,
        )
        if review_receipt is None:
            raise StudioConflictError("blocked S0 review receipt failed authentication")
        result = self._execute_s0_rejection_recovery(
            workspace,
            finding_set=finding_set,
            finding_set_artifact_hash=outcome.finding_set_hash,
            review_receipt=review_receipt,
            graph_attempt=outcome.attempt,
        )
        recovery_receipt = result["recovery_receipt"]
        repair_context = cast(
            S0RepairContextV66 | None,
            result["repair_context"],
        )
        self._append_event(
            task_id,
            event_type="s0_interrupted_recovery_resumed",
            status=("succeeded" if repair_context is not None else "blocked"),
            message=("Authenticated interrupted S0 recovery was replayed"),
            details={
                "blocked_attempt": outcome.attempt,
                "failure_signature": finding_set.failure_signature,
                "recovery_status": recovery_receipt.status,
                "repair_context_hash": (
                    repair_context.context_hash if repair_context is not None else None
                ),
                "reviewer_rationale_disclosed": False,
            },
        )
        return repair_context

    def _rebuild_s0_context_after_committed_recovery(
        self,
        workspace: StageWorkspaceV50,
    ) -> S0RepairContextV66 | None:
        """Close the crash window after graph recovery but before context write."""

        current_attempt = workspace._latest_attempt("S0")
        transitions: list[RecoveryTransitionReceiptV60] = []
        receipt_fields = set(RecoveryTransitionReceiptV60.model_fields)
        for _, payload in workspace._artifacts_of_kind(
            "recovery_transition_receipt_v60"
        ):
            if not isinstance(payload, dict):
                continue
            try:
                item = RecoveryTransitionReceiptV60.model_validate(
                    {
                        key: value
                        for key, value in payload.items()
                        if key in receipt_fields
                    }
                )
            except ValueError:
                continue
            if (
                item.status == "ATTEMPT_CREATED"
                and item.revoke_from == "S0"
                and item.successor_attempt == current_attempt
            ):
                transitions.append(item)
        if len(transitions) != 1:
            return None
        transition = transitions[0]
        diagnoses = [
            item
            for _, item in workspace._artifacts_of_kind(
                "failure_diagnosis_v60",
                FailureDiagnosisV60,
            )
            if item.diagnosis_hash == transition.diagnosis_hash
            and item.category == "contract_semantics"
            and item.failed_stage == "S0"
        ]
        if len(diagnoses) != 1:
            return None
        diagnosis = diagnoses[0]
        candidates = [
            (ref, item)
            for ref, item in workspace._artifacts_of_kind(
                "s0_review_finding_set_v66",
                S0ReviewFindingSetV66,
            )
            if ref.sha256 in diagnosis.evidence_refs
            and item.attempt_id == f"s0-a{transition.predecessor_attempt}"
        ]
        if len(candidates) != 1:
            return None
        _, finding_set = candidates[0]
        finding_set.assert_sealed()
        authorization = authorize_s0_semantic_repair_v66(
            finding_set,
            repair_attempts_used=((transition.predecessor_attempt or 1) - 1),
        )
        if not authorization.authorized:
            return None
        workspace.commit_evidence(
            "s0_semantic_repair_decision_v66",
            authorization.model_dump(mode="json"),
        )
        context = build_s0_repair_context_v66(
            finding_set=finding_set,
            authorization=authorization,
            new_attempt_id=f"s0-a{current_attempt}",
        )
        workspace.commit_evidence(
            "s0_repair_context_v66",
            context.model_dump(mode="json"),
        )
        return context

    def run_s0(self, task_id: str) -> dict[str, Any]:
        workspace = self._workspace(task_id)
        if workspace.current_gate("S0"):
            return self.snapshot(task_id)
        initial_stage_status = workspace.status().stage_statuses["S0"]
        repair_context: S0RepairContextV66 | None = None
        if initial_stage_status == "blocked":
            repair_context = self._resume_blocked_s0_recovery(
                workspace,
                task_id=task_id,
            )
            if repair_context is None:
                return self.snapshot(task_id)
            initial_stage_status = workspace.status().stage_statuses["S0"]
        elif initial_stage_status in {"awaiting_gate_evidence", "failed"}:
            # A submitted attempt is never silently overwritten or sent to a
            # second reviewer lottery.  Only an already authenticated review
            # may complete its interrupted typed gate/recovery transition.
            resumed, repair_context = self._resume_authenticated_s0_review(
                workspace,
                task_id=task_id,
            )
            if not resumed or repair_context is None:
                return self.snapshot(task_id)
            initial_stage_status = workspace.status().stage_statuses["S0"]
        if any((workspace.root / relative).exists() for relative in _S0_OWNED_PATHS):
            raise StudioConflictError(
                "S0 contains partial artifacts; refusing a second model call"
            )
        decision_intent = _load_decision_intent(workspace)
        profile = frozen_s0_evaluation_profile_v66()
        profile.assert_sealed()
        graph_attempt = workspace._latest_attempt("S0")
        if repair_context is None:
            repair_context = self._current_s0_repair_context(workspace)
        if repair_context is None and graph_attempt > 1:
            repair_context = self._rebuild_s0_context_after_committed_recovery(
                workspace
            )
            if repair_context is not None:
                self._append_event(
                    task_id,
                    event_type="s0_repair_context_rebuilt",
                    status="succeeded",
                    message=(
                        "S0 repair context was rebuilt from committed graph "
                        "recovery evidence"
                    ),
                    details={
                        "graph_attempt": graph_attempt,
                        "repair_context_hash": repair_context.context_hash,
                        "reviewer_rationale_disclosed": False,
                    },
                )
        if graph_attempt > 1 and repair_context is None:
            raise StudioConflictError(
                "recovered S0 attempt lacks its authenticated repair context"
            )

        while True:
            graph_attempt = workspace._latest_attempt("S0")
            driver = StageRoleDriverV51(self._transport(task_id))
            self._append_event(
                task_id,
                event_type="s0_generator_started",
                status="running",
                message="Fresh Codex generator process started for S0",
                details={
                    "graph_attempt": graph_attempt,
                    "semantic_repair": repair_context is not None,
                },
            )
            generator_inputs: dict[str, Any] = {
                "user_objective": workspace.spec.objective,
                "mission_hash": workspace.spec.mission_hash,
                "evidence_snapshot_hash": (workspace.spec.evidence_snapshot_hash),
                "evidence_scope": workspace.spec.evidence_scope,
                "frozen_evaluation_profile": profile.model_dump(mode="json"),
                "required_artifacts": {
                    "decision_function": (DecisionFunctionDraftV66.model_json_schema()),
                    "regime_diagnosis": (RegimeDiagnosisDraftV66.model_json_schema()),
                },
                "requirements": [
                    "Return exactly two proposed_artifacts.",
                    "Use artifact_type decision_function and regime_diagnosis.",
                    "Each content field must contain only one JSON object.",
                    "Use the exact V6.6 schema versions supplied by the harness.",
                    "Bind regime evidence_hashes to both evidence_snapshot_hash "
                    "and the exact frozen evaluation profile hash.",
                    "Keep every identifier and hash list sorted and unique.",
                    "Every regime narrative and limitation must be a complete "
                    "sentence of at most 200 characters.",
                    "Omit canary tolerance; the harness owns and injects it.",
                    "Use report_only when the user's decision loss is not specified.",
                    "State limitations and a concrete abandon condition.",
                    "decision_function.expression must be only a bare arithmetic "
                    "expression over input_names; put prose in regime_diagnosis.",
                    "Canary input_values must align positionally with input_names.",
                    "Do not invent or loosen split, baseline, uncertainty, "
                    "threshold, qualification, or stopping policies; they are "
                    "owned by the frozen evaluation profile and downstream "
                    "typed adapters.",
                    "Candidate generation and registry population belong to S1.",
                ],
            }
            if repair_context is not None:
                repair_context.assert_sealed()
                generator_inputs["semantic_repair_context"] = repair_context.model_dump(
                    mode="json"
                )
            if decision_intent is not None:
                generator_inputs["decision_value_intent"] = decision_intent.model_dump(
                    mode="json"
                )
                generator_inputs["requirements"].extend(
                    [
                        "Use input_names exactly [prediction, target].",
                        "Use sense minimize.",
                        "Implement the frozen asymmetric loss as "
                        "underage_unit_cost*max(target-prediction,0) + "
                        "overage_unit_cost*max(prediction-target,0).",
                        "Include executable canaries for underage, overage, "
                        "and zero-error cases.",
                    ]
                )

            producer: RoleProcessOutcomeV51 | None = None
            validation_error: str | None = None
            materialized_payloads: (
                tuple[
                    dict[str, Any],
                    dict[str, Any],
                    dict[str, Any],
                ]
                | None
            ) = None
            for generator_attempt in (1, 2):
                attempt_inputs = dict(generator_inputs)
                if validation_error is not None and producer is not None:
                    attempt_inputs["repair"] = {
                        "previous_output_hash": producer.receipt.output_hash,
                        "validation_error": validation_error[:500],
                        "instruction": (
                            "Return a complete corrected replacement. Do not "
                            "weaken the frozen contract or evidence bindings."
                        ),
                    }
                producer = driver.run(
                    task_id=task_id,
                    stage="S0",
                    role_name=(
                        "problem_formulator"
                        if generator_attempt == 1 and repair_context is None
                        else "problem_formulator_repair"
                    ),
                    role_kind="generator",
                    subject_id="s0_problem_contract",
                    objective=(
                        "Formalize the frozen real modeling objective into a "
                        "falsifiable S0 diagnosis and computable decision "
                        "function without claiming scientific authority."
                    ),
                    public_inputs=attempt_inputs,
                    allowed_candidate_ids=[],
                )
                if producer.draft.authority_claimed:
                    raise StudioValidationError("generator claimed reserved authority")
                commit_generator_outcome_v51(
                    workspace,
                    producer,
                    execution_role="modeler",
                    input_authority_hash=str(workspace.spec.spec_hash),
                )
                try:
                    materialized_payloads = self._materialize_s0(
                        workspace,
                        producer,
                    )
                    break
                except (StudioValidationError, ValueError) as exc:
                    validation_error = str(exc)
                    self._append_event(
                        task_id,
                        event_type="s0_generator_rejected",
                        status="blocked",
                        message=(
                            "Generator output failed typed validation"
                            if generator_attempt == 1
                            else "Repair output failed typed validation"
                        ),
                        details={
                            "graph_attempt": graph_attempt,
                            "generator_attempt": generator_attempt,
                            "failure_signature": validation_error[:500],
                            "output_hash": producer.receipt.output_hash,
                        },
                    )
                    if generator_attempt == 2:
                        raise StudioValidationError(
                            "S0 generator exhausted its two-attempt typed "
                            "validation budget"
                        ) from exc
            assert producer is not None
            assert materialized_payloads is not None
            decision_payload, regime_payload, profile_payload = materialized_payloads
            s0_extra_paths = [S0_EVALUATION_PROFILE_PATH_V66]
            if decision_intent is not None:
                s0_extra_paths.append(DECISION_INTENT_PATH)
            workspace.submit_stage(
                "S0",
                actor="model",
                extra_paths=s0_extra_paths,
            )
            check = workspace.run_mechanical_check("S0")
            self._append_event(
                task_id,
                event_type="s0_generator_completed",
                status="succeeded",
                message=(
                    "S0 artifacts validated and committed; independent review required"
                ),
                details={
                    "run_id": producer.request.run_id,
                    "graph_attempt": graph_attempt,
                    "check_status": check.status,
                    "generator_attempts": 2 if validation_error else 1,
                    "evaluation_profile_hash": profile.profile_hash,
                },
            )

            manifest = workspace._manifest_for_stage("S0")
            self._append_event(
                task_id,
                event_type="s0_reviewer_started",
                status="running",
                message="Fresh independent Codex referee process started",
                details={"graph_attempt": graph_attempt},
            )
            reviewer = driver.run(
                task_id=task_id,
                stage="S0",
                role_name="s0_referee",
                role_kind="reviewer",
                subject_id="s0_problem_contract",
                objective=(
                    "Independently review whether this S0 contract preserves "
                    "the frozen objective, is falsifiable and computable, binds "
                    "public evidence, and stays within its authority."
                ),
                public_inputs={
                    "producer_output_hash": producer.receipt.output_hash,
                    "manifest": manifest.model_dump(mode="json"),
                    "artifacts": {
                        relative: json.loads(
                            (workspace.root / relative).read_text(encoding="utf-8")
                        )
                        for relative in (
                            *_S0_OWNED_PATHS,
                            *(
                                [DECISION_INTENT_PATH]
                                if decision_intent is not None
                                else []
                            ),
                        )
                    },
                    "mechanical_check": check.model_dump(mode="json"),
                    "gate_policy_hash": POLICIES["S0"].policy_hash,
                    "finding_code_contract": (
                        S0ReviewerFindingCodesV66.model_json_schema()
                    ),
                    "stage_boundary": {
                        "candidate_registry_stage": "S1",
                        "empty_s0_candidate_registry_is_expected": True,
                        "numeric_evaluation_policies_are_profile_owned": True,
                        "reviewer_must_not_invent_or_loosen_policy": True,
                    },
                    "review_rule": (
                        "APPROVE only with an empty findings list when the "
                        "question remains the frozen objective, the decision "
                        "function is computable, uncertainty and limitations "
                        "are explicit, the frozen profile is bound, and no "
                        "scientific or action authority is claimed. Use only "
                        "the supplied finding codes for REJECT or HUMAN. The "
                        "V5 projection's canary tolerance is harness-injected. "
                        "Do not reject S0 for having no S1 candidates."
                    ),
                },
                allowed_candidate_ids=[],
            )
            review_commit = self._commit_review(
                workspace,
                producer=producer,
                reviewer=reviewer,
                decision_payload=decision_payload,
                regime_payload=regime_payload,
                evaluation_profile_payload=profile_payload,
            )
            finding_set = cast(
                S0ReviewFindingSetV66 | None,
                review_commit["finding_set"],
            )
            finding_set_artifact_hash = cast(
                str | None,
                review_commit["finding_set_artifact_hash"],
            )
            review_receipt = review_commit["receipt"]
            next_repair_context = self._apply_s0_review_gate_transition(
                workspace,
                task_id=task_id,
                graph_attempt=graph_attempt,
                review_verdict=reviewer.draft.verdict,
                review_receipt=review_receipt,
                finding_set=finding_set,
                finding_set_artifact_hash=finding_set_artifact_hash,
                resumed_authenticated_review=False,
            )
            if next_repair_context is not None:
                repair_context = next_repair_context
                continue
            return self.snapshot(task_id)

    def start_s0(self, task_id: str) -> dict[str, Any]:
        self._workspace(task_id)
        with self._lock:
            if task_id in self._active_tasks:
                raise StudioConflictError("task already has an active S0 run")
        claim = self._prepare_operator_run_v70(task_id, "run_s0")
        with self._lock:
            if task_id in self._active_tasks:
                conflict = StudioConflictError("task already has an active S0 run")
                self._fail_operator_claim_v70(claim, conflict)
                raise conflict
            self._active_tasks.add(task_id)
        try:
            self._append_event(
                task_id,
                event_type="s0_run_accepted",
                status="accepted",
                message="Bounded S0 run accepted by the local bridge",
                details={
                    "operator_work_id": claim.lease.work_id,
                    "operator_packet_hash": claim.packet.packet_hash,
                },
            )
        except Exception as exc:
            with self._lock:
                self._active_tasks.discard(task_id)
            self._fail_operator_claim_v70(claim, exc)
            raise

        def worker() -> None:
            try:
                self._execute_operator_run_v70(
                    claim,
                    lambda: self.run_s0(task_id),
                )
            except OperatorPlaneError as exc:
                self._append_event(
                    task_id,
                    event_type="operator_reconcile_required",
                    status="blocked",
                    message="S0 authority may have advanced; operator reconciliation is required",
                    details={
                        "operator_work_id": claim.lease.work_id,
                        "error_type": type(exc).__name__,
                        "scientific_qualification_granted": False,
                        "real_world_action_authorized": False,
                    },
                )
            except Exception as exc:
                self._append_event(
                    task_id,
                    event_type="s0_run_failed",
                    status="failed",
                    message="S0 run failed closed",
                    details={
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                        "scientific_qualification_granted": False,
                        "real_world_action_authorized": False,
                    },
                )
            finally:
                with self._lock:
                    self._active_tasks.discard(task_id)

        thread = threading.Thread(
            target=worker,
            name=f"fma-studio-{task_id}-s0",
            daemon=True,
        )
        try:
            thread.start()
        except Exception as exc:
            with self._lock:
                self._active_tasks.discard(task_id)
            self._fail_operator_claim_v70(claim, exc)
            raise
        return self.snapshot(task_id)

    @staticmethod
    def _assert_predata_frontier_v67(workspace: StageWorkspaceV50) -> None:
        if (
            workspace.current_gate("S1") is not None
            or workspace.status().stage_statuses["S1"] != "frontier"
            or any((workspace.root / relative).exists() for relative in _S1_PATHS)
        ):
            raise StudioConflictError(
                "V6.7 pre-data preparation must precede all S1 work"
            )
        observed_data_paths = (
            RAW_RELATIVE_PATH,
            SOURCE_RAW_PATH,
            SOURCE_RECEIPT_PATH,
            SOURCE_VERIFICATION_PATH,
            SOURCE_ACQUISITION_AUTH_PATH,
            S2_SOURCE_REVERIFICATION_PATH,
            MEASUREMENT_SCHEMA_PATH,
        )
        if any(
            (workspace.root / relative).exists() for relative in observed_data_paths
        ):
            raise StudioConflictError(
                "V6.7 pre-data preparation cannot follow data access"
            )

    @classmethod
    def _build_predata_intent_v67(
        cls,
        workspace: StageWorkspaceV50,
        bundle: PreDataBundleV67,
    ) -> PreDataPreparationIntentV67:
        workspace_spec_hash = workspace.spec.spec_hash
        s0_gate_hash = workspace.current_gate("S0")
        workflow_mode = cls._workflow_mode_contract_v67(workspace)
        if workspace_spec_hash is None or s0_gate_hash is None or workflow_mode is None:
            raise StudioConflictError(
                "V6.7 pre-data intent requires current S0 and workflow-mode authority"
            )
        workflow_ref, workflow_contract = workflow_mode
        source, measurement, protocol = bundle
        file_bytes = predata_contract_file_bytes_v67(
            source,
            measurement,
            protocol,
        )
        artifact_file_hashes = {
            relative_path: hashlib.sha256(payload).hexdigest()
            for relative_path, payload in file_bytes.items()
        }
        preparation_payload = predata_preparation_payload_v67(
            workspace_spec_hash=workspace_spec_hash,
            s0_gate_hash=s0_gate_hash,
            source_contract=source,
            measurement_contract=measurement,
            predata_protocol=protocol,
            artifact_file_hashes=artifact_file_hashes,
        )
        draft = PreDataPreparationIntentV67(
            workspace_spec_hash=workspace_spec_hash,
            s0_gate_hash=s0_gate_hash,
            workflow_mode_contract_hash=str(workflow_contract.contract_hash),
            workflow_mode_artifact_hash=workflow_ref.sha256,
            evidence_scope=workspace.spec.evidence_scope,
            source_contract=source,
            measurement_contract=measurement,
            predata_protocol=protocol,
            artifact_file_hashes=artifact_file_hashes,
            preparation_evidence_payload_hash=sha256_value(preparation_payload),
            authority_key_id=workspace.authority_key_id,
        )
        intent = draft.authenticate(
            workspace._mac(
                PREDATA_PREPARATION_INTENT_KIND_V67,
                draft.unsigned_hash(),
            )
        )
        intent.assert_sealed()
        return intent

    @classmethod
    def _materialize_predata_transaction_locked_v67(
        cls,
        workspace: StageWorkspaceV50,
        state: _PreDataTransactionStateV67,
    ) -> _PreDataTransactionStateV67:
        if (
            state.status != "RECOVERY_PENDING"
            or state.intent is None
            or state.intent_ref is None
        ):
            raise StudioConflictError(
                "V6.7 pre-data reconciliation requires one pending intent"
            )
        cls._assert_predata_frontier_v67(workspace)
        intent = state.intent
        file_bytes = predata_contract_file_bytes_v67(
            intent.source_contract,
            intent.measurement_contract,
            intent.predata_protocol,
        )
        for relative_path, expected_bytes in file_bytes.items():
            path = workspace.root / relative_path
            if path.exists():
                try:
                    actual = path.read_bytes()
                except OSError as exc:
                    raise StudioConflictError(
                        "V6.7 pre-data projection cannot be read"
                    ) from exc
                if actual != expected_bytes:
                    raise StudioConflictError(
                        "V6.7 pre-data recovery refuses to overwrite a "
                        "conflicting projection"
                    )
                continue
            _write_bytes_new(path, expected_bytes)

        if not cls._intent_projection_files_complete_v67(
            workspace,
            intent,
        ):
            raise StudioConflictError(
                "V6.7 pre-data recovery did not materialize all projections"
            )
        preparation_payload = predata_preparation_payload_v67(
            workspace_spec_hash=intent.workspace_spec_hash,
            s0_gate_hash=intent.s0_gate_hash,
            source_contract=intent.source_contract,
            measurement_contract=intent.measurement_contract,
            predata_protocol=intent.predata_protocol,
            artifact_file_hashes=intent.artifact_file_hashes,
        )
        if sha256_value(preparation_payload) != (
            intent.preparation_evidence_payload_hash
        ):
            raise StudioConflictError(
                "V6.7 pre-data preparation payload differs from its intent"
            )
        preparation_ref = cls._commit_evidence_once(
            workspace,
            PREDATA_PREPARATION_EVIDENCE_KIND_V67,
            preparation_payload,
        )
        if state.completion is None:
            completion_draft = PreDataPreparationCompletionV67(
                workspace_spec_hash=intent.workspace_spec_hash,
                s0_gate_hash=intent.s0_gate_hash,
                workflow_mode_contract_hash=intent.workflow_mode_contract_hash,
                workflow_mode_artifact_hash=intent.workflow_mode_artifact_hash,
                evidence_scope=intent.evidence_scope,
                intent_hash=str(intent.intent_hash),
                intent_artifact_hash=state.intent_ref.sha256,
                preparation_evidence_artifact_hash=preparation_ref.sha256,
                preparation_evidence_payload_hash=(
                    intent.preparation_evidence_payload_hash
                ),
                source_contract_hash=str(intent.source_contract.contract_hash),
                measurement_contract_hash=str(
                    intent.measurement_contract.contract_hash
                ),
                protocol_hash=str(intent.predata_protocol.protocol_hash),
                artifact_file_hashes=intent.artifact_file_hashes,
                completed_at=_utc_now(),
                authority_key_id=workspace.authority_key_id,
            )
            completion = completion_draft.authenticate(
                workspace._mac(
                    PREDATA_PREPARATION_COMPLETION_KIND_V67,
                    completion_draft.unsigned_hash(),
                )
            )
            completion.assert_sealed()
            cls._commit_evidence_once(
                workspace,
                PREDATA_PREPARATION_COMPLETION_KIND_V67,
                completion.model_dump(mode="json"),
            )
        elif (
            state.completion.preparation_evidence_artifact_hash
            != preparation_ref.sha256
        ):
            raise StudioConflictError(
                "existing V6.7 completion differs from recovered preparation evidence"
            )
        completed = cls._predata_transaction_state_v67(workspace)
        if completed.status != "COMPLETED":
            raise StudioConflictError(
                "V6.7 pre-data transaction failed deterministic replay"
            )
        return completed

    def _append_predata_completion_event_v67(
        self,
        task_id: str,
        state: _PreDataTransactionStateV67,
    ) -> None:
        if (
            state.status != "COMPLETED"
            or state.bundle is None
            or state.intent is None
            or state.completion is None
            or state.preparation_ref is None
            or state.completion.completion_hash is None
        ):
            raise StudioConflictError(
                "V6.7 completion event requires a completed transaction"
            )
        source, measurement, protocol = state.bundle
        self._append_event(
            task_id,
            event_type="predata_bundle_prepared_v67",
            status="succeeded",
            message=(
                "Source semantics, measurement design, and execution protocol "
                "were frozen before S1 and observation access"
            ),
            details={
                "source_contract_hash": source.contract_hash,
                "measurement_contract_hash": measurement.contract_hash,
                "protocol_hash": protocol.protocol_hash,
                "preparation_intent_hash": state.intent.intent_hash,
                "preparation_evidence_hash": state.preparation_ref.sha256,
                "preparation_completion_hash": (state.completion.completion_hash),
                "network_accessed": False,
                "observation_values_accessed": False,
                "scientific_qualification_granted": False,
                "real_world_action_authorized": False,
            },
            idempotency_key=(f"predata-v67:{state.completion.completion_hash}"),
        )

    def reconcile_predata_v67(self, task_id: str) -> dict[str, Any]:
        """Resume exactly one authenticated pre-data intent without observation."""

        with self._lock:
            if task_id in self._active_tasks:
                raise StudioConflictError("task already has an active stage run")
        workspace = self._workspace(task_id)
        transaction_factory = getattr(
            getattr(getattr(workspace, "graph", None), "store", None),
            "writer_transaction",
            None,
        )
        if not callable(transaction_factory):
            raise StudioConflictError(
                "workspace does not expose the graph writer transaction"
            )
        with transaction_factory():
            workspace = self._workspace(task_id)
            if not workspace.verify():
                raise StudioConflictError(
                    "workspace failed verification inside pre-data recovery"
                )
            state = self._predata_transaction_state_v67(workspace)
            if state.status == "NOT_STARTED":
                raise StudioConflictError(
                    "no V6.7 pre-data transaction is available to reconcile"
                )
            if state.status == "STALE_PENDING":
                raise StudioConflictError(
                    "V6.7 pre-data intent belongs to a stale S0 authority and "
                    "requires explicit graph recovery"
                )
            if state.status == "RECOVERY_PENDING":
                state = self._materialize_predata_transaction_locked_v67(
                    workspace,
                    state,
                )
            if not workspace.verify():
                raise StudioConflictError(
                    "workspace failed verification after pre-data recovery"
                )
        if state.status == "COMPLETED":
            self._append_predata_completion_event_v67(task_id, state)
        return self.snapshot(task_id)

    def prepare_predata_v67(
        self,
        task_id: str,
        request: StudioWorldBankDataRequestV62 | dict[str, Any],
    ) -> dict[str, Any]:
        """Commit and complete an exact pre-data transaction before S1 or data."""

        try:
            validated = (
                request
                if isinstance(request, StudioWorldBankDataRequestV62)
                else StudioWorldBankDataRequestV62.model_validate(request)
            )
        except ValueError as exc:
            raise StudioValidationError(str(exc)) from exc
        with self._lock:
            if task_id in self._active_tasks:
                raise StudioConflictError("task already has an active stage run")
        workspace = self._workspace(task_id)
        transaction_factory = getattr(
            getattr(getattr(workspace, "graph", None), "store", None),
            "writer_transaction",
            None,
        )
        if not callable(transaction_factory):
            raise StudioConflictError(
                "workspace does not expose the graph writer transaction"
            )
        with transaction_factory():
            workspace = self._workspace(task_id)
            if not workspace.verify():
                raise StudioConflictError(
                    "workspace failed verification inside pre-data preparation"
                )
            if self._effective_workflow_mode(workspace) != "v67":
                raise StudioConflictError(
                    "V6.7 pre-data preparation requires a task frozen in "
                    "workflow_mode=v67"
                )
            if (
                validated.fixture_only
                and workspace.spec.evidence_scope != "development"
            ):
                raise StudioConflictError(
                    "fixture-only V6.7 pre-data requires development evidence scope"
                )
            if (
                not validated.fixture_only
                and workspace.spec.evidence_scope != "public_data"
            ):
                raise StudioConflictError(
                    "live V6.7 pre-data requires public_data evidence scope"
                )
            state = self._predata_transaction_state_v67(workspace)
            if state.status == "STALE_PENDING":
                raise StudioConflictError(
                    "V6.7 pre-data intent belongs to a stale S0 authority and "
                    "cannot be replaced in place"
                )
            s0_gate_hash = workspace.current_gate("S0")
            workspace_spec_hash = workspace.spec.spec_hash
            if s0_gate_hash is None or workspace_spec_hash is None:
                raise StudioConflictError(
                    "V6.7 pre-data preparation requires a sealed workspace and "
                    "an open current S0 gate"
                )
            try:
                expected_bundle = build_world_bank_predata_bundle_v67(
                    request=validated,
                    workspace_spec_hash=workspace_spec_hash,
                    s0_gate_hash=s0_gate_hash,
                )
            except (TypeError, ValueError) as exc:
                raise StudioValidationError(str(exc)) from exc
            if state.status != "NOT_STARTED":
                if state.bundle != expected_bundle:
                    raise StudioConflictError(
                        "request differs from the frozen V6.7 pre-data bundle"
                    )
                if state.status == "RECOVERY_PENDING":
                    state = self._materialize_predata_transaction_locked_v67(
                        workspace,
                        state,
                    )
            else:
                self._assert_predata_frontier_v67(workspace)
                if workspace._artifacts_of_kind(PREDATA_PREPARATION_EVIDENCE_KIND_V67):
                    raise StudioConflictError(
                        "V6.7 pre-data evidence exists without its intent"
                    )
                intent = self._build_predata_intent_v67(
                    workspace,
                    expected_bundle,
                )
                self._commit_evidence_once(
                    workspace,
                    PREDATA_PREPARATION_INTENT_KIND_V67,
                    intent.model_dump(mode="json"),
                )
                state = self._predata_transaction_state_v67(workspace)
                state = self._materialize_predata_transaction_locked_v67(
                    workspace,
                    state,
                )
            if not workspace.verify():
                raise StudioConflictError(
                    "workspace failed verification after pre-data preparation"
                )
        if state.status == "COMPLETED":
            self._append_predata_completion_event_v67(task_id, state)
        return self.snapshot(task_id)

    @staticmethod
    def _portfolio_query_v69(
        workspace: StageWorkspaceV50,
        request: StudioPortfolioPrepareRequestV69,
    ) -> CapabilityQueryV68:
        workspace_spec_hash = workspace.spec.spec_hash
        s0_gate_hash = workspace.current_gate("S0")
        if workspace_spec_hash is None or s0_gate_hash is None:
            raise StudioConflictError(
                "V6.9 portfolio preparation requires a sealed workspace and "
                "an open current S0 gate"
            )
        measurement_contract_hash = sha256_value(
            {
                "schema_version": "6.9-studio-portfolio-measurement-freeze",
                "workspace_spec_hash": workspace_spec_hash,
                "s0_gate_hash": s0_gate_hash,
                "planned_observation_count": request.planned_observation_count,
                "state_unit": request.state_unit,
                "time_unit": request.time_unit,
                "observation_values_included": False,
                "observed_statistics_included": False,
            }
        )
        return CapabilityQueryV68.seal(
            workspace_spec_hash=workspace_spec_hash,
            s0_gate_hash=s0_gate_hash,
            problem_signature=ProblemSignatureV60(
                state_kind="scalar",
                time_kind="continuous",
                dynamics_kind="autonomous",
                observation_kind="complete",
                task_kind="prediction",
                observation_count=request.planned_observation_count,
                positive_observations=True,
                strictly_increasing_time=True,
            ),
            claim_kind="predictive",
            measurement=MeasurementSignatureV68(
                measurement_contract_hash=measurement_contract_hash,
                scale_type="ratio",
                study_design_type="time_series",
                missingness_policy="reject_incomplete_series",
                measurement_unit=request.state_unit,
                time_basis=request.time_unit,
                minimum_planned_observations=(
                    request.planned_observation_count
                ),
            ),
        )

    def _append_portfolio_completion_event_v69(
        self,
        task_id: str,
        state: PortfolioTransactionStateV69,
    ) -> None:
        if (
            state.status != "COMPLETED"
            or state.run is None
            or state.completion is None
            or state.completion.completion_hash is None
        ):
            raise StudioConflictError(
                "V6.9 completion event requires a completed transaction"
            )
        run = state.run
        self._append_event(
            task_id,
            event_type="portfolio_v69_run_completed",
            status="succeeded",
            message=(
                "Development portfolio execution completed with "
                f"{run.final_decision}; no scientific qualification was granted"
            ),
            details={
                "protocol_hash": state.protocol_hash,
                "snapshot_hash": state.snapshot_hash,
                "outer_origin_plan_hash": state.plan_hash,
                "run_hash": state.run_hash,
                "completion_hash": state.completion.completion_hash,
                "decision": run.final_decision,
                "selected_branch_id": run.selected_branch_id,
                "reason_code": run.reason_code,
                "persistence_relative_improvement": (
                    run.persistence_relative_improvement
                ),
                "engineering_status": "COMPLETED",
                "scientific_evidence_status": "NOT_RUN",
                "claim_ceiling": "development_protocol_only",
                "s1_s6_gates_touched": False,
                "scientific_qualification_granted": False,
                "real_world_action_authorized": False,
            },
            idempotency_key=(
                f"portfolio-v69-completion:{state.completion.completion_hash}"
            ),
        )

    def prepare_portfolio_v69(
        self,
        task_id: str,
        request: StudioPortfolioPrepareRequestV69 | dict[str, Any],
    ) -> dict[str, Any]:
        """Freeze the exact two-pack development portfolio before data access."""

        try:
            validated = (
                request
                if isinstance(request, StudioPortfolioPrepareRequestV69)
                else StudioPortfolioPrepareRequestV69.model_validate(request)
            )
        except ValidationError as exc:
            raise StudioValidationError(_safe_validation_message(exc)) from exc
        except ValueError as exc:
            raise StudioValidationError(str(exc)) from exc
        with self._portfolio_persistent_mutation_claim_v69(task_id):
            workspace = self._workspace(task_id)
            if not self._portfolio_lane_eligible_v69(workspace):
                raise StudioConflictError(
                    "V6.9 portfolio preparation requires a development-scope "
                    "legacy task"
                )
            query = self._portfolio_query_v69(workspace, validated)
            transaction = self._portfolio_transaction_v69(workspace)
            maximum_origins = (
                validated.max_origins
                if validated.max_origins is not None
                else min(
                    12,
                    validated.planned_observation_count
                    - validated.initial_training_count,
                )
            )
            try:
                state = transaction.freeze(
                    query=query,
                    registry=default_development_capability_registry_v68(),
                    branch_budget=BranchBudgetV68.seal(
                        max_wall_seconds=60,
                        max_cpu_seconds=60,
                        max_memory_megabytes=512,
                        max_artifact_bytes=2_000_000,
                        max_model_calls=0,
                        max_tool_calls=2,
                    ),
                    portfolio_budget=PortfolioBudgetV68.seal(
                        max_parallel_branches=2,
                        total_wall_seconds=120,
                        total_cpu_seconds=120,
                        total_memory_megabytes=1024,
                        total_artifact_bytes=4_000_000,
                        total_model_calls=0,
                        total_tool_calls=4,
                    ),
                    time_unit=validated.time_unit,
                    snapshot_task_id=task_id,
                    initial_training_count=validated.initial_training_count,
                    maximum_origins=maximum_origins,
                    baseline_policy=PersistenceBaselinePolicyV69.seal(
                        minimum_relative_improvement=(
                            validated.min_relative_improvement
                        )
                    ),
                )
            except ValidationError as exc:
                raise StudioValidationError(
                    _safe_validation_message(exc)
                ) from exc
            except ValueError as exc:
                raise StudioValidationError(str(exc)) from exc
            except PermissionError as exc:
                raise StudioConflictError(str(exc)) from exc
            except StageWorkspaceError as exc:
                if str(exc).startswith(
                    "a different current V6.9 freeze intent"
                ):
                    raise StudioConflictError(
                        "request differs from the frozen V6.9 portfolio"
                    ) from exc
                raise
        if state.intent is None or state.intent.intent_hash is None:
            raise StudioConflictError(
                "V6.9 portfolio freeze did not produce an authenticated intent"
            )
        self._append_event(
            task_id,
            event_type="portfolio_v69_frozen",
            status="succeeded",
            message=(
                "Exact ODE and positive-log-increment branches were frozen; "
                "the freeze request contained no observations or statistics"
            ),
            details={
                "intent_hash": state.intent.intent_hash,
                "protocol_hash": state.protocol_hash,
                "planned_observation_count": (
                    state.intent.planned_observation_count
                ),
                "initial_training_count": state.intent.initial_training_count,
                "maximum_origins": state.intent.maximum_origins,
                "problem_signature_source": (
                    "caller_selected_v69_narrow_lane"
                ),
                "derived_from_s0_typed_problem_signature": False,
                "observation_values_accessed": False,
                "observed_statistics_accessed": False,
                "private_acceptance_data_accessed": False,
                "scientific_evidence_status": "NOT_RUN",
                "scientific_qualification_granted": False,
                "real_world_action_authorized": False,
            },
            idempotency_key=f"portfolio-v69-freeze:{state.intent.intent_hash}",
        )
        return self.snapshot(task_id)

    def ingest_portfolio_series_v69(
        self,
        task_id: str,
        request: StudioPortfolioSeriesRequestV69 | dict[str, Any],
    ) -> dict[str, Any]:
        """Bind one public series to the frozen portfolio without executing it."""

        try:
            validated = (
                request
                if isinstance(request, StudioPortfolioSeriesRequestV69)
                else StudioPortfolioSeriesRequestV69.model_validate(request)
            )
        except ValidationError as exc:
            raise StudioValidationError(_safe_validation_message(exc)) from exc
        except ValueError as exc:
            raise StudioValidationError(str(exc)) from exc
        with self._portfolio_persistent_mutation_claim_v69(task_id):
            workspace = self._workspace(task_id)
            if not self._portfolio_lane_eligible_v69(workspace):
                raise StudioConflictError(
                    "V6.9 portfolio data requires a development-scope legacy task"
                )
            transaction = self._portfolio_transaction_v69(workspace)
            before = self._project_portfolio_transaction_v69(transaction)
            if before.status == "STALE_PENDING":
                raise StudioConflictError(
                    "V6.9 portfolio intent belongs to a stale S0 authority"
                )
            if before.intent is None:
                raise StudioConflictError(
                    "V6.9 portfolio data requires a frozen portfolio intent"
                )
            try:
                snapshot = PositiveSeriesSnapshotV69.seal(
                    task_id=task_id,
                    time_unit=before.intent.time_unit,
                    state_unit=before.intent.state_unit,
                    times=validated.times,
                    observations=validated.observations,
                    source_id=validated.source_id,
                )
                state = transaction.stage_snapshot(snapshot)
            except ValidationError as exc:
                raise StudioValidationError(
                    _safe_validation_message(exc)
                ) from exc
            except ValueError as exc:
                raise StudioValidationError(str(exc)) from exc
            except PermissionError as exc:
                raise StudioConflictError(str(exc)) from exc
            except StageWorkspaceError as exc:
                if str(exc).startswith(
                    "a different V6.9 snapshot intent"
                ):
                    raise StudioConflictError(
                        "series differs from the staged V6.9 snapshot"
                    ) from exc
                raise
        if state.status == "STALE_PENDING":
            raise StudioConflictError(
                "V6.9 portfolio intent belongs to a stale S0 authority"
            )
        if state.run_intent is None or state.run_intent.run_intent_hash is None:
            raise StudioConflictError(
                "V6.9 data staging did not produce an authenticated run intent"
            )
        self._append_event(
            task_id,
            event_type="portfolio_v69_data_staged",
            status="succeeded",
            message=(
                "Caller-declared public positive series and common rolling "
                "origins were staged; no capability branch ran"
            ),
            details={
                "run_intent_hash": state.run_intent.run_intent_hash,
                "snapshot_hash": state.snapshot_hash,
                "outer_origin_plan_hash": state.plan_hash,
                "observation_count": len(validated.observations),
                "caller_declared_public_data": True,
                "public_source_verified": False,
                "private_acceptance_data_accessed": False,
                "branch_execution_started": False,
                "scientific_evidence_status": "NOT_RUN",
                "scientific_qualification_granted": False,
                "real_world_action_authorized": False,
            },
            idempotency_key=(
                f"portfolio-v69-data:{state.run_intent.run_intent_hash}"
            ),
        )
        return self.snapshot(task_id)

    def run_portfolio_v69(self, task_id: str) -> dict[str, Any]:
        """Execute or resume the authenticated development portfolio."""

        with self._portfolio_mutation_claim_v69(task_id):
            self._run_portfolio_v69_claimed(task_id)
        return self.snapshot(task_id)

    def _run_portfolio_v69_claimed(self, task_id: str) -> dict[str, Any]:
        with exclusive_file_lock(
            self._portfolio_lane_lock_path_v69(task_id)
        ):
            return self._run_portfolio_v69_under_lane_lock(task_id)

    def _run_portfolio_v69_under_lane_lock(
        self,
        task_id: str,
    ) -> dict[str, Any]:
        workspace = self._workspace(task_id)
        if not self._portfolio_lane_eligible_v69(workspace):
            raise StudioConflictError(
                "V6.9 portfolio execution requires a development-scope legacy task"
            )
        transaction = self._portfolio_transaction_v69(workspace)
        before = self._project_portfolio_transaction_v69(transaction)
        if before.status == "STALE_PENDING":
            raise StudioConflictError(
                "V6.9 portfolio intent belongs to a stale S0 authority"
            )
        if before.status == "COMPLETED":
            self._append_portfolio_completion_event_v69(task_id, before)
            return self.snapshot(task_id)
        if before.status not in {"DATA_STAGED", "RECOVERY_PENDING"}:
            raise StudioConflictError(
                "V6.9 portfolio run requires a committed public-series intent"
            )
        try:
            transaction.execute()
            state = self._project_portfolio_transaction_v69(transaction)
        except PermissionError as exc:
            raise StudioConflictError(str(exc)) from exc
        except (StageWorkspaceError, ValidationError, ValueError) as exc:
            raise StudioBridgeError(
                "V6.9 portfolio execution failed integrity verification"
            ) from exc
        self._append_portfolio_completion_event_v69(task_id, state)
        return self.snapshot(task_id)

    def start_portfolio_v69(self, task_id: str) -> dict[str, Any]:
        """Start the data-bound portfolio in a background worker."""

        workspace = self._workspace(task_id)
        if not self._portfolio_lane_eligible_v69(workspace):
            raise StudioConflictError(
                "V6.9 portfolio execution requires a development-scope legacy task"
            )
        state = self._project_portfolio_transaction_v69(
            self._portfolio_transaction_v69(workspace)
        )
        if state.status == "COMPLETED":
            self._append_portfolio_completion_event_v69(task_id, state)
            return self.snapshot(task_id)
        if state.status not in {"DATA_STAGED", "RECOVERY_PENDING"}:
            raise StudioConflictError(
                "V6.9 portfolio run requires a committed public-series intent"
            )
        with self._lock:
            if task_id in self._active_tasks:
                raise StudioConflictError("task already has an active stage run")
            workspace = self._workspace(task_id)
            if not self._portfolio_lane_eligible_v69(workspace):
                raise StudioConflictError(
                    "V6.9 portfolio execution requires a development-scope "
                    "legacy task"
                )
            state = self._project_portfolio_transaction_v69(
                self._portfolio_transaction_v69(workspace)
            )
            if state.status not in {"DATA_STAGED", "RECOVERY_PENDING"}:
                raise StudioConflictError(
                    "V6.9 portfolio run is no longer staged or recoverable"
                )
        claim = self._prepare_operator_run_v70(task_id, "run_portfolio_v69")
        with self._lock:
            if task_id in self._active_tasks:
                conflict = StudioConflictError(
                    "task already has an active stage run"
                )
                self._fail_operator_claim_v70(claim, conflict)
                raise conflict
            self._active_tasks.add(task_id)
        run_intent_hash = (
            state.run_intent.run_intent_hash
            if state.run_intent is not None
            else None
        )
        try:
            self._append_event(
                task_id,
                event_type="portfolio_v69_run_accepted",
                status="accepted",
                message="Authenticated V6.9 development portfolio run accepted",
                details={
                    "run_intent_hash": run_intent_hash,
                    "operator_work_id": claim.lease.work_id,
                    "operator_packet_hash": claim.packet.packet_hash,
                    "scientific_evidence_status": "NOT_RUN",
                    "scientific_qualification_granted": False,
                    "real_world_action_authorized": False,
                },
                idempotency_key=(
                    f"portfolio-v69-run-accepted:{run_intent_hash}"
                ),
            )
        except Exception as exc:
            with self._lock:
                self._active_tasks.discard(task_id)
            self._fail_operator_claim_v70(claim, exc)
            raise

        def worker() -> None:
            try:
                self._execute_operator_run_v70(
                    claim,
                    lambda: self._run_portfolio_v69_claimed(task_id),
                )
            except OperatorPlaneError as exc:
                self._append_event(
                    task_id,
                    event_type="operator_reconcile_required",
                    status="blocked",
                    message="Portfolio authority may have advanced; operator reconciliation is required",
                    details={
                        "operator_work_id": claim.lease.work_id,
                        "error_type": type(exc).__name__,
                        "scientific_evidence_status": "NOT_RUN",
                        "scientific_qualification_granted": False,
                        "real_world_action_authorized": False,
                    },
                )
            except Exception as exc:
                self._append_event(
                    task_id,
                    event_type="portfolio_v69_run_failed",
                    status="failed",
                    message="V6.9 development portfolio run failed closed",
                    details={
                        "error_type": type(exc).__name__,
                        "scientific_evidence_status": "NOT_RUN",
                        "scientific_qualification_granted": False,
                        "real_world_action_authorized": False,
                    },
                )
            finally:
                with self._lock:
                    self._active_tasks.discard(task_id)

        thread = threading.Thread(
            target=worker,
            name=f"fma-studio-{task_id}-portfolio-v69",
            daemon=True,
        )
        try:
            thread.start()
        except Exception as exc:
            with self._lock:
                self._active_tasks.discard(task_id)
            self._fail_operator_claim_v70(claim, exc)
            raise
        return self.snapshot(task_id)

    def reconcile_portfolio_v69(self, task_id: str) -> dict[str, Any]:
        """Explicitly replay a staged or partially committed V6.9 run."""

        with self._portfolio_persistent_mutation_claim_v69(task_id):
            workspace = self._workspace(task_id)
            if not self._portfolio_lane_eligible_v69(workspace):
                raise StudioConflictError(
                    "V6.9 portfolio recovery requires a development-scope "
                    "legacy task"
                )
            transaction = self._portfolio_transaction_v69(workspace)
            before = self._project_portfolio_transaction_v69(transaction)
            if before.status == "STALE_PENDING":
                raise StudioConflictError(
                    "V6.9 portfolio intent belongs to a stale S0 authority"
                )
            if before.status not in {
                "DATA_STAGED",
                "RECOVERY_PENDING",
                "COMPLETED",
            }:
                raise StudioConflictError(
                    "no staged or partial V6.9 portfolio run is available "
                    "to reconcile"
                )
            try:
                state = transaction.reconcile()
            except PermissionError as exc:
                raise StudioConflictError(str(exc)) from exc
            except (StageWorkspaceError, ValidationError, ValueError) as exc:
                raise StudioBridgeError(
                    "V6.9 portfolio recovery failed integrity verification"
                ) from exc
        if state.status != "COMPLETED":
            raise StudioConflictError(
                "V6.9 portfolio reconciliation did not reach completion"
            )
        self._append_portfolio_completion_event_v69(task_id, state)
        return self.snapshot(task_id)

    @staticmethod
    def _commit_evidence_once(
        workspace: StageWorkspaceV50,
        kind: str,
        payload: dict[str, Any],
    ) -> ArtifactRef:
        matches = [
            reference
            for reference, existing in workspace._artifacts_of_kind(kind)
            if existing == payload
        ]
        if len(matches) > 1:
            raise StudioConflictError(
                f"multiple committed {kind} artifacts have identical content"
            )
        if matches:
            return matches[0]
        return workspace.commit_evidence(kind, payload)

    @staticmethod
    def _s1_repair_context_records_v67(
        workspace: StageWorkspaceV50,
        protocol: PreDataExecutionProtocolV67,
    ) -> dict[int, tuple[ArtifactRef, S1BoundedRepairContextV67]]:
        records: dict[
            int,
            tuple[ArtifactRef, S1BoundedRepairContextV67],
        ] = {}
        for reference, payload in workspace._artifacts_of_kind(
            "s1_bounded_repair_context_v67"
        ):
            try:
                context = S1BoundedRepairContextV67.model_validate(payload)
                context.assert_sealed()
            except (TypeError, ValidationError, ValueError) as exc:
                raise StudioConflictError(
                    "committed V6.7 S1 repair context is invalid"
                ) from exc
            if (
                context.workspace_spec_hash != workspace.spec.spec_hash
                or context.s0_gate_hash != workspace.current_gate("S0")
                or context.predata_protocol_hash != protocol.protocol_hash
            ):
                continue
            if context.successor_attempt in records:
                raise StudioConflictError(
                    "multiple V6.7 S1 repair contexts claim one attempt"
                )
            records[context.successor_attempt] = (reference, context)
        return records

    @staticmethod
    def _s1_rejection_handoff_records_v67(
        workspace: StageWorkspaceV50,
        protocol: PreDataExecutionProtocolV67,
    ) -> dict[
        int,
        tuple[ArtifactRef, S1FormalizationRejectionHandoffV67],
    ]:
        records: dict[
            int,
            tuple[ArtifactRef, S1FormalizationRejectionHandoffV67],
        ] = {}
        for reference, payload in workspace._artifacts_of_kind(
            "s1_formalization_rejection_handoff_v67"
        ):
            try:
                handoff = S1FormalizationRejectionHandoffV67.model_validate(payload)
                handoff.assert_sealed()
            except (TypeError, ValidationError, ValueError) as exc:
                raise StudioConflictError(
                    "committed V6.7 S1 rejection handoff is invalid"
                ) from exc
            if (
                handoff.workspace_spec_hash != workspace.spec.spec_hash
                or handoff.s0_gate_hash != workspace.current_gate("S0")
                or handoff.predata_protocol_hash != protocol.protocol_hash
            ):
                continue
            if handoff.predecessor_attempt in records:
                raise StudioConflictError(
                    "multiple V6.7 S1 rejection handoffs claim one attempt"
                )
            records[handoff.predecessor_attempt] = (reference, handoff)
        return records

    def _commit_s1_rejection_handoff_v67(
        self,
        *,
        workspace: StageWorkspaceV50,
        protocol: PreDataExecutionProtocolV67,
        handoff: S1FormalizationRejectionHandoffV67,
    ) -> tuple[ArtifactRef, S1FormalizationRejectionHandoffV67]:
        records = self._s1_rejection_handoff_records_v67(
            workspace,
            protocol,
        )
        existing = records.get(handoff.predecessor_attempt)
        if existing is not None:
            if existing[1].handoff_hash != handoff.handoff_hash:
                raise StudioConflictError(
                    "another V6.7 S1 rejection handoff already owns the current attempt"
                )
            return existing
        reference = self._commit_evidence_once(
            workspace,
            "s1_formalization_rejection_handoff_v67",
            handoff.model_dump(mode="json"),
        )
        return reference, handoff

    def _append_s1_recovery_event_once(
        self,
        task_id: str,
        *,
        event_type: str,
        binding_key: str,
        binding_value: str,
        status: Literal[
            "accepted",
            "running",
            "succeeded",
            "failed",
            "blocked",
        ],
        message: str,
        details: dict[str, Any],
    ) -> None:
        if any(
            event.event_type == event_type
            and event.details.get(binding_key) == binding_value
            for event in self._events(task_id)
        ):
            return
        self._append_event(
            task_id,
            event_type=event_type,
            status=status,
            message=message,
            details=details,
        )

    @staticmethod
    def _s1_transition_for_handoff_v67(
        *,
        workspace: StageWorkspaceV50,
        recovery_kernel: RecoveryKernelV60,
        handoff_ref: ArtifactRef,
        handoff: S1FormalizationRejectionHandoffV67,
    ) -> tuple[
        FailureDiagnosisV60,
        RecoveryPlanV60,
        RecoveryTransitionReceiptV60,
    ]:
        recovery_refs = s1_recovery_evidence_refs_v67(
            handoff,
            handoff_artifact_hash=handoff_ref.sha256,
        )
        matches = [
            (diagnosis, plan, receipt)
            for _, receipt, diagnosis, plan in (
                recovery_kernel.completed_transition_records()
            )
            if diagnosis.workspace_spec_hash == workspace.spec.spec_hash
            and diagnosis.failed_stage == "S1"
            and diagnosis.category == "review_rejection"
            and diagnosis.failure_code == S1_FORMALIZATION_FAILURE_CODE_V67
            and diagnosis.failure_signature == handoff.failure_signature
            and diagnosis.evidence_refs == recovery_refs
        ]
        if len(matches) > 1:
            raise StudioConflictError(
                "V6.7 S1 rejection handoff has multiple recovery transitions"
            )
        if matches:
            diagnosis, plan, receipt = matches[0]
        else:
            try:
                diagnosis, plan, receipt = recovery_kernel.recover(
                    failed_stage="S1",
                    category="review_rejection",
                    failure_code=S1_FORMALIZATION_FAILURE_CODE_V67,
                    evidence_refs=recovery_refs,
                    expected_information_gain=(handoff.expected_information_gain),
                    holdout_exposed=False,
                    private_evidence_used=False,
                )
            except (
                OSError,
                PermissionError,
                StageWorkspaceError,
                ValidationError,
                ValueError,
            ) as exc:
                raise StudioValidationError(
                    f"V6.7 S1 graph recovery failed: {exc}"
                ) from exc
        if (
            diagnosis.evidence_refs != recovery_refs
            or diagnosis.failure_signature != handoff.failure_signature
            or plan.diagnosis_hash != diagnosis.diagnosis_hash
            or receipt.diagnosis_hash != diagnosis.diagnosis_hash
            or receipt.plan_hash != plan.plan_hash
        ):
            raise StudioConflictError(
                "V6.7 S1 recovery transition differs from its handoff"
            )
        if handoff.recovery_disposition == "bounded_patch":
            if (
                plan.action != "PATCH"
                or plan.revoke_from != "S1"
                or not plan.automatic_execution_permitted
                or receipt.status != "ATTEMPT_CREATED"
                or receipt.predecessor_attempt != handoff.predecessor_attempt
                or receipt.successor_attempt != handoff.predecessor_attempt + 1
            ):
                raise StudioConflictError(
                    "V6.7 S1 bounded recovery differs from its handoff"
                )
        elif (
            plan.action != "HUMAN"
            or plan.revoke_from is not None
            or plan.automatic_execution_permitted
            or receipt.status != "HUMAN_REQUIRED"
            or receipt.predecessor_attempt is not None
            or receipt.successor_attempt is not None
        ):
            raise StudioConflictError(
                "V6.7 repeated S1 rejection did not terminate for human review"
            )
        return diagnosis, plan, receipt

    def _materialize_s1_recovery_handoff_v67(
        self,
        *,
        task_id: str,
        workspace: StageWorkspaceV50,
        protocol: PreDataExecutionProtocolV67,
        recovery_kernel: RecoveryKernelV60,
        handoff_ref: ArtifactRef,
        handoff: S1FormalizationRejectionHandoffV67,
        context_records: dict[
            int,
            tuple[ArtifactRef, S1BoundedRepairContextV67],
        ],
    ) -> S1BoundedRepairContextV67:
        diagnosis, plan, receipt = self._s1_transition_for_handoff_v67(
            workspace=workspace,
            recovery_kernel=recovery_kernel,
            handoff_ref=handoff_ref,
            handoff=handoff,
        )
        if handoff.recovery_disposition == "terminal_human":
            existing_context = context_records.get(handoff.predecessor_attempt)
            if (
                existing_context is None
                or existing_context[1].context_hash
                != handoff.existing_repair_context_hash
            ):
                raise StudioConflictError(
                    "terminal V6.7 S1 rejection is not bound to the current "
                    "repair context"
                )
            terminal_ref = self._commit_evidence_once(
                workspace,
                "s1_formalization_rejection_terminal_v67",
                {
                    "schema_version": ("6.7-s1-formalization-rejection-terminal"),
                    "workspace_spec_hash": handoff.workspace_spec_hash,
                    "s0_gate_hash": handoff.s0_gate_hash,
                    "predata_protocol_hash": handoff.predata_protocol_hash,
                    "reviewer_receipt_hash": handoff.reviewer_receipt_hash,
                    "reviewer_finding_signature": (handoff.reviewer_finding_signature),
                    "failure_code": S1_FORMALIZATION_FAILURE_CODE_V67,
                    "failure_signature": handoff.failure_signature,
                    "handoff_hash": handoff.handoff_hash,
                    "handoff_artifact_hash": handoff_ref.sha256,
                    "diagnosis_hash": diagnosis.diagnosis_hash,
                    "plan_hash": plan.plan_hash,
                    "recovery_receipt_hash": receipt.receipt_hash,
                    "predecessor_attempt": handoff.predecessor_attempt,
                    "terminal_status": receipt.status,
                    "private_evidence_used": False,
                    "holdout_exposed": False,
                    "scientific_failure_established": False,
                    "scientific_qualification_granted": False,
                    "real_world_action_authorized": False,
                },
            )
            self._append_s1_recovery_event_once(
                task_id,
                event_type="s1_graph_recovery_stopped_v67",
                binding_key="terminal_evidence_hash",
                binding_value=terminal_ref.sha256,
                status="blocked",
                message=(
                    "S1 formalization recovery stopped under the code-owned "
                    "recovery policy"
                ),
                details={
                    "failure_code": S1_FORMALIZATION_FAILURE_CODE_V67,
                    "failure_signature": handoff.failure_signature,
                    "recovery_status": receipt.status,
                    "handoff_artifact_hash": handoff_ref.sha256,
                    "terminal_evidence_hash": terminal_ref.sha256,
                    "scientific_qualification_granted": False,
                    "real_world_action_authorized": False,
                },
            )
            raise StudioValidationError(
                f"V6.7 S1 formalization recovery terminated with {receipt.status}"
            )

        rejection = build_s1_formalization_rejection_evidence_v67(
            workspace_spec_hash=handoff.workspace_spec_hash,
            s0_gate_hash=handoff.s0_gate_hash,
            predata_protocol_hash=handoff.predata_protocol_hash,
            reviewer_receipt_hash=handoff.reviewer_receipt_hash,
            findings=[item.normalized_finding for item in handoff.findings],
            reviewer_finding_signature=(handoff.reviewer_finding_signature),
            diagnosis=diagnosis,
            plan=plan,
            recovery_receipt=receipt,
        )
        rejection_ref = self._commit_evidence_once(
            workspace,
            "s1_formalization_rejection_evidence_v67",
            rejection.model_dump(mode="json"),
        )
        repair_context = build_s1_bounded_repair_context_v67(rejection)
        successor_attempt = repair_context.successor_attempt
        existing_context = context_records.get(successor_attempt)
        if existing_context is not None:
            if existing_context[1].context_hash != repair_context.context_hash:
                raise StudioConflictError(
                    "V6.7 S1 successor has a conflicting repair context"
                )
            context_ref, repair_context = existing_context
        else:
            context_ref = self._commit_evidence_once(
                workspace,
                "s1_bounded_repair_context_v67",
                repair_context.model_dump(mode="json"),
            )
            context_records[successor_attempt] = (
                context_ref,
                repair_context,
            )
        self._append_s1_recovery_event_once(
            task_id,
            event_type="s1_graph_repair_attempt_created_v67",
            binding_key="repair_context_artifact_hash",
            binding_value=context_ref.sha256,
            status="succeeded",
            message=(
                "Independent S1 rejection created one bounded successor graph attempt"
            ),
            details={
                "failure_code": S1_FORMALIZATION_FAILURE_CODE_V67,
                "failure_signature": handoff.failure_signature,
                "reviewer_finding_signature": (handoff.reviewer_finding_signature),
                "handoff_artifact_hash": handoff_ref.sha256,
                "recovery_receipt_hash": receipt.receipt_hash,
                "predecessor_attempt": receipt.predecessor_attempt,
                "successor_attempt": receipt.successor_attempt,
                "rejection_evidence_artifact_hash": rejection_ref.sha256,
                "repair_context_artifact_hash": context_ref.sha256,
                "protocol_change_permitted": False,
                "adapter_change_permitted": False,
                "threshold_change_permitted": False,
                "scientific_qualification_granted": False,
                "real_world_action_authorized": False,
            },
        )
        return repair_context

    def _reconcile_s1_recovery_handoffs_v67(
        self,
        *,
        task_id: str,
        workspace: StageWorkspaceV50,
        protocol: PreDataExecutionProtocolV67,
        recovery_kernel: RecoveryKernelV60,
    ) -> S1BoundedRepairContextV67 | None:
        """Finish any write-ahead S1 recovery before applying restart guards."""

        context_records = self._s1_repair_context_records_v67(
            workspace,
            protocol,
        )
        handoff_records = self._s1_rejection_handoff_records_v67(
            workspace,
            protocol,
        )
        processed_attempts: set[int] = set()

        while True:
            current_attempt = workspace._latest_attempt("S1")
            pending: list[
                tuple[
                    ArtifactRef,
                    S1FormalizationRejectionHandoffV67,
                ]
            ] = []
            if current_attempt > 1:
                predecessor = handoff_records.get(current_attempt - 1)
                if (
                    predecessor is not None
                    and predecessor[1].recovery_disposition == "bounded_patch"
                    and predecessor[1].predecessor_attempt not in processed_attempts
                ):
                    pending.append(predecessor)
            current = handoff_records.get(current_attempt)
            if (
                current is not None
                and current[1].predecessor_attempt not in processed_attempts
            ):
                pending.append(current)
            if not pending:
                break

            for handoff_ref, handoff in pending:
                predecessor_attempt = handoff.predecessor_attempt
                latest_attempt = workspace._latest_attempt("S1")
                if handoff.recovery_disposition == "bounded_patch":
                    if predecessor_attempt not in {
                        latest_attempt,
                        latest_attempt - 1,
                    }:
                        raise StudioConflictError(
                            "V6.7 S1 recovery handoff is outside the current "
                            "attempt lineage"
                        )
                elif predecessor_attempt != latest_attempt:
                    raise StudioConflictError(
                        "terminal V6.7 S1 recovery handoff is not current"
                    )
                self._materialize_s1_recovery_handoff_v67(
                    task_id=task_id,
                    workspace=workspace,
                    protocol=protocol,
                    recovery_kernel=recovery_kernel,
                    handoff_ref=handoff_ref,
                    handoff=handoff,
                    context_records=context_records,
                )
                processed_attempts.add(predecessor_attempt)

        current_attempt = workspace._latest_attempt("S1")
        current_context = context_records.get(current_attempt)
        return current_context[1] if current_context is not None else None

    def _workspace_after_predata_reconciliation_v67(
        self,
        task_id: str,
    ) -> StageWorkspaceV50:
        workspace = self._workspace(task_id)
        if self._effective_workflow_mode(workspace) != "v67":
            return workspace
        state = self._predata_transaction_state_v67(workspace)
        if state.status == "STALE_PENDING":
            raise StudioConflictError(
                "V6.7 pre-data intent belongs to a stale S0 authority"
            )
        if state.status == "RECOVERY_PENDING":
            self.reconcile_predata_v67(task_id)
            workspace = self._workspace(task_id)
            state = self._predata_transaction_state_v67(workspace)
        if state.status == "COMPLETED":
            self._append_predata_completion_event_v67(task_id, state)
        return workspace

    def run_s1(self, task_id: str) -> dict[str, Any]:
        workspace = self._workspace_after_predata_reconciliation_v67(task_id)
        with self._portfolio_mutation_claim_v69(task_id):
            workspace = self._workspace(task_id)
            self._assert_no_portfolio_lane_v69(workspace)
            self._run_s1_claimed(task_id, workspace)
        return self.snapshot(task_id)

    def _run_s1_claimed(
        self,
        task_id: str,
        workspace: StageWorkspaceV50,
    ) -> dict[str, Any]:
        del workspace
        with exclusive_file_lock(
            self._portfolio_lane_lock_path_v69(task_id)
        ):
            current = self._workspace(task_id)
            self._assert_no_portfolio_lane_v69(current)
            return self._run_s1_under_lane_lock_v69(task_id, current)

    def _run_s1_under_lane_lock_v69(
        self,
        task_id: str,
        workspace: StageWorkspaceV50,
    ) -> dict[str, Any]:
        self._assert_no_portfolio_lane_v69(workspace)
        workflow_mode = self._effective_workflow_mode(workspace)
        predata_bundle = self._authoritative_predata_bundle_v67(workspace)
        if workflow_mode == "v67" and predata_bundle is None:
            raise StudioConflictError("V6.7 pre-data bundle must be prepared before S1")
        if workspace.current_gate("S1"):
            return self.snapshot(task_id)
        if not workspace.current_gate("S0"):
            raise StudioConflictError("S1 requires an open current S0 gate")
        measurement_contract = predata_bundle[1] if predata_bundle is not None else None
        predata_protocol = predata_bundle[2] if predata_bundle is not None else None
        recovery_kernel = RecoveryKernelV60(workspace)
        repair_context: S1BoundedRepairContextV67 | None = None
        if predata_protocol is not None:
            repair_context = self._reconcile_s1_recovery_handoffs_v67(
                task_id=task_id,
                workspace=workspace,
                protocol=predata_protocol,
                recovery_kernel=recovery_kernel,
            )
        recovery_state = recovery_kernel.load_state()
        if predata_protocol is not None and (
            (
                recovery_state.human_required
                and recovery_state.human_reason == S1_FORMALIZATION_FAILURE_CODE_V67
            )
            or (
                recovery_state.stopped
                and recovery_state.stop_reason == S1_FORMALIZATION_FAILURE_CODE_V67
            )
        ):
            raise StudioConflictError(
                "V6.7 S1 recovery is terminal and requires inspection or "
                "human resolution"
            )
        if predata_protocol is not None:
            current_attempt = workspace._latest_attempt("S1")
            if (
                current_attempt > 1
                and recovery_state.last_action == "PATCH"
                and recovery_state.last_revoke_from == "S1"
                and repair_context is None
            ):
                raise StudioConflictError(
                    "V6.7 S1 successor attempt lacks its committed bounded "
                    "repair context; reconciliation is required"
                )

        while True:
            if any((workspace.root / relative).exists() for relative in _S1_PATHS):
                raise StudioConflictError(
                    "S1 contains partial artifacts; automatic re-execution is blocked"
                )
            orchestrator = StudioS1OrchestratorV58(
                workspace=workspace,
                task_id=task_id,
                driver_factory=lambda: StageRoleDriverV51(self._transport(task_id)),
                event_callback=lambda event_type, status, message, details: (
                    self._append_event(
                        task_id,
                        event_type=event_type,
                        status=status,
                        message=message,
                        details=details,
                    )
                ),
                measurement_contract_v67=measurement_contract,
                predata_protocol_v67=predata_protocol,
                repair_context_v67=repair_context,
            )
            try:
                orchestrator.run()
                return self.snapshot(task_id)
            except S1FormalizationRejectedV67 as exc:
                if predata_protocol is None or predata_protocol.protocol_hash is None:
                    raise StudioValidationError(str(exc)) from exc
                workspace_hash = workspace.spec.spec_hash
                s0_gate_hash = workspace.current_gate("S0")
                if workspace_hash is None or s0_gate_hash is None:
                    raise StudioConflictError(
                        "V6.7 S1 recovery requires current workspace authority"
                    ) from exc
                if exc.protocol_hash != predata_protocol.protocol_hash:
                    raise StudioConflictError(
                        "V6.7 S1 rejection belongs to another pre-data protocol"
                    ) from exc
                handoff = build_s1_formalization_rejection_handoff_v67(
                    workspace_spec_hash=str(workspace_hash),
                    s0_gate_hash=str(s0_gate_hash),
                    predata_protocol_hash=exc.protocol_hash,
                    reviewer_receipt_hash=exc.reviewer_receipt_hash,
                    findings=list(exc.normalized_findings),
                    reviewer_finding_signature=(exc.normalized_finding_signature),
                    predecessor_attempt=workspace._latest_attempt("S1"),
                    existing_repair_context_hash=(
                        str(repair_context.context_hash)
                        if repair_context is not None
                        and repair_context.context_hash is not None
                        else None
                    ),
                )
                handoff_ref, handoff = self._commit_s1_rejection_handoff_v67(
                    workspace=workspace,
                    protocol=predata_protocol,
                    handoff=handoff,
                )
                if (
                    exc.handoff_artifact_hash is not None
                    and exc.handoff_artifact_hash != handoff_ref.sha256
                ):
                    raise StudioConflictError(
                        "runtime and service disagree on the V6.7 S1 rejection handoff"
                    ) from exc
                context_records = self._s1_repair_context_records_v67(
                    workspace,
                    predata_protocol,
                )
                repair_context = self._materialize_s1_recovery_handoff_v67(
                    task_id=task_id,
                    workspace=workspace,
                    protocol=predata_protocol,
                    recovery_kernel=recovery_kernel,
                    handoff_ref=handoff_ref,
                    handoff=handoff,
                    context_records=context_records,
                )
            except (S1RuntimeError, ValidationError) as exc:
                raise StudioValidationError(str(exc)) from exc

    def start_s1(self, task_id: str) -> dict[str, Any]:
        workspace = self._workspace_after_predata_reconciliation_v67(task_id)
        if not workspace.current_gate("S0"):
            raise StudioConflictError("S1 requires an open current S0 gate")
        if (
            self._effective_workflow_mode(workspace) == "v67"
            and self._authoritative_predata_bundle_v67(workspace) is None
        ):
            raise StudioConflictError("V6.7 pre-data bundle must be prepared before S1")
        with self._lock:
            if task_id in self._active_tasks:
                raise StudioConflictError("task already has an active stage run")
            workspace = self._workspace(task_id)
            self._assert_no_portfolio_lane_v69(workspace)
            if not workspace.current_gate("S0"):
                raise StudioConflictError("S1 requires an open current S0 gate")
            if (
                self._effective_workflow_mode(workspace) == "v67"
                and self._authoritative_predata_bundle_v67(workspace) is None
            ):
                raise StudioConflictError(
                    "V6.7 pre-data bundle must be prepared before S1"
                )
        claim = self._prepare_operator_run_v70(task_id, "run_s1")
        with self._lock:
            if task_id in self._active_tasks:
                conflict = StudioConflictError(
                    "task already has an active stage run"
                )
                self._fail_operator_claim_v70(claim, conflict)
                raise conflict
            self._active_tasks.add(task_id)
        try:
            self._append_event(
                task_id,
                event_type="s1_run_accepted",
                status="accepted",
                message="Bounded graph-native S1 run accepted by the local bridge",
                details={
                    "operator_work_id": claim.lease.work_id,
                    "operator_packet_hash": claim.packet.packet_hash,
                },
            )
        except Exception as exc:
            with self._lock:
                self._active_tasks.discard(task_id)
            self._fail_operator_claim_v70(claim, exc)
            raise

        def worker() -> None:
            try:
                self._execute_operator_run_v70(
                    claim,
                    lambda: self._run_s1_claimed(
                        task_id,
                        self._workspace(task_id),
                    ),
                )
            except OperatorPlaneError as exc:
                self._append_event(
                    task_id,
                    event_type="operator_reconcile_required",
                    status="blocked",
                    message="S1 authority may have advanced; operator reconciliation is required",
                    details={
                        "operator_work_id": claim.lease.work_id,
                        "error_type": type(exc).__name__,
                        "scientific_qualification_granted": False,
                        "real_world_action_authorized": False,
                    },
                )
            except Exception as exc:
                self._append_event(
                    task_id,
                    event_type="s1_run_failed",
                    status="failed",
                    message="S1 run failed closed",
                    details={
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                        "scientific_qualification_granted": False,
                        "real_world_action_authorized": False,
                    },
                )
            finally:
                with self._lock:
                    self._active_tasks.discard(task_id)

        thread = threading.Thread(
            target=worker,
            name=f"fma-studio-{task_id}-s1",
            daemon=True,
        )
        try:
            thread.start()
        except Exception as exc:
            with self._lock:
                self._active_tasks.discard(task_id)
            self._fail_operator_claim_v70(claim, exc)
            raise
        return self.snapshot(task_id)

    def ingest_ode_data(
        self,
        task_id: str,
        request: StudioODEDataRequestV59 | dict[str, Any],
    ) -> dict[str, Any]:
        workspace = self._workspace(task_id)
        if self._effective_workflow_mode(workspace) == "v67":
            raise StudioConflictError(
                "V6.7 tasks require the frozen official-source intake; "
                "legacy ODE data ingestion is forbidden"
            )
        try:
            validated = (
                request
                if isinstance(request, StudioODEDataRequestV59)
                else StudioODEDataRequestV59.model_validate(request)
            )
        except ValueError as exc:
            raise StudioValidationError(str(exc)) from exc
        with self._lock:
            if task_id in self._active_tasks:
                raise StudioConflictError("task already has an active stage run")
            try:
                raw_path = ingest_ode_data_v59(workspace, validated)
            except (BackhalfRuntimeError, ValidationError) as exc:
                raise StudioValidationError(str(exc)) from exc
            self._append_event(
                task_id,
                event_type="s2_raw_data_received",
                status="succeeded",
                message=("Positive scalar ODE data were received; S2 remains unfrozen"),
                details={
                    "adapter_id": validated.adapter_id,
                    "point_count": len(validated.times),
                    "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                    "fixture_only": validated.fixture_only,
                    "scientific_qualification_granted": False,
                    "real_world_action_authorized": False,
                },
            )
        return self.snapshot(task_id)

    def ingest_world_bank_data(
        self,
        task_id: str,
        request: StudioWorldBankDataRequestV62 | dict[str, Any],
    ) -> dict[str, Any]:
        """Fetch one registered official series and prepare an unfrozen S2 input."""

        workspace = self._workspace_after_predata_reconciliation_v67(task_id)
        try:
            validated = (
                request
                if isinstance(request, StudioWorldBankDataRequestV62)
                else StudioWorldBankDataRequestV62.model_validate(request)
            )
            contract = world_bank_source_contract_from_studio_request_v67(validated)
        except ValueError as exc:
            raise StudioValidationError(str(exc)) from exc
        workflow_mode = self._effective_workflow_mode(workspace)
        predata_bundle = self._authoritative_predata_bundle_v67(workspace)
        if workflow_mode == "v67" and predata_bundle is None:
            raise StudioConflictError(
                "V6.7 official-source intake requires the frozen pre-data bundle"
            )
        if predata_bundle is not None:
            workspace_spec_hash = workspace.spec.spec_hash
            s0_gate_hash = workspace.current_gate("S0")
            if workspace_spec_hash is None or s0_gate_hash is None:
                raise StudioConflictError(
                    "V6.7 source intake requires current workspace authority"
                )
            try:
                expected_predata = build_world_bank_predata_bundle_v67(
                    request=validated,
                    workspace_spec_hash=workspace_spec_hash,
                    s0_gate_hash=s0_gate_hash,
                )
            except (TypeError, ValueError) as exc:
                raise StudioValidationError(str(exc)) from exc
            if predata_bundle != expected_predata:
                raise StudioConflictError(
                    "official-source request differs from the frozen V6.7 "
                    "source, measurement, or execution semantics"
                )
        if workspace.current_gate("S1") is None:
            raise StudioConflictError("official-source intake requires an open S1 gate")
        if workspace.current_gate("S2") is not None:
            raise StudioConflictError("S2 is already frozen")
        if (
            not validated.fixture_only
            and workspace.spec.evidence_scope != "public_data"
        ):
            raise StudioConflictError(
                "live official-source intake requires public_data evidence scope"
            )
        if (
            validated.fixture_only
            and workflow_mode == "v67"
            and workspace.spec.evidence_scope != "development"
        ):
            raise StudioConflictError(
                "fixture-only V6.7 source intake requires development evidence scope"
            )
        artifact_paths = [
            RAW_RELATIVE_PATH,
            SOURCE_RAW_PATH,
            SOURCE_RECEIPT_PATH,
            SOURCE_VERIFICATION_PATH,
            SOURCE_ACQUISITION_AUTH_PATH,
            S2_SOURCE_REVERIFICATION_PATH,
            MEASUREMENT_SCHEMA_PATH,
        ]
        if predata_bundle is None:
            artifact_paths.append(SOURCE_CONTRACT_PATH)
        if any((workspace.root / item).exists() for item in artifact_paths):
            raise StudioConflictError("official-source intake artifacts already exist")
        with self._lock:
            if task_id in self._active_tasks:
                raise StudioConflictError("task already has an active stage run")
            try:
                validate_v67_pre_acquisition_v67(
                    workspace,
                    adapter_id=validated.adapter_id,
                    source_contract=contract,
                    measurement_unit=validated.state_unit,
                    time_basis=validated.observation_time_basis,
                    missing_value_policy="reject_incomplete_series",
                )
                source_authority = SourceTransportAuthorityV62.from_stage_workspace(
                    workspace
                )
                acquisition = source_authority.acquire_world_bank_series(
                    workspace_spec=workspace.spec,
                    task_id=task_id,
                    contract=contract,
                    fetcher=self.world_bank_fetcher,
                )
                fetched = acquisition.fetched
                measurement = MeasurementSchemaV62.seal(
                    measurement_id=validated.contract_id,
                    source_contract_hash=fetched.contract.contract_hash,
                    indicator_id=fetched.contract.indicator_id,
                    semantic_name=validated.semantic_name,
                    operational_definition=validated.operational_definition,
                    observation_time_basis=(validated.observation_time_basis),
                    aggregation_level=validated.aggregation_level,
                    time_unit=fetched.snapshot.time_unit,
                    state_unit=fetched.snapshot.state_unit,
                )
                raw_request = StudioODEDataRequestV59(
                    adapter_id=validated.adapter_id,
                    time_unit=fetched.snapshot.time_unit,
                    state_unit=fetched.snapshot.state_unit,
                    times=fetched.snapshot.times,
                    observations=fetched.snapshot.observations,
                    source_id=fetched.receipt.source_id,
                    license_status=(
                        "world_bank_default_open_data_recorded;"
                        "independent_license_review_absent"
                    ),
                    fixture_only=fetched.snapshot.fixture_only,
                )
                validate_v67_data_compatibility_v67(
                    workspace,
                    raw_request,
                    source_contract=fetched.contract,
                    source_receipt=fetched.receipt,
                    source_acquisition_receipt=(acquisition.authority_receipt),
                    measurement_schema=measurement,
                    source_raw_body=fetched.raw_body,
                    require_source_evidence=True,
                )
                if predata_bundle is None:
                    materialize_world_bank_series_v62(
                        workspace_root=workspace.root,
                        fetched=fetched,
                    )
                else:
                    if fetched.contract != predata_bundle[0]:
                        raise StudioConflictError(
                            "fetched source contract differs from V6.7 freeze"
                        )
                    _write_bytes_new(
                        workspace.root / SOURCE_RAW_PATH,
                        fetched.raw_body,
                    )
                    _write_json_new(
                        workspace.root / SOURCE_RECEIPT_PATH,
                        fetched.receipt.model_dump(mode="json"),
                    )
                _write_json_new(
                    workspace.root / SOURCE_ACQUISITION_AUTH_PATH,
                    acquisition.authority_receipt.model_dump(mode="json"),
                )
                _write_json_new(
                    workspace.root / MEASUREMENT_SCHEMA_PATH,
                    measurement.model_dump(mode="json"),
                )
                verification = verify_world_bank_source_v62(
                    workspace_root=workspace.root,
                    contract=fetched.contract,
                    receipt=fetched.receipt,
                    snapshot=fetched.snapshot,
                )
                if verification.status != "PASS":
                    raise StudioValidationError(
                        "official source failed code-owned replay"
                    )
                raw_path = ingest_ode_data_v59(
                    workspace,
                    raw_request,
                )
            except (
                BackhalfRuntimeError,
                ValidationError,
                ValueError,
            ) as exc:
                raise StudioValidationError(str(exc)) from exc
            self._append_event(
                task_id,
                event_type="s2_official_source_received_v62",
                status="succeeded",
                message=(
                    "Registered World Bank bytes were fetched, replayed, and "
                    "prepared for S2; semantic review remains separate"
                ),
                details={
                    "adapter_id": validated.adapter_id,
                    "contract_hash": fetched.contract.contract_hash,
                    "receipt_hash": fetched.receipt.receipt_hash,
                    "acquisition_authority_receipt_hash": (
                        acquisition.authority_receipt.receipt_hash
                    ),
                    "verification_hash": verification.verification_hash,
                    "source_integrity_status": verification.status,
                    "scientific_provenance_status": (
                        verification.scientific_provenance_status
                    ),
                    "point_count": len(fetched.snapshot.times),
                    "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                    "fixture_only": fetched.snapshot.fixture_only,
                    "scientific_qualification_granted": False,
                    "real_world_action_authorized": False,
                },
            )
        return self.snapshot(task_id)

    def _execute_recovery(
        self,
        task_id: str,
        workspace: StageWorkspaceV50,
        request: StudioRecoveryRequestV60,
    ) -> dict[str, Any]:
        kernel = RecoveryKernelV60(workspace)
        evidence_refs = kernel.evidence_refs_for_stage(request.failed_stage)
        try:
            diagnosis, plan, receipt = kernel.recover(
                failed_stage=request.failed_stage,
                category=request.category,
                failure_code=request.failure_code,
                evidence_refs=evidence_refs,
                expected_information_gain=request.expected_information_gain,
                holdout_exposed=request.holdout_exposed,
                private_evidence_used=request.private_evidence_used,
            )
        except (
            OSError,
            PermissionError,
            StageWorkspaceError,
            ValidationError,
            ValueError,
        ) as exc:
            raise StudioValidationError(str(exc)) from exc
        details = {
            "diagnosis_hash": diagnosis.diagnosis_hash,
            "failure_signature": diagnosis.failure_signature,
            "action": plan.action,
            "revoke_from": plan.revoke_from,
            "transition_status": receipt.status,
            "predecessor_attempt": receipt.predecessor_attempt,
            "successor_attempt": receipt.successor_attempt,
            "quarantined_paths": sorted(receipt.quarantined_file_hashes),
            "scientific_qualification_granted": False,
            "real_world_action_authorized": False,
        }
        self._append_event(
            task_id,
            event_type="recovery_transition_v60",
            status=(
                "blocked"
                if receipt.status in {"ABSTAINED", "HUMAN_REQUIRED"}
                else "succeeded"
            ),
            message=(
                "Recovery stopped without mutating scientific state"
                if receipt.status in {"ABSTAINED", "HUMAN_REQUIRED"}
                else "Graph-native recovery prepared the next bounded run"
            ),
            details=details,
        )
        return details

    def recover(
        self,
        task_id: str,
        request: StudioRecoveryRequestV60 | dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize one failure and execute only the code-authorized recovery."""

        workspace = self._workspace(task_id)
        try:
            validated = (
                request
                if isinstance(request, StudioRecoveryRequestV60)
                else StudioRecoveryRequestV60.model_validate(request)
            )
        except ValueError as exc:
            raise StudioValidationError(str(exc)) from exc
        if validated.failed_stage == "S0" and validated.category in {
            "contract_semantics",
            "review_rejection",
        }:
            raise StudioValidationError(
                "S0 semantic or review recovery requires an authenticated "
                "typed S0 review handoff"
            )
        with self._lock:
            if task_id in self._active_tasks:
                raise StudioConflictError("task already has an active stage run")
            self._execute_recovery(task_id, workspace, validated)
        return self.snapshot(task_id)

    def _backhalf_orchestrator(
        self,
        task_id: str,
        workspace: StageWorkspaceV50,
    ) -> StudioBackhalfOrchestratorV59:
        return StudioBackhalfOrchestratorV59(
            workspace=workspace,
            task_id=task_id,
            driver_factory=lambda: StageRoleDriverV51(self._transport(task_id)),
            event_callback=lambda event_type, status, message, details: (
                self._append_event(
                    task_id,
                    event_type=event_type,
                    status=status,
                    message=message,
                    details=details,
                )
            ),
        )

    @staticmethod
    def _closed_stage_recovery_request(
        workspace: StageWorkspaceV50,
        decisions: dict[str, str],
    ) -> StudioRecoveryRequestV60 | None:
        failed_stage = next(
            (stage for stage, decision in decisions.items() if decision == "BLOCKED"),
            None,
        )
        if failed_stage is None:
            return None
        stage = cast(StageId, failed_stage)
        try:
            manifest = workspace._manifest_for_stage(stage)
        except StageWorkspaceError:
            return StudioRecoveryRequestV60(
                failed_stage=stage,
                category="numerical_implementation",
                failure_code=f"{stage.lower()}_mechanical_failure",
            )
        checks = workspace._latest_checks(stage, str(manifest.manifest_hash))
        failed = [item for item in checks.values() if item.status in {"FAIL", "ERROR"}]
        if not failed:
            # A reviewer rejection or policy mismatch needs independent human
            # interpretation; do not turn it into an automatic model mutation.
            return None
        levels = {item.level for item in failed}
        adapter_id = StudioTaskService._adapter_before_recovery(workspace)
        if stage == "S2":
            category: FailureCategoryV60 = "data_contract"
        elif stage == "S3":
            if adapter_id == ADAPTIVE_ADAPTER_ID and "L0" not in levels:
                category = "capability_gap"
            else:
                category = (
                    "numerical_implementation" if "L0" in levels else "model_assumption"
                )
        elif stage == "S4":
            if adapter_id == ADAPTIVE_ADAPTER_ID:
                category = "capability_gap"
            else:
                category = (
                    "uncertainty_calibration"
                    if levels.issubset({"L4"})
                    else "model_assumption"
                )
        elif stage == "S5":
            category = "decision_support"
        elif stage == "S6":
            category = "paper_consistency"
        else:
            return None
        codes = sorted(item.reason_code for item in failed)
        return StudioRecoveryRequestV60(
            failed_stage=stage,
            category=category,
            failure_code=f"{stage.lower()}_{hashlib.sha256(canonical_json(codes).encode('utf-8')).hexdigest()[:16]}",
        )

    def _run_backhalf_attempt(
        self,
        task_id: str,
        workspace: StageWorkspaceV50,
    ) -> dict[str, str]:
        try:
            decisions = self._backhalf_orchestrator(
                task_id,
                workspace,
            ).run()
        except BackhalfRuntimeError as exc:
            match = re.search(r"\b(S[2-6]) contains partial artifacts\b", str(exc))
            if match is None:
                raise StudioValidationError(str(exc)) from exc
            stage = cast(StageId, match.group(1))
            self._execute_recovery(
                task_id,
                workspace,
                StudioRecoveryRequestV60(
                    failed_stage=stage,
                    category="partial_artifact",
                    failure_code=f"{stage.lower()}_interrupted_before_manifest",
                ),
            )
            try:
                decisions = self._backhalf_orchestrator(
                    task_id,
                    workspace,
                ).run()
            except (BackhalfRuntimeError, ValidationError, ValueError) as retry_exc:
                raise StudioValidationError(str(retry_exc)) from retry_exc
        except (BackhalfRuntimeError, ValidationError, ValueError) as exc:
            raise StudioValidationError(str(exc)) from exc
        return decisions

    @staticmethod
    def _adapter_before_recovery(
        workspace: StageWorkspaceV50,
    ) -> str | None:
        path = workspace.root / "docs" / "adapter_binding.json"
        if not path.is_file():
            return None
        try:
            return str(json.loads(path.read_text(encoding="utf-8"))["adapter_id"])
        except (KeyError, TypeError, json.JSONDecodeError):
            return None

    @staticmethod
    def _registered_branch_can_resume(
        workspace: StageWorkspaceV50,
        adapter_before_recovery: str | None,
        recovery_details: dict[str, Any] | None,
    ) -> bool:
        if (
            adapter_before_recovery != ODE_ADAPTER_ID
            or recovery_details is None
            or recovery_details.get("action") != "BRANCH"
            or recovery_details.get("revoke_from") != "S1"
            or recovery_details.get("transition_status") != "ATTEMPT_CREATED"
        ):
            return False
        raw_path = workspace.root / "data" / "raw" / "ode_series.json"
        if not raw_path.is_file():
            return False
        try:
            request = StudioODEDataRequestV59.model_validate_json(
                raw_path.read_text(encoding="utf-8")
            )
            return (
                len(request.observations) >= 26
                and RecoveryKernelV60(workspace).load_state().last_action == "BRANCH"
            )
        except (OSError, ValidationError, ValueError):
            return False

    def run_backhalf(self, task_id: str) -> dict[str, Any]:
        workspace = self._workspace(task_id)
        while True:
            decisions = self._run_backhalf_attempt(task_id, workspace)
            recovery_details: dict[str, Any] | None = None
            adapter_before_recovery = self._adapter_before_recovery(workspace)
            recovery_request = self._closed_stage_recovery_request(workspace, decisions)
            if recovery_request is not None:
                recovery_details = self._execute_recovery(
                    task_id, workspace, recovery_request
                )
            if decisions.get("S6") == "OPEN":
                report, evidence_hash = materialize_scientific_success_v61(workspace)
                self._append_event(
                    task_id,
                    event_type="scientific_success_evaluated_v61",
                    status="succeeded",
                    message=(
                        "Claim-relative scientific success was evaluated "
                        "without granting qualification"
                    ),
                    details={
                        "report_hash": report.report_hash,
                        "evidence_hash": evidence_hash,
                        "local_predictive_gate_status": (
                            report.local_predictive_gate_status
                        ),
                        "scientific_success_status": (report.scientific_success_status),
                        "claim_ceiling": report.claim_ceiling,
                        "scientific_qualification_granted": False,
                        "real_world_action_authorized": False,
                    },
                )
                if (workspace.root / PROVENANCE_BINDING_PATH).is_file():
                    try:
                        (
                            admission,
                            closure,
                            verification,
                            closure_evidence_hashes,
                        ) = materialize_scientific_closure_v62(workspace)
                    except (OSError, TypeError, ValueError) as exc:
                        self._append_event(
                            task_id,
                            event_type="scientific_closure_blocked_v62",
                            status="blocked",
                            message=(
                                "S0-S6 remained complete, but current evidence "
                                "could not be admitted into scientific closure"
                            ),
                            details={
                                "error_type": type(exc).__name__,
                                "reason": str(exc)[:500],
                                "scientific_qualification_granted": False,
                                "real_world_action_authorized": False,
                            },
                        )
                    else:
                        self._append_event(
                            task_id,
                            event_type="scientific_closure_evaluated_v62",
                            status="succeeded",
                            message=(
                                "Current typed S1-S5 evidence was admitted and "
                                "the post-S6 scientific closure was replayed"
                            ),
                            details={
                                "admission_hash": admission.admission_hash,
                                "report_hash": closure.report_hash,
                                "verification_hash": (verification.verification_hash),
                                "evidence_hashes": closure_evidence_hashes,
                                "local_evidence_status": (
                                    closure.local_evidence_status
                                ),
                                "scientific_closure_status": (
                                    closure.scientific_closure_status
                                ),
                                "claim_ceiling": closure.claim_ceiling,
                                "scientific_qualification_granted": False,
                                "real_world_action_authorized": False,
                            },
                        )
                else:
                    self._append_event(
                        task_id,
                        event_type="scientific_closure_not_run_v62",
                        status="blocked",
                        message=(
                            "V6.2 closure remained NOT_RUN because no "
                            "authenticated provenance binding exists"
                        ),
                        details={
                            "reason_code": ("authenticated_provenance_binding_absent"),
                            "scientific_qualification_granted": False,
                            "real_world_action_authorized": False,
                        },
                    )
            self._append_event(
                task_id,
                event_type="backhalf_run_completed",
                status=("succeeded" if decisions.get("S6") == "OPEN" else "blocked"),
                message=(
                    "Registered S2-S6 path completed"
                    if decisions.get("S6") == "OPEN"
                    else "Registered S2-S6 path stopped at a closed gate"
                ),
                details={
                    "decisions": decisions,
                    "recovery": recovery_details,
                    "scientific_qualification_granted": False,
                    "real_world_action_authorized": False,
                },
            )
            if not self._registered_branch_can_resume(
                workspace,
                adapter_before_recovery,
                recovery_details,
            ):
                break
            self._append_event(
                task_id,
                event_type="registered_branch_resume_started",
                status="running",
                message=(
                    "The new S1 attempt will use the registered adaptive "
                    "positive-series capability pack"
                ),
                details={
                    "predecessor_adapter_id": adapter_before_recovery,
                    "successor_adapter_id": ADAPTIVE_ADAPTER_ID,
                    "successor_attempt": recovery_details["successor_attempt"],
                    "scientific_qualification_granted": False,
                    "real_world_action_authorized": False,
                },
            )
            self._run_s1_claimed(task_id, self._workspace(task_id))
        return self.snapshot(task_id)

    def start_backhalf(self, task_id: str) -> dict[str, Any]:
        workspace = self._workspace(task_id)
        if not workspace.current_gate("S1"):
            raise StudioConflictError(
                "S2-S6 execution requires an open current S1 gate"
            )
        with self._lock:
            if task_id in self._active_tasks:
                raise StudioConflictError("task already has an active stage run")
        claim = self._prepare_operator_run_v70(task_id, "run_backhalf")
        with self._lock:
            if task_id in self._active_tasks:
                conflict = StudioConflictError(
                    "task already has an active stage run"
                )
                self._fail_operator_claim_v70(claim, conflict)
                raise conflict
            self._active_tasks.add(task_id)
        try:
            self._append_event(
                task_id,
                event_type="backhalf_run_accepted",
                status="accepted",
                message="Registered scalar ODE S2-S6 run accepted by the local bridge",
                details={
                    "operator_work_id": claim.lease.work_id,
                    "operator_packet_hash": claim.packet.packet_hash,
                },
            )
        except Exception as exc:
            with self._lock:
                self._active_tasks.discard(task_id)
            self._fail_operator_claim_v70(claim, exc)
            raise

        def worker() -> None:
            try:
                self._execute_operator_run_v70(
                    claim,
                    lambda: self.run_backhalf(task_id),
                )
            except OperatorPlaneError as exc:
                self._append_event(
                    task_id,
                    event_type="operator_reconcile_required",
                    status="blocked",
                    message="S2-S6 authority may have advanced; operator reconciliation is required",
                    details={
                        "operator_work_id": claim.lease.work_id,
                        "error_type": type(exc).__name__,
                        "scientific_qualification_granted": False,
                        "real_world_action_authorized": False,
                    },
                )
            except Exception as exc:
                self._append_event(
                    task_id,
                    event_type="backhalf_run_failed",
                    status="failed",
                    message="S2-S6 run failed closed",
                    details={
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                        "scientific_qualification_granted": False,
                        "real_world_action_authorized": False,
                    },
                )
            finally:
                with self._lock:
                    self._active_tasks.discard(task_id)

        thread = threading.Thread(
            target=worker,
            name=f"fma-studio-{task_id}-backhalf",
            daemon=True,
        )
        try:
            thread.start()
        except Exception as exc:
            with self._lock:
                self._active_tasks.discard(task_id)
            self._fail_operator_claim_v70(claim, exc)
            raise
        return self.snapshot(task_id)
