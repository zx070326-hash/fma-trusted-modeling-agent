"""Typed, additive S0--S6 workspace protocol for FMA V5.

These schemas deliberately separate four kinds of evidence:

* scientific computation,
* integrity verification,
* workflow presence, and
* reviewer judgement.

A stage gate is a workflow transition only.  It can never grant scientific
qualification or authorize a real-world action; those authorities remain in
the existing private promotion/human-approval protocols.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256, _assert_timezone


StageId = Literal["S0", "S1", "S2", "S3", "S4", "S5", "S6"]
Actor = Literal["model", "harness", "verifier", "human"]
CheckLevel = Literal["L0", "L1", "L2", "L3", "L4", "L5"]
CheckStatus = Literal["PASS", "FAIL", "NOT_RUN", "NOT_APPLICABLE", "ERROR"]
EvidenceClass = Literal[
    "scientific_computation",
    "integrity",
    "workflow_presence",
    "reviewer_judgment",
]
EvidenceScope = Literal[
    "structural",
    "synthetic_fixture",
    "development",
    "public_data",
    "private_holdout",
    "real_world",
]
ReviewRole = Literal[
    "literature_scout",
    "referee",
    "red_team",
    "data_auditor",
    "numerics_auditor",
    "final_red_team",
]
ReviewVerdict = Literal["APPROVE", "REJECT", "HUMAN"]
ExecutionRole = Literal["modeler", "literature_scout", "writer"]
GateDecision = Literal["OPEN", "BLOCKED", "NEEDS_EVIDENCE"]
Applicability = Literal["applicable", "not_applicable"]


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _assert_unique(values: list[str], label: str) -> None:
    if values != sorted(set(values)):
        raise ValueError(f"{label} must be sorted and unique")


class WorkflowProfileV50(StrictModel):
    """Frozen mechanism assignment, recorded before a task starts."""

    schema_version: Literal["5.0"] = "5.0"
    gate: bool = True
    redteam: bool = True
    compete: bool = True
    checks_advanced: bool = True
    buildpaper: bool = True
    skills: bool = True
    ensemble: bool = True
    gold_injection_allowed: Literal[False] = False
    profile_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_profile(self) -> "WorkflowProfileV50":
        if self.profile_hash and self.profile_hash != self.content_hash():
            raise ValueError("profile_hash does not match workflow profile")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "profile_hash")

    def assert_sealed(self) -> None:
        if not self.profile_hash or self.profile_hash != self.content_hash():
            raise ValueError("workflow profile is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "WorkflowProfileV50":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"profile_hash"}),
            profile_hash=draft.content_hash(),
        )


class TaskWorkspaceSpecV50(StrictModel):
    """Immutable public task/control envelope bound into the V4 graph."""

    schema_version: Literal["5.0"] = "5.0"
    workspace_id: Identifier
    graph_id: Identifier
    objective: Annotated[str, Field(min_length=5, max_length=4_000)]
    mission_hash: Sha256
    evidence_snapshot_hash: Sha256
    evaluator_epoch: Identifier
    profile: WorkflowProfileV50
    evidence_scope: EvidenceScope = "development"
    max_nodes: Annotated[int, Field(ge=14, le=512)] = 64
    max_outcomes: Annotated[int, Field(ge=14, le=512)] = 64
    max_failures: Annotated[int, Field(ge=1, le=128)] = 16
    permitted_actions: list[str] = Field(
        default_factory=lambda: ["local_compute", "write_local_run_artifacts"]
    )
    forbidden_actions: list[str] = Field(
        default_factory=lambda: ["external_action", "read_private_acceptance"]
    )
    created_at: datetime
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    private_acceptance_data_exposed: Literal[False] = False
    spec_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "TaskWorkspaceSpecV50":
        _assert_timezone(self.created_at, "created_at")
        self.profile.assert_sealed()
        if len(self.workspace_id) > 60 or len(self.graph_id) > 64:
            raise ValueError("workspace_id/graph_id exceed safe RunStore lengths")
        if self.graph_id != f"v5-{self.workspace_id}":
            raise ValueError("graph_id must be 'v5-' followed by workspace_id")
        _assert_unique(self.permitted_actions, "permitted_actions")
        _assert_unique(self.forbidden_actions, "forbidden_actions")
        if set(self.permitted_actions) & set(self.forbidden_actions):
            raise ValueError("permitted and forbidden actions overlap")
        if not {"external_action", "read_private_acceptance"}.issubset(
            self.forbidden_actions
        ):
            raise ValueError(
                "V5 workspaces must forbid external action and private acceptance"
            )
        if not all(
            (
                self.profile.gate,
                self.profile.redteam,
                self.profile.compete,
                self.profile.checks_advanced,
                self.profile.buildpaper,
                self.profile.skills,
                self.profile.ensemble,
            )
        ):
            raise ValueError(
                "V5 runtime currently supports only the full mechanism profile; "
                "ablations require an external materialized runner"
            )
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match task workspace spec")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "spec_hash")

    def assert_sealed(self) -> None:
        if not self.spec_hash or self.spec_hash != self.content_hash():
            raise ValueError("task workspace spec is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "TaskWorkspaceSpecV50":
        data.setdefault("created_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"spec_hash"}),
            spec_hash=draft.content_hash(),
        )


class AuthorityGenesisV50(StrictModel):
    """Immutable binding between one graph genesis and its external key."""

    schema_version: Literal["5.0"] = "5.0"
    workspace_spec_hash: Sha256
    graph_id: Identifier
    authority_key_id: Identifier
    authority_key_commitment: Sha256
    created_at: datetime
    authority_auth_tag: Sha256 | None = None
    genesis_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_genesis(self) -> "AuthorityGenesisV50":
        _assert_timezone(self.created_at, "created_at")
        if self.authority_auth_tag and not self.genesis_hash:
            raise ValueError("authenticated genesis requires genesis_hash")
        if self.genesis_hash:
            if _hash_without(self, "genesis_hash") != self.genesis_hash:
                raise ValueError("genesis_hash does not match authority genesis")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "authority_auth_tag", "genesis_hash")


class RegimeDiagnosisV50(StrictModel):
    """S0's four irreducible modelling questions plus a computable decision."""

    schema_version: Literal["5.0"] = "5.0"
    system_boundary: Annotated[str, Field(min_length=10)]
    state_and_memory: Annotated[str, Field(min_length=10)]
    uncertainty_and_data: Annotated[str, Field(min_length=10)]
    decision_and_loss: Annotated[str, Field(min_length=10)]
    query_type: Literal[
        "explanation", "prediction", "control", "optimization", "design", "mixed"
    ]
    downstream_decision: Annotated[str, Field(min_length=5)]
    decision_function_id: Identifier
    computable_decision_function: Annotated[str, Field(min_length=5)]
    evidence_hashes: Annotated[list[Sha256], Field(min_length=1)]
    limitations: Annotated[list[str], Field(min_length=1)]
    diagnosis_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_diagnosis(self) -> "RegimeDiagnosisV50":
        _assert_unique(self.evidence_hashes, "evidence_hashes")
        if self.diagnosis_hash and self.diagnosis_hash != self.content_hash():
            raise ValueError("diagnosis_hash does not match regime diagnosis")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "diagnosis_hash")

    @classmethod
    def seal(cls, **data: object) -> "RegimeDiagnosisV50":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"diagnosis_hash"}),
            diagnosis_hash=draft.content_hash(),
        )


class DecisionFunctionCanaryV50(StrictModel):
    canary_id: Identifier
    inputs: dict[Identifier, Annotated[float, Field(allow_inf_nan=False)]]
    expected: Annotated[float, Field(allow_inf_nan=False)]
    tolerance: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1e-9


class DecisionFunctionSpecV50(StrictModel):
    """A small auditable arithmetic decision/loss function for S0."""

    schema_version: Literal["5.0"] = "5.0"
    function_id: Identifier
    input_names: Annotated[list[Identifier], Field(min_length=1)]
    expression: Annotated[str, Field(min_length=1, max_length=1_000)]
    sense: Literal["minimize", "maximize", "report_only"]
    output_unit: Annotated[str, Field(min_length=1)]
    canaries: Annotated[list[DecisionFunctionCanaryV50], Field(min_length=1)]
    function_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_function(self) -> "DecisionFunctionSpecV50":
        _assert_unique(self.input_names, "input_names")
        expected = set(self.input_names)
        canary_ids: set[str] = set()
        for canary in self.canaries:
            if set(canary.inputs) != expected:
                raise ValueError(
                    f"canary {canary.canary_id} inputs differ from input_names"
                )
            if canary.canary_id in canary_ids:
                raise ValueError("decision function canary IDs must be unique")
            canary_ids.add(canary.canary_id)
        if self.function_hash and self.function_hash != self.content_hash():
            raise ValueError("function_hash does not match decision function")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "function_hash")

    @classmethod
    def seal(cls, **data: object) -> "DecisionFunctionSpecV50":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"function_hash"}),
            function_hash=draft.content_hash(),
        )


class AssumptionRecordV50(StrictModel):
    assumption_id: Identifier
    statement: Annotated[str, Field(min_length=5)]
    failure_consequence: Annotated[str, Field(min_length=5)]
    falsification_test: Annotated[str, Field(min_length=5)]
    abandon_criterion: Annotated[str, Field(min_length=5)]


class AssumptionSetV50(StrictModel):
    schema_version: Literal["5.0"] = "5.0"
    assumptions: Annotated[list[AssumptionRecordV50], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_assumptions(self) -> "AssumptionSetV50":
        ids = [item.assumption_id for item in self.assumptions]
        if len(ids) != len(set(ids)):
            raise ValueError("assumption IDs must be unique")
        return self


class SymbolRecordV50(StrictModel):
    symbol_id: Identifier
    meaning: Annotated[str, Field(min_length=3)]
    unit: Annotated[str, Field(min_length=1)]
    role: Literal["state", "parameter", "decision", "observable", "derived"]
    lower_bound: float | None = Field(default=None, allow_inf_nan=False)
    upper_bound: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_bounds(self) -> "SymbolRecordV50":
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError(f"invalid bounds for {self.symbol_id}")
        return self


class SymbolTableV50(StrictModel):
    schema_version: Literal["5.0"] = "5.0"
    symbols: Annotated[list[SymbolRecordV50], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_symbols(self) -> "SymbolTableV50":
        ids = [item.symbol_id for item in self.symbols]
        if len(ids) != len(set(ids)):
            raise ValueError("symbol IDs must be unique")
        return self


class CandidateFormalizationV50(StrictModel):
    candidate_id: Identifier
    model_family: Annotated[str, Field(min_length=3)]
    mathematical_form: Annotated[str, Field(min_length=5)]
    assumption_ids: Annotated[list[Identifier], Field(min_length=1)]
    symbol_ids: Annotated[list[Identifier], Field(min_length=1)]
    data_requirement_ids: Annotated[list[Identifier], Field(min_length=1)]
    validation_obligation_ids: Annotated[list[Identifier], Field(min_length=1)]
    abandon_criteria: Annotated[list[str], Field(min_length=1)]
    lineage: Annotated[str, Field(min_length=3)]

    @model_validator(mode="after")
    def validate_candidate(self) -> "CandidateFormalizationV50":
        for field_name in (
            "assumption_ids",
            "symbol_ids",
            "data_requirement_ids",
            "validation_obligation_ids",
        ):
            values = list(getattr(self, field_name))
            _assert_unique(values, field_name)
        return self

    def structural_hash(self) -> str:
        # Identity, prose lineage, and a family label cannot make two copies of
        # the same mathematical proposal count as competing structures.
        return sha256_value(
            self.model_dump(
                mode="json",
                exclude={"candidate_id", "lineage", "model_family"},
            )
        )


class CandidateSetV50(StrictModel):
    schema_version: Literal["5.0"] = "5.0"
    candidates: Annotated[list[CandidateFormalizationV50], Field(min_length=1)]
    generation_receipt_hashes: list[Sha256] = Field(default_factory=list)
    literature_scout_receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_candidates(self) -> "CandidateSetV50":
        ids = [item.candidate_id for item in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate IDs must be unique")
        hashes = [item.structural_hash() for item in self.candidates]
        if len(hashes) != len(set(hashes)):
            raise ValueError("candidate formalizations must be structurally distinct")
        _assert_unique(self.generation_receipt_hashes, "generation_receipt_hashes")
        return self


class RoleExecutionReceiptV50(StrictModel):
    """Harness-authenticated evidence that a role ran in a named fresh context."""

    schema_version: Literal["5.0"] = "5.0"
    execution_id: Identifier
    stage: StageId
    role: ExecutionRole
    subject_id: Identifier
    input_authority_hash: Sha256
    run_id: Identifier
    context_id: Identifier
    provider: Annotated[str, Field(min_length=1)]
    model: Annotated[str, Field(min_length=1)]
    prompt_hash: Sha256
    output_schema_hash: Sha256
    transport_trace_hash: Sha256
    output_artifact_hash: Sha256
    completed: Literal[True] = True
    issued_by: Literal["harness"]
    issued_at: datetime
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_execution(self) -> "RoleExecutionReceiptV50":
        _assert_timezone(self.issued_at, "issued_at")
        if self.authority_auth_tag and not self.receipt_hash:
            raise ValueError("authenticated role execution requires receipt_hash")
        if self.receipt_hash:
            expected = _hash_without(self, "receipt_hash")
            if expected != self.receipt_hash:
                raise ValueError("receipt_hash does not match role execution")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "authority_auth_tag", "receipt_hash")


class ModelSpecV50(StrictModel):
    schema_version: Literal["5.0"] = "5.0"
    selected_candidate_id: Identifier
    selected_candidate_structural_hash: Sha256
    selection_rationale: Annotated[str, Field(min_length=5)]
    assumption_ids: Annotated[list[Identifier], Field(min_length=1)]
    symbol_ids: Annotated[list[Identifier], Field(min_length=1)]
    data_requirement_ids: Annotated[list[Identifier], Field(min_length=1)]
    declared_conservation_laws: list[str] = Field(default_factory=list)
    declared_limit_cases: Annotated[list[str], Field(min_length=2)]
    identifiability_risks: Annotated[list[str], Field(min_length=1)]
    model_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_model(self) -> "ModelSpecV50":
        for field_name in (
            "assumption_ids",
            "symbol_ids",
            "data_requirement_ids",
        ):
            _assert_unique(list(getattr(self, field_name)), field_name)
        if self.model_hash and self.model_hash != self.content_hash():
            raise ValueError("model_hash does not match model spec")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "model_hash")

    @classmethod
    def seal(cls, **data: object) -> "ModelSpecV50":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"model_hash"}),
            model_hash=draft.content_hash(),
        )


class ValidationObligationV50(StrictModel):
    check_id: Identifier
    stage: Literal["S3", "S4"]
    level: CheckLevel
    evidence_class: EvidenceClass
    applicability: Applicability = "applicable"
    applicability_rule: Annotated[str, Field(min_length=5)]
    required: bool = True

    @model_validator(mode="after")
    def validate_obligation(self) -> "ValidationObligationV50":
        if self.stage == "S3" and self.level not in {"L0", "L1", "L2"}:
            raise ValueError("S3 obligations must be L0-L2")
        if self.stage == "S4" and self.level not in {"L1", "L2", "L3", "L4"}:
            raise ValueError("S4 obligations must be L1-L4")
        if self.level in {"L0", "L1", "L2", "L3", "L4"} and (
            self.evidence_class != "scientific_computation"
        ):
            raise ValueError("L0-L4 obligations require scientific computation")
        return self


class ValidationPlanV50(StrictModel):
    schema_version: Literal["5.0"] = "5.0"
    obligations: Annotated[list[ValidationObligationV50], Field(min_length=1)]
    frozen_by: Literal["verifier", "human"]
    frozen_at: datetime
    ensemble_disagreement_threshold: float = Field(
        default=0.25, gt=0, allow_inf_nan=False
    )
    unsupported_support_action: Literal["return_to_data_acquisition"] = (
        "return_to_data_acquisition"
    )
    plan_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_plan(self) -> "ValidationPlanV50":
        _assert_timezone(self.frozen_at, "frozen_at")
        ids = [item.check_id for item in self.obligations]
        if len(ids) != len(set(ids)):
            raise ValueError("validation check IDs must be unique")
        required_coverage = {
            (item.stage, item.level)
            for item in self.obligations
            if item.required and item.applicability == "applicable"
        }
        missing = {
            ("S3", "L0"),
            ("S3", "L1"),
            ("S3", "L2"),
            ("S4", "L3"),
            ("S4", "L4"),
        } - required_coverage
        if missing:
            raise ValueError(
                "validation plan lacks required applicable level coverage: "
                + ", ".join(f"{stage}/{level}" for stage, level in sorted(missing))
            )
        if self.plan_hash and self.plan_hash != self.content_hash():
            raise ValueError("plan_hash does not match validation plan")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "plan_hash")

    @classmethod
    def seal(cls, **data: object) -> "ValidationPlanV50":
        data.setdefault("frozen_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"plan_hash"}),
            plan_hash=draft.content_hash(),
        )


class DataLedgerEntryV50(StrictModel):
    data_item_id: Identifier
    semantic_name: Annotated[str, Field(min_length=3)]
    units: Annotated[str, Field(min_length=1)]
    source_kind: Literal[
        "official", "local", "user", "literature", "synthetic", "estimated"
    ]
    source_ref: Annotated[str, Field(min_length=1)]
    raw_relative_path: Annotated[str, Field(min_length=1)] | None = None
    accessed_at: datetime | None = None
    license_status: Annotated[str, Field(min_length=2)]
    request_hash: Sha256 | None = None
    raw_response_hash: Sha256 | None = None
    transform_script_relative_path: Annotated[str, Field(min_length=1)]
    transform_script_hash: Sha256
    transform_params: dict[str, Any] = Field(default_factory=dict)
    transform_params_hash: Sha256
    processed_artifact_hash: Sha256
    quality_flags: list[str] = Field(default_factory=list)
    unavailability_reason: str | None = None
    sensitivity_requirement_id: Identifier | None = None

    @model_validator(mode="after")
    def validate_entry(self) -> "DataLedgerEntryV50":
        if self.accessed_at is not None:
            _assert_timezone(self.accessed_at, "accessed_at")
        if self.source_kind in {"synthetic", "estimated"}:
            if not self.unavailability_reason or not self.sensitivity_requirement_id:
                raise ValueError(
                    "synthetic/estimated data requires unavailability_reason and "
                    "sensitivity_requirement_id"
                )
        elif self.raw_response_hash is None or self.raw_relative_path is None:
            raise ValueError(
                "non-synthetic data requires raw_response_hash and raw_relative_path"
            )
        if self.raw_relative_path is not None and not (
            self.raw_relative_path.startswith("data/raw/")
        ):
            raise ValueError("raw_relative_path must be under data/raw/")
        if not self.transform_script_relative_path.startswith("src/"):
            raise ValueError(
                "transform_script_relative_path must be under src/"
            )
        if self.transform_params_hash != sha256_value(self.transform_params):
            raise ValueError("transform_params_hash does not match parameters")
        return self


class DataLedgerV50(StrictModel):
    schema_version: Literal["5.0"] = "5.0"
    entries: Annotated[list[DataLedgerEntryV50], Field(min_length=1)]
    raw_baseline_tree_hash: Sha256
    ledger_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_ledger(self) -> "DataLedgerV50":
        ids = [item.data_item_id for item in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("data ledger IDs must be unique")
        if self.ledger_hash and self.ledger_hash != self.content_hash():
            raise ValueError("ledger_hash does not match data ledger")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "ledger_hash")

    @classmethod
    def seal(cls, **data: object) -> "DataLedgerV50":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"ledger_hash"}),
            ledger_hash=draft.content_hash(),
        )


class ProcessedArtifactV50(StrictModel):
    data_item_id: Identifier
    relative_path: Annotated[str, Field(min_length=1)]
    artifact_hash: Sha256

    @model_validator(mode="after")
    def validate_path(self) -> "ProcessedArtifactV50":
        if not self.relative_path.startswith("data/processed/"):
            raise ValueError(
                "processed artifact relative_path must be under data/processed/"
            )
        return self


class ProcessedManifestV50(StrictModel):
    schema_version: Literal["5.0"] = "5.0"
    raw_baseline_tree_hash: Sha256
    artifacts: Annotated[list[ProcessedArtifactV50], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_artifacts(self) -> "ProcessedManifestV50":
        ids = [item.data_item_id for item in self.artifacts]
        if len(ids) != len(set(ids)):
            raise ValueError("processed data item IDs must be unique")
        return self


class CodeManifestV50(StrictModel):
    schema_version: Literal["5.0"] = "5.0"
    source_tree_hash: Sha256
    environment_ref: Annotated[str, Field(min_length=1)]
    environment_hash: Sha256
    replay_command: Annotated[str, Field(min_length=3)]
    replay_receipt_ref: Annotated[str, Field(min_length=1)]
    replay_receipt_hash: Sha256
    random_seed: int
    tolerance_policy: Annotated[str, Field(min_length=3)]
    fermi_estimate_ref: Annotated[str, Field(min_length=1)]
    fermi_estimate_hash: Sha256
    toy_oracle_refs: Annotated[list[str], Field(min_length=1)]
    toy_oracle_hashes: dict[str, Sha256]

    @model_validator(mode="after")
    def validate_references(self) -> "CodeManifestV50":
        _assert_unique(self.toy_oracle_refs, "toy_oracle_refs")
        if not self.environment_ref.startswith("results/"):
            raise ValueError("environment_ref must be under results/")
        if not self.replay_receipt_ref.startswith("checks/"):
            raise ValueError("replay_receipt_ref must be under checks/")
        if not self.fermi_estimate_ref.startswith("results/"):
            raise ValueError("fermi_estimate_ref must be under results/")
        if any(
            not relative_path.startswith("checks/")
            for relative_path in self.toy_oracle_refs
        ):
            raise ValueError("toy oracle refs must be under checks/")
        if set(self.toy_oracle_hashes) != set(self.toy_oracle_refs):
            raise ValueError(
                "toy_oracle_hashes keys must exactly match toy_oracle_refs"
            )
        return self


class ResultRecordV50(StrictModel):
    result_id: Identifier
    relative_path: Annotated[str, Field(min_length=1)]
    artifact_hash: Sha256
    value: float | None = Field(default=None, allow_inf_nan=False)
    interval_low: float | None = Field(default=None, allow_inf_nan=False)
    interval_high: float | None = Field(default=None, allow_inf_nan=False)
    units: Annotated[str, Field(min_length=1)] = "unitless"

    @model_validator(mode="after")
    def validate_interval(self) -> "ResultRecordV50":
        if not self.relative_path.startswith("results/"):
            raise ValueError("result relative_path must be under results/")
        if (self.interval_low is None) != (self.interval_high is None):
            raise ValueError("result intervals require both bounds")
        if (
            self.interval_low is not None
            and self.interval_high is not None
            and self.interval_low > self.interval_high
        ):
            raise ValueError("result interval is reversed")
        return self


class ResultIndexV50(StrictModel):
    schema_version: Literal["5.0"] = "5.0"
    records: Annotated[list[ResultRecordV50], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_records(self) -> "ResultIndexV50":
        ids = [item.result_id for item in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("result IDs must be unique")
        return self


class UQClaimV50(StrictModel):
    claim_id: Identifier
    result_id: Identifier
    interval_result_id: Identifier
    support_status: Literal["in_support", "extrapolation", "unknown"]
    ensemble_disagreement: float = Field(ge=0, allow_inf_nan=False)


class UQSummaryV50(StrictModel):
    schema_version: Literal["5.0"] = "5.0"
    claims: Annotated[list[UQClaimV50], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_claims(self) -> "UQSummaryV50":
        claim_ids = [item.claim_id for item in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("UQ claim IDs must be unique")
        return self


class DecisionAssertionV50(StrictModel):
    assertion_id: Identifier
    statement: Annotated[str, Field(min_length=5)]
    result_ids: Annotated[list[Identifier], Field(min_length=1)]
    uq_claim_ids: Annotated[list[Identifier], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_references(self) -> "DecisionAssertionV50":
        _assert_unique(self.result_ids, "result_ids")
        _assert_unique(self.uq_claim_ids, "uq_claim_ids")
        return self


class DecisionDossierV50(StrictModel):
    schema_version: Literal["5.0"] = "5.0"
    assertions: Annotated[list[DecisionAssertionV50], Field(min_length=1)]
    high_disagreement_detected: bool
    next_action: Literal[
        "draft_report_only",
        "register_prediction",
        "return_to_data_acquisition",
        "request_human_decision",
    ]
    prediction_seal_hash: Sha256 | None = None
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_decision(self) -> "DecisionDossierV50":
        if self.high_disagreement_detected and (
            self.next_action != "return_to_data_acquisition"
        ):
            raise ValueError(
                "high ensemble disagreement must return to data acquisition"
            )
        if self.next_action == "register_prediction" and not (
            self.prediction_seal_hash
        ):
            raise ValueError(
                "register_prediction requires an authenticated prediction seal hash"
            )
        assertion_ids = [item.assertion_id for item in self.assertions]
        if len(assertion_ids) != len(set(assertion_ids)):
            raise ValueError("decision assertion IDs must be unique")
        return self


class FileBindingV50(StrictModel):
    relative_path: Annotated[str, Field(min_length=1)]
    sha256: Sha256
    size_bytes: Annotated[int, Field(ge=0)]
    snapshot_artifact_hash: Sha256


class RawDataBaselineV50(StrictModel):
    """Harness-frozen raw input tree for one S2 attempt."""

    schema_version: Literal["5.0"] = "5.0"
    workspace_spec_hash: Sha256
    s1_gate_hash: Sha256
    s2_attempt: Annotated[int, Field(ge=1)]
    raw_tree_hash: Sha256
    files: list[FileBindingV50] = Field(default_factory=list)
    frozen_at: datetime
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    baseline_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_baseline(self) -> "RawDataBaselineV50":
        _assert_timezone(self.frozen_at, "frozen_at")
        paths = [item.relative_path for item in self.files]
        if paths != sorted(set(paths)):
            raise ValueError("raw baseline paths must be sorted and unique")
        if self.authority_auth_tag and not self.baseline_hash:
            raise ValueError("authenticated raw baseline requires baseline_hash")
        if self.baseline_hash and (
            _hash_without(self, "baseline_hash") != self.baseline_hash
        ):
            raise ValueError("baseline_hash does not match raw baseline")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "authority_auth_tag", "baseline_hash")


class StageArtifactManifestV50(StrictModel):
    schema_version: Literal["5.0"] = "5.0"
    workspace_spec_hash: Sha256
    stage: StageId
    attempt: Annotated[int, Field(ge=1)]
    predecessor_gate_hash: Sha256 | None = None
    files: Annotated[list[FileBindingV50], Field(min_length=1)]
    captured_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "StageArtifactManifestV50":
        _assert_timezone(self.captured_at, "captured_at")
        paths = [item.relative_path for item in self.files]
        if paths != sorted(set(paths)):
            raise ValueError("manifest file paths must be sorted and unique")
        if self.stage != "S0" and self.predecessor_gate_hash is None:
            raise ValueError("non-S0 manifests require a predecessor gate")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match stage manifest")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    @classmethod
    def seal(cls, **data: object) -> "StageArtifactManifestV50":
        data.setdefault("captured_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


class AdapterExecutionReceiptV50(StrictModel):
    """Authenticated proof of how one frozen scientific obligation was run."""

    schema_version: Literal["5.0"] = "5.0"
    execution_id: Identifier
    check_id: Identifier
    stage: StageId
    level: Literal["L0", "L1", "L2", "L3", "L4"]
    input_manifest_hash: Sha256
    protocol_hash: Sha256
    adapter_id: Identifier
    adapter_version: Annotated[str, Field(min_length=1)]
    adapter_code_hash: Sha256
    applicability: Applicability
    status: CheckStatus
    execution_mode: Literal[
        "adapter_run",
        "adapter_exception",
        "adapter_missing",
        "frozen_not_applicable",
    ]
    adapter_invoked: bool
    scientific_computation_performed: bool
    evidence_refs: list[Sha256] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_execution(self) -> "AdapterExecutionReceiptV50":
        _assert_timezone(self.started_at, "started_at")
        _assert_timezone(self.finished_at, "finished_at")
        if self.finished_at < self.started_at:
            raise ValueError("adapter execution finished before it started")
        _assert_unique(self.evidence_refs, "evidence_refs")
        if self.execution_mode in {"adapter_run", "adapter_exception"}:
            if not self.adapter_invoked:
                raise ValueError("adapter execution mode requires invocation")
        elif self.adapter_invoked:
            raise ValueError("missing/N/A adapter cannot be marked invoked")
        if self.execution_mode != "adapter_run" and (
            self.scientific_computation_performed
        ):
            raise ValueError(
                "only an adapter run may claim scientific computation"
            )
        if self.status == "PASS":
            if (
                self.applicability != "applicable"
                or self.execution_mode != "adapter_run"
                or not self.adapter_invoked
                or not self.scientific_computation_performed
                or not self.evidence_refs
            ):
                raise ValueError(
                    "scientific PASS requires an invoked adapter run and evidence"
                )
        if self.status == "NOT_RUN" and self.execution_mode != "adapter_missing":
            raise ValueError("NOT_RUN requires adapter_missing")
        if self.status == "NOT_APPLICABLE":
            if (
                self.applicability != "not_applicable"
                or self.execution_mode != "frozen_not_applicable"
            ):
                raise ValueError(
                    "NOT_APPLICABLE requires frozen_not_applicable"
                )
        if self.status == "ERROR" and self.execution_mode != "adapter_exception":
            raise ValueError("ERROR requires adapter_exception")
        if self.authority_auth_tag and not self.receipt_hash:
            raise ValueError(
                "authenticated adapter execution requires receipt_hash"
            )
        if self.receipt_hash and (
            _hash_without(self, "receipt_hash") != self.receipt_hash
        ):
            raise ValueError(
                "receipt_hash does not match adapter execution receipt"
            )
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "authority_auth_tag", "receipt_hash")


class CheckResultV50(StrictModel):
    """Authenticated checker evidence bound to one exact stage manifest."""

    schema_version: Literal["5.0"] = "5.0"
    check_id: Identifier
    stage: StageId
    level: CheckLevel
    evidence_class: EvidenceClass
    applicability: Applicability
    status: CheckStatus
    reason_code: Identifier
    subject_hashes: list[Sha256] = Field(default_factory=list)
    input_manifest_hash: Sha256
    protocol_hash: Sha256
    adapter_id: Identifier
    adapter_version: Annotated[str, Field(min_length=1)]
    adapter_code_hash: Sha256
    adapter_execution_receipt_hash: Sha256 | None = None
    thresholds: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[Sha256] = Field(default_factory=list)
    scope: EvidenceScope
    executed_by: Literal["harness", "verifier"]
    started_at: datetime
    finished_at: datetime
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    result_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_result(self) -> "CheckResultV50":
        _assert_timezone(self.started_at, "started_at")
        _assert_timezone(self.finished_at, "finished_at")
        if self.finished_at < self.started_at:
            raise ValueError("check finished before it started")
        _assert_unique(self.subject_hashes, "subject_hashes")
        _assert_unique(self.evidence_refs, "evidence_refs")
        if self.status == "PASS" and self.applicability != "applicable":
            raise ValueError("PASS requires applicable")
        if self.status == "NOT_APPLICABLE":
            if self.applicability != "not_applicable":
                raise ValueError("NOT_APPLICABLE requires frozen not_applicable")
        elif self.applicability == "not_applicable":
            raise ValueError("not_applicable checks must use NOT_APPLICABLE")
        if self.status == "PASS" and self.level in {"L0", "L1", "L2", "L3", "L4"}:
            if self.evidence_class != "scientific_computation":
                raise ValueError(
                    "workflow presence/integrity cannot emit a scientific PASS"
                )
            if not self.evidence_refs:
                raise ValueError("scientific PASS requires computation evidence")
        if self.level in {"L0", "L1", "L2", "L3", "L4"}:
            if self.adapter_execution_receipt_hash is None:
                raise ValueError(
                    "L0-L4 results require an adapter execution receipt"
                )
        elif self.adapter_execution_receipt_hash is not None:
            raise ValueError("L5 results cannot cite a scientific adapter receipt")
        if self.authority_auth_tag and not self.result_hash:
            raise ValueError("authenticated check result requires result_hash")
        if self.result_hash:
            expected = _hash_without(self, "result_hash")
            if expected != self.result_hash:
                raise ValueError("result_hash does not match check result")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "authority_auth_tag", "result_hash")


class IndependentReviewReceiptV50(StrictModel):
    schema_version: Literal["5.0"] = "5.0"
    review_id: Identifier
    stage: StageId
    role: ReviewRole
    input_manifest_hash: Sha256
    producer_run_id: Identifier
    reviewer_run_id: Identifier
    producer_context_id: Identifier
    reviewer_context_id: Identifier
    prompt_hash: Sha256
    output_schema_hash: Sha256
    allowed_input_hashes: list[Sha256]
    context_isolation_attested: bool
    transport_trace_hash: Sha256
    output_artifact_hash: Sha256
    verdict: ReviewVerdict
    finding_ids: list[Identifier] = Field(default_factory=list)
    issued_by: Literal["verifier", "human"]
    issued_at: datetime
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_review(self) -> "IndependentReviewReceiptV50":
        _assert_timezone(self.issued_at, "issued_at")
        _assert_unique(self.allowed_input_hashes, "allowed_input_hashes")
        _assert_unique(self.finding_ids, "finding_ids")
        if self.producer_run_id == self.reviewer_run_id:
            raise ValueError("reviewer run must be independent from producer run")
        if self.producer_context_id == self.reviewer_context_id:
            raise ValueError("reviewer context must be fresh")
        if not self.context_isolation_attested:
            raise ValueError("independent review requires context isolation")
        if self.authority_auth_tag and not self.receipt_hash:
            raise ValueError("authenticated review requires receipt_hash")
        if self.receipt_hash:
            expected = _hash_without(self, "receipt_hash")
            if expected != self.receipt_hash:
                raise ValueError("receipt_hash does not match review receipt")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "authority_auth_tag", "receipt_hash")


class GateCertificateV50(StrictModel):
    """A workflow certificate; explicitly not a scientific promotion."""

    schema_version: Literal["5.0"] = "5.0"
    workspace_spec_hash: Sha256
    stage: StageId
    attempt: Annotated[int, Field(ge=1)]
    policy_hash: Sha256
    manifest: StageArtifactManifestV50
    upstream_gate_hashes: list[Sha256]
    check_result_hashes: list[Sha256]
    reviewer_receipt_hashes: list[Sha256]
    graph_gate_node_hash: Sha256
    graph_snapshot_before_hash: Sha256
    decision: Literal["OPEN"] = "OPEN"
    evaluator_epoch: Identifier
    authority: Literal["verifier", "human"]
    authority_key_id: Identifier
    issued_at: datetime
    scientific_qualification_granted: Literal[False] = False
    private_scientific_gate_passed: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    private_acceptance_data_exposed: Literal[False] = False
    authority_auth_tag: Sha256 | None = None
    certificate_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_certificate(self) -> "GateCertificateV50":
        _assert_timezone(self.issued_at, "issued_at")
        self.manifest.content_hash()
        if self.manifest.stage != self.stage or self.manifest.attempt != self.attempt:
            raise ValueError("gate certificate and manifest stage/attempt differ")
        _assert_unique(self.upstream_gate_hashes, "upstream_gate_hashes")
        _assert_unique(self.check_result_hashes, "check_result_hashes")
        _assert_unique(self.reviewer_receipt_hashes, "reviewer_receipt_hashes")
        if self.authority_auth_tag and not self.certificate_hash:
            raise ValueError("authenticated certificate requires certificate_hash")
        if self.certificate_hash:
            expected = _hash_without(self, "certificate_hash")
            if expected != self.certificate_hash:
                raise ValueError(
                    "certificate_hash does not match gate certificate"
                )
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "authority_auth_tag", "certificate_hash")


class GateEvaluationV50(StrictModel):
    schema_version: Literal["5.0"] = "5.0"
    stage: StageId
    manifest_hash: Sha256
    decision: GateDecision
    reasons: list[str]
    accepted_check_hashes: list[Sha256] = Field(default_factory=list)
    accepted_review_hashes: list[Sha256] = Field(default_factory=list)
    certificate_hash: Sha256 | None = None
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False


class PredictionSealV50(StrictModel):
    """Task-side binding to a snapshot held outside the task workspace."""

    schema_version: Literal["5.0"] = "5.0"
    workspace_spec_hash: Sha256
    s4_gate_hash: Sha256
    task_id: Identifier
    training_snapshot_hash: Sha256
    candidate_hash: Sha256
    prediction_artifact_hash: Sha256
    external_registration_hash: Sha256
    external_snapshot_hash: Sha256
    holdout_commitment_hash: Sha256
    registered_at: datetime
    authority_key_id: Identifier
    private_holdout_accessed_before_registration: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    authority_auth_tag: Sha256 | None = None
    seal_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_prediction(self) -> "PredictionSealV50":
        _assert_timezone(self.registered_at, "registered_at")
        if self.authority_auth_tag and not self.seal_hash:
            raise ValueError("authenticated prediction seal requires seal_hash")
        if self.seal_hash and _hash_without(self, "seal_hash") != self.seal_hash:
            raise ValueError("seal_hash does not match prediction seal")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "authority_auth_tag", "seal_hash")


class WorkflowStatusV50(StrictModel):
    schema_version: Literal["5.0"] = "5.0"
    workspace_id: Identifier
    graph_id: Identifier
    graph_verified: bool
    stage_statuses: dict[StageId, str]
    current_gate_hashes: dict[StageId, Sha256]
    stale_gate_hashes: dict[StageId, Sha256]
    frontier_stages: list[StageId]
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    claim_scope: Literal["workflow_control_only"] = "workflow_control_only"


__all__ = [
    "Actor",
    "AdapterExecutionReceiptV50",
    "Applicability",
    "AssumptionRecordV50",
    "AssumptionSetV50",
    "AuthorityGenesisV50",
    "CandidateFormalizationV50",
    "CandidateSetV50",
    "CheckLevel",
    "CheckResultV50",
    "CheckStatus",
    "CodeManifestV50",
    "DataLedgerEntryV50",
    "DataLedgerV50",
    "DecisionAssertionV50",
    "DecisionFunctionCanaryV50",
    "DecisionFunctionSpecV50",
    "DecisionDossierV50",
    "EvidenceClass",
    "EvidenceScope",
    "ExecutionRole",
    "FileBindingV50",
    "GateCertificateV50",
    "GateDecision",
    "GateEvaluationV50",
    "IndependentReviewReceiptV50",
    "ModelSpecV50",
    "PredictionSealV50",
    "ProcessedArtifactV50",
    "ProcessedManifestV50",
    "RegimeDiagnosisV50",
    "RawDataBaselineV50",
    "RoleExecutionReceiptV50",
    "ResultIndexV50",
    "ResultRecordV50",
    "ReviewRole",
    "ReviewVerdict",
    "StageArtifactManifestV50",
    "StageId",
    "SymbolRecordV50",
    "SymbolTableV50",
    "TaskWorkspaceSpecV50",
    "UQClaimV50",
    "UQSummaryV50",
    "ValidationObligationV50",
    "ValidationPlanV50",
    "WorkflowProfileV50",
    "WorkflowStatusV50",
]
