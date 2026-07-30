"""Additive V6.5 control plane for the V6.3 qualification protocol.

V6.3 remains the scientific protocol and verifier.  This module adds the
missing single-writer ingress boundary around every state-changing V6.3
operation.  It never signs an external-role envelope and never upgrades a
V6.3 protocol result into scientific qualification or real-world authority.

Each attempted operation is represented by authority-authenticated,
append-only events:

``intent -> (completion | failure -> RESUME_EXACT -> ... | ABORT_ATTEMPT)``

The graph writer lock covers intent, action, and terminal receipt.  Recoverable
failures remain visible, exact retries are content-bound, and V6.3 authority
artifacts committed outside an operation segment make the V6.5 projection
``INCONSISTENT``.  A hard process death after a V6.3 write is intentionally
not auto-recovered: without a durable V6.5 terminal receipt, provenance is
ambiguous and the attempt remains human-reconciliation only.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
from pathlib import Path
import platform
import re
import sys
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5.stage_workspace import StageWorkspaceV50
from fma.v5.workspace_schemas import PredictionSealV50

from . import external_qualification as qualification
from .external_qualification import (
    CurrentModelPredictionBindingV63,
    ExternalAggregateEvaluationV63,
    ExternalCustodyAdmissionV63,
    ExternalEvaluationConsumptionV63,
    ExternalEvaluationReservationV63,
    ExternalEvidenceCustodyV63,
    ExternalPredictionRegistrationV63,
    ExternalPredictivePromotionV63,
    ExternalPredictiveQualificationReceiptV63,
    PredictiveExternalQualificationContractV63,
)
from .external_qualification_coordinator import (
    CUSTODY_ADMISSION_KIND_V63,
    CUSTODY_KIND_V63,
    DISPATCH_PACKET_KIND_V63,
    EVALUATION_KIND_V63,
    FORECAST_INPUT_KIND_V63,
    OPERATION_INTENT_KIND_V63,
    OPERATION_RECEIPT_KIND_V63,
    PREDICTION_BINDING_KIND_V63,
    PREDICTION_SEAL_KIND_V50,
    PREDICTION_VECTOR_KIND_V63,
    PROMOTION_KIND_V63,
    QUALIFICATION_KIND_V63,
    REGISTRATION_KIND_V63,
    RESERVATION_KIND_V63,
    CONTRACT_KIND_V63,
    ExternalQualificationCoordinatorV63,
    ExternalQualificationOperationReceiptV63,
    ExternalQualificationPhaseV63,
    ExternalQualificationStateV63,
    _claim_v65_mutation_coordinator_constructor,
    _artifact_hash_for_model,
    _typed_one,
    project_external_qualification_state_v63,
)

_v63_mutation_constructor = (
    _claim_v65_mutation_coordinator_constructor()
)
del _claim_v65_mutation_coordinator_constructor


ExternalControlOperationV65 = Literal[
    "ingest_custody",
    "run_prediction",
    "ingest_registration",
    "reserve_evaluation",
    "ingest_evaluation",
    "ingest_promotion",
]
ExternalControlActorV65 = Literal["operator", "server"]
ExternalControlCapabilityV65 = Literal[
    "activate",
    "ingest_custody",
    "run_prediction",
    "ingest_registration",
    "reserve_evaluation",
    "ingest_evaluation",
    "ingest_promotion",
    "abort_attempt",
]
ExternalControlFailureClassV65 = Literal["RETRYABLE", "HUMAN_REQUIRED"]
ExternalControlResolutionDecisionV65 = Literal[
    "RESUME_EXACT", "ABORT_ATTEMPT"
]
ExternalControlStatusV65 = Literal[
    "ACTIVE",
    "PENDING_FAILURE",
    "ABORTED",
    "INCONSISTENT",
    "LEGACY_UNMANAGED",
]

CONTROL_PRINCIPAL_KIND_V65 = "external_control_principal_v65"
CONTROL_ACTIVATION_KIND_V65 = "external_control_activation_v65"
CONTROL_INTENT_KIND_V65 = "external_control_intent_v65"
CONTROL_COMPLETION_KIND_V65 = "external_control_completion_v65"
CONTROL_FAILURE_KIND_V65 = "external_control_failure_v65"
CONTROL_RESOLUTION_KIND_V65 = "external_control_resolution_v65"

_V63_PROTECTED_KINDS = {
    CONTRACT_KIND_V63,
    FORECAST_INPUT_KIND_V63,
    CUSTODY_KIND_V63,
    CUSTODY_ADMISSION_KIND_V63,
    PREDICTION_VECTOR_KIND_V63,
    PREDICTION_BINDING_KIND_V63,
    REGISTRATION_KIND_V63,
    PREDICTION_SEAL_KIND_V50,
    RESERVATION_KIND_V63,
    DISPATCH_PACKET_KIND_V63,
    EVALUATION_KIND_V63,
    "external_evaluation_consumption_v63",
    PROMOTION_KIND_V63,
    QUALIFICATION_KIND_V63,
    OPERATION_INTENT_KIND_V63,
    OPERATION_RECEIPT_KIND_V63,
}

_ALL_CONTROL_CAPABILITIES: tuple[ExternalControlCapabilityV65, ...] = (
    "abort_attempt",
    "activate",
    "ingest_custody",
    "ingest_evaluation",
    "ingest_promotion",
    "ingest_registration",
    "reserve_evaluation",
    "run_prediction",
)

_START_PHASES: dict[ExternalControlOperationV65, set[str]] = {
    "ingest_custody": {"INPUT_FROZEN"},
    "run_prediction": {"CUSTODY_VERIFIED"},
    "ingest_registration": {"PREDICTION_BOUND"},
    "reserve_evaluation": {"PREDICTION_REGISTERED"},
    "ingest_evaluation": {"EVALUATION_RESERVED"},
    "ingest_promotion": {"AWAITING_PROMOTION"},
}
_RESULT_PHASES: dict[ExternalControlOperationV65, set[str]] = {
    "ingest_custody": {"CUSTODY_VERIFIED", "CUSTODY_REJECTED"},
    "run_prediction": {"PREDICTION_BOUND"},
    "ingest_registration": {"PREDICTION_REGISTERED"},
    "reserve_evaluation": {"EVALUATION_RESERVED"},
    "ingest_evaluation": {"AWAITING_PROMOTION"},
    "ingest_promotion": {"EXTERNALLY_QUALIFIED", "REJECTED"},
}
_RECOVERY_PHASES: dict[ExternalControlOperationV65, set[str]] = {
    "ingest_custody": {"CUSTODY_COMMITTED"},
    "run_prediction": set(),
    "ingest_registration": {"REGISTRATION_COMMITTED"},
    "reserve_evaluation": set(),
    "ingest_evaluation": {"EVALUATION_COMMITTED"},
    "ingest_promotion": {"PROMOTION_COMMITTED"},
}
_NEXT_OPERATION: dict[str, list[str]] = {
    "INPUT_FROZEN": ["ingest_custody"],
    "CUSTODY_VERIFIED": ["run_prediction"],
    "PREDICTION_BOUND": ["ingest_registration"],
    "PREDICTION_REGISTERED": ["reserve_evaluation"],
    "EVALUATION_RESERVED": ["ingest_evaluation"],
    "AWAITING_PROMOTION": ["ingest_promotion"],
}


class ExternalControlPlaneErrorV65(RuntimeError):
    """The V6.5 control ledger or requested transition failed closed."""


class ExternalControlOperationFailedV65(ExternalControlPlaneErrorV65):
    """An action failed after a durable failure receipt was committed."""

    def __init__(
        self,
        *,
        operation_id: str,
        failure_class: ExternalControlFailureClassV65,
        failure_receipt_hash: str,
    ) -> None:
        self.operation_id = operation_id
        self.failure_class = failure_class
        self.failure_receipt_hash = failure_receipt_hash
        super().__init__(
            f"{operation_id} failed as {failure_class}; "
            f"receipt={failure_receipt_hash}"
        )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _assert_aware(value: datetime, field_name: str) -> None:
    if value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


class ExternalControlRuntimeIdentityV65(StrictModel):
    """Deterministic identity of the code and dependencies executing a request."""

    schema_version: Literal["6.5-external-control-runtime-identity"] = (
        "6.5-external-control-runtime-identity"
    )
    source_sha256_by_module: dict[Identifier, Sha256]
    dependency_versions: dict[Identifier, str]
    python_implementation: Identifier
    python_version: str
    python_cache_tag: str
    python_executable_sha256: Sha256 | None
    runtime_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_runtime(self) -> "ExternalControlRuntimeIdentityV65":
        if not self.source_sha256_by_module:
            raise ValueError("V6.5 runtime identity lacks source hashes")
        if not self.dependency_versions:
            raise ValueError("V6.5 runtime identity lacks dependency versions")
        if self.runtime_hash and self.runtime_hash != self.content_hash():
            raise ValueError("V6.5 runtime identity hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "runtime_hash")

    def assert_sealed(self) -> None:
        if not self.runtime_hash or self.runtime_hash != self.content_hash():
            raise ValueError("V6.5 runtime identity is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ExternalControlRuntimeIdentityV65":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"runtime_hash"})
        payload["runtime_hash"] = draft.content_hash()
        return cls(**payload)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_external_control_runtime_identity_v65(
) -> ExternalControlRuntimeIdentityV65:
    """Capture the installed implementation used for exact-retry binding."""

    module_paths = {
        "external_control_plane": Path(__file__).resolve(),
        "external_qualification": Path(qualification.__file__).resolve(),
        "external_qualification_coordinator": (
            Path(__file__).with_name(
                "external_qualification_coordinator.py"
            ).resolve()
        ),
        "external_prediction_runtime": (
            Path(__file__).with_name("external_prediction_runtime.py").resolve()
        ),
        "stage_workspace": (
            Path(__file__).parents[1] / "v5" / "stage_workspace.py"
        ).resolve(),
    }
    dependency_versions: dict[str, str] = {}
    for distribution in ("cryptography", "numpy", "pydantic", "scipy"):
        try:
            dependency_versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            dependency_versions[distribution] = "NOT_INSTALLED"
    executable = Path(sys.executable).resolve()
    executable_hash = (
        _file_sha256(executable) if executable.is_file() else None
    )
    return ExternalControlRuntimeIdentityV65.seal(
        source_sha256_by_module={
            name: _file_sha256(path)
            for name, path in sorted(module_paths.items())
        },
        dependency_versions=dict(sorted(dependency_versions.items())),
        python_implementation=platform.python_implementation().lower(),
        python_version=platform.python_version(),
        python_cache_tag=sys.implementation.cache_tag or "none",
        python_executable_sha256=executable_hash,
    )


class ExternalControlPrincipalV65(StrictModel):
    """Workspace-authority authenticated mutation capability."""

    schema_version: Literal["6.5-external-control-principal"] = (
        "6.5-external-control-principal"
    )
    principal_id: Identifier
    actor_type: ExternalControlActorV65
    workspace_spec_hash: Sha256
    workspace_authority_genesis_hash: Sha256
    qualification_id: Identifier
    allowed_operations: Annotated[
        list[ExternalControlCapabilityV65], Field(min_length=1)
    ]
    issued_at: datetime
    expires_at: datetime
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    principal_hash: Sha256 | None = None
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_principal(self) -> "ExternalControlPrincipalV65":
        _assert_aware(self.issued_at, "issued_at")
        _assert_aware(self.expires_at, "expires_at")
        if self.expires_at <= self.issued_at:
            raise ValueError("V6.5 principal capability has an empty lifetime")
        if self.allowed_operations != sorted(set(self.allowed_operations)):
            raise ValueError(
                "V6.5 principal operations must be sorted and unique"
            )
        if self.authority_auth_tag and not self.principal_hash:
            raise ValueError("authenticated V6.5 principal lacks a hash")
        if self.principal_hash and self.principal_hash != self.content_hash():
            raise ValueError("V6.5 principal hash differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(
            self, "authority_auth_tag", "principal_hash"
        )

    def content_hash(self) -> str:
        return _hash_without(self, "principal_hash")

    def assert_sealed(self) -> None:
        if (
            not self.authority_auth_tag
            or not self.principal_hash
            or self.principal_hash != self.content_hash()
        ):
            raise ValueError("V6.5 principal is not sealed")


class ExternalControlActivationV65(StrictModel):
    """Exact graph boundary after which V6.5 owns all V6.3 mutations."""

    schema_version: Literal["6.5-external-control-activation"] = (
        "6.5-external-control-activation"
    )
    activation_id: Identifier
    qualification_id: Identifier
    v63_state_hash: Sha256
    v63_phase: ExternalQualificationPhaseV63
    pre_graph_tip: Sha256
    principal_id: Identifier
    principal_hash: Sha256
    runtime_identity_hash: Sha256
    activated_at: datetime
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    activation_hash: Sha256 | None = None
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_activation(self) -> "ExternalControlActivationV65":
        _assert_aware(self.activated_at, "activated_at")
        if self.authority_auth_tag and not self.activation_hash:
            raise ValueError("authenticated V6.5 activation lacks a hash")
        if self.activation_hash and self.activation_hash != self.content_hash():
            raise ValueError("V6.5 activation hash differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(
            self, "authority_auth_tag", "activation_hash"
        )

    def content_hash(self) -> str:
        return _hash_without(self, "activation_hash")

    def assert_sealed(self) -> None:
        if (
            not self.authority_auth_tag
            or not self.activation_hash
            or self.activation_hash != self.content_hash()
        ):
            raise ValueError("V6.5 activation is not sealed")


class ExternalControlIntentV65(StrictModel):
    schema_version: Literal["6.5-external-control-intent"] = (
        "6.5-external-control-intent"
    )
    operation_id: Identifier
    qualification_id: Identifier
    operation_type: ExternalControlOperationV65
    request_hash: Sha256
    input_artifact_hash: Sha256
    expected_v63_state_hash: Sha256
    expected_v63_phase: ExternalQualificationPhaseV63
    pre_graph_tip: Sha256
    actor: ExternalControlActorV65
    principal_id: Identifier
    principal_hash: Sha256
    runtime_identity_hash: Sha256
    started_at: datetime
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    intent_hash: Sha256 | None = None
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_intent(self) -> "ExternalControlIntentV65":
        _assert_aware(self.started_at, "started_at")
        if self.authority_auth_tag and not self.intent_hash:
            raise ValueError("authenticated V6.5 intent lacks a hash")
        if self.intent_hash and self.intent_hash != self.content_hash():
            raise ValueError("V6.5 intent hash differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "authority_auth_tag", "intent_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "intent_hash")

    def assert_sealed(self) -> None:
        if (
            not self.authority_auth_tag
            or not self.intent_hash
            or self.intent_hash != self.content_hash()
        ):
            raise ValueError("V6.5 intent is not sealed")


class ExternalControlCompletionV65(StrictModel):
    schema_version: Literal["6.5-external-control-completion"] = (
        "6.5-external-control-completion"
    )
    operation_id: Identifier
    qualification_id: Identifier
    operation_type: ExternalControlOperationV65
    intent_hash: Sha256
    request_hash: Sha256
    input_artifact_hash: Sha256
    expected_v63_state_hash: Sha256
    expected_v63_phase: ExternalQualificationPhaseV63
    result_v63_state_hash: Sha256
    result_v63_phase: ExternalQualificationPhaseV63
    pre_graph_tip: Sha256
    post_graph_tip: Sha256
    output_artifact_hashes: dict[Identifier, Sha256]
    principal_id: Identifier
    principal_hash: Sha256
    runtime_identity_hash: Sha256
    completed_at: datetime
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    completion_hash: Sha256 | None = None
    v63_protocol_qualification_granted: bool = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_completion(self) -> "ExternalControlCompletionV65":
        _assert_aware(self.completed_at, "completed_at")
        if not self.output_artifact_hashes:
            raise ValueError("V6.5 completion lacks action outputs")
        if self.v63_protocol_qualification_granted != (
            self.result_v63_phase == "EXTERNALLY_QUALIFIED"
        ):
            raise ValueError("V6.5 protocol flag differs from V6.3 phase")
        if self.authority_auth_tag and not self.completion_hash:
            raise ValueError("authenticated V6.5 completion lacks a hash")
        if self.completion_hash and (
            self.completion_hash != self.content_hash()
        ):
            raise ValueError("V6.5 completion hash differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(
            self, "authority_auth_tag", "completion_hash"
        )

    def content_hash(self) -> str:
        return _hash_without(self, "completion_hash")

    def assert_sealed(self) -> None:
        if (
            not self.authority_auth_tag
            or not self.completion_hash
            or self.completion_hash != self.content_hash()
        ):
            raise ValueError("V6.5 completion is not sealed")


class ExternalControlFailureV65(StrictModel):
    schema_version: Literal["6.5-external-control-failure"] = (
        "6.5-external-control-failure"
    )
    operation_id: Identifier
    qualification_id: Identifier
    operation_type: ExternalControlOperationV65
    intent_hash: Sha256
    request_hash: Sha256
    input_artifact_hash: Sha256
    expected_v63_state_hash: Sha256
    expected_v63_phase: ExternalQualificationPhaseV63
    attempt_number: Annotated[int, Field(ge=1)]
    failure_class: ExternalControlFailureClassV65
    error_code: Identifier
    error_message_hash: Sha256
    observed_v63_state_hash: Sha256
    observed_v63_phase: ExternalQualificationPhaseV63
    pre_graph_tip: Sha256
    post_graph_tip: Sha256
    principal_id: Identifier
    principal_hash: Sha256
    runtime_identity_hash: Sha256
    failed_at: datetime
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    failure_hash: Sha256 | None = None
    v63_protocol_qualification_granted: bool = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_failure(self) -> "ExternalControlFailureV65":
        _assert_aware(self.failed_at, "failed_at")
        if self.v63_protocol_qualification_granted != (
            self.observed_v63_phase == "EXTERNALLY_QUALIFIED"
        ):
            raise ValueError("failure protocol flag differs from V6.3 phase")
        if self.authority_auth_tag and not self.failure_hash:
            raise ValueError("authenticated V6.5 failure lacks a hash")
        if self.failure_hash and self.failure_hash != self.content_hash():
            raise ValueError("V6.5 failure hash differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "authority_auth_tag", "failure_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "failure_hash")

    def assert_sealed(self) -> None:
        if (
            not self.authority_auth_tag
            or not self.failure_hash
            or self.failure_hash != self.content_hash()
        ):
            raise ValueError("V6.5 failure is not sealed")


class ExternalControlResolutionV65(StrictModel):
    schema_version: Literal["6.5-external-control-resolution"] = (
        "6.5-external-control-resolution"
    )
    operation_id: Identifier
    qualification_id: Identifier
    operation_type: ExternalControlOperationV65
    intent_hash: Sha256
    request_hash: Sha256
    input_artifact_hash: Sha256
    expected_v63_state_hash: Sha256
    expected_v63_phase: ExternalQualificationPhaseV63
    failure_hash: Sha256
    decision: ExternalControlResolutionDecisionV65
    actor: ExternalControlActorV65
    principal_id: Identifier
    principal_hash: Sha256
    runtime_identity_hash: Sha256
    pre_graph_tip: Sha256
    post_graph_tip: Sha256
    resolved_at: datetime
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    resolution_hash: Sha256 | None = None
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_resolution(self) -> "ExternalControlResolutionV65":
        _assert_aware(self.resolved_at, "resolved_at")
        if self.post_graph_tip != self.pre_graph_tip:
            raise ValueError(
                "resolution has no V6.3 action and must preserve graph tip"
            )
        if self.authority_auth_tag and not self.resolution_hash:
            raise ValueError("authenticated V6.5 resolution lacks a hash")
        if self.resolution_hash and (
            self.resolution_hash != self.content_hash()
        ):
            raise ValueError("V6.5 resolution hash differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(
            self, "authority_auth_tag", "resolution_hash"
        )

    def content_hash(self) -> str:
        return _hash_without(self, "resolution_hash")

    def assert_sealed(self) -> None:
        if (
            not self.authority_auth_tag
            or not self.resolution_hash
            or self.resolution_hash != self.content_hash()
        ):
            raise ValueError("V6.5 resolution is not sealed")


class ExternalControlStateV65(StrictModel):
    schema_version: Literal["6.5-external-control-state"] = (
        "6.5-external-control-state"
    )
    qualification_id: Identifier
    v63_state_hash: Sha256
    v63_phase: ExternalQualificationPhaseV63
    graph_event_sequence: Annotated[int, Field(ge=1)]
    graph_event_tip: Sha256
    control_status: ExternalControlStatusV65
    activation_hash: Sha256 | None = None
    activation_event_sequence: Annotated[int, Field(ge=1)] | None = None
    activation_runtime_identity_hash: Sha256 | None = None
    pending_operation_id: Identifier | None = None
    pending_failure_class: ExternalControlFailureClassV65 | None = None
    reason_codes: list[Identifier] = Field(default_factory=list)
    next_valid_operations: list[ExternalControlOperationV65] = Field(
        default_factory=list
    )
    v63_protocol_qualification_granted: bool = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    state_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_state(self) -> "ExternalControlStateV65":
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("V6.5 state reasons must be sorted and unique")
        if self.v63_protocol_qualification_granted != (
            self.v63_phase == "EXTERNALLY_QUALIFIED"
        ):
            raise ValueError("V6.5 state protocol flag differs")
        activation_fields = (
            self.activation_hash,
            self.activation_event_sequence,
            self.activation_runtime_identity_hash,
        )
        if self.control_status == "LEGACY_UNMANAGED":
            if any(value is not None for value in activation_fields):
                raise ValueError(
                    "legacy-unmanaged state cannot expose an activation"
                )
        elif (
            self.control_status != "INCONSISTENT"
            or any(value is not None for value in activation_fields)
        ) and not all(value is not None for value in activation_fields):
            raise ValueError("managed V6.5 state lacks its activation binding")
        if self.control_status == "PENDING_FAILURE":
            if not self.pending_operation_id or not self.pending_failure_class:
                raise ValueError("pending failure lacks operation metadata")
        elif self.pending_failure_class is not None:
            raise ValueError("non-pending state exposes a failure class")
        if self.control_status != "ACTIVE" and self.next_valid_operations:
            raise ValueError("non-active V6.5 state exposes operations")
        if self.state_hash and self.state_hash != self.content_hash():
            raise ValueError("V6.5 state hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "state_hash")

    @classmethod
    def seal(cls, **data: object) -> "ExternalControlStateV65":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"state_hash"})
        payload["state_hash"] = draft.content_hash()
        return cls(**payload)


class ExternalControlResultV65(StrictModel):
    schema_version: Literal["6.5-external-control-result"] = (
        "6.5-external-control-result"
    )
    operation_id: Identifier
    operation_type: ExternalControlOperationV65
    request_hash: Sha256
    resumed: bool
    completion_artifact_hash: Sha256
    output_artifact_hashes: dict[Identifier, Sha256]
    state: ExternalControlStateV65
    v63_protocol_qualification_granted: bool
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_result(self) -> "ExternalControlResultV65":
        if self.v63_protocol_qualification_granted != (
            self.state.v63_protocol_qualification_granted
        ):
            raise ValueError("V6.5 result protocol flag differs")
        return self


class ExternalQualificationProjectionV65(StrictModel):
    """Newest read-only claim projection; V6.3 alone is never final authority."""

    schema_version: Literal["6.5-external-qualification-projection"] = (
        "6.5-external-qualification-projection"
    )
    qualification_id: Identifier
    control_state_hash: Sha256
    control_status: ExternalControlStatusV65
    v63_state_hash: Sha256
    v63_phase: ExternalQualificationPhaseV63
    v63_verification_hash: Sha256 | None
    v63_verification_status: Literal["PASS", "FAIL", "NOT_RUN"]
    projection_status: Literal[
        "NOT_RUN", "REJECTED", "WORKFLOW_VERIFIED"
    ]
    reason_codes: list[Identifier] = Field(default_factory=list)
    claim_ceiling: Literal["workflow_integrity_only"] = (
        "workflow_integrity_only"
    )
    v63_protocol_qualification_granted: bool
    v65_control_verified: bool
    v64_deployment_anchor_verified: Literal[False] = False
    v65_predictive_quality_authority_verified: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    projection_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_projection(self) -> "ExternalQualificationProjectionV65":
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError(
                "V6.5 qualification reasons must be sorted and unique"
            )
        workflow_verified = self.projection_status == "WORKFLOW_VERIFIED"
        expected_workflow = bool(
            self.control_status == "ACTIVE"
            and self.v63_verification_status == "PASS"
            and self.v63_phase == "EXTERNALLY_QUALIFIED"
            and self.v63_protocol_qualification_granted
            and self.v65_control_verified
        )
        if workflow_verified != expected_workflow:
            raise ValueError(
                "V6.5 workflow projection differs from component evidence"
            )
        if self.projection_hash and (
            self.projection_hash != self.content_hash()
        ):
            raise ValueError("V6.5 qualification projection hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "projection_hash")

    @classmethod
    def seal(cls, **data: object) -> "ExternalQualificationProjectionV65":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"projection_hash"})
        payload["projection_hash"] = draft.content_hash()
        return cls(**payload)


@dataclass(frozen=True)
class _EventItem:
    sequence: int
    event_hash: str
    reference: ArtifactRef
    payload: Any


@dataclass(frozen=True)
class _ControlLedger:
    events: list[dict[str, object]]
    items: list[_EventItem]
    activations: list[tuple[_EventItem, ExternalControlActivationV65]]
    intents: dict[str, tuple[_EventItem, ExternalControlIntentV65]]
    completions: dict[
        str, tuple[_EventItem, ExternalControlCompletionV65]
    ]
    failures: dict[
        str, list[tuple[_EventItem, ExternalControlFailureV65]]
    ]
    resolutions: dict[
        str, list[tuple[_EventItem, ExternalControlResolutionV65]]
    ]


@dataclass(frozen=True)
class _ActionOutput:
    artifact_hashes: dict[str, str]


def _graph_events(workspace: StageWorkspaceV50) -> list[dict[str, object]]:
    if not workspace.verify():
        raise ExternalControlPlaneErrorV65(
            "workspace verification failed before V6.5 replay"
        )
    return workspace.graph._read_events(workspace.graph.store)


def _graph_tip(
    workspace: StageWorkspaceV50,
) -> tuple[int, str]:
    events = _graph_events(workspace)
    if not events:
        raise ExternalControlPlaneErrorV65("workspace graph is empty")
    return int(events[-1]["sequence"]), str(events[-1]["event_hash"])


def _seal_envelope(
    *,
    workspace: StageWorkspaceV50,
    kind: str,
    model_type: type[StrictModel],
    hash_field: str,
    data: Mapping[str, object],
) -> StrictModel:
    unsigned = model_type.model_validate(dict(data))
    payload = unsigned.model_dump(mode="json")
    payload["authority_auth_tag"] = workspace._mac(
        kind, getattr(unsigned, "unsigned_hash")()
    )
    payload[hash_field] = sha256_value(
        {key: value for key, value in payload.items() if key != hash_field}
    )
    return model_type.model_validate(payload)


def _verify_envelope(
    *,
    workspace: StageWorkspaceV50,
    kind: str,
    model: Any,
) -> None:
    model.assert_sealed()
    if (
        model.authority_key_id != workspace.authority_key_id
        or not workspace._verify_mac(
            kind, model.unsigned_hash(), model.authority_auth_tag
        )
    ):
        raise ExternalControlPlaneErrorV65(
            f"{kind} workspace authority authentication failed"
        )


def issue_external_control_principal_v65(
    *,
    workspace: StageWorkspaceV50,
    principal_id: str,
    actor_type: ExternalControlActorV65,
    qualification_id: str,
    allowed_operations: list[ExternalControlCapabilityV65] | None = None,
    issued_at: datetime,
    expires_at: datetime,
) -> ExternalControlPrincipalV65:
    """Issue a local capability from the workspace authority boundary."""

    authority_genesis = workspace._artifacts_of_kind(
        "authority_genesis_v50"
    )
    if len(authority_genesis) != 1:
        raise ExternalControlPlaneErrorV65(
            "workspace authority genesis is not unique"
        )
    principal = _seal_envelope(
        workspace=workspace,
        kind=CONTROL_PRINCIPAL_KIND_V65,
        model_type=ExternalControlPrincipalV65,
        hash_field="principal_hash",
        data={
            "principal_id": principal_id,
            "actor_type": actor_type,
            "workspace_spec_hash": workspace.spec.spec_hash,
            "workspace_authority_genesis_hash": (
                authority_genesis[0][0].sha256
            ),
            "qualification_id": qualification_id,
            "allowed_operations": sorted(
                set(allowed_operations or _ALL_CONTROL_CAPABILITIES)
            ),
            "issued_at": issued_at,
            "expires_at": expires_at,
            "authority_key_id": workspace.authority_key_id,
        },
    )
    assert isinstance(principal, ExternalControlPrincipalV65)
    return principal


_CONTROL_MODEL_BY_KIND: dict[str, type[StrictModel]] = {
    CONTROL_ACTIVATION_KIND_V65: ExternalControlActivationV65,
    CONTROL_INTENT_KIND_V65: ExternalControlIntentV65,
    CONTROL_COMPLETION_KIND_V65: ExternalControlCompletionV65,
    CONTROL_FAILURE_KIND_V65: ExternalControlFailureV65,
    CONTROL_RESOLUTION_KIND_V65: ExternalControlResolutionV65,
}


def _read_control_ledger(
    workspace: StageWorkspaceV50,
    *,
    qualification_id: str,
) -> _ControlLedger:
    events = _graph_events(workspace)
    items: list[_EventItem] = []
    activations: list[
        tuple[_EventItem, ExternalControlActivationV65]
    ] = []
    intents: dict[str, tuple[_EventItem, ExternalControlIntentV65]] = {}
    completions: dict[
        str, tuple[_EventItem, ExternalControlCompletionV65]
    ] = {}
    failures: dict[
        str, list[tuple[_EventItem, ExternalControlFailureV65]]
    ] = {}
    resolutions: dict[
        str, list[tuple[_EventItem, ExternalControlResolutionV65]]
    ] = {}
    event_hash_by_sequence = {
        int(event["sequence"]): str(event["event_hash"])
        for event in events
    }
    for event in events:
        if event.get("event_type") != "artifact_committed":
            continue
        reference = ArtifactRef.model_validate(event.get("payload"))
        payload = workspace.graph.store.load_artifact(reference)
        item = _EventItem(
            sequence=int(event["sequence"]),
            event_hash=str(event["event_hash"]),
            reference=reference,
            payload=payload,
        )
        items.append(item)
        model_type = _CONTROL_MODEL_BY_KIND.get(reference.kind)
        if model_type is None:
            continue
        try:
            model = model_type.model_validate(payload)
            _verify_envelope(
                workspace=workspace,
                kind=reference.kind,
                model=model,
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ExternalControlPlaneErrorV65(
                "V6.5 control ledger contains an invalid envelope"
            ) from exc
        if model.qualification_id != qualification_id:
            continue
        if isinstance(model, ExternalControlActivationV65):
            activations.append((item, model))
            continue
        operation_id = model.operation_id
        if isinstance(model, ExternalControlIntentV65):
            if operation_id in intents:
                raise ExternalControlPlaneErrorV65(
                    "duplicate V6.5 operation intent"
                )
            intents[operation_id] = (item, model)
        elif isinstance(model, ExternalControlCompletionV65):
            if operation_id in completions:
                raise ExternalControlPlaneErrorV65(
                    "duplicate V6.5 operation completion"
                )
            completions[operation_id] = (item, model)
        elif isinstance(model, ExternalControlFailureV65):
            failures.setdefault(operation_id, []).append((item, model))
        elif isinstance(model, ExternalControlResolutionV65):
            resolutions.setdefault(operation_id, []).append((item, model))

    if len(activations) > 1:
        raise ExternalControlPlaneErrorV65(
            "duplicate V6.5 control activation"
        )
    if activations:
        activation_item, activation = activations[0]
        if (
            activation_item.sequence <= 1
            or activation.pre_graph_tip
            != event_hash_by_sequence.get(activation_item.sequence - 1)
        ):
            raise ExternalControlPlaneErrorV65(
                "V6.5 activation pre-graph tip differs"
            )
        if any(
            item.sequence <= activation_item.sequence
            for item, _intent in intents.values()
        ):
            raise ExternalControlPlaneErrorV65(
                "V6.5 operation predates control activation"
            )

    for operation_id, (intent_item, intent) in intents.items():
        if (
            intent_item.sequence <= 1
            or intent.pre_graph_tip
            != event_hash_by_sequence.get(intent_item.sequence - 1)
        ):
            raise ExternalControlPlaneErrorV65(
                "V6.5 intent pre-graph tip differs"
            )
        completion_entry = completions.get(operation_id)
        failure_entries = sorted(
            failures.get(operation_id, []), key=lambda entry: entry[0].sequence
        )
        resolution_entries = sorted(
            resolutions.get(operation_id, []),
            key=lambda entry: entry[0].sequence,
        )
        expected_attempts = list(range(1, len(failure_entries) + 1))
        if [failure.attempt_number for _, failure in failure_entries] != (
            expected_attempts
        ):
            raise ExternalControlPlaneErrorV65(
                "V6.5 failure attempts are not contiguous"
            )
        failure_by_hash = {
            failure.failure_hash: (failure_item, failure)
            for failure_item, failure in failure_entries
        }
        if len(failure_by_hash) != len(failure_entries):
            raise ExternalControlPlaneErrorV65(
                "duplicate V6.5 failure receipt"
            )
        resolution_by_failure: dict[
            str, tuple[_EventItem, ExternalControlResolutionV65]
        ] = {}
        for resolution_item, resolution in resolution_entries:
            failure_entry = failure_by_hash.get(resolution.failure_hash)
            if failure_entry is None:
                raise ExternalControlPlaneErrorV65(
                    "V6.5 resolution lacks its failure receipt"
                )
            if resolution.failure_hash in resolution_by_failure:
                raise ExternalControlPlaneErrorV65(
                    "V6.5 failure has duplicate resolutions"
                )
            if resolution_item.sequence <= failure_entry[0].sequence:
                raise ExternalControlPlaneErrorV65(
                    "V6.5 resolution predates its failure"
                )
            if (
                resolution.pre_graph_tip != failure_entry[0].event_hash
                or resolution.post_graph_tip != failure_entry[0].event_hash
                or event_hash_by_sequence.get(
                    resolution_item.sequence - 1
                )
                != failure_entry[0].event_hash
            ):
                raise ExternalControlPlaneErrorV65(
                    "V6.5 resolution graph-tip binding differs"
                )
            resolution_by_failure[resolution.failure_hash] = (
                resolution_item,
                resolution,
            )
        envelopes = [
            *(model for _, model in failure_entries),
            *(model for _, model in resolution_entries),
            *(
                [completion_entry[1]]
                if completion_entry is not None
                else []
            ),
        ]
        for envelope in envelopes:
            if (
                envelope.qualification_id != intent.qualification_id
                or envelope.operation_type != intent.operation_type
                or envelope.intent_hash != intent.intent_hash
                or envelope.request_hash != intent.request_hash
            ):
                raise ExternalControlPlaneErrorV65(
                    "V6.5 operation envelope differs from its intent"
                )
            if isinstance(
                envelope,
                (ExternalControlCompletionV65, ExternalControlFailureV65),
            ) and (
                envelope.principal_id != intent.principal_id
                or envelope.principal_hash != intent.principal_hash
                or envelope.runtime_identity_hash
                != intent.runtime_identity_hash
            ):
                raise ExternalControlPlaneErrorV65(
                    "V6.5 action receipt authority differs from intent"
                )
            if (
                isinstance(envelope, ExternalControlResolutionV65)
                and envelope.decision == "RESUME_EXACT"
                and (
                    envelope.principal_id != intent.principal_id
                    or envelope.principal_hash != intent.principal_hash
                    or envelope.runtime_identity_hash
                    != intent.runtime_identity_hash
                )
            ):
                raise ExternalControlPlaneErrorV65(
                    "V6.5 exact-resume authority differs from intent"
                )
            if isinstance(
                envelope,
                (ExternalControlCompletionV65, ExternalControlFailureV65),
            ) and (
                envelope.input_artifact_hash
                != intent.input_artifact_hash
                or envelope.expected_v63_state_hash
                != intent.expected_v63_state_hash
                or envelope.expected_v63_phase
                != intent.expected_v63_phase
            ):
                raise ExternalControlPlaneErrorV65(
                    "V6.5 operation receipt binding differs from intent"
                )
            if isinstance(envelope, ExternalControlResolutionV65) and (
                envelope.input_artifact_hash
                != intent.input_artifact_hash
                or envelope.expected_v63_state_hash
                != intent.expected_v63_state_hash
                or envelope.expected_v63_phase
                != intent.expected_v63_phase
            ):
                raise ExternalControlPlaneErrorV65(
                    "V6.5 resolution binding differs from intent"
                )
        if completion_entry is not None and (
            completion_entry[1].result_v63_phase
            not in _RESULT_PHASES[intent.operation_type]
        ):
            raise ExternalControlPlaneErrorV65(
                "V6.5 completion has an invalid result phase"
            )
        attempt_start = intent_item
        for failure_item, failure in failure_entries:
            if (
                failure_item.sequence <= attempt_start.sequence
                or failure.pre_graph_tip != attempt_start.event_hash
                or failure.post_graph_tip
                != event_hash_by_sequence.get(failure_item.sequence - 1)
            ):
                raise ExternalControlPlaneErrorV65(
                    "V6.5 failure graph-tip binding differs"
                )
            resolution_entry = resolution_by_failure.get(
                failure.failure_hash
            )
            if resolution_entry is None:
                if failure_item != failure_entries[-1][0]:
                    raise ExternalControlPlaneErrorV65(
                        "later attempt bypasses an unresolved failure"
                    )
                continue
            if resolution_entry[1].decision == "RESUME_EXACT":
                attempt_start = resolution_entry[0]
        if completion_entry is not None:
            completion_item, completion = completion_entry
            if (
                completion.pre_graph_tip != attempt_start.event_hash
                or completion.post_graph_tip
                != event_hash_by_sequence.get(completion_item.sequence - 1)
            ):
                raise ExternalControlPlaneErrorV65(
                    "V6.5 completion graph-tip binding differs"
                )
            artifact_sequence_by_hash = {
                item.reference.sha256: item.sequence for item in items
            }
            operation_segments: list[tuple[int, int]] = []
            segment_start = intent_item.sequence
            for failure_item, failure in failure_entries:
                operation_segments.append(
                    (segment_start, failure_item.sequence)
                )
                resolution_entry = resolution_by_failure.get(
                    failure.failure_hash
                )
                if (
                    resolution_entry is not None
                    and resolution_entry[1].decision == "RESUME_EXACT"
                ):
                    segment_start = resolution_entry[0].sequence
            operation_segments.append(
                (segment_start, completion_item.sequence)
            )
            if any(
                not any(
                    start
                    < artifact_sequence_by_hash.get(artifact_hash, -1)
                    < end
                    for start, end in operation_segments
                )
                for artifact_hash in completion.output_artifact_hashes.values()
            ):
                raise ExternalControlPlaneErrorV65(
                    "V6.5 completion output lies outside its action segment"
                )
        aborts = [
            entry
            for entry in resolution_entries
            if entry[1].decision == "ABORT_ATTEMPT"
        ]
        if len(aborts) > 1 or (aborts and completion_entry is not None):
            raise ExternalControlPlaneErrorV65(
                "aborted V6.5 operation has invalid terminal events"
            )
        if completion_entry is not None:
            if completion_entry[0].sequence <= intent_item.sequence:
                raise ExternalControlPlaneErrorV65(
                    "V6.5 completion predates its intent"
                )
            unresolved = [
                failure
                for _item, failure in failure_entries
                if failure.failure_hash not in resolution_by_failure
            ]
            if unresolved:
                raise ExternalControlPlaneErrorV65(
                    "V6.5 completion bypasses an unresolved failure"
                )

    orphan_ids = (
        set(completions) | set(failures) | set(resolutions)
    ) - set(intents)
    if orphan_ids:
        raise ExternalControlPlaneErrorV65(
            "V6.5 control receipt lacks its intent"
        )
    return _ControlLedger(
        events=events,
        items=items,
        activations=activations,
        intents=intents,
        completions=completions,
        failures=failures,
        resolutions=resolutions,
    )


def _resolved_failure_hashes(
    ledger: _ControlLedger,
    operation_id: str,
) -> set[str]:
    return {
        resolution.failure_hash
        for _item, resolution in ledger.resolutions.get(operation_id, [])
    }


def _abort_resolution(
    ledger: _ControlLedger,
    operation_id: str,
) -> tuple[_EventItem, ExternalControlResolutionV65] | None:
    matches = [
        entry
        for entry in ledger.resolutions.get(operation_id, [])
        if entry[1].decision == "ABORT_ATTEMPT"
    ]
    return matches[0] if matches else None


def _unresolved_failure(
    ledger: _ControlLedger,
    operation_id: str,
) -> tuple[_EventItem, ExternalControlFailureV65] | None:
    resolved = _resolved_failure_hashes(ledger, operation_id)
    matches = [
        entry
        for entry in ledger.failures.get(operation_id, [])
        if entry[1].failure_hash not in resolved
    ]
    if len(matches) > 1:
        raise ExternalControlPlaneErrorV65(
            "operation has multiple unresolved failures"
        )
    return matches[0] if matches else None


def _operation_open(
    ledger: _ControlLedger,
    operation_id: str,
) -> bool:
    return (
        operation_id not in ledger.completions
        and _abort_resolution(ledger, operation_id) is None
    )


def _controlled_segments(
    ledger: _ControlLedger,
) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    for operation_id, (intent_item, _intent) in ledger.intents.items():
        starts = [intent_item.sequence]
        starts.extend(
            item.sequence
            for item, resolution in ledger.resolutions.get(operation_id, [])
            if resolution.decision == "RESUME_EXACT"
        )
        outcomes: list[int] = [
            item.sequence
            for item, _failure in ledger.failures.get(operation_id, [])
        ]
        completion = ledger.completions.get(operation_id)
        if completion is not None:
            outcomes.append(completion[0].sequence)
        for start in sorted(starts):
            later = [sequence for sequence in outcomes if sequence > start]
            if later:
                segments.append((start, min(later)))
    return segments


def _belongs_to_qualification(
    item: _EventItem,
    *,
    qualification_id: str,
    task_id: str | None,
) -> bool:
    payload = item.payload
    if not isinstance(payload, dict):
        return False
    item_qualification_id = payload.get("qualification_id")
    if isinstance(item_qualification_id, str):
        return item_qualification_id == qualification_id
    return bool(
        item.reference.kind == PREDICTION_SEAL_KIND_V50
        and task_id is not None
        and payload.get("task_id") == task_id
    )


def _uncontrolled_v63_artifacts(
    ledger: _ControlLedger,
    *,
    qualification_id: str,
    task_id: str | None,
    after_sequence: int,
) -> list[_EventItem]:
    segments = _controlled_segments(ledger)
    return [
        item
        for item in ledger.items
        if item.sequence > after_sequence
        and item.reference.kind in _V63_PROTECTED_KINDS
        and _belongs_to_qualification(
            item,
            qualification_id=qualification_id,
            task_id=task_id,
        )
        and not any(
            start < item.sequence < end for start, end in segments
        )
    ]


def _project_locked(
    workspace: StageWorkspaceV50,
    *,
    qualification_id: str,
    trusted_public_keys: Mapping[str, bytes],
) -> ExternalControlStateV65:
    v63_state = project_external_qualification_state_v63(
        workspace,
        qualification_id=qualification_id,
        trusted_public_keys=trusted_public_keys,
    )
    ledger = _read_control_ledger(
        workspace, qualification_id=qualification_id
    )
    reasons: list[str] = []
    status: ExternalControlStatusV65 = "ACTIVE"
    pending_operation_id: str | None = None
    pending_failure_class: ExternalControlFailureClassV65 | None = None

    activation_entry = ledger.activations[0] if ledger.activations else None
    if activation_entry is None:
        if ledger.intents or ledger.completions or ledger.failures or (
            ledger.resolutions
        ):
            status = "INCONSISTENT"
            reasons.append("control_event_without_activation")
        else:
            status = "LEGACY_UNMANAGED"
            reasons.append("v65_control_not_activated")
    else:
        uncontrolled = _uncontrolled_v63_artifacts(
            ledger,
            qualification_id=qualification_id,
            task_id=v63_state.task_id,
            after_sequence=activation_entry[0].sequence,
        )
        if uncontrolled:
            status = "INCONSISTENT"
            reasons.append("uncontrolled_v63_artifact")
        if v63_state.phase == "INCONSISTENT":
            status = "INCONSISTENT"
            reasons.append("v63_projection_inconsistent")

    open_operations = [
        operation_id
        for operation_id in ledger.intents
        if _operation_open(ledger, operation_id)
    ]
    if len(open_operations) > 1:
        status = "INCONSISTENT"
        reasons.append("multiple_unresolved_intents")
    elif open_operations:
        candidate_operation_id = open_operations[0]
        failure = _unresolved_failure(ledger, candidate_operation_id)
        if failure is None:
            status = "INCONSISTENT"
            reasons.append("unexplained_pending_intent")
        elif status != "INCONSISTENT":
            status = "PENDING_FAILURE"
            pending_operation_id = candidate_operation_id
            pending_failure_class = failure[1].failure_class

    aborted = [
        operation_id
        for operation_id in ledger.intents
        if _abort_resolution(ledger, operation_id) is not None
    ]
    if aborted and status == "ACTIVE":
        status = "ABORTED"
        reasons.append("qualification_attempt_aborted")

    sequence, tip = _graph_tip(workspace)
    return ExternalControlStateV65.seal(
        qualification_id=qualification_id,
        v63_state_hash=v63_state.state_hash,
        v63_phase=v63_state.phase,
        graph_event_sequence=sequence,
        graph_event_tip=tip,
        control_status=status,
        activation_hash=(
            activation_entry[1].activation_hash
            if activation_entry is not None
            else None
        ),
        activation_event_sequence=(
            activation_entry[0].sequence
            if activation_entry is not None
            else None
        ),
        activation_runtime_identity_hash=(
            activation_entry[1].runtime_identity_hash
            if activation_entry is not None
            else None
        ),
        pending_operation_id=pending_operation_id,
        pending_failure_class=pending_failure_class,
        reason_codes=sorted(set(reasons)),
        next_valid_operations=(
            _NEXT_OPERATION.get(v63_state.phase, [])
            if status == "ACTIVE"
            else []
        ),
        v63_protocol_qualification_granted=(
            v63_state.v63_protocol_qualification_granted
        ),
    )


def project_external_control_state_v65(
    workspace: StageWorkspaceV50,
    *,
    qualification_id: str,
    trusted_public_keys: Mapping[str, bytes],
) -> ExternalControlStateV65:
    """Read-only V6.5 projection over the V6.3 and control ledgers."""

    with workspace.graph.store.writer_transaction():
        return _project_locked(
            workspace,
            qualification_id=qualification_id,
            trusted_public_keys=trusted_public_keys,
        )


def _require_principal(
    *,
    workspace: StageWorkspaceV50,
    principal: ExternalControlPrincipalV65 | None,
    operation: ExternalControlCapabilityV65,
    qualification_id: str,
    now: datetime,
) -> ExternalControlPrincipalV65:
    if principal is None:
        raise PermissionError(
            "V6.5 mutation requires an authenticated principal capability"
        )
    try:
        principal.assert_sealed()
    except ValueError as exc:
        raise PermissionError(
            "V6.5 principal capability is unsealed"
        ) from exc
    if (
        principal.authority_key_id != workspace.authority_key_id
        or not workspace._verify_mac(
            CONTROL_PRINCIPAL_KIND_V65,
            principal.unsigned_hash(),
            principal.authority_auth_tag,
        )
    ):
        raise PermissionError(
            "V6.5 principal capability authentication failed"
        )
    if (
        principal.workspace_spec_hash != workspace.spec.spec_hash
        or principal.qualification_id != qualification_id
    ):
        raise PermissionError(
            "V6.5 principal capability audience differs from workspace "
            "or qualification"
        )
    authority_genesis = workspace._artifacts_of_kind(
        "authority_genesis_v50"
    )
    if (
        len(authority_genesis) != 1
        or principal.workspace_authority_genesis_hash
        != authority_genesis[0][0].sha256
    ):
        raise PermissionError(
            "V6.5 principal capability workspace genesis differs"
        )
    if not principal.issued_at <= now < principal.expires_at:
        raise PermissionError("V6.5 principal capability is not current")
    if operation not in principal.allowed_operations:
        raise PermissionError(
            f"V6.5 principal lacks capability {operation}"
        )
    return principal


def _input_hash(
    *,
    operation_type: ExternalControlOperationV65,
    qualification_id: str,
    payload: object,
) -> str:
    if isinstance(payload, StrictModel):
        serialized: object = payload.model_dump(mode="json")
    else:
        serialized = payload
    # For signed ingress this is the exact content-addressed hash that the
    # V6.3 action will commit.  Internal operations hash their typed argument
    # payload.  Operation and qualification bindings live in request_hash.
    return sha256_value(serialized)


def _request_hash(
    *,
    operation_type: ExternalControlOperationV65,
    qualification_id: str,
    input_artifact_hash: str,
    expected_v63_state_hash: str,
    expected_v63_phase: ExternalQualificationPhaseV63,
    principal: ExternalControlPrincipalV65,
    runtime_identity_hash: str,
) -> str:
    return sha256_value(
        {
            "schema_version": "6.5-external-control-request",
            "operation_type": operation_type,
            "qualification_id": qualification_id,
            "input_artifact_hash": input_artifact_hash,
            "expected_v63_state_hash": expected_v63_state_hash,
            "expected_v63_phase": expected_v63_phase,
            "actor": principal.actor_type,
            "principal_id": principal.principal_id,
            "principal_hash": principal.principal_hash,
            "runtime_identity_hash": runtime_identity_hash,
        }
    )


def _operation_id(
    operation_type: ExternalControlOperationV65,
    request_hash: str,
) -> str:
    return f"v65-{operation_type.replace('_', '-')}-{request_hash[:20]}"


def _error_code(exc: Exception) -> str:
    normalized = re.sub(
        r"[^A-Za-z0-9_.-]+", "-", type(exc).__name__
    ).strip("-")
    return normalized or "ExternalControlError"


def _failure_class(
    exc: Exception,
) -> ExternalControlFailureClassV65:
    if isinstance(
        exc,
        (
            ConnectionError,
            TimeoutError,
            InterruptedError,
        ),
    ):
        return "RETRYABLE"
    return "HUMAN_REQUIRED"


def _load_chain(
    workspace: StageWorkspaceV50,
    *,
    qualification_id: str,
) -> dict[str, Any]:
    contract = _typed_one(
        workspace,
        kind=CONTRACT_KIND_V63,
        model_type=PredictiveExternalQualificationContractV63,
        qualification_id=qualification_id,
    )
    custody = _typed_one(
        workspace,
        kind=CUSTODY_KIND_V63,
        model_type=ExternalEvidenceCustodyV63,
        qualification_id=qualification_id,
        required=False,
    )
    binding = _typed_one(
        workspace,
        kind=PREDICTION_BINDING_KIND_V63,
        model_type=CurrentModelPredictionBindingV63,
        qualification_id=qualification_id,
        required=False,
    )
    registration = _typed_one(
        workspace,
        kind=REGISTRATION_KIND_V63,
        model_type=ExternalPredictionRegistrationV63,
        qualification_id=qualification_id,
        task_id=contract.task_id,
        required=False,
    )
    seal = _typed_one(
        workspace,
        kind=PREDICTION_SEAL_KIND_V50,
        model_type=PredictionSealV50,
        qualification_id=qualification_id,
        task_id=contract.task_id,
        required=False,
    )
    reservation = _typed_one(
        workspace,
        kind=RESERVATION_KIND_V63,
        model_type=ExternalEvaluationReservationV63,
        qualification_id=qualification_id,
        task_id=contract.task_id,
        required=False,
    )
    evaluation = _typed_one(
        workspace,
        kind=EVALUATION_KIND_V63,
        model_type=ExternalAggregateEvaluationV63,
        qualification_id=qualification_id,
        required=False,
    )
    return {
        "contract": contract,
        "custody": custody,
        "binding": binding,
        "registration": registration,
        "seal": seal,
        "reservation": reservation,
        "evaluation": evaluation,
    }


def _recover_v63_coordinator_output(
    workspace: StageWorkspaceV50,
    *,
    qualification_id: str,
    operation_type: Literal["run_prediction", "reserve_evaluation"],
) -> _ActionOutput:
    matches = [
        (reference, receipt)
        for reference, receipt in workspace._artifacts_of_kind(
            OPERATION_RECEIPT_KIND_V63,
            ExternalQualificationOperationReceiptV63,
        )
        if receipt.qualification_id == qualification_id
        and receipt.operation_type == operation_type
    ]
    if len(matches) != 1:
        raise ExternalControlPlaneErrorV65(
            f"completed V6.3 {operation_type} output is ambiguous"
        )
    reference, receipt = matches[0]
    outputs = dict(receipt.output_artifact_hashes)
    outputs["v63_operation_receipt"] = reference.sha256
    return _ActionOutput(artifact_hashes=outputs)


def _exact_artifact_hash(
    workspace: StageWorkspaceV50,
    *,
    kind: str,
    model_type: type[StrictModel],
    qualification_id: str | None = None,
    task_id: str | None = None,
    predicate: Any = None,
) -> str:
    matches = [
        reference.sha256
        for reference, model in workspace._artifacts_of_kind(
            kind, model_type
        )
        if (
            qualification_id is None
            or getattr(model, "qualification_id", None) == qualification_id
        )
        and (
            task_id is None
            or getattr(model, "task_id", None) == task_id
        )
        and (predicate is None or predicate(model))
    ]
    if len(matches) != 1:
        raise ExternalControlPlaneErrorV65(
            f"completed V6.3 {kind} output is ambiguous"
        )
    return matches[0]


def _recover_completed_action_output(
    workspace: StageWorkspaceV50,
    *,
    qualification_id: str,
    operation_type: ExternalControlOperationV65,
) -> _ActionOutput:
    """Recover exact action hashes after V6.3 committed but V6.5 did not."""

    if operation_type in {"run_prediction", "reserve_evaluation"}:
        return _recover_v63_coordinator_output(
            workspace,
            qualification_id=qualification_id,
            operation_type=operation_type,
        )
    chain = _load_chain(workspace, qualification_id=qualification_id)
    contract = chain["contract"]
    if operation_type == "ingest_custody":
        return _ActionOutput(
            artifact_hashes={
                "custody": _exact_artifact_hash(
                    workspace,
                    kind=CUSTODY_KIND_V63,
                    model_type=ExternalEvidenceCustodyV63,
                    qualification_id=qualification_id,
                ),
                "custody_admission": _exact_artifact_hash(
                    workspace,
                    kind=CUSTODY_ADMISSION_KIND_V63,
                    model_type=ExternalCustodyAdmissionV63,
                    qualification_id=qualification_id,
                ),
            }
        )
    if operation_type == "ingest_registration":
        return _ActionOutput(
            artifact_hashes={
                "registration": _exact_artifact_hash(
                    workspace,
                    kind=REGISTRATION_KIND_V63,
                    model_type=ExternalPredictionRegistrationV63,
                    qualification_id=qualification_id,
                    task_id=contract.task_id,
                ),
                "prediction_seal": _exact_artifact_hash(
                    workspace,
                    kind=PREDICTION_SEAL_KIND_V50,
                    model_type=PredictionSealV50,
                    task_id=contract.task_id,
                ),
            }
        )
    if operation_type == "ingest_evaluation":
        return _ActionOutput(
            artifact_hashes={
                "evaluation": _exact_artifact_hash(
                    workspace,
                    kind=EVALUATION_KIND_V63,
                    model_type=ExternalAggregateEvaluationV63,
                    qualification_id=qualification_id,
                ),
                "evaluation_consumption": _exact_artifact_hash(
                    workspace,
                    kind="external_evaluation_consumption_v63",
                    model_type=ExternalEvaluationConsumptionV63,
                    qualification_id=qualification_id,
                    task_id=contract.task_id,
                ),
                "v63_not_run_receipt": _exact_artifact_hash(
                    workspace,
                    kind=QUALIFICATION_KIND_V63,
                    model_type=ExternalPredictiveQualificationReceiptV63,
                    qualification_id=qualification_id,
                    predicate=lambda model: model.status == "NOT_RUN",
                ),
            }
        )
    if operation_type == "ingest_promotion":
        return _ActionOutput(
            artifact_hashes={
                "promotion": _exact_artifact_hash(
                    workspace,
                    kind=PROMOTION_KIND_V63,
                    model_type=ExternalPredictivePromotionV63,
                    qualification_id=qualification_id,
                ),
                "v63_qualification_receipt": _exact_artifact_hash(
                    workspace,
                    kind=QUALIFICATION_KIND_V63,
                    model_type=ExternalPredictiveQualificationReceiptV63,
                    qualification_id=qualification_id,
                    predicate=lambda model: model.status != "NOT_RUN",
                ),
            }
        )
    raise AssertionError(f"unsupported V6.5 recovery {operation_type}")


def _execute_operation_v65(
    control_plane: "ExternalQualificationControlPlaneV65",
    workspace: StageWorkspaceV50,
    operation_type: ExternalControlOperationV65,
    payload: object,
    current_v63_state: ExternalQualificationStateV63,
) -> _ActionOutput:
    """Execute one typed V6.3 action; tests may replace this narrow seam."""

    qualification_id = current_v63_state.qualification_id
    chain = _load_chain(
        workspace, qualification_id=qualification_id
    )
    contract = chain["contract"]

    if operation_type == "ingest_custody":
        if not isinstance(payload, ExternalEvidenceCustodyV63):
            raise TypeError("ingest_custody requires typed V6.3 custody")
        admission = qualification.admit_external_evidence_custody_v63(
            workspace=workspace,
            contract=contract,
            custody=payload,
            trusted_public_keys=control_plane._trusted_public_keys,
        )
        return _ActionOutput(
            artifact_hashes={
                "custody": _artifact_hash_for_model(
                    workspace,
                    kind=CUSTODY_KIND_V63,
                    model_type=ExternalEvidenceCustodyV63,
                    model=payload,
                ),
                "custody_admission": _artifact_hash_for_model(
                    workspace,
                    kind=CUSTODY_ADMISSION_KIND_V63,
                    model_type=type(admission),
                    model=admission,
                ),
            }
        )

    if operation_type == "run_prediction":
        if current_v63_state.phase == "PREDICTION_BOUND":
            return _recover_v63_coordinator_output(
                workspace,
                qualification_id=qualification_id,
                operation_type="run_prediction",
            )
        result = control_plane._v63_coordinator().run_prediction(
            qualification_id=qualification_id,
            expected_state_hash=current_v63_state.state_hash,
            actor="server",
        )
        outputs = dict(result.output_artifact_hashes)
        outputs["v63_operation_receipt"] = result.operation_receipt_hash
        return _ActionOutput(artifact_hashes=outputs)

    if operation_type == "ingest_registration":
        if not isinstance(payload, ExternalPredictionRegistrationV63):
            raise TypeError(
                "ingest_registration requires typed V6.3 registration"
            )
        seal = qualification.register_external_prediction_v63(
            workspace=workspace,
            contract=contract,
            custody=chain["custody"],
            prediction_binding=chain["binding"],
            registration=payload,
            trusted_public_keys=control_plane._trusted_public_keys,
        )
        return _ActionOutput(
            artifact_hashes={
                "registration": _artifact_hash_for_model(
                    workspace,
                    kind=REGISTRATION_KIND_V63,
                    model_type=ExternalPredictionRegistrationV63,
                    model=payload,
                ),
                "prediction_seal": _artifact_hash_for_model(
                    workspace,
                    kind=PREDICTION_SEAL_KIND_V50,
                    model_type=PredictionSealV50,
                    model=seal,
                ),
            }
        )

    if operation_type == "reserve_evaluation":
        if not isinstance(payload, dict):
            raise TypeError("reserve_evaluation input must be typed")
        if current_v63_state.phase == "EVALUATION_RESERVED":
            return _recover_v63_coordinator_output(
                workspace,
                qualification_id=qualification_id,
                operation_type="reserve_evaluation",
            )
        result = control_plane._v63_coordinator().reserve_evaluation(
            qualification_id=qualification_id,
            expected_state_hash=current_v63_state.state_hash,
            evaluator_key_id=str(payload["evaluator_key_id"]),
            evaluator_host_id=str(payload["evaluator_host_id"]),
            actor="server",
        )
        outputs = dict(result.output_artifact_hashes)
        outputs["v63_operation_receipt"] = result.operation_receipt_hash
        return _ActionOutput(artifact_hashes=outputs)

    if operation_type == "ingest_evaluation":
        if not isinstance(payload, ExternalAggregateEvaluationV63):
            raise TypeError(
                "ingest_evaluation requires typed V6.3 evaluation"
            )
        receipt = qualification.assess_external_predictive_qualification_v63(
            workspace=workspace,
            contract=contract,
            custody=chain["custody"],
            prediction_binding=chain["binding"],
            registration=chain["registration"],
            prediction_seal=chain["seal"],
            reservation=chain["reservation"],
            evaluation=payload,
            promotion=None,
            trusted_public_keys=control_plane._trusted_public_keys,
        )
        consumption = _typed_one(
            workspace,
            kind="external_evaluation_consumption_v63",
            model_type=ExternalEvaluationConsumptionV63,
            qualification_id=qualification_id,
            task_id=contract.task_id,
        )
        return _ActionOutput(
            artifact_hashes={
                "evaluation": _artifact_hash_for_model(
                    workspace,
                    kind=EVALUATION_KIND_V63,
                    model_type=ExternalAggregateEvaluationV63,
                    model=payload,
                ),
                "evaluation_consumption": _artifact_hash_for_model(
                    workspace,
                    kind="external_evaluation_consumption_v63",
                    model_type=ExternalEvaluationConsumptionV63,
                    model=consumption,
                ),
                "v63_not_run_receipt": _artifact_hash_for_model(
                    workspace,
                    kind=QUALIFICATION_KIND_V63,
                    model_type=ExternalPredictiveQualificationReceiptV63,
                    model=receipt,
                ),
            }
        )

    if operation_type == "ingest_promotion":
        if not isinstance(payload, ExternalPredictivePromotionV63):
            raise TypeError(
                "ingest_promotion requires typed V6.3 promotion"
            )
        receipt = qualification.assess_external_predictive_qualification_v63(
            workspace=workspace,
            contract=contract,
            custody=chain["custody"],
            prediction_binding=chain["binding"],
            registration=chain["registration"],
            prediction_seal=chain["seal"],
            reservation=chain["reservation"],
            evaluation=chain["evaluation"],
            promotion=payload,
            trusted_public_keys=control_plane._trusted_public_keys,
        )
        return _ActionOutput(
            artifact_hashes={
                "promotion": _artifact_hash_for_model(
                    workspace,
                    kind=PROMOTION_KIND_V63,
                    model_type=ExternalPredictivePromotionV63,
                    model=payload,
                ),
                "v63_qualification_receipt": _artifact_hash_for_model(
                    workspace,
                    kind=QUALIFICATION_KIND_V63,
                    model_type=ExternalPredictiveQualificationReceiptV63,
                    model=receipt,
                ),
            }
        )
    raise AssertionError(f"unsupported V6.5 operation {operation_type}")


class ExternalQualificationControlPlaneV65:
    """Single-writer, failure-accounted ingress for one V6.3 workspace."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        authority_key: bytes,
        authority_key_id: str,
        trusted_public_keys: Mapping[str, bytes],
        principal: ExternalControlPrincipalV65 | None = None,
    ) -> None:
        if len(authority_key) < 32:
            raise ValueError("workspace authority key must contain 32 bytes")
        self._workspace_root = Path(workspace_root).resolve()
        self._authority_key = bytes(authority_key)
        self._authority_key_id = authority_key_id
        self._trusted_public_keys = {
            key_id: bytes(value)
            for key_id, value in trusted_public_keys.items()
        }
        self._principal = principal
        self._runtime_identity = (
            capture_external_control_runtime_identity_v65()
        )

    def _open(self) -> StageWorkspaceV50:
        return StageWorkspaceV50.open_existing(
            self._workspace_root,
            authority_key=self._authority_key,
            authority_key_id=self._authority_key_id,
        )

    def _v63_coordinator(
        self,
        _constructor: Any = _v63_mutation_constructor,
    ) -> ExternalQualificationCoordinatorV63:
        return _constructor(
            self._workspace_root,
            authority_key=self._authority_key,
            authority_key_id=self._authority_key_id,
            trusted_public_keys=self._trusted_public_keys,
        )

    @contextmanager
    def _locked_reopen(self) -> Iterator[StageWorkspaceV50]:
        bootstrap = self._open()
        with bootstrap.graph.store.writer_transaction():
            workspace = self._open()
            if not workspace.verify():
                raise ExternalControlPlaneErrorV65(
                    "workspace failed verification inside V6.5 writer lock"
                )
            yield workspace

    def state(
        self, *, qualification_id: str
    ) -> ExternalControlStateV65:
        with self._locked_reopen() as workspace:
            return _project_locked(
                workspace,
                qualification_id=qualification_id,
                trusted_public_keys=self._trusted_public_keys,
            )

    def latest_qualification(
        self,
        *,
        qualification_id: str,
    ) -> ExternalQualificationProjectionV65:
        """Project the newest claim ceiling without trusting V6.3 alone."""

        before = self.state(qualification_id=qualification_id)
        verification = self._v63_coordinator().verify(
            qualification_id=qualification_id
        )
        after = self.state(qualification_id=qualification_id)
        stable = bool(
            before.state_hash == after.state_hash
            and verification.state_hash == after.v63_state_hash
        )
        control_verified = bool(
            stable
            and after.control_status == "ACTIVE"
            and after.activation_hash
        )
        reasons = list(after.reason_codes)
        reasons.extend(verification.reason_codes)
        if not stable:
            reasons.append("qualification_projection_changed_during_read")
        if not control_verified:
            reasons.append("v65_control_not_current_and_consistent")
        if verification.status != "PASS":
            reasons.append("v63_replay_not_pass")
        reasons.extend(
            [
                "v64_deployment_anchor_not_verified",
                "v65_predictive_quality_authority_not_verified",
            ]
        )
        workflow_verified = bool(
            control_verified
            and verification.status == "PASS"
            and verification.phase == "EXTERNALLY_QUALIFIED"
            and after.v63_protocol_qualification_granted
        )
        if workflow_verified:
            projection_status: Literal[
                "NOT_RUN", "REJECTED", "WORKFLOW_VERIFIED"
            ] = "WORKFLOW_VERIFIED"
        elif (
            after.control_status in {"ABORTED", "INCONSISTENT"}
            or verification.status == "FAIL"
            or after.v63_phase == "REJECTED"
        ):
            projection_status = "REJECTED"
        else:
            projection_status = "NOT_RUN"
        return ExternalQualificationProjectionV65.seal(
            qualification_id=qualification_id,
            control_state_hash=after.state_hash,
            control_status=after.control_status,
            v63_state_hash=after.v63_state_hash,
            v63_phase=after.v63_phase,
            v63_verification_hash=verification.verification_hash,
            v63_verification_status=verification.status,
            projection_status=projection_status,
            reason_codes=sorted(set(reasons)),
            v63_protocol_qualification_granted=(
                after.v63_protocol_qualification_granted
            ),
            v65_control_verified=control_verified,
        )

    def activate(
        self,
        *,
        qualification_id: str,
        expected_v63_state_hash: str,
        expected_v63_phase: ExternalQualificationPhaseV63,
    ) -> ExternalControlStateV65:
        """Explicitly establish the non-retroactive V6.5 ownership boundary."""

        with self._locked_reopen() as workspace:
            principal = _require_principal(
                workspace=workspace,
                principal=self._principal,
                operation="activate",
                qualification_id=qualification_id,
                now=_utc_now(),
            )
            current = project_external_qualification_state_v63(
                workspace,
                qualification_id=qualification_id,
                trusted_public_keys=self._trusted_public_keys,
            )
            ledger = _read_control_ledger(
                workspace, qualification_id=qualification_id
            )
            if ledger.activations:
                return _project_locked(
                    workspace,
                    qualification_id=qualification_id,
                    trusted_public_keys=self._trusted_public_keys,
                )
            if ledger.intents or ledger.completions or ledger.failures or (
                ledger.resolutions
            ):
                raise ExternalControlPlaneErrorV65(
                    "V6.5 control events exist without an activation"
                )
            if (
                current.state_hash != expected_v63_state_hash
                or current.phase != expected_v63_phase
            ):
                raise ExternalControlPlaneErrorV65(
                    "activation expected V6.3 state hash or phase is stale"
                )
            if current.phase != "INPUT_FROZEN":
                raise ExternalControlPlaneErrorV65(
                    "V6.5 activation requires INPUT_FROZEN; "
                    "CONTRACT_FROZEN has no controlled forecast-input "
                    "transition and progressed V6.3 cannot be adopted"
                )
            _sequence, tip = _graph_tip(workspace)
            activation_suffix = sha256_value(
                {
                    "qualification_id": qualification_id,
                    "state_hash": current.state_hash,
                    "graph_tip": tip,
                }
            )[:20]
            activation = _seal_envelope(
                workspace=workspace,
                kind=CONTROL_ACTIVATION_KIND_V65,
                model_type=ExternalControlActivationV65,
                hash_field="activation_hash",
                data={
                    "activation_id": f"v65-activation-{activation_suffix}",
                    "qualification_id": qualification_id,
                    "v63_state_hash": current.state_hash,
                    "v63_phase": current.phase,
                    "pre_graph_tip": tip,
                    "principal_id": principal.principal_id,
                    "principal_hash": principal.principal_hash,
                    "runtime_identity_hash": (
                        self._runtime_identity.runtime_hash
                    ),
                    "activated_at": _utc_now(),
                    "authority_key_id": workspace.authority_key_id,
                },
            )
            workspace.commit_evidence(
                CONTROL_ACTIVATION_KIND_V65,
                activation.model_dump(mode="json"),
            )
            return _project_locked(
                workspace,
                qualification_id=qualification_id,
                trusted_public_keys=self._trusted_public_keys,
            )

    @staticmethod
    def _assert_output_hashes(
        workspace: StageWorkspaceV50,
        output: _ActionOutput,
    ) -> None:
        for artifact_hash in output.artifact_hashes.values():
            workspace._artifact_payload_by_hash(artifact_hash)

    def _completion_result(
        self,
        *,
        workspace: StageWorkspaceV50,
        completion_item: _EventItem,
        completion: ExternalControlCompletionV65,
        resumed: bool,
    ) -> ExternalControlResultV65:
        state = _project_locked(
            workspace,
            qualification_id=completion.qualification_id,
            trusted_public_keys=self._trusted_public_keys,
        )
        return ExternalControlResultV65(
            operation_id=completion.operation_id,
            operation_type=completion.operation_type,
            request_hash=completion.request_hash,
            resumed=resumed,
            completion_artifact_hash=completion_item.reference.sha256,
            output_artifact_hashes=completion.output_artifact_hashes,
            state=state,
            v63_protocol_qualification_granted=(
                state.v63_protocol_qualification_granted
            ),
        )

    def _commit_failure(
        self,
        *,
        workspace: StageWorkspaceV50,
        intent: ExternalControlIntentV65,
        attempt_number: int,
        pre_graph_tip: str,
        exc: Exception,
    ) -> tuple[ExternalControlFailureV65, str]:
        observed = project_external_qualification_state_v63(
            workspace,
            qualification_id=intent.qualification_id,
            trusted_public_keys=self._trusted_public_keys,
        )
        _sequence, post_graph_tip = _graph_tip(workspace)
        failure = _seal_envelope(
            workspace=workspace,
            kind=CONTROL_FAILURE_KIND_V65,
            model_type=ExternalControlFailureV65,
            hash_field="failure_hash",
            data={
                "operation_id": intent.operation_id,
                "qualification_id": intent.qualification_id,
                "operation_type": intent.operation_type,
                "intent_hash": intent.intent_hash,
                "request_hash": intent.request_hash,
                "input_artifact_hash": intent.input_artifact_hash,
                "expected_v63_state_hash": (
                    intent.expected_v63_state_hash
                ),
                "expected_v63_phase": intent.expected_v63_phase,
                "attempt_number": attempt_number,
                "failure_class": _failure_class(exc),
                "error_code": _error_code(exc),
                "error_message_hash": sha256_value(
                    {
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                ),
                "observed_v63_state_hash": observed.state_hash,
                "observed_v63_phase": observed.phase,
                "pre_graph_tip": pre_graph_tip,
                "post_graph_tip": post_graph_tip,
                "principal_id": intent.principal_id,
                "principal_hash": intent.principal_hash,
                "runtime_identity_hash": intent.runtime_identity_hash,
                "failed_at": _utc_now(),
                "authority_key_id": workspace.authority_key_id,
                "v63_protocol_qualification_granted": (
                    observed.v63_protocol_qualification_granted
                ),
            },
        )
        assert isinstance(failure, ExternalControlFailureV65)
        reference = workspace.commit_evidence(
            CONTROL_FAILURE_KIND_V65,
            failure.model_dump(mode="json"),
        )
        return failure, reference.sha256

    def _mutate(
        self,
        *,
        operation_type: ExternalControlOperationV65,
        qualification_id: str,
        expected_v63_state_hash: str,
        expected_v63_phase: ExternalQualificationPhaseV63,
        payload: object,
    ) -> ExternalControlResultV65:
        input_artifact_hash = _input_hash(
            operation_type=operation_type,
            qualification_id=qualification_id,
            payload=payload,
        )

        with self._locked_reopen() as workspace:
            principal = _require_principal(
                workspace=workspace,
                principal=self._principal,
                operation=operation_type,
                qualification_id=qualification_id,
                now=_utc_now(),
            )
            request_hash = _request_hash(
                operation_type=operation_type,
                qualification_id=qualification_id,
                input_artifact_hash=input_artifact_hash,
                expected_v63_state_hash=expected_v63_state_hash,
                expected_v63_phase=expected_v63_phase,
                principal=principal,
                runtime_identity_hash=self._runtime_identity.runtime_hash,
            )
            operation_id = _operation_id(operation_type, request_hash)
            current = project_external_qualification_state_v63(
                workspace,
                qualification_id=qualification_id,
                trusted_public_keys=self._trusted_public_keys,
            )
            ledger = _read_control_ledger(
                workspace, qualification_id=qualification_id
            )
            exact_intent = ledger.intents.get(operation_id)
            exact_completion = ledger.completions.get(operation_id)

            same_transition = [
                intent
                for _item, intent in ledger.intents.values()
                if intent.operation_type == operation_type
            ]
            if any(
                intent.request_hash != request_hash
                for intent in same_transition
            ):
                raise ExternalControlPlaneErrorV65(
                    "a divergent request already exists for this transition"
                )
            open_other = [
                candidate_id
                for candidate_id in ledger.intents
                if candidate_id != operation_id
                and _operation_open(ledger, candidate_id)
            ]
            if open_other:
                raise ExternalControlPlaneErrorV65(
                    "a second unresolved V6.5 intent is forbidden"
                )

            if exact_completion is not None:
                return self._completion_result(
                    workspace=workspace,
                    completion_item=exact_completion[0],
                    completion=exact_completion[1],
                    resumed=True,
                )
            if _abort_resolution(ledger, operation_id) is not None:
                raise ExternalControlPlaneErrorV65(
                    "qualification attempt was explicitly aborted"
                )

            resumed = exact_intent is not None
            if exact_intent is None:
                projected = _project_locked(
                    workspace,
                    qualification_id=qualification_id,
                    trusted_public_keys=self._trusted_public_keys,
                )
                if projected.control_status != "ACTIVE":
                    raise ExternalControlPlaneErrorV65(
                        "V6.5 control projection is not active"
                    )
                if (
                    current.state_hash != expected_v63_state_hash
                    or current.phase != expected_v63_phase
                ):
                    raise ExternalControlPlaneErrorV65(
                        "expected V6.3 state hash or phase is stale"
                    )
                if current.phase not in _START_PHASES[operation_type]:
                    raise ExternalControlPlaneErrorV65(
                        f"{operation_type} is invalid from {current.phase}"
                    )
                _sequence, pre_graph_tip = _graph_tip(workspace)
                intent = _seal_envelope(
                    workspace=workspace,
                    kind=CONTROL_INTENT_KIND_V65,
                    model_type=ExternalControlIntentV65,
                    hash_field="intent_hash",
                    data={
                        "operation_id": operation_id,
                        "qualification_id": qualification_id,
                        "operation_type": operation_type,
                        "request_hash": request_hash,
                        "input_artifact_hash": input_artifact_hash,
                        "expected_v63_state_hash": (
                            expected_v63_state_hash
                        ),
                        "expected_v63_phase": expected_v63_phase,
                        "pre_graph_tip": pre_graph_tip,
                        "actor": principal.actor_type,
                        "principal_id": principal.principal_id,
                        "principal_hash": principal.principal_hash,
                        "runtime_identity_hash": (
                            self._runtime_identity.runtime_hash
                        ),
                        "started_at": _utc_now(),
                        "authority_key_id": workspace.authority_key_id,
                    },
                )
                assert isinstance(intent, ExternalControlIntentV65)
                workspace.commit_evidence(
                    CONTROL_INTENT_KIND_V65,
                    intent.model_dump(mode="json"),
                )
            else:
                intent = exact_intent[1]
                if (
                    intent.request_hash != request_hash
                    or intent.input_artifact_hash != input_artifact_hash
                    or intent.expected_v63_state_hash
                    != expected_v63_state_hash
                    or intent.expected_v63_phase != expected_v63_phase
                    or intent.actor != principal.actor_type
                    or intent.principal_id != principal.principal_id
                    or intent.principal_hash != principal.principal_hash
                    or intent.runtime_identity_hash
                    != self._runtime_identity.runtime_hash
                ):
                    raise ExternalControlPlaneErrorV65(
                        "operation retry differs from its V6.5 intent"
                    )
                unresolved = _unresolved_failure(ledger, operation_id)
                if unresolved is None:
                    latest = max(
                        [
                            exact_intent[0],
                            *(
                                item
                                for item, resolution in ledger.resolutions.get(
                                    operation_id, []
                                )
                                if resolution.decision == "RESUME_EXACT"
                            ),
                        ],
                        key=lambda item: item.sequence,
                    )
                    _sequence, tip = _graph_tip(workspace)
                    if tip != latest.event_hash:
                        raise ExternalControlPlaneErrorV65(
                            "unexplained pending intent moved the graph tip; "
                            "unreceipted V6.3 artifacts cannot be adopted"
                        )
                    failure, failure_ref = self._commit_failure(
                        workspace=workspace,
                        intent=intent,
                        attempt_number=len(
                            ledger.failures.get(operation_id, [])
                        )
                        + 1,
                        pre_graph_tip=latest.event_hash,
                        exc=RuntimeError(
                            "interrupted after action without a terminal "
                            "V6.5 receipt"
                        ),
                    )
                    raise ExternalControlOperationFailedV65(
                        operation_id=operation_id,
                        failure_class=failure.failure_class,
                        failure_receipt_hash=failure_ref,
                    )
                _sequence, tip = _graph_tip(workspace)
                if tip != unresolved[0].event_hash:
                    raise ExternalControlPlaneErrorV65(
                        "exact retry is stale relative to its failure"
                    )
                if current.phase not in (
                    _START_PHASES[operation_type]
                    | _RESULT_PHASES[operation_type]
                    | _RECOVERY_PHASES[operation_type]
                ):
                    raise ExternalControlPlaneErrorV65(
                        "failed operation cannot resume from current phase"
                    )
                resolution = _seal_envelope(
                    workspace=workspace,
                    kind=CONTROL_RESOLUTION_KIND_V65,
                    model_type=ExternalControlResolutionV65,
                    hash_field="resolution_hash",
                    data={
                        "operation_id": operation_id,
                        "qualification_id": qualification_id,
                        "operation_type": operation_type,
                        "intent_hash": intent.intent_hash,
                        "request_hash": request_hash,
                        "input_artifact_hash": (
                            intent.input_artifact_hash
                        ),
                        "expected_v63_state_hash": (
                            intent.expected_v63_state_hash
                        ),
                        "expected_v63_phase": intent.expected_v63_phase,
                        "failure_hash": unresolved[1].failure_hash,
                        "decision": "RESUME_EXACT",
                        "actor": principal.actor_type,
                        "principal_id": principal.principal_id,
                        "principal_hash": principal.principal_hash,
                        "runtime_identity_hash": (
                            self._runtime_identity.runtime_hash
                        ),
                        "pre_graph_tip": tip,
                        "post_graph_tip": tip,
                        "resolved_at": _utc_now(),
                        "authority_key_id": workspace.authority_key_id,
                    },
                )
                workspace.commit_evidence(
                    CONTROL_RESOLUTION_KIND_V65,
                    resolution.model_dump(mode="json"),
                )

            current = project_external_qualification_state_v63(
                workspace,
                qualification_id=qualification_id,
                trusted_public_keys=self._trusted_public_keys,
            )
            _sequence, action_pre_tip = _graph_tip(workspace)
            output: _ActionOutput | None = None
            action_state: ExternalQualificationStateV63 | None = None
            try:
                if current.phase in _RESULT_PHASES[operation_type]:
                    output = _recover_completed_action_output(
                        workspace,
                        qualification_id=qualification_id,
                        operation_type=operation_type,
                    )
                else:
                    output = _execute_operation_v65(
                        self,
                        workspace,
                        operation_type,
                        payload,
                        current,
                    )
                self._assert_output_hashes(workspace, output)
                action_state = project_external_qualification_state_v63(
                    workspace,
                    qualification_id=qualification_id,
                    trusted_public_keys=self._trusted_public_keys,
                )
                if action_state.phase not in _RESULT_PHASES[operation_type]:
                    raise ExternalControlPlaneErrorV65(
                        f"{operation_type} produced {action_state.phase}"
                    )
                _sequence, action_post_tip = _graph_tip(workspace)
                completion = _seal_envelope(
                    workspace=workspace,
                    kind=CONTROL_COMPLETION_KIND_V65,
                    model_type=ExternalControlCompletionV65,
                    hash_field="completion_hash",
                    data={
                        "operation_id": operation_id,
                        "qualification_id": qualification_id,
                        "operation_type": operation_type,
                        "intent_hash": intent.intent_hash,
                        "request_hash": request_hash,
                        "input_artifact_hash": input_artifact_hash,
                        "expected_v63_state_hash": (
                            expected_v63_state_hash
                        ),
                        "expected_v63_phase": expected_v63_phase,
                        "result_v63_state_hash": action_state.state_hash,
                        "result_v63_phase": action_state.phase,
                        "pre_graph_tip": action_pre_tip,
                        "post_graph_tip": action_post_tip,
                        "output_artifact_hashes": dict(
                            sorted(output.artifact_hashes.items())
                        ),
                        "principal_id": intent.principal_id,
                        "principal_hash": intent.principal_hash,
                        "runtime_identity_hash": (
                            intent.runtime_identity_hash
                        ),
                        "completed_at": _utc_now(),
                        "authority_key_id": workspace.authority_key_id,
                        "v63_protocol_qualification_granted": (
                            action_state.v63_protocol_qualification_granted
                        ),
                    },
                )
                assert isinstance(
                    completion, ExternalControlCompletionV65
                )
                completion_ref = workspace.commit_evidence(
                    CONTROL_COMPLETION_KIND_V65,
                    completion.model_dump(mode="json"),
                )
            except Exception as exc:
                refreshed = _read_control_ledger(
                    workspace, qualification_id=qualification_id
                )
                committed_completion = refreshed.completions.get(
                    operation_id
                )
                if committed_completion is not None:
                    committed = committed_completion[1]
                    if (
                        output is None
                        or action_state is None
                        or committed.result_v63_state_hash
                        != action_state.state_hash
                        or committed.result_v63_phase != action_state.phase
                        or committed.output_artifact_hashes
                        != dict(sorted(output.artifact_hashes.items()))
                        or committed.pre_graph_tip != action_pre_tip
                    ):
                        raise ExternalControlPlaneErrorV65(
                            "committed V6.5 completion differs from the "
                            "current action result"
                        ) from exc
                    return self._completion_result(
                        workspace=workspace,
                        completion_item=committed_completion[0],
                        completion=committed,
                        resumed=resumed,
                    )
                failure, failure_ref = self._commit_failure(
                    workspace=workspace,
                    intent=intent,
                    attempt_number=len(
                        refreshed.failures.get(operation_id, [])
                    )
                    + 1,
                    pre_graph_tip=action_pre_tip,
                    exc=exc,
                )
                raise ExternalControlOperationFailedV65(
                    operation_id=operation_id,
                    failure_class=failure.failure_class,
                    failure_receipt_hash=failure_ref,
                ) from exc

            state = _project_locked(
                workspace,
                qualification_id=qualification_id,
                trusted_public_keys=self._trusted_public_keys,
            )
            return ExternalControlResultV65(
                operation_id=operation_id,
                operation_type=operation_type,
                request_hash=request_hash,
                resumed=resumed,
                completion_artifact_hash=completion_ref.sha256,
                output_artifact_hashes=completion.output_artifact_hashes,
                state=state,
                v63_protocol_qualification_granted=(
                    state.v63_protocol_qualification_granted
                ),
            )

    def ingest_custody(
        self,
        *,
        qualification_id: str,
        expected_v63_state_hash: str,
        expected_v63_phase: ExternalQualificationPhaseV63,
        custody: ExternalEvidenceCustodyV63,
    ) -> ExternalControlResultV65:
        if not isinstance(custody, ExternalEvidenceCustodyV63):
            raise TypeError("custody must be ExternalEvidenceCustodyV63")
        custody.assert_sealed()
        return self._mutate(
            operation_type="ingest_custody",
            qualification_id=qualification_id,
            expected_v63_state_hash=expected_v63_state_hash,
            expected_v63_phase=expected_v63_phase,
            payload=custody,
        )

    def run_prediction(
        self,
        *,
        qualification_id: str,
        expected_v63_state_hash: str,
        expected_v63_phase: ExternalQualificationPhaseV63,
    ) -> ExternalControlResultV65:
        return self._mutate(
            operation_type="run_prediction",
            qualification_id=qualification_id,
            expected_v63_state_hash=expected_v63_state_hash,
            expected_v63_phase=expected_v63_phase,
            payload={"runtime": "current-v63-model"},
        )

    def ingest_registration(
        self,
        *,
        qualification_id: str,
        expected_v63_state_hash: str,
        expected_v63_phase: ExternalQualificationPhaseV63,
        registration: ExternalPredictionRegistrationV63,
    ) -> ExternalControlResultV65:
        if not isinstance(registration, ExternalPredictionRegistrationV63):
            raise TypeError(
                "registration must be ExternalPredictionRegistrationV63"
            )
        registration.assert_sealed()
        return self._mutate(
            operation_type="ingest_registration",
            qualification_id=qualification_id,
            expected_v63_state_hash=expected_v63_state_hash,
            expected_v63_phase=expected_v63_phase,
            payload=registration,
        )

    def reserve_evaluation(
        self,
        *,
        qualification_id: str,
        expected_v63_state_hash: str,
        expected_v63_phase: ExternalQualificationPhaseV63,
        evaluator_key_id: str,
        evaluator_host_id: str,
    ) -> ExternalControlResultV65:
        payload = {
            "schema_version": "6.5-evaluation-reservation-input",
            "evaluator_key_id": evaluator_key_id,
            "evaluator_host_id": evaluator_host_id,
        }
        return self._mutate(
            operation_type="reserve_evaluation",
            qualification_id=qualification_id,
            expected_v63_state_hash=expected_v63_state_hash,
            expected_v63_phase=expected_v63_phase,
            payload=payload,
        )

    def ingest_evaluation(
        self,
        *,
        qualification_id: str,
        expected_v63_state_hash: str,
        expected_v63_phase: ExternalQualificationPhaseV63,
        evaluation: ExternalAggregateEvaluationV63,
    ) -> ExternalControlResultV65:
        if not isinstance(evaluation, ExternalAggregateEvaluationV63):
            raise TypeError(
                "evaluation must be ExternalAggregateEvaluationV63"
            )
        evaluation.assert_sealed()
        return self._mutate(
            operation_type="ingest_evaluation",
            qualification_id=qualification_id,
            expected_v63_state_hash=expected_v63_state_hash,
            expected_v63_phase=expected_v63_phase,
            payload=evaluation,
        )

    def ingest_promotion(
        self,
        *,
        qualification_id: str,
        expected_v63_state_hash: str,
        expected_v63_phase: ExternalQualificationPhaseV63,
        promotion: ExternalPredictivePromotionV63,
    ) -> ExternalControlResultV65:
        if not isinstance(promotion, ExternalPredictivePromotionV63):
            raise TypeError(
                "promotion must be ExternalPredictivePromotionV63"
            )
        promotion.assert_sealed()
        return self._mutate(
            operation_type="ingest_promotion",
            qualification_id=qualification_id,
            expected_v63_state_hash=expected_v63_state_hash,
            expected_v63_phase=expected_v63_phase,
            payload=promotion,
        )

    def abort_attempt(
        self,
        *,
        qualification_id: str,
        operation_id: str,
    ) -> ExternalControlStateV65:
        with self._locked_reopen() as workspace:
            principal = _require_principal(
                workspace=workspace,
                principal=self._principal,
                operation="abort_attempt",
                qualification_id=qualification_id,
                now=_utc_now(),
            )
            ledger = _read_control_ledger(
                workspace, qualification_id=qualification_id
            )
            intent_entry = ledger.intents.get(operation_id)
            if intent_entry is None:
                raise ExternalControlPlaneErrorV65(
                    "abort target has no V6.5 intent"
                )
            if operation_id in ledger.completions:
                raise ExternalControlPlaneErrorV65(
                    "completed operation cannot be aborted"
                )
            if _abort_resolution(ledger, operation_id) is not None:
                return _project_locked(
                    workspace,
                    qualification_id=qualification_id,
                    trusted_public_keys=self._trusted_public_keys,
                )
            failure = _unresolved_failure(ledger, operation_id)
            if failure is None:
                raise ExternalControlPlaneErrorV65(
                    "abort requires an unresolved failure receipt"
                )
            current = project_external_qualification_state_v63(
                workspace,
                qualification_id=qualification_id,
                trusted_public_keys=self._trusted_public_keys,
            )
            intent = intent_entry[1]
            resume_starts = [
                item
                for item, resolution in ledger.resolutions.get(
                    operation_id, []
                )
                if resolution.decision == "RESUME_EXACT"
                and item.sequence < failure[0].sequence
            ]
            attempt_start = max(
                [intent_entry[0], *resume_starts],
                key=lambda item: item.sequence,
            )
            progressed_artifacts = [
                item
                for item in ledger.items
                if attempt_start.sequence
                < item.sequence
                < failure[0].sequence
                and item.reference.kind in _V63_PROTECTED_KINDS
                and _belongs_to_qualification(
                    item,
                    qualification_id=qualification_id,
                    task_id=current.task_id,
                )
            ]
            if (
                current.phase != intent.expected_v63_phase
                or progressed_artifacts
            ):
                raise ExternalControlPlaneErrorV65(
                    "progressed V6.3 operation cannot abort; resume exact "
                    "completion reconciliation"
                )
            if failure[1].failure_class != "HUMAN_REQUIRED":
                raise ExternalControlPlaneErrorV65(
                    "retryable operation must resume exact and cannot abort"
                )
            _sequence, tip = _graph_tip(workspace)
            if tip != failure[0].event_hash:
                raise ExternalControlPlaneErrorV65(
                    "abort is stale relative to the failure receipt"
                )
            resolution = _seal_envelope(
                workspace=workspace,
                kind=CONTROL_RESOLUTION_KIND_V65,
                model_type=ExternalControlResolutionV65,
                hash_field="resolution_hash",
                data={
                    "operation_id": operation_id,
                    "qualification_id": qualification_id,
                    "operation_type": intent.operation_type,
                    "intent_hash": intent.intent_hash,
                    "request_hash": intent.request_hash,
                    "input_artifact_hash": intent.input_artifact_hash,
                    "expected_v63_state_hash": (
                        intent.expected_v63_state_hash
                    ),
                    "expected_v63_phase": intent.expected_v63_phase,
                    "failure_hash": failure[1].failure_hash,
                    "decision": "ABORT_ATTEMPT",
                    "actor": principal.actor_type,
                    "principal_id": principal.principal_id,
                    "principal_hash": principal.principal_hash,
                    "runtime_identity_hash": (
                        self._runtime_identity.runtime_hash
                    ),
                    "pre_graph_tip": tip,
                    "post_graph_tip": tip,
                    "resolved_at": _utc_now(),
                    "authority_key_id": workspace.authority_key_id,
                },
            )
            workspace.commit_evidence(
                CONTROL_RESOLUTION_KIND_V65,
                resolution.model_dump(mode="json"),
            )
            return _project_locked(
                workspace,
                qualification_id=qualification_id,
                trusted_public_keys=self._trusted_public_keys,
            )


del _v63_mutation_constructor


__all__ = [
    "CONTROL_ACTIVATION_KIND_V65",
    "CONTROL_COMPLETION_KIND_V65",
    "CONTROL_FAILURE_KIND_V65",
    "CONTROL_INTENT_KIND_V65",
    "CONTROL_PRINCIPAL_KIND_V65",
    "CONTROL_RESOLUTION_KIND_V65",
    "ExternalControlActivationV65",
    "ExternalControlCompletionV65",
    "ExternalControlFailureClassV65",
    "ExternalControlFailureV65",
    "ExternalControlIntentV65",
    "ExternalControlOperationFailedV65",
    "ExternalControlOperationV65",
    "ExternalControlPlaneErrorV65",
    "ExternalControlPrincipalV65",
    "ExternalControlResolutionV65",
    "ExternalControlResolutionDecisionV65",
    "ExternalControlResultV65",
    "ExternalControlRuntimeIdentityV65",
    "ExternalControlStateV65",
    "ExternalQualificationControlPlaneV65",
    "ExternalQualificationProjectionV65",
    "capture_external_control_runtime_identity_v65",
    "issue_external_control_principal_v65",
    "project_external_control_state_v65",
]
