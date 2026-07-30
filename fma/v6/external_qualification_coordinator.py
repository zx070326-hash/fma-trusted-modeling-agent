"""Graph-native coordinator for the V6.3 external qualification protocol.

The coordinator is intentionally a small code-owned state machine.  It does
not sign external-role envelopes and it never receives private holdout target
values.  Models may prepare public prediction work, but only this harness
surface may serialize an authority transition.

Every mutation follows the same protocol:

``lock -> reopen -> verify -> project -> intent -> one action -> receipt``

The append-only intent makes an interrupted operation visible.  Only the exact
same request may resume it; a different request fails closed.  The underlying
``RunStore`` lock is re-entrant, so existing typed V5/V6.3 APIs can safely take
their own nested writer lock.

This candidate currently owns prediction generation and evaluation
reservation only.  Signed custody, registry, evaluation, and promotion ingress
still require a protected service boundary.  Consequently a V6.3 terminal
phase is exposed only as a protocol result; every coordinator-facing
``scientific_qualification_granted`` field remains false until an additive
deployment-anchored authority layer can establish real external provenance.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5.stage_workspace import StageWorkspaceError, StageWorkspaceV50
from fma.v5.workspace_schemas import PredictionSealV50, RoleExecutionReceiptV50

from . import external_qualification as qualification
from .external_prediction_runtime import (
    PREDICTION_RUNTIME_ADAPTER_ID_V63,
    PREDICTION_TRACE_KIND_V63,
    ExternalPredictionRuntimeError,
    run_current_model_external_prediction_v63,
    verify_current_model_external_prediction_v63,
)
from .external_qualification import (
    CurrentModelPredictionBindingV63,
    ExternalAggregateEvaluationV63,
    ExternalCustodyAdmissionV63,
    ExternalEvaluationConsumptionV63,
    ExternalEvaluationReservationV63,
    ExternalEvidenceCustodyV63,
    ExternalForecastInputV63,
    ExternalPredictionRegistrationV63,
    ExternalPredictionVectorV63,
    ExternalPredictivePromotionV63,
    ExternalPredictiveQualificationReceiptV63,
    ExternalQualificationError,
    PredictiveExternalQualificationContractV63,
)


ExternalQualificationPhaseV63 = Literal[
    "LOCAL_READY",
    "CONTRACT_FROZEN",
    "INPUT_FROZEN",
    "CUSTODY_COMMITTED",
    "CUSTODY_REJECTED",
    "CUSTODY_VERIFIED",
    "PREDICTION_BOUND",
    "REGISTRATION_COMMITTED",
    "PREDICTION_REGISTERED",
    "EVALUATION_RESERVED",
    "EVALUATION_COMMITTED",
    "AWAITING_PROMOTION",
    "PROMOTION_COMMITTED",
    "EXTERNALLY_QUALIFIED",
    "REJECTED",
    "STALE",
    "INCONSISTENT",
]
CoordinatorActorV63 = Literal["operator", "server"]
CoordinatorOperationV63 = Literal[
    "run_prediction",
    "reserve_evaluation",
]

CONTRACT_KIND_V63 = "predictive_external_qualification_contract_v63"
FORECAST_INPUT_KIND_V63 = "external_forecast_input_v63"
CUSTODY_KIND_V63 = "external_evidence_custody_v63"
CUSTODY_ADMISSION_KIND_V63 = "external_custody_admission_v63"
PREDICTION_VECTOR_KIND_V63 = "external_prediction_vector_v63"
PREDICTION_BINDING_KIND_V63 = "current_model_prediction_binding_v63"
REGISTRATION_KIND_V63 = "external_prediction_registration_v63"
PREDICTION_SEAL_KIND_V50 = "prediction_seal_v50"
RESERVATION_KIND_V63 = "external_evaluation_reservation_v63"
EVALUATION_KIND_V63 = "external_aggregate_evaluation_v63"
CONSUMPTION_KIND_V63 = "external_evaluation_consumption_v63"
PROMOTION_KIND_V63 = "external_predictive_promotion_v63"
QUALIFICATION_KIND_V63 = "external_predictive_qualification_v63"
OPERATION_INTENT_KIND_V63 = "external_qualification_operation_intent_v63"
OPERATION_RECEIPT_KIND_V63 = "external_qualification_operation_receipt_v63"
DISPATCH_PACKET_KIND_V63 = "external_evaluation_dispatch_packet_v63"

_TERMINAL_PHASES = {
    "CUSTODY_REJECTED",
    "EXTERNALLY_QUALIFIED",
    "REJECTED",
    "STALE",
    "INCONSISTENT",
}
_NEXT_ACTIONS: dict[str, list[str]] = {
    "LOCAL_READY": ["freeze_contract"],
    "CONTRACT_FROZEN": ["freeze_forecast_input"],
    "INPUT_FROZEN": ["ingest_custody"],
    "CUSTODY_COMMITTED": ["resume_custody_admission"],
    "CUSTODY_REJECTED": [],
    "CUSTODY_VERIFIED": ["run_prediction"],
    "PREDICTION_BOUND": ["ingest_registration"],
    "REGISTRATION_COMMITTED": ["resume_registration"],
    "PREDICTION_REGISTERED": ["reserve_evaluation"],
    "EVALUATION_RESERVED": ["dispatch_evaluation"],
    "EVALUATION_COMMITTED": ["resume_evaluation_ingest"],
    "AWAITING_PROMOTION": ["ingest_promotion"],
    "PROMOTION_COMMITTED": ["resume_promotion"],
    "EXTERNALLY_QUALIFIED": [],
    "REJECTED": [],
    "STALE": ["start_new_attempt"],
    "INCONSISTENT": ["inspect_authority_ledger"],
}


class ExternalQualificationCoordinatorError(RuntimeError):
    """A coordinator or state replay failed closed."""


def _make_v65_mutation_constructor_gate() -> tuple[Any, Any]:
    """Create a one-claim constructor whose token never becomes a module value."""

    token = object()
    claimed = False

    def claim() -> Any:
        nonlocal claimed
        if claimed:
            raise RuntimeError(
                "V6.5 mutation-coordinator constructor was already claimed"
            )
        claimed = True

        def construct(*args: object, **kwargs: object) -> Any:
            if "_mutation_gate" in kwargs:
                raise TypeError("mutation gate is process-owned")
            return ExternalQualificationCoordinatorV63(
                *args,
                **kwargs,
                _mutation_gate=token,
            )

        return construct

    def validates(candidate: object) -> bool:
        return candidate is token

    return claim, validates


(
    _claim_v65_mutation_coordinator_constructor,
    _valid_v65_mutation_gate,
) = _make_v65_mutation_constructor_gate()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _assert_aware(value: datetime, field_name: str) -> None:
    if value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


class ExternalQualificationAuthorityArtifactV63(StrictModel):
    """One authority artifact in graph-event order."""

    event_sequence: Annotated[int, Field(ge=1)]
    role: Identifier
    kind: Identifier
    artifact_hash: Sha256


class ExternalQualificationStateV63(StrictModel):
    """Pure projection of one V6.3 qualification from the graph ledger."""

    schema_version: Literal["6.3-external-qualification-state"] = (
        "6.3-external-qualification-state"
    )
    qualification_id: Identifier
    task_id: Identifier | None
    workspace_spec_hash: Sha256
    phase: ExternalQualificationPhaseV63
    current: bool
    terminal: bool
    graph_event_sequence: Annotated[int, Field(ge=1)]
    graph_event_tip: Sha256
    authority_artifacts: list[ExternalQualificationAuthorityArtifactV63]
    ordered_authority_artifact_hashes: list[Sha256]
    pending_operation_id: Identifier | None = None
    reason_codes: list[Identifier] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)
    next_valid_actions: list[Identifier] = Field(default_factory=list)
    v63_protocol_qualification_granted: bool = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    state_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_state(self) -> "ExternalQualificationStateV63":
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("state reason codes must be sorted and unique")
        if self.ordered_authority_artifact_hashes != [
            item.artifact_hash for item in self.authority_artifacts
        ]:
            raise ValueError("ordered authority hashes differ from artifacts")
        if [item.event_sequence for item in self.authority_artifacts] != sorted(
            item.event_sequence for item in self.authority_artifacts
        ):
            raise ValueError("authority artifacts are not in graph-event order")
        qualified = self.phase == "EXTERNALLY_QUALIFIED"
        if self.v63_protocol_qualification_granted != qualified:
            raise ValueError("V6.3 protocol flag differs from phase")
        if self.current != (self.phase not in {"STALE", "INCONSISTENT"}):
            raise ValueError("state current flag differs from phase")
        if self.terminal != (self.phase in _TERMINAL_PHASES):
            raise ValueError("state terminal flag differs from phase")
        if self.next_valid_actions != _NEXT_ACTIONS[self.phase]:
            raise ValueError("state next actions differ from phase")
        if self.state_hash and self.state_hash != self.content_hash():
            raise ValueError("external qualification state hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "state_hash")

    def assert_sealed(self) -> None:
        if not self.state_hash or self.state_hash != self.content_hash():
            raise ValueError("external qualification state is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ExternalQualificationStateV63":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"state_hash"})
        payload["state_hash"] = draft.content_hash()
        return cls(**payload)


class ExternalQualificationOperationIntentV63(StrictModel):
    """Harness-authenticated, append-only coordinator operation intent."""

    schema_version: Literal["6.3-external-qualification-operation-intent"] = (
        "6.3-external-qualification-operation-intent"
    )
    operation_id: Identifier
    qualification_id: Identifier
    operation_type: CoordinatorOperationV63
    request_hash: Sha256
    expected_state_hash: Sha256
    expected_phase: ExternalQualificationPhaseV63
    actor: CoordinatorActorV63
    started_at: datetime
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    intent_hash: Sha256 | None = None
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_intent(self) -> "ExternalQualificationOperationIntentV63":
        _assert_aware(self.started_at, "started_at")
        if self.authority_auth_tag and not self.intent_hash:
            raise ValueError("authenticated operation intent lacks a hash")
        if self.intent_hash and self.intent_hash != self.content_hash():
            raise ValueError("operation intent hash differs")
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
            raise ValueError("operation intent is not sealed")


class ExternalQualificationOperationReceiptV63(StrictModel):
    """Harness-authenticated completion of one exact intent."""

    schema_version: Literal["6.3-external-qualification-operation-receipt"] = (
        "6.3-external-qualification-operation-receipt"
    )
    operation_id: Identifier
    qualification_id: Identifier
    operation_type: CoordinatorOperationV63
    intent_hash: Sha256
    request_hash: Sha256
    expected_state_hash: Sha256
    action_state_hash: Sha256
    result_phase: ExternalQualificationPhaseV63
    output_artifact_hashes: dict[Identifier, Sha256]
    completed_at: datetime
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    receipt_hash: Sha256 | None = None
    v63_protocol_qualification_granted: bool = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_receipt(self) -> "ExternalQualificationOperationReceiptV63":
        _assert_aware(self.completed_at, "completed_at")
        if self.v63_protocol_qualification_granted != (
            self.result_phase == "EXTERNALLY_QUALIFIED"
        ):
            raise ValueError("operation receipt protocol flag differs from phase")
        if self.authority_auth_tag and not self.receipt_hash:
            raise ValueError("authenticated operation receipt lacks a hash")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("operation receipt hash differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "authority_auth_tag", "receipt_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if (
            not self.authority_auth_tag
            or not self.receipt_hash
            or self.receipt_hash != self.content_hash()
        ):
            raise ValueError("operation receipt is not sealed")


class ExternalEvaluationDispatchPacketV63(StrictModel):
    """Local dispatch description; creating it performs no external call."""

    schema_version: Literal["6.3-external-evaluation-dispatch-packet"] = (
        "6.3-external-evaluation-dispatch-packet"
    )
    qualification_id: Identifier
    task_id: Identifier
    graph_id: Identifier
    contract_hash: Sha256
    forecast_input_hash: Sha256
    custody_hash: Sha256
    registration_hash: Sha256
    prediction_seal_hash: Sha256
    reservation_hash: Sha256
    reservation_artifact_hash: Sha256
    prediction_artifact_hash: Sha256
    evaluator_key_id: Identifier
    evaluator_host_id: Identifier
    issued_at: datetime
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    packet_hash: Sha256 | None = None
    private_target_values_included: Literal[False] = False
    external_call_performed: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_packet(self) -> "ExternalEvaluationDispatchPacketV63":
        _assert_aware(self.issued_at, "issued_at")
        if self.authority_auth_tag and not self.packet_hash:
            raise ValueError("authenticated dispatch packet lacks a hash")
        if self.packet_hash and self.packet_hash != self.content_hash():
            raise ValueError("evaluation dispatch packet hash differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "authority_auth_tag", "packet_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "packet_hash")

    def assert_sealed(self) -> None:
        if (
            not self.authority_auth_tag
            or not self.packet_hash
            or self.packet_hash != self.content_hash()
        ):
            raise ValueError("evaluation dispatch packet is not sealed")


class ExternalQualificationOperationResultV63(StrictModel):
    """Safe caller view of a completed or resumed coordinator operation."""

    schema_version: Literal["6.3-external-qualification-operation-result"] = (
        "6.3-external-qualification-operation-result"
    )
    operation_id: Identifier
    operation_type: CoordinatorOperationV63
    request_hash: Sha256
    resumed: bool
    operation_receipt_hash: Sha256
    output_artifact_hashes: dict[Identifier, Sha256]
    dispatch_packet: ExternalEvaluationDispatchPacketV63 | None = None
    state: ExternalQualificationStateV63
    v63_protocol_qualification_granted: bool
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_result(self) -> "ExternalQualificationOperationResultV63":
        if (
            self.v63_protocol_qualification_granted
            != self.state.v63_protocol_qualification_granted
        ):
            raise ValueError("operation result protocol flag differs from state")
        if (self.dispatch_packet is not None) != (
            self.operation_type == "reserve_evaluation"
        ):
            raise ValueError("dispatch packet is only valid for reservation")
        return self


class ExternalQualificationVerificationV63(StrictModel):
    """Read-only replay result for the projected phase."""

    schema_version: Literal["6.3-external-qualification-verification"] = (
        "6.3-external-qualification-verification"
    )
    qualification_id: Identifier
    state_hash: Sha256
    phase: ExternalQualificationPhaseV63
    status: Literal["PASS", "FAIL", "NOT_RUN"]
    checks: dict[Identifier, bool]
    reason_codes: list[Identifier]
    prediction_runtime_replay_hash: Sha256 | None = None
    qualification_replay_hash: Sha256 | None = None
    v63_protocol_qualification_granted: bool
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    verification_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_verification(self) -> "ExternalQualificationVerificationV63":
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("verification reasons must be sorted and unique")
        if self.status == "PASS" and (
            not self.checks or not all(self.checks.values())
        ):
            raise ValueError("PASS verification has a failing check")
        if self.status == "FAIL" and (
            not self.checks or all(self.checks.values())
        ):
            raise ValueError("FAIL verification lacks a failing check")
        if self.v63_protocol_qualification_granted != (
            self.status == "PASS"
            and self.phase == "EXTERNALLY_QUALIFIED"
        ):
            raise ValueError("verification protocol flag differs")
        if self.verification_hash and (
            self.verification_hash != self.content_hash()
        ):
            raise ValueError("qualification verification hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "verification_hash")

    @classmethod
    def seal(cls, **data: object) -> "ExternalQualificationVerificationV63":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"verification_hash"})
        payload["verification_hash"] = draft.content_hash()
        return cls(**payload)


@dataclass(frozen=True)
class _LedgerItem:
    event_sequence: int
    event_hash: str
    reference: ArtifactRef
    payload: Any


@dataclass(frozen=True)
class _ProjectedLedger:
    events: list[dict[str, object]]
    items: list[_LedgerItem]
    qualification_id: str
    task_id: str | None
    by_role: dict[str, list[_LedgerItem]]
    authority_artifacts: list[ExternalQualificationAuthorityArtifactV63]
    operation_intents: dict[
        str,
        tuple[_LedgerItem, ExternalQualificationOperationIntentV63],
    ]
    operation_receipts: dict[
        str,
        tuple[_LedgerItem, ExternalQualificationOperationReceiptV63],
    ]
    pending_intent: ExternalQualificationOperationIntentV63 | None
    reason_codes: list[str]
    diagnostics: list[str]


_MODEL_BY_KIND: dict[str, type[StrictModel]] = {
    CONTRACT_KIND_V63: PredictiveExternalQualificationContractV63,
    FORECAST_INPUT_KIND_V63: ExternalForecastInputV63,
    CUSTODY_KIND_V63: ExternalEvidenceCustodyV63,
    CUSTODY_ADMISSION_KIND_V63: ExternalCustodyAdmissionV63,
    PREDICTION_VECTOR_KIND_V63: ExternalPredictionVectorV63,
    PREDICTION_BINDING_KIND_V63: CurrentModelPredictionBindingV63,
    REGISTRATION_KIND_V63: ExternalPredictionRegistrationV63,
    PREDICTION_SEAL_KIND_V50: PredictionSealV50,
    RESERVATION_KIND_V63: ExternalEvaluationReservationV63,
    EVALUATION_KIND_V63: ExternalAggregateEvaluationV63,
    CONSUMPTION_KIND_V63: ExternalEvaluationConsumptionV63,
    PROMOTION_KIND_V63: ExternalPredictivePromotionV63,
    QUALIFICATION_KIND_V63: ExternalPredictiveQualificationReceiptV63,
    OPERATION_INTENT_KIND_V63: ExternalQualificationOperationIntentV63,
    OPERATION_RECEIPT_KIND_V63: ExternalQualificationOperationReceiptV63,
    DISPATCH_PACKET_KIND_V63: ExternalEvaluationDispatchPacketV63,
}

_EXPECTED_KIND_BY_SCHEMA = {
    "6.3-predictive-qualification-contract": CONTRACT_KIND_V63,
    "6.3-external-forecast-input": FORECAST_INPUT_KIND_V63,
    "6.3-external-evidence-custody": CUSTODY_KIND_V63,
    "6.3-external-custody-admission": CUSTODY_ADMISSION_KIND_V63,
    "6.3-external-prediction-vector": PREDICTION_VECTOR_KIND_V63,
    "6.3-current-model-prediction-binding": PREDICTION_BINDING_KIND_V63,
    "6.3-external-prediction-registration": REGISTRATION_KIND_V63,
    "5.0-prediction-seal": PREDICTION_SEAL_KIND_V50,
    "6.3-external-evaluation-reservation": RESERVATION_KIND_V63,
    "6.3-external-aggregate-evaluation": EVALUATION_KIND_V63,
    "6.3-external-evaluation-consumption": CONSUMPTION_KIND_V63,
    "6.3-external-predictive-promotion": PROMOTION_KIND_V63,
    "6.3-external-predictive-qualification": QUALIFICATION_KIND_V63,
    "6.3-external-qualification-operation-intent": (
        OPERATION_INTENT_KIND_V63
    ),
    "6.3-external-qualification-operation-receipt": (
        OPERATION_RECEIPT_KIND_V63
    ),
    "6.3-external-evaluation-dispatch-packet": DISPATCH_PACKET_KIND_V63,
}

_ROLE_BY_KIND = {
    CONTRACT_KIND_V63: "contract",
    FORECAST_INPUT_KIND_V63: "forecast_input",
    CUSTODY_KIND_V63: "custody",
    CUSTODY_ADMISSION_KIND_V63: "custody_admission",
    PREDICTION_VECTOR_KIND_V63: "prediction_vector",
    PREDICTION_BINDING_KIND_V63: "prediction_binding",
    REGISTRATION_KIND_V63: "registration",
    PREDICTION_SEAL_KIND_V50: "prediction_seal",
    RESERVATION_KIND_V63: "evaluation_reservation",
    EVALUATION_KIND_V63: "evaluation",
    CONSUMPTION_KIND_V63: "evaluation_consumption",
    PROMOTION_KIND_V63: "promotion",
    DISPATCH_PACKET_KIND_V63: "dispatch_packet",
    OPERATION_INTENT_KIND_V63: "operation_intent",
    OPERATION_RECEIPT_KIND_V63: "operation_receipt",
}


def _read_ledger(workspace: StageWorkspaceV50) -> tuple[
    list[dict[str, object]], list[_LedgerItem]
]:
    if not workspace.verify():
        raise ExternalQualificationCoordinatorError(
            "workspace verification failed before state projection"
        )
    try:
        events = workspace.graph._read_events(workspace.graph.store)
        items: list[_LedgerItem] = []
        for event in events:
            if event.get("event_type") != "artifact_committed":
                continue
            reference = ArtifactRef.model_validate(event.get("payload"))
            payload = workspace.graph.store.load_artifact(reference)
            items.append(
                _LedgerItem(
                    event_sequence=int(event["sequence"]),
                    event_hash=str(event["event_hash"]),
                    reference=reference,
                    payload=payload,
                )
            )
        return events, items
    except (
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise ExternalQualificationCoordinatorError(
            "graph authority ledger could not be replayed"
        ) from exc


def _payload_qualification_id(payload: Any) -> str | None:
    if isinstance(payload, StrictModel):
        value = getattr(payload, "qualification_id", None)
    elif isinstance(payload, dict):
        value = payload.get("qualification_id")
    else:
        value = None
    return value if isinstance(value, str) else None


def _payload_task_id(payload: Any) -> str | None:
    if isinstance(payload, StrictModel):
        value = getattr(payload, "task_id", None)
    elif isinstance(payload, dict):
        value = payload.get("task_id")
    else:
        value = None
    return value if isinstance(value, str) else None


def _operation_ledger(
    workspace: StageWorkspaceV50,
    items: list[_LedgerItem],
) -> tuple[
    dict[str, tuple[_LedgerItem, ExternalQualificationOperationIntentV63]],
    dict[str, tuple[_LedgerItem, ExternalQualificationOperationReceiptV63]],
]:
    intents: dict[
        str, tuple[_LedgerItem, ExternalQualificationOperationIntentV63]
    ] = {}
    receipts: dict[
        str, tuple[_LedgerItem, ExternalQualificationOperationReceiptV63]
    ] = {}
    for item in items:
        if item.reference.kind == OPERATION_INTENT_KIND_V63:
            try:
                model = ExternalQualificationOperationIntentV63.model_validate(
                    item.payload
                )
                model.assert_sealed()
            except ValueError as exc:
                raise ExternalQualificationCoordinatorError(
                    "operation-intent ledger contains an invalid envelope"
                ) from exc
            if (
                model.authority_key_id != workspace.authority_key_id
                or not workspace._verify_mac(
                    OPERATION_INTENT_KIND_V63,
                    model.unsigned_hash(),
                    model.authority_auth_tag,
                )
            ):
                raise ExternalQualificationCoordinatorError(
                    "operation-intent authority authentication failed"
                )
            if model.operation_id in intents:
                raise ExternalQualificationCoordinatorError(
                    "operation-intent ledger contains duplicates"
                )
            intents[model.operation_id] = (item, model)
        elif item.reference.kind == OPERATION_RECEIPT_KIND_V63:
            try:
                model = ExternalQualificationOperationReceiptV63.model_validate(
                    item.payload
                )
                model.assert_sealed()
            except ValueError as exc:
                raise ExternalQualificationCoordinatorError(
                    "operation-receipt ledger contains an invalid envelope"
                ) from exc
            if (
                model.authority_key_id != workspace.authority_key_id
                or not workspace._verify_mac(
                    OPERATION_RECEIPT_KIND_V63,
                    model.unsigned_hash(),
                    model.authority_auth_tag,
                )
            ):
                raise ExternalQualificationCoordinatorError(
                    "operation-receipt authority authentication failed"
                )
            if model.operation_id in receipts:
                raise ExternalQualificationCoordinatorError(
                    "operation-receipt ledger contains duplicates"
                )
            receipts[model.operation_id] = (item, model)
    for operation_id, (receipt_item, receipt) in receipts.items():
        prior = intents.get(operation_id)
        if prior is None:
            raise ExternalQualificationCoordinatorError(
                "operation receipt has no matching intent"
            )
        intent_item, intent = prior
        if (
            receipt_item.event_sequence <= intent_item.event_sequence
            or receipt.completed_at < intent.started_at
            or receipt.qualification_id != intent.qualification_id
            or receipt.operation_type != intent.operation_type
            or receipt.intent_hash != intent.intent_hash
            or receipt.request_hash != intent.request_hash
            or receipt.expected_state_hash != intent.expected_state_hash
        ):
            raise ExternalQualificationCoordinatorError(
                "operation receipt differs from its intent"
            )
    return intents, receipts


def _project_ledger(
    workspace: StageWorkspaceV50,
    *,
    qualification_id: str | None,
) -> _ProjectedLedger:
    events, items = _read_ledger(workspace)
    reasons: list[str] = []
    diagnostics: list[str] = []

    contract_items = [
        item for item in items if item.reference.kind == CONTRACT_KIND_V63
    ]
    parsed_contracts: list[
        tuple[_LedgerItem, PredictiveExternalQualificationContractV63]
    ] = []
    for item in contract_items:
        try:
            parsed_contracts.append(
                (
                    item,
                    PredictiveExternalQualificationContractV63.model_validate(
                        item.payload
                    ),
                )
            )
        except ValueError as exc:
            reasons.append("invalid_contract_envelope")
            diagnostics.append(str(exc))
    if qualification_id is None:
        identities = sorted(
            {
                contract.qualification_id
                for _item, contract in parsed_contracts
            }
        )
        if len(identities) > 1:
            reasons.append("ambiguous_qualification_identity")
            qualification_id = identities[0]
        elif identities:
            qualification_id = identities[0]
        else:
            qualification_id = f"local.{workspace.spec.workspace_id}"

    selected_contracts = [
        (item, contract)
        for item, contract in parsed_contracts
        if contract.qualification_id == qualification_id
    ]
    if len(selected_contracts) > 1:
        reasons.append("duplicate_contract")
    contract = selected_contracts[0][1] if selected_contracts else None
    task_id = contract.task_id if contract is not None else None

    by_role: dict[str, list[_LedgerItem]] = {}
    selected_items: list[_LedgerItem] = []
    parsed_by_item: dict[int, Any] = {}

    for item in items:
        payload = item.payload
        schema_version = (
            payload.get("schema_version")
            if isinstance(payload, dict)
            else None
        )
        expected_kind = (
            _EXPECTED_KIND_BY_SCHEMA.get(schema_version)
            if isinstance(schema_version, str)
            else None
        )
        payload_qualification = _payload_qualification_id(payload)
        payload_task = _payload_task_id(payload)
        if payload_qualification is not None:
            belongs = payload_qualification == qualification_id and (
                task_id is None
                or payload_task is None
                or payload_task == task_id
            )
        else:
            belongs = (
                expected_kind == PREDICTION_SEAL_KIND_V50
                and task_id is not None
                and payload_task == task_id
            )
        if expected_kind is not None and belongs:
            if item.reference.kind != expected_kind:
                reasons.append("wrong_artifact_kind")
                diagnostics.append(
                    f"{schema_version} committed as {item.reference.kind}"
                )
                selected_items.append(item)
                by_role.setdefault("wrong_kind", []).append(item)
                continue
            model_type = _MODEL_BY_KIND[expected_kind]
            try:
                parsed = model_type.model_validate(payload)
                parsed_by_item[item.event_sequence] = parsed
            except ValueError as exc:
                reasons.append("invalid_authority_envelope")
                diagnostics.append(f"{expected_kind}: {exc}")
                parsed = payload
            role = _ROLE_BY_KIND.get(expected_kind, "authority_artifact")
            if expected_kind == QUALIFICATION_KIND_V63 and isinstance(
                parsed, ExternalPredictiveQualificationReceiptV63
            ):
                role = (
                    "qualification_not_run"
                    if parsed.status == "NOT_RUN"
                    else "qualification_final"
                )
            selected_items.append(item)
            by_role.setdefault(role, []).append(item)

    # A prediction seal has no qualification_id, so bind it through task_id.
    if task_id is not None:
        for item in items:
            if item.reference.kind != PREDICTION_SEAL_KIND_V50:
                continue
            try:
                seal = PredictionSealV50.model_validate(item.payload)
            except ValueError as exc:
                reasons.append("invalid_prediction_seal")
                diagnostics.append(str(exc))
                continue
            if seal.task_id != task_id:
                continue
            parsed_by_item[item.event_sequence] = seal
            if item not in selected_items:
                selected_items.append(item)
                by_role.setdefault("prediction_seal", []).append(item)

    # Bind the generator receipt and its trace through the prediction binding.
    bindings = by_role.get("prediction_binding", [])
    binding_model = (
        parsed_by_item.get(bindings[0].event_sequence)
        if len(bindings) == 1
        else None
    )
    receipt_hash = (
        binding_model.generator_execution_receipt_hash
        if isinstance(binding_model, CurrentModelPredictionBindingV63)
        else None
    )
    role_receipt_items: list[_LedgerItem] = []
    for item in items:
        if item.reference.kind != "role_execution_receipt_v50":
            continue
        try:
            receipt = RoleExecutionReceiptV50.model_validate(item.payload)
        except ValueError:
            continue
        if (
            receipt.receipt_hash == receipt_hash
            or (
                receipt_hash is None
                and task_id is not None
                and receipt.subject_id == task_id
                and receipt.model == PREDICTION_RUNTIME_ADAPTER_ID_V63
            )
        ):
            parsed_by_item[item.event_sequence] = receipt
            role_receipt_items.append(item)
    if role_receipt_items:
        by_role["generator_receipt"] = role_receipt_items
        selected_items.extend(
            item for item in role_receipt_items if item not in selected_items
        )
        for receipt_item in role_receipt_items:
            receipt = parsed_by_item[receipt_item.event_sequence]
            for item in items:
                if item.reference.sha256 == receipt.transport_trace_hash:
                    by_role.setdefault("prediction_trace", []).append(item)
                    selected_items.append(item)

    # A crashed runtime may have a trace before the binding exists.
    for item in items:
        if item.reference.kind != PREDICTION_TRACE_KIND_V63:
            continue
        if (
            isinstance(item.payload, dict)
            and item.payload.get("qualification_id") == qualification_id
        ):
            by_role.setdefault("prediction_trace", []).append(item)
            selected_items.append(item)

    try:
        intents, receipts = _operation_ledger(workspace, items)
    except ExternalQualificationCoordinatorError as exc:
        reasons.append("invalid_operation_ledger")
        diagnostics.append(str(exc))
        intents, receipts = {}, {}
    pending = [
        intent
        for operation_id, (_item, intent) in intents.items()
        if operation_id not in receipts
        and intent.qualification_id == qualification_id
    ]
    if len(pending) > 1:
        reasons.append("multiple_pending_operations")
    pending_intent = pending[0] if len(pending) == 1 else None
    for item, intent in intents.values():
        if intent.qualification_id == qualification_id:
            selected_items.append(item)
            by_role.setdefault("operation_intent", []).append(item)
    for item, receipt in receipts.values():
        if receipt.qualification_id == qualification_id:
            selected_items.append(item)
            by_role.setdefault("operation_receipt", []).append(item)

    # A repeated commit event is still a duplicate authority statement.
    selected_items = sorted(
        selected_items,
        key=lambda item: item.event_sequence,
    )
    seen_sequences: set[int] = set()
    unique_selected: list[_LedgerItem] = []
    for item in selected_items:
        if item.event_sequence in seen_sequences:
            continue
        seen_sequences.add(item.event_sequence)
        unique_selected.append(item)
    selected_items = unique_selected

    singleton_roles = {
        "contract",
        "forecast_input",
        "custody",
        "custody_admission",
        "prediction_vector",
        "prediction_trace",
        "generator_receipt",
        "prediction_binding",
        "registration",
        "prediction_seal",
        "evaluation_reservation",
        "dispatch_packet",
        "evaluation",
        "evaluation_consumption",
        "promotion",
    }
    for role in singleton_roles:
        role_items = by_role.get(role, [])
        if len({item.event_sequence for item in role_items}) > 1:
            reasons.append(f"duplicate_{role}")

    not_run = by_role.get("qualification_not_run", [])
    terminal = by_role.get("qualification_final", [])
    if len(not_run) > 1:
        reasons.append("duplicate_qualification_not_run")
    if len(terminal) > 1:
        reasons.append("duplicate_qualification_final")

    authority_artifacts: list[
        ExternalQualificationAuthorityArtifactV63
    ] = []
    for item in selected_items:
        role = next(
            (
                candidate
                for candidate, role_items in by_role.items()
                if item in role_items
            ),
            "authority_artifact",
        )
        authority_artifacts.append(
            ExternalQualificationAuthorityArtifactV63(
                event_sequence=item.event_sequence,
                role=role,
                kind=item.reference.kind,
                artifact_hash=item.reference.sha256,
            )
        )
    return _ProjectedLedger(
        events=events,
        items=items,
        qualification_id=qualification_id,
        task_id=task_id,
        by_role=by_role,
        authority_artifacts=authority_artifacts,
        operation_intents=intents,
        operation_receipts=receipts,
        pending_intent=pending_intent,
        reason_codes=sorted(set(reasons)),
        diagnostics=diagnostics,
    )


def _typed_one(
    workspace: StageWorkspaceV50,
    *,
    kind: str,
    model_type: type[Any],
    qualification_id: str,
    task_id: str | None = None,
    required: bool = True,
) -> Any | None:
    try:
        matches = [
            item
            for _reference, item in workspace._artifacts_of_kind(
                kind, model_type
            )
            if (
                (
                    getattr(item, "qualification_id", None)
                    == qualification_id
                    and (
                        task_id is None
                        or getattr(item, "task_id", None) is None
                        or getattr(item, "task_id", None) == task_id
                    )
                )
                if getattr(item, "qualification_id", None) is not None
                else (
                    kind == PREDICTION_SEAL_KIND_V50
                    and task_id is not None
                    and getattr(item, "task_id", None) == task_id
                )
            )
        ]
    except (
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise ExternalQualificationCoordinatorError(
            f"{kind} could not be loaded from the authority ledger"
        ) from exc
    if not matches and not required:
        return None
    if len(matches) != 1:
        raise ExternalQualificationCoordinatorError(
            f"{kind} is absent, duplicated, or ambiguous"
        )
    return matches[0]


def _artifact_hash_for_model(
    workspace: StageWorkspaceV50,
    *,
    kind: str,
    model_type: type[Any],
    model: Any,
) -> str:
    try:
        hashes = [
            reference.sha256
            for reference, item in workspace._artifacts_of_kind(
                kind, model_type
            )
            if item == model
        ]
    except (
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise ExternalQualificationCoordinatorError(
            f"{kind} artifact hash could not be replayed"
        ) from exc
    if len(hashes) != 1:
        raise ExternalQualificationCoordinatorError(
            f"{kind} exact artifact is absent or duplicated"
        )
    return hashes[0]


def _validate_role_order(ledger: _ProjectedLedger) -> list[str]:
    reasons: list[str] = []
    ranks = {
        "contract": 10,
        "forecast_input": 20,
        "custody": 30,
        "custody_admission": 40,
        "prediction_vector": 50,
        "prediction_trace": 55,
        "generator_receipt": 60,
        "prediction_binding": 70,
        "registration": 80,
        "prediction_seal": 90,
        "evaluation_reservation": 100,
        "dispatch_packet": 105,
        "evaluation": 110,
        "evaluation_consumption": 120,
        "qualification_not_run": 130,
        "promotion": 140,
        "qualification_final": 150,
    }
    ranked: list[tuple[int, int]] = []
    for role, role_items in ledger.by_role.items():
        if role not in ranks:
            continue
        ranked.extend(
            (item.event_sequence, ranks[role]) for item in role_items
        )
    ordered_ranks = [rank for _sequence, rank in sorted(ranked)]
    if ordered_ranks != sorted(ordered_ranks):
        reasons.append("authority_artifacts_out_of_order")

    dependencies = {
        "forecast_input": {"contract"},
        "custody": {"contract", "forecast_input"},
        "custody_admission": {"custody"},
        "prediction_vector": {"custody_admission"},
        "prediction_trace": {"prediction_vector"},
        "generator_receipt": {"prediction_vector", "prediction_trace"},
        "prediction_binding": {"prediction_vector", "generator_receipt"},
        "registration": {"prediction_binding"},
        "prediction_seal": {"registration"},
        "evaluation_reservation": {"prediction_seal"},
        "dispatch_packet": {"evaluation_reservation"},
        "evaluation": {"evaluation_reservation"},
        "evaluation_consumption": {"evaluation"},
        "qualification_not_run": {"evaluation_consumption"},
        "promotion": {"evaluation_consumption"},
        "qualification_final": {"promotion", "evaluation_consumption"},
    }
    present = {
        role for role, role_items in ledger.by_role.items() if role_items
    }
    for role, required in dependencies.items():
        if role in present and not required.issubset(present):
            reasons.append(f"{role}_missing_predecessor")
    return sorted(set(reasons))


def _coordinator_operation_status(
    ledger: _ProjectedLedger,
    *,
    operation_type: CoordinatorOperationV63,
    expected_result_phase: ExternalQualificationPhaseV63,
    output_roles: Mapping[str, str],
) -> Literal["PENDING", "COMPLETED"]:
    matching = [
        (operation_id, item, intent)
        for operation_id, (item, intent) in ledger.operation_intents.items()
        if intent.qualification_id == ledger.qualification_id
        and intent.operation_type == operation_type
    ]
    if len(matching) != 1:
        raise ExternalQualificationCoordinatorError(
            f"{operation_type} lacks one exact coordinator intent"
        )
    operation_id, intent_item, _intent = matching[0]
    receipt_entry = ledger.operation_receipts.get(operation_id)
    if receipt_entry is None:
        return "PENDING"

    receipt_item, receipt = receipt_entry
    if (
        receipt.result_phase != expected_result_phase
        or set(receipt.output_artifact_hashes) != set(output_roles)
    ):
        raise ExternalQualificationCoordinatorError(
            f"{operation_type} receipt has an invalid result contract"
        )
    for output_name, role in output_roles.items():
        role_items = ledger.by_role.get(role, [])
        if len(role_items) != 1:
            raise ExternalQualificationCoordinatorError(
                f"{operation_type} output {output_name} is ambiguous"
            )
        artifact = role_items[0]
        if (
            receipt.output_artifact_hashes[output_name]
            != artifact.reference.sha256
            or artifact.event_sequence <= intent_item.event_sequence
            or artifact.event_sequence >= receipt_item.event_sequence
        ):
            raise ExternalQualificationCoordinatorError(
                f"{operation_type} output {output_name} differs from ledger"
            )
    return "COMPLETED"


def _state_from_ledger(
    workspace: StageWorkspaceV50,
    ledger: _ProjectedLedger,
    *,
    phase: ExternalQualificationPhaseV63,
    reason_codes: list[str],
    diagnostics: list[str],
) -> ExternalQualificationStateV63:
    event_tip = str(ledger.events[-1]["event_hash"])
    qualified = phase == "EXTERNALLY_QUALIFIED"
    return ExternalQualificationStateV63.seal(
        qualification_id=ledger.qualification_id,
        task_id=ledger.task_id,
        workspace_spec_hash=workspace.spec.spec_hash,
        phase=phase,
        current=phase not in {"STALE", "INCONSISTENT"},
        terminal=phase in _TERMINAL_PHASES,
        graph_event_sequence=int(ledger.events[-1]["sequence"]),
        graph_event_tip=event_tip,
        authority_artifacts=ledger.authority_artifacts,
        ordered_authority_artifact_hashes=[
            item.artifact_hash for item in ledger.authority_artifacts
        ],
        pending_operation_id=(
            ledger.pending_intent.operation_id
            if ledger.pending_intent is not None
            else None
        ),
        reason_codes=sorted(set(reason_codes)),
        diagnostics=diagnostics,
        next_valid_actions=_NEXT_ACTIONS[phase],
        v63_protocol_qualification_granted=qualified,
    )


def _project_external_qualification_state_locked_v63(
    workspace: StageWorkspaceV50,
    *,
    qualification_id: str | None,
    trusted_public_keys: Mapping[str, bytes],
) -> ExternalQualificationStateV63:
    ledger = _project_ledger(
        workspace,
        qualification_id=qualification_id,
    )
    reasons = list(ledger.reason_codes)
    diagnostics = list(ledger.diagnostics)
    reasons.extend(_validate_role_order(ledger))
    if reasons:
        return _state_from_ledger(
            workspace,
            ledger,
            phase="INCONSISTENT",
            reason_codes=reasons,
            diagnostics=diagnostics,
        )

    roles = ledger.by_role
    if not roles.get("contract"):
        # Any V6.3 authority artifact without its contract is inconsistent.
        orphan_roles = {
            role
            for role, items in roles.items()
            if items and role not in {"operation_intent", "operation_receipt"}
        }
        if orphan_roles:
            return _state_from_ledger(
                workspace,
                ledger,
                phase="INCONSISTENT",
                reason_codes=["contract_missing"],
                diagnostics=sorted(orphan_roles),
            )
        try:
            qualification.derive_predictive_local_context_v63(workspace)
        except ExternalQualificationError as exc:
            return _state_from_ledger(
                workspace,
                ledger,
                phase="INCONSISTENT",
                reason_codes=["local_prerequisites_unavailable"],
                diagnostics=[str(exc)],
            )
        return _state_from_ledger(
            workspace,
            ledger,
            phase="LOCAL_READY",
            reason_codes=[],
            diagnostics=[],
        )

    try:
        contract = _typed_one(
            workspace,
            kind=CONTRACT_KIND_V63,
            model_type=PredictiveExternalQualificationContractV63,
            qualification_id=ledger.qualification_id,
        )
        assert isinstance(
            contract, PredictiveExternalQualificationContractV63
        )
        contract.assert_sealed()
        if trusted_public_keys:
            qualification._assert_trusted_authority_set(
                contract=contract,
                trusted_public_keys=trusted_public_keys,
            )
        try:
            qualification._assert_contract_current(
                workspace=workspace,
                contract=contract,
            )
        except ExternalQualificationError as exc:
            return _state_from_ledger(
                workspace,
                ledger,
                phase="STALE",
                reason_codes=["contract_stale"],
                diagnostics=[str(exc)],
            )

        forecast_input = _typed_one(
            workspace,
            kind=FORECAST_INPUT_KIND_V63,
            model_type=ExternalForecastInputV63,
            qualification_id=ledger.qualification_id,
            task_id=contract.task_id,
            required=False,
        )
        if forecast_input is None:
            return _state_from_ledger(
                workspace,
                ledger,
                phase="CONTRACT_FROZEN",
                reason_codes=[],
                diagnostics=[],
            )
        qualification._current_external_forecast_input(
            workspace=workspace,
            contract=contract,
        )

        custody = _typed_one(
            workspace,
            kind=CUSTODY_KIND_V63,
            model_type=ExternalEvidenceCustodyV63,
            qualification_id=ledger.qualification_id,
            required=False,
        )
        admission = _typed_one(
            workspace,
            kind=CUSTODY_ADMISSION_KIND_V63,
            model_type=ExternalCustodyAdmissionV63,
            qualification_id=ledger.qualification_id,
            required=False,
        )
        if custody is None and admission is None:
            return _state_from_ledger(
                workspace,
                ledger,
                phase="INPUT_FROZEN",
                reason_codes=[],
                diagnostics=[],
            )
        if custody is None:
            raise ExternalQualificationCoordinatorError(
                "custody admission exists without its custody envelope"
            )
        assert isinstance(custody, ExternalEvidenceCustodyV63)
        if not trusted_public_keys:
            raise ExternalQualificationCoordinatorError(
                "trusted public keys are required to replay signed custody"
            )
        qualification._require_signed(
            model=custody,
            key_id=custody.custody_key_id,
            signature_base64=custody.signature_base64,
            trusted_public_keys=trusted_public_keys,
            hash_field="custody_hash",
            label="custody",
        )
        expected_custody_reasons = sorted(
            set(
                qualification._custody_reason_codes(
                    contract=contract,
                    custody=custody,
                )
                + qualification._external_forecast_input_reason_codes(
                    workspace=workspace,
                    contract=contract,
                    custody=custody,
                )
            )
        )
        if admission is None:
            return _state_from_ledger(
                workspace,
                ledger,
                phase="CUSTODY_COMMITTED",
                reason_codes=["custody_admission_pending"],
                diagnostics=expected_custody_reasons,
            )
        assert isinstance(admission, ExternalCustodyAdmissionV63)
        if (
            admission.contract_hash != contract.contract_hash
            or admission.custody_hash != custody.custody_hash
            or admission.status
            != ("REJECTED" if expected_custody_reasons else "VERIFIED")
            or admission.reason_codes != expected_custody_reasons
        ):
            raise ExternalQualificationCoordinatorError(
                "custody admission differs from code-derived result"
            )
        if admission.status == "REJECTED":
            return _state_from_ledger(
                workspace,
                ledger,
                phase="CUSTODY_REJECTED",
                reason_codes=admission.reason_codes,
                diagnostics=[],
            )
        qualification._verified_custody_admission(
            workspace=workspace,
            contract=contract,
            custody=custody,
        )

        binding = _typed_one(
            workspace,
            kind=PREDICTION_BINDING_KIND_V63,
            model_type=CurrentModelPredictionBindingV63,
            qualification_id=ledger.qualification_id,
            required=False,
        )
        runtime_prefix = any(
            roles.get(role)
            for role in (
                "prediction_vector",
                "prediction_trace",
                "generator_receipt",
            )
        )
        if binding is None:
            if runtime_prefix:
                pending = ledger.pending_intent
                if not (
                    pending is not None
                    and pending.operation_type == "run_prediction"
                ):
                    raise ExternalQualificationCoordinatorError(
                        "prediction runtime prefix lacks a binding"
                    )
            return _state_from_ledger(
                workspace,
                ledger,
                phase="CUSTODY_VERIFIED",
                reason_codes=[],
                diagnostics=(
                    ["prediction operation is pending"]
                    if runtime_prefix
                    else []
                ),
            )
        assert isinstance(binding, CurrentModelPredictionBindingV63)
        qualification._assert_current_model_prediction_binding(
            workspace=workspace,
            contract=contract,
            custody=custody,
            binding=binding,
        )
        replayed_prediction = verify_current_model_external_prediction_v63(
            workspace=workspace,
            contract=contract,
            forecast_input=forecast_input,
            custody=custody,
        )
        if replayed_prediction.binding != binding:
            raise ExternalQualificationCoordinatorError(
                "current-model prediction differs from numeric replay"
            )
        _coordinator_operation_status(
            ledger,
            operation_type="run_prediction",
            expected_result_phase="PREDICTION_BOUND",
            output_roles={
                "prediction_vector": "prediction_vector",
                "generator_execution_receipt": "generator_receipt",
                "prediction_binding": "prediction_binding",
            },
        )

        registration = _typed_one(
            workspace,
            kind=REGISTRATION_KIND_V63,
            model_type=ExternalPredictionRegistrationV63,
            qualification_id=ledger.qualification_id,
            task_id=contract.task_id,
            required=False,
        )
        prediction_seal = _typed_one(
            workspace,
            kind=PREDICTION_SEAL_KIND_V50,
            model_type=PredictionSealV50,
            qualification_id=ledger.qualification_id,
            task_id=contract.task_id,
            required=False,
        )
        if registration is None and prediction_seal is None:
            return _state_from_ledger(
                workspace,
                ledger,
                phase="PREDICTION_BOUND",
                reason_codes=[],
                diagnostics=[],
            )
        if registration is None:
            raise ExternalQualificationCoordinatorError(
                "prediction seal exists without registration"
            )
        assert isinstance(registration, ExternalPredictionRegistrationV63)
        qualification._require_signed(
            model=registration,
            key_id=registration.registry_key_id,
            signature_base64=registration.signature_base64,
            trusted_public_keys=trusted_public_keys,
            hash_field="registration_hash",
            label="prediction registration",
        )
        if prediction_seal is None:
            return _state_from_ledger(
                workspace,
                ledger,
                phase="REGISTRATION_COMMITTED",
                reason_codes=[],
                diagnostics=[],
            )
        assert isinstance(prediction_seal, PredictionSealV50)
        qualification._assert_registered_prediction_chain(
            workspace=workspace,
            contract=contract,
            custody=custody,
            prediction_binding=binding,
            registration=registration,
            prediction_seal=prediction_seal,
        )

        reservation = _typed_one(
            workspace,
            kind=RESERVATION_KIND_V63,
            model_type=ExternalEvaluationReservationV63,
            qualification_id=ledger.qualification_id,
            task_id=contract.task_id,
            required=False,
        )
        if reservation is None:
            return _state_from_ledger(
                workspace,
                ledger,
                phase="PREDICTION_REGISTERED",
                reason_codes=[],
                diagnostics=[],
            )
        assert isinstance(reservation, ExternalEvaluationReservationV63)
        qualification._assert_evaluation_reservation(
            workspace=workspace,
            contract=contract,
            custody=custody,
            custody_admission=admission,
            prediction_binding=binding,
            registration=registration,
            prediction_seal=prediction_seal,
            reservation=reservation,
        )
        reservation_operation_status = _coordinator_operation_status(
            ledger,
            operation_type="reserve_evaluation",
            expected_result_phase="EVALUATION_RESERVED",
            output_roles={
                "evaluation_reservation": "evaluation_reservation",
                "dispatch_packet": "dispatch_packet",
            },
        )

        packets = roles.get("dispatch_packet", [])
        if reservation_operation_status == "COMPLETED" and not packets:
            raise ExternalQualificationCoordinatorError(
                "completed evaluation reservation lacks a dispatch packet"
            )
        if packets:
            packet = ExternalEvaluationDispatchPacketV63.model_validate(
                packets[0].payload
            )
            packet.assert_sealed()
            if (
                packet.authority_key_id != workspace.authority_key_id
                or not workspace._verify_mac(
                    DISPATCH_PACKET_KIND_V63,
                    packet.unsigned_hash(),
                    packet.authority_auth_tag,
                )
                or packet.reservation_hash != reservation.reservation_hash
            ):
                raise ExternalQualificationCoordinatorError(
                    "evaluation dispatch packet authority rejected"
                )

        evaluation = _typed_one(
            workspace,
            kind=EVALUATION_KIND_V63,
            model_type=ExternalAggregateEvaluationV63,
            qualification_id=ledger.qualification_id,
            required=False,
        )
        consumption = _typed_one(
            workspace,
            kind=CONSUMPTION_KIND_V63,
            model_type=ExternalEvaluationConsumptionV63,
            qualification_id=ledger.qualification_id,
            task_id=contract.task_id,
            required=False,
        )
        if evaluation is None and consumption is None:
            return _state_from_ledger(
                workspace,
                ledger,
                phase="EVALUATION_RESERVED",
                reason_codes=[],
                diagnostics=[],
            )
        if evaluation is None:
            raise ExternalQualificationCoordinatorError(
                "evaluation consumption exists without its evaluation"
            )
        assert isinstance(evaluation, ExternalAggregateEvaluationV63)
        qualification._require_signed(
            model=evaluation,
            key_id=evaluation.evaluator_key_id,
            signature_base64=evaluation.signature_base64,
            trusted_public_keys=trusted_public_keys,
            hash_field="evaluation_hash",
            label="external evaluation",
        )
        qualification._assert_chain_bindings(
            workspace=workspace,
            contract=contract,
            custody=custody,
            custody_admission=admission,
            prediction_binding=binding,
            registration=registration,
            prediction_seal=prediction_seal,
            reservation=reservation,
            evaluation=evaluation,
        )
        if consumption is None:
            return _state_from_ledger(
                workspace,
                ledger,
                phase="EVALUATION_COMMITTED",
                reason_codes=["evaluation_consumption_pending"],
                diagnostics=[],
            )

        promotions = roles.get("promotion", [])
        terminal_receipts = roles.get("qualification_final", [])
        not_run_receipts = roles.get("qualification_not_run", [])
        promotion = (
            ExternalPredictivePromotionV63.model_validate(
                promotions[0].payload
            )
            if promotions
            else None
        )
        if promotion is None:
            if terminal_receipts:
                raise ExternalQualificationCoordinatorError(
                    "terminal qualification exists without promotion"
                )
            # If a NOT_RUN receipt exists, replay it exactly.  A crash before
            # that receipt remains an explicit pending evaluation operation.
            if not_run_receipts:
                not_run = ExternalPredictiveQualificationReceiptV63.model_validate(
                    not_run_receipts[0].payload
                )
                recomputed = (
                    qualification.assess_external_predictive_qualification_v63(
                        workspace=workspace,
                        contract=contract,
                        custody=custody,
                        prediction_binding=binding,
                        registration=registration,
                        prediction_seal=prediction_seal,
                        reservation=reservation,
                        evaluation=evaluation,
                        promotion=None,
                        trusted_public_keys=trusted_public_keys,
                        _persist=False,
                    )
                )
                if recomputed != not_run:
                    raise ExternalQualificationCoordinatorError(
                        "NOT_RUN qualification receipt differs from replay"
                    )
            return _state_from_ledger(
                workspace,
                ledger,
                phase="AWAITING_PROMOTION",
                reason_codes=["external_promotion_missing"],
                diagnostics=[],
            )

        qualification._require_signed(
            model=promotion,
            key_id=promotion.promotion_key_id,
            signature_base64=promotion.signature_base64,
            trusted_public_keys=trusted_public_keys,
            hash_field="promotion_hash",
            label="external promotion",
        )
        if not terminal_receipts:
            return _state_from_ledger(
                workspace,
                ledger,
                phase="PROMOTION_COMMITTED",
                reason_codes=[],
                diagnostics=[],
            )
        terminal_receipt = (
            ExternalPredictiveQualificationReceiptV63.model_validate(
                terminal_receipts[0].payload
            )
        )
        replay = qualification.verify_external_predictive_qualification_v63(
            workspace=workspace,
            receipt=terminal_receipt,
            trusted_public_keys=trusted_public_keys,
        )
        if replay.status != "PASS":
            raise ExternalQualificationCoordinatorError(
                "terminal qualification receipt failed full replay"
            )
        return _state_from_ledger(
            workspace,
            ledger,
            phase=(
                "EXTERNALLY_QUALIFIED"
                if terminal_receipt.status == "EXTERNALLY_QUALIFIED"
                else "REJECTED"
            ),
            reason_codes=terminal_receipt.reason_codes,
            diagnostics=[],
        )
    except (
        AssertionError,
        ExternalQualificationCoordinatorError,
        ExternalQualificationError,
        StageWorkspaceError,
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        return _state_from_ledger(
            workspace,
            ledger,
            phase="INCONSISTENT",
            reason_codes=["authority_replay_failed"],
            diagnostics=[str(exc)],
        )


def project_external_qualification_state_v63(
    workspace: StageWorkspaceV50,
    *,
    qualification_id: str | None = None,
    trusted_public_keys: Mapping[str, bytes] | None = None,
) -> ExternalQualificationStateV63:
    """Project state without committing, repairing, or writing any artifact."""

    transaction_factory = getattr(
        getattr(getattr(workspace, "graph", None), "store", None),
        "writer_transaction",
        None,
    )
    if not callable(transaction_factory):
        raise ExternalQualificationCoordinatorError(
            "workspace does not expose the graph writer lock"
        )
    with transaction_factory():
        state = _project_external_qualification_state_locked_v63(
            workspace,
            qualification_id=qualification_id,
            trusted_public_keys=dict(trusted_public_keys or {}),
        )
        state.assert_sealed()
        return state


def _operation_request_hash(
    *,
    operation_type: CoordinatorOperationV63,
    qualification_id: str,
    expected_state_hash: str,
    actor: CoordinatorActorV63,
    arguments: Mapping[str, object],
) -> str:
    return sha256_value(
        {
            "schema_version": "6.3-external-qualification-request",
            "operation_type": operation_type,
            "qualification_id": qualification_id,
            "expected_state_hash": expected_state_hash,
            "actor": actor,
            "arguments": dict(arguments),
        }
    )


def _operation_id(
    operation_type: CoordinatorOperationV63,
    request_hash: str,
) -> str:
    return f"v63-{operation_type.replace('_', '-')}-{request_hash[:20]}"


def _seal_intent(
    workspace: StageWorkspaceV50,
    *,
    operation_id: str,
    qualification_id: str,
    operation_type: CoordinatorOperationV63,
    request_hash: str,
    expected_state_hash: str,
    expected_phase: ExternalQualificationPhaseV63,
    actor: CoordinatorActorV63,
) -> ExternalQualificationOperationIntentV63:
    unsigned = ExternalQualificationOperationIntentV63(
        operation_id=operation_id,
        qualification_id=qualification_id,
        operation_type=operation_type,
        request_hash=request_hash,
        expected_state_hash=expected_state_hash,
        expected_phase=expected_phase,
        actor=actor,
        started_at=_utc_now(),
        authority_key_id=workspace.authority_key_id,
    )
    payload = unsigned.model_dump(mode="json")
    payload["authority_auth_tag"] = workspace._mac(
        OPERATION_INTENT_KIND_V63,
        unsigned.unsigned_hash(),
    )
    payload["intent_hash"] = sha256_value(
        {key: value for key, value in payload.items() if key != "intent_hash"}
    )
    return ExternalQualificationOperationIntentV63.model_validate(payload)


def _seal_operation_receipt(
    workspace: StageWorkspaceV50,
    *,
    intent: ExternalQualificationOperationIntentV63,
    action_state: ExternalQualificationStateV63,
    output_artifact_hashes: Mapping[str, str],
) -> ExternalQualificationOperationReceiptV63:
    unsigned = ExternalQualificationOperationReceiptV63(
        operation_id=intent.operation_id,
        qualification_id=intent.qualification_id,
        operation_type=intent.operation_type,
        intent_hash=intent.intent_hash,
        request_hash=intent.request_hash,
        expected_state_hash=intent.expected_state_hash,
        action_state_hash=action_state.state_hash,
        result_phase=action_state.phase,
        output_artifact_hashes=dict(sorted(output_artifact_hashes.items())),
        completed_at=_utc_now(),
        authority_key_id=workspace.authority_key_id,
        v63_protocol_qualification_granted=(
            action_state.phase == "EXTERNALLY_QUALIFIED"
        ),
    )
    payload = unsigned.model_dump(mode="json")
    payload["authority_auth_tag"] = workspace._mac(
        OPERATION_RECEIPT_KIND_V63,
        unsigned.unsigned_hash(),
    )
    payload["receipt_hash"] = sha256_value(
        {key: value for key, value in payload.items() if key != "receipt_hash"}
    )
    return ExternalQualificationOperationReceiptV63.model_validate(payload)


def _dispatch_packet(
    workspace: StageWorkspaceV50,
    *,
    contract: PredictiveExternalQualificationContractV63,
    forecast_input: ExternalForecastInputV63,
    custody: ExternalEvidenceCustodyV63,
    registration: ExternalPredictionRegistrationV63,
    prediction_seal: PredictionSealV50,
    reservation: ExternalEvaluationReservationV63,
    reservation_artifact_hash: str,
) -> ExternalEvaluationDispatchPacketV63:
    unsigned = ExternalEvaluationDispatchPacketV63(
        qualification_id=contract.qualification_id,
        task_id=contract.task_id,
        graph_id=workspace.spec.graph_id,
        contract_hash=contract.contract_hash,
        forecast_input_hash=forecast_input.input_hash,
        custody_hash=custody.custody_hash,
        registration_hash=registration.registration_hash,
        prediction_seal_hash=prediction_seal.seal_hash,
        reservation_hash=reservation.reservation_hash,
        reservation_artifact_hash=reservation_artifact_hash,
        prediction_artifact_hash=reservation.prediction_artifact_hash,
        evaluator_key_id=reservation.evaluator_key_id,
        evaluator_host_id=reservation.evaluator_host_id,
        issued_at=reservation.reserved_at,
        authority_key_id=workspace.authority_key_id,
    )
    payload = unsigned.model_dump(mode="json")
    payload["authority_auth_tag"] = workspace._mac(
        DISPATCH_PACKET_KIND_V63,
        unsigned.unsigned_hash(),
    )
    payload["packet_hash"] = sha256_value(
        {key: value for key, value in payload.items() if key != "packet_hash"}
    )
    return ExternalEvaluationDispatchPacketV63.model_validate(payload)


def _commit_or_recover_dispatch_packet(
    workspace: StageWorkspaceV50,
    packet: ExternalEvaluationDispatchPacketV63,
) -> tuple[ExternalEvaluationDispatchPacketV63, str]:
    try:
        prior = [
            (reference, item)
            for reference, item in workspace._artifacts_of_kind(
                DISPATCH_PACKET_KIND_V63,
                ExternalEvaluationDispatchPacketV63,
            )
            if item.qualification_id == packet.qualification_id
            or item.task_id == packet.task_id
        ]
    except (
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise ExternalQualificationCoordinatorError(
            "evaluation dispatch-packet ledger could not be replayed"
        ) from exc
    if prior:
        if len(prior) != 1 or prior[0][1] != packet:
            raise ExternalQualificationCoordinatorError(
                "evaluation dispatch packet is immutable"
            )
        return prior[0][1], prior[0][0].sha256
    reference = workspace.commit_evidence(
        DISPATCH_PACKET_KIND_V63,
        packet.model_dump(mode="json"),
    )
    return packet, reference.sha256


@dataclass(frozen=True)
class _ActionOutput:
    artifact_hashes: dict[str, str]
    dispatch_packet: ExternalEvaluationDispatchPacketV63 | None = None


class ExternalQualificationCoordinatorV63:
    """V6.3 replay surface plus a V6.5-internal mutation adapter.

    Public callers may use :meth:`state` and :meth:`verify`.  Mutation
    methods fail closed unless the process-owned V6.5 ingress capability was
    injected by the additive control plane.
    """

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        authority_key: bytes,
        authority_key_id: str,
        trusted_public_keys: Mapping[str, bytes],
        _mutation_gate: object | None = None,
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
        self._mutation_gate = _mutation_gate

    def _open(self) -> StageWorkspaceV50:
        return StageWorkspaceV50.open_existing(
            self._workspace_root,
            authority_key=self._authority_key,
            authority_key_id=self._authority_key_id,
        )

    @contextmanager
    def _locked_reopen(self) -> Iterator[StageWorkspaceV50]:
        bootstrap = self._open()
        with bootstrap.graph.store.writer_transaction():
            workspace = self._open()
            if not workspace.verify():
                raise ExternalQualificationCoordinatorError(
                    "workspace failed verification inside writer lock"
                )
            yield workspace

    @staticmethod
    def _actor(actor: str) -> CoordinatorActorV63:
        if actor not in {"operator", "server"}:
            raise PermissionError(
                "model, Codex, and external-role actors cannot commit "
                "coordinator authority transitions"
            )
        return actor  # type: ignore[return-value]

    def state(
        self,
        *,
        qualification_id: str | None = None,
    ) -> ExternalQualificationStateV63:
        with self._locked_reopen() as workspace:
            return _project_external_qualification_state_locked_v63(
                workspace,
                qualification_id=qualification_id,
                trusted_public_keys=self._trusted_public_keys,
            )

    def _completed_result(
        self,
        *,
        workspace: StageWorkspaceV50,
        receipt_item: _LedgerItem,
        receipt: ExternalQualificationOperationReceiptV63,
        resumed: bool,
    ) -> ExternalQualificationOperationResultV63:
        state = _project_external_qualification_state_locked_v63(
            workspace,
            qualification_id=receipt.qualification_id,
            trusted_public_keys=self._trusted_public_keys,
        )
        dispatch_packet = None
        if receipt.operation_type == "reserve_evaluation":
            dispatch_packet = _typed_one(
                workspace,
                kind=DISPATCH_PACKET_KIND_V63,
                model_type=ExternalEvaluationDispatchPacketV63,
                qualification_id=receipt.qualification_id,
            )
        return ExternalQualificationOperationResultV63(
            operation_id=receipt.operation_id,
            operation_type=receipt.operation_type,
            request_hash=receipt.request_hash,
            resumed=resumed,
            operation_receipt_hash=receipt_item.reference.sha256,
            output_artifact_hashes=receipt.output_artifact_hashes,
            dispatch_packet=dispatch_packet,
            state=state,
            v63_protocol_qualification_granted=(
                state.v63_protocol_qualification_granted
            ),
        )

    def _mutate(
        self,
        *,
        operation_type: CoordinatorOperationV63,
        qualification_id: str,
        expected_state_hash: str,
        actor: str,
        arguments: Mapping[str, object],
        allowed_start_phases: set[str],
        allowed_result_phases: set[str],
        action: Callable[
            [StageWorkspaceV50, ExternalQualificationStateV63],
            _ActionOutput,
        ],
    ) -> ExternalQualificationOperationResultV63:
        normalized_actor = self._actor(actor)
        request_hash = _operation_request_hash(
            operation_type=operation_type,
            qualification_id=qualification_id,
            expected_state_hash=expected_state_hash,
            actor=normalized_actor,
            arguments=arguments,
        )
        operation_id = _operation_id(operation_type, request_hash)

        with self._locked_reopen() as workspace:
            if (
                not _valid_v65_mutation_gate(self._mutation_gate)
            ):
                raise ExternalQualificationCoordinatorError(
                    "V6.3 mutation is internal to the V6.5 control plane; "
                    "the public V6.3 coordinator surface is read-only"
                )
            before = _project_external_qualification_state_locked_v63(
                workspace,
                qualification_id=qualification_id,
                trusted_public_keys=self._trusted_public_keys,
            )
            _events, items = _read_ledger(workspace)
            intents, receipts = _operation_ledger(workspace, items)
            exact_intent = intents.get(operation_id)
            exact_receipt = receipts.get(operation_id)

            same_transition = [
                intent
                for _item, intent in intents.values()
                if intent.qualification_id == qualification_id
                and intent.operation_type == operation_type
            ]
            divergent = [
                intent
                for intent in same_transition
                if intent.request_hash != request_hash
            ]
            if divergent:
                raise ExternalQualificationCoordinatorError(
                    "a divergent request already exists for this transition"
                )
            pending_other = [
                intent
                for candidate_id, (_item, intent) in intents.items()
                if candidate_id not in receipts
                and candidate_id != operation_id
                and intent.qualification_id == qualification_id
            ]
            if pending_other:
                raise ExternalQualificationCoordinatorError(
                    "an unfinished coordinator intent blocks this operation"
                )

            if exact_receipt is not None:
                if exact_intent is None:
                    raise ExternalQualificationCoordinatorError(
                        "completed operation lacks its intent"
                    )
                return self._completed_result(
                    workspace=workspace,
                    receipt_item=exact_receipt[0],
                    receipt=exact_receipt[1],
                    resumed=True,
                )

            resumed = exact_intent is not None
            if exact_intent is None:
                if before.state_hash != expected_state_hash:
                    raise ExternalQualificationCoordinatorError(
                        "expected_state_hash is stale"
                    )
                if before.phase not in allowed_start_phases:
                    raise ExternalQualificationCoordinatorError(
                        f"{operation_type} is invalid from phase {before.phase}"
                    )
                intent = _seal_intent(
                    workspace,
                    operation_id=operation_id,
                    qualification_id=qualification_id,
                    operation_type=operation_type,
                    request_hash=request_hash,
                    expected_state_hash=expected_state_hash,
                    expected_phase=before.phase,
                    actor=normalized_actor,
                )
                workspace.commit_evidence(
                    OPERATION_INTENT_KIND_V63,
                    intent.model_dump(mode="json"),
                )
            else:
                intent = exact_intent[1]
                if (
                    intent.request_hash != request_hash
                    or intent.expected_state_hash != expected_state_hash
                    or intent.actor != normalized_actor
                ):
                    raise ExternalQualificationCoordinatorError(
                        "operation retry differs from pending intent"
                    )
                if before.phase not in (
                    allowed_start_phases | allowed_result_phases
                ):
                    raise ExternalQualificationCoordinatorError(
                        "pending operation cannot resume from current phase"
                    )

            output = action(workspace, before)
            action_state = _project_external_qualification_state_locked_v63(
                workspace,
                qualification_id=qualification_id,
                trusted_public_keys=self._trusted_public_keys,
            )
            if action_state.phase not in allowed_result_phases:
                raise ExternalQualificationCoordinatorError(
                    f"{operation_type} produced phase {action_state.phase}"
                )
            receipt = _seal_operation_receipt(
                workspace,
                intent=intent,
                action_state=action_state,
                output_artifact_hashes=output.artifact_hashes,
            )
            receipt_ref = workspace.commit_evidence(
                OPERATION_RECEIPT_KIND_V63,
                receipt.model_dump(mode="json"),
            )
            final_state = _project_external_qualification_state_locked_v63(
                workspace,
                qualification_id=qualification_id,
                trusted_public_keys=self._trusted_public_keys,
            )
            return ExternalQualificationOperationResultV63(
                operation_id=operation_id,
                operation_type=operation_type,
                request_hash=request_hash,
                resumed=resumed,
                operation_receipt_hash=receipt_ref.sha256,
                output_artifact_hashes=output.artifact_hashes,
                dispatch_packet=output.dispatch_packet,
                state=final_state,
                v63_protocol_qualification_granted=(
                    final_state.v63_protocol_qualification_granted
                ),
            )

    def run_prediction(
        self,
        *,
        qualification_id: str,
        expected_state_hash: str,
        actor: str = "operator",
    ) -> ExternalQualificationOperationResultV63:
        """Generate or resume the deterministic public-input prediction."""

        def action(
            workspace: StageWorkspaceV50,
            _state: ExternalQualificationStateV63,
        ) -> _ActionOutput:
            contract = _typed_one(
                workspace,
                kind=CONTRACT_KIND_V63,
                model_type=PredictiveExternalQualificationContractV63,
                qualification_id=qualification_id,
            )
            forecast_input = _typed_one(
                workspace,
                kind=FORECAST_INPUT_KIND_V63,
                model_type=ExternalForecastInputV63,
                qualification_id=qualification_id,
                task_id=contract.task_id,
            )
            custody = _typed_one(
                workspace,
                kind=CUSTODY_KIND_V63,
                model_type=ExternalEvidenceCustodyV63,
                qualification_id=qualification_id,
            )
            result = run_current_model_external_prediction_v63(
                workspace=workspace,
                contract=contract,
                forecast_input=forecast_input,
                custody=custody,
            )
            return _ActionOutput(
                artifact_hashes={
                    "prediction_vector": _artifact_hash_for_model(
                        workspace,
                        kind=PREDICTION_VECTOR_KIND_V63,
                        model_type=ExternalPredictionVectorV63,
                        model=result.prediction_vector,
                    ),
                    "generator_execution_receipt": _artifact_hash_for_model(
                        workspace,
                        kind="role_execution_receipt_v50",
                        model_type=RoleExecutionReceiptV50,
                        model=result.execution_receipt,
                    ),
                    "prediction_binding": _artifact_hash_for_model(
                        workspace,
                        kind=PREDICTION_BINDING_KIND_V63,
                        model_type=CurrentModelPredictionBindingV63,
                        model=result.binding,
                    ),
                }
            )

        return self._mutate(
            operation_type="run_prediction",
            qualification_id=qualification_id,
            expected_state_hash=expected_state_hash,
            actor=actor,
            arguments={
                "runtime_adapter_id": PREDICTION_RUNTIME_ADAPTER_ID_V63
            },
            allowed_start_phases={"CUSTODY_VERIFIED"},
            allowed_result_phases={"PREDICTION_BOUND"},
            action=action,
        )

    def reserve_evaluation(
        self,
        *,
        qualification_id: str,
        expected_state_hash: str,
        evaluator_key_id: str,
        evaluator_host_id: str,
        actor: str = "operator",
    ) -> ExternalQualificationOperationResultV63:
        """Reserve exactly one evaluator dispatch without contacting it."""

        def action(
            workspace: StageWorkspaceV50,
            _state: ExternalQualificationStateV63,
        ) -> _ActionOutput:
            contract = _typed_one(
                workspace,
                kind=CONTRACT_KIND_V63,
                model_type=PredictiveExternalQualificationContractV63,
                qualification_id=qualification_id,
            )
            forecast_input = _typed_one(
                workspace,
                kind=FORECAST_INPUT_KIND_V63,
                model_type=ExternalForecastInputV63,
                qualification_id=qualification_id,
                task_id=contract.task_id,
            )
            custody = _typed_one(
                workspace,
                kind=CUSTODY_KIND_V63,
                model_type=ExternalEvidenceCustodyV63,
                qualification_id=qualification_id,
            )
            binding = _typed_one(
                workspace,
                kind=PREDICTION_BINDING_KIND_V63,
                model_type=CurrentModelPredictionBindingV63,
                qualification_id=qualification_id,
            )
            registration = _typed_one(
                workspace,
                kind=REGISTRATION_KIND_V63,
                model_type=ExternalPredictionRegistrationV63,
                qualification_id=qualification_id,
                task_id=contract.task_id,
            )
            prediction_seal = _typed_one(
                workspace,
                kind=PREDICTION_SEAL_KIND_V50,
                model_type=PredictionSealV50,
                qualification_id=qualification_id,
                task_id=contract.task_id,
            )
            reservation = qualification.reserve_external_evaluation_v63(
                workspace=workspace,
                contract=contract,
                custody=custody,
                prediction_binding=binding,
                registration=registration,
                prediction_seal=prediction_seal,
                evaluator_key_id=evaluator_key_id,
                evaluator_host_id=evaluator_host_id,
            )
            reservation_artifact_hash = _artifact_hash_for_model(
                workspace,
                kind=RESERVATION_KIND_V63,
                model_type=ExternalEvaluationReservationV63,
                model=reservation,
            )
            packet = _dispatch_packet(
                workspace,
                contract=contract,
                forecast_input=forecast_input,
                custody=custody,
                registration=registration,
                prediction_seal=prediction_seal,
                reservation=reservation,
                reservation_artifact_hash=reservation_artifact_hash,
            )
            packet, packet_artifact_hash = (
                _commit_or_recover_dispatch_packet(workspace, packet)
            )
            return _ActionOutput(
                artifact_hashes={
                    "evaluation_reservation": reservation_artifact_hash,
                    "dispatch_packet": packet_artifact_hash,
                },
                dispatch_packet=packet,
            )

        return self._mutate(
            operation_type="reserve_evaluation",
            qualification_id=qualification_id,
            expected_state_hash=expected_state_hash,
            actor=actor,
            arguments={
                "evaluator_key_id": evaluator_key_id,
                "evaluator_host_id": evaluator_host_id,
            },
            allowed_start_phases={"PREDICTION_REGISTERED"},
            allowed_result_phases={"EVALUATION_RESERVED"},
            action=action,
        )

    def _qualification_chain(
        self,
        workspace: StageWorkspaceV50,
        qualification_id: str,
    ) -> dict[str, Any]:
        contract = _typed_one(
            workspace,
            kind=CONTRACT_KIND_V63,
            model_type=PredictiveExternalQualificationContractV63,
            qualification_id=qualification_id,
        )
        return {
            "contract": contract,
            "forecast_input": _typed_one(
                workspace,
                kind=FORECAST_INPUT_KIND_V63,
                model_type=ExternalForecastInputV63,
                qualification_id=qualification_id,
                task_id=contract.task_id,
            ),
            "custody": _typed_one(
                workspace,
                kind=CUSTODY_KIND_V63,
                model_type=ExternalEvidenceCustodyV63,
                qualification_id=qualification_id,
            ),
            "binding": _typed_one(
                workspace,
                kind=PREDICTION_BINDING_KIND_V63,
                model_type=CurrentModelPredictionBindingV63,
                qualification_id=qualification_id,
            ),
            "registration": _typed_one(
                workspace,
                kind=REGISTRATION_KIND_V63,
                model_type=ExternalPredictionRegistrationV63,
                qualification_id=qualification_id,
                task_id=contract.task_id,
            ),
            "prediction_seal": _typed_one(
                workspace,
                kind=PREDICTION_SEAL_KIND_V50,
                model_type=PredictionSealV50,
                qualification_id=qualification_id,
                task_id=contract.task_id,
            ),
            "reservation": _typed_one(
                workspace,
                kind=RESERVATION_KIND_V63,
                model_type=ExternalEvaluationReservationV63,
                qualification_id=qualification_id,
                task_id=contract.task_id,
            ),
        }

    def verify(
        self,
        *,
        qualification_id: str,
    ) -> ExternalQualificationVerificationV63:
        """Read-only replay of every authority chain available in the phase."""

        with self._locked_reopen() as workspace:
            state = _project_external_qualification_state_locked_v63(
                workspace,
                qualification_id=qualification_id,
                trusted_public_keys=self._trusted_public_keys,
            )
            checks: dict[str, bool] = {
                "workspace_verified": workspace.verify(),
                "state_current": state.current,
                "state_consistent": state.phase != "INCONSISTENT",
            }
            reasons = list(state.reason_codes)
            prediction_replay_hash = None
            qualification_replay_hash = None

            if state.phase in {
                "PREDICTION_BOUND",
                "REGISTRATION_COMMITTED",
                "PREDICTION_REGISTERED",
                "EVALUATION_RESERVED",
                "EVALUATION_COMMITTED",
                "AWAITING_PROMOTION",
                "PROMOTION_COMMITTED",
                "EXTERNALLY_QUALIFIED",
                "REJECTED",
            }:
                try:
                    chain = self._qualification_chain(
                        workspace, qualification_id
                    )
                    runtime = verify_current_model_external_prediction_v63(
                        workspace=workspace,
                        contract=chain["contract"],
                        forecast_input=chain["forecast_input"],
                        custody=chain["custody"],
                    )
                    checks["prediction_runtime_replayed"] = True
                    prediction_replay_hash = sha256_value(
                        {
                            "forecast_input_hash": (
                                runtime.forecast_input.input_hash
                            ),
                            "prediction_vector_hash": (
                                runtime.prediction_vector.vector_hash
                            ),
                            "execution_receipt_hash": (
                                runtime.execution_receipt.receipt_hash
                            ),
                            "binding_hash": runtime.binding.binding_hash,
                        }
                    )
                except (
                    ExternalPredictionRuntimeError,
                    ExternalQualificationCoordinatorError,
                ):
                    checks["prediction_runtime_replayed"] = False
                    reasons.append("prediction_runtime_replay_failed")

            if state.phase in {"EXTERNALLY_QUALIFIED", "REJECTED"}:
                try:
                    terminal = [
                        item
                        for _reference, item in workspace._artifacts_of_kind(
                            QUALIFICATION_KIND_V63,
                            ExternalPredictiveQualificationReceiptV63,
                        )
                        if item.qualification_id == qualification_id
                        and item.status != "NOT_RUN"
                    ]
                    if len(terminal) != 1:
                        raise ExternalQualificationCoordinatorError(
                            "terminal qualification receipt is ambiguous"
                        )
                    replay = (
                        qualification.verify_external_predictive_qualification_v63(
                            workspace=workspace,
                            receipt=terminal[0],
                            trusted_public_keys=self._trusted_public_keys,
                        )
                    )
                    checks["qualification_receipt_replayed"] = (
                        replay.status == "PASS"
                    )
                    qualification_replay_hash = replay.replay_hash
                except (
                    ExternalQualificationCoordinatorError,
                    ExternalQualificationError,
                    AttributeError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ):
                    checks["qualification_receipt_replayed"] = False
                    reasons.append("qualification_receipt_replay_failed")

            fail = any(not value for value in checks.values())
            if fail:
                status: Literal["PASS", "FAIL", "NOT_RUN"] = "FAIL"
            elif state.phase in {
                "LOCAL_READY",
                "CONTRACT_FROZEN",
                "INPUT_FROZEN",
                "CUSTODY_COMMITTED",
                "CUSTODY_VERIFIED",
                "PREDICTION_BOUND",
                "REGISTRATION_COMMITTED",
                "PREDICTION_REGISTERED",
                "EVALUATION_RESERVED",
                "EVALUATION_COMMITTED",
                "AWAITING_PROMOTION",
                "PROMOTION_COMMITTED",
            }:
                status = "NOT_RUN"
            else:
                status = "PASS"
            return ExternalQualificationVerificationV63.seal(
                qualification_id=qualification_id,
                state_hash=state.state_hash,
                phase=state.phase,
                status=status,
                checks=checks,
                reason_codes=sorted(set(reasons)),
                prediction_runtime_replay_hash=prediction_replay_hash,
                qualification_replay_hash=qualification_replay_hash,
                v63_protocol_qualification_granted=(
                    status == "PASS"
                    and state.phase == "EXTERNALLY_QUALIFIED"
                ),
            )
