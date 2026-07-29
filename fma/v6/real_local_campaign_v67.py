"""Explicit V6.7 real-local campaign with a pre-data action.

V6.5 artifacts remain governed by :mod:`fma.v6.real_local_campaign`.  This
module owns separate spec, event, freeze, terminal, and lock paths so that the
six-step V6.7 workflow cannot silently reinterpret a five-step V6.5 journal.
"""

from __future__ import annotations

import hashlib
import hmac
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Callable, Literal

from pydantic import Field, ValidationError, model_validator

from fma._file_lock import exclusive_file_lock
from fma.codex_driver import CodexCLIConfig
from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5.stage_workspace import StageWorkspaceV50
from fma.v6.measurement_study_design import (
    MEASUREMENT_STUDY_DESIGN_PATH_V67,
)
from fma.v6.predata_protocol import (
    PREDATA_EXECUTION_PROTOCOL_PATH_V67,
    minimum_predata_observation_count_v67,
    registered_positive_series_capability_pack_v67,
    verify_predata_execution_protocol_v67,
)
from fma.v6.public_source import SOURCE_CONTRACT_PATH
from fma.studio.service import (
    StudioTaskService,
    build_world_bank_predata_bundle_v67,
)

from .real_local_campaign import (
    EVENTS_PATH_V65,
    FREEZE_RECEIPT_PATH_V65,
    LOCK_PATH_V65,
    SPEC_PATH_V65,
    TERMINAL_RECEIPTS_PATH_V65,
    CampaignConflictError,
    CodexRuntimeBudgetsV65,
    CodexRuntimeContractV65,
    EventStatusV65,
    EventTypeV65,
    RealLocalCampaignError,
    RealLocalCampaignEventV65,
    RealLocalCampaignFreezeReceiptV65,
    RealLocalCampaignRunnerV65,
    RealLocalCampaignSpecV65,
    TerminalStatusV65,
    _append_line,
    _assert_aware,
    _build_codex_runtime_contract,
    _file_hash,
    _hash_without,
    _read_json_object,
    _utc_now,
    _write_new,
)


SPEC_PATH_V67 = "campaign_spec_v67.json"
EVENTS_PATH_V67 = "campaign_events_v67.jsonl"
FREEZE_RECEIPT_PATH_V67 = "campaign_freeze_receipt_v67.json"
TERMINAL_RECEIPTS_PATH_V67 = "terminal_receipts_v67.jsonl"
LOCK_PATH_V67 = ".real_local_campaign_v67.lock"

ActionV67 = Literal[
    "create_task",
    "run_s0",
    "prepare_predata_v67",
    "run_s1",
    "ingest_world_bank_data",
    "run_backhalf",
]
ACTION_ORDER_V67: tuple[ActionV67, ...] = (
    "create_task",
    "run_s0",
    "prepare_predata_v67",
    "run_s1",
    "ingest_world_bank_data",
    "run_backhalf",
)
_PREDATA_PATHS_V67 = (
    SOURCE_CONTRACT_PATH,
    MEASUREMENT_STUDY_DESIGN_PATH_V67,
    PREDATA_EXECUTION_PROTOCOL_PATH_V67,
)


class CodexRuntimeBudgetsV67(CodexRuntimeBudgetsV65):
    """V6.7 runtime budgets for its explicit six-action campaign."""

    campaign_action_limit: Literal[6] = 6


class CodexRuntimeContractV67(CodexRuntimeContractV65):
    """Frozen V6.7 runtime identity with the six-action budget."""

    schema_version: Literal["6.7-codex-runtime-contract"] = "6.7-codex-runtime-contract"
    budgets: CodexRuntimeBudgetsV67


def build_codex_runtime_contract_v67(
    *,
    config: CodexCLIConfig,
    source_adapter_id: str,
    process_runner: Any | None = None,
    cli_locator: Callable[..., Path] | None = None,
) -> CodexRuntimeContractV67:
    """Inspect, version-check, and freeze one trusted V6.7 Codex runtime."""

    contract = _build_codex_runtime_contract(
        config=config,
        source_adapter_id=source_adapter_id,
        budgets=CodexRuntimeBudgetsV67.from_config(config),
        contract_type=CodexRuntimeContractV67,
        process_runner=process_runner,
        cli_locator=cli_locator,
    )
    if not isinstance(contract, CodexRuntimeContractV67):
        raise RealLocalCampaignError(
            "V6.7 runtime builder returned another contract version"
        )
    return contract


class RealLocalCampaignSpecV67(RealLocalCampaignSpecV65):
    """Frozen V6.7 campaign inputs with an explicit pre-data workflow."""

    schema_version: Literal["6.7-real-local-campaign-spec"] = (
        "6.7-real-local-campaign-spec"
    )
    predata_workflow: Literal["source_measurement_protocol_before_s1"] = (
        "source_measurement_protocol_before_s1"
    )
    codex_runtime_contract: CodexRuntimeContractV67 | None = None

    @model_validator(mode="after")
    def validate_predata_observation_budget(
        self,
    ) -> "RealLocalCampaignSpecV67":
        required = minimum_predata_observation_count_v67(
            self.world_bank_request.adapter_id
        )
        if self.world_bank_request.minimum_observations < required:
            raise ValueError(
                "V6.7 campaign minimum_observations cannot execute the frozen "
                f"rolling-origin protocol; requires at least {required}"
            )
        return self


class RealLocalCampaignEventV67(RealLocalCampaignEventV65):
    schema_version: Literal["6.7-real-local-campaign-event"] = (
        "6.7-real-local-campaign-event"
    )
    action: ActionV67 | None = None


class RealLocalCampaignFreezeReceiptV67(RealLocalCampaignFreezeReceiptV65):
    schema_version: Literal["6.7-real-local-campaign-freeze"] = (
        "6.7-real-local-campaign-freeze"
    )


class RealLocalCampaignTerminalReceiptV67(StrictModel):
    """Authority-bound terminal for exactly the six-step V6.7 action order."""

    schema_version: Literal["6.7-real-local-campaign-terminal"] = (
        "6.7-real-local-campaign-terminal"
    )
    execution_id: Identifier
    attempt_index: Annotated[int, Field(ge=1, le=3)]
    previous_receipt_hash: Sha256 | None = None
    campaign_id: Identifier
    task_id: Identifier
    spec_hash: Sha256
    terminal_status: TerminalStatusV65
    reason_codes: list[Identifier] = Field(default_factory=list)
    completed_actions: list[ActionV67] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime
    last_event_hash: Sha256
    pending_intent_hash: Literal[None] = None
    snapshot_hash: Sha256 | None = None
    workspace_spec_hash: Sha256 | None = None
    runtime_contract_hash: Sha256 | None = None
    source_evidence_hash: Sha256 | None = None
    predata_evidence_hash: Sha256 | None = None
    studio_event_tip_hash: Sha256 | None = None
    campaign_event_chain_verified: bool
    workspace_verified: bool
    studio_event_chain_verified: bool
    snapshot_fixture_only: bool | None
    workflow_complete: bool
    fixture_or_control: bool
    external_scientific_qualification_status: Literal["NOT_RUN"] = "NOT_RUN"
    claim_ceiling: Literal[
        "no_scientific_claim",
        "control_protocol_only",
        "local_workflow_evidence_only",
    ]
    receipt_is_authority: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "RealLocalCampaignTerminalReceiptV67":
        _assert_aware(self.started_at, "started_at")
        _assert_aware(self.finished_at, "finished_at")
        if self.finished_at < self.started_at:
            raise ValueError("terminal receipt finishes before it starts")
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("terminal reason codes must be sorted and unique")
        if len(self.completed_actions) != len(set(self.completed_actions)):
            raise ValueError("completed actions must be unique")
        indices = [ACTION_ORDER_V67.index(item) for item in self.completed_actions]
        if indices != sorted(indices):
            raise ValueError("completed actions are out of V6.7 workflow order")
        if (self.attempt_index == 1) != (self.previous_receipt_hash is None):
            raise ValueError("terminal receipt predecessor differs from attempt index")
        if self.fixture_or_control:
            if self.terminal_status == "COMPLETED_LOCAL":
                raise ValueError("control execution cannot be completed-local")
            if self.claim_ceiling != "control_protocol_only":
                raise ValueError("control execution has an invalid claim ceiling")
        elif self.terminal_status == "COMPLETED_CONTROL":
            raise ValueError("real execution cannot be completed-control")
        if self.terminal_status in {
            "COMPLETED_LOCAL",
            "COMPLETED_CONTROL",
        } and (
            self.completed_actions != list(ACTION_ORDER_V67)
            or self.predata_evidence_hash is None
        ):
            raise ValueError(
                "completed V6.7 receipt lacks the six-step pre-data evidence"
            )
        if self.terminal_status == "COMPLETED_LOCAL" and not (
            not self.reason_codes
            and self.campaign_event_chain_verified
            and self.workspace_verified
            and self.studio_event_chain_verified
            and self.snapshot_fixture_only is False
            and self.workflow_complete
            and self.claim_ceiling == "local_workflow_evidence_only"
            and self.runtime_contract_hash is not None
            and self.source_evidence_hash is not None
            and self.studio_event_tip_hash is not None
            and self.snapshot_hash is not None
            and self.workspace_spec_hash is not None
        ):
            raise ValueError(
                "completed-local V6.7 receipt lacks real local verification"
            )
        if self.receipt_hash and not self.authority_auth_tag:
            raise ValueError("terminal receipt hash requires an authority auth tag")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("V6.7 terminal receipt hash differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "authority_auth_tag", "receipt_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_content_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("V6.7 terminal receipt is not sealed")

    def assert_sealed(
        self,
        *,
        authority_key: bytes,
        authority_key_id: str,
    ) -> None:
        self.assert_content_sealed()
        expected_tag = hmac.new(
            authority_key,
            f"real-local-terminal-v67:{self.unsigned_hash()}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if (
            self.authority_key_id != authority_key_id
            or not self.authority_auth_tag
            or not hmac.compare_digest(
                self.authority_auth_tag,
                expected_tag,
            )
        ):
            raise ValueError("V6.7 terminal receipt authority differs")

    @classmethod
    def seal(
        cls,
        *,
        authority_key: bytes,
        **data: object,
    ) -> "RealLocalCampaignTerminalReceiptV67":
        unsigned = cls(**data)
        payload = unsigned.model_dump(
            mode="json",
            exclude={"authority_auth_tag", "receipt_hash"},
        )
        payload["authority_auth_tag"] = hmac.new(
            authority_key,
            f"real-local-terminal-v67:{unsigned.unsigned_hash()}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        tagged = cls(**payload)
        payload = tagged.model_dump(mode="json", exclude={"receipt_hash"})
        payload["receipt_hash"] = tagged.content_hash()
        return cls(**payload)


class RealLocalCampaignRunnerV67(RealLocalCampaignRunnerV65):
    """V6.7 runner with separate artifacts and one explicit pre-data action."""

    @property
    def spec_path(self) -> Path:
        return self.root / SPEC_PATH_V67

    @property
    def events_path(self) -> Path:
        return self.root / EVENTS_PATH_V67

    @property
    def freeze_receipt_path(self) -> Path:
        return self.root / FREEZE_RECEIPT_PATH_V67

    @property
    def terminal_receipts_path(self) -> Path:
        return self.root / TERMINAL_RECEIPTS_PATH_V67

    @property
    def lock_path(self) -> Path:
        return self.root / LOCK_PATH_V67

    @property
    def action_order(self) -> tuple[ActionV67, ...]:
        return ACTION_ORDER_V67

    @staticmethod
    def _runtime_budgets_from_config(
        config: CodexCLIConfig,
    ) -> CodexRuntimeBudgetsV67:
        return CodexRuntimeBudgetsV67.from_config(config)

    @staticmethod
    def _seal_runtime_contract(
        **data: object,
    ) -> CodexRuntimeContractV67:
        return CodexRuntimeContractV67.seal(**data)

    def prepare(
        self,
        spec: RealLocalCampaignSpecV67 | dict[str, Any],
    ) -> RealLocalCampaignSpecV67:
        if isinstance(spec, RealLocalCampaignSpecV67):
            validated = spec
            validated.assert_sealed()
        else:
            payload = dict(spec)
            supplied_hash = payload.pop("spec_hash", None)
            validated = RealLocalCampaignSpecV67.seal(**payload)
            if supplied_hash is not None and supplied_hash != validated.spec_hash:
                raise CampaignConflictError("supplied V6.7 campaign hash differs")
        authority_key = self._authority_key_required()
        self.root.mkdir(parents=True, exist_ok=True)
        with exclusive_file_lock(self.lock_path):
            foreign = (
                self.root / SPEC_PATH_V65,
                self.root / EVENTS_PATH_V65,
                self.root / FREEZE_RECEIPT_PATH_V65,
                self.root / TERMINAL_RECEIPTS_PATH_V65,
                self.root / LOCK_PATH_V65,
            )
            if any(path.exists() for path in foreign):
                raise CampaignConflictError(
                    "V6.7 campaign root contains V6.5 authority artifacts"
                )
            core_paths = (
                self.spec_path,
                self.events_path,
                self.freeze_receipt_path,
            )
            material_state = any(
                path.exists()
                for path in (
                    *core_paths,
                    self.terminal_receipts_path,
                    self.workspace_root,
                )
            )
            if material_state and not all(path.is_file() for path in core_paths):
                raise CampaignConflictError(
                    "partial V6.7 campaign state cannot be upgraded silently"
                )
            if material_state:
                if self.load_spec() != validated:
                    raise CampaignConflictError(
                        "campaign root contains another V6.7 spec"
                    )
            else:
                _write_new(
                    self.spec_path,
                    validated.model_dump(mode="json"),
                )
                self._append_event_locked(
                    validated,
                    event_type="CAMPAIGN_PREPARED",
                    status="PREPARED",
                    details=self._genesis_details(validated),
                )
                events = self._read_events(validated)
                if len(events) != 1 or events[0].event_hash is None:
                    raise CampaignConflictError(
                        "V6.7 campaign failed to create one exact genesis"
                    )
                freeze = RealLocalCampaignFreezeReceiptV67.seal(
                    authority_key=authority_key,
                    campaign_id=validated.campaign_id,
                    task_id=validated.task_id,
                    spec_hash=validated.spec_hash,
                    genesis_event_hash=events[0].event_hash,
                    authority_key_id=self.authority_key_id,
                    frozen_at=_utc_now(),
                )
                _write_new(
                    self.freeze_receipt_path,
                    freeze.model_dump(mode="json"),
                )
        return validated

    @staticmethod
    def _genesis_details(
        spec: RealLocalCampaignSpecV67,
    ) -> dict[str, Any]:
        return {
            "spec_hash": spec.spec_hash,
            "execution_mode": spec.execution_mode,
            "live_codex": spec.live_codex,
            "live_world_bank": spec.live_world_bank,
            "default_live_execution": False,
            "execution_semantics": spec.execution_semantics,
            "max_execution_attempts": spec.max_execution_attempts,
            "campaign_schema_version": spec.schema_version,
            "action_order": list(ACTION_ORDER_V67),
            "predata_workflow": spec.predata_workflow,
        }

    def load_spec(self) -> RealLocalCampaignSpecV67:
        try:
            spec = RealLocalCampaignSpecV67.model_validate_json(
                self.spec_path.read_text(encoding="utf-8")
            )
            spec.assert_sealed()
            events = self._read_events(spec)
            if not events or events[0].event_hash is None:
                raise CampaignConflictError("V6.7 campaign lacks its prepared genesis")
            self._read_freeze_receipt(spec, events[0])
            return spec
        except FileNotFoundError as exc:
            raise RealLocalCampaignError("V6.7 campaign is not prepared") from exc

    def _read_freeze_receipt(
        self,
        spec: RealLocalCampaignSpecV67,
        genesis: RealLocalCampaignEventV67,
    ) -> RealLocalCampaignFreezeReceiptV67:
        receipt = RealLocalCampaignFreezeReceiptV67.model_validate(
            _read_json_object(self.freeze_receipt_path)
        )
        receipt.assert_sealed(
            authority_key=self._authority_key_required(),
            authority_key_id=self.authority_key_id,
        )
        if (
            genesis.event_type != "CAMPAIGN_PREPARED"
            or genesis.sequence != 1
            or genesis.previous_event_hash is not None
            or genesis.event_hash is None
            or receipt.campaign_id != spec.campaign_id
            or receipt.task_id != spec.task_id
            or receipt.spec_hash != spec.spec_hash
            or receipt.genesis_event_hash != genesis.event_hash
        ):
            raise CampaignConflictError(
                "V6.7 freeze receipt differs from spec or genesis"
            )
        return receipt

    def _read_events(
        self,
        spec: RealLocalCampaignSpecV67 | None = None,
    ) -> list[RealLocalCampaignEventV67]:
        if not self.events_path.is_file():
            return []
        expected_spec = spec or self.load_spec()
        events: list[RealLocalCampaignEventV67] = []
        previous: str | None = None
        intents: dict[str, RealLocalCampaignEventV67] = {}
        resolved: set[str] = set()
        for raw in self.events_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                raise CampaignConflictError("V6.7 campaign journal contains blank line")
            event = RealLocalCampaignEventV67.model_validate_json(raw)
            event.assert_sealed()
            if (
                event.campaign_id != expected_spec.campaign_id
                or event.sequence != len(events) + 1
                or event.previous_event_hash != previous
            ):
                raise CampaignConflictError("V6.7 campaign event chain differs")
            if event.event_type == "ACTION_INTENT":
                assert event.action_id is not None
                if event.action_id in intents:
                    raise CampaignConflictError("duplicate V6.7 campaign action ID")
                intents[event.action_id] = event
            elif event.event_type == "ACTION_RESULT":
                assert event.action_id is not None
                intent = intents.get(event.action_id)
                if (
                    intent is None
                    or event.action_id in resolved
                    or event.intent_event_hash != intent.event_hash
                    or event.action != intent.action
                    or event.request_hash != intent.request_hash
                ):
                    raise CampaignConflictError(
                        "V6.7 result does not bind one pending intent"
                    )
                resolved.add(event.action_id)
            events.append(event)
            previous = event.event_hash
        prepared = [
            event for event in events if event.event_type == "CAMPAIGN_PREPARED"
        ]
        if (
            not events
            or events[0].event_type != "CAMPAIGN_PREPARED"
            or len(prepared) != 1
            or events[0].details != self._genesis_details(expected_spec)
            or len(set(intents) - resolved) > 1
        ):
            raise CampaignConflictError("V6.7 campaign genesis differs")
        return events

    def _append_event_locked(
        self,
        spec: RealLocalCampaignSpecV67,
        *,
        event_type: EventTypeV65,
        status: EventStatusV65,
        action_id: str | None = None,
        action: ActionV67 | None = None,
        request_hash: str | None = None,
        intent_event_hash: str | None = None,
        result_hash: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> RealLocalCampaignEventV67:
        events = self._read_events(spec)
        event = RealLocalCampaignEventV67.seal(
            campaign_id=spec.campaign_id,
            sequence=len(events) + 1,
            event_type=event_type,
            status=status,
            action_id=action_id,
            action=action,
            request_hash=request_hash,
            intent_event_hash=intent_event_hash,
            result_hash=result_hash,
            details=details or {},
            recorded_at=_utc_now(),
            previous_event_hash=events[-1].event_hash if events else None,
        )
        _append_line(self.events_path, event.model_dump(mode="json"))
        return event

    def _read_receipts(
        self,
        spec: RealLocalCampaignSpecV67 | None = None,
        *,
        verify_authority: bool = True,
    ) -> list[RealLocalCampaignTerminalReceiptV67]:
        if not self.terminal_receipts_path.is_file():
            return []
        expected_spec = spec or self.load_spec()
        events = self._read_events(expected_spec)
        event_indices = {event.event_hash: event.sequence for event in events}
        receipts: list[RealLocalCampaignTerminalReceiptV67] = []
        previous_receipt_hash: str | None = None
        previous_event_sequence = 0
        execution_ids: set[str] = set()
        for raw in self.terminal_receipts_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                raise CampaignConflictError("V6.7 terminal journal contains blank line")
            receipt = RealLocalCampaignTerminalReceiptV67.model_validate_json(raw)
            receipt.assert_content_sealed()
            if verify_authority:
                receipt.assert_sealed(
                    authority_key=self._authority_key_required(),
                    authority_key_id=self.authority_key_id,
                )
            if (
                receipt.campaign_id != expected_spec.campaign_id
                or receipt.task_id != expected_spec.task_id
                or receipt.spec_hash != expected_spec.spec_hash
                or receipt.last_event_hash not in event_indices
                or receipt.attempt_index != len(receipts) + 1
                or receipt.previous_receipt_hash != previous_receipt_hash
                or receipt.execution_id in execution_ids
                or event_indices[receipt.last_event_hash] < previous_event_sequence
            ):
                raise CampaignConflictError("V6.7 terminal receipt chain differs")
            receipts.append(receipt)
            execution_ids.add(receipt.execution_id)
            previous_receipt_hash = receipt.receipt_hash
            previous_event_sequence = event_indices[receipt.last_event_hash]
        return receipts

    def _validate_predata_workspace(
        self,
        spec: RealLocalCampaignSpecV67,
        snapshot: dict[str, Any],
    ) -> tuple[Literal["ABSENT", "COMPLETE", "PARTIAL"], str | None]:
        if self._is_control_execution():
            if snapshot.get("predata_v67_prepared") is True:
                return (
                    "COMPLETE",
                    sha256_value(
                        {
                            "schema_version": "6.7-control-predata",
                            "spec_hash": spec.spec_hash,
                        }
                    ),
                )
            return "ABSENT", None
        root = self._workspace_path(spec)
        present = [(root / relative).is_file() for relative in _PREDATA_PATHS_V67]
        if not any(present):
            return "ABSENT", None
        if not all(present):
            return "PARTIAL", None
        try:
            workspace = StageWorkspaceV50.open_existing(
                root,
                authority_key=self.authority_key,
                authority_key_id=self.authority_key_id,
            )
            workspace_hash = workspace.spec.spec_hash
            s0_gate_hash = workspace.current_gate("S0")
            if workspace_hash is None or s0_gate_hash is None or not workspace.verify():
                return "PARTIAL", None
            expected = build_world_bank_predata_bundle_v67(
                request=spec.world_bank_request,
                workspace_spec_hash=workspace_hash,
                s0_gate_hash=s0_gate_hash,
            )
            transaction = StudioTaskService._predata_transaction_state_v67(workspace)
            if (
                transaction.status not in {"COMPLETED", "LEGACY_COMPLETED"}
                or transaction.bundle is None
                or transaction.preparation_ref is None
            ):
                return "PARTIAL", None
            source, measurement, protocol = transaction.bundle
            if transaction.bundle != expected:
                return "PARTIAL", None
            pack = registered_positive_series_capability_pack_v67(
                protocol.adapter_binding.adapter_id
            )
            if not verify_predata_execution_protocol_v67(
                measurement_contract=measurement,
                capability_pack=pack,
                protocol=protocol,
            ):
                return "PARTIAL", None
            file_hashes = {
                relative: _file_hash(root / relative) for relative in _PREDATA_PATHS_V67
            }
            return (
                "COMPLETE",
                sha256_value(
                    {
                        "spec_hash": spec.spec_hash,
                        "workspace_spec_hash": workspace_hash,
                        "s0_gate_hash": s0_gate_hash,
                        "source_contract_hash": source.contract_hash,
                        "measurement_contract_hash": (measurement.contract_hash),
                        "protocol_hash": protocol.protocol_hash,
                        "transaction_status": transaction.status,
                        "intent_artifact_hash": (
                            transaction.intent_ref.sha256
                            if transaction.intent_ref is not None
                            else None
                        ),
                        "intent_hash": (
                            transaction.intent.intent_hash
                            if transaction.intent is not None
                            else None
                        ),
                        "evidence_hash": transaction.preparation_ref.sha256,
                        "completion_artifact_hash": (
                            transaction.completion_ref.sha256
                            if transaction.completion_ref is not None
                            else None
                        ),
                        "completion_hash": (
                            transaction.completion.completion_hash
                            if transaction.completion is not None
                            else None
                        ),
                        "file_hashes": file_hashes,
                    }
                ),
            )
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            ValidationError,
        ):
            return "PARTIAL", None

    def _validate_source_workspace(
        self,
        spec: RealLocalCampaignSpecV67,
        snapshot: dict[str, Any],
        *,
        require_s2: bool = False,
    ) -> tuple[Literal["ABSENT", "COMPLETE", "PARTIAL"], str | None]:
        root = self._workspace_path(spec)
        data_received = bool(snapshot.get("backhalf", {}).get("data_received"))
        source_only_predata = bool(
            not data_received
            and all((root / relative).is_file() for relative in _PREDATA_PATHS_V67)
            and not any(
                (root / relative).is_file()
                for relative in (
                    "data/raw/ode_input_v59.json",
                    "data/source_provenance_v62/world_bank_response.json",
                    "data/source_provenance_v62/receipt.json",
                )
            )
        )
        if source_only_predata:
            return "ABSENT", None
        return super()._validate_source_workspace(
            spec,
            snapshot,
            require_s2=require_s2,
        )

    def _action_postcondition(
        self,
        spec: RealLocalCampaignSpecV67,
        action: ActionV67,
        snapshot: dict[str, Any],
    ) -> tuple[bool, str | None]:
        if action == "prepare_predata_v67":
            state, evidence_hash = self._validate_predata_workspace(
                spec,
                snapshot,
            )
            return state == "COMPLETE", evidence_hash
        return super()._action_postcondition(spec, action, snapshot)

    def _action_request(
        self,
        spec: RealLocalCampaignSpecV67,
        action: ActionV67,
    ) -> object:
        if action == "create_task":
            request = super()._action_request(spec, action)
            if not isinstance(request, dict):
                raise RealLocalCampaignError(
                    "V6.7 create-task request is not an object"
                )
            return {**request, "workflow_mode": "v67"}
        if action == "prepare_predata_v67":
            return spec.world_bank_request.model_dump(mode="json")
        return super()._action_request(spec, action)

    def _call_action(
        self,
        service: Any,
        spec: RealLocalCampaignSpecV67,
        action: ActionV67,
    ) -> dict[str, Any]:
        if action == "prepare_predata_v67":
            return service.prepare_predata_v67(
                spec.task_id,
                spec.world_bank_request.model_dump(mode="json"),
            )
        return super()._call_action(service, spec, action)

    def _append_terminal(
        self,
        spec: RealLocalCampaignSpecV67,
        *,
        execution_id: str,
        started_at: datetime,
        terminal_status: TerminalStatusV65,
        reason_codes: list[str],
        snapshot: dict[str, Any] | None,
        workspace_spec_hash: str | None,
        campaign_event_chain_verified: bool,
        workspace_verified: bool,
        studio_event_chain_verified: bool,
        snapshot_fixture_only: bool | None,
        workflow_complete: bool,
        runtime_contract_hash: str | None = None,
        source_evidence_hash: str | None = None,
        studio_event_tip_hash: str | None = None,
    ) -> RealLocalCampaignTerminalReceiptV67:
        with exclusive_file_lock(self.lock_path):
            events = self._read_events(spec)
            if (
                not events
                or events[-1].event_hash is None
                or self._pending_intent(events) is not None
            ):
                raise CampaignConflictError(
                    "V6.7 terminal requires a resolved event tip"
                )
            receipts = self._read_receipts(spec)
            control = self._is_control_execution()
            claim_ceiling = (
                "control_protocol_only"
                if control
                else (
                    "local_workflow_evidence_only"
                    if terminal_status == "COMPLETED_LOCAL"
                    else "no_scientific_claim"
                )
            )
            predata_hash = None
            if snapshot is not None:
                _, predata_hash = self._validate_predata_workspace(
                    spec,
                    snapshot,
                )
            receipt = RealLocalCampaignTerminalReceiptV67.seal(
                authority_key=self._authority_key_required(),
                execution_id=execution_id,
                attempt_index=len(receipts) + 1,
                previous_receipt_hash=(receipts[-1].receipt_hash if receipts else None),
                campaign_id=spec.campaign_id,
                task_id=spec.task_id,
                spec_hash=spec.spec_hash,
                terminal_status=terminal_status,
                reason_codes=sorted(set(reason_codes)),
                completed_actions=self._completed_actions(events),
                started_at=started_at,
                finished_at=_utc_now(),
                last_event_hash=events[-1].event_hash,
                pending_intent_hash=None,
                snapshot_hash=(
                    sha256_value(snapshot) if snapshot is not None else None
                ),
                workspace_spec_hash=workspace_spec_hash,
                runtime_contract_hash=runtime_contract_hash,
                source_evidence_hash=source_evidence_hash,
                predata_evidence_hash=predata_hash,
                studio_event_tip_hash=studio_event_tip_hash,
                campaign_event_chain_verified=campaign_event_chain_verified,
                workspace_verified=workspace_verified,
                studio_event_chain_verified=studio_event_chain_verified,
                snapshot_fixture_only=(True if control else snapshot_fixture_only),
                workflow_complete=workflow_complete,
                fixture_or_control=control,
                claim_ceiling=claim_ceiling,
                authority_key_id=self.authority_key_id,
            )
            _append_line(
                self.terminal_receipts_path,
                receipt.model_dump(mode="json"),
            )
            return receipt

    def verify(self, *, require_real: bool = True) -> bool:
        if not super().verify(require_real=require_real):
            return False
        try:
            spec = self.load_spec()
            receipts = self._read_receipts(spec)
            service = self._make_service()
            snapshot = service.snapshot(spec.task_id)
            state, evidence_hash = self._validate_predata_workspace(
                spec,
                snapshot,
            )
            return bool(
                receipts
                and state == "COMPLETE"
                and receipts[-1].predata_evidence_hash == evidence_hash
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            return False

    def status(
        self,
        *,
        lock_timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        output = super().status(
            lock_timeout_seconds=lock_timeout_seconds,
        )
        output["schema_version"] = "6.7-real-local-campaign-status"
        output["action_order"] = list(ACTION_ORDER_V67)
        output["predata_workflow"] = "source_measurement_protocol_before_s1"
        return output


RealLocalCampaignV67 = RealLocalCampaignRunnerV67


__all__ = [
    "ACTION_ORDER_V67",
    "ActionV67",
    "CodexRuntimeBudgetsV67",
    "CodexRuntimeContractV67",
    "EVENTS_PATH_V67",
    "FREEZE_RECEIPT_PATH_V67",
    "LOCK_PATH_V67",
    "RealLocalCampaignEventV67",
    "RealLocalCampaignFreezeReceiptV67",
    "RealLocalCampaignRunnerV67",
    "RealLocalCampaignSpecV67",
    "RealLocalCampaignTerminalReceiptV67",
    "RealLocalCampaignV67",
    "SPEC_PATH_V67",
    "TERMINAL_RECEIPTS_PATH_V67",
    "build_codex_runtime_contract_v67",
]
