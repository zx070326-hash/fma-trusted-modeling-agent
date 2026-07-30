"""Bounded parallel S1 execution with controlled epistemic exchange."""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import uuid4

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier
from fma.v5.stage_workspace import POLICIES, StageWorkspaceV50
from fma.v5.workspace_schemas import (
    AssumptionRecordV50,
    AssumptionSetV50,
    CandidateFormalizationV50,
    CandidateSetV50,
    ModelSpecV50,
    SymbolRecordV50,
    SymbolTableV50,
    ValidationObligationV50,
    ValidationPlanV50,
)
from fma.v5_1.codex_stage_driver import (
    RoleProcessOutcomeV51,
    StageRoleDriverV51,
)
from fma.v5_8.epistemic import (
    DisclosurePacketV58,
    EpistemicGraphBuilderV58,
    EpistemicGraphStoreV58,
    KnowledgeBrokerV58,
    KnowledgeDraftV58,
    KnowledgeUnitV58,
    S1ExplorationBudgetV58,
    TransferAssessmentDraftV58,
    TransferAssessmentV58,
    TransferDraftV58,
    TransferHypothesisV58,
)
from fma.v6.executable_candidate import (
    EXECUTABLE_CANDIDATE_INTENT_PATH,
    EXECUTABLE_CANDIDATE_IR_PATH,
    RegisteredFamilySearchIntentV62,
    compile_registered_family_search_ir_v62,
)
from fma.v6.measurement_study_design import (
    MEASUREMENT_STUDY_DESIGN_PATH_V67,
    MeasurementStudyDesignContractV67,
)
from fma.v6.predata_protocol import (
    CANDIDATE_EXECUTION_BINDING_PATH_V67,
    PREDATA_EXECUTION_PROTOCOL_PATH_V67,
    CandidateExecutionBindingV67,
    PreDataExecutionProtocolV67,
    bind_candidate_to_predata_protocol_v67,
    registered_positive_series_capability_pack_v67,
    verify_predata_execution_protocol_v67,
)
from fma.v6.public_source import (
    SOURCE_CONTRACT_PATH,
    WorldBankSourceContractV62,
)
from fma.v6.s1_review_recovery import (
    S1BoundedRepairContextV67,
    S1FormalizationRejectionHandoffV67,
    build_s1_formalization_rejection_handoff_v67,
)


EventCallback = Callable[
    [
        str,
        Literal["accepted", "running", "succeeded", "failed", "blocked"],
        str,
        dict[str, Any],
    ],
    None,
]
DriverFactory = Callable[[], StageRoleDriverV51]


REQUIRED_VALIDATION_IDS = [
    "s3_l0_replay",
    "s3_l1_structural",
    "s3_l2_numerical",
    "s4_l3_holdout",
    "s4_l4_uncertainty",
]
VALIDATION_RULE_ARTIFACTS = {
    check_id: f"validation_rule_{check_id}"
    for check_id in REQUIRED_VALIDATION_IDS
}


class S1RuntimeError(RuntimeError):
    pass


class S1FormalizationRejectedV67(S1RuntimeError):
    """Typed V6.7 handoff from independent S1 rejection to graph recovery."""

    recovery_category: Literal["review_rejection"] = "review_rejection"
    recovery_failure_code: Literal["s1_formalization_review_rejected"] = (
        "s1_formalization_review_rejected"
    )

    def __init__(
        self,
        *,
        findings: list[str],
        reviewer_receipt_hash: str,
        protocol_hash: str,
        handoff_artifact_hash: str | None = None,
    ) -> None:
        normalized = self.normalize_findings(findings)
        if handoff_artifact_hash is not None and (
            len(handoff_artifact_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in handoff_artifact_hash
            )
        ):
            raise ValueError("S1 rejection handoff artifact hash is invalid")
        self._findings = tuple(findings)
        self._normalized_findings = normalized
        self._reviewer_receipt_hash = reviewer_receipt_hash
        self._protocol_hash = protocol_hash
        self._handoff_artifact_hash = handoff_artifact_hash
        self._normalized_finding_signature = self.finding_signature(normalized)
        super().__init__(
            "V6.7 S1 formalization was rejected after bounded repair "
            f"(finding_signature={self._normalized_finding_signature})"
        )

    @staticmethod
    def normalize_findings(findings: list[str]) -> tuple[str, ...]:
        normalized = {
            " ".join(item.split()).casefold()
            for item in findings
            if item.strip()
        }
        if not normalized:
            normalized = {"reviewer rejected without a normalized finding"}
        return tuple(sorted(normalized))

    @staticmethod
    def finding_signature(normalized_findings: tuple[str, ...]) -> str:
        return sha256_value(
            {
                "schema_version": "6.7-s1-formalization-rejection",
                "stage": "S1",
                "recovery_category": "review_rejection",
                "normalized_findings": list(normalized_findings),
            }
        )

    @property
    def findings(self) -> tuple[str, ...]:
        return self._findings

    @property
    def normalized_findings(self) -> tuple[str, ...]:
        return self._normalized_findings

    @property
    def normalized_finding_signature(self) -> str:
        return self._normalized_finding_signature

    @property
    def reviewer_receipt_hash(self) -> str:
        return self._reviewer_receipt_hash

    @property
    def protocol_hash(self) -> str:
        return self._protocol_hash

    @property
    def handoff_artifact_hash(self) -> str | None:
        return self._handoff_artifact_hash


class LiteratureMapDraftV58(StrictModel):
    scope: Literal["supplied_public_inputs_only"]
    candidate_family_hints: list[str] = Field(min_length=3, max_length=12)
    source_claims_verified: Literal[False] = False
    limitations: list[str] = Field(min_length=1, max_length=12)


class CandidateCoreDraftV58(StrictModel):
    """Model-owned mathematical content before harness reference binding."""

    candidate_id: Identifier
    model_family: str = Field(min_length=3, max_length=200)
    mathematical_form: str = Field(min_length=5, max_length=2000)
    data_requirement_ids: list[Identifier] = Field(min_length=1, max_length=8)
    abandon_criteria: list[str] = Field(min_length=1, max_length=8)
    lineage: str = Field(min_length=3, max_length=1000)


class CandidateStructureDraftV58(StrictModel):
    candidate_id: Identifier
    model_family: str = Field(min_length=3, max_length=200)
    data_requirement_ids: list[Identifier] = Field(min_length=1, max_length=8)
    abandon_criteria: list[str] = Field(min_length=1, max_length=8)
    lineage: str = Field(min_length=3, max_length=1000)


class CandidateMathematicalFormDraftV58(StrictModel):
    candidate_id: Identifier
    mathematical_form: str = Field(min_length=5, max_length=2600)


class TransferCoreDraftV58(StrictModel):
    """Model-owned translation before harness target/source normalization."""

    transfer_id: Identifier
    source_unit_ids: list[Identifier] = Field(min_length=1, max_length=4)
    target_interpretation: str = Field(min_length=10, max_length=1500)
    proposed_modification: str = Field(min_length=10, max_length=1500)
    falsification_test: str = Field(min_length=5, max_length=1000)


class S1SelectionDraftV58(StrictModel):
    selected_candidate_id: Identifier
    selection_rationale: str = Field(min_length=10, max_length=2000)
    declared_conservation_laws: list[str] = Field(default_factory=list, max_length=12)
    declared_limit_cases: list[str] = Field(min_length=2, max_length=12)
    identifiability_risks: list[str] = Field(min_length=1, max_length=12)


class ValidationRuleDraftV58(StrictModel):
    check_id: Identifier
    applicability_rule: str = Field(min_length=40, max_length=1200)


class ValidationRulesDraftV58(StrictModel):
    rules: list[ValidationRuleDraftV58] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def validate_rules(self) -> "ValidationRulesDraftV58":
        check_ids = [item.check_id for item in self.rules]
        if check_ids != REQUIRED_VALIDATION_IDS:
            raise ValueError(
                "validation rules must cover the ordered harness check registry"
            )
        return self


@dataclass(frozen=True)
class BranchResultV58:
    branch_id: str
    candidate: CandidateFormalizationV50
    assumptions: AssumptionSetV50
    symbols: SymbolTableV50
    knowledge_drafts: list[KnowledgeDraftV58]
    outcome: RoleProcessOutcomeV51


class _CallBudget:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0
        self._lock = threading.Lock()

    def consume(self) -> None:
        with self._lock:
            if self.used >= self.limit:
                raise S1RuntimeError("S1 model-call budget exhausted")
            self.used += 1


def _json_value(text: str, label: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise S1RuntimeError(f"{label} is not valid JSON") from exc


def _artifact_map(outcome: RoleProcessOutcomeV51) -> dict[str, str]:
    artifacts = {
        item.artifact_type: item.content for item in outcome.draft.proposed_artifacts
    }
    if len(artifacts) != len(outcome.draft.proposed_artifacts):
        raise S1RuntimeError("role returned duplicate artifact types")
    return artifacts


def _write_json_new(path: Path, payload: object) -> None:
    if path.exists():
        raise S1RuntimeError(f"refusing to overwrite S1 artifact: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


class StudioS1OrchestratorV58:
    """A real S1 slice: blind fan-out, controlled exchange, and dual review."""

    def __init__(
        self,
        *,
        workspace: StageWorkspaceV50,
        task_id: str,
        driver_factory: DriverFactory,
        event_callback: EventCallback,
        budget: S1ExplorationBudgetV58 | None = None,
        measurement_contract_v67: MeasurementStudyDesignContractV67 | None = None,
        predata_protocol_v67: PreDataExecutionProtocolV67 | None = None,
        repair_context_v67: S1BoundedRepairContextV67 | None = None,
    ) -> None:
        self.workspace = workspace
        self.task_id = task_id
        self.driver_factory = driver_factory
        self.event_callback = event_callback
        self.budget = budget or S1ExplorationBudgetV58()
        self.call_budget = _CallBudget(self.budget.max_model_calls)
        self.s0_gate = workspace.current_gate("S0")
        if self.s0_gate is None:
            raise S1RuntimeError("S1 requires an open current S0 gate")
        if (measurement_contract_v67 is None) != (predata_protocol_v67 is None):
            raise S1RuntimeError(
                "V6.7 S1 requires both the measurement contract and pre-data "
                "execution protocol"
            )
        self.measurement_contract_v67 = measurement_contract_v67
        self.predata_protocol_v67 = predata_protocol_v67
        self.repair_context_v67 = repair_context_v67
        self.source_contract_v62: WorldBankSourceContractV62 | None = None
        self._v67_protocol_verification: dict[str, Any] | None = None
        if self.measurement_contract_v67 is not None:
            self._assert_v67_predata_integrity()
        elif self.repair_context_v67 is not None:
            raise S1RuntimeError(
                "V6.7 S1 repair context requires the sealed pre-data pair"
            )

    @property
    def v67_predata_enabled(self) -> bool:
        return self.measurement_contract_v67 is not None

    def _assert_v67_predata_integrity(self) -> None:
        contract = self.measurement_contract_v67
        protocol = self.predata_protocol_v67
        if contract is None or protocol is None:
            raise S1RuntimeError(
                "V6.7 S1 pre-data artifacts are incomplete"
            )
        try:
            contract.assert_sealed()
            protocol.assert_sealed()
        except (TypeError, ValueError) as exc:
            raise S1RuntimeError(
                "V6.7 S1 requires sealed pre-data artifacts"
            ) from exc

        workspace_hash = self.workspace.spec.spec_hash
        if workspace_hash is None:
            raise S1RuntimeError("V6.7 S1 requires a sealed workspace spec")
        if (
            contract.workspace_spec_hash != workspace_hash
            or protocol.workspace_spec_hash != workspace_hash
        ):
            raise S1RuntimeError(
                "V6.7 pre-data artifacts reference another workspace spec"
            )
        if (
            contract.s0_gate_hash != self.s0_gate
            or protocol.s0_gate_hash != self.s0_gate
        ):
            raise S1RuntimeError(
                "V6.7 pre-data artifacts reference a stale or foreign S0 gate"
            )
        if (
            contract.contract_hash is None
            or protocol.measurement_contract_id != contract.contract_id
            or protocol.measurement_contract_hash != contract.contract_hash
        ):
            raise S1RuntimeError(
                "V6.7 protocol is not bound to the supplied measurement contract"
            )
        if self.repair_context_v67 is not None:
            try:
                self.repair_context_v67.assert_sealed()
            except (TypeError, ValueError) as exc:
                raise S1RuntimeError(
                    "V6.7 S1 repair context is not sealed"
                ) from exc
            if (
                self.repair_context_v67.workspace_spec_hash != workspace_hash
                or self.repair_context_v67.s0_gate_hash != self.s0_gate
                or self.repair_context_v67.predata_protocol_hash
                != protocol.protocol_hash
                or self.repair_context_v67.successor_attempt
                != self.workspace._latest_attempt("S1")
            ):
                raise S1RuntimeError(
                    "V6.7 S1 repair context is stale or belongs to another "
                    "workspace, protocol, or graph attempt"
                )

        try:
            capability_pack = registered_positive_series_capability_pack_v67(
                protocol.adapter_binding.adapter_id
            )
        except (TypeError, ValueError) as exc:
            raise S1RuntimeError(
                "V6.7 protocol references an unavailable capability pack"
            ) from exc
        if not verify_predata_execution_protocol_v67(
            measurement_contract=contract,
            capability_pack=capability_pack,
            protocol=protocol,
        ):
            raise S1RuntimeError(
                "V6.7 protocol failed deterministic compiler replay"
            )

        source_path = self.workspace.root / SOURCE_CONTRACT_PATH
        if not source_path.is_file():
            raise S1RuntimeError(
                f"V6.7 S1 artifact is absent: {SOURCE_CONTRACT_PATH}"
            )
        try:
            source_contract = WorldBankSourceContractV62.model_validate(
                json.loads(source_path.read_text(encoding="utf-8"))
            )
            source_contract.assert_sealed()
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise S1RuntimeError(
                f"V6.7 S1 artifact cannot be replayed: {SOURCE_CONTRACT_PATH}"
            ) from exc
        if (
            source_contract.contract_id != contract.source_contract_id
            or source_contract.contract_hash != contract.source_contract_hash
            or protocol.source_contract_id != source_contract.contract_id
            or protocol.source_contract_hash != source_contract.contract_hash
        ):
            raise S1RuntimeError(
                "V6.7 source, measurement, and execution contracts differ"
            )
        self.source_contract_v62 = source_contract

        artifact_models = (
            (
                MEASUREMENT_STUDY_DESIGN_PATH_V67,
                MeasurementStudyDesignContractV67,
                contract,
            ),
            (
                PREDATA_EXECUTION_PROTOCOL_PATH_V67,
                PreDataExecutionProtocolV67,
                protocol,
            ),
        )
        for relative_path, model_type, expected in artifact_models:
            path = self.workspace.root / relative_path
            if not path.is_file():
                raise S1RuntimeError(
                    f"V6.7 S1 artifact is absent: {relative_path}"
                )
            try:
                persisted = model_type.model_validate(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                raise S1RuntimeError(
                    f"V6.7 S1 artifact cannot be replayed: {relative_path}"
                ) from exc
            if persisted != expected:
                raise S1RuntimeError(
                    f"V6.7 supplied and persisted artifacts differ: {relative_path}"
                )

        self._v67_protocol_verification = {
            "schema_version": "6.7-s1-predata-protocol-verification",
            "verification_method": (
                "deterministic_recompile_from_sealed_measurement_contract_"
                "and_code_registered_capability_pack"
            ),
            "replay_verified": True,
            "workspace_spec_hash": workspace_hash,
            "s0_gate_hash": self.s0_gate,
            "measurement_contract_hash": contract.contract_hash,
            "protocol_hash": protocol.protocol_hash,
            "source_contract_hash": source_contract.contract_hash,
            "capability_pack_hash": protocol.adapter_binding.capability_pack_hash,
            "selected_adapter_id": protocol.adapter_binding.adapter_id,
            "selected_adapter_version": protocol.adapter_binding.adapter_version,
            "required_sections_verified": [
                "adapter_binding",
                "adapter_resolution",
                "compatibility",
                "thresholds",
                "fitting",
                "training_and_confirmation",
                "uncertainty_and_stress",
                "level_rules",
                "abstention",
            ],
            "ordered_level_rules_verified": [
                item.level for item in protocol.level_rules
            ],
            "observation_values_accessed": False,
            "scientific_qualification_granted": False,
            "real_world_action_authorized": False,
        }

    def _commit_s1_rejection_handoff_v67(
        self,
        *,
        findings: list[str],
        reviewer_receipt_hash: str,
    ) -> str:
        protocol = self.predata_protocol_v67
        workspace_hash = self.workspace.spec.spec_hash
        if (
            protocol is None
            or protocol.protocol_hash is None
            or workspace_hash is None
            or self.s0_gate is None
        ):
            raise S1RuntimeError(
                "V6.7 S1 rejection handoff requires current pre-data authority"
            )
        existing_context_hash = (
            str(self.repair_context_v67.context_hash)
            if self.repair_context_v67 is not None
            and self.repair_context_v67.context_hash is not None
            else None
        )
        handoff = build_s1_formalization_rejection_handoff_v67(
            workspace_spec_hash=workspace_hash,
            s0_gate_hash=self.s0_gate,
            predata_protocol_hash=protocol.protocol_hash,
            reviewer_receipt_hash=reviewer_receipt_hash,
            findings=findings,
            predecessor_attempt=self.workspace._latest_attempt("S1"),
            existing_repair_context_hash=existing_context_hash,
        )
        matches: list[str] = []
        for reference, payload in self.workspace._artifacts_of_kind(
            "s1_formalization_rejection_handoff_v67"
        ):
            try:
                existing = (
                    S1FormalizationRejectionHandoffV67.model_validate(payload)
                )
                existing.assert_sealed()
            except (TypeError, ValueError) as exc:
                raise S1RuntimeError(
                    "committed V6.7 S1 rejection handoff is invalid"
                ) from exc
            if (
                existing.workspace_spec_hash == workspace_hash
                and existing.s0_gate_hash == self.s0_gate
                and existing.predata_protocol_hash == protocol.protocol_hash
                and existing.predecessor_attempt
                == handoff.predecessor_attempt
            ):
                if existing.handoff_hash != handoff.handoff_hash:
                    raise S1RuntimeError(
                        "another S1 rejection handoff owns this graph attempt"
                    )
                matches.append(reference.sha256)
        if len(matches) > 1:
            raise S1RuntimeError(
                "multiple S1 rejection handoffs own this graph attempt"
            )
        if matches:
            return matches[0]
        reference = self.workspace.commit_evidence(
            "s1_formalization_rejection_handoff_v67",
            handoff.model_dump(mode="json"),
        )
        return reference.sha256

    def _event(
        self,
        event_type: str,
        status: Literal["accepted", "running", "succeeded", "failed", "blocked"],
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.event_callback(event_type, status, message, details or {})

    def _run_role(
        self,
        *,
        role_name: str,
        role_kind: Literal["generator", "reviewer"],
        subject_id: str,
        objective: str,
        public_inputs: dict[str, Any],
        allowed_candidate_ids: list[str],
    ) -> RoleProcessOutcomeV51:
        self.call_budget.consume()
        return self.driver_factory().run(
            task_id=self.task_id,
            stage="S1",
            role_name=role_name,
            role_kind=role_kind,
            subject_id=subject_id,
            objective=objective,
            public_inputs=public_inputs,
            allowed_candidate_ids=allowed_candidate_ids,
        )

    def _s0_public_context(self) -> dict[str, Any]:
        context = {
            "objective": self.workspace.spec.objective,
            "mission_hash": self.workspace.spec.mission_hash,
            "evidence_snapshot_hash": self.workspace.spec.evidence_snapshot_hash,
            "evidence_scope": self.workspace.spec.evidence_scope,
            "s0_gate_hash": self.s0_gate,
            "problem_contract": json.loads(
                (self.workspace.root / "problem" / "contract.json").read_text(
                    encoding="utf-8"
                )
            ),
            "decision_function": json.loads(
                (self.workspace.root / "problem" / "decision_function.json").read_text(
                    encoding="utf-8"
                )
            ),
            "regime": json.loads(
                (self.workspace.root / "docs" / "regime.json").read_text(
                    encoding="utf-8"
                )
            ),
            "available_execution_adapters": [
                {
                    "adapter_id": "scalar_autonomous_ode_v52",
                    "applicability": (
                        "Strictly increasing times with at least twelve finite "
                        "positive scalar observations."
                    ),
                    "registered_families": [
                        "constant",
                        "exponential",
                        "gompertz",
                        "logistic",
                    ],
                    "required_model_family_marker": (
                        "ODE, ordinary differential, or autonomous differential"
                    ),
                    "required_mathematical_marker": "dx/dt",
                    "authority_note": (
                        "This is an available executor, not a requirement to "
                        "select it and not scientific qualification."
                    ),
                }
            ],
        }
        if self.v67_predata_enabled:
            self._assert_v67_predata_integrity()
            assert self.measurement_contract_v67 is not None
            assert self.predata_protocol_v67 is not None
            assert self.source_contract_v62 is not None
            assert self._v67_protocol_verification is not None
            protocol = self.predata_protocol_v67
            context.update(
                {
                    "measurement_study_design_contract_v67": (
                        self.measurement_contract_v67.model_dump(mode="json")
                    ),
                    "source_contract_v62": self.source_contract_v62.model_dump(
                        mode="json"
                    ),
                    "predata_execution_protocol_v67": protocol.model_dump(
                        mode="json"
                    ),
                    "predata_protocol_mechanical_verification_v67": (
                        self._v67_protocol_verification
                    ),
                    "available_execution_adapters": [
                        {
                            "adapter_id": protocol.adapter_binding.adapter_id,
                            "adapter_version": (
                                protocol.adapter_binding.adapter_version
                            ),
                            "capability_pack_hash": (
                                protocol.adapter_binding.capability_pack_hash
                            ),
                            "registered_families": [
                                item
                                for item in protocol.adapter_binding.allowed_families
                            ],
                            "authority_note": (
                                "This exact adapter was frozen by the code-owned "
                                "pre-data compiler. S1 may test candidate "
                                "compatibility but may not replace, broaden, or "
                                "reinterpret the protocol."
                            ),
                        }
                    ],
                }
            )
        return context

    def _parse_branch(
        self, branch_id: str, outcome: RoleProcessOutcomeV51
    ) -> BranchResultV58:
        if outcome.draft.authority_claimed:
            raise S1RuntimeError(f"{branch_id} claimed reserved authority")
        artifacts = _artifact_map(outcome)
        required = {"candidate", "assumptions", "symbols", "knowledge_units"}
        if set(artifacts) != required:
            raise S1RuntimeError(
                f"{branch_id} must return exactly {sorted(required)}; "
                f"received {sorted(artifacts)}"
            )
        candidate_core = CandidateCoreDraftV58.model_validate(
            _json_value(artifacts["candidate"], f"{branch_id}.candidate")
        )
        expected_candidate_id = f"candidate.{branch_id}"
        if candidate_core.candidate_id != expected_candidate_id:
            raise S1RuntimeError(
                f"{branch_id} candidate_id must be {expected_candidate_id}"
            )
        assumptions = AssumptionSetV50.model_validate(
            {
                "schema_version": "5.0",
                "assumptions": _json_value(
                    artifacts["assumptions"], f"{branch_id}.assumptions"
                ),
            }
        )
        symbols = SymbolTableV50.model_validate(
            {
                "schema_version": "5.0",
                "symbols": _json_value(artifacts["symbols"], f"{branch_id}.symbols"),
            }
        )
        candidate = CandidateFormalizationV50(
            candidate_id=candidate_core.candidate_id,
            model_family=candidate_core.model_family,
            mathematical_form=candidate_core.mathematical_form,
            assumption_ids=sorted(
                item.assumption_id for item in assumptions.assumptions
            ),
            symbol_ids=sorted(item.symbol_id for item in symbols.symbols),
            data_requirement_ids=sorted(set(candidate_core.data_requirement_ids)),
            validation_obligation_ids=REQUIRED_VALIDATION_IDS,
            abandon_criteria=candidate_core.abandon_criteria,
            lineage=candidate_core.lineage,
        )
        raw_knowledge = _json_value(
            artifacts["knowledge_units"], f"{branch_id}.knowledge_units"
        )
        if not isinstance(raw_knowledge, list) or not raw_knowledge:
            raise S1RuntimeError(f"{branch_id} must publish knowledge units")
        knowledge = [KnowledgeDraftV58.model_validate(item) for item in raw_knowledge]
        if any(not item.unit_id.startswith(f"{branch_id}.") for item in knowledge):
            raise S1RuntimeError(f"{branch_id} knowledge IDs must be branch-namespaced")
        return BranchResultV58(
            branch_id=branch_id,
            candidate=candidate,
            assumptions=assumptions,
            symbols=symbols,
            knowledge_drafts=knowledge,
            outcome=outcome,
        )

    def _run_branch(self, branch_id: str) -> BranchResultV58:
        candidate_id = f"candidate.{branch_id}"
        base_inputs = {
            **self._s0_public_context(),
            "branch_id": branch_id,
            "candidate_id": candidate_id,
            "blind_generation": True,
            "peer_branch_knowledge_visible": False,
            "required_artifacts": {
                "candidate": CandidateCoreDraftV58.model_json_schema(),
                "assumptions": {
                    "type": "array",
                    "items": AssumptionRecordV50.model_json_schema(),
                },
                "symbols": {
                    "type": "array",
                    "items": SymbolRecordV50.model_json_schema(),
                },
                "knowledge_units": {
                    "type": "array",
                    "items": KnowledgeDraftV58.model_json_schema(),
                },
            },
            "requirements": [
                "Return exactly candidate, assumptions, symbols, knowledge_units.",
                f"Use candidate_id {candidate_id}.",
                f"Namespace assumption, symbol, and knowledge IDs with {branch_id}.",
                "Keep every identifier list sorted and unique.",
                "The harness binds assumption, symbol, and validation IDs into the candidate.",
                "Do not claim empirical support; no S2 data has been frozen.",
                "Make the candidate structurally distinct from generic alternatives.",
                "Keep mathematical_form complete and self-contained within 1600 characters.",
                "Each knowledge unit needs conditions and a falsification test.",
            ],
        }
        validation_error: str | None = None
        for attempt in (1, 2):
            inputs = dict(base_inputs)
            if validation_error is not None:
                inputs["repair"] = {
                    "validation_error": validation_error[:700],
                    "instruction": "Return a complete corrected replacement.",
                }
            try:
                outcome = self._run_role(
                    role_name=(
                        f"s1_{branch_id}_blind"
                        if attempt == 1
                        else f"s1_{branch_id}_repair"
                    ),
                    role_kind="generator",
                    subject_id=candidate_id,
                    objective=(
                        f"Develop one falsifiable {branch_id} modelling branch "
                        "without seeing peer branch conclusions."
                    ),
                    public_inputs=inputs,
                    allowed_candidate_ids=[candidate_id],
                )
            except S1RuntimeError:
                raise
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                validation_error = f"transport {type(exc).__name__}: {str(exc)[:400]}"
                self._event(
                    "s1_branch_transport_failed",
                    "blocked",
                    f"{branch_id} role process failed before a valid receipt",
                    {
                        "branch_id": branch_id,
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "failure_signature": validation_error,
                    },
                )
                if attempt == 2:
                    raise S1RuntimeError(
                        f"{branch_id} exhausted its transport retry budget"
                    ) from exc
                continue
            try:
                branch = self._parse_branch(branch_id, outcome)
                if attempt == 2:
                    self._event(
                        "s1_branch_repaired",
                        "succeeded",
                        f"{branch_id} passed typed validation after one repair",
                        {
                            "branch_id": branch_id,
                            "attempt": attempt,
                            "run_id": outcome.request.run_id,
                            "output_hash": outcome.receipt.output_hash,
                        },
                    )
                return branch
            except (ValueError, S1RuntimeError) as exc:
                validation_error = str(exc)
                self._event(
                    "s1_branch_attempt_rejected",
                    "blocked",
                    f"{branch_id} output failed typed validation",
                    {
                        "branch_id": branch_id,
                        "attempt": attempt,
                        "run_id": outcome.request.run_id,
                        "output_hash": outcome.receipt.output_hash,
                        "failure_signature": validation_error[:500],
                    },
                )
                if attempt == 2:
                    raise S1RuntimeError(
                        f"{branch_id} exhausted its typed repair budget"
                    ) from exc
        raise AssertionError("unreachable branch repair state")

    def _commit_role(
        self,
        outcome: RoleProcessOutcomeV51,
        *,
        execution_role: Literal["modeler", "literature_scout", "writer"],
        subject_id: str,
        output_payload: dict[str, Any],
    ):
        trace = self.workspace.commit_evidence(
            "codex_role_transport_trace_v58",
            {
                "role": execution_role,
                "role_name": outcome.request.role_name,
                "subject_id": subject_id,
                "input_authority_hash": self.s0_gate,
                "run_id": outcome.request.run_id,
                "context_id": outcome.request.context_id,
                "request_hash": outcome.request.request_hash,
                "process_receipt": outcome.receipt.model_dump(mode="json"),
            },
        )
        output = self.workspace.commit_evidence(
            "codex_role_output_v58",
            {
                **output_payload,
                "stage": "S1",
                "role": execution_role,
                "role_name": outcome.request.role_name,
                "request_hash": outcome.request.request_hash,
                "draft_hash": outcome.receipt.output_hash,
            },
        )
        receipt = self.workspace.issue_role_execution(
            stage="S1",
            execution_id=f"exec-{outcome.request.run_id}",
            role=execution_role,
            subject_id=subject_id,
            input_authority_hash=self.s0_gate,
            run_id=outcome.request.run_id,
            context_id=outcome.request.context_id,
            provider=outcome.receipt.provider,
            model=outcome.receipt.requested_model or "served_model_unattested",
            prompt_hash=outcome.receipt.prompt_hash,
            output_schema_hash=outcome.receipt.output_schema_hash,
            transport_trace_hash=trace.sha256,
            output_artifact_hash=output.sha256,
        )
        return receipt, output

    def _run_scout(self) -> tuple[RoleProcessOutcomeV51, LiteratureMapDraftV58]:
        outcome = self._run_role(
            role_name="s1_prior_model_scout",
            role_kind="generator",
            subject_id="literature-map",
            objective=(
                "Map plausible classical model families from supplied public "
                "context only, with no claim of live literature verification."
            ),
            public_inputs={
                **self._s0_public_context(),
                "external_search_permitted": False,
                "required_artifact": {
                    "literature_map": LiteratureMapDraftV58.model_json_schema()
                },
                "requirements": [
                    "Return exactly one literature_map artifact.",
                    "Set scope to supplied_public_inputs_only.",
                    "Set source_claims_verified to false.",
                    "Include at least three family hints and explicit limitations.",
                ],
            },
            allowed_candidate_ids=[],
        )
        artifacts = _artifact_map(outcome)
        if set(artifacts) != {"literature_map"}:
            raise S1RuntimeError("scout must return exactly literature_map")
        report = LiteratureMapDraftV58.model_validate(
            _json_value(artifacts["literature_map"], "literature_map")
        )
        return outcome, report

    def _run_translator(
        self,
        *,
        graph_hash: str,
        packet: DisclosurePacketV58,
        units_by_hash: dict[str, KnowledgeUnitV58],
    ) -> tuple[RoleProcessOutcomeV51, TransferDraftV58]:
        target_id = packet.recipient_branch_id
        outcome = self._run_role(
            role_name=f"s1_cross_paradigm_translator_{target_id}",
            role_kind="generator",
            subject_id=f"translator.{target_id}",
            objective=(
                f"Translate disclosed peer findings for {target_id} without "
                "converting analogy or consensus into evidence."
            ),
            public_inputs={
                "s0_gate_hash": self.s0_gate,
                "source_graph_hash": graph_hash,
                "disclosure_packet": packet.model_dump(mode="json"),
                "disclosed_units": [
                    units_by_hash[item].model_dump(mode="json")
                    for item in packet.disclosed_unit_hashes
                ],
                "allowed_target_branch_id": target_id,
                "max_transfer_hypotheses": 1,
                "required_artifact": {
                    "transfer_hypotheses": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": TransferCoreDraftV58.model_json_schema(),
                    }
                },
                "requirements": [
                    "Return exactly one transfer_hypotheses artifact.",
                    "Use only disclosed source unit IDs.",
                    f"Return exactly one hypothesis targeting {target_id}.",
                    "Every translation is an unverified hypothesis.",
                    "Include a concrete target-side falsification test.",
                ],
            },
            allowed_candidate_ids=[f"candidate.{target_id}"],
        )
        artifacts = _artifact_map(outcome)
        if set(artifacts) != {"transfer_hypotheses"}:
            raise S1RuntimeError("translator must return exactly transfer_hypotheses")
        raw = _json_value(artifacts["transfer_hypotheses"], "transfer_hypotheses")
        if not isinstance(raw, list) or len(raw) != 1:
            raise S1RuntimeError("translator must return exactly one hypothesis")
        core = TransferCoreDraftV58.model_validate(raw[0])
        draft = TransferDraftV58(
            transfer_id=core.transfer_id,
            source_unit_ids=sorted(set(core.source_unit_ids)),
            target_branch_id=target_id,
            target_interpretation=core.target_interpretation,
            proposed_modification=core.proposed_modification,
            falsification_test=core.falsification_test,
        )
        return outcome, draft

    def _run_mentor(
        self,
        *,
        branch: BranchResultV58,
        packet: DisclosurePacketV58,
        transfers: list[TransferHypothesisV58],
        units_by_hash: dict[str, KnowledgeUnitV58],
    ) -> tuple[RoleProcessOutcomeV51, list[TransferAssessmentDraftV58]]:
        relevant = [
            item for item in transfers if item.target_branch_id == branch.branch_id
        ]
        if not relevant:
            raise S1RuntimeError(
                f"no transfer targets scheduled branch {branch.branch_id}"
            )
        outcome = self._run_role(
            role_name=f"s1_{branch.branch_id}_recipient",
            role_kind="generator",
            subject_id=f"mentor.{branch.branch_id}",
            objective=(
                "Assess whether translated peer knowledge is coherent enough to "
                "test in this branch; do not claim scientific support."
            ),
            public_inputs={
                "branch_id": branch.branch_id,
                "candidate": branch.candidate.model_dump(mode="json"),
                "disclosure_packet": packet.model_dump(mode="json"),
                "disclosed_units": [
                    units_by_hash[item].model_dump(mode="json")
                    for item in packet.disclosed_unit_hashes
                ],
                "transfer_hypotheses": [
                    item.model_dump(mode="json") for item in relevant
                ],
                "required_artifact": {
                    "transfer_assessments": {
                        "type": "array",
                        "items": TransferAssessmentDraftV58.model_json_schema(),
                    }
                },
                "requirements": [
                    "Assess every targeted transfer exactly once.",
                    "ACCEPT_FOR_TEST means coherent but scientifically unverified.",
                    "Name the later S2-S4 test required to support or reject it.",
                ],
            },
            allowed_candidate_ids=[branch.candidate.candidate_id],
        )
        artifacts = _artifact_map(outcome)
        if set(artifacts) != {"transfer_assessments"}:
            raise S1RuntimeError(
                f"{branch.branch_id} recipient returned wrong artifacts"
            )
        raw = _json_value(
            artifacts["transfer_assessments"],
            f"{branch.branch_id}.transfer_assessments",
        )
        if not isinstance(raw, list):
            raise S1RuntimeError("transfer assessments must be an array")
        drafts = [TransferAssessmentDraftV58.model_validate(item) for item in raw]
        expected_ids = sorted(item.transfer_id for item in relevant)
        actual_ids = sorted(item.transfer_id for item in drafts)
        if actual_ids != expected_ids or len(actual_ids) != len(set(actual_ids)):
            raise S1RuntimeError(
                f"{branch.branch_id} did not assess every targeted transfer"
            )
        return outcome, drafts

    def _run_synthesizer(
        self,
        *,
        branches: list[BranchResultV58],
        epistemic_summary: dict[str, Any],
        repair: dict[str, Any] | None = None,
    ) -> tuple[
        RoleProcessOutcomeV51,
        S1SelectionDraftV58,
        ValidationRulesDraftV58,
        CandidateStructureDraftV58,
        CandidateMathematicalFormDraftV58,
        RegisteredFamilySearchIntentV62,
    ]:
        candidate_ids = sorted(item.candidate.candidate_id for item in branches)
        if self.v67_predata_enabled:
            refinement_rule = (
                "Return complete selected_candidate_structure and "
                "selected_mathematical_form artifacts plus one typed "
                "executable_candidate_intent with the same candidate_id as "
                "selection. The sealed V6.7 pre-data protocol already fixes "
                "the exact adapter, registered families, feature and fitting "
                "semantics, optimizer, split, baselines, dependence limits, "
                "uncertainty, stress tests, and L0-L4 implementations. Do not "
                "replace, broaden, or silently reinterpret those code-owned "
                "rules. Refine the candidate mathematics and its assumptions "
                "so they are compatible with the exact protocol, or preserve "
                "an explicit incompatibility as a finding."
            )
            validation_rule_requirements = [
                "Return one candidate-specific applicability rule for each "
                "check_id in the supplied order.",
                "Treat the sealed V6.7 protocol as authoritative for executable "
                "thresholds, fitting, baselines, dependence, uncertainty, "
                "stress, and L0-L4 implementation details; do not restate or "
                "weaken them.",
                "State how each exact registered check applies to this "
                "candidate, including any contradiction or evidence condition, "
                "and keep the outcome decidable as PASS, FAIL, NOT_RUN, or HUMAN.",
                "L0-L2 cannot claim empirical performance; L3-L4 cannot claim "
                "results before the frozen data and confirmation evidence exist.",
            ]
        else:
            refinement_rule = (
                "Return complete selected_candidate_structure and "
                "selected_mathematical_form artifacts plus one typed "
                "executable_candidate_intent with the same candidate_id as "
                "selection. Resolve feature construction, estimation, "
                "dependence, and decision details enough for an independent S1 "
                "structural audit. The harness preserves the selected blind "
                "branch's assumption and symbol registries."
            )
            validation_rule_requirements = [
                "Return one rule for each check_id in the supplied order.",
                "Make rules task-specific, quantitative where meaningful, "
                "dependence-aware, and decidable as PASS, FAIL, NOT_RUN, or HUMAN.",
                "Do not reference an undefined prespecified criterion, stress "
                "distribution, feature set, or uncertainty procedure.",
                "L0-L2 cannot claim empirical performance; L3-L4 must compare "
                "against the best frozen baseline and stress decision stability.",
            ]
        public_inputs = {
            **self._s0_public_context(),
            "candidates": [
                item.candidate.model_dump(mode="json") for item in branches
            ],
            "epistemic_summary": epistemic_summary,
            "selection_rule": (
                "Prefer decision relevance, falsifiability, parsimony, "
                "identifiability, and baseline competitiveness; this is "
                "development selection, not scientific acceptance."
            ),
            "refinement_rule": refinement_rule,
            "validation_rule_requirements": validation_rule_requirements,
            "required_artifacts": {
                "selection": S1SelectionDraftV58.model_json_schema(),
                "selected_candidate_structure": (
                    CandidateStructureDraftV58.model_json_schema()
                ),
                "selected_mathematical_form": (
                    CandidateMathematicalFormDraftV58.model_json_schema()
                ),
                "executable_candidate_intent": (
                    RegisteredFamilySearchIntentV62.model_json_schema()
                ),
                **{
                    artifact_type: ValidationRuleDraftV58.model_json_schema()
                    for artifact_type in VALIDATION_RULE_ARTIFACTS.values()
                },
            },
        }
        if self.repair_context_v67 is not None:
            public_inputs["s1_graph_repair_context_v67"] = (
                self.repair_context_v67.model_dump(mode="json")
            )
            public_inputs["s1_graph_repair_rule_v67"] = (
                "Use only the necessary normalized findings in the sealed "
                "graph repair context. Repair candidate formalization against "
                "the already-frozen protocol. Protocol, adapter, thresholds, "
                "observation scope, and gate authority are immutable."
            )
        if repair is not None:
            public_inputs["repair"] = repair
        outcome = self._run_role(
            role_name=(
                "s1_candidate_synthesizer"
                if repair is None
                else "s1_candidate_synthesizer_repair"
            ),
            role_kind="generator",
            subject_id="s1-selection",
            objective=(
                "Select and fully formalize one development candidate while "
                "preserving alternatives, uncertainty, and falsification duties."
            ),
            public_inputs=public_inputs,
            allowed_candidate_ids=candidate_ids,
        )
        artifacts = _artifact_map(outcome)
        expected_artifacts = {
            "selection",
            "selected_candidate_structure",
            "selected_mathematical_form",
            "executable_candidate_intent",
            *VALIDATION_RULE_ARTIFACTS.values(),
        }
        if set(artifacts) != expected_artifacts:
            raise S1RuntimeError(
                "synthesizer returned an incomplete atomic formalization packet"
            )
        selection = S1SelectionDraftV58.model_validate(
            _json_value(artifacts["selection"], "selection")
        )
        validation_rules = ValidationRulesDraftV58.model_validate(
            {
                "rules": [
                    ValidationRuleDraftV58.model_validate(
                        _json_value(
                            artifacts[VALIDATION_RULE_ARTIFACTS[check_id]],
                            VALIDATION_RULE_ARTIFACTS[check_id],
                        )
                    )
                    for check_id in REQUIRED_VALIDATION_IDS
                ]
            }
        )
        selected_structure = CandidateStructureDraftV58.model_validate(
            _json_value(
                artifacts["selected_candidate_structure"],
                "selected_candidate_structure",
            )
        )
        selected_math = CandidateMathematicalFormDraftV58.model_validate(
            _json_value(
                artifacts["selected_mathematical_form"],
                "selected_mathematical_form",
            )
        )
        execution_intent = RegisteredFamilySearchIntentV62.model_validate(
            _json_value(
                artifacts["executable_candidate_intent"],
                "executable_candidate_intent",
            )
        )
        if selection.selected_candidate_id not in candidate_ids:
            raise S1RuntimeError("synthesizer selected an unknown candidate")
        if {
            selected_structure.candidate_id,
            selected_math.candidate_id,
            execution_intent.candidate_id,
        } != {selection.selected_candidate_id}:
            raise S1RuntimeError(
                "selected candidate artifacts and selection identify different "
                "candidates"
            )
        if outcome.draft.selected_candidate_id != selection.selected_candidate_id:
            raise S1RuntimeError("selection artifact and role draft differ")
        return (
            outcome,
            selection,
            validation_rules,
            selected_structure,
            selected_math,
            execution_intent,
        )

    @staticmethod
    def _refined_candidate(
        branches: list[BranchResultV58],
        selected_structure: CandidateStructureDraftV58,
        selected_math: CandidateMathematicalFormDraftV58,
    ) -> CandidateFormalizationV50:
        original = next(
            item.candidate
            for item in branches
            if item.candidate.candidate_id == selected_structure.candidate_id
        )
        return CandidateFormalizationV50(
            candidate_id=selected_structure.candidate_id,
            model_family=selected_structure.model_family,
            mathematical_form=selected_math.mathematical_form,
            data_requirement_ids=selected_structure.data_requirement_ids,
            assumption_ids=original.assumption_ids,
            symbol_ids=original.symbol_ids,
            validation_obligation_ids=REQUIRED_VALIDATION_IDS,
            abandon_criteria=selected_structure.abandon_criteria,
            lineage=(
                f"{original.lineage} | Graph-guided S1 synthesis refinement: "
                f"{selected_structure.lineage}"
            ),
        )

    def _run_formalization_auditor(
        self,
        *,
        branches: list[BranchResultV58],
        selection: S1SelectionDraftV58,
        validation_rules: ValidationRulesDraftV58,
        selected_structure: CandidateStructureDraftV58,
        selected_math: CandidateMathematicalFormDraftV58,
        execution_intent: RegisteredFamilySearchIntentV62,
        epistemic_review_proof: dict[str, Any],
    ) -> RoleProcessOutcomeV51:
        refined = self._refined_candidate(
            branches,
            selected_structure,
            selected_math,
        )
        candidate_binding_v67: CandidateExecutionBindingV67 | None = None
        if self.v67_predata_enabled:
            assert self.predata_protocol_v67 is not None
            candidate_binding_v67 = bind_candidate_to_predata_protocol_v67(
                candidate=refined,
                execution_intent=execution_intent,
                execution_ir=compile_registered_family_search_ir_v62(
                    refined,
                    execution_intent,
                ),
                protocol=self.predata_protocol_v67,
            )
            audit_objective = (
                "Independently audit whether the selected S1 candidate and its "
                "candidate-specific L0-L4 applicability rules are compatible "
                "with, and do not contradict, the mechanically replay-verified "
                "V6.7 pre-data protocol."
            )
            audit_rule = (
                "The harness has already deterministically replayed the sealed "
                "V6.7 protocol from the sealed measurement contract and exact "
                "code-registered capability pack. Do not ask the model to "
                "restate optimizer, split, baseline, dependence, uncertainty, "
                "stress, threshold, or L0-L4 implementation details. APPROVE "
                "only if the selected mathematics, assumptions, typed intent, "
                "and candidate-specific applicability rules are compatible "
                "with and do not loosen or contradict that exact protocol, and "
                "the harness-owned candidate_execution_binding_v67 resolves "
                "the legacy V6.2 two-adapter envelope to the single pre-data "
                "adapter. The legacy envelope is not S2 resolution authority. "
                "Genuine task-specific limits must remain explicit. Empirical "
                "results are not required at S1; use HUMAN only for an "
                "irreducible user or domain choice."
            )
        else:
            audit_objective = (
                "Audit the selected S1 candidate and L0-L4 rules for "
                "structural completeness before artifact freezing."
            )
            audit_rule = (
                "APPROVE only if executable feature construction, "
                "estimation/dependence semantics, and every L0-L4 "
                "decision rule are fully specified. Empirical results "
                "are not required at S1; use HUMAN only for an "
                "irreducible user or domain choice."
            )
        for attempt in (1, 2):
            try:
                outcome = self._run_role(
                    role_name="s1_formalization_auditor",
                    role_kind="reviewer",
                    subject_id="s1-selection-preflight",
                    objective=audit_objective,
                    public_inputs={
                        **self._s0_public_context(),
                        "selected_candidate": refined.model_dump(mode="json"),
                        "executable_candidate_intent": (
                            execution_intent.model_dump(mode="json")
                        ),
                        **(
                            {
                                "candidate_execution_binding_v67": (
                                    candidate_binding_v67.model_dump(mode="json")
                                )
                            }
                            if candidate_binding_v67 is not None
                            else {}
                        ),
                        "selection": selection.model_dump(mode="json"),
                        "validation_rules": validation_rules.model_dump(mode="json"),
                        "epistemic_review_proof": epistemic_review_proof,
                        "audit_rule": audit_rule,
                        "response_budget": {
                            "maximum_findings": 8,
                            "maximum_rationale_characters": 800,
                        },
                    },
                    allowed_candidate_ids=[
                        item.candidate.candidate_id for item in branches
                    ],
                )
            except S1RuntimeError:
                raise
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                self._event(
                    "s1_preflight_transport_failed",
                    "blocked",
                    "Pre-freeze auditor failed before a valid advisory receipt",
                    {
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "failure_signature": (
                            f"transport {type(exc).__name__}: {str(exc)[:400]}"
                        ),
                    },
                )
                if attempt == 2:
                    raise S1RuntimeError(
                        "pre-freeze auditor exhausted its transport retry budget"
                    ) from exc
                continue
            if attempt == 2:
                self._event(
                    "s1_preflight_transport_recovered",
                    "succeeded",
                    "Pre-freeze auditor returned in a fresh retry process",
                    {
                        "attempt": attempt,
                        "run_id": outcome.request.run_id,
                        "output_hash": outcome.receipt.output_hash,
                    },
                )
            return outcome
        raise AssertionError("unreachable preflight transport retry state")

    def _commit_advisory_review(
        self,
        outcome: RoleProcessOutcomeV51,
        *,
        attempt: int,
    ) -> None:
        rejection_evidence: dict[str, Any] = {}
        if self.v67_predata_enabled and outcome.draft.verdict != "APPROVE":
            assert self.predata_protocol_v67 is not None
            assert self.predata_protocol_v67.protocol_hash is not None
            normalized = S1FormalizationRejectedV67.normalize_findings(
                outcome.draft.findings
            )
            rejection_evidence = {
                "predata_protocol_hash": (
                    self.predata_protocol_v67.protocol_hash
                ),
                "normalized_findings": list(normalized),
                "normalized_finding_signature": (
                    S1FormalizationRejectedV67.finding_signature(normalized)
                ),
                "recovery_category": "review_rejection",
                "rollback_root_selected_by_model": False,
            }
        self.workspace.commit_evidence(
            "codex_advisory_review_v58",
            {
                "stage": "S1",
                "purpose": "pre_freeze_formalization_audit",
                "attempt": attempt,
                "graph_attempt": self.workspace._latest_attempt("S1"),
                "workspace_spec_hash": self.workspace.spec.spec_hash,
                "s0_gate_hash": self.s0_gate,
                "request_hash": outcome.request.request_hash,
                "run_id": outcome.request.run_id,
                "context_id": outcome.request.context_id,
                "verdict": outcome.draft.verdict,
                "findings": outcome.draft.findings,
                "uncertainties": outcome.draft.uncertainties,
                "process_receipt": outcome.receipt.model_dump(mode="json"),
                "gate_authority": False,
                "scientific_support_established": False,
                **rejection_evidence,
            },
        )

    @staticmethod
    def _epistemic_review_proof(
        graph,
    ) -> dict[str, Any]:
        return {
            "origin_separation": graph.independence.model_dump(mode="json"),
            "disclosures": [
                {
                    "packet_hash": item.packet_hash,
                    "recipient_branch_id": item.recipient_branch_id,
                    "source_branch_ids": item.source_branch_ids,
                    "peer_context_only": item.peer_context_only,
                }
                for item in graph.disclosure_packets
            ],
            "transfers": [
                {
                    "transfer_hash": item.transfer_hash,
                    "source_branch_ids": item.source_branch_ids,
                    "target_branch_id": item.target_branch_id,
                    "status": item.status,
                }
                for item in graph.transfers
            ],
            "transfer_assessments": [
                {
                    "assessment_hash": item.assessment_hash,
                    "transfer_hash": item.transfer_hash,
                    "target_branch_id": item.target_branch_id,
                    "verdict": item.verdict,
                    "scientific_support_established": (
                        item.scientific_support_established
                    ),
                }
                for item in graph.transfer_assessments
            ],
        }

    @staticmethod
    def _validation_plan(
        validation_rules: ValidationRulesDraftV58,
    ) -> ValidationPlanV50:
        rules_by_id = {
            item.check_id: item.applicability_rule
            for item in validation_rules.rules
        }
        obligations = [
            ValidationObligationV50(
                check_id="s3_l0_replay",
                stage="S3",
                level="L0",
                evidence_class="scientific_computation",
                applicability_rule=rules_by_id["s3_l0_replay"],
            ),
            ValidationObligationV50(
                check_id="s3_l1_structural",
                stage="S3",
                level="L1",
                evidence_class="scientific_computation",
                applicability_rule=rules_by_id["s3_l1_structural"],
            ),
            ValidationObligationV50(
                check_id="s3_l2_numerical",
                stage="S3",
                level="L2",
                evidence_class="scientific_computation",
                applicability_rule=rules_by_id["s3_l2_numerical"],
            ),
            ValidationObligationV50(
                check_id="s4_l3_holdout",
                stage="S4",
                level="L3",
                evidence_class="scientific_computation",
                applicability_rule=rules_by_id["s4_l3_holdout"],
            ),
            ValidationObligationV50(
                check_id="s4_l4_uncertainty",
                stage="S4",
                level="L4",
                evidence_class="scientific_computation",
                applicability_rule=rules_by_id["s4_l4_uncertainty"],
            ),
        ]
        return ValidationPlanV50.seal(
            obligations=obligations,
            frozen_by="verifier",
        )

    def _materialize_s1(
        self,
        *,
        branches: list[BranchResultV58],
        generation_receipt_hashes: list[str],
        scout_receipt_hash: str,
        selection: S1SelectionDraftV58,
        validation_rules: ValidationRulesDraftV58,
        selected_structure: CandidateStructureDraftV58,
        selected_math: CandidateMathematicalFormDraftV58,
        execution_intent: RegisteredFamilySearchIntentV62,
    ) -> ValidationPlanV50:
        refined_candidate = self._refined_candidate(
            branches,
            selected_structure,
            selected_math,
        )
        candidates = CandidateSetV50(
            candidates=sorted(
                [
                    refined_candidate
                    if item.candidate.candidate_id
                    == selection.selected_candidate_id
                    else item.candidate
                    for item in branches
                ],
                key=lambda item: item.candidate_id,
            ),
            generation_receipt_hashes=sorted(generation_receipt_hashes),
            literature_scout_receipt_hash=scout_receipt_hash,
        )
        assumptions_by_id: dict[str, AssumptionRecordV50] = {}
        symbols_by_id: dict[str, SymbolRecordV50] = {}
        for branch in branches:
            for assumption in branch.assumptions.assumptions:
                existing = assumptions_by_id.get(assumption.assumption_id)
                if existing is not None and existing != assumption:
                    raise S1RuntimeError("conflicting assumption definitions")
                assumptions_by_id[assumption.assumption_id] = assumption
            for symbol in branch.symbols.symbols:
                existing_symbol = symbols_by_id.get(symbol.symbol_id)
                if existing_symbol is not None and existing_symbol != symbol:
                    raise S1RuntimeError("conflicting symbol definitions")
                symbols_by_id[symbol.symbol_id] = symbol
        selected = refined_candidate
        model = ModelSpecV50.seal(
            selected_candidate_id=selected.candidate_id,
            selected_candidate_structural_hash=selected.structural_hash(),
            selection_rationale=selection.selection_rationale,
            assumption_ids=selected.assumption_ids,
            symbol_ids=selected.symbol_ids,
            data_requirement_ids=selected.data_requirement_ids,
            declared_conservation_laws=selection.declared_conservation_laws,
            declared_limit_cases=selection.declared_limit_cases,
            identifiability_risks=selection.identifiability_risks,
        )
        plan = self._validation_plan(validation_rules)
        execution_ir = compile_registered_family_search_ir_v62(
            selected,
            execution_intent,
        )
        candidate_binding_v67: CandidateExecutionBindingV67 | None = None
        if self.v67_predata_enabled:
            assert self.predata_protocol_v67 is not None
            candidate_binding_v67 = bind_candidate_to_predata_protocol_v67(
                candidate=selected,
                execution_intent=execution_intent,
                execution_ir=execution_ir,
                protocol=self.predata_protocol_v67,
            )
        root = self.workspace.root
        _write_json_new(
            root / "docs" / "candidates.json",
            candidates.model_dump(mode="json"),
        )
        _write_json_new(
            root / "docs" / "assumptions.json",
            AssumptionSetV50(
                assumptions=[
                    assumptions_by_id[key] for key in sorted(assumptions_by_id)
                ]
            ).model_dump(mode="json"),
        )
        _write_json_new(
            root / "docs" / "symbols.json",
            SymbolTableV50(
                symbols=[symbols_by_id[key] for key in sorted(symbols_by_id)]
            ).model_dump(mode="json"),
        )
        _write_json_new(
            root / "docs" / "model_spec.json",
            model.model_dump(mode="json"),
        )
        _write_json_new(
            root / "docs" / "validation_plan.json",
            plan.model_dump(mode="json"),
        )
        _write_json_new(
            root / EXECUTABLE_CANDIDATE_INTENT_PATH,
            execution_intent.model_dump(mode="json"),
        )
        _write_json_new(
            root / EXECUTABLE_CANDIDATE_IR_PATH,
            execution_ir.model_dump(mode="json"),
        )
        if candidate_binding_v67 is not None:
            _write_json_new(
                root / CANDIDATE_EXECUTION_BINDING_PATH_V67,
                candidate_binding_v67.model_dump(mode="json"),
            )
        return plan

    def _commit_review(
        self,
        *,
        producer: RoleProcessOutcomeV51,
        reviewer: RoleProcessOutcomeV51,
        role: Literal["referee", "red_team"],
    ) -> None:
        manifest = self.workspace._manifest_for_stage("S1")
        checks = self.workspace._latest_checks("S1", str(manifest.manifest_hash))
        allowed_inputs = sorted(
            {item.sha256 for item in manifest.files}
            | {
                str(result.result_hash)
                for result in checks.values()
                if result.result_hash is not None
            }
        )
        finding_ids = sorted(
            {
                f"finding-{hashlib.sha256(item.encode('utf-8')).hexdigest()[:16]}"
                for item in reviewer.draft.findings
            }
        )
        trace = self.workspace.commit_evidence(
            "codex_review_transport_trace_v58",
            {
                "stage": "S1",
                "role": role,
                "producer_run_id": producer.request.run_id,
                "reviewer_run_id": reviewer.request.run_id,
                "producer_context_id": producer.request.context_id,
                "reviewer_context_id": reviewer.request.context_id,
                "context_isolation_attested": True,
                "allowed_input_hashes": allowed_inputs,
                "process_receipt": reviewer.receipt.model_dump(mode="json"),
            },
        )
        output = self.workspace.commit_evidence(
            "codex_review_output_v58",
            {
                "stage": "S1",
                "role": role,
                "verdict": reviewer.draft.verdict,
                "finding_ids": finding_ids,
                "draft": reviewer.draft.model_dump(mode="json"),
            },
        )
        self.workspace.issue_review(
            stage="S1",
            review_id=f"review-{reviewer.request.run_id}",
            role=role,
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

    def run(self) -> dict[str, Any]:
        root = self.workspace.root
        if self.v67_predata_enabled:
            self._assert_v67_predata_integrity()
        s1_paths = [
            root / "docs" / name
            for name in (
                "candidates.json",
                "assumptions.json",
                "symbols.json",
                "model_spec.json",
                "validation_plan.json",
                Path(EXECUTABLE_CANDIDATE_INTENT_PATH).name,
                Path(EXECUTABLE_CANDIDATE_IR_PATH).name,
            )
        ]
        v67_manifest_paths = (
            [
                SOURCE_CONTRACT_PATH,
                MEASUREMENT_STUDY_DESIGN_PATH_V67,
                PREDATA_EXECUTION_PROTOCOL_PATH_V67,
                CANDIDATE_EXECUTION_BINDING_PATH_V67,
            ]
            if self.v67_predata_enabled
            else []
        )
        s1_review_paths = [
            *s1_paths,
            *(root / relative_path for relative_path in v67_manifest_paths),
        ]
        if self.workspace.current_gate("S1"):
            graph = EpistemicGraphStoreV58(root).load_head()
            return {
                "gate_decision": "OPEN",
                "epistemic_graph": graph,
                "model_calls": 0,
            }
        generated_v67_paths = (
            [root / CANDIDATE_EXECUTION_BINDING_PATH_V67]
            if self.v67_predata_enabled
            else []
        )
        if any(path.exists() for path in [*s1_paths, *generated_v67_paths]):
            raise S1RuntimeError(
                "S1 contains partial artifacts; refusing automatic re-execution"
            )

        self._event(
            "s1_parallel_exploration_started",
            "running",
            "Independent blind S1 branches and the public-context scout started",
            {
                "branch_ids": self.budget.branch_ids,
                "max_parallel_workers": self.budget.max_parallel_workers,
                "peer_knowledge_visible": False,
            },
        )
        with ThreadPoolExecutor(max_workers=self.budget.max_parallel_workers) as pool:
            branch_futures = {
                branch: pool.submit(self._run_branch, branch)
                for branch in self.budget.branch_ids
            }
            scout_future = pool.submit(self._run_scout)
            branches = [
                branch_futures[branch].result() for branch in self.budget.branch_ids
            ]
            scout_outcome, scout_report = scout_future.result()

        generation_receipts: dict[str, str] = {}
        units: list[KnowledgeUnitV58] = []
        for branch in branches:
            receipt, output = self._commit_role(
                branch.outcome,
                execution_role="modeler",
                subject_id=branch.candidate.candidate_id,
                output_payload={
                    "candidate_id": branch.candidate.candidate_id,
                    "candidate_hash": branch.candidate.structural_hash(),
                    "branch_id": branch.branch_id,
                    "blind_generation": True,
                    "peer_knowledge_visible": False,
                },
            )
            assert receipt.receipt_hash is not None
            generation_receipts[branch.candidate.candidate_id] = (
                receipt.receipt_hash
            )
            for draft in branch.knowledge_drafts:
                units.append(
                    KnowledgeUnitV58.seal(
                        **draft.model_dump(),
                        task_id=self.task_id,
                        branch_id=branch.branch_id,
                        shared_input_hashes=[self.s0_gate],
                        independent_origin_hashes=[receipt.receipt_hash],
                        evidence_refs=sorted([self.s0_gate, output.sha256]),
                        ancestor_unit_hashes=[],
                        status="mechanically_valid",
                        independent_support_count=0,
                        privacy_scope=(
                            "public_only"
                            if self.workspace.spec.evidence_scope == "public_data"
                            else "development_public"
                        ),
                        created_by="model",
                    )
                )
        scout_receipt, _ = self._commit_role(
            scout_outcome,
            execution_role="literature_scout",
            subject_id="literature-map",
            output_payload={
                "subject_id": "literature-map",
                "scope": scout_report.scope,
                "report_hash": sha256_value(scout_report),
                "source_claims_verified": False,
            },
        )
        assert scout_receipt.receipt_hash is not None

        builder = EpistemicGraphBuilderV58(
            task_id=self.task_id,
            workspace_spec_hash=str(self.workspace.spec.spec_hash),
            s0_gate_hash=self.s0_gate,
        )
        builder.add_units(units)
        builder.freeze_initial_frontier()
        graph_store = EpistemicGraphStoreV58(root)
        initial_graph = builder.build()
        graph_store.save(initial_graph)
        if not initial_graph.independence.passed:
            raise S1RuntimeError(
                "blind branch origins are not independently attributable"
            )
        self._event(
            "s1_initial_frontier_frozen",
            "succeeded",
            "Blind candidate frontier and independent origins were frozen",
            {
                "candidate_count": len(branches),
                "knowledge_unit_count": len(units),
                "effective_independent_branches": (
                    initial_graph.independence.effective_independent_branches
                ),
            },
        )

        broker = KnowledgeBrokerV58()
        packets = [
            broker.disclose(
                initial_graph,
                recipient_branch_id=branch.branch_id,
                limit=self.budget.disclosure_limit_per_branch,
            )
            for branch in branches
        ]
        selected_packets = broker.select_for_cross_pollination(
            packets,
            limit=self.budget.max_cross_pollination_branches,
        )
        builder.add_disclosures(packets)
        disclosure_graph = builder.build()
        graph_store.save(disclosure_graph)
        self._event(
            "s1_controlled_disclosure_opened",
            "succeeded",
            "Knowledge Broker opened bounded peer packets after blind freeze",
            {
                "packet_count": len(packets),
                "scheduled_target_branches": [
                    item.recipient_branch_id for item in selected_packets
                ],
            },
        )

        units_by_id = {item.unit_id: item for item in units}
        units_by_hash = {str(item.unit_hash): item for item in units if item.unit_hash}
        with ThreadPoolExecutor(
            max_workers=min(self.budget.max_parallel_workers, len(selected_packets))
        ) as pool:
            translator_futures = {
                packet.recipient_branch_id: pool.submit(
                    self._run_translator,
                    graph_hash=str(disclosure_graph.graph_hash),
                    packet=packet,
                    units_by_hash=units_by_hash,
                )
                for packet in selected_packets
            }
            translator_results = {
                target: translator_futures[target].result()
                for target in sorted(translator_futures)
            }
        transfers: list[TransferHypothesisV58] = []
        packet_by_target = {item.recipient_branch_id: item for item in selected_packets}
        for target, (translator_outcome, draft) in translator_results.items():
            translator_receipt, _ = self._commit_role(
                translator_outcome,
                execution_role="modeler",
                subject_id=f"translator.{target}",
                output_payload={
                    "subject_id": f"translator.{target}",
                    "target_branch_id": target,
                    "translation_report_hash": sha256_value(
                        draft.model_dump(mode="json")
                    ),
                },
            )
            assert translator_receipt.receipt_hash is not None
            try:
                source_units = [units_by_id[item] for item in draft.source_unit_ids]
            except KeyError as exc:
                raise S1RuntimeError(
                    "translator referenced an unknown knowledge unit"
                ) from exc
            source_hashes = sorted(
                str(item.unit_hash) for item in source_units if item.unit_hash
            )
            if not set(source_hashes).issubset(
                set(packet_by_target[target].disclosed_unit_hashes)
            ):
                raise S1RuntimeError(
                    "translator used knowledge outside the target disclosure packet"
                )
            transfers.append(
                TransferHypothesisV58.seal(
                    transfer_id=draft.transfer_id,
                    source_unit_hashes=source_hashes,
                    source_branch_ids=sorted({item.branch_id for item in source_units}),
                    target_branch_id=draft.target_branch_id,
                    target_interpretation=draft.target_interpretation,
                    proposed_modification=draft.proposed_modification,
                    falsification_test=draft.falsification_test,
                    translator_receipt_hash=translator_receipt.receipt_hash,
                )
            )
        builder.add_transfers(transfers)

        branch_by_id = {item.branch_id: item for item in branches}
        with ThreadPoolExecutor(
            max_workers=min(self.budget.max_parallel_workers, len(selected_packets))
        ) as pool:
            mentor_futures = {
                target: pool.submit(
                    self._run_mentor,
                    branch=branch_by_id[target],
                    packet=packet_by_target[target],
                    transfers=transfers,
                    units_by_hash=units_by_hash,
                )
                for target in sorted(packet_by_target)
            }
            mentor_results = {
                target: mentor_futures[target].result()
                for target in sorted(mentor_futures)
            }
        transfers_by_id = {item.transfer_id: item for item in transfers}
        assessments: list[TransferAssessmentV58] = []
        for target, (outcome, drafts) in mentor_results.items():
            receipt, _ = self._commit_role(
                outcome,
                execution_role="modeler",
                subject_id=f"mentor.{target}",
                output_payload={
                    "subject_id": f"mentor.{target}",
                    "target_branch_id": target,
                    "assessment_count": len(drafts),
                    "scientific_support_established": False,
                },
            )
            assert receipt.receipt_hash is not None
            for draft in drafts:
                transfer = transfers_by_id[draft.transfer_id]
                assessments.append(
                    TransferAssessmentV58.seal(
                        transfer_hash=transfer.transfer_hash,
                        target_branch_id=target,
                        verdict=draft.verdict,
                        rationale=draft.rationale,
                        required_test=draft.required_test,
                        assessor_receipt_hash=receipt.receipt_hash,
                    )
                )
        builder.add_transfer_assessments(assessments)
        final_graph = builder.build(promotion_authority_present=False)
        graph_store.save(final_graph)
        graph_artifact = self.workspace.commit_evidence(
            "epistemic_graph_v58",
            final_graph.model_dump(mode="json"),
        )
        self._event(
            "s1_cross_branch_learning_completed",
            "succeeded",
            "Cross-paradigm hypotheses were assessed without becoming science",
            {
                "epistemic_graph_hash": final_graph.graph_hash,
                "committed_artifact_hash": graph_artifact.sha256,
                "transfer_count": len(transfers),
                "assessment_count": len(assessments),
                "cross_task_use_permitted": False,
            },
        )

        epistemic_review_proof = self._epistemic_review_proof(final_graph)
        (
            synthesizer_outcome,
            selection,
            validation_rules,
            selected_structure,
            selected_math,
            execution_intent,
        ) = self._run_synthesizer(
            branches=branches,
            epistemic_summary={
                **graph_store.summary(),
                "accepted_for_later_test": sum(
                    item.verdict == "ACCEPT_FOR_TEST" for item in assessments
                ),
                "rejected_transfers": sum(
                    item.verdict == "REJECT" for item in assessments
                ),
                "epistemic_graph_artifact_hash": graph_artifact.sha256,
            },
        )
        advisory = self._run_formalization_auditor(
            branches=branches,
            selection=selection,
            validation_rules=validation_rules,
            selected_structure=selected_structure,
            selected_math=selected_math,
            execution_intent=execution_intent,
            epistemic_review_proof=epistemic_review_proof,
        )
        self._commit_advisory_review(advisory, attempt=1)
        if advisory.draft.verdict != "APPROVE":
            if self.repair_context_v67 is not None:
                self._event(
                    "s1_graph_repair_review_exhausted_v67",
                    "blocked",
                    "The one graph-authorized S1 repair remained rejected",
                    {
                        "verdict": advisory.draft.verdict,
                        "finding_count": len(advisory.draft.findings),
                        "model_calls_used": self.call_budget.used,
                        "graph_attempt": self.workspace._latest_attempt("S1"),
                        "repair_context_hash": (
                            self.repair_context_v67.context_hash
                        ),
                    },
                )
                assert self.predata_protocol_v67 is not None
                assert self.predata_protocol_v67.protocol_hash is not None
                assert advisory.receipt.receipt_hash is not None
                handoff_artifact_hash = (
                    self._commit_s1_rejection_handoff_v67(
                        findings=advisory.draft.findings,
                        reviewer_receipt_hash=(
                            advisory.receipt.receipt_hash
                        ),
                    )
                )
                raise S1FormalizationRejectedV67(
                    findings=advisory.draft.findings,
                    reviewer_receipt_hash=advisory.receipt.receipt_hash,
                    protocol_hash=self.predata_protocol_v67.protocol_hash,
                    handoff_artifact_hash=handoff_artifact_hash,
                )
            self._event(
                "s1_preflight_review_requested_repair",
                "blocked",
                "Pre-freeze auditor requested one bounded formalization repair",
                {
                    "verdict": advisory.draft.verdict,
                    "finding_count": len(advisory.draft.findings),
                    "model_calls_used": self.call_budget.used,
                },
            )
            self._commit_role(
                synthesizer_outcome,
                execution_role="modeler",
                subject_id="s1-selection-attempt-1",
                output_payload={
                    "subject_id": "s1-selection-attempt-1",
                    "selected_candidate_id": selection.selected_candidate_id,
                    "selection_hash": sha256_value(selection),
                    "validation_rules_hash": sha256_value(validation_rules),
                    "selected_structure_hash": sha256_value(selected_structure),
                    "selected_mathematical_form_hash": sha256_value(
                        selected_math
                    ),
                    "executable_candidate_intent_hash": (
                        execution_intent.content_hash()
                    ),
                    "repair_requested": True,
                    "scientific_acceptance": False,
                },
            )
            (
                synthesizer_outcome,
                selection,
                validation_rules,
                selected_structure,
                selected_math,
                execution_intent,
            ) = self._run_synthesizer(
                branches=branches,
                epistemic_summary={
                    **graph_store.summary(),
                    "epistemic_graph_artifact_hash": graph_artifact.sha256,
                },
                repair={
                    "previous_selection": selection.model_dump(mode="json"),
                    "previous_validation_rules": validation_rules.model_dump(
                        mode="json"
                    ),
                    "previous_selected_candidate_structure": (
                        selected_structure.model_dump(mode="json")
                    ),
                    "previous_selected_mathematical_form": (
                        selected_math.model_dump(mode="json")
                    ),
                    "previous_executable_candidate_intent": (
                        execution_intent.model_dump(mode="json")
                    ),
                    "auditor_verdict": advisory.draft.verdict,
                    "auditor_findings": advisory.draft.findings,
                    "auditor_uncertainties": advisory.draft.uncertainties,
                    "instruction": (
                        "Return complete replacement artifacts that resolve "
                        "every auditor finding without inventing evidence."
                    ),
                },
            )
            advisory = self._run_formalization_auditor(
                branches=branches,
                selection=selection,
                validation_rules=validation_rules,
                selected_structure=selected_structure,
                selected_math=selected_math,
                execution_intent=execution_intent,
                epistemic_review_proof=epistemic_review_proof,
            )
            self._commit_advisory_review(advisory, attempt=2)
            if advisory.draft.verdict != "APPROVE":
                self._event(
                    "s1_preflight_review_exhausted",
                    "blocked",
                    "Formalization remained incomplete after one bounded repair",
                    {
                        "verdict": advisory.draft.verdict,
                        "finding_count": len(advisory.draft.findings),
                        "model_calls_used": self.call_budget.used,
                    },
                )
                if self.v67_predata_enabled:
                    assert self.predata_protocol_v67 is not None
                    assert self.predata_protocol_v67.protocol_hash is not None
                    assert advisory.receipt.receipt_hash is not None
                    handoff_artifact_hash = (
                        self._commit_s1_rejection_handoff_v67(
                            findings=advisory.draft.findings,
                            reviewer_receipt_hash=(
                                advisory.receipt.receipt_hash
                            ),
                        )
                    )
                    raise S1FormalizationRejectedV67(
                        findings=advisory.draft.findings,
                        reviewer_receipt_hash=advisory.receipt.receipt_hash,
                        protocol_hash=self.predata_protocol_v67.protocol_hash,
                        handoff_artifact_hash=handoff_artifact_hash,
                    )
                raise S1RuntimeError(
                    "S1 formalization failed the second pre-freeze audit"
                )
            self._event(
                "s1_preflight_review_recovered",
                "succeeded",
                "Formalization passed a fresh audit after one bounded repair",
                {
                    "model_calls_used": self.call_budget.used,
                    "selected_candidate_id": selection.selected_candidate_id,
                },
            )
        else:
            self._event(
                "s1_preflight_review_passed",
                "succeeded",
                "Formalization passed the independent pre-freeze audit",
                {
                    "model_calls_used": self.call_budget.used,
                    "selected_candidate_id": selection.selected_candidate_id,
                },
            )
        refined_candidate = self._refined_candidate(
            branches,
            selected_structure,
            selected_math,
        )
        synthesizer_receipt, _ = self._commit_role(
            synthesizer_outcome,
            execution_role="modeler",
            subject_id=selection.selected_candidate_id,
            output_payload={
                "subject_id": selection.selected_candidate_id,
                "candidate_id": selection.selected_candidate_id,
                "candidate_hash": refined_candidate.structural_hash(),
                "selected_candidate_id": selection.selected_candidate_id,
                "selection_hash": sha256_value(selection),
                "validation_rules_hash": sha256_value(validation_rules),
                "selected_structure_hash": sha256_value(selected_structure),
                "selected_mathematical_form_hash": sha256_value(selected_math),
                "executable_candidate_intent_hash": (
                    execution_intent.content_hash()
                ),
                "scientific_acceptance": False,
            },
        )
        assert synthesizer_receipt.receipt_hash is not None
        final_generation_receipts = [
            synthesizer_receipt.receipt_hash
            if candidate_id == selection.selected_candidate_id
            else generation_receipts[candidate_id]
            for candidate_id in sorted(generation_receipts)
        ]
        self._materialize_s1(
            branches=branches,
            generation_receipt_hashes=final_generation_receipts,
            scout_receipt_hash=scout_receipt.receipt_hash,
            selection=selection,
            validation_rules=validation_rules,
            selected_structure=selected_structure,
            selected_math=selected_math,
            execution_intent=execution_intent,
        )
        self.workspace.submit_stage(
            "S1",
            actor="model",
            extra_paths=[
                EXECUTABLE_CANDIDATE_INTENT_PATH,
                EXECUTABLE_CANDIDATE_IR_PATH,
                *v67_manifest_paths,
            ],
        )
        check = self.workspace.run_mechanical_check("S1")
        self._event(
            "s1_formalization_completed",
            "succeeded" if check.status == "PASS" else "blocked",
            "S1 candidate set, selected model, and validation plan were frozen",
            {
                "selected_candidate_id": selection.selected_candidate_id,
                "mechanical_check_status": check.status,
                "model_calls_used_before_review": self.call_budget.used,
                "scientific_acceptance": False,
            },
        )

        manifest = self.workspace._manifest_for_stage("S1")
        review_artifacts = {
            path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in s1_review_paths
        }
        candidates_for_review = review_artifacts["candidates"]["candidates"]
        review_evidence_packet = {
            "artifact_hashes": {
                item.relative_path: item.sha256 for item in manifest.files
            },
            "candidates": [
                {
                    key: candidate[key]
                    for key in (
                        "candidate_id",
                        "model_family",
                        "mathematical_form",
                        "assumption_ids",
                        "symbol_ids",
                        "validation_obligation_ids",
                        "abandon_criteria",
                    )
                }
                for candidate in candidates_for_review
            ],
            "assumptions": [
                {
                    key: assumption[key]
                    for key in (
                        "assumption_id",
                        "statement",
                        "falsification_test",
                        "abandon_criterion",
                    )
                }
                for assumption in review_artifacts["assumptions"]["assumptions"]
            ],
            "symbols": [
                {
                    key: symbol[key]
                    for key in ("symbol_id", "meaning", "role", "unit")
                }
                for symbol in review_artifacts["symbols"]["symbols"]
            ],
            "selected_model": review_artifacts["model_spec"],
            "validation_plan": review_artifacts["validation_plan"],
            "executable_candidate_intent": review_artifacts[
                Path(EXECUTABLE_CANDIDATE_INTENT_PATH).stem
            ],
            "executable_candidate_ir": review_artifacts[
                Path(EXECUTABLE_CANDIDATE_IR_PATH).stem
            ],
        }
        if self.v67_predata_enabled:
            assert self._v67_protocol_verification is not None
            review_evidence_packet.update(
                {
                    "source_contract_v62": review_artifacts[
                        Path(SOURCE_CONTRACT_PATH).stem
                    ],
                    "measurement_study_design_contract_v67": review_artifacts[
                        Path(MEASUREMENT_STUDY_DESIGN_PATH_V67).stem
                    ],
                    "predata_execution_protocol_v67": review_artifacts[
                        Path(PREDATA_EXECUTION_PROTOCOL_PATH_V67).stem
                    ],
                    "candidate_execution_binding_v67": review_artifacts[
                        Path(CANDIDATE_EXECUTION_BINDING_PATH_V67).stem
                    ],
                    "predata_protocol_mechanical_verification_v67": (
                        self._v67_protocol_verification
                    ),
                }
            )
            final_review_rule = (
                "APPROVE only if candidates are structurally distinct; the "
                "origin-separation audit passes without cross-branch overlap; "
                "disclosures exclude the recipient; transfers remain proposed "
                "and assessments establish no scientific support; the selected "
                "model is registry-bound; its typed execution intent and "
                "candidate-specific validation applicability rules agree with "
                "and do not loosen or contradict the exact sealed V6.7 "
                "pre-data protocol; and the deterministic protocol replay is "
                "verified. The harness-owned candidate execution binding, not "
                "the legacy V6.2 two-adapter envelope, is the single-adapter "
                "execution authority. The code-owned protocol, not model prose, owns "
                "optimizer, split, baseline, dependence, uncertainty, stress, "
                "threshold, and L0-L4 execution details. Origin separation is "
                "the S1 workflow condition, not a claim of independent "
                "scientific replication. Do not require empirical results at S1."
            )
        else:
            final_review_rule = (
                "APPROVE only if candidates are structurally distinct; the "
                "origin-separation audit passes without cross-branch overlap; "
                "disclosures exclude the recipient; transfers remain proposed "
                "and assessments establish no scientific support; the selected "
                "model is registry-bound; its typed execution intent agrees "
                "with the candidate while free text remains non-executable; "
                "and every L0-L4 rule is task-specific and decidable. Origin "
                "separation is the S1 workflow condition, "
                "not a claim of independent scientific replication. Do not "
                "require empirical results at S1."
            )
        review_inputs = {
            "producer_output_hash": synthesizer_outcome.receipt.output_hash,
            "manifest": manifest.model_dump(mode="json"),
            "review_evidence_packet": review_evidence_packet,
            "mechanical_check": check.model_dump(mode="json"),
            "epistemic_summary": graph_store.summary(),
            "epistemic_review_proof": epistemic_review_proof,
            "gate_policy_hash": POLICIES["S1"].policy_hash,
            "review_rule": final_review_rule,
        }

        def review(role: Literal["referee", "red_team"]):
            for attempt in (1, 2):
                try:
                    outcome = self._run_role(
                        role_name=f"s1_{role}",
                        role_kind="reviewer",
                        subject_id="s1-selection",
                        objective=(
                            "Check the bounded S1 evidence packet against the "
                            "declared gate rule and return a concise verdict."
                            if role == "referee"
                            else "Identify up to eight decisive S1 structural "
                            "failure modes, then return a concise verdict."
                        ),
                        public_inputs={
                            **review_inputs,
                            "review_role": role,
                            "response_budget": {
                                "maximum_findings": 8,
                                "maximum_rationale_characters": 800,
                                "instruction": (
                                    "Judge only supplied evidence. Return HUMAN "
                                    "when evidence is insufficient; do not derive "
                                    "a replacement model."
                                ),
                            },
                        },
                        allowed_candidate_ids=[
                            item.candidate.candidate_id for item in branches
                        ],
                    )
                except S1RuntimeError:
                    raise
                except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    self._event(
                        "s1_review_transport_failed",
                        "blocked",
                        f"{role} process failed before a valid review receipt",
                        {
                            "review_role": role,
                            "attempt": attempt,
                            "error_type": type(exc).__name__,
                            "failure_signature": (
                                f"transport {type(exc).__name__}: {str(exc)[:400]}"
                            ),
                        },
                    )
                    if attempt == 2:
                        raise S1RuntimeError(
                            f"{role} exhausted its review transport retry budget"
                        ) from exc
                    continue
                if attempt == 2:
                    self._event(
                        "s1_review_transport_recovered",
                        "succeeded",
                        f"{role} produced a valid receipt in a fresh retry process",
                        {
                            "review_role": role,
                            "attempt": attempt,
                            "run_id": outcome.request.run_id,
                            "output_hash": outcome.receipt.output_hash,
                        },
                    )
                return outcome
            raise AssertionError("unreachable review retry state")

        with ThreadPoolExecutor(max_workers=2) as pool:
            review_futures = {
                role: pool.submit(review, role) for role in ("referee", "red_team")
            }
            reviewers = {
                role: review_futures[role].result() for role in ("referee", "red_team")
            }
        for role in ("referee", "red_team"):
            self._commit_review(
                producer=synthesizer_outcome,
                reviewer=reviewers[role],
                role=role,
            )
        gate = self.workspace.evaluate_gate("S1")
        self._event(
            "s1_gate_evaluated",
            "succeeded" if gate.decision == "OPEN" else "blocked",
            (
                "S1 gate opened; S2 data freezing is now available"
                if gate.decision == "OPEN"
                else f"S1 gate did not open: {gate.decision}"
            ),
            {
                "decision": gate.decision,
                "reasons": gate.reasons,
                "review_verdicts": {
                    role: reviewers[role].draft.verdict
                    for role in ("referee", "red_team")
                },
                "model_calls_used": self.call_budget.used,
                "scientific_qualification_granted": False,
                "real_world_action_authorized": False,
            },
        )
        return {
            "gate_decision": gate.decision,
            "selected_candidate_id": selection.selected_candidate_id,
            "epistemic_graph": final_graph,
            "model_calls": self.call_budget.used,
            "synthesizer_receipt_hash": synthesizer_receipt.receipt_hash,
        }


__all__ = [
    "CandidateCoreDraftV58",
    "CandidateMathematicalFormDraftV58",
    "CandidateStructureDraftV58",
    "LiteratureMapDraftV58",
    "REQUIRED_VALIDATION_IDS",
    "S1FormalizationRejectedV67",
    "S1RuntimeError",
    "S1SelectionDraftV58",
    "StudioS1OrchestratorV58",
    "TransferCoreDraftV58",
    "VALIDATION_RULE_ARTIFACTS",
    "ValidationRuleDraftV58",
    "ValidationRulesDraftV58",
]
