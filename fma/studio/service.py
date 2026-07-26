"""Typed local service that connects the web studio to the FMA V5 kernel.

The browser never receives the V5 authority key and cannot write graph state
directly.  It may request a task or bounded S0/S1 runs; this service validates
the request, invokes isolated Codex role processes, and asks the existing
harness to authenticate checks, reviews, and graph transitions.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import Field, ValidationError, model_validator

from fma.codex_driver import CodexCLIConfig
from fma.hashing import canonical_json, sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier
from fma.v5.scaffold import scaffold_task_workspace
from fma.v5.stage_workspace import (
    POLICIES,
    StageWorkspaceV50,
    _evaluate_arithmetic,
)
from fma.v5.workspace_schemas import (
    DecisionFunctionCanaryV50,
    DecisionFunctionSpecV50,
    RegimeDiagnosisV50,
    TaskWorkspaceSpecV50,
    WorkflowProfileV50,
)
from fma.v5_1.codex_stage_driver import (
    RoleProcessOutcomeV51,
    StageRoleDriverV51,
    StageRoleTransportV51,
    commit_generator_outcome_v51,
)
from fma.v5_8.epistemic import EpistemicGraphStoreV58
from fma.v5_8.stage_driver import CodexStageRoleTransportV58

from .s1_runtime import S1RuntimeError, StudioS1OrchestratorV58


_TASK_ID_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,58}[A-Za-z0-9])?")
_S0_PATHS = (
    "problem/contract.json",
    "problem/decision_function.json",
    "docs/regime.json",
)
_S1_PATHS = (
    "docs/candidates.json",
    "docs/assumptions.json",
    "docs/symbols.json",
    "docs/model_spec.json",
    "docs/validation_plan.json",
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


class CreateTaskRequest(StrictModel):
    objective: str = Field(min_length=12, max_length=4000)
    workspace_id: str | None = Field(default=None, max_length=60)
    evidence_scope: Literal["development", "public_data"] = "development"


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
            len(item.input_values) != len(self.input_names)
            for item in self.canaries
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


class RoleTransportFactory(Protocol):
    def __call__(self, output_root: Path) -> StageRoleTransportV51: ...


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _write_json_new(path: Path, payload: object) -> None:
    if path.exists():
        raise StudioConflictError(
            f"refusing to overwrite existing artifact: {path.name}"
        )
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


def _sealed_without_hash(model: StrictModel, hash_field: str, model_type):
    payload = model.model_dump(mode="json", exclude={hash_field})
    return model_type.seal(**payload)


class StudioTaskService:
    """Single-process bridge service with fail-closed task-level concurrency."""

    def __init__(
        self,
        task_root: str | Path,
        *,
        authority_key: bytes,
        authority_key_id: str,
        codex_config: CodexCLIConfig | None = None,
        role_transport_factory: RoleTransportFactory | None = None,
    ) -> None:
        if len(authority_key) < 32:
            raise ValueError("authority_key must contain at least 32 bytes")
        self.task_root = Path(task_root).resolve()
        self.task_root.mkdir(parents=True, exist_ok=True)
        self.authority_key = bytes(authority_key)
        self.authority_key_id = authority_key_id
        self.codex_config = codex_config or CodexCLIConfig()
        self.role_transport_factory = role_transport_factory
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

    def _event_path(self, task_id: str) -> Path:
        return self._task_path(task_id) / ".fma" / "studio_events.jsonl"

    def _events(self, task_id: str) -> list[StudioEvent]:
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

    def _append_event(
        self,
        task_id: str,
        *,
        event_type: str,
        status: Literal["accepted", "running", "succeeded", "failed", "blocked"],
        message: str,
        details: dict[str, Any] | None = None,
    ) -> StudioEvent:
        with self._lock:
            events = self._events(task_id)
            payload = {
                "schema_version": "1.0",
                "sequence": len(events) + 1,
                "event_type": event_type,
                "status": status,
                "message": message,
                "details": details or {},
                "recorded_at": _utc_now(),
                "previous_event_hash": events[-1].event_hash if events else None,
            }
            unsigned = StudioEvent(**payload, event_hash="0" * 64)
            event = StudioEvent(
                **payload,
                event_hash=sha256_value(
                    unsigned.model_dump(mode="json", exclude={"event_hash"})
                ),
            )
            path = self._event_path(task_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical_json(event.model_dump(mode="json")) + "\n")
            return event

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
        derived = hashlib.sha256(objective.encode("utf-8")).hexdigest()[:12]
        task_id = validated.workspace_id or f"task-{derived}"
        if not _TASK_ID_PATTERN.fullmatch(task_id) or task_id in {".", ".."}:
            raise StudioValidationError("workspace_id is not a safe identifier")
        root = self._task_path(task_id)

        with self._lock:
            if root.exists():
                workspace = self._workspace(task_id)
                if workspace.spec.objective != objective:
                    raise StudioConflictError(
                        "workspace_id already exists with another objective"
                    )
                return self.snapshot(task_id)

            mission_hash = sha256_value(
                {
                    "schema_version": "studio-mission-1",
                    "objective": objective,
                    "value_owner": "user",
                }
            )
            evidence_snapshot_hash = sha256_value(
                {
                    "schema_version": "studio-evidence-1",
                    "objective_hash": hashlib.sha256(
                        objective.encode("utf-8")
                    ).hexdigest(),
                    "items": [],
                    "data_ingested": False,
                }
            )
            scaffold_task_workspace(root, task_id, objective)
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
            StageWorkspaceV50.create(
                root,
                spec,
                authority_key=self.authority_key,
                authority_key_id=self.authority_key_id,
            )
            self._append_event(
                task_id,
                event_type="task_created",
                status="succeeded",
                message="FMA task workspace and S0 frontier created",
                details={
                    "evidence_scope": validated.evidence_scope,
                    "scientific_qualification_granted": False,
                    "real_world_action_authorized": False,
                },
            )
            return self.snapshot(task_id)

    def snapshot(self, task_id: str) -> dict[str, Any]:
        workspace = self._workspace(task_id)
        status = workspace.status()
        events = self._events(task_id)
        s0_open = workspace.current_gate("S0") is not None
        s1_open = workspace.current_gate("S1") is not None
        with self._lock:
            active = task_id in self._active_tasks
        return {
            "status": "success",
            "task_id": task_id,
            "objective": workspace.spec.objective,
            "workflow": status.model_dump(mode="json"),
            "activity": "running"
            if active
            else (events[-1].status if events else "idle"),
            "events": [event.model_dump(mode="json") for event in events[-30:]],
            "epistemic": EpistemicGraphStoreV58(workspace.root).summary(),
            "next_valid_actions": (
                []
                if active
                else (
                    ["inspect_s1", "continue_s2"]
                    if s1_open
                    else ["inspect_s0", "run_s1"]
                    if s0_open
                    else ["run_s0"]
                )
            ),
            "scientific_qualification_granted": False,
            "real_world_action_authorized": False,
        }

    def list_tasks(self) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        for child in sorted(self.task_root.iterdir()):
            if (
                not child.is_dir()
                or not (child / ".fma" / "workspace_spec.json").is_file()
            ):
                continue
            try:
                snapshot = self.snapshot(child.name)
            except (OSError, ValueError, RuntimeError):
                continue
            items.append(
                {
                    "task_id": snapshot["task_id"],
                    "objective": snapshot["objective"],
                    "activity": snapshot["activity"],
                    "stage_statuses": snapshot["workflow"]["stage_statuses"],
                }
            )
        return {"status": "success", "items": items}

    def _transport(self, task_id: str) -> StageRoleTransportV51:
        output_root = self._task_path(task_id) / ".fma" / "roles"
        if self.role_transport_factory is not None:
            return self.role_transport_factory(output_root)
        return CodexStageRoleTransportV58(
            output_root,
            self.codex_config,
        )

    def _materialize_s0(
        self,
        workspace: StageWorkspaceV50,
        outcome: RoleProcessOutcomeV51,
    ) -> None:
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
        if decision_payload.get("schema_version") == "5.8":
            decision_draft = DecisionFunctionDraftV58.model_validate(
                decision_payload
            )
            decision_unsealed = DecisionFunctionSpecV50(
                function_id=decision_draft.function_id,
                input_names=decision_draft.input_names,
                expression=decision_draft.expression,
                sense=decision_draft.sense,
                output_unit=decision_draft.output_unit,
                canaries=[
                    DecisionFunctionCanaryV50(
                        canary_id=item.canary_id,
                        inputs=dict(
                            zip(
                                decision_draft.input_names,
                                item.input_values,
                                strict=True,
                            )
                        ),
                        expected=item.expected,
                        tolerance=item.tolerance,
                    )
                    for item in decision_draft.canaries
                ],
            )
        else:
            decision_unsealed = DecisionFunctionSpecV50.model_validate(
                decision_payload
            )
        regime_unsealed = RegimeDiagnosisV50.model_validate(
            _safe_json(
                artifacts["regime_diagnosis"],
                artifact_type="regime_diagnosis",
            )
        )
        decision = _sealed_without_hash(
            decision_unsealed, "function_hash", DecisionFunctionSpecV50
        )
        regime = _sealed_without_hash(
            regime_unsealed, "diagnosis_hash", RegimeDiagnosisV50
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
        if regime.decision_function_id != decision.function_id:
            raise StudioValidationError(
                "regime decision_function_id does not match decision function"
            )
        if workspace.spec.evidence_snapshot_hash not in regime.evidence_hashes:
            raise StudioValidationError(
                "regime diagnosis is not bound to the frozen evidence snapshot"
            )
        root = workspace.root
        if any((root / relative).exists() for relative in _S0_PATHS):
            raise StudioConflictError(
                "S0 artifacts already exist; automatic re-execution is blocked"
            )
        _write_json_new(
            root / "problem" / "contract.json",
            {
                "schema_version": "5.0",
                "mission_hash": workspace.spec.mission_hash,
                "evidence_snapshot_hash": workspace.spec.evidence_snapshot_hash,
                "question": workspace.spec.objective,
            },
        )
        _write_json_new(
            root / "problem" / "decision_function.json",
            decision.model_dump(mode="json"),
        )
        _write_json_new(
            root / "docs" / "regime.json",
            regime.model_dump(mode="json"),
        )

    def _commit_review(
        self,
        workspace: StageWorkspaceV50,
        *,
        producer: RoleProcessOutcomeV51,
        reviewer: RoleProcessOutcomeV51,
    ) -> None:
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
        finding_ids = sorted(
            {
                f"finding-{hashlib.sha256(item.encode('utf-8')).hexdigest()[:16]}"
                for item in reviewer.draft.findings
            }
        )
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
        workspace.issue_review(
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

    def run_s0(self, task_id: str) -> dict[str, Any]:
        workspace = self._workspace(task_id)
        if workspace.current_gate("S0"):
            return self.snapshot(task_id)
        if any((workspace.root / relative).exists() for relative in _S0_PATHS):
            raise StudioConflictError(
                "S0 contains partial artifacts; refusing a second model call"
            )
        driver = StageRoleDriverV51(self._transport(task_id))
        self._append_event(
            task_id,
            event_type="s0_generator_started",
            status="running",
            message="Fresh Codex generator process started for S0",
        )
        generator_inputs: dict[str, Any] = {
            "user_objective": workspace.spec.objective,
            "mission_hash": workspace.spec.mission_hash,
            "evidence_snapshot_hash": workspace.spec.evidence_snapshot_hash,
            "evidence_scope": workspace.spec.evidence_scope,
            "required_artifacts": {
                "decision_function": DecisionFunctionDraftV58.model_json_schema(),
                "regime_diagnosis": RegimeDiagnosisV50.model_json_schema(),
            },
            "requirements": [
                "Return exactly two proposed_artifacts.",
                "Use artifact_type decision_function and regime_diagnosis.",
                "Each content field must contain only a JSON object string.",
                "Bind regime evidence_hashes to evidence_snapshot_hash.",
                "Keep every identifier list sorted and unique.",
                "Use report_only when the user's decision loss is not specified.",
                "State limitations and a concrete abandon condition.",
                "decision_function.expression must be only a bare arithmetic "
                "expression over input_names; put constraints, abstention rules, "
                "and prose in regime_diagnosis.",
                "For each decision canary, input_values must align positionally "
                "with input_names; the harness binds the names.",
            ],
        }
        producer: RoleProcessOutcomeV51 | None = None
        validation_error: str | None = None
        for attempt in (1, 2):
            attempt_inputs = dict(generator_inputs)
            if validation_error is not None and producer is not None:
                attempt_inputs["repair"] = {
                    "previous_output_hash": producer.receipt.output_hash,
                    "validation_error": validation_error[:500],
                    "instruction": (
                        "Return a complete corrected replacement. Do not weaken "
                        "the contract or omit required evidence bindings."
                    ),
                }
            producer = driver.run(
                task_id=task_id,
                stage="S0",
                role_name=(
                    "problem_formulator"
                    if attempt == 1
                    else "problem_formulator_repair"
                ),
                role_kind="generator",
                subject_id="s0_problem_contract",
                objective=(
                    "Formalize the user's real modeling objective into a falsifiable "
                    "S0 regime diagnosis and a computable decision function."
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
                self._materialize_s0(workspace, producer)
                break
            except (StudioValidationError, ValueError) as exc:
                validation_error = str(exc)
                self._append_event(
                    task_id,
                    event_type="s0_generator_rejected",
                    status="blocked",
                    message=(
                        "Generator output failed typed validation"
                        if attempt == 1
                        else "Repair output failed typed validation"
                    ),
                    details={
                        "attempt": attempt,
                        "failure_signature": validation_error[:500],
                        "output_hash": producer.receipt.output_hash,
                    },
                )
                if attempt == 2:
                    raise StudioValidationError(
                        "S0 generator exhausted its two-attempt validation budget"
                    ) from exc
        assert producer is not None
        workspace.submit_stage("S0", actor="model")
        check = workspace.run_mechanical_check("S0")
        self._append_event(
            task_id,
            event_type="s0_generator_completed",
            status="succeeded",
            message="S0 artifacts validated and committed; independent review required",
            details={
                "run_id": producer.request.run_id,
                "check_status": check.status,
                "generator_attempts": 2 if validation_error else 1,
            },
        )

        manifest = workspace._manifest_for_stage("S0")
        self._append_event(
            task_id,
            event_type="s0_reviewer_started",
            status="running",
            message="Fresh independent Codex referee process started",
        )
        reviewer = driver.run(
            task_id=task_id,
            stage="S0",
            role_name="s0_referee",
            role_kind="reviewer",
            subject_id="s0_problem_contract",
            objective=(
                "Independently review whether the S0 contract is coherent, "
                "falsifiable, evidence-bound, and honest about missing decisions."
            ),
            public_inputs={
                "producer_output_hash": producer.receipt.output_hash,
                "manifest": manifest.model_dump(mode="json"),
                "artifacts": {
                    relative: json.loads(
                        (workspace.root / relative).read_text(encoding="utf-8")
                    )
                    for relative in _S0_PATHS
                },
                "mechanical_check": check.model_dump(mode="json"),
                "gate_policy_hash": POLICIES["S0"].policy_hash,
                "review_rule": (
                    "APPROVE only if the question remains the frozen objective, "
                    "the loss is computable, uncertainty is explicit, and no "
                    "scientific or action authority is claimed. Otherwise REJECT "
                    "or HUMAN."
                ),
            },
            allowed_candidate_ids=[],
        )
        self._commit_review(workspace, producer=producer, reviewer=reviewer)
        gate = workspace.evaluate_gate("S0")
        final_status = "succeeded" if gate.decision == "OPEN" else "blocked"
        self._append_event(
            task_id,
            event_type="s0_gate_evaluated",
            status=final_status,
            message=(
                "S0 gate opened; S1 candidate formalization is now available"
                if gate.decision == "OPEN"
                else f"S0 gate did not open: {gate.decision}"
            ),
            details={
                "decision": gate.decision,
                "reasons": gate.reasons,
                "review_verdict": reviewer.draft.verdict,
                "scientific_qualification_granted": False,
                "real_world_action_authorized": False,
            },
        )
        return self.snapshot(task_id)

    def start_s0(self, task_id: str) -> dict[str, Any]:
        self._workspace(task_id)
        with self._lock:
            if task_id in self._active_tasks:
                raise StudioConflictError("task already has an active S0 run")
            self._active_tasks.add(task_id)
        self._append_event(
            task_id,
            event_type="s0_run_accepted",
            status="accepted",
            message="Bounded S0 run accepted by the local bridge",
        )

        def worker() -> None:
            try:
                self.run_s0(task_id)
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
        thread.start()
        return self.snapshot(task_id)

    def run_s1(self, task_id: str) -> dict[str, Any]:
        workspace = self._workspace(task_id)
        if workspace.current_gate("S1"):
            return self.snapshot(task_id)
        if not workspace.current_gate("S0"):
            raise StudioConflictError("S1 requires an open current S0 gate")
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
        )
        try:
            orchestrator.run()
        except (S1RuntimeError, ValidationError) as exc:
            raise StudioValidationError(str(exc)) from exc
        return self.snapshot(task_id)

    def start_s1(self, task_id: str) -> dict[str, Any]:
        workspace = self._workspace(task_id)
        if not workspace.current_gate("S0"):
            raise StudioConflictError("S1 requires an open current S0 gate")
        with self._lock:
            if task_id in self._active_tasks:
                raise StudioConflictError("task already has an active stage run")
            self._active_tasks.add(task_id)
        self._append_event(
            task_id,
            event_type="s1_run_accepted",
            status="accepted",
            message="Bounded graph-native S1 run accepted by the local bridge",
        )

        def worker() -> None:
            try:
                self.run_s1(task_id)
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
        thread.start()
        return self.snapshot(task_id)
