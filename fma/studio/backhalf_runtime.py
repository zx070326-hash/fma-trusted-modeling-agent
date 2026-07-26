"""Narrow, auditable S2--S6 Studio runtime for positive scalar ODE series.

This is an additive vertical slice.  It does not pretend to execute arbitrary
S1 mathematics: the selected S1 candidate must explicitly describe an
autonomous ODE family, and the supplied data must satisfy the registered V5.2
adapter contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import uuid4

import numpy as np
import scipy
from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier
from fma.v5.check_registry import (
    AdapterContextV50,
    AdapterOutcomeV50,
    CheckRegistryV50,
)
from fma.v5.paper import build_paper
from fma.v5.stage_workspace import POLICIES, StageWorkspaceV50, _tree_hash
from fma.v5.workspace_schemas import (
    CodeManifestV50,
    DataLedgerEntryV50,
    DataLedgerV50,
    DecisionAssertionV50,
    DecisionDossierV50,
    ModelSpecV50,
    ProcessedArtifactV50,
    ProcessedManifestV50,
    ResultIndexV50,
    ResultRecordV50,
    StageId,
    UQClaimV50,
    UQSummaryV50,
    ValidationObligationV50,
    ValidationPlanV50,
)
from fma.v5_1.codex_stage_driver import (
    RoleProcessOutcomeV51,
    StageRoleDriverV51,
    commit_generator_outcome_v51,
)
from fma.v5_2.ode_system import (
    ODEScientificBundleV52,
    ODEThresholdsV52,
    ODETimeSeriesSnapshotV52,
    build_ode_bundle_v52,
    run_ode_replays_v52,
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

ODE_ADAPTER_ID = "scalar_autonomous_ode_v52"
RAW_RELATIVE_PATH = "data/raw/ode_series.json"
PROCESSED_RELATIVE_PATH = "data/processed/ode_snapshot.json"
ADAPTER_BINDING_PATH = "docs/adapter_binding.json"
BUNDLE_PATH = "results/ode_scientific_bundle.json"
REPLAY_INPUT_PATH = "checks/ode_replay_input.json"


class BackhalfRuntimeError(RuntimeError):
    pass


class StudioODEDataRequestV59(StrictModel):
    schema_version: Literal["5.9"] = "5.9"
    adapter_id: Literal["scalar_autonomous_ode_v52"] = ODE_ADAPTER_ID
    time_unit: Identifier
    state_unit: Identifier
    times: list[float] = Field(min_length=12, max_length=4096)
    observations: list[float] = Field(min_length=12, max_length=4096)
    source_id: str = Field(min_length=3, max_length=300)
    license_status: str = Field(min_length=2, max_length=300)
    fixture_only: bool = False

    @model_validator(mode="after")
    def validate_series(self) -> "StudioODEDataRequestV59":
        if len(self.times) != len(self.observations):
            raise ValueError("times and observations must have equal length")
        if any(not math.isfinite(value) for value in self.times):
            raise ValueError("times must be finite")
        if any(
            right <= left for left, right in zip(self.times, self.times[1:])
        ):
            raise ValueError("times must be strictly increasing")
        if any(
            not math.isfinite(value) or value <= 0
            for value in self.observations
        ):
            raise ValueError("ODE observations must be finite and positive")
        return self


class DataMappingDraftV59(StrictModel):
    schema_version: Literal["5.9"] = "5.9"
    data_requirement_ids: list[Identifier] = Field(min_length=1, max_length=8)
    semantic_name: str = Field(min_length=5, max_length=300)
    units: str = Field(min_length=1, max_length=100)
    transform_rule: str = Field(min_length=10, max_length=600)
    quality_flags: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_ids(self) -> "DataMappingDraftV59":
        if self.data_requirement_ids != sorted(set(self.data_requirement_ids)):
            raise ValueError("data requirement IDs must be sorted and unique")
        return self


class DecisionNarrativeDraftV59(StrictModel):
    schema_version: Literal["5.9"] = "5.9"
    statement: str = Field(min_length=20, max_length=1200)
    limitations: list[str] = Field(min_length=1, max_length=8)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_new(path: Path, payload: object) -> None:
    if path.exists():
        raise BackhalfRuntimeError(
            f"refusing to overwrite existing artifact: {path.as_posix()}"
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


def _artifact_map(outcome: RoleProcessOutcomeV51) -> dict[str, str]:
    artifacts = {
        item.artifact_type: item.content
        for item in outcome.draft.proposed_artifacts
    }
    if len(artifacts) != len(outcome.draft.proposed_artifacts):
        raise BackhalfRuntimeError("role returned duplicate artifact types")
    return artifacts


def _artifact_json(
    outcome: RoleProcessOutcomeV51,
    artifact_type: str,
) -> dict[str, Any]:
    artifacts = _artifact_map(outcome)
    if set(artifacts) != {artifact_type}:
        raise BackhalfRuntimeError(
            f"{outcome.request.role_name} must return only {artifact_type}"
        )
    try:
        payload = json.loads(artifacts[artifact_type])
    except json.JSONDecodeError as exc:
        raise BackhalfRuntimeError(
            f"{artifact_type} is not valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise BackhalfRuntimeError(f"{artifact_type} must be a JSON object")
    return payload


def _selected_candidate(
    workspace: StageWorkspaceV50,
) -> tuple[ModelSpecV50, dict[str, Any]]:
    model = ModelSpecV50.model_validate_json(
        (workspace.root / "docs" / "model_spec.json").read_text(
            encoding="utf-8"
        )
    )
    candidate_payload = json.loads(
        (workspace.root / "docs" / "candidates.json").read_text(
            encoding="utf-8"
        )
    )
    candidates = candidate_payload.get("candidates", [])
    selected = next(
        (
            item
            for item in candidates
            if item.get("candidate_id") == model.selected_candidate_id
        ),
        None,
    )
    if not isinstance(selected, dict):
        raise BackhalfRuntimeError("selected S1 candidate is unavailable")
    return model, selected


def _assert_ode_compatible(
    model: ModelSpecV50,
    selected: dict[str, Any],
) -> None:
    if selected.get("candidate_id") != model.selected_candidate_id:
        raise BackhalfRuntimeError(
            "selected S1 candidate does not match the frozen model spec"
        )
    family = str(selected.get("model_family", "")).lower()
    form = str(selected.get("mathematical_form", "")).lower()
    family_ok = (
        "ode" in family
        or "ordinary differential" in family
        or "autonomous differential" in family
    )
    form_ok = "dx/dt" in form and any(
        token in form for token in ("r*x", "log(k/x)", "1-x/k", "dx/dt = 0")
    )
    if not family_ok or not form_ok:
        raise BackhalfRuntimeError(
            "selected S1 candidate is not compatible with the registered "
            "scalar autonomous ODE adapter"
        )


def ingest_ode_data_v59(
    workspace: StageWorkspaceV50,
    request: StudioODEDataRequestV59,
) -> Path:
    if workspace.current_gate("S1") is None:
        raise BackhalfRuntimeError("ODE data intake requires an open S1 gate")
    if workspace.current_gate("S2") is not None:
        raise BackhalfRuntimeError("S2 is already frozen")
    _, selected = _selected_candidate(workspace)
    model = ModelSpecV50.model_validate_json(
        (workspace.root / "docs" / "model_spec.json").read_text(
            encoding="utf-8"
        )
    )
    _assert_ode_compatible(model, selected)
    path = workspace.root / RAW_RELATIVE_PATH
    _write_json_new(path, request.model_dump(mode="json"))
    return path


def _read_bound_file(context: AdapterContextV50, relative_path: str) -> bytes:
    binding = next(
        (
            item
            for item in context.manifest.files
            if item.relative_path == relative_path
        ),
        None,
    )
    if binding is None:
        raise ValueError(f"{relative_path} is absent from the frozen manifest")
    path = (context.workspace_root / relative_path).resolve()
    root = context.workspace_root.resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError(relative_path)
    payload = path.read_bytes()
    if (
        len(payload) != binding.size_bytes
        or hashlib.sha256(payload).hexdigest() != binding.sha256
    ):
        raise ValueError(f"{relative_path} differs from the frozen manifest")
    return payload


class StudioODEObligationAdapterV59:
    adapter_id = "studio_scalar_ode_obligation_adapter"
    adapter_version = "5.9"

    def __init__(self, obligation: ValidationObligationV50) -> None:
        self.check_id = obligation.check_id
        self.level = obligation.level

    def run(self, context: AdapterContextV50) -> AdapterOutcomeV50:
        bundle = ODEScientificBundleV52.model_validate_json(
            _read_bound_file(context, BUNDLE_PATH)
        )
        binding = json.loads(
            _read_bound_file(context, ADAPTER_BINDING_PATH).decode("utf-8")
        )
        if binding.get("adapter_id") != ODE_ADAPTER_ID:
            raise ValueError("S2 adapter binding is not scalar ODE V5.2")
        evidence = next(
            item for item in bundle.levels if item.level == self.level
        )
        adapter_binding = next(
            item
            for item in context.manifest.files
            if item.relative_path == ADAPTER_BINDING_PATH
        )
        payload: dict[str, Any] = {
            "adapter_binding_hash": adapter_binding.sha256,
            "bundle_hash": bundle.bundle_hash,
            "level_evidence": evidence.model_dump(mode="json"),
            "fixture_only": bundle.fixture_only,
            "scientific_qualification_granted": False,
            "real_world_action_authorized": False,
        }
        if self.level == "L0":
            code = CodeManifestV50.model_validate_json(
                _read_bound_file(
                    context,
                    "results/code_manifest.json",
                )
            )
            payload["computation_artifact_sha256"] = code.replay_receipt_hash
        return AdapterOutcomeV50(
            status="PASS" if evidence.status == "PASS" else "FAIL",
            reason_code=(
                "studio_scalar_ode_level_passed"
                if evidence.status == "PASS"
                else f"studio_scalar_ode_level_{evidence.status.lower()}"
            ),
            thresholds=evidence.thresholds,
            metrics=evidence.metrics,
            evidence_payloads=[payload],
        )


class StudioBackhalfOrchestratorV59:
    """Drive the registered scalar ODE path from S2 through S6."""

    def __init__(
        self,
        *,
        workspace: StageWorkspaceV50,
        task_id: str,
        driver_factory: DriverFactory,
        event_callback: EventCallback,
    ) -> None:
        self.workspace = workspace
        self.task_id = task_id
        self.driver_factory = driver_factory
        self.event_callback = event_callback
        if workspace.current_gate("S1") is None:
            raise BackhalfRuntimeError("back-half execution requires an open S1 gate")

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
        stage: StageId,
        role_name: str,
        role_kind: Literal["generator", "reviewer"],
        subject_id: str,
        objective: str,
        public_inputs: dict[str, Any],
        allowed_candidate_ids: list[str],
    ) -> RoleProcessOutcomeV51:
        return self.driver_factory().run(
            task_id=self.task_id,
            stage=stage,
            role_name=role_name,
            role_kind=role_kind,
            subject_id=subject_id,
            objective=objective,
            public_inputs=public_inputs,
            allowed_candidate_ids=allowed_candidate_ids,
        )

    def _commit_review(
        self,
        *,
        stage: StageId,
        role: str,
        reviewer: RoleProcessOutcomeV51,
        producer_run_id: str,
        producer_context_id: str,
    ) -> None:
        manifest = self.workspace._manifest_for_stage(stage)
        checks = self.workspace._latest_checks(
            stage,
            str(manifest.manifest_hash),
        )
        allowed_inputs = sorted(
            {item.sha256 for item in manifest.files}
            | {
                str(item.result_hash)
                for item in checks.values()
                if item.result_hash is not None
            }
        )
        finding_ids = sorted(
            {
                f"finding-{hashlib.sha256(item.encode('utf-8')).hexdigest()[:16]}"
                for item in reviewer.draft.findings
            }
        )
        trace = self.workspace.commit_evidence(
            "codex_review_transport_trace_v59",
            {
                "stage": stage,
                "role": role,
                "producer_run_id": producer_run_id,
                "reviewer_run_id": reviewer.request.run_id,
                "producer_context_id": producer_context_id,
                "reviewer_context_id": reviewer.request.context_id,
                "context_isolation_attested": True,
                "allowed_input_hashes": allowed_inputs,
                "process_receipt": reviewer.receipt.model_dump(mode="json"),
            },
        )
        output = self.workspace.commit_evidence(
            "codex_review_output_v59",
            {
                "stage": stage,
                "role": role,
                "verdict": reviewer.draft.verdict,
                "finding_ids": finding_ids,
                "draft": reviewer.draft.model_dump(mode="json"),
            },
        )
        self.workspace.issue_review(
            stage=stage,
            review_id=f"review-{reviewer.request.run_id}",
            role=role,
            producer_run_id=producer_run_id,
            reviewer_run_id=reviewer.request.run_id,
            producer_context_id=producer_context_id,
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

    def _review_stage(
        self,
        *,
        stage: StageId,
        role: str,
        producer_run_id: str,
        producer_context_id: str,
        summary: dict[str, Any],
    ) -> RoleProcessOutcomeV51:
        manifest = self.workspace._manifest_for_stage(stage)
        checks = self.workspace._latest_checks(
            stage,
            str(manifest.manifest_hash),
        )
        reviewer = self._run_role(
            stage=stage,
            role_name=f"{stage.lower()}_{role}",
            role_kind="reviewer",
            subject_id=f"{stage.lower()}-work",
            objective=(
                f"Independently audit {stage} evidence against its frozen "
                "workflow and scientific boundary."
            ),
            public_inputs={
                "manifest": manifest.model_dump(mode="json"),
                "checks": {
                    key: value.model_dump(mode="json")
                    for key, value in checks.items()
                },
                "summary": summary,
                "gate_policy_hash": POLICIES[stage].policy_hash,
                "review_rule": (
                    "APPROVE only when the supplied frozen evidence is coherent "
                    "and every reserved authority remains false. Scientific "
                    "FAIL or NOT_RUN must not be reinterpreted as PASS."
                ),
            },
            allowed_candidate_ids=[],
        )
        if reviewer.draft.authority_claimed:
            raise BackhalfRuntimeError(f"{stage} reviewer claimed authority")
        if reviewer.draft.proposed_artifacts:
            raise BackhalfRuntimeError(f"{stage} reviewer proposed artifacts")
        self._commit_review(
            stage=stage,
            role=role,
            reviewer=reviewer,
            producer_run_id=producer_run_id,
            producer_context_id=producer_context_id,
        )
        return reviewer

    def _evaluate(
        self,
        *,
        stage: StageId,
        producer_run_id: str,
        producer_context_id: str,
        summary: dict[str, Any],
        extra_paths: list[str] | None = None,
        scientific_obligations: list[ValidationObligationV50] | None = None,
    ) -> str:
        actor: Literal["model", "harness"] = (
            "model" if stage in {"S2", "S5"} else "harness"
        )
        manifest = self.workspace.submit_stage(
            stage,
            actor=actor,
            extra_paths=extra_paths or [],
        )
        mechanical = self.workspace.run_mechanical_check(stage)
        if mechanical.status != "PASS":
            self._event(
                f"{stage.lower()}_mechanical_check_failed",
                "blocked",
                f"{stage} failed its harness-owned mechanical check",
                {
                    "reasons": mechanical.metrics,
                    "manifest_hash": manifest.manifest_hash,
                },
            )
            return "BLOCKED"
        if scientific_obligations:
            registry = CheckRegistryV50()
            for obligation in scientific_obligations:
                if obligation.applicability == "applicable":
                    registry.register(StudioODEObligationAdapterV59(obligation))
            for obligation in scientific_obligations:
                registry.execute(self.workspace, obligation)
        reviewers: dict[str, str] = {}
        for role in POLICIES[stage].required_review_roles:
            outcome = self._review_stage(
                stage=stage,
                role=role,
                producer_run_id=producer_run_id,
                producer_context_id=producer_context_id,
                summary=summary,
            )
            reviewers[role] = outcome.draft.verdict
        evaluation = self.workspace.evaluate_gate(stage)
        self._event(
            f"{stage.lower()}_gate_evaluated",
            "succeeded" if evaluation.decision == "OPEN" else "blocked",
            (
                f"{stage} gate opened"
                if evaluation.decision == "OPEN"
                else f"{stage} stopped: {evaluation.decision}"
            ),
            {
                "decision": evaluation.decision,
                "reasons": evaluation.reasons,
                "review_verdicts": reviewers,
                "scientific_qualification_granted": False,
                "real_world_action_authorized": False,
            },
        )
        return evaluation.decision

    def run_s2(self) -> str:
        if self.workspace.current_gate("S2"):
            return "OPEN"
        root = self.workspace.root
        raw_path = root / RAW_RELATIVE_PATH
        if not raw_path.is_file():
            raise BackhalfRuntimeError(
                "S2 requires user-supplied scalar ODE data"
            )
        for relative in (
            "data/ledger.json",
            "data/processed/manifest.json",
            PROCESSED_RELATIVE_PATH,
            ADAPTER_BINDING_PATH,
        ):
            if (root / relative).exists():
                raise BackhalfRuntimeError(
                    "S2 contains partial artifacts; automatic rerun is blocked"
                )
        request = StudioODEDataRequestV59.model_validate_json(
            raw_path.read_text(encoding="utf-8")
        )
        model, selected = _selected_candidate(self.workspace)
        _assert_ode_compatible(model, selected)
        baseline = self.workspace._raw_baseline_for_current_s2()
        if baseline is None:
            baseline = self.workspace.freeze_raw_inputs(actor="harness")
        snapshot = ODETimeSeriesSnapshotV52.seal(
            task_id=self.task_id,
            time_unit=request.time_unit,
            state_unit=request.state_unit,
            times=request.times,
            observations=request.observations,
            source_id=request.source_id,
            fixture_only=request.fixture_only,
        )
        required_ids = sorted(set(model.data_requirement_ids))
        producer = self._run_role(
            stage="S2",
            role_name="s2_data_steward",
            role_kind="generator",
            subject_id=model.selected_candidate_id,
            objective=(
                "Map the harness-frozen positive scalar series to every selected "
                "model data requirement without changing bytes or claiming quality."
            ),
            public_inputs={
                "objective": self.workspace.spec.objective,
                "selected_candidate": selected,
                "raw_baseline_hash": baseline.baseline_hash,
                "data_summary": {
                    "adapter_id": request.adapter_id,
                    "point_count": len(request.times),
                    "time_unit": request.time_unit,
                    "state_unit": request.state_unit,
                    "source_id": request.source_id,
                    "fixture_only": request.fixture_only,
                },
                "required_data_requirement_ids": required_ids,
                "required_artifacts": {
                    "data_mapping": DataMappingDraftV59.model_json_schema()
                },
            },
            allowed_candidate_ids=[model.selected_candidate_id],
        )
        if producer.draft.authority_claimed:
            raise BackhalfRuntimeError("S2 data steward claimed authority")
        mapping = DataMappingDraftV59.model_validate(
            _artifact_json(producer, "data_mapping")
        )
        if mapping.data_requirement_ids != required_ids:
            raise BackhalfRuntimeError(
                "S2 data mapping does not cover the selected model requirements"
            )
        commit_generator_outcome_v51(
            self.workspace,
            producer,
            execution_role="modeler",
            input_authority_hash=str(self.workspace.current_gate("S1")),
        )
        transform_path = root / "src" / "models" / "prepare_ode_data.py"
        transform_path.parent.mkdir(parents=True, exist_ok=True)
        if transform_path.exists():
            raise BackhalfRuntimeError("S2 transform already exists")
        transform_path.write_text(
            "\"\"\"Code-owned identity transform for a frozen ODE snapshot.\"\"\"\n"
            "\n"
            "def transform(payload: dict) -> dict:\n"
            "    return payload\n",
            encoding="utf-8",
            newline="\n",
        )
        _write_json_new(
            root / PROCESSED_RELATIVE_PATH,
            snapshot.model_dump(mode="json"),
        )
        processed_hash = _sha(root / PROCESSED_RELATIVE_PATH)
        transform_params = {
            "adapter_id": ODE_ADAPTER_ID,
            "identity_transform": True,
            "drop_missing": False,
        }
        entries = [
            DataLedgerEntryV50(
                data_item_id=requirement_id,
                semantic_name=f"{mapping.semantic_name}: {requirement_id}",
                units=mapping.units,
                source_kind="user",
                source_ref=request.source_id,
                raw_relative_path=RAW_RELATIVE_PATH,
                accessed_at=datetime.now(timezone.utc),
                license_status=request.license_status,
                raw_response_hash=_sha(raw_path),
                transform_script_relative_path="src/models/prepare_ode_data.py",
                transform_script_hash=_sha(transform_path),
                transform_params=transform_params,
                transform_params_hash=sha256_value(transform_params),
                processed_artifact_hash=processed_hash,
                quality_flags=[
                    *mapping.quality_flags,
                    "positive_scalar_series_contract",
                ],
            )
            for requirement_id in required_ids
        ]
        _write_json_new(
            root / "data" / "ledger.json",
            DataLedgerV50.seal(
                entries=entries,
                raw_baseline_tree_hash=baseline.raw_tree_hash,
            ).model_dump(mode="json"),
        )
        _write_json_new(
            root / "data" / "processed" / "manifest.json",
            ProcessedManifestV50(
                raw_baseline_tree_hash=baseline.raw_tree_hash,
                artifacts=[
                    ProcessedArtifactV50(
                        data_item_id=requirement_id,
                        relative_path=PROCESSED_RELATIVE_PATH,
                        artifact_hash=processed_hash,
                    )
                    for requirement_id in required_ids
                ],
            ).model_dump(mode="json"),
        )
        _write_json_new(
            root / ADAPTER_BINDING_PATH,
            {
                "schema_version": "5.9",
                "adapter_id": ODE_ADAPTER_ID,
                "selected_candidate_id": model.selected_candidate_id,
                "selected_candidate_structural_hash": (
                    model.selected_candidate_structural_hash
                ),
                "registered_families": [
                    "constant",
                    "exponential",
                    "gompertz",
                    "logistic",
                ],
                "raw_baseline_hash": baseline.baseline_hash,
                "scientific_qualification_granted": False,
                "real_world_action_authorized": False,
            },
        )
        self._event(
            "s2_data_materialized",
            "succeeded",
            "Frozen user data were mapped to the registered scalar ODE adapter",
            {
                "point_count": len(request.times),
                "data_requirement_count": len(required_ids),
                "raw_baseline_hash": baseline.baseline_hash,
                "fixture_only": request.fixture_only,
            },
        )
        return self._evaluate(
            stage="S2",
            producer_run_id=producer.request.run_id,
            producer_context_id=producer.request.context_id,
            summary={
                "adapter_id": ODE_ADAPTER_ID,
                "point_count": len(request.times),
                "fixture_only": request.fixture_only,
                "mapping": mapping.model_dump(mode="json"),
            },
            extra_paths=[ADAPTER_BINDING_PATH],
        )

    def _materialize_s3(self) -> tuple[ODEScientificBundleV52, ValidationPlanV50]:
        root = self.workspace.root
        for relative in (
            BUNDLE_PATH,
            REPLAY_INPUT_PATH,
            "results/index.json",
            "results/code_manifest.json",
        ):
            if (root / relative).exists():
                raise BackhalfRuntimeError(
                    "S3 contains partial artifacts; automatic rerun is blocked"
                )
        snapshot = ODETimeSeriesSnapshotV52.model_validate_json(
            (root / PROCESSED_RELATIVE_PATH).read_text(encoding="utf-8")
        )
        thresholds = ODEThresholdsV52.seal(bootstrap_replicates=40)
        _write_json_new(
            root / REPLAY_INPUT_PATH,
            {
                "snapshot": snapshot.model_dump(mode="json"),
                "thresholds": thresholds.model_dump(mode="json"),
            },
        )
        replay_hashes = run_ode_replays_v52(root / REPLAY_INPUT_PATH)
        bundle = build_ode_bundle_v52(
            snapshot=snapshot,
            thresholds=thresholds,
            replay_output_hashes=replay_hashes,
        )
        _write_json_new(root / BUNDLE_PATH, bundle.model_dump(mode="json"))
        source_path = root / "src" / "models" / "run_scalar_ode.py"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.exists():
            raise BackhalfRuntimeError("S3 adapter source already exists")
        source_path.write_text(
            "\"\"\"Registered execution entrypoint: fma.v5_2.ode_system.\"\"\"\n"
            "ADAPTER_ID = \"scalar_autonomous_ode_v52\"\n",
            encoding="utf-8",
            newline="\n",
        )
        environment_path = root / "results" / "environment.json"
        fermi_path = root / "results" / "fermi_estimate.json"
        toy_path = root / "checks" / "ode_toy_oracle.json"
        replay_receipt_path = root / "checks" / "ode_replay_receipt.json"
        _write_json_new(
            environment_path,
            {
                "schema_version": "5.9-environment",
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "adapter_id": ODE_ADAPTER_ID,
            },
        )
        _write_json_new(
            fermi_path,
            {
                "schema_version": "5.9-fermi",
                "observation_count": len(snapshot.times),
                "registered_family_count": len(bundle.candidates),
                "fit_scale": len(snapshot.times) * len(bundle.candidates),
            },
        )
        l2 = next(item for item in bundle.levels if item.level == "L2")
        _write_json_new(toy_path, l2.model_dump(mode="json"))
        replay_command = (
            "python -m fma.v5_2.ode_system replay "
            "checks/ode_replay_input.json"
        )
        source_tree_hash = _tree_hash(root / "src")
        _write_json_new(
            replay_receipt_path,
            {
                "schema_version": "5.9-replay",
                "replay_command": replay_command,
                "source_tree_hash": source_tree_hash,
                "environment_hash": _sha(environment_path),
                "random_seed": 104729,
                "exit_code": 0,
                "passed": len(replay_hashes) == 2
                and len(set(replay_hashes)) == 1,
                "deterministic_output_hashes": replay_hashes,
            },
        )
        _write_json_new(
            root / "results" / "code_manifest.json",
            CodeManifestV50(
                source_tree_hash=source_tree_hash,
                environment_ref="results/environment.json",
                environment_hash=_sha(environment_path),
                replay_command=replay_command,
                replay_receipt_ref="checks/ode_replay_receipt.json",
                replay_receipt_hash=_sha(replay_receipt_path),
                random_seed=104729,
                tolerance_policy=(
                    "Frozen ODE V5.2 thresholds and deterministic replay hashes"
                ),
                fermi_estimate_ref="results/fermi_estimate.json",
                fermi_estimate_hash=_sha(fermi_path),
                toy_oracle_refs=["checks/ode_toy_oracle.json"],
                toy_oracle_hashes={
                    "checks/ode_toy_oracle.json": _sha(toy_path)
                },
            ).model_dump(mode="json"),
        )
        selected = next(
            item
            for item in bundle.candidates
            if item.candidate_id == bundle.selected_candidate_id
        )
        l4 = next(item for item in bundle.levels if item.level == "L4")
        low = l4.metrics.get("forecast_interval_low")
        high = l4.metrics.get("forecast_interval_high")
        if not isinstance(low, (int, float)) or not isinstance(
            high, (int, float)
        ):
            candidate_forecasts = [
                item.forecast_value for item in bundle.candidates
            ]
            low = min(candidate_forecasts)
            high = max(candidate_forecasts)
        point_path = root / "results" / "artifacts" / "forecast.json"
        interval_path = (
            root / "results" / "artifacts" / "forecast_interval.json"
        )
        _write_json_new(
            point_path,
            {
                "schema_version": "5.9-result",
                "result_id": "forecast",
                "value": selected.forecast_value,
                "interval_low": None,
                "interval_high": None,
                "units": snapshot.state_unit,
            },
        )
        _write_json_new(
            interval_path,
            {
                "schema_version": "5.9-result",
                "result_id": "forecast_interval",
                "value": None,
                "interval_low": float(low),
                "interval_high": float(high),
                "units": snapshot.state_unit,
            },
        )
        _write_json_new(
            root / "results" / "index.json",
            ResultIndexV50(
                records=[
                    ResultRecordV50(
                        result_id="forecast",
                        relative_path=(
                            "results/artifacts/forecast.json"
                        ),
                        artifact_hash=_sha(point_path),
                        value=selected.forecast_value,
                        units=snapshot.state_unit,
                    ),
                    ResultRecordV50(
                        result_id="forecast_interval",
                        relative_path=(
                            "results/artifacts/forecast_interval.json"
                        ),
                        artifact_hash=_sha(interval_path),
                        interval_low=float(low),
                        interval_high=float(high),
                        units=snapshot.state_unit,
                    ),
                ]
            ).model_dump(mode="json"),
        )
        plan = ValidationPlanV50.model_validate_json(
            (root / "docs" / "validation_plan.json").read_text(
                encoding="utf-8"
            )
        )
        return bundle, plan

    def run_s3(self) -> str:
        if self.workspace.current_gate("S3"):
            return "OPEN"
        if self.workspace.current_gate("S2") is None:
            raise BackhalfRuntimeError("S3 requires an open S2 gate")
        bundle, plan = self._materialize_s3()
        obligations = [
            item for item in plan.obligations if item.stage == "S3"
        ]
        self._event(
            "s3_computation_completed",
            "succeeded",
            "Registered scalar ODE candidates were fitted and replayed",
            {
                "selected_family": bundle.selected_candidate_id,
                "bundle_hash": bundle.bundle_hash,
                "level_statuses": {
                    item.level: item.status for item in bundle.levels
                },
                "scientific_acceptance": bundle.scientific_acceptance,
                "fixture_only": bundle.fixture_only,
            },
        )
        return self._evaluate(
            stage="S3",
            producer_run_id="s3-harness-ode-executor",
            producer_context_id=f"s3-harness-{uuid4().hex[:16]}",
            summary={
                "adapter_id": ODE_ADAPTER_ID,
                "bundle_hash": bundle.bundle_hash,
                "levels": {
                    item.level: item.status for item in bundle.levels
                },
            },
            extra_paths=[BUNDLE_PATH, ADAPTER_BINDING_PATH],
            scientific_obligations=obligations,
        )

    def run_s4(self) -> str:
        if self.workspace.current_gate("S4"):
            return "OPEN"
        if self.workspace.current_gate("S3") is None:
            raise BackhalfRuntimeError("S4 requires an open S3 gate")
        root = self.workspace.root
        for relative in (
            "results/verification_summary.json",
            "results/uq_summary.json",
        ):
            if (root / relative).exists():
                raise BackhalfRuntimeError(
                    "S4 contains partial artifacts; automatic rerun is blocked"
                )
        bundle = ODEScientificBundleV52.model_validate_json(
            (root / BUNDLE_PATH).read_text(encoding="utf-8")
        )
        plan = ValidationPlanV50.model_validate_json(
            (root / "docs" / "validation_plan.json").read_text(
                encoding="utf-8"
            )
        )
        obligations = [
            item for item in plan.obligations if item.stage == "S4"
        ]
        _write_json_new(
            root / "results" / "verification_summary.json",
            {
                "schema_version": "5.9",
                "validation_plan_hash": plan.plan_hash,
                "check_ids": [item.check_id for item in obligations],
                "adapter_id": ODE_ADAPTER_ID,
                "bundle_hash": bundle.bundle_hash,
                "level_statuses": {
                    item.level: item.status for item in bundle.levels
                },
                "scientific_acceptance": bundle.scientific_acceptance,
                "scientific_qualification_granted": False,
            },
        )
        l4 = next(item for item in bundle.levels if item.level == "L4")
        disagreement = l4.metrics.get(
            "ensemble_forecast_coefficient_of_variation"
        )
        _write_json_new(
            root / "results" / "uq_summary.json",
            UQSummaryV50(
                claims=[
                    UQClaimV50(
                        claim_id="forecast_claim",
                        result_id="forecast",
                        interval_result_id="forecast_interval",
                        support_status=(
                            "in_support"
                            if l4.status == "PASS"
                            else "unknown"
                        ),
                        ensemble_disagreement=(
                            float(disagreement)
                            if isinstance(disagreement, (int, float))
                            and math.isfinite(float(disagreement))
                            else 1.0
                        ),
                    )
                ]
            ).model_dump(mode="json"),
        )
        self._event(
            "s4_verification_materialized",
            "succeeded",
            "Holdout and uncertainty evidence were projected from the ODE bundle",
            {
                "l3_status": next(
                    item.status for item in bundle.levels if item.level == "L3"
                ),
                "l4_status": l4.status,
                "scientific_acceptance": bundle.scientific_acceptance,
            },
        )
        return self._evaluate(
            stage="S4",
            producer_run_id="s4-harness-ode-verifier",
            producer_context_id=f"s4-harness-{uuid4().hex[:16]}",
            summary={
                "adapter_id": ODE_ADAPTER_ID,
                "bundle_hash": bundle.bundle_hash,
                "scientific_acceptance": bundle.scientific_acceptance,
                "fixture_only": bundle.fixture_only,
            },
            extra_paths=[BUNDLE_PATH, ADAPTER_BINDING_PATH],
            scientific_obligations=obligations,
        )

    def run_s5(self) -> str:
        if self.workspace.current_gate("S5"):
            return "OPEN"
        if self.workspace.current_gate("S4") is None:
            raise BackhalfRuntimeError("S5 requires an open S4 gate")
        root = self.workspace.root
        dossier_path = root / "results" / "decision_dossier.json"
        if dossier_path.exists():
            raise BackhalfRuntimeError(
                "S5 contains partial artifacts; automatic rerun is blocked"
            )
        results = ResultIndexV50.model_validate_json(
            (root / "results" / "index.json").read_text(encoding="utf-8")
        )
        uq = UQSummaryV50.model_validate_json(
            (root / "results" / "uq_summary.json").read_text(
                encoding="utf-8"
            )
        )
        plan = ValidationPlanV50.model_validate_json(
            (root / "docs" / "validation_plan.json").read_text(
                encoding="utf-8"
            )
        )
        model = ModelSpecV50.model_validate_json(
            (root / "docs" / "model_spec.json").read_text(
                encoding="utf-8"
            )
        )
        producer = self._run_role(
            stage="S5",
            role_name="s5_decision_writer",
            role_kind="generator",
            subject_id=model.selected_candidate_id,
            objective=(
                "Draft one bounded report-only interpretation of the frozen "
                "result and uncertainty evidence."
            ),
            public_inputs={
                "objective": self.workspace.spec.objective,
                "selected_candidate_id": model.selected_candidate_id,
                "results": results.model_dump(mode="json"),
                "uq": uq.model_dump(mode="json"),
                "authority_rule": (
                    "Narrative only. The harness owns bindings, next_action, "
                    "prediction registration, and every external action."
                ),
                "required_artifacts": {
                    "decision_narrative": (
                        DecisionNarrativeDraftV59.model_json_schema()
                    )
                },
            },
            allowed_candidate_ids=[model.selected_candidate_id],
        )
        if producer.draft.authority_claimed:
            raise BackhalfRuntimeError("S5 writer claimed authority")
        narrative = DecisionNarrativeDraftV59.model_validate(
            _artifact_json(producer, "decision_narrative")
        )
        commit_generator_outcome_v51(
            self.workspace,
            producer,
            execution_role="writer",
            input_authority_hash=str(self.workspace.current_gate("S4")),
        )
        high_disagreement = any(
            item.ensemble_disagreement
            >= plan.ensemble_disagreement_threshold
            for item in uq.claims
        )
        unsupported = any(
            item.support_status != "in_support" for item in uq.claims
        )
        next_action = (
            "return_to_data_acquisition"
            if high_disagreement or unsupported
            else "draft_report_only"
        )
        _write_json_new(
            dossier_path,
            DecisionDossierV50(
                assertions=[
                    DecisionAssertionV50(
                        assertion_id="forecast_interpretation",
                        statement=narrative.statement,
                        result_ids=["forecast"],
                        uq_claim_ids=["forecast_claim"],
                    )
                ],
                high_disagreement_detected=high_disagreement,
                next_action=next_action,
                real_world_action_authorized=False,
            ).model_dump(mode="json"),
        )
        self._event(
            "s5_decision_dossier_materialized",
            "succeeded",
            "A bounded decision dossier was bound to results and UQ",
            {
                "next_action": next_action,
                "high_disagreement_detected": high_disagreement,
                "limitations": narrative.limitations,
                "real_world_action_authorized": False,
            },
        )
        return self._evaluate(
            stage="S5",
            producer_run_id=producer.request.run_id,
            producer_context_id=producer.request.context_id,
            summary={
                "next_action": next_action,
                "high_disagreement_detected": high_disagreement,
                "real_world_action_authorized": False,
            },
        )

    def run_s6(self) -> str:
        if self.workspace.current_gate("S6"):
            return "OPEN"
        if self.workspace.current_gate("S5") is None:
            raise BackhalfRuntimeError("S6 requires an open S5 gate")
        root = self.workspace.root
        required = (
            "results/values.json",
            "paper/main.template.tex",
            "paper/build/main.tex",
            "paper/build/main.pdf",
            "paper/build/build_receipt.json",
        )
        if any((root / relative).exists() for relative in required):
            raise BackhalfRuntimeError(
                "S6 contains partial artifacts; automatic rerun is blocked"
            )
        results = ResultIndexV50.model_validate_json(
            (root / "results" / "index.json").read_text(encoding="utf-8")
        )
        values: dict[str, float] = {}
        for record in results.records:
            if record.value is not None:
                values[record.result_id] = record.value
            if (
                record.interval_low is not None
                and record.interval_high is not None
            ):
                values[f"{record.result_id}_low"] = record.interval_low
                values[f"{record.result_id}_high"] = record.interval_high
        _write_json_new(root / "results" / "values.json", values)
        template_path = root / "paper" / "main.template.tex"
        template_path.parent.mkdir(parents=True, exist_ok=True)
        template_path.write_text(
            "\\documentclass{article}\n"
            "\\begin{document}\n"
            "\\section*{FMA Modeling Report}\n"
            "Registered scalar model forecast: {{result.forecast}}.\\\\\n"
            "Frozen uncertainty interval: "
            "{{result.forecast_interval_low}} to "
            "{{result.forecast_interval_high}}.\\\\\n"
            "This report grants neither scientific qualification nor "
            "real-world action authority.\n"
            "\\end{document}\n",
            encoding="utf-8",
            newline="\n",
        )
        receipt = build_paper(root)
        self._event(
            "s6_paper_built",
            "succeeded",
            "The report PDF was built from machine-readable result values",
            {
                "receipt_hash": receipt.receipt_hash,
                "scientific_correctness_established": False,
                "real_world_action_authorized": False,
            },
        )
        return self._evaluate(
            stage="S6",
            producer_run_id="s6-harness-paper-builder",
            producer_context_id=f"s6-harness-{uuid4().hex[:16]}",
            summary={
                "paper_build_receipt_hash": receipt.receipt_hash,
                "scientific_correctness_established": False,
                "real_world_action_authorized": False,
            },
        )

    def run(self) -> dict[str, str]:
        decisions: dict[str, str] = {}
        for stage, runner in (
            ("S2", self.run_s2),
            ("S3", self.run_s3),
            ("S4", self.run_s4),
            ("S5", self.run_s5),
            ("S6", self.run_s6),
        ):
            decision = runner()
            decisions[stage] = decision
            if decision != "OPEN":
                break
        return decisions


def backhalf_summary_v59(workspace: StageWorkspaceV50) -> dict[str, Any]:
    root = workspace.root
    bundle_path = root / BUNDLE_PATH
    bundle = (
        ODEScientificBundleV52.model_validate_json(
            bundle_path.read_text(encoding="utf-8")
        )
        if bundle_path.is_file()
        else None
    )
    return {
        "schema_version": "5.9",
        "adapter_id": ODE_ADAPTER_ID,
        "data_received": (root / RAW_RELATIVE_PATH).is_file(),
        "workflow_complete": workspace.current_gate("S6") is not None,
        "selected_scientific_family": (
            bundle.selected_candidate_id if bundle is not None else None
        ),
        "level_statuses": (
            {item.level: item.status for item in bundle.levels}
            if bundle is not None
            else {}
        ),
        "scientific_acceptance": (
            bundle.scientific_acceptance if bundle is not None else False
        ),
        "fixture_only": bundle.fixture_only if bundle is not None else None,
        "scientific_qualification_granted": False,
        "real_world_action_authorized": False,
    }


__all__ = [
    "BackhalfRuntimeError",
    "DataMappingDraftV59",
    "DecisionNarrativeDraftV59",
    "ODE_ADAPTER_ID",
    "StudioBackhalfOrchestratorV59",
    "StudioODEDataRequestV59",
    "backhalf_summary_v59",
    "ingest_ode_data_v59",
]
