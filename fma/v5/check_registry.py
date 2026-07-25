"""Typed domain-check adapter boundary for V5.

The registry does not contain universal scientific validators.  It binds one
pre-registered validation obligation to one concrete adapter implementation
and records missing/crashed adapters as ``NOT_RUN``/``ERROR`` rather than
manufacturing a green result.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from pydantic import Field, model_validator

from fma.schemas import StrictModel

from .workspace_schemas import (
    CheckStatus,
    StageArtifactManifestV50,
    ValidationObligationV50,
)


class AdapterOutcomeV50(StrictModel):
    status: CheckStatus
    reason_code: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
    thresholds: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    evidence_payloads: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_outcome(self) -> "AdapterOutcomeV50":
        if self.status == "PASS" and not self.evidence_payloads:
            raise ValueError("scientific PASS requires computation evidence payloads")
        if self.status in {"NOT_APPLICABLE", "NOT_RUN", "ERROR"}:
            raise ValueError(
                "adapter outcomes are PASS/FAIL; missing, N/A, and exceptions "
                "are assigned by the harness"
            )
        return self


@dataclass(frozen=True)
class AdapterContextV50:
    workspace_root: Path
    manifest: StageArtifactManifestV50
    obligation: ValidationObligationV50


class ScientificCheckAdapterV50(Protocol):
    adapter_id: str
    adapter_version: str
    check_id: str
    level: str

    def run(self, context: AdapterContextV50) -> AdapterOutcomeV50:
        ...


class CheckRegistryV50:
    """In-memory implementation registry; identity is frozen into each receipt."""

    def __init__(self) -> None:
        self._adapters: dict[str, ScientificCheckAdapterV50] = {}

    def register(self, adapter: ScientificCheckAdapterV50) -> None:
        if adapter.check_id in self._adapters:
            raise ValueError(f"duplicate adapter for check {adapter.check_id}")
        if adapter.level not in {"L0", "L1", "L2", "L3", "L4"}:
            raise ValueError("scientific adapters must implement L0-L4 checks")
        self._adapters[adapter.check_id] = adapter

    @staticmethod
    def _implementation_hash(adapter: object) -> str:
        source = inspect.getsourcefile(adapter.__class__)
        if source is None:
            raise RuntimeError("adapter implementation has no auditable source file")
        path = Path(source).resolve()
        if not path.is_file():
            raise RuntimeError("adapter implementation source is unavailable")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def execute(
        self,
        workspace: Any,
        obligation: ValidationObligationV50,
    ) -> Any:
        """Execute or explicitly record the obligation on a StageWorkspaceV50."""

        manifest = workspace._manifest_for_stage(obligation.stage)
        registry_code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if obligation.applicability == "not_applicable":
            now = datetime.now(timezone.utc)
            execution = workspace._record_adapter_execution(
                stage=obligation.stage,
                check_id=obligation.check_id,
                level=obligation.level,
                applicability="not_applicable",
                status="NOT_APPLICABLE",
                execution_mode="frozen_not_applicable",
                adapter_invoked=False,
                scientific_computation_performed=False,
                adapter_id="v5_applicability_registry",
                adapter_version="5.0",
                adapter_code_hash=registry_code_hash,
                evidence_refs=[],
                started_at=now,
                finished_at=now,
            )
            return workspace.issue_check(
                stage=obligation.stage,
                check_id=obligation.check_id,
                level=obligation.level,
                evidence_class=obligation.evidence_class,
                applicability="not_applicable",
                status="NOT_APPLICABLE",
                reason_code="frozen_not_applicable",
                adapter_id="v5_applicability_registry",
                adapter_version="5.0",
                adapter_code_hash=registry_code_hash,
                evidence_refs=[],
                adapter_execution_receipt_hash=execution.receipt_hash,
                subject_hashes=[str(manifest.manifest_hash)],
                scope=workspace.spec.evidence_scope,
                started_at=execution.started_at,
                finished_at=execution.finished_at,
            )

        adapter = self._adapters.get(obligation.check_id)
        if adapter is None:
            now = datetime.now(timezone.utc)
            execution = workspace._record_adapter_execution(
                stage=obligation.stage,
                check_id=obligation.check_id,
                level=obligation.level,
                applicability="applicable",
                status="NOT_RUN",
                execution_mode="adapter_missing",
                adapter_invoked=False,
                scientific_computation_performed=False,
                adapter_id="v5_missing_adapter",
                adapter_version="5.0",
                adapter_code_hash=registry_code_hash,
                evidence_refs=[],
                started_at=now,
                finished_at=now,
            )
            return workspace.issue_check(
                stage=obligation.stage,
                check_id=obligation.check_id,
                level=obligation.level,
                evidence_class=obligation.evidence_class,
                applicability="applicable",
                status="NOT_RUN",
                reason_code="adapter_missing",
                adapter_id="v5_missing_adapter",
                adapter_version="5.0",
                adapter_code_hash=registry_code_hash,
                evidence_refs=[],
                adapter_execution_receipt_hash=execution.receipt_hash,
                subject_hashes=[str(manifest.manifest_hash)],
                scope=workspace.spec.evidence_scope,
                started_at=execution.started_at,
                finished_at=execution.finished_at,
            )

        if adapter.check_id != obligation.check_id or adapter.level != obligation.level:
            raise ValueError("adapter metadata does not match frozen obligation")
        implementation_hash = self._implementation_hash(adapter)
        started_at = datetime.now(timezone.utc)
        execution_mode = "adapter_run"
        try:
            outcome = adapter.run(
                AdapterContextV50(
                    workspace_root=workspace.root,
                    manifest=manifest,
                    obligation=obligation,
                )
            )
        except Exception as exc:
            outcome = AdapterOutcomeV50(
                status="FAIL",
                reason_code="adapter_exception_placeholder",
                metrics={"exception_type": type(exc).__name__},
            )
            status = "ERROR"
            reason_code = "adapter_exception"
            execution_mode = "adapter_exception"
        else:
            status = outcome.status
            reason_code = outcome.reason_code
        evidence_refs = []
        for index, payload in enumerate(
            outcome.evidence_payloads if execution_mode == "adapter_run" else []
        ):
            evidence = workspace.commit_evidence(
                "scientific_adapter_evidence_v50",
                {
                    "check_id": obligation.check_id,
                    "adapter_id": adapter.adapter_id,
                    "adapter_version": adapter.adapter_version,
                    "implementation_hash": implementation_hash,
                    "manifest_hash": manifest.manifest_hash,
                    "evidence_index": index,
                    "payload": payload,
                },
            )
            evidence_refs.append(evidence.sha256)
        finished_at = datetime.now(timezone.utc)
        execution = workspace._record_adapter_execution(
            stage=obligation.stage,
            check_id=obligation.check_id,
            level=obligation.level,
            applicability="applicable",
            status=status,
            execution_mode=execution_mode,
            adapter_invoked=True,
            scientific_computation_performed=(
                execution_mode == "adapter_run" and bool(evidence_refs)
            ),
            adapter_id=adapter.adapter_id,
            adapter_version=adapter.adapter_version,
            adapter_code_hash=implementation_hash,
            evidence_refs=evidence_refs,
            started_at=started_at,
            finished_at=finished_at,
        )
        return workspace.issue_check(
            stage=obligation.stage,
            check_id=obligation.check_id,
            level=obligation.level,
            evidence_class=obligation.evidence_class,
            applicability="applicable",
            status=status,
            reason_code=reason_code,
            adapter_id=adapter.adapter_id,
            adapter_version=adapter.adapter_version,
            adapter_code_hash=implementation_hash,
            evidence_refs=evidence_refs,
            adapter_execution_receipt_hash=execution.receipt_hash,
            subject_hashes=[str(manifest.manifest_hash)],
            thresholds=outcome.thresholds,
            metrics=outcome.metrics,
            scope=workspace.spec.evidence_scope,
            started_at=execution.started_at,
            finished_at=execution.finished_at,
        )


__all__ = [
    "AdapterContextV50",
    "AdapterOutcomeV50",
    "CheckRegistryV50",
    "ScientificCheckAdapterV50",
]
