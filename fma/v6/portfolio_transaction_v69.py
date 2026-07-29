"""Authenticated, restart-safe transaction for the V6.9 development portfolio.

The transaction has three explicit mutations:

``freeze`` (observation-free) -> ``stage_snapshot`` (data only) -> ``execute``.

``project`` is read-only.  ``reconcile`` is the only recovery mutation.  This
module commits development evidence only; it never writes a stage gate or
grants scientific qualification/action authority.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Callable, Iterator, Literal

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5.stage_workspace import StageWorkspaceError, StageWorkspaceV50
from fma.v5_2.ode_system import ODEThresholdsV52

from .capability_sdk_v68 import CapabilityQueryV68, CapabilityRegistryV68
from .portfolio_protocol_v68 import (
    BranchBudgetV68,
    ModelingPortfolioProtocolV68,
    PortfolioBudgetV68,
)
from .portfolio_runtime_v69 import (
    BranchExecutionReceiptV69,
    CommonRollingOriginPlanV69,
    DevelopmentPortfolioRunV69,
    MAXIMUM_ROLLING_ORIGINS_V69,
    PersistenceBaselinePolicyV69,
    PositiveSeriesSnapshotV69,
    build_common_rolling_origin_plan_v69,
    compile_default_development_portfolio_protocol_v69,
    execute_development_portfolio_v69,
)
from .positive_log_increment_v68 import PositiveLogIncrementThresholdsV68


PORTFOLIO_TRANSACTION_INTENT_KIND_V69 = "portfolio_transaction_intent_v69"
PORTFOLIO_RUN_INTENT_KIND_V69 = "portfolio_run_intent_v69"
PORTFOLIO_BRANCH_RECEIPT_KIND_V69 = "portfolio_branch_receipt_v69"
PORTFOLIO_RUN_KIND_V69 = "portfolio_run_v69"
PORTFOLIO_COMPLETION_KIND_V69 = "portfolio_completion_v69"

PortfolioTransactionStatusV69 = Literal[
    "NOT_STARTED",
    "FROZEN",
    "DATA_STAGED",
    "RECOVERY_PENDING",
    "COMPLETED",
    "STALE_PENDING",
]


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _assert_authenticated(
    model: StrictModel,
    *,
    auth_tag: str | None,
    object_hash: str | None,
    expected_hash: str,
    label: str,
) -> None:
    type(model).model_validate(model.model_dump(mode="json"))
    if not auth_tag or not object_hash or object_hash != expected_hash:
        raise ValueError(f"V6.9 {label} is not authenticated and sealed")


class PortfolioTransactionIntentV69(StrictModel):
    """Observation-free freeze of protocol, budgets, thresholds, and policy."""

    schema_version: Literal["6.9-portfolio-transaction-intent"] = (
        "6.9-portfolio-transaction-intent"
    )
    workspace_spec_hash: Sha256
    s0_gate_hash: Sha256
    protocol: ModelingPortfolioProtocolV68
    planned_observation_count: Annotated[int, Field(ge=35)]
    state_unit: Identifier
    time_unit: Identifier
    snapshot_task_id: Identifier | None = None
    initial_training_count: Annotated[int, Field(ge=34)] = 34
    maximum_origins: Annotated[
        int,
        Field(ge=1, le=MAXIMUM_ROLLING_ORIGINS_V69),
    ] | None = None
    baseline_policy: PersistenceBaselinePolicyV69
    ode_thresholds: ODEThresholdsV52
    log_increment_thresholds: PositiveLogIncrementThresholdsV68
    observation_values_accessed: Literal[False] = False
    observed_statistics_accessed: Literal[False] = False
    private_acceptance_data_accessed: Literal[False] = False
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    intent_hash: Sha256 | None = None
    intent_is_stage_evidence: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_intent(self) -> "PortfolioTransactionIntentV69":
        self.protocol.assert_sealed()
        self.baseline_policy.assert_sealed()
        self.ode_thresholds.assert_sealed()
        self.log_increment_thresholds.assert_sealed()
        if (
            self.protocol.runtime_mode != "development_sandbox"
            or self.protocol.workspace_spec_hash != self.workspace_spec_hash
            or self.protocol.s0_gate_hash != self.s0_gate_hash
            or self.protocol.common_loss.loss_unit != self.state_unit
        ):
            raise ValueError("V6.9 freeze protocol authority binding differs")
        available_origins = (
            self.planned_observation_count - self.initial_training_count
        )
        if available_origins < 1 or (
            self.maximum_origins is not None
            and self.maximum_origins > available_origins
        ):
            raise ValueError("V6.9 frozen rolling-origin budget is impossible")
        if self.authority_auth_tag and not self.intent_hash:
            raise ValueError("authenticated V6.9 freeze intent requires intent hash")
        if self.intent_hash and self.intent_hash != self.content_hash():
            raise ValueError("V6.9 freeze intent hash differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "authority_auth_tag", "intent_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "intent_hash")

    def authenticate(self, tag: str) -> "PortfolioTransactionIntentV69":
        if self.authority_auth_tag is not None or self.intent_hash is not None:
            raise ValueError("V6.9 freeze intent is already authenticated")
        payload = self.model_dump(mode="json")
        payload["authority_auth_tag"] = tag
        payload["intent_hash"] = sha256_value(
            {key: value for key, value in payload.items() if key != "intent_hash"}
        )
        return type(self).model_validate(payload)

    def assert_sealed(self) -> None:
        _assert_authenticated(
            self,
            auth_tag=self.authority_auth_tag,
            object_hash=self.intent_hash,
            expected_hash=self.content_hash(),
            label="freeze intent",
        )


class PortfolioRunIntentV69(StrictModel):
    """Authenticated data-only intent; committing it executes no capability."""

    schema_version: Literal["6.9-portfolio-run-intent"] = (
        "6.9-portfolio-run-intent"
    )
    workspace_spec_hash: Sha256
    s0_gate_hash: Sha256
    freeze_intent_hash: Sha256
    freeze_artifact_hash: Sha256
    protocol_hash: Sha256
    snapshot: PositiveSeriesSnapshotV69
    origin_plan: CommonRollingOriginPlanV69
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    run_intent_hash: Sha256 | None = None
    run_started: Literal[False] = False
    intent_is_stage_evidence: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_intent(self) -> "PortfolioRunIntentV69":
        self.snapshot.assert_sealed()
        self.origin_plan.assert_sealed()
        if self.origin_plan.snapshot_hash != self.snapshot.snapshot_hash:
            raise ValueError("V6.9 run intent plan uses another snapshot")
        if self.authority_auth_tag and not self.run_intent_hash:
            raise ValueError("authenticated V6.9 run intent requires intent hash")
        if self.run_intent_hash and self.run_intent_hash != self.content_hash():
            raise ValueError("V6.9 run intent hash differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "authority_auth_tag", "run_intent_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "run_intent_hash")

    def authenticate(self, tag: str) -> "PortfolioRunIntentV69":
        if self.authority_auth_tag is not None or self.run_intent_hash is not None:
            raise ValueError("V6.9 run intent is already authenticated")
        payload = self.model_dump(mode="json")
        payload["authority_auth_tag"] = tag
        payload["run_intent_hash"] = sha256_value(
            {
                key: value
                for key, value in payload.items()
                if key != "run_intent_hash"
            }
        )
        return type(self).model_validate(payload)

    def assert_sealed(self) -> None:
        _assert_authenticated(
            self,
            auth_tag=self.authority_auth_tag,
            object_hash=self.run_intent_hash,
            expected_hash=self.content_hash(),
            label="run intent",
        )


class PortfolioTransactionCompletionV69(StrictModel):
    """Authenticated completion binding every committed transaction output."""

    schema_version: Literal["6.9-portfolio-transaction-completion"] = (
        "6.9-portfolio-transaction-completion"
    )
    workspace_spec_hash: Sha256
    s0_gate_hash: Sha256
    freeze_intent_hash: Sha256
    freeze_artifact_hash: Sha256
    run_intent_hash: Sha256
    run_intent_artifact_hash: Sha256
    branch_artifact_hashes: dict[Identifier, Sha256] = Field(
        min_length=2,
        max_length=2,
    )
    branch_receipt_hashes: dict[Identifier, Sha256] = Field(
        min_length=2,
        max_length=2,
    )
    run_artifact_hash: Sha256
    run_hash: Sha256
    completed_at: datetime
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    completion_hash: Sha256 | None = None
    completion_is_stage_evidence: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_completion(self) -> "PortfolioTransactionCompletionV69":
        if self.completed_at.tzinfo is None or self.completed_at.utcoffset() is None:
            raise ValueError("V6.9 completion time must include a timezone")
        for mapping in (
            self.branch_artifact_hashes,
            self.branch_receipt_hashes,
        ):
            if list(mapping) != sorted(mapping):
                raise ValueError("V6.9 completion branch maps must be sorted")
        if set(self.branch_artifact_hashes) != set(
            self.branch_receipt_hashes
        ):
            raise ValueError("V6.9 completion branch maps differ")
        if self.authority_auth_tag and not self.completion_hash:
            raise ValueError("authenticated V6.9 completion requires hash")
        if self.completion_hash and self.completion_hash != self.content_hash():
            raise ValueError("V6.9 completion hash differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "authority_auth_tag", "completion_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "completion_hash")

    def authenticate(self, tag: str) -> "PortfolioTransactionCompletionV69":
        if self.authority_auth_tag is not None or self.completion_hash is not None:
            raise ValueError("V6.9 completion is already authenticated")
        payload = self.model_dump(mode="json")
        payload["authority_auth_tag"] = tag
        payload["completion_hash"] = sha256_value(
            {
                key: value
                for key, value in payload.items()
                if key != "completion_hash"
            }
        )
        return type(self).model_validate(payload)

    def assert_sealed(self) -> None:
        _assert_authenticated(
            self,
            auth_tag=self.authority_auth_tag,
            object_hash=self.completion_hash,
            expected_hash=self.content_hash(),
            label="completion",
        )


@dataclass(frozen=True)
class PortfolioTransactionStateV69:
    status: PortfolioTransactionStatusV69
    intent_ref: ArtifactRef | None = None
    intent: PortfolioTransactionIntentV69 | None = None
    run_intent_ref: ArtifactRef | None = None
    run_intent: PortfolioRunIntentV69 | None = None
    branch_refs: dict[str, ArtifactRef] = field(default_factory=dict)
    branch_receipts: dict[str, BranchExecutionReceiptV69] = field(
        default_factory=dict
    )
    run_ref: ArtifactRef | None = None
    run: DevelopmentPortfolioRunV69 | None = None
    completion_ref: ArtifactRef | None = None
    completion: PortfolioTransactionCompletionV69 | None = None

    @property
    def protocol_hash(self) -> str | None:
        return (
            str(self.intent.protocol.protocol_hash)
            if self.intent is not None
            else None
        )

    @property
    def snapshot_hash(self) -> str | None:
        return (
            str(self.run_intent.snapshot.snapshot_hash)
            if self.run_intent is not None
            else None
        )

    @property
    def plan_hash(self) -> str | None:
        return (
            str(self.run_intent.origin_plan.plan_hash)
            if self.run_intent is not None
            else None
        )

    @property
    def branch_statuses(self) -> dict[str, str]:
        return {
            branch_id: receipt.execution_status
            for branch_id, receipt in sorted(self.branch_receipts.items())
        }

    @property
    def evaluation_hashes(self) -> dict[str, str]:
        if self.run is None:
            return {}
        return {
            item.branch_id: str(item.evaluation_hash)
            for item in self.run.evaluations
        }

    @property
    def final_decision(self) -> str | None:
        return self.run.final_decision if self.run is not None else None

    @property
    def run_hash(self) -> str | None:
        return str(self.run.run_hash) if self.run is not None else None


class DevelopmentPortfolioTransactionV69:
    """Small authenticated transaction around the pure V6.9 runtime."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        authority_key: bytes,
        authority_key_id: str,
        _fault_hook: Callable[[str], None] | None = None,
    ) -> None:
        if len(authority_key) < 32:
            raise ValueError("V6.9 transaction authority key must contain 32 bytes")
        self._workspace_root = Path(workspace_root).resolve()
        self._authority_key = bytes(authority_key)
        self._authority_key_id = authority_key_id
        self._fault_hook = _fault_hook

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
                raise StageWorkspaceError(
                    "V6.9 workspace failed verification inside writer lock"
                )
            self._assert_workspace_boundary(workspace)
            yield workspace

    def _fault(self, phase: str) -> None:
        if self._fault_hook is not None:
            self._fault_hook(phase)

    @staticmethod
    def _assert_workspace_boundary(workspace: StageWorkspaceV50) -> None:
        if workspace.spec.evidence_scope != "development":
            raise PermissionError(
                "V6.9 portfolio transaction requires development evidence scope"
            )
        status = workspace.status()
        if any(
            workspace.current_gate(stage) is not None
            for stage in ("S1", "S2", "S3", "S4", "S5", "S6")
        ):
            raise PermissionError(
                "V6.9 portfolio transaction cannot run after S1 opens"
            )
        if workspace.current_gate("S0") is not None and (
            status.stage_statuses.get("S1") != "frontier"
        ):
            raise PermissionError(
                "V6.9 portfolio transaction requires S1 on the frontier"
            )

    def project(self) -> PortfolioTransactionStateV69:
        """Read and verify committed state without committing or repairing."""

        workspace = self._open()
        self._assert_workspace_boundary(workspace)
        return self._project_workspace(workspace)

    def freeze(
        self,
        *,
        query: CapabilityQueryV68,
        registry: CapabilityRegistryV68,
        branch_budget: BranchBudgetV68,
        portfolio_budget: PortfolioBudgetV68,
        time_unit: str,
        snapshot_task_id: str | None = None,
        initial_training_count: int = 34,
        maximum_origins: int | None = None,
        baseline_policy: PersistenceBaselinePolicyV69 | None = None,
        ode_thresholds: ODEThresholdsV52 | None = None,
        log_increment_thresholds: PositiveLogIncrementThresholdsV68 | None = None,
        tie_tolerance: float = 1e-12,
    ) -> PortfolioTransactionStateV69:
        """Commit the authenticated pre-observation freeze only."""

        policy = baseline_policy or PersistenceBaselinePolicyV69.seal()
        ode = ode_thresholds or ODEThresholdsV52.seal()
        log = (
            log_increment_thresholds
            or PositiveLogIncrementThresholdsV68.seal()
        )
        with self._locked_reopen() as workspace:
            before = self._project_workspace(workspace)
            current_s0 = workspace.current_gate("S0")
            if workspace.spec.spec_hash is None or current_s0 is None:
                raise StageWorkspaceError(
                    "V6.9 freeze requires a current open S0 gate"
                )
            if (
                query.workspace_spec_hash != workspace.spec.spec_hash
                or query.s0_gate_hash != current_s0
            ):
                raise ValueError("V6.9 query uses another workspace or S0 gate")
            if query.problem_signature.observation_count < 35:
                raise ValueError(
                    "V6.9 freeze requires at least 35 planned observations"
                )
            if (
                query.problem_signature.observation_count
                != query.measurement.minimum_planned_observations
                or time_unit != query.measurement.time_basis
            ):
                raise ValueError(
                    "V6.9 query count/state/time measurement bindings differ"
                )
            protocol = compile_default_development_portfolio_protocol_v69(
                query=query,
                registry=registry,
                branch_budget=branch_budget,
                portfolio_budget=portfolio_budget,
                tie_tolerance=tie_tolerance,
            )
            draft = PortfolioTransactionIntentV69(
                workspace_spec_hash=workspace.spec.spec_hash,
                s0_gate_hash=current_s0,
                protocol=protocol,
                planned_observation_count=(
                    query.problem_signature.observation_count
                ),
                state_unit=query.measurement.measurement_unit,
                time_unit=time_unit,
                snapshot_task_id=snapshot_task_id,
                initial_training_count=initial_training_count,
                maximum_origins=maximum_origins,
                baseline_policy=policy,
                ode_thresholds=ode,
                log_increment_thresholds=log,
                authority_key_id=workspace.authority_key_id,
            )
            intent = draft.authenticate(
                workspace._mac(
                    PORTFOLIO_TRANSACTION_INTENT_KIND_V69,
                    draft.unsigned_hash(),
                )
            )
            if before.intent is not None:
                if before.intent == intent:
                    return before
                raise StageWorkspaceError(
                    "a different current V6.9 freeze intent already exists"
                )
            self._commit_exact_or_replay(
                workspace,
                PORTFOLIO_TRANSACTION_INTENT_KIND_V69,
                intent,
            )
            return self._project_workspace(workspace)

    def stage_snapshot(
        self,
        snapshot: PositiveSeriesSnapshotV69,
    ) -> PortfolioTransactionStateV69:
        """Commit an authenticated snapshot/plan intent; execute no pack."""

        snapshot.assert_sealed()
        with self._locked_reopen() as workspace:
            state = self._project_workspace(workspace)
            if state.status == "STALE_PENDING":
                return state
            if state.intent is None or state.intent_ref is None:
                raise StageWorkspaceError("V6.9 portfolio is not frozen")
            if (
                snapshot.state_unit != state.intent.state_unit
                or snapshot.state_unit
                != state.intent.protocol.common_loss.loss_unit
                or snapshot.time_unit != state.intent.time_unit
                or len(snapshot.observations)
                != state.intent.planned_observation_count
                or (
                    state.intent.snapshot_task_id is not None
                    and snapshot.task_id != state.intent.snapshot_task_id
                )
            ):
                raise ValueError("V6.9 snapshot differs from frozen units/count")
            plan = build_common_rolling_origin_plan_v69(
                snapshot,
                initial_training_count=state.intent.initial_training_count,
                maximum_origins=state.intent.maximum_origins,
            )
            draft = PortfolioRunIntentV69(
                workspace_spec_hash=state.intent.workspace_spec_hash,
                s0_gate_hash=state.intent.s0_gate_hash,
                freeze_intent_hash=str(state.intent.intent_hash),
                freeze_artifact_hash=state.intent_ref.sha256,
                protocol_hash=str(state.intent.protocol.protocol_hash),
                snapshot=snapshot,
                origin_plan=plan,
                authority_key_id=workspace.authority_key_id,
            )
            run_intent = draft.authenticate(
                workspace._mac(
                    PORTFOLIO_RUN_INTENT_KIND_V69,
                    draft.unsigned_hash(),
                )
            )
            if state.run_intent is not None:
                if state.run_intent == run_intent:
                    return state
                raise StageWorkspaceError(
                    "a different V6.9 snapshot intent already exists"
                )
            self._commit_exact_or_replay(
                workspace,
                PORTFOLIO_RUN_INTENT_KIND_V69,
                run_intent,
            )
            result = self._project_workspace(workspace)
        self._fault("after_run_intent")
        return result

    def execute(
        self,
        snapshot: PositiveSeriesSnapshotV69 | None = None,
    ) -> DevelopmentPortfolioRunV69:
        """Execute only from a committed data intent, then commit outputs."""

        state = self.project()
        if snapshot is not None:
            snapshot.assert_sealed()
            if (
                state.run_intent is not None
                and snapshot != state.run_intent.snapshot
            ):
                raise ValueError("V6.9 supplied snapshot differs from run intent")
            if state.status == "FROZEN":
                state = self.stage_snapshot(snapshot)
        if state.status == "COMPLETED" and state.run is not None:
            return state.run
        if state.status not in {"DATA_STAGED", "RECOVERY_PENDING"}:
            raise StageWorkspaceError(
                "V6.9 execute requires a committed snapshot intent"
            )
        recovered = self.reconcile(snapshot)
        if recovered.status != "COMPLETED" or recovered.run is None:
            raise StageWorkspaceError("V6.9 transaction did not complete")
        return recovered.run

    def reconcile(
        self,
        snapshot: PositiveSeriesSnapshotV69 | None = None,
    ) -> PortfolioTransactionStateV69:
        """Resume an explicit staged/incomplete run; never start from FROZEN."""

        state = self.project()
        if state.status in {
            "NOT_STARTED",
            "FROZEN",
            "STALE_PENDING",
        }:
            if (
                snapshot is not None
                and state.run_intent is not None
                and snapshot != state.run_intent.snapshot
            ):
                raise ValueError("V6.9 supplied snapshot differs from run intent")
            return state
        if state.status == "COMPLETED":
            if (
                state.intent is None
                or state.run_intent is None
                or state.run is None
            ):
                raise StageWorkspaceError(
                    "V6.9 completed replay lacks authenticated inputs"
                )
            if (
                snapshot is not None
                and snapshot != state.run_intent.snapshot
            ):
                raise ValueError(
                    "V6.9 supplied snapshot differs from run intent"
                )
            expected = execute_development_portfolio_v69(
                protocol=state.intent.protocol,
                snapshot=state.run_intent.snapshot,
                origin_plan=state.run_intent.origin_plan,
                ode_thresholds=state.intent.ode_thresholds,
                log_increment_thresholds=(
                    state.intent.log_increment_thresholds
                ),
                baseline_policy=state.intent.baseline_policy,
            )
            if expected != state.run:
                raise StageWorkspaceError(
                    "V6.9 completed run differs from exact replay"
                )
            return state
        if state.intent is None or state.run_intent is None:
            raise StageWorkspaceError("V6.9 recovery lacks authenticated intents")
        if snapshot is not None and snapshot != state.run_intent.snapshot:
            raise ValueError("V6.9 supplied snapshot differs from run intent")
        expected = execute_development_portfolio_v69(
            protocol=state.intent.protocol,
            snapshot=state.run_intent.snapshot,
            origin_plan=state.run_intent.origin_plan,
            ode_thresholds=state.intent.ode_thresholds,
            log_increment_thresholds=state.intent.log_increment_thresholds,
            baseline_policy=state.intent.baseline_policy,
        )
        return self._commit_expected(expected)

    @staticmethod
    def _commit_exact_or_replay(
        workspace: StageWorkspaceV50,
        kind: str,
        model: StrictModel,
    ) -> ArtifactRef:
        payload = model.model_dump(mode="json")
        exact = [
            reference
            for reference, existing in workspace._artifacts_of_kind(kind)
            if existing == payload
        ]
        if len(exact) > 1:
            raise StageWorkspaceError(
                f"multiple identical committed V6.9 artifacts: {kind}"
            )
        if exact:
            return exact[0]
        return workspace.commit_evidence(kind, payload)

    @staticmethod
    def _verify_authenticated_model(
        workspace: StageWorkspaceV50,
        *,
        kind: str,
        model: PortfolioTransactionIntentV69
        | PortfolioRunIntentV69
        | PortfolioTransactionCompletionV69,
    ) -> None:
        model.assert_sealed()
        if (
            model.authority_key_id != workspace.authority_key_id
            or not workspace._verify_mac(
                kind,
                model.unsigned_hash(),
                model.authority_auth_tag,
            )
        ):
            raise StageWorkspaceError(
                f"V6.9 authenticated artifact failed authority check: {kind}"
            )

    def _project_workspace(
        self,
        workspace: StageWorkspaceV50,
    ) -> PortfolioTransactionStateV69:
        """Pure projection over committed artifacts."""

        freeze_records = workspace._artifacts_of_kind(
            PORTFOLIO_TRANSACTION_INTENT_KIND_V69,
            PortfolioTransactionIntentV69,
        )
        for _, intent in freeze_records:
            self._verify_authenticated_model(
                workspace,
                kind=PORTFOLIO_TRANSACTION_INTENT_KIND_V69,
                model=intent,
            )
            if intent.workspace_spec_hash != workspace.spec.spec_hash:
                raise StageWorkspaceError(
                    "V6.9 freeze intent uses another workspace"
                )
        current_s0 = workspace.current_gate("S0")
        current_freezes = [
            item
            for item in freeze_records
            if item[1].s0_gate_hash == current_s0
        ]
        if len(current_freezes) > 1:
            raise StageWorkspaceError(
                "multiple current V6.9 freeze intents are committed"
            )
        if not current_freezes:
            if freeze_records:
                stale_ref, stale_intent = freeze_records[-1]
                return PortfolioTransactionStateV69(
                    status="STALE_PENDING",
                    intent_ref=stale_ref,
                    intent=stale_intent,
                )
            return PortfolioTransactionStateV69(status="NOT_STARTED")
        intent_ref, intent = current_freezes[0]

        run_intent_records = workspace._artifacts_of_kind(
            PORTFOLIO_RUN_INTENT_KIND_V69,
            PortfolioRunIntentV69,
        )
        matching_run_intents: list[tuple[ArtifactRef, PortfolioRunIntentV69]] = []
        for reference, run_intent in run_intent_records:
            self._verify_authenticated_model(
                workspace,
                kind=PORTFOLIO_RUN_INTENT_KIND_V69,
                model=run_intent,
            )
            if (
                run_intent.freeze_intent_hash == intent.intent_hash
                and run_intent.freeze_artifact_hash == intent_ref.sha256
            ):
                if (
                    run_intent.workspace_spec_hash
                    != intent.workspace_spec_hash
                    or run_intent.s0_gate_hash != intent.s0_gate_hash
                    or run_intent.protocol_hash != intent.protocol.protocol_hash
                ):
                    raise StageWorkspaceError(
                        "V6.9 run intent freeze binding differs"
                    )
                expected_plan = build_common_rolling_origin_plan_v69(
                    run_intent.snapshot,
                    initial_training_count=intent.initial_training_count,
                    maximum_origins=intent.maximum_origins,
                )
                if run_intent.origin_plan != expected_plan:
                    raise StageWorkspaceError(
                        "V6.9 run intent rolling plan differs"
                    )
                matching_run_intents.append((reference, run_intent))
        if len(matching_run_intents) > 1:
            raise StageWorkspaceError(
                "multiple current V6.9 run intents are committed"
            )
        if not matching_run_intents:
            return PortfolioTransactionStateV69(
                status="FROZEN",
                intent_ref=intent_ref,
                intent=intent,
            )
        run_intent_ref, run_intent = matching_run_intents[0]

        expected_branch_ids = {
            item.branch_id for item in intent.protocol.branches
        }
        branch_refs: dict[str, ArtifactRef] = {}
        branch_receipts: dict[str, BranchExecutionReceiptV69] = {}
        for reference, receipt in workspace._artifacts_of_kind(
            PORTFOLIO_BRANCH_RECEIPT_KIND_V69,
            BranchExecutionReceiptV69,
        ):
            receipt.assert_sealed()
            if (
                receipt.protocol_hash != intent.protocol.protocol_hash
                or receipt.snapshot_hash != run_intent.snapshot.snapshot_hash
                or receipt.origin_plan_hash != run_intent.origin_plan.plan_hash
            ):
                continue
            if receipt.branch_id not in expected_branch_ids:
                raise StageWorkspaceError(
                    "V6.9 branch receipt names an unknown branch"
                )
            if receipt.branch_id in branch_receipts:
                raise StageWorkspaceError(
                    "multiple V6.9 receipts exist for one branch"
                )
            branch_refs[receipt.branch_id] = reference
            branch_receipts[receipt.branch_id] = receipt

        matching_runs: list[tuple[ArtifactRef, DevelopmentPortfolioRunV69]] = []
        for reference, run in workspace._artifacts_of_kind(
            PORTFOLIO_RUN_KIND_V69,
            DevelopmentPortfolioRunV69,
        ):
            run.assert_sealed()
            if (
                run.protocol_hash == intent.protocol.protocol_hash
                and run.snapshot_hash == run_intent.snapshot.snapshot_hash
                and run.origin_plan_hash == run_intent.origin_plan.plan_hash
            ):
                matching_runs.append((reference, run))
        if len(matching_runs) > 1:
            raise StageWorkspaceError("multiple current V6.9 runs are committed")
        run_ref: ArtifactRef | None = None
        run: DevelopmentPortfolioRunV69 | None = None
        if matching_runs:
            run_ref, run = matching_runs[0]
            expected_receipts = {
                item.branch_id: item for item in run.branch_receipts
            }
            if set(branch_receipts) != expected_branch_ids:
                raise StageWorkspaceError(
                    "V6.9 run exists without both committed branch receipts"
                )
            if branch_receipts != expected_receipts:
                raise StageWorkspaceError(
                    "V6.9 committed branch receipts differ from the run"
                )

        completion_records = workspace._artifacts_of_kind(
            PORTFOLIO_COMPLETION_KIND_V69,
            PortfolioTransactionCompletionV69,
        )
        matching_completions: list[
            tuple[ArtifactRef, PortfolioTransactionCompletionV69]
        ] = []
        for reference, completion in completion_records:
            self._verify_authenticated_model(
                workspace,
                kind=PORTFOLIO_COMPLETION_KIND_V69,
                model=completion,
            )
            if (
                completion.freeze_artifact_hash == intent_ref.sha256
                and completion.run_intent_artifact_hash
                == run_intent_ref.sha256
            ):
                matching_completions.append((reference, completion))
        if len(matching_completions) > 1:
            raise StageWorkspaceError(
                "multiple current V6.9 completions are committed"
            )
        completion_ref: ArtifactRef | None = None
        completion: PortfolioTransactionCompletionV69 | None = None
        if matching_completions:
            if run is None or run_ref is None:
                raise StageWorkspaceError(
                    "V6.9 completion exists without a committed run"
                )
            completion_ref, completion = matching_completions[0]
            if (
                completion.workspace_spec_hash != intent.workspace_spec_hash
                or completion.s0_gate_hash != intent.s0_gate_hash
                or completion.freeze_intent_hash != intent.intent_hash
                or completion.run_intent_hash != run_intent.run_intent_hash
                or completion.run_artifact_hash != run_ref.sha256
                or completion.run_hash != run.run_hash
                or completion.branch_artifact_hashes
                != {
                    key: branch_refs[key].sha256
                    for key in sorted(branch_refs)
                }
                or completion.branch_receipt_hashes
                != {
                    key: str(branch_receipts[key].receipt_hash)
                    for key in sorted(branch_receipts)
                }
            ):
                raise StageWorkspaceError(
                    "V6.9 completion output binding differs"
                )
            status: PortfolioTransactionStatusV69 = "COMPLETED"
        elif branch_receipts or run is not None:
            status = "RECOVERY_PENDING"
        else:
            status = "DATA_STAGED"
        return PortfolioTransactionStateV69(
            status=status,
            intent_ref=intent_ref,
            intent=intent,
            run_intent_ref=run_intent_ref,
            run_intent=run_intent,
            branch_refs=dict(sorted(branch_refs.items())),
            branch_receipts=dict(sorted(branch_receipts.items())),
            run_ref=run_ref,
            run=run,
            completion_ref=completion_ref,
            completion=completion,
        )

    def _commit_expected(
        self,
        expected: DevelopmentPortfolioRunV69,
    ) -> PortfolioTransactionStateV69:
        with self._locked_reopen() as workspace:
            state = self._project_workspace(workspace)
            if state.status == "STALE_PENDING":
                return state
            if (
                state.intent is None
                or state.intent_ref is None
                or state.run_intent is None
                or state.run_intent_ref is None
            ):
                raise StageWorkspaceError(
                    "V6.9 output commit lacks authenticated intents"
                )
            if (
                expected.protocol_hash != state.intent.protocol.protocol_hash
                or expected.snapshot_hash
                != state.run_intent.snapshot.snapshot_hash
                or expected.origin_plan_hash
                != state.run_intent.origin_plan.plan_hash
            ):
                raise StageWorkspaceError("V6.9 recomputed run binding differs")
            expected_receipts = {
                item.branch_id: item for item in expected.branch_receipts
            }
            for branch_id, existing in state.branch_receipts.items():
                if existing != expected_receipts[branch_id]:
                    raise StageWorkspaceError(
                        "V6.9 existing branch receipt differs from replay"
                    )
            branch_refs = dict(state.branch_refs)
            for index, branch_id in enumerate(sorted(expected_receipts), start=1):
                if branch_id not in branch_refs:
                    branch_refs[branch_id] = self._commit_exact_or_replay(
                        workspace,
                        PORTFOLIO_BRANCH_RECEIPT_KIND_V69,
                        expected_receipts[branch_id],
                    )
                    self._fault(f"after_branch_{index}")
            if state.run is not None and state.run != expected:
                raise StageWorkspaceError(
                    "V6.9 existing run differs from deterministic replay"
                )
            run_ref = state.run_ref
            if run_ref is None:
                run_ref = self._commit_exact_or_replay(
                    workspace,
                    PORTFOLIO_RUN_KIND_V69,
                    expected,
                )
                self._fault("after_run")
            if state.completion is None:
                completion_draft = PortfolioTransactionCompletionV69(
                    workspace_spec_hash=state.intent.workspace_spec_hash,
                    s0_gate_hash=state.intent.s0_gate_hash,
                    freeze_intent_hash=str(state.intent.intent_hash),
                    freeze_artifact_hash=state.intent_ref.sha256,
                    run_intent_hash=str(state.run_intent.run_intent_hash),
                    run_intent_artifact_hash=state.run_intent_ref.sha256,
                    branch_artifact_hashes={
                        key: branch_refs[key].sha256
                        for key in sorted(branch_refs)
                    },
                    branch_receipt_hashes={
                        key: str(expected_receipts[key].receipt_hash)
                        for key in sorted(expected_receipts)
                    },
                    run_artifact_hash=run_ref.sha256,
                    run_hash=str(expected.run_hash),
                    completed_at=datetime.now(timezone.utc),
                    authority_key_id=workspace.authority_key_id,
                )
                completion = completion_draft.authenticate(
                    workspace._mac(
                        PORTFOLIO_COMPLETION_KIND_V69,
                        completion_draft.unsigned_hash(),
                    )
                )
                self._commit_exact_or_replay(
                    workspace,
                    PORTFOLIO_COMPLETION_KIND_V69,
                    completion,
                )
                self._fault("after_completion")
            return self._project_workspace(workspace)


__all__ = [
    "PORTFOLIO_BRANCH_RECEIPT_KIND_V69",
    "PORTFOLIO_COMPLETION_KIND_V69",
    "PORTFOLIO_RUN_INTENT_KIND_V69",
    "PORTFOLIO_RUN_KIND_V69",
    "PORTFOLIO_TRANSACTION_INTENT_KIND_V69",
    "DevelopmentPortfolioTransactionV69",
    "PortfolioRunIntentV69",
    "PortfolioTransactionCompletionV69",
    "PortfolioTransactionIntentV69",
    "PortfolioTransactionStateV69",
    "PortfolioTransactionStatusV69",
]
