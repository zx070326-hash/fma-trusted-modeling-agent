from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Annotated, Callable, Concatenate, Literal, ParamSpec, TypeVar

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore
from fma.v2.schemas import Identifier, Sha256, _assert_timezone


LoopLayer = Literal["modeling", "development"]
Actor = Literal["model", "harness", "verifier", "human"]
NodeKind = Literal[
    # Mathematical-modeling product graph.
    "mission",
    "problem_contract",
    "workflow_plan",
    "model_space",
    "model_proposal",
    "model_validation",
    "model_admission",
    "generation_call",
    "model_candidate",
    "evolution_operator",
    "experiment",
    "execution",
    "evaluation",
    "decision",
    "experience",
    "failure",
    "checkpoint",
    "incident",
    "recovery_patch",
    "runtime_release",
    # Agent-development graph.
    "objective",
    "issue",
    "design",
    "patch",
    "test",
    "review",
    "release",
    "retrospective",
]
EdgeRelation = Literal[
    "requires_success",
    "requires_active",
    "requires_terminal",
    "derived_from",
    "evaluated_by",
    "learned_from_failure",
    "supersedes",
    "invalidates",
]
OutcomeStatus = Literal["succeeded", "failed", "blocked"]
EffectiveStatus = Literal[
    "pending",
    "succeeded",
    "failed",
    "blocked",
    "qualified",
    "active",
    "rejected",
    "revoked",
]
PromotionDecision = Literal["qualified", "active", "rejected"]


MODELING_KINDS = frozenset(
    {
        "mission",
        "problem_contract",
        "workflow_plan",
        "model_space",
        "model_proposal",
        "model_validation",
        "model_admission",
        "generation_call",
        "model_candidate",
        "evolution_operator",
        "experiment",
        "execution",
        "evaluation",
        "decision",
        "experience",
        "failure",
        "checkpoint",
        "incident",
        "recovery_patch",
        "runtime_release",
    }
)
DEVELOPMENT_KINDS = frozenset(
    {
        "objective",
        "issue",
        "design",
        "patch",
        "test",
        "review",
        "release",
        "retrospective",
    }
)
ALLOWED_EXECUTORS: dict[str, frozenset[str]] = {
    "mission": frozenset({"harness", "human"}),
    "problem_contract": frozenset({"harness", "human"}),
    "workflow_plan": frozenset({"model", "harness"}),
    "model_space": frozenset({"harness", "human"}),
    "model_proposal": frozenset({"model", "harness"}),
    "model_validation": frozenset({"verifier"}),
    "model_admission": frozenset({"verifier"}),
    "generation_call": frozenset({"harness"}),
    "model_candidate": frozenset({"model"}),
    "evolution_operator": frozenset({"model", "harness"}),
    "experiment": frozenset({"model", "harness"}),
    "execution": frozenset({"harness"}),
    "evaluation": frozenset({"verifier"}),
    "decision": frozenset({"verifier", "human"}),
    "experience": frozenset({"harness"}),
    "failure": frozenset({"harness", "verifier"}),
    "checkpoint": frozenset({"harness"}),
    "incident": frozenset({"harness"}),
    "recovery_patch": frozenset({"harness"}),
    "runtime_release": frozenset({"harness"}),
    "objective": frozenset({"human"}),
    "issue": frozenset({"harness", "human"}),
    "design": frozenset({"model", "human"}),
    "patch": frozenset({"model", "human"}),
    "test": frozenset({"harness"}),
    "review": frozenset({"verifier"}),
    "release": frozenset({"human"}),
    "retrospective": frozenset({"harness", "verifier", "human"}),
}
SUCCESS_STATUSES = frozenset({"succeeded", "qualified", "active"})
TERMINAL_STATUSES = frozenset(
    {"succeeded", "failed", "blocked", "qualified", "active", "rejected"}
)
PROPAGATING_RELATIONS = frozenset(
    {
        "requires_success",
        "requires_active",
        "requires_terminal",
        "derived_from",
        "evaluated_by",
        "learned_from_failure",
    }
)

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _single_writer_mutation(
    method: Callable[Concatenate["GraphLoopStoreV40", _P], _R],
) -> Callable[Concatenate["GraphLoopStoreV40", _P], _R]:
    @wraps(method)
    def serialized(
        self: "GraphLoopStoreV40", *args: _P.args, **kwargs: _P.kwargs
    ) -> _R:
        with self.store.writer_transaction():
            return method(self, *args, **kwargs)

    return serialized


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class GraphLoopContractV40(StrictModel):
    """Frozen authority and budget envelope for one graph-native loop."""

    schema_version: Literal["4.0"] = "4.0"
    graph_id: Identifier
    layer: LoopLayer
    evaluator_epoch: Identifier
    objective: Annotated[str, Field(min_length=5)]
    max_nodes: Annotated[int, Field(ge=1)] = 128
    max_outcomes: Annotated[int, Field(ge=1)] = 64
    max_failures: Annotated[int, Field(ge=1)] = 16
    max_promotions: Annotated[int, Field(ge=1)] = 8
    allowed_actions: list[Annotated[str, Field(min_length=1)]] = Field(
        default_factory=lambda: ["local_compute", "write_local_run_artifacts"]
    )
    forbidden_actions: list[Annotated[str, Field(min_length=1)]] = Field(
        default_factory=lambda: ["external_action"]
    )
    created_at: datetime
    contract_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "GraphLoopContractV40":
        _assert_timezone(self.created_at, "created_at")
        if set(self.allowed_actions) & set(self.forbidden_actions):
            raise ValueError("allowed_actions and forbidden_actions overlap")
        if "external_action" not in self.forbidden_actions:
            raise ValueError("V4.0 loops must forbid external_action")
        if self.contract_hash and self.contract_hash != self.content_hash():
            raise ValueError("contract_hash does not match graph-loop contract")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "contract_hash")

    def assert_sealed(self) -> None:
        if not self.contract_hash or self.contract_hash != self.content_hash():
            raise ValueError("graph-loop contract is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "GraphLoopContractV40":
        data.setdefault("created_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"contract_hash"}),
            contract_hash=draft.content_hash(),
        )


class GraphNodeV40(StrictModel):
    """Immutable work instance; logical retries become new nodes, never cycles."""

    schema_version: Literal["4.0"] = "4.0"
    node_id: Identifier
    layer: LoopLayer
    node_kind: NodeKind
    executor: Actor
    created_by: Actor
    artifact_hash: Sha256
    purpose: Annotated[str, Field(min_length=3)]
    created_at: datetime
    node_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_node(self) -> "GraphNodeV40":
        _assert_timezone(self.created_at, "created_at")
        allowed_kinds = MODELING_KINDS if self.layer == "modeling" else DEVELOPMENT_KINDS
        if self.node_kind not in allowed_kinds:
            raise ValueError(f"{self.node_kind} is not valid in the {self.layer} layer")
        if self.executor not in ALLOWED_EXECUTORS[self.node_kind]:
            raise ValueError(
                f"{self.executor} cannot execute node kind {self.node_kind}"
            )
        if self.created_by == "model" and self.node_kind in {
            "model_validation",
            "model_admission",
            "generation_call",
            "evaluation",
            "decision",
            "checkpoint",
            "incident",
            "recovery_patch",
            "test",
            "review",
            "release",
        }:
            raise ValueError("model cannot create verifier, decision, or release nodes")
        if self.node_hash and self.node_hash != self.content_hash():
            raise ValueError("node_hash does not match graph node")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "node_hash")

    def assert_sealed(self) -> None:
        if not self.node_hash or self.node_hash != self.content_hash():
            raise ValueError("graph node is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "GraphNodeV40":
        data.setdefault("created_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"node_hash"}),
            node_hash=draft.content_hash(),
        )


class GraphEdgeV40(StrictModel):
    schema_version: Literal["4.0"] = "4.0"
    edge_id: Identifier
    layer: LoopLayer
    source_node_hash: Sha256
    target_node_hash: Sha256
    relation: EdgeRelation
    rationale: Annotated[str, Field(min_length=3)]
    created_at: datetime
    edge_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_edge(self) -> "GraphEdgeV40":
        _assert_timezone(self.created_at, "created_at")
        if self.source_node_hash == self.target_node_hash:
            raise ValueError("graph self-edges are forbidden")
        if self.edge_hash and self.edge_hash != self.content_hash():
            raise ValueError("edge_hash does not match graph edge")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "edge_hash")

    def assert_sealed(self) -> None:
        if not self.edge_hash or self.edge_hash != self.content_hash():
            raise ValueError("graph edge is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "GraphEdgeV40":
        data.setdefault("created_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"edge_hash"}),
            edge_hash=draft.content_hash(),
        )


class NodeOutcomeV40(StrictModel):
    schema_version: Literal["4.0"] = "4.0"
    outcome_id: Identifier
    graph_id: Identifier
    layer: LoopLayer
    node_hash: Sha256
    base_snapshot_hash: Sha256
    actor: Actor
    status: OutcomeStatus
    output_artifacts: Annotated[list[ArtifactRef], Field(min_length=1)]
    summary: Annotated[str, Field(min_length=3)]
    started_at: datetime
    finished_at: datetime
    outcome_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> "NodeOutcomeV40":
        _assert_timezone(self.started_at, "started_at")
        _assert_timezone(self.finished_at, "finished_at")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at precedes started_at")
        hashes = [ref.sha256 for ref in self.output_artifacts]
        if hashes != sorted(set(hashes)):
            raise ValueError("output_artifacts must be sorted by unique sha256")
        if self.outcome_hash and self.outcome_hash != self.content_hash():
            raise ValueError("outcome_hash does not match node outcome")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "outcome_hash")

    def assert_sealed(self) -> None:
        if not self.outcome_hash or self.outcome_hash != self.content_hash():
            raise ValueError("node outcome is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "NodeOutcomeV40":
        now = datetime.now(timezone.utc)
        data.setdefault("started_at", now)
        data.setdefault("finished_at", now)
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"outcome_hash"}),
            outcome_hash=draft.content_hash(),
        )


class PromotionReceiptV40(StrictModel):
    """Atomic, evaluator-bound status transition; never authored by the model."""

    schema_version: Literal["4.0"] = "4.0"
    promotion_id: Identifier
    graph_id: Identifier
    layer: LoopLayer
    candidate_node_hash: Sha256
    evaluator_node_hash: Sha256
    evidence_node_hashes: Annotated[list[Sha256], Field(min_length=1)]
    base_snapshot_hash: Sha256
    decision: PromotionDecision
    authority: Literal["verifier", "human"]
    evaluator_epoch: Identifier
    independent_gate_passed: bool
    private_scientific_gate_passed: bool
    scope: Annotated[str, Field(min_length=3)]
    decided_at: datetime
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "PromotionReceiptV40":
        _assert_timezone(self.decided_at, "decided_at")
        if self.evidence_node_hashes != sorted(set(self.evidence_node_hashes)):
            raise ValueError("evidence_node_hashes must be sorted and unique")
        if self.evaluator_node_hash not in self.evidence_node_hashes:
            raise ValueError("evaluator node must be included in promotion evidence")
        if self.decision == "qualified" and self.authority != "verifier":
            raise ValueError("only verifier may qualify a candidate")
        if self.decision == "active" and self.authority != "human":
            raise ValueError("only human may activate a qualified candidate")
        if self.decision in {"qualified", "active"} and not self.independent_gate_passed:
            raise ValueError("promotion requires an independent gate")
        if (
            self.layer == "modeling"
            and self.decision in {"qualified", "active"}
            and not self.private_scientific_gate_passed
        ):
            raise ValueError("modeling promotion requires the private scientific gate")
        if self.layer == "development" and self.private_scientific_gate_passed:
            raise ValueError("development promotion cannot claim a scientific gate")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match promotion receipt")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("promotion receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PromotionReceiptV40":
        data.setdefault("decided_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


class RevocationReceiptV40(StrictModel):
    schema_version: Literal["4.0"] = "4.0"
    revocation_id: Identifier
    graph_id: Identifier
    layer: LoopLayer
    root_node_hash: Sha256
    affected_node_hashes: Annotated[list[Sha256], Field(min_length=1)]
    base_snapshot_hash: Sha256
    authority: Literal["verifier", "human"]
    reason: Annotated[str, Field(min_length=3)]
    revoked_at: datetime
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "RevocationReceiptV40":
        _assert_timezone(self.revoked_at, "revoked_at")
        if self.affected_node_hashes != sorted(set(self.affected_node_hashes)):
            raise ValueError("affected_node_hashes must be sorted and unique")
        if self.root_node_hash not in self.affected_node_hashes:
            raise ValueError("revocation root must be affected")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match revocation receipt")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("revocation receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "RevocationReceiptV40":
        data.setdefault("revoked_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


class GraphLoopSnapshotV40(StrictModel):
    schema_version: Literal["4.0"] = "4.0"
    graph_id: Identifier
    layer: LoopLayer
    graph_event_count: Annotated[int, Field(ge=1)]
    last_graph_event_hash: Sha256
    node_statuses: dict[Sha256, EffectiveStatus]
    frontier_node_hashes: list[Sha256]
    outcome_count: Annotated[int, Field(ge=0)]
    failure_count: Annotated[int, Field(ge=0)]
    promotion_count: Annotated[int, Field(ge=0)]
    budget_exhausted: bool
    stop_reason: str | None = None
    snapshot_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_snapshot(self) -> "GraphLoopSnapshotV40":
        if self.frontier_node_hashes != sorted(set(self.frontier_node_hashes)):
            raise ValueError("frontier_node_hashes must be sorted and unique")
        if self.snapshot_hash and self.snapshot_hash != self.content_hash():
            raise ValueError("snapshot_hash does not match graph-loop snapshot")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "snapshot_hash")

    def assert_sealed(self) -> None:
        if not self.snapshot_hash or self.snapshot_hash != self.content_hash():
            raise ValueError("graph-loop snapshot is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "GraphLoopSnapshotV40":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"snapshot_hash"}),
            snapshot_hash=draft.content_hash(),
        )


class GraphLoopStateV40(StrictModel):
    contract: GraphLoopContractV40
    nodes: list[GraphNodeV40]
    edges: list[GraphEdgeV40]
    outcomes: list[NodeOutcomeV40]
    promotions: list[PromotionReceiptV40]
    revocations: list[RevocationReceiptV40]
    snapshot: GraphLoopSnapshotV40

    def node_by_hash(self, node_hash: str) -> GraphNodeV40:
        for node in self.nodes:
            if node.node_hash == node_hash:
                return node
        raise KeyError(node_hash)


class CrossLayerBridgeV40(StrictModel):
    """One-way receipt; software release never becomes scientific evidence."""

    schema_version: Literal["4.0"] = "4.0"
    bridge_id: Identifier
    direction: Literal[
        "development_release_to_modeling_runtime",
        "modeling_failure_to_development_issue",
    ]
    source_graph_id: Identifier
    source_snapshot_hash: Sha256
    source_node_hash: Sha256
    target_graph_id: Identifier
    sanitized_payload_hash: Sha256
    approved_by: Literal["harness", "human"]
    scientific_validity_granted: Literal[False] = False
    private_acceptance_data_exposed: Literal[False] = False
    created_at: datetime
    bridge_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bridge(self) -> "CrossLayerBridgeV40":
        _assert_timezone(self.created_at, "created_at")
        if (
            self.direction == "development_release_to_modeling_runtime"
            and self.approved_by != "human"
        ):
            raise ValueError("runtime release bridge requires human approval")
        if self.bridge_hash and self.bridge_hash != self.content_hash():
            raise ValueError("bridge_hash does not match cross-layer bridge")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "bridge_hash")

    def assert_sealed(self) -> None:
        if not self.bridge_hash or self.bridge_hash != self.content_hash():
            raise ValueError("cross-layer bridge is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "CrossLayerBridgeV40":
        data.setdefault("created_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"bridge_hash"}),
            bridge_hash=draft.content_hash(),
        )


class GraphLoopStoreV40:
    """Event-sourced graph scheduler with code-owned authority transitions."""

    def __init__(self, output_root: str | Path, contract: GraphLoopContractV40) -> None:
        contract.assert_sealed()
        self.contract = contract
        self.store = RunStore(output_root, run_id=contract.graph_id)
        ref = self.store.put_artifact("graph_loop_contract_v40", contract)
        self.store.emit("graph_loop_started_v40", ref.model_dump(mode="json"))
        self.project_state()

    @classmethod
    def open_existing(cls, graph_directory: str | Path) -> "GraphLoopStoreV40":
        store = RunStore.open_existing(graph_directory)
        with store.writer_transaction():
            events = cls._read_events(store)
            committed = cls._committed_artifacts(store, events)
            starts = [
                event
                for event in events
                if event["event_type"] == "graph_loop_started_v40"
            ]
            if len(starts) != 1:
                raise RuntimeError("graph loop needs exactly one start event")
            ref = cls._committed_ref(
                starts[0], committed, "graph_loop_contract_v40"
            )
            contract = GraphLoopContractV40.model_validate(
                store.load_artifact(ref)
            )
            contract.assert_sealed()
            instance = cls.__new__(cls)
            instance.contract = contract
            instance.store = store
            instance.project_state()
        return instance

    @property
    def run_directory(self) -> Path:
        return self.store.run_directory

    def put_output(self, kind: str, payload: object) -> ArtifactRef:
        return self.store.put_artifact(kind, payload)

    @_single_writer_mutation
    def add_node(self, node: GraphNodeV40) -> ArtifactRef:
        node.assert_sealed()
        state = self.project_state()
        if node.layer != self.contract.layer:
            raise ValueError("node belongs to another loop layer")
        if len(state.nodes) >= self.contract.max_nodes:
            raise RuntimeError("graph node budget is exhausted")
        if any(item.node_hash == node.node_hash for item in state.nodes):
            raise ValueError("graph node already exists")
        if any(item.node_id == node.node_id for item in state.nodes):
            raise ValueError("graph node_id already exists")
        ref = self.store.put_artifact("graph_node_v40", node)
        self.store.emit("graph_node_added_v40", ref.model_dump(mode="json"))
        self.project_state()
        return ref

    @_single_writer_mutation
    def add_edge(self, edge: GraphEdgeV40) -> ArtifactRef:
        edge.assert_sealed()
        state = self.project_state()
        if edge.layer != self.contract.layer:
            raise ValueError("edge belongs to another loop layer")
        node_hashes = {node.node_hash for node in state.nodes}
        if edge.source_node_hash not in node_hashes or edge.target_node_hash not in node_hashes:
            raise ValueError("graph edge references an unknown node")
        if any(item.edge_hash == edge.edge_hash for item in state.edges):
            raise ValueError("graph edge already exists")
        if any(item.edge_id == edge.edge_id for item in state.edges):
            raise ValueError("graph edge_id already exists")
        if self._has_path(edge.target_node_hash, edge.source_node_hash, state.edges):
            raise ValueError("graph edge would create a cycle")
        ref = self.store.put_artifact("graph_edge_v40", edge)
        self.store.emit("graph_edge_added_v40", ref.model_dump(mode="json"))
        self.project_state()
        return ref

    @_single_writer_mutation
    def record_outcome(
        self,
        node_hash: str,
        *,
        actor: Actor,
        status: OutcomeStatus,
        output_artifacts: list[ArtifactRef],
        summary: str,
        outcome_id: str,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        bridge_source: "GraphLoopStoreV40 | None" = None,
        bridge_reconciliation_ref: ArtifactRef | None = None,
    ) -> NodeOutcomeV40:
        state = self.project_state()
        node = state.node_by_hash(node_hash)
        if actor != node.executor:
            raise PermissionError("outcome actor does not own this graph node")
        if node.node_kind == "runtime_release":
            if bridge_source is None or bridge_reconciliation_ref is None:
                raise RuntimeError(
                    "runtime release requires a current bridge reconciliation"
                )
            from .bridge_reconciliation import (
                assert_current_bridge_reconciliation_v40,
            )

            assert_current_bridge_reconciliation_v40(
                bridge_source,
                self,
                node_hash,
                bridge_reconciliation_ref,
            )
        if node_hash not in state.snapshot.frontier_node_hashes:
            raise RuntimeError("node is not on the executable frontier")
        if state.snapshot.budget_exhausted:
            raise RuntimeError("graph-loop budget is exhausted")
        self._assert_output_refs(output_artifacts)
        now = datetime.now(timezone.utc)
        outcome = NodeOutcomeV40.seal(
            outcome_id=outcome_id,
            graph_id=self.contract.graph_id,
            layer=self.contract.layer,
            node_hash=node_hash,
            base_snapshot_hash=state.snapshot.snapshot_hash,
            actor=actor,
            status=status,
            output_artifacts=sorted(output_artifacts, key=lambda ref: ref.sha256),
            summary=summary,
            started_at=started_at or now,
            finished_at=finished_at or now,
        )
        ref = self.store.put_artifact("node_outcome_v40", outcome)
        self.store.emit("node_outcome_recorded_v40", ref.model_dump(mode="json"))
        self.project_state()
        return outcome

    @_single_writer_mutation
    def decide_promotion(
        self,
        candidate_node_hash: str,
        evaluator_node_hash: str,
        *,
        evidence_node_hashes: list[str],
        decision: PromotionDecision,
        authority: Literal["verifier", "human"],
        independent_gate_passed: bool,
        private_scientific_gate_passed: bool,
        scope: str,
        promotion_id: str,
        decided_at: datetime | None = None,
    ) -> PromotionReceiptV40:
        state = self.project_state()
        if state.snapshot.promotion_count >= self.contract.max_promotions:
            raise RuntimeError("promotion budget is exhausted")
        receipt = PromotionReceiptV40.seal(
            promotion_id=promotion_id,
            graph_id=self.contract.graph_id,
            layer=self.contract.layer,
            candidate_node_hash=candidate_node_hash,
            evaluator_node_hash=evaluator_node_hash,
            evidence_node_hashes=sorted(set(evidence_node_hashes)),
            base_snapshot_hash=state.snapshot.snapshot_hash,
            decision=decision,
            authority=authority,
            evaluator_epoch=self.contract.evaluator_epoch,
            independent_gate_passed=independent_gate_passed,
            private_scientific_gate_passed=private_scientific_gate_passed,
            scope=scope,
            decided_at=decided_at or datetime.now(timezone.utc),
        )
        self._validate_promotion(receipt, state)
        ref = self.store.put_artifact("promotion_receipt_v40", receipt)
        self.store.emit("promotion_decided_v40", ref.model_dump(mode="json"))
        self.project_state()
        return receipt

    @_single_writer_mutation
    def revoke_node(
        self,
        root_node_hash: str,
        *,
        authority: Literal["verifier", "human"],
        reason: str,
        revocation_id: str,
        revoked_at: datetime | None = None,
    ) -> RevocationReceiptV40:
        state = self.project_state()
        if root_node_hash not in state.snapshot.node_statuses:
            raise ValueError("cannot revoke an unknown graph node")
        if state.snapshot.node_statuses[root_node_hash] == "revoked":
            raise ValueError("graph node is already revoked")
        affected = sorted(self._revocation_closure(root_node_hash, state.edges))
        receipt = RevocationReceiptV40.seal(
            revocation_id=revocation_id,
            graph_id=self.contract.graph_id,
            layer=self.contract.layer,
            root_node_hash=root_node_hash,
            affected_node_hashes=affected,
            base_snapshot_hash=state.snapshot.snapshot_hash,
            authority=authority,
            reason=reason,
            revoked_at=revoked_at or datetime.now(timezone.utc),
        )
        ref = self.store.put_artifact("revocation_receipt_v40", receipt)
        self.store.emit("nodes_revoked_v40", ref.model_dump(mode="json"))
        self.project_state()
        return receipt

    def verify(self) -> bool:
        try:
            self.project_state()
        except (
            OSError,
            RuntimeError,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ):
            return False
        return True

    @_single_writer_mutation
    def project_state(self) -> GraphLoopStateV40:
        event_stat = self.store.event_path.stat()
        artifact_stats = tuple(
            (path.name, item.st_size, item.st_mtime_ns)
            for path in sorted(self.store.artifact_directory.glob("*.json"))
            for item in (path.stat(),)
        )
        revision = (
            event_stat.st_size,
            event_stat.st_mtime_ns,
            artifact_stats,
        )
        if (
            getattr(self, "_projected_state_revision", None) == revision
            and getattr(self, "_projected_state_cache", None) is not None
        ):
            return self._projected_state_cache
        events = self._read_events(self.store)
        committed = self._committed_artifacts(self.store, events)
        nodes: list[GraphNodeV40] = []
        edges: list[GraphEdgeV40] = []
        outcomes: list[NodeOutcomeV40] = []
        promotions: list[PromotionReceiptV40] = []
        revocations: list[RevocationReceiptV40] = []
        revoked: set[str] = set()
        started = False
        graph_event_count = 0
        last_graph_event_hash: str | None = None

        for event in events:
            event_type = event["event_type"]
            if event_type in {"run_created", "artifact_committed"}:
                continue
            if event_type == "graph_loop_started_v40":
                if started:
                    raise RuntimeError("duplicate graph-loop start event")
                ref = self._committed_ref(event, committed, "graph_loop_contract_v40")
                contract = GraphLoopContractV40.model_validate(
                    self.store.load_artifact(ref)
                )
                contract.assert_sealed()
                if contract != self.contract:
                    raise RuntimeError("graph-loop start contract changed")
                started = True
                graph_event_count = 1
                last_graph_event_hash = str(event["event_hash"])
                continue
            if not started:
                raise RuntimeError("graph mutation occurred before graph-loop start")

            before = self._snapshot(
                nodes,
                edges,
                outcomes,
                promotions,
                revoked,
                graph_event_count,
                last_graph_event_hash,
            )
            if event_type == "graph_node_added_v40":
                ref = self._committed_ref(event, committed, "graph_node_v40")
                node = GraphNodeV40.model_validate(self.store.load_artifact(ref))
                node.assert_sealed()
                if node.layer != self.contract.layer:
                    raise RuntimeError("graph node belongs to another layer")
                if len(nodes) >= self.contract.max_nodes:
                    raise RuntimeError("graph contains more nodes than its budget")
                if any(
                    item.node_hash == node.node_hash or item.node_id == node.node_id
                    for item in nodes
                ):
                    raise RuntimeError("duplicate graph node")
                nodes.append(node)
            elif event_type == "graph_edge_added_v40":
                ref = self._committed_ref(event, committed, "graph_edge_v40")
                edge = GraphEdgeV40.model_validate(self.store.load_artifact(ref))
                edge.assert_sealed()
                node_hashes = {node.node_hash for node in nodes}
                if edge.layer != self.contract.layer:
                    raise RuntimeError("graph edge belongs to another layer")
                if edge.source_node_hash not in node_hashes or edge.target_node_hash not in node_hashes:
                    raise RuntimeError("graph edge references unknown nodes")
                if any(
                    item.edge_hash == edge.edge_hash or item.edge_id == edge.edge_id
                    for item in edges
                ):
                    raise RuntimeError("duplicate graph edge")
                if self._has_path(edge.target_node_hash, edge.source_node_hash, edges):
                    raise RuntimeError("graph contains a cycle")
                edges.append(edge)
            elif event_type == "node_outcome_recorded_v40":
                ref = self._committed_ref(event, committed, "node_outcome_v40")
                outcome = NodeOutcomeV40.model_validate(self.store.load_artifact(ref))
                outcome.assert_sealed()
                self._validate_outcome(outcome, nodes, committed, before)
                outcomes.append(outcome)
            elif event_type == "promotion_decided_v40":
                ref = self._committed_ref(event, committed, "promotion_receipt_v40")
                receipt = PromotionReceiptV40.model_validate(self.store.load_artifact(ref))
                receipt.assert_sealed()
                state = GraphLoopStateV40(
                    contract=self.contract,
                    nodes=nodes,
                    edges=edges,
                    outcomes=outcomes,
                    promotions=promotions,
                    revocations=revocations,
                    snapshot=before,
                )
                self._validate_promotion(receipt, state)
                if len(promotions) >= self.contract.max_promotions:
                    raise RuntimeError("graph contains more promotions than its budget")
                promotions.append(receipt)
            elif event_type == "nodes_revoked_v40":
                ref = self._committed_ref(event, committed, "revocation_receipt_v40")
                receipt = RevocationReceiptV40.model_validate(
                    self.store.load_artifact(ref)
                )
                receipt.assert_sealed()
                if receipt.graph_id != self.contract.graph_id or receipt.layer != self.contract.layer:
                    raise RuntimeError("revocation receipt belongs to another graph")
                if receipt.base_snapshot_hash != before.snapshot_hash:
                    raise RuntimeError("revocation is bound to another graph state")
                if receipt.root_node_hash in revoked:
                    raise RuntimeError("revocation root was already revoked")
                expected = sorted(self._revocation_closure(receipt.root_node_hash, edges))
                if receipt.affected_node_hashes != expected:
                    raise RuntimeError("revocation receipt has an incorrect cascade")
                revoked.update(expected)
                revocations.append(receipt)
            else:
                raise RuntimeError(f"unsupported graph-loop event: {event_type}")

            graph_event_count += 1
            last_graph_event_hash = str(event["event_hash"])

        if not started or last_graph_event_hash is None:
            raise RuntimeError("graph loop has not been started")
        snapshot = self._snapshot(
            nodes,
            edges,
            outcomes,
            promotions,
            revoked,
            graph_event_count,
            last_graph_event_hash,
        )
        projected = GraphLoopStateV40(
            contract=self.contract,
            nodes=nodes,
            edges=edges,
            outcomes=outcomes,
            promotions=promotions,
            revocations=revocations,
            snapshot=snapshot,
        )
        self._projected_state_revision = revision
        self._projected_state_cache = projected
        return projected

    def _snapshot(
        self,
        nodes: list[GraphNodeV40],
        edges: list[GraphEdgeV40],
        outcomes: list[NodeOutcomeV40],
        promotions: list[PromotionReceiptV40],
        revoked: set[str],
        graph_event_count: int,
        last_graph_event_hash: str | None,
    ) -> GraphLoopSnapshotV40:
        if last_graph_event_hash is None:
            raise RuntimeError("graph-loop snapshot needs an event tip")
        statuses: dict[str, EffectiveStatus] = {
            str(node.node_hash): "pending" for node in nodes
        }
        for outcome in outcomes:
            statuses[outcome.node_hash] = outcome.status
        for receipt in promotions:
            statuses[receipt.candidate_node_hash] = receipt.decision
        for node_hash in revoked:
            statuses[node_hash] = "revoked"

        failure_count = sum(
            outcome.status in {"failed", "blocked"} for outcome in outcomes
        )
        budget_exhausted = (
            len(outcomes) >= self.contract.max_outcomes
            or failure_count >= self.contract.max_failures
            or len(promotions) >= self.contract.max_promotions
        )
        frontier: list[str] = []
        if not budget_exhausted:
            for node in nodes:
                assert node.node_hash is not None
                if statuses[node.node_hash] != "pending":
                    continue
                incoming = [
                    edge for edge in edges if edge.target_node_hash == node.node_hash
                ]
                if self._dependencies_satisfied(incoming, statuses):
                    frontier.append(node.node_hash)

        pending_count = sum(status == "pending" for status in statuses.values())
        if budget_exhausted:
            stop_reason = "budget_exhausted"
        elif not frontier and pending_count == 0 and nodes:
            stop_reason = "graph_complete"
        elif not frontier and pending_count:
            stop_reason = "waiting_on_dependencies"
        else:
            stop_reason = None
        return GraphLoopSnapshotV40.seal(
            graph_id=self.contract.graph_id,
            layer=self.contract.layer,
            graph_event_count=graph_event_count,
            last_graph_event_hash=last_graph_event_hash,
            node_statuses=dict(sorted(statuses.items())),
            frontier_node_hashes=sorted(frontier),
            outcome_count=len(outcomes),
            failure_count=failure_count,
            promotion_count=len(promotions),
            budget_exhausted=budget_exhausted,
            stop_reason=stop_reason,
        )

    def _validate_outcome(
        self,
        outcome: NodeOutcomeV40,
        nodes: list[GraphNodeV40],
        committed: dict[tuple[str, str], ArtifactRef],
        before: GraphLoopSnapshotV40,
    ) -> None:
        if outcome.graph_id != self.contract.graph_id or outcome.layer != self.contract.layer:
            raise RuntimeError("node outcome belongs to another graph")
        if outcome.base_snapshot_hash != before.snapshot_hash:
            raise RuntimeError("node outcome is bound to another graph state")
        if outcome.node_hash not in before.frontier_node_hashes:
            raise RuntimeError("node outcome was not recorded from the frontier")
        node = next((item for item in nodes if item.node_hash == outcome.node_hash), None)
        if node is None or node.executor != outcome.actor:
            raise RuntimeError("node outcome actor lacks authority")
        if before.budget_exhausted:
            raise RuntimeError("node outcome exceeds graph-loop budget")
        for ref in outcome.output_artifacts:
            if committed.get((ref.kind, ref.sha256)) != ref:
                raise RuntimeError("node outcome references an uncommitted artifact")
            self.store.load_artifact(ref)

    def _validate_promotion(
        self,
        receipt: PromotionReceiptV40,
        state: GraphLoopStateV40,
    ) -> None:
        if receipt.graph_id != self.contract.graph_id or receipt.layer != self.contract.layer:
            raise RuntimeError("promotion receipt belongs to another graph")
        if receipt.base_snapshot_hash != state.snapshot.snapshot_hash:
            raise RuntimeError("promotion is bound to another graph state")
        if receipt.evaluator_epoch != self.contract.evaluator_epoch:
            raise RuntimeError("promotion used another evaluator epoch")
        candidate = state.node_by_hash(receipt.candidate_node_hash)
        evaluator = state.node_by_hash(receipt.evaluator_node_hash)
        if evaluator.executor != "verifier" or evaluator.node_kind not in {"evaluation", "review"}:
            raise RuntimeError("promotion requires an independent evaluator node")
        if not any(
            edge.source_node_hash == candidate.node_hash
            and edge.target_node_hash == evaluator.node_hash
            and edge.relation == "evaluated_by"
            for edge in state.edges
        ):
            raise RuntimeError("candidate is not bound to the named evaluator")
        statuses = state.snapshot.node_statuses
        if statuses[evaluator.node_hash] != "succeeded":
            raise RuntimeError("evaluator node has not succeeded")
        for node_hash in receipt.evidence_node_hashes:
            if statuses.get(node_hash) not in SUCCESS_STATUSES:
                raise RuntimeError("promotion evidence is not successful and active")
        candidate_status = statuses[candidate.node_hash]
        if receipt.decision == "qualified" and candidate_status != "succeeded":
            raise RuntimeError("only a succeeded candidate may be qualified")
        if receipt.decision == "active" and candidate_status != "qualified":
            raise RuntimeError("only a qualified candidate may become active")
        if receipt.decision == "rejected" and candidate_status not in TERMINAL_STATUSES:
            raise RuntimeError("only a terminal candidate may be rejected")

    def _assert_output_refs(self, refs: list[ArtifactRef]) -> None:
        if not refs:
            raise ValueError("node outcome requires at least one output artifact")
        events = self._read_events(self.store)
        committed = self._committed_artifacts(self.store, events)
        for ref in refs:
            if committed.get((ref.kind, ref.sha256)) != ref:
                raise ValueError("output artifact was not committed by this graph store")
            self.store.load_artifact(ref)

    @staticmethod
    def _dependencies_satisfied(
        incoming: list[GraphEdgeV40], statuses: dict[str, EffectiveStatus]
    ) -> bool:
        for edge in incoming:
            source_status = statuses[edge.source_node_hash]
            if edge.relation in {"requires_success", "derived_from", "evaluated_by"}:
                if source_status not in SUCCESS_STATUSES:
                    return False
            elif edge.relation == "requires_active":
                if source_status != "active":
                    return False
            elif edge.relation in {"requires_terminal", "learned_from_failure"}:
                if source_status not in TERMINAL_STATUSES:
                    return False
                if edge.relation == "learned_from_failure" and source_status not in {
                    "failed",
                    "blocked",
                    "rejected",
                }:
                    return False
            elif edge.relation == "invalidates":
                if source_status not in TERMINAL_STATUSES:
                    return False
            # supersedes is semantic lineage and does not gate readiness.
        return True

    @staticmethod
    def _read_events(store: RunStore) -> list[dict[str, object]]:
        if not store.verify_event_chain():
            raise RuntimeError("graph-loop event hash chain is invalid")
        return [
            json.loads(line)
            for line in store.event_path.read_text(encoding="utf-8").splitlines()
        ]

    @staticmethod
    def _committed_artifacts(
        store: RunStore, events: list[dict[str, object]]
    ) -> dict[tuple[str, str], ArtifactRef]:
        committed: dict[tuple[str, str], ArtifactRef] = {}
        for event in events:
            if event["event_type"] != "artifact_committed":
                continue
            ref = ArtifactRef.model_validate(event["payload"])
            store.load_artifact(ref)
            committed[(ref.kind, ref.sha256)] = ref
        return committed

    @staticmethod
    def _committed_ref(
        event: dict[str, object],
        committed: dict[tuple[str, str], ArtifactRef],
        required_kind: str,
    ) -> ArtifactRef:
        ref = ArtifactRef.model_validate(event["payload"])
        if ref.kind != required_kind:
            raise RuntimeError(f"graph event requires {required_kind}")
        if committed.get((ref.kind, ref.sha256)) != ref:
            raise RuntimeError("graph event references an uncommitted artifact")
        return ref

    @staticmethod
    def _has_path(start: str, target: str, edges: list[GraphEdgeV40]) -> bool:
        adjacency: dict[str, set[str]] = {}
        for edge in edges:
            adjacency.setdefault(edge.source_node_hash, set()).add(edge.target_node_hash)
        pending = [start]
        seen: set[str] = set()
        while pending:
            current = pending.pop()
            if current == target:
                return True
            if current in seen:
                continue
            seen.add(current)
            pending.extend(adjacency.get(current, set()) - seen)
        return False

    @staticmethod
    def _revocation_closure(root: str, edges: list[GraphEdgeV40]) -> set[str]:
        adjacency: dict[str, set[str]] = {}
        for edge in edges:
            if edge.relation in PROPAGATING_RELATIONS:
                adjacency.setdefault(edge.source_node_hash, set()).add(
                    edge.target_node_hash
                )
        affected: set[str] = set()
        pending = [root]
        while pending:
            current = pending.pop()
            if current in affected:
                continue
            affected.add(current)
            pending.extend(adjacency.get(current, set()) - affected)
        return affected


def import_cross_layer_bridge_v40(
    source: GraphLoopStoreV40,
    target: GraphLoopStoreV40,
    source_node_hash: str,
    *,
    target_node_id: str,
    sanitized_payload: object,
    bridge_id: str,
    approved_by: Literal["harness", "human"],
    created_at: datetime | None = None,
) -> tuple[CrossLayerBridgeV40, GraphNodeV40]:
    """Import a release/failure as a pending target-layer proposal.

    The receipt intentionally grants neither scientific validity nor target
    execution.  The target loop must still execute and evaluate the new node.
    """

    source_state = source.project_state()
    source_node = source_state.node_by_hash(source_node_hash)
    source_status = source_state.snapshot.node_statuses[source_node_hash]
    now = created_at or datetime.now(timezone.utc)
    if source.contract.layer == "development" and target.contract.layer == "modeling":
        if source_node.node_kind != "release" or source_status != "succeeded":
            raise ValueError("only a succeeded development release may cross to modeling")
        active_patch_sources = [
            edge.source_node_hash
            for edge in source_state.edges
            if edge.target_node_hash == source_node_hash
            and edge.relation == "requires_active"
            and source_state.node_by_hash(edge.source_node_hash).node_kind == "patch"
            and source_state.snapshot.node_statuses[edge.source_node_hash] == "active"
        ]
        if not active_patch_sources:
            raise ValueError("development release is not bound to an active patch")
        direction = "development_release_to_modeling_runtime"
        target_kind = "runtime_release"
        executor: Actor = "harness"
    elif source.contract.layer == "modeling" and target.contract.layer == "development":
        if source_node.node_kind != "failure" or source_status != "succeeded":
            raise ValueError("only a recorded modeling failure may cross to development")
        direction = "modeling_failure_to_development_issue"
        target_kind = "issue"
        executor = "harness"
    else:
        raise ValueError("unsupported cross-layer bridge direction")

    payload_ref = target.put_output("cross_layer_sanitized_payload_v40", sanitized_payload)
    bridge = CrossLayerBridgeV40.seal(
        bridge_id=bridge_id,
        direction=direction,
        source_graph_id=source.contract.graph_id,
        source_snapshot_hash=source_state.snapshot.snapshot_hash,
        source_node_hash=source_node_hash,
        target_graph_id=target.contract.graph_id,
        sanitized_payload_hash=payload_ref.sha256,
        approved_by=approved_by,
        created_at=now,
    )
    target.put_output("cross_layer_bridge_v40", bridge)
    node = GraphNodeV40.seal(
        node_id=target_node_id,
        layer=target.contract.layer,
        node_kind=target_kind,
        executor=executor,
        created_by=approved_by,
        artifact_hash=bridge.bridge_hash,
        purpose=(
            "evaluate a development release inside the modeling runtime"
            if direction == "development_release_to_modeling_runtime"
            else "diagnose a sanitized modeling failure without private acceptance data"
        ),
        created_at=now,
    )
    target.add_node(node)
    return bridge, node
