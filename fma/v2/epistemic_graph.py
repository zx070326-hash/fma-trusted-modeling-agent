from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore

from .schemas import Identifier, Sha256, _assert_timezone


NodeKind = Literal[
    "source",
    "evidence",
    "data_snapshot",
    "method_claim",
    "model_candidate",
    "validation_report",
    "skill_report",
    "shift_report",
    "decision_report",
    "failure_signature",
    "evolution_operator",
    "benchmark_report",
]
Relation = Literal[
    "supports",
    "refutes",
    "derived_from",
    "evaluates",
    "justifies_use",
    "learned_from",
    "supersedes",
    "invalidates",
]
EffectiveStatus = Literal["active", "refuted", "superseded", "revoked"]

# Edges point from an upstream dependency to its downstream product.  Only
# these relations propagate revocation; refutation and supersession describe
# semantic state rather than derivation.
PROPAGATING_RELATIONS: frozenset[str] = frozenset(
    {"supports", "derived_from", "evaluates", "justifies_use", "learned_from"}
)


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class EpistemicNodeV22(StrictModel):
    """An immutable pointer to a claim or evidence artifact.

    ``artifact_hash`` identifies the original artifact, while ``node_hash``
    seals the modeling-specific interpretation and intended use recorded here.
    """

    schema_version: Literal["2.2"] = "2.2"
    node_id: Identifier
    node_kind: NodeKind
    artifact_hash: Sha256
    mission_spec_hash: Sha256 | None = None
    intended_uses: list[Annotated[str, Field(min_length=3)]] = Field(
        default_factory=list
    )
    created_at: datetime
    node_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_node(self) -> "EpistemicNodeV22":
        _assert_timezone(self.created_at, "created_at")
        if self.node_hash and self.node_hash != self.content_hash():
            raise ValueError("node_hash does not match epistemic node content")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "node_hash")

    def assert_sealed(self) -> None:
        if not self.node_hash or self.node_hash != self.content_hash():
            raise ValueError("epistemic node is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EpistemicNodeV22":
        data.setdefault("created_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"node_hash"}),
            node_hash=draft.content_hash(),
        )


class EpistemicEdgeV22(StrictModel):
    """A typed, immutable influence edge: upstream source -> downstream target."""

    schema_version: Literal["2.2"] = "2.2"
    edge_id: Identifier
    source_node_hash: Sha256
    target_node_hash: Sha256
    relation: Relation
    rationale: Annotated[str, Field(min_length=3)]
    created_at: datetime
    edge_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_edge(self) -> "EpistemicEdgeV22":
        _assert_timezone(self.created_at, "created_at")
        if self.source_node_hash == self.target_node_hash:
            raise ValueError("epistemic self-edges are forbidden")
        if self.edge_hash and self.edge_hash != self.content_hash():
            raise ValueError("edge_hash does not match epistemic edge content")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "edge_hash")

    def assert_sealed(self) -> None:
        if not self.edge_hash or self.edge_hash != self.content_hash():
            raise ValueError("epistemic edge is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EpistemicEdgeV22":
        data.setdefault("created_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"edge_hash"}),
            edge_hash=draft.content_hash(),
        )


class EpistemicGraphSnapshotV22(StrictModel):
    """Deterministic graph projection at one graph-event tip."""

    schema_version: Literal["2.2"] = "2.2"
    graph_id: Identifier
    graph_event_count: Annotated[int, Field(ge=1)]
    last_graph_event_hash: Sha256
    node_statuses: dict[Sha256, EffectiveStatus]
    active_edge_hashes: list[Sha256]
    snapshot_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_snapshot(self) -> "EpistemicGraphSnapshotV22":
        if self.active_edge_hashes != sorted(set(self.active_edge_hashes)):
            raise ValueError("active_edge_hashes must be sorted and unique")
        if self.snapshot_hash and self.snapshot_hash != self.content_hash():
            raise ValueError("snapshot_hash does not match graph projection")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "snapshot_hash")

    def assert_sealed(self) -> None:
        if not self.snapshot_hash or self.snapshot_hash != self.content_hash():
            raise ValueError("epistemic graph snapshot is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EpistemicGraphSnapshotV22":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"snapshot_hash"}),
            snapshot_hash=draft.content_hash(),
        )


class EpistemicRevocationReceiptV22(StrictModel):
    """Append-only invalidation receipt bound to the exact pre-revocation graph."""

    schema_version: Literal["2.2"] = "2.2"
    receipt_id: Identifier
    graph_id: Identifier
    root_node_hash: Sha256
    base_snapshot_hash: Sha256
    affected_node_hashes: list[Sha256] = Field(min_length=1)
    reason: Annotated[str, Field(min_length=3)]
    revoked_at: datetime
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "EpistemicRevocationReceiptV22":
        _assert_timezone(self.revoked_at, "revoked_at")
        if self.affected_node_hashes != sorted(set(self.affected_node_hashes)):
            raise ValueError("affected_node_hashes must be sorted and unique")
        if self.root_node_hash not in self.affected_node_hashes:
            raise ValueError("revocation root must appear in affected_node_hashes")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match revocation receipt")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("revocation receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EpistemicRevocationReceiptV22":
        data.setdefault("revoked_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


class EpistemicGraphStateV22(StrictModel):
    graph_id: Identifier
    nodes: list[EpistemicNodeV22]
    edges: list[EpistemicEdgeV22]
    revocation_receipts: list[EpistemicRevocationReceiptV22]
    snapshot: EpistemicGraphSnapshotV22

    def node_by_hash(self, node_hash: str) -> EpistemicNodeV22:
        for node in self.nodes:
            if node.node_hash == node_hash:
                return node
        raise KeyError(node_hash)


class EpistemicGraphStore:
    """Event-sourced V2.2 evidence graph with code-owned revocation semantics.

    Nodes and edges are content-addressed artifacts.  Mutations are graph
    events in the existing hash-chained ``RunStore``.  Effective state is never
    trusted from a caller; it is rebuilt from artifacts and events on replay.
    """

    def __init__(self, output_root: str | Path, graph_id: str) -> None:
        self.graph_id = graph_id
        self.store = RunStore(output_root, run_id=graph_id)
        self.store.emit(
            "epistemic_graph_started",
            {"graph_id": graph_id, "schema_version": "2.2"},
        )

    @classmethod
    def open_existing(cls, graph_directory: str | Path) -> "EpistemicGraphStore":
        store = RunStore.open_existing(graph_directory)
        events = cls._read_events(store)
        starts = [e for e in events if e["event_type"] == "epistemic_graph_started"]
        if len(starts) != 1:
            raise RuntimeError("epistemic graph needs exactly one start event")
        payload = starts[0]["payload"]
        if payload.get("schema_version") != "2.2":
            raise RuntimeError("unsupported epistemic graph version")
        instance = cls.__new__(cls)
        instance.graph_id = str(payload["graph_id"])
        instance.store = store
        instance.project_state()
        return instance

    @property
    def run_directory(self) -> Path:
        return self.store.run_directory

    def add_node(self, node: EpistemicNodeV22) -> ArtifactRef:
        node.assert_sealed()
        state = self.project_state()
        if any(existing.node_hash == node.node_hash for existing in state.nodes):
            raise ValueError("epistemic node already exists")
        if any(existing.node_id == node.node_id for existing in state.nodes):
            raise ValueError("epistemic node_id already exists")
        ref = self.store.put_artifact("epistemic_node_v22", node)
        self.store.emit("epistemic_node_added", ref.model_dump(mode="json"))
        self.project_state()
        return ref

    def add_edge(self, edge: EpistemicEdgeV22) -> ArtifactRef:
        edge.assert_sealed()
        state = self.project_state()
        node_hashes = {node.node_hash for node in state.nodes}
        if edge.source_node_hash not in node_hashes or edge.target_node_hash not in node_hashes:
            raise ValueError("epistemic edge references an unknown node")
        if any(existing.edge_hash == edge.edge_hash for existing in state.edges):
            raise ValueError("epistemic edge already exists")
        if any(existing.edge_id == edge.edge_id for existing in state.edges):
            raise ValueError("epistemic edge_id already exists")
        if self._has_path(
            edge.target_node_hash,
            edge.source_node_hash,
            state.edges,
        ):
            raise ValueError("epistemic edge would create a cycle")
        ref = self.store.put_artifact("epistemic_edge_v22", edge)
        self.store.emit("epistemic_edge_added", ref.model_dump(mode="json"))
        self.project_state()
        return ref

    def revoke_node(
        self,
        root_node_hash: str,
        *,
        reason: str,
        receipt_id: str,
        revoked_at: datetime | None = None,
    ) -> EpistemicRevocationReceiptV22:
        state = self.project_state()
        if root_node_hash not in state.snapshot.node_statuses:
            raise ValueError("cannot revoke an unknown epistemic node")
        if state.snapshot.node_statuses[root_node_hash] == "revoked":
            raise ValueError("epistemic node is already revoked")
        affected = sorted(self._revocation_closure(root_node_hash, state.edges))
        receipt = EpistemicRevocationReceiptV22.seal(
            receipt_id=receipt_id,
            graph_id=self.graph_id,
            root_node_hash=root_node_hash,
            base_snapshot_hash=state.snapshot.snapshot_hash,
            affected_node_hashes=affected,
            reason=reason,
            revoked_at=revoked_at or datetime.now(timezone.utc),
        )
        ref = self.store.put_artifact("epistemic_revocation_receipt_v22", receipt)
        self.store.emit("epistemic_nodes_revoked", ref.model_dump(mode="json"))
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

    def project_state(self) -> EpistemicGraphStateV22:
        events = self._read_events(self.store)
        committed: dict[tuple[str, str], ArtifactRef] = {}
        nodes: list[EpistemicNodeV22] = []
        edges: list[EpistemicEdgeV22] = []
        receipts: list[EpistemicRevocationReceiptV22] = []
        revoked: set[str] = set()
        graph_event_count = 0
        last_graph_event_hash: str | None = None
        started = False

        for event in events:
            event_type = event["event_type"]
            if event_type == "artifact_committed":
                ref = ArtifactRef.model_validate(event["payload"])
                self.store.load_artifact(ref)
                committed[(ref.kind, ref.sha256)] = ref
                continue
            if event_type == "run_created":
                continue
            if event_type == "epistemic_graph_started":
                if started or event["payload"] != {
                    "graph_id": self.graph_id,
                    "schema_version": "2.2",
                }:
                    raise RuntimeError("invalid epistemic graph start event")
                started = True
                graph_event_count += 1
                last_graph_event_hash = event["event_hash"]
                continue
            if not started:
                raise RuntimeError("graph mutation occurred before graph start")

            if event_type == "epistemic_node_added":
                ref = self._committed_ref(event, committed, "epistemic_node_v22")
                node = EpistemicNodeV22.model_validate(self.store.load_artifact(ref))
                node.assert_sealed()
                if any(n.node_hash == node.node_hash or n.node_id == node.node_id for n in nodes):
                    raise RuntimeError("duplicate epistemic node")
                nodes.append(node)
            elif event_type == "epistemic_edge_added":
                ref = self._committed_ref(event, committed, "epistemic_edge_v22")
                edge = EpistemicEdgeV22.model_validate(self.store.load_artifact(ref))
                edge.assert_sealed()
                node_hashes = {node.node_hash for node in nodes}
                if edge.source_node_hash not in node_hashes or edge.target_node_hash not in node_hashes:
                    raise RuntimeError("epistemic edge references unknown nodes")
                if any(e.edge_hash == edge.edge_hash or e.edge_id == edge.edge_id for e in edges):
                    raise RuntimeError("duplicate epistemic edge")
                if self._has_path(edge.target_node_hash, edge.source_node_hash, edges):
                    raise RuntimeError("epistemic graph contains a cycle")
                edges.append(edge)
            elif event_type == "epistemic_nodes_revoked":
                ref = self._committed_ref(
                    event, committed, "epistemic_revocation_receipt_v22"
                )
                receipt = EpistemicRevocationReceiptV22.model_validate(
                    self.store.load_artifact(ref)
                )
                receipt.assert_sealed()
                if receipt.graph_id != self.graph_id:
                    raise RuntimeError("revocation receipt belongs to another graph")
                if receipt.root_node_hash in revoked:
                    raise RuntimeError("revocation root was already revoked")
                base = self._snapshot(
                    nodes,
                    edges,
                    revoked,
                    graph_event_count,
                    last_graph_event_hash,
                )
                if receipt.base_snapshot_hash != base.snapshot_hash:
                    raise RuntimeError("revocation receipt is bound to another graph state")
                expected = sorted(
                    self._revocation_closure(receipt.root_node_hash, edges)
                )
                if receipt.affected_node_hashes != expected:
                    raise RuntimeError("revocation receipt has an incorrect cascade")
                revoked.update(expected)
                receipts.append(receipt)
            else:
                raise RuntimeError(f"unsupported epistemic graph event: {event_type}")

            graph_event_count += 1
            last_graph_event_hash = event["event_hash"]

        if not started or last_graph_event_hash is None:
            raise RuntimeError("epistemic graph has not been started")
        snapshot = self._snapshot(
            nodes,
            edges,
            revoked,
            graph_event_count,
            last_graph_event_hash,
        )
        return EpistemicGraphStateV22(
            graph_id=self.graph_id,
            nodes=nodes,
            edges=edges,
            revocation_receipts=receipts,
            snapshot=snapshot,
        )

    def _snapshot(
        self,
        nodes: list[EpistemicNodeV22],
        edges: list[EpistemicEdgeV22],
        revoked: set[str],
        graph_event_count: int,
        last_graph_event_hash: str | None,
    ) -> EpistemicGraphSnapshotV22:
        if last_graph_event_hash is None:
            raise RuntimeError("graph snapshot needs an event tip")
        statuses: dict[str, EffectiveStatus] = {}
        superseded = {edge.target_node_hash for edge in edges if edge.relation == "supersedes"}
        refuted = {edge.target_node_hash for edge in edges if edge.relation == "refutes"}
        for node in nodes:
            assert node.node_hash is not None
            if node.node_hash in revoked:
                status: EffectiveStatus = "revoked"
            elif node.node_hash in superseded:
                status = "superseded"
            elif node.node_hash in refuted:
                status = "refuted"
            else:
                status = "active"
            statuses[node.node_hash] = status
        edge_hashes = sorted(
            edge.edge_hash for edge in edges if edge.edge_hash is not None
        )
        return EpistemicGraphSnapshotV22.seal(
            graph_id=self.graph_id,
            graph_event_count=graph_event_count,
            last_graph_event_hash=last_graph_event_hash,
            node_statuses=dict(sorted(statuses.items())),
            active_edge_hashes=edge_hashes,
        )

    @staticmethod
    def _read_events(store: RunStore) -> list[dict[str, object]]:
        if not store.verify_event_chain():
            raise RuntimeError("epistemic graph event hash chain is invalid")
        return [
            json.loads(line)
            for line in store.event_path.read_text(encoding="utf-8").splitlines()
        ]

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
    def _has_path(
        start: str, target: str, edges: list[EpistemicEdgeV22]
    ) -> bool:
        adjacency: dict[str, set[str]] = {}
        for edge in edges:
            adjacency.setdefault(edge.source_node_hash, set()).add(edge.target_node_hash)
        stack = [start]
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            if current == target:
                return True
            if current in seen:
                continue
            seen.add(current)
            stack.extend(adjacency.get(current, ()))
        return False

    @staticmethod
    def _revocation_closure(
        root: str, edges: list[EpistemicEdgeV22]
    ) -> set[str]:
        known = {
            node_hash
            for edge in edges
            for node_hash in (edge.source_node_hash, edge.target_node_hash)
        }
        if edges and root not in known:
            # An isolated node is still a valid root; the caller/projector
            # checks node existence independently.
            return {root}
        adjacency: dict[str, set[str]] = {}
        for edge in edges:
            if edge.relation in PROPAGATING_RELATIONS:
                adjacency.setdefault(edge.source_node_hash, set()).add(
                    edge.target_node_hash
                )
        affected: set[str] = set()
        stack = [root]
        while stack:
            current = stack.pop()
            if current in affected:
                continue
            affected.add(current)
            stack.extend(adjacency.get(current, ()))
        return affected


def register_official_shadow_run(
    graph: EpistemicGraphStore,
    run_directory: str | Path,
) -> dict[str, str]:
    """Import one independently verified official run as graph pointers.

    The function never trusts a manifest without first replaying the official
    run.  It records external artifact hashes rather than copying the artifacts.
    Evolved portfolios are candidate memories only; this function does not
    promote them for use.
    """

    from .empirical_schemas import ForecastCandidatePortfolio
    from .official_shadow import OfficialShadowManifest, verify_official_shadow_run

    directory = Path(run_directory).resolve()
    if not verify_official_shadow_run(directory):
        raise ValueError("official shadow run failed independent verification")
    source_store = RunStore.open_existing(directory)
    events = EpistemicGraphStore._read_events(source_store)
    committed = [
        ArtifactRef.model_validate(event["payload"])
        for event in events
        if event["event_type"] == "artifact_committed"
    ]
    manifest_refs = [ref for ref in committed if ref.kind == "official_shadow_manifest"]
    if len(manifest_refs) != 1:
        raise RuntimeError("official run needs exactly one manifest")
    manifest = OfficialShadowManifest.model_validate(
        source_store.load_artifact(manifest_refs[0])
    )
    manifest.assert_sealed()

    refs: dict[str, ArtifactRef] = {}
    for kind in [
        "official_api_raw_response",
        "official_data_receipt",
        "timeseries_snapshot",
        "forecast_candidate_portfolio",
        "forecast_validation_report",
        "forecast_skill_uncertainty_report",
        "window_shift_report",
        "official_shadow_report",
    ]:
        matches = [ref for ref in manifest.artifact_refs if ref.kind == kind]
        if len(matches) != 1:
            raise RuntimeError(f"official manifest needs exactly one {kind}")
        source_store.load_artifact(matches[0])
        refs[kind] = matches[0]

    portfolio = ForecastCandidatePortfolio.model_validate(
        source_store.load_artifact(refs["forecast_candidate_portfolio"])
    )
    portfolio.assert_sealed()
    created_at = manifest.created_at
    node_plan: list[tuple[str, NodeKind, ArtifactRef, list[str]]] = [
        ("raw_source", "source", refs["official_api_raw_response"], ["replay_only"]),
        ("source_receipt", "evidence", refs["official_data_receipt"], ["provenance_check"]),
        ("data_snapshot", "data_snapshot", refs["timeseries_snapshot"], ["retrospective_shadow"]),
        ("model_portfolio", "model_candidate", refs["forecast_candidate_portfolio"], ["shadow_evaluation"]),
        ("validation", "validation_report", refs["forecast_validation_report"], ["shadow_evaluation"]),
        ("skill", "skill_report", refs["forecast_skill_uncertainty_report"], ["relative_skill_only"]),
        ("shift", "shift_report", refs["window_shift_report"], ["distribution_shift_check"]),
        ("decision", "decision_report", refs["official_shadow_report"], ["shadow_only_no_real_action"]),
    ]
    if portfolio.generator_id == "failure_evolved_forecast_generator_v1":
        node_plan.append(
            (
                "operator_memory",
                "evolution_operator",
                refs["forecast_candidate_portfolio"],
                ["candidate_only_requires_hidden_worldpack"],
            )
        )

    hashes: dict[str, str] = {}
    for label, kind, ref, uses in node_plan:
        node = EpistemicNodeV22.seal(
            node_id=f"{label}_{ref.sha256[:16]}",
            node_kind=kind,
            artifact_hash=ref.sha256,
            intended_uses=uses,
            created_at=created_at,
        )
        graph.add_node(node)
        assert node.node_hash is not None
        hashes[label] = node.node_hash

    dependencies = [
        ("raw_source", "source_receipt", "derived_from"),
        ("raw_source", "data_snapshot", "derived_from"),
        ("data_snapshot", "validation", "evaluates"),
        ("model_portfolio", "validation", "evaluates"),
        ("validation", "skill", "derived_from"),
        ("data_snapshot", "shift", "evaluates"),
        ("validation", "decision", "justifies_use"),
        ("skill", "decision", "justifies_use"),
        ("shift", "decision", "justifies_use"),
    ]
    if "operator_memory" in hashes:
        dependencies.append(("skill", "operator_memory", "learned_from"))
    for index, (source, target, relation) in enumerate(dependencies, start=1):
        edge = EpistemicEdgeV22.seal(
            edge_id=f"official_edge_{index}_{hashes[source][:8]}_{hashes[target][:8]}",
            source_node_hash=hashes[source],
            target_node_hash=hashes[target],
            relation=relation,
            rationale=f"verified official run dependency: {source} to {target}",
            created_at=created_at,
        )
        graph.add_edge(edge)
    return hashes


def register_method_learning_run(
    graph: EpistemicGraphStore,
    run_directory: str | Path,
) -> dict[str, str]:
    """Register a verified web-method candidate without upgrading its status."""

    from .method_knowledge import (
        MethodLearningManifestV22,
        verify_method_learning_run,
    )

    directory = Path(run_directory).resolve()
    if not verify_method_learning_run(directory):
        raise ValueError("method learning run failed independent verification")
    source_store = RunStore.open_existing(directory)
    events = EpistemicGraphStore._read_events(source_store)
    committed = [
        ArtifactRef.model_validate(event["payload"])
        for event in events
        if event["event_type"] == "artifact_committed"
    ]
    manifest_refs = [ref for ref in committed if ref.kind == "method_learning_manifest_v22"]
    if len(manifest_refs) != 1:
        raise RuntimeError("method learning run needs exactly one manifest")
    manifest = MethodLearningManifestV22.model_validate(
        source_store.load_artifact(manifest_refs[0])
    )
    manifest.assert_sealed()
    by_kind: dict[str, ArtifactRef] = {}
    for kind in [
        "method_evidence_snapshot_v22",
        "method_claim_draft_v22",
        "evolution_operator_knowledge_v22",
    ]:
        matches = [ref for ref in manifest.artifact_refs if ref.kind == kind]
        if len(matches) != 1:
            raise RuntimeError(f"method manifest needs exactly one {kind}")
        source_store.load_artifact(matches[0])
        by_kind[kind] = matches[0]

    plan: list[tuple[str, NodeKind, ArtifactRef, list[str]]] = [
        (
            "method_source",
            "source",
            by_kind["method_evidence_snapshot_v22"],
            ["untrusted_method_evidence"],
        ),
        (
            "method_claim",
            "method_claim",
            by_kind["method_claim_draft_v22"],
            ["candidate_interpretation_only"],
        ),
        (
            "operator_memory",
            "evolution_operator",
            by_kind["evolution_operator_knowledge_v22"],
            ["candidate_only_requires_hidden_worldpack"],
        ),
    ]
    hashes: dict[str, str] = {}
    for label, kind, ref, uses in plan:
        node = EpistemicNodeV22.seal(
            node_id=f"{label}_{ref.sha256[:16]}",
            node_kind=kind,
            artifact_hash=ref.sha256,
            intended_uses=uses,
            created_at=manifest.created_at,
        )
        graph.add_node(node)
        assert node.node_hash is not None
        hashes[label] = node.node_hash
    for index, (source, target, relation) in enumerate(
        [
            ("method_source", "method_claim", "supports"),
            ("method_claim", "operator_memory", "learned_from"),
        ],
        start=1,
    ):
        graph.add_edge(
            EpistemicEdgeV22.seal(
                edge_id=f"method_edge_{index}_{hashes[source][:8]}_{hashes[target][:8]}",
                source_node_hash=hashes[source],
                target_node_hash=hashes[target],
                relation=relation,
                rationale=f"verified method-learning dependency: {source} to {target}",
                created_at=manifest.created_at,
            )
        )
    return hashes


def register_worldpack_run(
    graph: EpistemicGraphStore,
    run_directory: str | Path,
) -> dict[str, str]:
    """Register the exact tested arm policies and their hidden ablation verdict."""

    from .worldpack import (
        WorldPackAblationReportV22,
        WorldPackManifestV22,
        verify_worldpack_run,
    )

    directory = Path(run_directory).resolve()
    if not verify_worldpack_run(directory):
        raise ValueError("WorldPack run failed independent verification")
    source_store = RunStore.open_existing(directory)
    events = EpistemicGraphStore._read_events(source_store)
    committed = [
        ArtifactRef.model_validate(event["payload"])
        for event in events
        if event["event_type"] == "artifact_committed"
    ]
    manifest_refs = [ref for ref in committed if ref.kind == "worldpack_manifest_v22"]
    if len(manifest_refs) != 1:
        raise RuntimeError("WorldPack run needs exactly one manifest")
    manifest = WorldPackManifestV22.model_validate(
        source_store.load_artifact(manifest_refs[0])
    )
    manifest.assert_sealed()
    required = [
        "private_worldpack_v22",
        "worldpack_direct_policy_v22",
        "worldpack_memory_policy_v22",
        "worldpack_direct_selections_v22",
        "worldpack_memory_selections_v22",
        "worldpack_ablation_report_v22",
    ]
    refs: dict[str, ArtifactRef] = {}
    for kind in required:
        matches = [ref for ref in manifest.artifact_refs if ref.kind == kind]
        if len(matches) != 1:
            raise RuntimeError(f"WorldPack manifest needs exactly one {kind}")
        source_store.load_artifact(matches[0])
        refs[kind] = matches[0]
    report = WorldPackAblationReportV22.model_validate(
        source_store.load_artifact(refs["worldpack_ablation_report_v22"])
    )
    report.assert_sealed()

    node_plan: list[tuple[str, NodeKind, ArtifactRef, list[str]]] = [
        (
            "worldpack_direct_policy",
            "model_candidate",
            refs["worldpack_direct_policy_v22"],
            ["paired_ablation_baseline_policy"],
        ),
        (
            "worldpack_memory_policy",
            "evolution_operator",
            refs["worldpack_memory_policy_v22"],
            ["candidate_only_worldpack_scope"],
        ),
        (
            "worldpack_private",
            "evidence",
            refs["private_worldpack_v22"],
            ["hidden_benchmark_only"],
        ),
        (
            "worldpack_direct",
            "evidence",
            refs["worldpack_direct_selections_v22"],
            ["paired_ablation_baseline"],
        ),
        (
            "worldpack_memory",
            "evidence",
            refs["worldpack_memory_selections_v22"],
            ["paired_ablation_treatment"],
        ),
        (
            "worldpack_report",
            "benchmark_report",
            refs["worldpack_ablation_report_v22"],
            ["worldpack_scope_only"],
        ),
    ]
    hashes: dict[str, str] = {}
    for label, kind, ref, uses in node_plan:
        node = EpistemicNodeV22.seal(
            node_id=f"{label}_{ref.sha256[:16]}",
            node_kind=kind,
            artifact_hash=ref.sha256,
            intended_uses=uses,
            created_at=manifest.created_at,
        )
        graph.add_node(node)
        assert node.node_hash is not None
        hashes[label] = node.node_hash
    for index, source in enumerate(
        ["worldpack_private", "worldpack_direct", "worldpack_memory"], start=1
    ):
        graph.add_edge(
            EpistemicEdgeV22.seal(
                edge_id=f"worldpack_evidence_{index}_{hashes[source][:8]}",
                source_node_hash=hashes[source],
                target_node_hash=hashes["worldpack_report"],
                relation="derived_from",
                rationale=f"WorldPack report replays from {source}",
                created_at=manifest.created_at,
            )
        )
    verdict_relation: Relation = (
        "supports" if report.status == "promoted_for_worldpack_scope" else "refutes"
    )
    graph.add_edge(
        EpistemicEdgeV22.seal(
            edge_id=f"worldpack_verdict_{hashes['worldpack_report'][:8]}_{hashes['worldpack_memory_policy'][:8]}",
            source_node_hash=hashes["worldpack_report"],
            target_node_hash=hashes["worldpack_memory_policy"],
            relation=verdict_relation,
            rationale=f"pre-frozen WorldPack verdict: {report.status}",
            created_at=manifest.created_at,
        )
    )
    return hashes


def register_worldpack_confirmation_run(
    graph: EpistemicGraphStore,
    run_directory: str | Path,
    *,
    existing_memory_policy_node_hash: str | None = None,
) -> dict[str, str]:
    """Register a V2.3 confirmation and create qualification only on PASS."""

    from .worldpack import WorldPackArmPolicyV22
    from .worldpack_v23 import (
        WorldPackConfirmationManifestV23,
        WorldPackConfirmationReportV23,
        qualify_worldpack_policy_v23,
        verify_worldpack_confirmation_v23,
    )

    directory = Path(run_directory).resolve()
    if not verify_worldpack_confirmation_v23(directory):
        raise ValueError("V2.3 WorldPack confirmation failed independent verification")
    source_store = RunStore.open_existing(directory)
    events = EpistemicGraphStore._read_events(source_store)
    committed = [
        ArtifactRef.model_validate(event["payload"])
        for event in events
        if event["event_type"] == "artifact_committed"
    ]
    manifests = [
        ref for ref in committed if ref.kind == "worldpack_confirmation_manifest_v23"
    ]
    if len(manifests) != 1:
        raise RuntimeError("V2.3 confirmation needs exactly one manifest")
    manifest = WorldPackConfirmationManifestV23.model_validate(
        source_store.load_artifact(manifests[0])
    )
    manifest.assert_sealed()
    required = [
        "worldpack_memory_policy_v22",
        "private_worldpack_v22",
        "worldpack_direct_selections_v22",
        "worldpack_memory_selections_v22",
        "worldpack_confirmation_report_v23",
    ]
    refs: dict[str, ArtifactRef] = {}
    for kind in required:
        matches = [ref for ref in manifest.artifact_refs if ref.kind == kind]
        if len(matches) != 1:
            raise RuntimeError(f"V2.3 manifest needs exactly one {kind}")
        source_store.load_artifact(matches[0])
        refs[kind] = matches[0]
    policy = WorldPackArmPolicyV22.model_validate(
        source_store.load_artifact(refs["worldpack_memory_policy_v22"])
    )
    policy.assert_sealed()
    report = WorldPackConfirmationReportV23.model_validate(
        source_store.load_artifact(refs["worldpack_confirmation_report_v23"])
    )
    report.assert_sealed()

    hashes: dict[str, str] = {}
    if existing_memory_policy_node_hash is None:
        policy_node = EpistemicNodeV22.seal(
            node_id=f"confirmed_memory_policy_{refs['worldpack_memory_policy_v22'].sha256[:16]}",
            node_kind="evolution_operator",
            artifact_hash=refs["worldpack_memory_policy_v22"].sha256,
            intended_uses=["candidate_only_until_qualification"],
            created_at=manifest.created_at,
        )
        graph.add_node(policy_node)
        assert policy_node.node_hash is not None
        hashes["memory_policy"] = policy_node.node_hash
    else:
        existing = graph.project_state().node_by_hash(existing_memory_policy_node_hash)
        if (
            existing.node_kind != "evolution_operator"
            or existing.artifact_hash != refs["worldpack_memory_policy_v22"].sha256
        ):
            raise ValueError("existing memory-policy node is not the confirmed artifact")
        hashes["memory_policy"] = existing_memory_policy_node_hash

    node_plan: list[tuple[str, NodeKind, ArtifactRef, list[str]]] = [
        (
            "confirmation_private",
            "evidence",
            refs["private_worldpack_v22"],
            ["hidden_confirmation_only"],
        ),
        (
            "confirmation_direct",
            "evidence",
            refs["worldpack_direct_selections_v22"],
            ["paired_confirmation_baseline"],
        ),
        (
            "confirmation_memory",
            "evidence",
            refs["worldpack_memory_selections_v22"],
            ["paired_confirmation_treatment"],
        ),
        (
            "confirmation_report",
            "benchmark_report",
            refs["worldpack_confirmation_report_v23"],
            ["synthetic_forecast_worldpack_v23"],
        ),
    ]
    for label, kind, ref, uses in node_plan:
        node = EpistemicNodeV22.seal(
            node_id=f"{label}_{ref.sha256[:16]}",
            node_kind=kind,
            artifact_hash=ref.sha256,
            intended_uses=uses,
            created_at=manifest.created_at,
        )
        graph.add_node(node)
        assert node.node_hash is not None
        hashes[label] = node.node_hash
    for index, source in enumerate(
        ["confirmation_private", "confirmation_direct", "confirmation_memory"],
        start=1,
    ):
        graph.add_edge(
            EpistemicEdgeV22.seal(
                edge_id=f"confirmation_evidence_{index}_{hashes[source][:8]}",
                source_node_hash=hashes[source],
                target_node_hash=hashes["confirmation_report"],
                relation="derived_from",
                rationale=f"V2.3 confirmation report replays from {source}",
                created_at=manifest.created_at,
            )
        )

    if report.status == "promoted_for_worldpack_scope_v23":
        qualification = qualify_worldpack_policy_v23(policy, report)
        qualification_ref = graph.store.put_artifact(
            "operator_qualification_v23", qualification
        )
        qualification_node = EpistemicNodeV22.seal(
            node_id=f"qualified_policy_{qualification_ref.sha256[:16]}",
            node_kind="evolution_operator",
            artifact_hash=qualification_ref.sha256,
            intended_uses=["qualified_synthetic_forecast_worldpack_v23_only"],
            created_at=manifest.created_at,
        )
        graph.add_node(qualification_node)
        assert qualification_node.node_hash is not None
        hashes["qualified_policy"] = qualification_node.node_hash
        for index, (source, relation) in enumerate(
            [
                ("memory_policy", "supports"),
                ("confirmation_report", "justifies_use"),
            ],
            start=1,
        ):
            graph.add_edge(
                EpistemicEdgeV22.seal(
                    edge_id=f"qualification_edge_{index}_{hashes[source][:8]}",
                    source_node_hash=hashes[source],
                    target_node_hash=hashes["qualified_policy"],
                    relation=relation,
                    rationale="exact policy and passing report jointly bind qualification",
                    created_at=manifest.created_at,
                )
            )
    else:
        graph.add_edge(
            EpistemicEdgeV22.seal(
                edge_id=f"confirmation_refutes_{hashes['confirmation_report'][:8]}_{hashes['memory_policy'][:8]}",
                source_node_hash=hashes["confirmation_report"],
                target_node_hash=hashes["memory_policy"],
                relation="refutes",
                rationale=f"pre-frozen V2.3 verdict: {report.status}",
                created_at=manifest.created_at,
            )
        )
    return hashes


def register_dynamics_knowledge_run(
    graph: EpistemicGraphStore,
    run_directory: str | Path,
) -> dict[str, str]:
    """Register source-bound Dynamics claims as candidate memory, never qualification."""

    from .dynamics_knowledge import (
        DynamicsKnowledgeBundleV24,
        DynamicsKnowledgeManifestV24,
        DynamicsMethodClaimV24,
        LiteratureEvidenceSnapshotV24,
        verify_dynamics_knowledge_run,
    )

    directory = Path(run_directory).resolve()
    if not verify_dynamics_knowledge_run(directory):
        raise ValueError("Dynamics knowledge run failed independent verification")
    source_store = RunStore.open_existing(directory)
    events = EpistemicGraphStore._read_events(source_store)
    committed = [
        ArtifactRef.model_validate(event["payload"])
        for event in events
        if event["event_type"] == "artifact_committed"
    ]
    manifest_refs = [
        ref for ref in committed if ref.kind == "dynamics_knowledge_manifest_v24"
    ]
    if len(manifest_refs) != 1:
        raise RuntimeError("Dynamics knowledge run needs exactly one manifest")
    manifest = DynamicsKnowledgeManifestV24.model_validate(
        source_store.load_artifact(manifest_refs[0])
    )
    manifest.assert_sealed()

    snapshot_refs = [
        ref
        for ref in manifest.artifact_refs
        if ref.kind == "literature_evidence_snapshot_v24"
    ]
    claim_refs = [
        ref for ref in manifest.artifact_refs if ref.kind == "dynamics_method_claim_v24"
    ]
    bundle_refs = [
        ref
        for ref in manifest.artifact_refs
        if ref.kind == "dynamics_knowledge_bundle_v24"
    ]
    if len(snapshot_refs) != 3 or len(claim_refs) != 3 or len(bundle_refs) != 1:
        raise RuntimeError("Dynamics knowledge manifest has an invalid evidence shape")
    snapshots = [
        (
            ref,
            LiteratureEvidenceSnapshotV24.model_validate(source_store.load_artifact(ref)),
        )
        for ref in snapshot_refs
    ]
    claims = [
        (ref, DynamicsMethodClaimV24.model_validate(source_store.load_artifact(ref)))
        for ref in claim_refs
    ]
    bundle_ref = bundle_refs[0]
    bundle = DynamicsKnowledgeBundleV24.model_validate(
        source_store.load_artifact(bundle_ref)
    )
    for _, item in [*snapshots, *claims]:
        item.assert_sealed()
    bundle.assert_sealed()

    hashes: dict[str, str] = {}
    snapshot_node_by_internal_hash: dict[str, str] = {}
    for ref, snapshot in snapshots:
        node = EpistemicNodeV22.seal(
            node_id=f"dynamics_source_{snapshot.source_id}_{ref.sha256[:10]}",
            node_kind="source",
            artifact_hash=ref.sha256,
            intended_uses=["untrusted_bibliographic_evidence_only"],
            created_at=manifest.created_at,
        )
        graph.add_node(node)
        assert node.node_hash is not None and snapshot.snapshot_hash is not None
        hashes[f"source_{snapshot.source_id}"] = node.node_hash
        snapshot_node_by_internal_hash[snapshot.snapshot_hash] = node.node_hash
    claim_node_by_hash: dict[str, str] = {}
    for ref, claim in claims:
        node = EpistemicNodeV22.seal(
            node_id=f"dynamics_claim_{claim.claim_id}_{ref.sha256[:10]}",
            node_kind="method_claim",
            artifact_hash=ref.sha256,
            intended_uses=["candidate_interpretation_only"],
            created_at=manifest.created_at,
        )
        graph.add_node(node)
        assert node.node_hash is not None and claim.claim_hash is not None
        hashes[f"claim_{claim.claim_id}"] = node.node_hash
        claim_node_by_hash[claim.claim_hash] = node.node_hash
        graph.add_edge(
            EpistemicEdgeV22.seal(
                edge_id=f"dynamics_source_claim_{ref.sha256[:12]}",
                source_node_hash=snapshot_node_by_internal_hash[claim.evidence_snapshot_hash],
                target_node_hash=node.node_hash,
                relation="supports",
                rationale="frozen source metadata supports only this candidate interpretation",
                created_at=manifest.created_at,
            )
        )
    bundle_node = EpistemicNodeV22.seal(
        node_id=f"dynamics_knowledge_{bundle_ref.sha256[:16]}",
        node_kind="evolution_operator",
        artifact_hash=bundle_ref.sha256,
        intended_uses=["candidate_only_requires_hidden_dynamics_worldpack"],
        created_at=manifest.created_at,
    )
    graph.add_node(bundle_node)
    assert bundle_node.node_hash is not None
    hashes["knowledge_bundle"] = bundle_node.node_hash
    for index, claim_hash in enumerate(bundle.claim_hashes, start=1):
        graph.add_edge(
            EpistemicEdgeV22.seal(
                edge_id=f"dynamics_claim_bundle_{index}_{claim_hash[:10]}",
                source_node_hash=claim_node_by_hash[claim_hash],
                target_node_hash=bundle_node.node_hash,
                relation="learned_from",
                rationale="candidate Dynamics design rule remains bound to its interpreted claim",
                created_at=manifest.created_at,
            )
        )
    return hashes


def register_dynamics_worldpack_run(
    graph: EpistemicGraphStore,
    run_directory: str | Path,
    *,
    knowledge_bundle_node_hash: str | None = None,
    existing_memory_policy_node_hash: str | None = None,
    evolution_evidence_node_hash: str | None = None,
) -> dict[str, str]:
    """Register an exact Dynamics policy verdict and optional limited qualification."""

    from .dynamics_ir import DynamicsArmPolicyV24
    from .dynamics_worldpack import (
        DynamicsOperatorQualificationV24,
        DynamicsWorldPackManifestV24,
        DynamicsWorldPackReportV24,
        verify_dynamics_worldpack_run,
    )

    directory = Path(run_directory).resolve()
    if not verify_dynamics_worldpack_run(directory):
        raise ValueError("Dynamics WorldPack run failed independent verification")
    source_store = RunStore.open_existing(directory)
    events = EpistemicGraphStore._read_events(source_store)
    committed = [
        ArtifactRef.model_validate(event["payload"])
        for event in events
        if event["event_type"] == "artifact_committed"
    ]
    manifest_refs = [
        ref for ref in committed if ref.kind == "dynamics_worldpack_manifest_v24"
    ]
    if len(manifest_refs) != 1:
        raise RuntimeError("Dynamics WorldPack run needs exactly one manifest")
    manifest = DynamicsWorldPackManifestV24.model_validate(
        source_store.load_artifact(manifest_refs[0])
    )
    manifest.assert_sealed()

    def ref_one(kind: str) -> ArtifactRef:
        refs = [ref for ref in manifest.artifact_refs if ref.kind == kind]
        if len(refs) != 1:
            raise RuntimeError(f"Dynamics manifest needs exactly one {kind}")
        source_store.load_artifact(refs[0])
        return refs[0]

    refs = {
        kind: ref_one(kind)
        for kind in [
            "dynamics_worldpack_spec_v24",
            "dynamics_memory_policy_v24",
            "private_dynamics_worldpack_v24",
            "dynamics_direct_selections_v24",
            "dynamics_memory_selections_v24",
            "dynamics_worldpack_report_v24",
        ]
    }
    policy = DynamicsArmPolicyV24.model_validate(
        source_store.load_artifact(refs["dynamics_memory_policy_v24"])
    )
    report = DynamicsWorldPackReportV24.model_validate(
        source_store.load_artifact(refs["dynamics_worldpack_report_v24"])
    )
    policy.assert_sealed()
    report.assert_sealed()
    hashes: dict[str, str] = {}
    if existing_memory_policy_node_hash is None:
        policy_node = EpistemicNodeV22.seal(
            node_id=f"dynamics_policy_{refs['dynamics_memory_policy_v24'].sha256[:16]}",
            node_kind="evolution_operator",
            artifact_hash=refs["dynamics_memory_policy_v24"].sha256,
            intended_uses=["candidate_until_dynamics_qualification"],
            created_at=manifest.created_at,
        )
        graph.add_node(policy_node)
        assert policy_node.node_hash is not None
        hashes["memory_policy"] = policy_node.node_hash
    else:
        existing = graph.project_state().node_by_hash(existing_memory_policy_node_hash)
        if existing.artifact_hash != refs["dynamics_memory_policy_v24"].sha256:
            raise ValueError("existing Dynamics policy node is bound to another artifact")
        hashes["memory_policy"] = existing_memory_policy_node_hash

    node_plan = [
        ("spec", "evidence", "dynamics_worldpack_spec_v24", ["frozen_protocol"]),
        ("private", "evidence", "private_dynamics_worldpack_v24", ["hidden_benchmark_only"]),
        ("direct", "evidence", "dynamics_direct_selections_v24", ["paired_baseline"]),
        ("memory", "evidence", "dynamics_memory_selections_v24", ["paired_treatment"]),
        ("report", "benchmark_report", "dynamics_worldpack_report_v24", ["synthetic_dynamics_scope_only"]),
    ]
    for label, kind, ref_kind, uses in node_plan:
        ref = refs[ref_kind]
        node = EpistemicNodeV22.seal(
            node_id=f"dynamics_{label}_{ref.sha256[:16]}",
            node_kind=kind,
            artifact_hash=ref.sha256,
            intended_uses=uses,
            created_at=manifest.created_at,
        )
        graph.add_node(node)
        assert node.node_hash is not None
        hashes[label] = node.node_hash
    for index, source in enumerate(["spec", "private", "direct", "memory"], start=1):
        graph.add_edge(
            EpistemicEdgeV22.seal(
                edge_id=f"dynamics_report_evidence_{index}_{hashes[source][:8]}",
                source_node_hash=hashes[source],
                target_node_hash=hashes["report"],
                relation="derived_from",
                rationale=f"Dynamics report independently replays from {source}",
                created_at=manifest.created_at,
            )
        )
    for label, source_hash in (
        ("knowledge", knowledge_bundle_node_hash),
        ("failure", evolution_evidence_node_hash),
    ):
        if source_hash is None:
            continue
        graph.project_state().node_by_hash(source_hash)
        graph.add_edge(
            EpistemicEdgeV22.seal(
                edge_id=f"dynamics_policy_{label}_{source_hash[:8]}_{hashes['memory_policy'][:8]}",
                source_node_hash=source_hash,
                target_node_hash=hashes["memory_policy"],
                relation="learned_from",
                rationale=f"exact Dynamics policy is bound to its {label} lineage",
                created_at=manifest.created_at,
            )
        )

    if report.status == "candidate_rejected_dynamics_v24":
        graph.add_edge(
            EpistemicEdgeV22.seal(
                edge_id=f"dynamics_policy_refuted_{hashes['report'][:8]}_{hashes['memory_policy'][:8]}",
                source_node_hash=hashes["report"],
                target_node_hash=hashes["memory_policy"],
                relation="refutes",
                rationale="pre-frozen Dynamics confirmation gate rejected this exact policy",
                created_at=manifest.created_at,
            )
        )
    qualification_refs = [
        ref
        for ref in manifest.artifact_refs
        if ref.kind == "dynamics_operator_qualification_v24"
    ]
    if qualification_refs:
        qualification = DynamicsOperatorQualificationV24.model_validate(
            source_store.load_artifact(qualification_refs[0])
        )
        qualification.assert_sealed()
        node = EpistemicNodeV22.seal(
            node_id=f"dynamics_qualification_{qualification_refs[0].sha256[:16]}",
            node_kind="evolution_operator",
            artifact_hash=qualification_refs[0].sha256,
            intended_uses=["qualified_synthetic_dynamics_worldpack_v24_only"],
            created_at=manifest.created_at,
        )
        graph.add_node(node)
        assert node.node_hash is not None
        hashes["qualification"] = node.node_hash
        for index, (source, relation) in enumerate(
            [("memory_policy", "supports"), ("report", "justifies_use")], start=1
        ):
            graph.add_edge(
                EpistemicEdgeV22.seal(
                    edge_id=f"dynamics_qualification_edge_{index}_{hashes[source][:8]}",
                    source_node_hash=hashes[source],
                    target_node_hash=node.node_hash,
                    relation=relation,
                    rationale="exact Dynamics policy and passing report jointly bind qualification",
                    created_at=manifest.created_at,
                )
            )
    return hashes


def register_dynamics_estimator_run_v25(
    graph: EpistemicGraphStore,
    run_directory: str | Path,
    *,
    knowledge_bundle_node_hash: str | None = None,
    evolution_evidence_node_hash: str | None = None,
) -> dict[str, str]:
    """Register an exact V2.5 estimator ablation without widening its scope."""

    from .dynamics_estimator_ablation import (
        DynamicsEstimatorAblationReportV25,
        DynamicsEstimatorManifestV25,
        verify_estimator_worldpack_run_v25,
    )
    from .dynamics_integral import DynamicsEstimatorPolicyV25
    from .dynamics_stability import DynamicsStabilityReportV25

    directory = Path(run_directory).resolve()
    if not verify_estimator_worldpack_run_v25(directory):
        raise ValueError("V2.5 estimator run failed independent verification")
    source_store = RunStore.open_existing(directory)
    events = EpistemicGraphStore._read_events(source_store)
    committed = [
        ArtifactRef.model_validate(event["payload"])
        for event in events
        if event["event_type"] == "artifact_committed"
    ]
    manifest_refs = [
        ref for ref in committed if ref.kind == "dynamics_estimator_manifest_v25"
    ]
    if len(manifest_refs) != 1:
        raise RuntimeError("V2.5 estimator run needs exactly one manifest")
    manifest = DynamicsEstimatorManifestV25.model_validate(
        source_store.load_artifact(manifest_refs[0])
    )
    manifest.assert_sealed()

    def ref_one(kind: str) -> ArtifactRef:
        refs = [ref for ref in manifest.artifact_refs if ref.kind == kind]
        if len(refs) != 1:
            raise RuntimeError(f"V2.5 estimator manifest needs exactly one {kind}")
        source_store.load_artifact(refs[0])
        return refs[0]

    required = [
        "dynamics_estimator_spec_v25",
        "dynamics_point_policy_v25",
        "dynamics_integral_policy_v25",
        "private_dynamics_worldpack_v24",
        "dynamics_point_selections_v25",
        "dynamics_integral_selections_v25",
        "dynamics_estimator_report_v25",
    ]
    refs = {kind: ref_one(kind) for kind in required}
    point_policy = DynamicsEstimatorPolicyV25.model_validate(
        source_store.load_artifact(refs["dynamics_point_policy_v25"])
    )
    integral_policy = DynamicsEstimatorPolicyV25.model_validate(
        source_store.load_artifact(refs["dynamics_integral_policy_v25"])
    )
    report = DynamicsEstimatorAblationReportV25.model_validate(
        source_store.load_artifact(refs["dynamics_estimator_report_v25"])
    )
    for item in [point_policy, integral_policy, report]:
        item.assert_sealed()
    hashes: dict[str, str] = {}
    policy_plan = [
        (
            "point_policy",
            "dynamics_point_policy_v25",
            ["paired_point_derivative_baseline_only"],
        ),
        (
            "integral_policy",
            "dynamics_integral_policy_v25",
            ["candidate_until_estimator_qualification"],
        ),
    ]
    for label, ref_kind, uses in policy_plan:
        ref = refs[ref_kind]
        node = EpistemicNodeV22.seal(
            node_id=f"dynamics_v25_{label}_{ref.sha256[:16]}",
            node_kind="evolution_operator",
            artifact_hash=ref.sha256,
            intended_uses=uses,
            created_at=manifest.created_at,
        )
        graph.add_node(node)
        assert node.node_hash is not None
        hashes[label] = node.node_hash
    node_plan = [
        ("spec", "evidence", "dynamics_estimator_spec_v25", ["frozen_single_component_protocol"]),
        ("private", "evidence", "private_dynamics_worldpack_v24", ["hidden_benchmark_only"]),
        ("point", "evidence", "dynamics_point_selections_v25", ["paired_point_baseline"]),
        ("integral", "evidence", "dynamics_integral_selections_v25", ["paired_integral_treatment"]),
        ("report", "benchmark_report", "dynamics_estimator_report_v25", ["synthetic_estimator_scope_only"]),
    ]
    for label, kind, ref_kind, uses in node_plan:
        ref = refs[ref_kind]
        node = EpistemicNodeV22.seal(
            node_id=f"dynamics_v25_{label}_{ref.sha256[:16]}",
            node_kind=kind,
            artifact_hash=ref.sha256,
            intended_uses=uses,
            created_at=manifest.created_at,
        )
        graph.add_node(node)
        assert node.node_hash is not None
        hashes[label] = node.node_hash
    for index, source in enumerate(["spec", "private", "point", "integral"], start=1):
        graph.add_edge(
            EpistemicEdgeV22.seal(
                edge_id=f"dynamics_v25_report_evidence_{index}_{hashes[source][:8]}",
                source_node_hash=hashes[source],
                target_node_hash=hashes["report"],
                relation="derived_from",
                rationale=f"V2.5 estimator report independently replays from {source}",
                created_at=manifest.created_at,
            )
        )
    for label, source_hash in (
        ("knowledge", knowledge_bundle_node_hash),
        ("failure", evolution_evidence_node_hash),
    ):
        if source_hash is None:
            continue
        graph.project_state().node_by_hash(source_hash)
        for target in ["point_policy", "integral_policy"]:
            graph.add_edge(
                EpistemicEdgeV22.seal(
                    edge_id=f"dynamics_v25_{label}_{source_hash[:8]}_{hashes[target][:8]}",
                    source_node_hash=source_hash,
                    target_node_hash=hashes[target],
                    relation="learned_from",
                    rationale=f"exact V2.5 estimator policy is bound to {label} lineage",
                    created_at=manifest.created_at,
                )
            )
    stability_refs = [
        ref for ref in manifest.artifact_refs if ref.kind == "dynamics_stability_report_v25"
    ]
    if stability_refs:
        stability = DynamicsStabilityReportV25.model_validate(
            source_store.load_artifact(stability_refs[0])
        )
        stability.assert_sealed()
        node = EpistemicNodeV22.seal(
            node_id=f"dynamics_v25_stability_{stability_refs[0].sha256[:16]}",
            node_kind="validation_report",
            artifact_hash=stability_refs[0].sha256,
            intended_uses=["resampling_stability_not_calibrated_uncertainty"],
            created_at=manifest.created_at,
        )
        graph.add_node(node)
        assert node.node_hash is not None
        hashes["stability"] = node.node_hash
        for index, source in enumerate(
            ["spec", "private", "point", "integral", "report"], start=1
        ):
            graph.add_edge(
                EpistemicEdgeV22.seal(
                    edge_id=f"dynamics_v25_stability_evidence_{index}_{hashes[source][:8]}",
                    source_node_hash=hashes[source],
                    target_node_hash=hashes["stability"],
                    relation="derived_from",
                    rationale=f"V2.5 stability diagnostic independently replays from {source}",
                    created_at=manifest.created_at,
                )
            )
        if stability.status == "stability_gate_failed":
            graph.add_edge(
                EpistemicEdgeV22.seal(
                    edge_id=f"dynamics_v25_stability_refutes_{hashes['stability'][:8]}_{hashes['integral_policy'][:8]}",
                    source_node_hash=hashes["stability"],
                    target_node_hash=hashes["integral_policy"],
                    relation="refutes",
                    rationale="frozen dependence-aware stability gate rejected the exact integral policy",
                    created_at=manifest.created_at,
                )
            )
    if report.status == "candidate_rejected_estimator_v25":
        graph.add_edge(
            EpistemicEdgeV22.seal(
                edge_id=f"dynamics_v25_report_refutes_{hashes['report'][:8]}_{hashes['integral_policy'][:8]}",
                source_node_hash=hashes["report"],
                target_node_hash=hashes["integral_policy"],
                relation="refutes",
                rationale="pre-frozen V2.5 confirmation gate rejected the exact integral policy",
                created_at=manifest.created_at,
            )
        )
    qualification_refs = [
        ref
        for ref in manifest.artifact_refs
        if ref.kind == "dynamics_estimator_qualification_v25"
    ]
    if qualification_refs:
        node = EpistemicNodeV22.seal(
            node_id=f"dynamics_v25_qualification_{qualification_refs[0].sha256[:16]}",
            node_kind="evolution_operator",
            artifact_hash=qualification_refs[0].sha256,
            intended_uses=["qualified_synthetic_estimator_worldpack_v25_only"],
            created_at=manifest.created_at,
        )
        graph.add_node(node)
        assert node.node_hash is not None
        hashes["qualification"] = node.node_hash
        for index, source in enumerate(
            ["integral_policy", "report", "stability"], start=1
        ):
            graph.add_edge(
                EpistemicEdgeV22.seal(
                    edge_id=f"dynamics_v25_qualification_{index}_{hashes[source][:8]}",
                    source_node_hash=hashes[source],
                    target_node_hash=hashes["qualification"],
                    relation="justifies_use" if source != "integral_policy" else "supports",
                    rationale="exact policy, performance, and stability jointly bind qualification",
                    created_at=manifest.created_at,
                )
            )
    return hashes


def register_active_design_run_v26(
    graph: EpistemicGraphStore,
    run_directory: str | Path,
    *,
    knowledge_bundle_node_hash: str | None = None,
    prior_failure_node_hash: str | None = None,
) -> dict[str, str]:
    """Register the first V2.6 exploratory run without treating it as qualification."""

    from .dynamics_active_design import (
        ActiveDesignManifestV26,
        ActiveDesignPolicyV26,
        ActiveDesignReportV26,
        verify_active_design_worldpack_run_v26,
    )

    directory = Path(run_directory).resolve()
    if not verify_active_design_worldpack_run_v26(directory):
        raise ValueError("V2.6 active-design run failed independent verification")
    source_store = RunStore.open_existing(directory)
    events = EpistemicGraphStore._read_events(source_store)
    committed = [
        ArtifactRef.model_validate(event["payload"])
        for event in events
        if event["event_type"] == "artifact_committed"
    ]
    manifest_refs = [
        ref for ref in committed if ref.kind == "active_design_manifest_v26"
    ]
    if len(manifest_refs) != 1:
        raise RuntimeError("V2.6 active-design run needs exactly one manifest")
    manifest = ActiveDesignManifestV26.model_validate(
        source_store.load_artifact(manifest_refs[0])
    )
    manifest.assert_sealed()

    def ref_one(kind: str) -> ArtifactRef:
        refs = [ref for ref in manifest.artifact_refs if ref.kind == kind]
        if len(refs) != 1:
            raise RuntimeError(f"V2.6 active-design manifest needs exactly one {kind}")
        source_store.load_artifact(refs[0])
        return refs[0]

    required = [
        "active_design_spec_v26",
        "active_design_baseline_policy_v26",
        "active_design_candidate_policy_v26",
        "private_active_design_worldpack_v26",
        "active_design_baseline_bundle_v26",
        "active_design_candidate_bundle_v26",
        "active_design_report_v26",
    ]
    refs = {kind: ref_one(kind) for kind in required}
    baseline_policy = ActiveDesignPolicyV26.model_validate(
        source_store.load_artifact(refs["active_design_baseline_policy_v26"])
    )
    active_policy = ActiveDesignPolicyV26.model_validate(
        source_store.load_artifact(refs["active_design_candidate_policy_v26"])
    )
    report = ActiveDesignReportV26.model_validate(
        source_store.load_artifact(refs["active_design_report_v26"])
    )
    for item in (baseline_policy, active_policy, report):
        item.assert_sealed()
    hashes: dict[str, str] = {}
    node_plan = [
        (
            "baseline_policy",
            "evolution_operator",
            "active_design_baseline_policy_v26",
            ["paired_random_safe_catalog_baseline_only"],
        ),
        (
            "active_policy",
            "evolution_operator",
            "active_design_candidate_policy_v26",
            ["candidate_until_active_design_qualification"],
        ),
        (
            "spec",
            "evidence",
            "active_design_spec_v26",
            ["frozen_action_selection_ablation"],
        ),
        (
            "private",
            "evidence",
            "private_active_design_worldpack_v26",
            ["hidden_synthetic_initial_condition_benchmark_only"],
        ),
        (
            "baseline",
            "evidence",
            "active_design_baseline_bundle_v26",
            ["paired_random_action_receipts"],
        ),
        (
            "active",
            "evidence",
            "active_design_candidate_bundle_v26",
            ["paired_sequential_action_receipts"],
        ),
        (
            "report",
            "failure_signature",
            "active_design_report_v26",
            ["exploratory_relative_effect_metric_failure_only"],
        ),
    ]
    for label, kind, ref_kind, uses in node_plan:
        ref = refs[ref_kind]
        node = EpistemicNodeV22.seal(
            node_id=f"active_design_v26_{label}_{ref.sha256[:16]}",
            node_kind=kind,
            artifact_hash=ref.sha256,
            intended_uses=uses,
            created_at=manifest.created_at,
        )
        graph.add_node(node)
        assert node.node_hash is not None
        hashes[label] = node.node_hash
    for index, source in enumerate(
        ["spec", "private", "baseline", "active"], start=1
    ):
        graph.add_edge(
            EpistemicEdgeV22.seal(
                edge_id=f"active_design_v26_report_{index}_{hashes[source][:8]}",
                source_node_hash=hashes[source],
                target_node_hash=hashes["report"],
                relation="derived_from",
                rationale=f"V2.6 exploratory report independently replays from {source}",
                created_at=manifest.created_at,
            )
        )
    for label, source_hash in (
        ("knowledge", knowledge_bundle_node_hash),
        ("failure", prior_failure_node_hash),
    ):
        if source_hash is None:
            continue
        graph.project_state().node_by_hash(source_hash)
        for target in ("baseline_policy", "active_policy"):
            graph.add_edge(
                EpistemicEdgeV22.seal(
                    edge_id=f"active_design_v26_{label}_{source_hash[:8]}_{hashes[target][:8]}",
                    source_node_hash=source_hash,
                    target_node_hash=hashes[target],
                    relation="learned_from",
                    rationale=f"exact V2.6 action policy is bound to {label} lineage",
                    created_at=manifest.created_at,
                )
            )
    return hashes


def register_active_design_effect_run_v261(
    graph: EpistemicGraphStore,
    run_directory: str | Path,
    *,
    knowledge_bundle_node_hash: str | None = None,
    prior_failure_node_hash: str | None = None,
    metric_failure_node_hash: str | None = None,
    prior_effect_node_hash: str | None = None,
) -> dict[str, str]:
    """Register an exact V2.6.1 effect run with its narrow synthetic scope."""

    from .dynamics_active_design import ActiveDesignPolicyV26
    from .dynamics_active_design_effect import (
        ActiveDesignEffectManifestV261,
        ActiveDesignEffectReportV261,
        verify_active_design_effect_run_v261,
    )

    directory = Path(run_directory).resolve()
    if not verify_active_design_effect_run_v261(directory):
        raise ValueError("V2.6.1 active-design run failed independent verification")
    source_store = RunStore.open_existing(directory)
    events = EpistemicGraphStore._read_events(source_store)
    committed = [
        ArtifactRef.model_validate(event["payload"])
        for event in events
        if event["event_type"] == "artifact_committed"
    ]
    manifest_refs = [
        ref for ref in committed if ref.kind == "active_design_effect_manifest_v261"
    ]
    if len(manifest_refs) != 1:
        raise RuntimeError("V2.6.1 active-design run needs exactly one manifest")
    manifest = ActiveDesignEffectManifestV261.model_validate(
        source_store.load_artifact(manifest_refs[0])
    )
    manifest.assert_sealed()

    def ref_one(kind: str) -> ArtifactRef:
        refs = [ref for ref in manifest.artifact_refs if ref.kind == kind]
        if len(refs) != 1:
            raise RuntimeError(f"V2.6.1 manifest needs exactly one {kind}")
        source_store.load_artifact(refs[0])
        return refs[0]

    required = [
        "active_design_spec_v26",
        "active_design_effect_protocol_v261",
        "active_design_baseline_policy_v26",
        "active_design_candidate_policy_v26",
        "private_active_design_worldpack_v26",
        "active_design_baseline_bundle_v26",
        "active_design_candidate_bundle_v26",
        "active_design_effect_report_v261",
    ]
    refs = {kind: ref_one(kind) for kind in required}
    baseline_policy = ActiveDesignPolicyV26.model_validate(
        source_store.load_artifact(refs["active_design_baseline_policy_v26"])
    )
    active_policy = ActiveDesignPolicyV26.model_validate(
        source_store.load_artifact(refs["active_design_candidate_policy_v26"])
    )
    report = ActiveDesignEffectReportV261.model_validate(
        source_store.load_artifact(refs["active_design_effect_report_v261"])
    )
    for item in (baseline_policy, active_policy, report):
        item.assert_sealed()
    hashes: dict[str, str] = {}
    node_plan = [
        (
            "baseline_policy",
            "evolution_operator",
            "active_design_baseline_policy_v26",
            ["paired_random_safe_catalog_baseline_only"],
        ),
        (
            "active_policy",
            "evolution_operator",
            "active_design_candidate_policy_v26",
            ["candidate_until_active_design_effect_qualification"],
        ),
        (
            "spec",
            "evidence",
            "active_design_spec_v26",
            ["frozen_action_selection_ablation"],
        ),
        (
            "protocol",
            "evolution_operator",
            "active_design_effect_protocol_v261",
            ["bounded_absolute_effect_measure_only"],
        ),
        (
            "private",
            "evidence",
            "private_active_design_worldpack_v26",
            ["hidden_synthetic_initial_condition_benchmark_only"],
        ),
        (
            "baseline",
            "evidence",
            "active_design_baseline_bundle_v26",
            ["paired_random_action_receipts"],
        ),
        (
            "active",
            "evidence",
            "active_design_candidate_bundle_v26",
            ["paired_sequential_action_receipts"],
        ),
        (
            "report",
            "benchmark_report",
            "active_design_effect_report_v261",
            ["synthetic_safe_initial_condition_design_scope_only"],
        ),
    ]
    for label, kind, ref_kind, uses in node_plan:
        ref = refs[ref_kind]
        node = EpistemicNodeV22.seal(
            node_id=f"active_design_v261_{label}_{ref.sha256[:16]}",
            node_kind=kind,
            artifact_hash=ref.sha256,
            intended_uses=uses,
            created_at=manifest.created_at,
        )
        assert node.node_hash is not None
        existing = {
            item.node_id: item for item in graph.project_state().nodes
        }.get(node.node_id)
        if existing is None:
            graph.add_node(node)
            hashes[label] = node.node_hash
        else:
            if (
                existing.node_kind != node.node_kind
                or existing.artifact_hash != node.artifact_hash
                or existing.intended_uses != node.intended_uses
            ):
                raise ValueError("existing V2.6.1 node_id has another interpretation")
            assert existing.node_hash is not None
            hashes[label] = existing.node_hash
    for index, source in enumerate(
        ["spec", "protocol", "private", "baseline", "active"], start=1
    ):
        graph.add_edge(
            EpistemicEdgeV22.seal(
                edge_id=f"active_design_v261_report_{index}_{hashes[source][:8]}",
                source_node_hash=hashes[source],
                target_node_hash=hashes["report"],
                relation="derived_from",
                rationale=f"V2.6.1 effect report independently replays from {source}",
                created_at=manifest.created_at,
            )
        )
    for label, source_hash, targets in (
        ("knowledge", knowledge_bundle_node_hash, ("baseline_policy", "active_policy")),
        ("failure", prior_failure_node_hash, ("baseline_policy", "active_policy")),
        ("metric_failure", metric_failure_node_hash, ("protocol",)),
        ("prior_effect", prior_effect_node_hash, ("protocol",)),
    ):
        if source_hash is None:
            continue
        graph.project_state().node_by_hash(source_hash)
        for target in targets:
            graph.add_edge(
                EpistemicEdgeV22.seal(
                    edge_id=f"active_design_v261_{label}_{source_hash[:8]}_{hashes[target][:8]}",
                    source_node_hash=source_hash,
                    target_node_hash=hashes[target],
                    relation="learned_from",
                    rationale=f"exact V2.6.1 component is bound to {label} lineage",
                    created_at=manifest.created_at,
                )
            )
    if report.status == "candidate_rejected_active_design_v261":
        graph.add_edge(
            EpistemicEdgeV22.seal(
                edge_id=f"active_design_v261_refutes_{hashes['report'][:8]}_{hashes['active_policy'][:8]}",
                source_node_hash=hashes["report"],
                target_node_hash=hashes["active_policy"],
                relation="refutes",
                rationale="pre-frozen V2.6.1 confirmation gate rejected the exact active policy",
                created_at=manifest.created_at,
            )
        )
    qualification_refs = [
        ref
        for ref in manifest.artifact_refs
        if ref.kind == "active_design_effect_qualification_v261"
    ]
    if qualification_refs:
        node = EpistemicNodeV22.seal(
            node_id=f"active_design_v261_qualification_{qualification_refs[0].sha256[:16]}",
            node_kind="evolution_operator",
            artifact_hash=qualification_refs[0].sha256,
            intended_uses=["qualified_synthetic_safe_initial_condition_design_v261_only"],
            created_at=manifest.created_at,
        )
        graph.add_node(node)
        assert node.node_hash is not None
        hashes["qualification"] = node.node_hash
        for index, source in enumerate(
            ["active_policy", "protocol", "report"], start=1
        ):
            graph.add_edge(
                EpistemicEdgeV22.seal(
                    edge_id=f"active_design_v261_qualification_{index}_{hashes[source][:8]}",
                    source_node_hash=hashes[source],
                    target_node_hash=hashes["qualification"],
                    relation="supports" if source == "active_policy" else "justifies_use",
                    rationale="exact policy, bounded effect protocol, and passing confirmation bind qualification",
                    created_at=manifest.created_at,
                )
            )
    return hashes


def _register_replayed_run_nodes(
    graph: EpistemicGraphStore,
    *,
    refs: dict[str, ArtifactRef],
    node_plan: list[tuple[str, NodeKind, str, list[str]]],
    node_prefix: str,
    created_at: datetime,
) -> dict[str, str]:
    """Add or reuse immutable nodes for a replay-verified external run."""

    hashes: dict[str, str] = {}
    for label, kind, ref_kind, uses in node_plan:
        ref = refs[ref_kind]
        node = EpistemicNodeV22.seal(
            node_id=f"{node_prefix}_{label}_{ref.sha256[:16]}",
            node_kind=kind,
            artifact_hash=ref.sha256,
            intended_uses=uses,
            created_at=created_at,
        )
        assert node.node_hash is not None
        existing = {
            item.node_id: item for item in graph.project_state().nodes
        }.get(node.node_id)
        if existing is None:
            graph.add_node(node)
            hashes[label] = node.node_hash
        else:
            if (
                existing.node_kind != node.node_kind
                or existing.artifact_hash != node.artifact_hash
                or existing.intended_uses != node.intended_uses
            ):
                raise ValueError("existing epistemic-loop node has another interpretation")
            assert existing.node_hash is not None
            hashes[label] = existing.node_hash
    return hashes


def _add_epistemic_edge_if_absent(
    graph: EpistemicGraphStore, edge: EpistemicEdgeV22
) -> None:
    edge.assert_sealed()
    state = graph.project_state()
    existing = {item.edge_id: item for item in state.edges}.get(edge.edge_id)
    if existing is None:
        graph.add_edge(edge)
        return
    if (
        existing.source_node_hash != edge.source_node_hash
        or existing.target_node_hash != edge.target_node_hash
        or existing.relation != edge.relation
        or existing.rationale != edge.rationale
    ):
        raise ValueError("existing epistemic-loop edge has another interpretation")


def register_epistemic_loop_run_v30(
    graph: EpistemicGraphStore,
    run_directory: str | Path,
) -> dict[str, str]:
    """Register the one-step V3.0 failure without upgrading its candidate."""

    from fma.v3.epistemic_loop import (
        EpistemicActionPolicyV30,
        EpistemicLoopManifestV30,
        ProblemReformulationReportV30,
        verify_problem_reformulation_run_v30,
    )

    directory = Path(run_directory).resolve()
    if not verify_problem_reformulation_run_v30(directory):
        raise ValueError("V3.0 epistemic-loop run failed independent verification")
    source_store = RunStore.open_existing(directory)
    events = EpistemicGraphStore._read_events(source_store)
    committed = [
        ArtifactRef.model_validate(event["payload"])
        for event in events
        if event["event_type"] == "artifact_committed"
    ]
    manifests = [
        ref for ref in committed if ref.kind == "epistemic_loop_manifest_v30"
    ]
    if len(manifests) != 1:
        raise RuntimeError("V3.0 run needs exactly one manifest")
    manifest = EpistemicLoopManifestV30.model_validate(
        source_store.load_artifact(manifests[0])
    )
    manifest.assert_sealed()

    def ref_one(kind: str) -> ArtifactRef:
        selected = [ref for ref in manifest.artifact_refs if ref.kind == kind]
        if len(selected) != 1:
            raise RuntimeError(f"V3.0 manifest needs exactly one {kind}")
        source_store.load_artifact(selected[0])
        return selected[0]

    kinds = [
        "epistemic_loop_spec_v30",
        "epistemic_loop_baseline_policy_v30",
        "epistemic_loop_candidate_policy_v30",
        "private_epistemic_worldpack_v30",
        "epistemic_loop_baseline_bundle_v30",
        "epistemic_loop_candidate_bundle_v30",
        "epistemic_loop_report_v30",
    ]
    refs = {kind: ref_one(kind) for kind in kinds}
    candidate = EpistemicActionPolicyV30.model_validate(
        source_store.load_artifact(refs["epistemic_loop_candidate_policy_v30"])
    )
    report = ProblemReformulationReportV30.model_validate(
        source_store.load_artifact(refs["epistemic_loop_report_v30"])
    )
    candidate.assert_sealed()
    report.assert_sealed()
    hashes = _register_replayed_run_nodes(
        graph,
        refs=refs,
        node_plan=[
            ("spec", "evidence", "epistemic_loop_spec_v30", ["frozen_one_step_problem_reformulation_ablation"]),
            ("baseline_policy", "evolution_operator", "epistemic_loop_baseline_policy_v30", ["fixed_contract_more_data_baseline_only"]),
            ("candidate_policy", "evolution_operator", "epistemic_loop_candidate_policy_v30", ["candidate_rejected_until_new_confirmation"]),
            ("private", "evidence", "private_epistemic_worldpack_v30", ["hidden_synthetic_capacity_semantics_only"]),
            ("baseline", "evidence", "epistemic_loop_baseline_bundle_v30", ["one_step_fixed_contract_receipts"]),
            ("candidate", "evidence", "epistemic_loop_candidate_bundle_v30", ["one_step_reformulation_receipts"]),
            ("report", "failure_signature", "epistemic_loop_report_v30", ["one_step_information_tradeoff_failure_only"]),
        ],
        node_prefix="epistemic_loop_v30",
        created_at=manifest.created_at,
    )
    for index, source in enumerate(
        ["spec", "private", "baseline", "candidate"], start=1
    ):
        _add_epistemic_edge_if_absent(
            graph,
            EpistemicEdgeV22.seal(
                edge_id=f"epistemic_loop_v30_report_{index}_{hashes[source][:8]}",
                source_node_hash=hashes[source],
                target_node_hash=hashes["report"],
                relation="derived_from",
                rationale=f"V3.0 failure report independently replays from {source}",
                created_at=manifest.created_at,
            ),
        )
    if not report.gate_results.get("negative_transfer_rate_gate", True):
        _add_epistemic_edge_if_absent(
            graph,
            EpistemicEdgeV22.seal(
                edge_id=f"epistemic_loop_v30_refutes_{hashes['report'][:8]}_{hashes['candidate_policy'][:8]}",
                source_node_hash=hashes["report"],
                target_node_hash=hashes["candidate_policy"],
                relation="refutes",
                rationale="one-step clarification policy failed the frozen negative-transfer gate",
                created_at=manifest.created_at,
            ),
        )
    return hashes


def register_epistemic_loop_run_v301(
    graph: EpistemicGraphStore,
    run_directory: str | Path,
    *,
    prior_failure_node_hash: str,
) -> dict[str, str]:
    """Register a V3.0.1 sequential run and its narrow qualification, if any."""

    from fma.v3.epistemic_loop_v301 import (
        EpistemicLoopManifestV301,
        ProblemReformulationReportV301,
        SequentialEpistemicPolicyV301,
        verify_problem_reformulation_run_v301,
    )

    directory = Path(run_directory).resolve()
    if not verify_problem_reformulation_run_v301(directory):
        raise ValueError("V3.0.1 epistemic-loop run failed independent verification")
    graph.project_state().node_by_hash(prior_failure_node_hash)
    source_store = RunStore.open_existing(directory)
    events = EpistemicGraphStore._read_events(source_store)
    committed = [
        ArtifactRef.model_validate(event["payload"])
        for event in events
        if event["event_type"] == "artifact_committed"
    ]
    manifests = [
        ref for ref in committed if ref.kind == "epistemic_loop_manifest_v301"
    ]
    if len(manifests) != 1:
        raise RuntimeError("V3.0.1 run needs exactly one manifest")
    manifest = EpistemicLoopManifestV301.model_validate(
        source_store.load_artifact(manifests[0])
    )
    manifest.assert_sealed()

    def ref_one(kind: str) -> ArtifactRef:
        selected = [ref for ref in manifest.artifact_refs if ref.kind == kind]
        if len(selected) != 1:
            raise RuntimeError(f"V3.0.1 manifest needs exactly one {kind}")
        source_store.load_artifact(selected[0])
        return selected[0]

    kinds = [
        "epistemic_loop_spec_v301",
        "epistemic_loop_baseline_policy_v301",
        "epistemic_loop_candidate_policy_v301",
        "private_epistemic_worldpack_v301",
        "epistemic_loop_baseline_bundle_v301",
        "epistemic_loop_candidate_bundle_v301",
        "epistemic_loop_report_v301",
    ]
    refs = {kind: ref_one(kind) for kind in kinds}
    report = ProblemReformulationReportV301.model_validate(
        source_store.load_artifact(refs["epistemic_loop_report_v301"])
    )
    candidate = SequentialEpistemicPolicyV301.model_validate(
        source_store.load_artifact(refs["epistemic_loop_candidate_policy_v301"])
    )
    report.assert_sealed()
    candidate.assert_sealed()
    node_plan: list[tuple[str, NodeKind, str, list[str]]] = [
        ("spec", "evolution_operator", "epistemic_loop_spec_v301", ["failure_bound_two_step_epistemic_protocol"]),
        ("baseline_policy", "evolution_operator", "epistemic_loop_baseline_policy_v301", ["two_batch_fixed_contract_baseline_only"]),
        ("candidate_policy", "evolution_operator", "epistemic_loop_candidate_policy_v301", ["clarify_then_collect_exact_policy"]),
        ("private", "evidence", "private_epistemic_worldpack_v301", ["hidden_synthetic_capacity_semantics_only"]),
        ("baseline", "evidence", "epistemic_loop_baseline_bundle_v301", ["paired_two_cost_fixed_contract_receipts"]),
        ("candidate", "evidence", "epistemic_loop_candidate_bundle_v301", ["paired_two_cost_reformulation_receipts"]),
        ("report", "benchmark_report", "epistemic_loop_report_v301", ["synthetic_sequential_problem_reformulation_scope_only"]),
    ]
    qualification_refs = [
        ref
        for ref in manifest.artifact_refs
        if ref.kind == "epistemic_loop_qualification_v301"
    ]
    if qualification_refs:
        refs["epistemic_loop_qualification_v301"] = qualification_refs[0]
        node_plan.append(
            ("qualification", "evolution_operator", "epistemic_loop_qualification_v301", ["qualified_synthetic_sequential_problem_reformulation_v301_only"])
        )
    hashes = _register_replayed_run_nodes(
        graph,
        refs=refs,
        node_plan=node_plan,
        node_prefix="epistemic_loop_v301",
        created_at=manifest.created_at,
    )
    for index, source in enumerate(
        ["spec", "private", "baseline", "candidate"], start=1
    ):
        _add_epistemic_edge_if_absent(
            graph,
            EpistemicEdgeV22.seal(
                edge_id=f"epistemic_loop_v301_report_{index}_{hashes[source][:8]}",
                source_node_hash=hashes[source],
                target_node_hash=hashes["report"],
                relation="derived_from",
                rationale=f"V3.0.1 report independently replays from {source}",
                created_at=manifest.created_at,
            ),
        )
    for target in ("spec", "baseline_policy", "candidate_policy"):
        _add_epistemic_edge_if_absent(
            graph,
            EpistemicEdgeV22.seal(
                edge_id=f"epistemic_loop_v301_failure_{prior_failure_node_hash[:8]}_{hashes[target][:8]}",
                source_node_hash=prior_failure_node_hash,
                target_node_hash=hashes[target],
                relation="learned_from",
                rationale="V3.0.1 changes only the action horizon after the V3.0 failure",
                created_at=manifest.created_at,
            ),
        )
    if "qualification" in hashes:
        for index, source in enumerate(
            ["candidate_policy", "spec", "report"], start=1
        ):
            _add_epistemic_edge_if_absent(
                graph,
                EpistemicEdgeV22.seal(
                    edge_id=f"epistemic_loop_v301_qualification_{index}_{hashes[source][:8]}",
                    source_node_hash=hashes[source],
                    target_node_hash=hashes["qualification"],
                    relation="supports" if source == "candidate_policy" else "justifies_use",
                    rationale="exact sequential policy, frozen protocol, and confirmation bind the narrow qualification",
                    created_at=manifest.created_at,
                ),
            )
    return hashes
