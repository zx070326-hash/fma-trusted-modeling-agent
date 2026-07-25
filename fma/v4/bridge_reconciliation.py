from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.v2.schemas import Identifier, Sha256, _assert_timezone

from .graph_loop import CrossLayerBridgeV40, GraphLoopStoreV40


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class BridgeReconciliationReceiptV40(StrictModel):
    schema_version: Literal["4.0"] = "4.0"
    reconciliation_id: Identifier
    bridge_hash: Sha256
    direction: Literal[
        "development_release_to_modeling_runtime",
        "modeling_failure_to_development_issue",
    ]
    source_graph_id: Identifier
    source_snapshot_hash: Sha256
    source_node_hash: Sha256
    source_node_status: str
    active_patch_node_hashes: list[Sha256]
    target_graph_id: Identifier
    target_base_snapshot_hash: Sha256
    target_node_hash: Sha256
    decision: Literal["valid", "invalid"]
    reasons: list[Annotated[str, Field(min_length=3)]] = Field(min_length=1)
    checked_by: Literal["harness", "verifier", "human"]
    checked_at: datetime
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "BridgeReconciliationReceiptV40":
        _assert_timezone(self.checked_at, "checked_at")
        if self.active_patch_node_hashes != sorted(set(self.active_patch_node_hashes)):
            raise ValueError("active_patch_node_hashes must be sorted and unique")
        if self.decision == "valid" and self.reasons != ["source_binding_current"]:
            raise ValueError("valid bridge receipt must use the exact success reason")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match bridge reconciliation")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("bridge reconciliation receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "BridgeReconciliationReceiptV40":
        data.setdefault("checked_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


def _artifact_refs(store: GraphLoopStoreV40, kind: str) -> list[ArtifactRef]:
    refs: list[ArtifactRef] = []
    for line in store.store.event_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event["event_type"] != "artifact_committed":
            continue
        ref = ArtifactRef.model_validate(event["payload"])
        if ref.kind == kind:
            refs.append(ref)
    return refs


def bridge_for_target_node_v40(
    target: GraphLoopStoreV40,
    target_node_hash: str,
) -> CrossLayerBridgeV40:
    node = target.project_state().node_by_hash(target_node_hash)
    matches: list[CrossLayerBridgeV40] = []
    for ref in _artifact_refs(target, "cross_layer_bridge_v40"):
        bridge = CrossLayerBridgeV40.model_validate(target.store.load_artifact(ref))
        bridge.assert_sealed()
        if bridge.bridge_hash == node.artifact_hash:
            matches.append(bridge)
    if len(matches) != 1:
        raise RuntimeError("runtime node needs exactly one committed bridge")
    return matches[0]


def _source_binding_status(
    source: GraphLoopStoreV40,
    bridge: CrossLayerBridgeV40,
) -> tuple[bool, list[str], list[str], str]:
    state = source.project_state()
    reasons: list[str] = []
    active_patches: list[str] = []
    try:
        node = state.node_by_hash(bridge.source_node_hash)
        status = state.snapshot.node_statuses[bridge.source_node_hash]
    except KeyError:
        return False, ["source_node_missing"], [], "missing"
    if bridge.source_graph_id != source.contract.graph_id:
        reasons.append("source_graph_changed")
    if bridge.direction == "development_release_to_modeling_runtime":
        if source.contract.layer != "development" or node.node_kind != "release":
            reasons.append("source_is_not_development_release")
        if status != "succeeded":
            reasons.append("release_is_not_succeeded")
        active_patches = sorted(
            edge.source_node_hash
            for edge in state.edges
            if edge.target_node_hash == node.node_hash
            and edge.relation == "requires_active"
            and state.node_by_hash(edge.source_node_hash).node_kind == "patch"
            and state.snapshot.node_statuses[edge.source_node_hash] == "active"
        )
        if not active_patches:
            reasons.append("active_patch_binding_missing")
    else:
        if source.contract.layer != "modeling" or node.node_kind != "failure":
            reasons.append("source_is_not_modeling_failure")
        if status != "succeeded":
            reasons.append("failure_record_is_not_succeeded")
    return not reasons, reasons or ["source_binding_current"], active_patches, status


def reconcile_cross_layer_bridge_v40(
    source: GraphLoopStoreV40,
    target: GraphLoopStoreV40,
    target_node_hash: str,
    *,
    reconciliation_id: str,
    checked_by: Literal["harness", "verifier", "human"],
    checked_at: datetime | None = None,
) -> tuple[BridgeReconciliationReceiptV40, ArtifactRef]:
    """Recheck a bridge against the source tip and persist the exact verdict."""

    target_state = target.project_state()
    bridge = bridge_for_target_node_v40(target, target_node_hash)
    if bridge.target_graph_id != target.contract.graph_id:
        raise ValueError("bridge target graph differs")
    valid, reasons, active_patches, source_status = _source_binding_status(source, bridge)
    receipt = BridgeReconciliationReceiptV40.seal(
        reconciliation_id=reconciliation_id,
        bridge_hash=bridge.bridge_hash,
        direction=bridge.direction,
        source_graph_id=source.contract.graph_id,
        source_snapshot_hash=source.project_state().snapshot.snapshot_hash,
        source_node_hash=bridge.source_node_hash,
        source_node_status=source_status,
        active_patch_node_hashes=active_patches,
        target_graph_id=target.contract.graph_id,
        target_base_snapshot_hash=target_state.snapshot.snapshot_hash,
        target_node_hash=target_node_hash,
        decision="valid" if valid else "invalid",
        reasons=reasons,
        checked_by=checked_by,
        checked_at=checked_at or datetime.now(timezone.utc),
    )
    ref = target.put_output("bridge_reconciliation_receipt_v40", receipt)
    if not valid and checked_by in {"verifier", "human"}:
        target.revoke_node(
            target_node_hash,
            authority=checked_by,
            reason="cross-layer bridge invalid: " + ", ".join(reasons),
            revocation_id=f"revoke_{reconciliation_id}",
            revoked_at=receipt.checked_at,
        )
    return receipt, ref


def assert_current_bridge_reconciliation_v40(
    source: GraphLoopStoreV40,
    target: GraphLoopStoreV40,
    target_node_hash: str,
    reconciliation_ref: ArtifactRef,
) -> BridgeReconciliationReceiptV40:
    if reconciliation_ref.kind != "bridge_reconciliation_receipt_v40":
        raise ValueError("runtime release needs a bridge reconciliation artifact")
    receipt = BridgeReconciliationReceiptV40.model_validate(
        target.store.load_artifact(reconciliation_ref)
    )
    receipt.assert_sealed()
    bridge = bridge_for_target_node_v40(target, target_node_hash)
    source_state = source.project_state()
    target_state = target.project_state()
    valid, reasons, active_patches, source_status = _source_binding_status(source, bridge)
    if (
        receipt.decision != "valid"
        or not valid
        or receipt.bridge_hash != bridge.bridge_hash
        or receipt.source_graph_id != source.contract.graph_id
        or receipt.source_snapshot_hash != source_state.snapshot.snapshot_hash
        or receipt.source_node_status != source_status
        or receipt.active_patch_node_hashes != active_patches
        or receipt.target_graph_id != target.contract.graph_id
        or receipt.target_base_snapshot_hash != target_state.snapshot.snapshot_hash
        or receipt.target_node_hash != target_node_hash
        or reasons != ["source_binding_current"]
    ):
        raise RuntimeError("bridge reconciliation is stale or invalid")
    return receipt
