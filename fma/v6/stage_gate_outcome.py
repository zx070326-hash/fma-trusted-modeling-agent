"""Authenticated negative workflow-gate outcomes for V6.6.

V5 gate certificates authenticate only an OPEN transition.  A verifier
rejection previously remained a pending graph node and was therefore easy for
an outer campaign to misread as an execution failure.  This additive overlay
records a BLOCKED decision as the gate node's terminal graph outcome while
preserving the important authority boundary:

* the reviewer supplies an untrusted verdict;
* the harness binds the exact manifest, checks, review receipts, and typed
  finding set;
* only the verifier-owned workspace key may authenticate and record the
  negative graph outcome.

The receipt establishes workflow history only.  It is neither scientific
qualification nor permission for a real-world action.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256, _assert_timezone
from fma.v5.stage_workspace import (
    POLICIES,
    StageWorkspaceError,
    StageWorkspaceV50,
)
from fma.v5.workspace_schemas import (
    CheckResultV50,
    IndependentReviewReceiptV50,
    StageArtifactManifestV50,
    StageId,
)
from fma.v6.stage_review_recovery import S0ReviewFindingSetV66


class StageGateOutcomeV66(StrictModel):
    """One verifier-authenticated, non-opening workflow-gate decision."""

    schema_version: Literal["6.6-stage-gate-outcome"] = (
        "6.6-stage-gate-outcome"
    )
    workspace_spec_hash: Sha256
    stage: StageId
    attempt: Annotated[int, Field(ge=1)]
    decision: Literal["BLOCKED"]
    manifest_hash: Sha256
    policy_hash: Sha256
    graph_gate_node_hash: Sha256
    evaluator_epoch: Identifier
    check_result_hashes: list[Sha256]
    review_receipt_hashes: Annotated[list[Sha256], Field(min_length=1)]
    finding_set_hash: Sha256
    reason_codes: Annotated[list[Identifier], Field(min_length=1)]
    authority_key_id: Identifier
    issued_at: datetime
    authority_auth_tag: Sha256 | None = None
    outcome_hash: Sha256 | None = None
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_outcome(self) -> "StageGateOutcomeV66":
        _assert_timezone(self.issued_at, "issued_at")
        for label, values in (
            ("check_result_hashes", self.check_result_hashes),
            ("review_receipt_hashes", self.review_receipt_hashes),
            ("reason_codes", self.reason_codes),
        ):
            if values != sorted(set(values)):
                raise ValueError(f"{label} must be sorted and unique")
        if self.authority_auth_tag and not self.outcome_hash:
            raise ValueError(
                "authenticated stage-gate outcome requires outcome_hash"
            )
        if self.outcome_hash and self.outcome_hash != self.content_hash():
            raise ValueError("stage-gate outcome hash differs")
        return self

    def unsigned_hash(self) -> str:
        return sha256_value(
            self.model_dump(
                mode="json",
                exclude={"authority_auth_tag", "outcome_hash"},
            )
        )

    def content_hash(self) -> str:
        return sha256_value(
            self.model_dump(mode="json", exclude={"outcome_hash"})
        )

    def assert_sealed(self) -> None:
        if (
            not self.authority_auth_tag
            or not self.outcome_hash
            or self.outcome_hash != self.content_hash()
        ):
            raise ValueError("stage-gate outcome is not sealed")


def record_blocked_stage_gate_v66(
    workspace: StageWorkspaceV50,
    *,
    stage: StageId,
    manifest_hash: str,
    policy_hash: str,
    check_result_hashes: list[str],
    review_receipt_hashes: list[str],
    finding_set_hash: str,
    reason_codes: list[str],
) -> StageGateOutcomeV66:
    """Authenticate and atomically terminate the current gate as BLOCKED."""

    if stage != "S0":
        raise ValueError("V6.6 typed blocked-gate outcomes are restricted to S0")
    gate = workspace._binding(stage, "gate")
    manifest = workspace._manifest_for_stage(stage)
    state = workspace.graph.project_state()
    if gate.node_hash not in state.snapshot.frontier_node_hashes:
        raise StageWorkspaceError(f"{stage} gate is not on the graph frontier")
    if str(manifest.manifest_hash) != manifest_hash:
        raise PermissionError("blocked gate manifest binding differs")
    if not workspace._manifest_is_current(manifest):
        raise PermissionError("cannot block a gate for a stale manifest")
    if policy_hash != POLICIES[stage].policy_hash:
        raise PermissionError("blocked gate policy binding differs")
    checks = workspace._latest_checks(stage, manifest_hash)
    expected_check_hashes = sorted(
        str(item.result_hash)
        for item in checks.values()
        if item.result_hash is not None
    )
    if sorted(set(check_result_hashes)) != expected_check_hashes:
        raise PermissionError("blocked gate check snapshot differs")
    reviews = workspace._latest_reviews(stage, manifest_hash)
    expected_review_hashes = sorted(
        str(item.receipt_hash)
        for item in reviews.values()
        if item.receipt_hash is not None
    )
    if (
        not review_receipt_hashes
        or sorted(set(review_receipt_hashes)) != expected_review_hashes
        or any(
            role not in reviews for role in POLICIES[stage].required_review_roles
        )
    ):
        raise ValueError("blocked gate requires independent review evidence")
    if any(
        reviews[role].verdict != "REJECT"
        for role in POLICIES[stage].required_review_roles
    ):
        raise PermissionError(
            "blocked S0 gate requires an authenticated rejecting review"
        )
    gate_evaluation = workspace.evaluate_gate(stage)
    if gate_evaluation.decision != "BLOCKED":
        raise PermissionError(
            "blocked S0 outcome differs from verifier gate evaluation"
        )
    if reason_codes != ["independent_review_rejected"]:
        raise PermissionError("blocked S0 reason code is code-owned")
    if finding_set_hash not in workspace._committed_artifact_hashes():
        raise PermissionError("finding set is not committed workspace evidence")
    try:
        finding_set = S0ReviewFindingSetV66.model_validate(
            workspace._artifact_payload_by_hash(finding_set_hash)
        )
        finding_set.assert_sealed()
    except (TypeError, ValueError) as exc:
        raise PermissionError(
            "blocked S0 outcome requires a typed sealed finding set"
        ) from exc
    expected_review = reviews["referee"]
    if (
        finding_set.task_id != workspace.spec.workspace_id
        or finding_set.attempt_id != f"s0-a{manifest.attempt}"
        or finding_set.reviewer_receipt_hash
        != expected_review.receipt_hash
        or sorted(item.finding_id for item in finding_set.findings)
        != expected_review.finding_ids
    ):
        raise PermissionError(
            "blocked S0 finding set differs from review or graph attempt"
        )
    unsigned = StageGateOutcomeV66(
        workspace_spec_hash=workspace.spec.spec_hash,
        stage=stage,
        attempt=manifest.attempt,
        decision="BLOCKED",
        manifest_hash=manifest_hash,
        policy_hash=policy_hash,
        graph_gate_node_hash=gate.node_hash,
        evaluator_epoch=workspace.spec.evaluator_epoch,
        check_result_hashes=sorted(set(check_result_hashes)),
        review_receipt_hashes=sorted(set(review_receipt_hashes)),
        finding_set_hash=finding_set_hash,
        reason_codes=sorted(set(reason_codes)),
        authority_key_id=workspace.authority_key_id,
        issued_at=datetime.now(timezone.utc),
    )
    payload = unsigned.model_dump(mode="json")
    payload["authority_auth_tag"] = workspace._mac(
        "stage_gate_outcome_v66", unsigned.unsigned_hash()
    )
    payload["outcome_hash"] = sha256_value(
        {key: value for key, value in payload.items() if key != "outcome_hash"}
    )
    outcome = StageGateOutcomeV66.model_validate(payload)
    outcome.assert_sealed()
    outcome_ref = workspace.commit_evidence(
        "stage_gate_outcome_v66", outcome.model_dump(mode="json")
    )
    workspace.graph.record_outcome(
        str(gate.node_hash),
        actor="verifier",
        status="blocked",
        output_artifacts=[outcome_ref],
        summary=(
            f"{stage} workflow gate blocked by authenticated independent "
            "review; no scientific qualification"
        ),
        outcome_id=f"{gate.node_id}-blocked-outcome",
    )
    workspace._write_graph_projection()
    if not verify_stage_gate_outcome_v66(workspace, outcome):
        raise StageWorkspaceError(
            "recorded stage-gate outcome failed authority verification"
        )
    return outcome


def verify_stage_gate_outcome_v66(
    workspace: StageWorkspaceV50,
    outcome: StageGateOutcomeV66,
) -> bool:
    """Verify authority, artifact, graph-node, and current-attempt bindings."""

    try:
        outcome.assert_sealed()
        if (
            outcome.workspace_spec_hash != workspace.spec.spec_hash
            or outcome.stage != "S0"
            or outcome.evaluator_epoch != workspace.spec.evaluator_epoch
            or outcome.authority_key_id != workspace.authority_key_id
            or outcome.authority_auth_tag
            != workspace._mac(
                "stage_gate_outcome_v66", outcome.unsigned_hash()
            )
        ):
            return False
        state = workspace.graph.project_state()
        graph_nodes = [
            item
            for item in state.nodes
            if item.node_hash == outcome.graph_gate_node_hash
            and item.executor == "verifier"
        ]
        if len(graph_nodes) != 1:
            return False
        manifests = [
            item
            for _, item in workspace._artifacts_of_kind(
                "stage_artifact_manifest_v50",
                StageArtifactManifestV50,
            )
            if item.manifest_hash == outcome.manifest_hash
            and item.workspace_spec_hash == workspace.spec.spec_hash
            and item.stage == outcome.stage
            and item.attempt == outcome.attempt
        ]
        if len(manifests) != 1 or outcome.policy_hash != POLICIES[
            outcome.stage
        ].policy_hash:
            return False
        checks = [
            item
            for _, item in workspace._artifacts_of_kind(
                "check_result_v50", CheckResultV50
            )
            if item.result_hash in outcome.check_result_hashes
            and item.stage == outcome.stage
            and item.input_manifest_hash == outcome.manifest_hash
            and workspace.verify_check(item)
        ]
        reviews = [
            item
            for _, item in workspace._artifacts_of_kind(
                "independent_review_receipt_v50",
                IndependentReviewReceiptV50,
            )
            if item.receipt_hash in outcome.review_receipt_hashes
            and item.stage == outcome.stage
            and item.input_manifest_hash == outcome.manifest_hash
            and workspace.verify_review(item)
        ]
        finding_set = S0ReviewFindingSetV66.model_validate(
            workspace._artifact_payload_by_hash(outcome.finding_set_hash)
        )
        finding_set.assert_sealed()
        if (
            sorted(str(item.result_hash) for item in checks)
            != outcome.check_result_hashes
            or sorted(str(item.receipt_hash) for item in reviews)
            != outcome.review_receipt_hashes
            or outcome.finding_set_hash
            not in workspace._committed_artifact_hashes()
            or outcome.reason_codes != ["independent_review_rejected"]
            or len(reviews) != 1
            or reviews[0].role != "referee"
            or reviews[0].verdict != "REJECT"
            or finding_set.task_id != workspace.spec.workspace_id
            or finding_set.attempt_id != f"s0-a{outcome.attempt}"
            or finding_set.reviewer_receipt_hash
            != reviews[0].receipt_hash
            or sorted(item.finding_id for item in finding_set.findings)
            != reviews[0].finding_ids
        ):
            return False
        matches = [
            item
            for item in state.outcomes
            if item.node_hash == outcome.graph_gate_node_hash
            and item.status == "blocked"
            and any(
                ref.kind == "stage_gate_outcome_v66"
                and workspace.graph.store.load_artifact(ref)
                == outcome.model_dump(mode="json")
                for ref in item.output_artifacts
            )
        ]
        return len(matches) == 1
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return False


def latest_stage_gate_outcome_v66(
    workspace: StageWorkspaceV50,
    stage: StageId,
) -> StageGateOutcomeV66 | None:
    """Return the unique authenticated outcome for the current stage attempt."""

    gate = workspace._binding(stage, "gate")
    state = workspace.graph.project_state()
    candidates: list[StageGateOutcomeV66] = []
    for graph_outcome in state.outcomes:
        if (
            graph_outcome.node_hash != gate.node_hash
            or graph_outcome.status != "blocked"
        ):
            continue
        for ref in graph_outcome.output_artifacts:
            if ref.kind != "stage_gate_outcome_v66":
                continue
            candidate = StageGateOutcomeV66.model_validate(
                workspace.graph.store.load_artifact(ref)
            )
            if verify_stage_gate_outcome_v66(workspace, candidate):
                candidates.append(candidate)
    if len(candidates) > 1:
        raise StageWorkspaceError(
            f"{stage} has multiple authenticated blocked-gate outcomes"
        )
    return candidates[0] if candidates else None


__all__ = [
    "StageGateOutcomeV66",
    "latest_stage_gate_outcome_v66",
    "record_blocked_stage_gate_v66",
    "verify_stage_gate_outcome_v66",
]
