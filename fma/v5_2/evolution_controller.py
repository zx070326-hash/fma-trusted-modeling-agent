"""Authenticated, budgeted graph-native recovery for V5 stage workspaces.

The model may propose a transition.  Only this code-owned controller can
authenticate it, apply its bounded file projection, revoke the affected V5
stage closure, and create a new attempt lineage.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, model_validator

from fma.hashing import canonical_json, sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256, _assert_timezone
from fma.v5.stage_workspace import StageWorkspaceV50
from fma.v5.workspace_schemas import StageId


EvolutionActionV52 = Literal[
    "PATCH_SAME_SKELETON",
    "SWITCH_REGISTERED_CANDIDATE",
    "ADMIT_NEW_CANDIDATE",
    "REVISE_DATA_CONTRACT",
    "INVALIDATE_TASK",
    "STOP_SCIENTIFICALLY_REJECTED",
]
TransitionStatusV52 = Literal[
    "ATTEMPT_CREATED",
    "TASK_INVALIDATED",
    "SCIENTIFICALLY_REJECTED",
]
_NODE_ID = re.compile(
    r"^s(?P<stage>[0-6])-a(?P<attempt>[1-9][0-9]*)-(?P<kind>work|gate)$"
)


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _graph_state_hash(workspace: StageWorkspaceV50) -> str:
    return sha256_value(
        workspace.graph.project_state().model_dump(mode="json")
    )


def _stage_attempt(workspace: StageWorkspaceV50, stage: StageId) -> int:
    wanted = str(int(stage[1:]))
    attempts: list[int] = []
    for node in workspace.graph.project_state().nodes:
        match = _NODE_ID.fullmatch(node.node_id)
        if (
            match
            and match.group("stage") == wanted
            and match.group("kind") == "work"
        ):
            attempts.append(int(match.group("attempt")))
    if not attempts:
        raise RuntimeError(f"workspace has no attempt for {stage}")
    return max(attempts)


class RecoveryBudgetV52(StrictModel):
    schema_version: Literal["5.2"] = "5.2"
    max_attempts: Annotated[int, Field(ge=2, le=32)] = 4
    max_same_skeleton_patches: Annotated[int, Field(ge=0, le=16)] = 2
    max_candidate_switches: Annotated[int, Field(ge=0, le=16)] = 2
    max_generated_candidates: Annotated[int, Field(ge=0, le=64)] = 4
    max_model_calls: Annotated[int, Field(ge=1, le=512)] = 32
    max_wall_time_seconds: Annotated[int, Field(ge=1, le=172800)] = 7200
    max_repeated_failure: Annotated[int, Field(ge=1, le=8)] = 2
    budget_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_budget(self) -> "RecoveryBudgetV52":
        if self.budget_hash and self.budget_hash != self.content_hash():
            raise ValueError("budget_hash does not match recovery budget")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "budget_hash")

    @classmethod
    def seal(cls, **data: object) -> "RecoveryBudgetV52":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"budget_hash"}),
            budget_hash=draft.content_hash(),
        )

    def assert_sealed(self) -> None:
        if not self.budget_hash or self.budget_hash != self.content_hash():
            raise ValueError("recovery budget is not sealed")


class RecoveryPolicyV52(StrictModel):
    schema_version: Literal["5.2"] = "5.2"
    policy_id: Identifier
    evaluator_epoch: Identifier
    budget: RecoveryBudgetV52
    minimum_score_improvement: Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ] = 0.0
    patchable_failure_codes: list[Identifier] = Field(default_factory=list)
    task_invalid_failure_codes: list[Identifier] = Field(default_factory=list)
    candidate_generation_allowed: bool = True
    private_feedback_permitted: Literal[False] = False
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "RecoveryPolicyV52":
        self.budget.assert_sealed()
        for label, values in (
            ("patchable_failure_codes", self.patchable_failure_codes),
            ("task_invalid_failure_codes", self.task_invalid_failure_codes),
        ):
            if values != sorted(set(values)):
                raise ValueError(f"{label} must be sorted and unique")
        overlap = set(self.patchable_failure_codes) & set(
            self.task_invalid_failure_codes
        )
        if overlap:
            raise ValueError("failure codes cannot be patchable and task-invalid")
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("policy_hash does not match recovery policy")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    @classmethod
    def seal(cls, **data: object) -> "RecoveryPolicyV52":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"policy_hash"}),
            policy_hash=draft.content_hash(),
        )

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("recovery policy is not sealed")


class RecoveryStateV52(StrictModel):
    schema_version: Literal["5.2"] = "5.2"
    workspace_spec_hash: Sha256
    policy_hash: Sha256
    attempt_count: Annotated[int, Field(ge=1)]
    same_skeleton_patch_count: Annotated[int, Field(ge=0)] = 0
    candidate_switch_count: Annotated[int, Field(ge=0)] = 0
    generated_candidate_count: Annotated[int, Field(ge=0)] = 0
    model_call_count: Annotated[int, Field(ge=0)] = 0
    repeated_failure_counts: dict[Sha256, Annotated[int, Field(ge=1)]] = Field(
        default_factory=dict
    )
    started_at: datetime
    state_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_state(self) -> "RecoveryStateV52":
        _assert_timezone(self.started_at, "started_at")
        if self.state_hash and self.state_hash != self.content_hash():
            raise ValueError("state_hash does not match recovery state")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "state_hash")

    @classmethod
    def seal(cls, **data: object) -> "RecoveryStateV52":
        data.setdefault("started_at", _utc_now())
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"state_hash"}),
            state_hash=draft.content_hash(),
        )

    def assert_sealed(self) -> None:
        if not self.state_hash or self.state_hash != self.content_hash():
            raise ValueError("recovery state is not sealed")


class RecoveryEvidenceV52(StrictModel):
    schema_version: Literal["5.2"] = "5.2"
    workspace_spec_hash: Sha256
    graph_state_hash: Sha256
    evaluator_epoch: Identifier
    failed_stage: StageId
    failed_check_hashes: Annotated[list[Sha256], Field(min_length=1)]
    failure_codes: Annotated[list[Identifier], Field(min_length=1)]
    selected_candidate_hash: Sha256
    candidate_registry_hash: Sha256
    candidate_scores: dict[Sha256, Annotated[float, Field(allow_inf_nan=False)]]
    public_evidence_hashes: Annotated[list[Sha256], Field(min_length=1)]
    private_evidence_used: Literal[False] = False
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> "RecoveryEvidenceV52":
        for label, values in (
            ("failed_check_hashes", self.failed_check_hashes),
            ("failure_codes", self.failure_codes),
            ("public_evidence_hashes", self.public_evidence_hashes),
        ):
            if values != sorted(set(values)):
                raise ValueError(f"{label} must be sorted and unique")
        if self.selected_candidate_hash not in self.candidate_scores:
            raise ValueError("selected candidate is absent from candidate scores")
        if self.evidence_hash and self.evidence_hash != self.content_hash():
            raise ValueError("evidence_hash does not match recovery evidence")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evidence_hash")

    @classmethod
    def seal(cls, **data: object) -> "RecoveryEvidenceV52":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evidence_hash"}),
            evidence_hash=draft.content_hash(),
        )

    def assert_sealed(self) -> None:
        if not self.evidence_hash or self.evidence_hash != self.content_hash():
            raise ValueError("recovery evidence is not sealed")

    @property
    def failure_fingerprint(self) -> str:
        return sha256_value(
            {
                "failed_stage": self.failed_stage,
                "failure_codes": self.failure_codes,
                "failed_check_hashes": self.failed_check_hashes,
            }
        )


class TransitionFileV52(StrictModel):
    relative_path: Annotated[str, Field(min_length=1, max_length=240)]
    utf8_text: Annotated[str, Field(max_length=1_000_000)]
    content_hash: Sha256

    @model_validator(mode="after")
    def validate_file(self) -> "TransitionFileV52":
        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.as_posix() != self.relative_path
        ):
            raise ValueError("transition file path must be canonical POSIX-relative")
        actual = hashlib.sha256(self.utf8_text.encode("utf-8")).hexdigest()
        if actual != self.content_hash:
            raise ValueError("transition file content hash differs")
        return self

    @classmethod
    def from_text(cls, relative_path: str, text: str) -> "TransitionFileV52":
        return cls(
            relative_path=relative_path,
            utf8_text=text,
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )


class EvolutionProposalV52(StrictModel):
    schema_version: Literal["5.2"] = "5.2"
    proposal_id: Identifier
    action: EvolutionActionV52
    proposer_role_receipt_hash: Sha256
    source_candidate_hash: Sha256
    target_candidate_hash: Sha256 | None = None
    admitted_candidate_receipt_hash: Sha256 | None = None
    patch_manifest_hash: Sha256 | None = None
    projection_files: Annotated[list[TransitionFileV52], Field(max_length=32)] = (
        Field(default_factory=list)
    )
    rationale: Annotated[str, Field(min_length=10, max_length=4000)]
    expected_failure_codes_addressed: list[Identifier] = Field(default_factory=list)
    private_evidence_used: Literal[False] = False
    priority: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0.5
    proposal_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_proposal(self) -> "EvolutionProposalV52":
        if self.expected_failure_codes_addressed != sorted(
            set(self.expected_failure_codes_addressed)
        ):
            raise ValueError("addressed failure codes must be sorted and unique")
        paths = [item.relative_path for item in self.projection_files]
        if paths != sorted(set(paths)):
            raise ValueError("transition file paths must be sorted and unique")
        if self.projection_files:
            expected = sha256_value(
                [item.model_dump(mode="json") for item in self.projection_files]
            )
            if self.patch_manifest_hash != expected:
                raise ValueError("patch manifest does not bind transition files")
        elif self.patch_manifest_hash is not None:
            raise ValueError("empty transition has a patch manifest")

        if self.action == "PATCH_SAME_SKELETON":
            if self.target_candidate_hash not in {
                None,
                self.source_candidate_hash,
            }:
                raise ValueError("same-skeleton patch cannot change candidate")
            if not self.projection_files:
                raise ValueError("same-skeleton patch needs projected files")
        elif self.action == "SWITCH_REGISTERED_CANDIDATE":
            if (
                not self.target_candidate_hash
                or self.target_candidate_hash == self.source_candidate_hash
                or self.admitted_candidate_receipt_hash is not None
            ):
                raise ValueError("registered switch needs another registered target")
        elif self.action == "ADMIT_NEW_CANDIDATE":
            if not self.target_candidate_hash or not self.admitted_candidate_receipt_hash:
                raise ValueError("new candidate needs target and admission receipt")
        elif self.action == "REVISE_DATA_CONTRACT":
            if not self.projection_files:
                raise ValueError("data-contract revision needs projected files")
        elif (
            self.target_candidate_hash is not None
            or self.projection_files
            or self.patch_manifest_hash is not None
        ):
            raise ValueError("terminal proposal cannot mutate candidate or files")

        if self.proposal_hash and self.proposal_hash != self.content_hash():
            raise ValueError("proposal_hash does not match evolution proposal")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "proposal_hash")

    @classmethod
    def seal(cls, **data: object) -> "EvolutionProposalV52":
        files = data.get("projection_files", [])
        if files and "patch_manifest_hash" not in data:
            data["patch_manifest_hash"] = sha256_value(
                [
                    item.model_dump(mode="json")
                    if isinstance(item, TransitionFileV52)
                    else item
                    for item in files
                ]
            )
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"proposal_hash"}),
            proposal_hash=draft.content_hash(),
        )

    def assert_sealed(self) -> None:
        if not self.proposal_hash or self.proposal_hash != self.content_hash():
            raise ValueError("evolution proposal is not sealed")


class EvolutionDecisionV52(StrictModel):
    schema_version: Literal["5.2"] = "5.2"
    decision_id: Identifier
    proposal_hash: Sha256
    evidence_hash: Sha256
    recovery_state_hash: Sha256
    policy_hash: Sha256
    before_graph_state_hash: Sha256
    action: EvolutionActionV52
    earliest_affected_stage: StageId | None = None
    source_candidate_hash: Sha256
    target_candidate_hash: Sha256 | None = None
    patch_manifest_hash: Sha256 | None = None
    admitted_candidate_receipt_hash: Sha256 | None = None
    authority_key_id: Identifier
    issued_at: datetime
    private_evidence_used: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    authority_auth_tag: Sha256 | None = None
    decision_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> "EvolutionDecisionV52":
        _assert_timezone(self.issued_at, "issued_at")
        expected_stage: StageId | None = {
            "PATCH_SAME_SKELETON": "S3",
            "SWITCH_REGISTERED_CANDIDATE": "S1",
            "ADMIT_NEW_CANDIDATE": "S1",
            "REVISE_DATA_CONTRACT": "S2",
            "INVALIDATE_TASK": None,
            "STOP_SCIENTIFICALLY_REJECTED": None,
        }[self.action]
        if self.earliest_affected_stage != expected_stage:
            raise ValueError("decision earliest stage differs from action policy")
        if self.authority_auth_tag and not self.decision_hash:
            raise ValueError("authenticated decision requires decision_hash")
        if self.decision_hash and self.decision_hash != self.content_hash():
            raise ValueError("decision_hash does not match evolution decision")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "authority_auth_tag", "decision_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "decision_hash")


class EvolutionTransitionReceiptV52(StrictModel):
    schema_version: Literal["5.2"] = "5.2"
    decision_hash: Sha256
    before_graph_state_hash: Sha256
    after_graph_state_hash: Sha256
    status: TransitionStatusV52
    earliest_affected_stage: StageId | None = None
    predecessor_attempt: Annotated[int, Field(ge=1)] | None = None
    successor_attempt: Annotated[int, Field(ge=2)] | None = None
    affected_node_hashes: list[Sha256] = Field(default_factory=list)
    applied_file_hashes: dict[str, Sha256] = Field(default_factory=dict)
    source_candidate_hash: Sha256
    target_candidate_hash: Sha256 | None = None
    executed_at: datetime
    private_evidence_used: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "EvolutionTransitionReceiptV52":
        _assert_timezone(self.executed_at, "executed_at")
        if self.affected_node_hashes != sorted(set(self.affected_node_hashes)):
            raise ValueError("affected node hashes must be sorted and unique")
        attempt_created = self.status == "ATTEMPT_CREATED"
        if attempt_created != (
            self.earliest_affected_stage is not None
            and self.predecessor_attempt is not None
            and self.successor_attempt is not None
            and bool(self.affected_node_hashes)
        ):
            raise ValueError("attempt transition fields disagree with status")
        if (
            attempt_created
            and self.successor_attempt <= self.predecessor_attempt
        ):
            raise ValueError("successor attempt does not advance lineage")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match transition receipt")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    @classmethod
    def seal(cls, **data: object) -> "EvolutionTransitionReceiptV52":
        data.setdefault("executed_at", _utc_now())
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


class RecoveryAuthorityV52:
    def __init__(self, key_id: str, secret: bytes) -> None:
        if not re.fullmatch(r"^[A-Za-z][A-Za-z0-9_.-]*$", key_id):
            raise ValueError("invalid recovery authority key_id")
        if len(secret) < 32:
            raise ValueError("recovery authority secret needs at least 32 bytes")
        self.key_id = key_id
        self._secret = bytes(secret)

    def _mac(self, unsigned_hash: str) -> str:
        return hmac.new(
            self._secret,
            f"evolution_decision_v52:{unsigned_hash}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def issue(self, **data: object) -> EvolutionDecisionV52:
        data["authority_key_id"] = self.key_id
        data.setdefault("issued_at", _utc_now())
        unsigned = EvolutionDecisionV52(**data)
        tag = self._mac(unsigned.unsigned_hash())
        final_payload = unsigned.model_dump(mode="json")
        final_payload["authority_auth_tag"] = tag
        final_payload["decision_hash"] = sha256_value(
            {
                key: value
                for key, value in final_payload.items()
                if key != "decision_hash"
            }
        )
        return EvolutionDecisionV52(**final_payload)

    def verify(self, decision: EvolutionDecisionV52) -> bool:
        try:
            return bool(
                decision.decision_hash
                and decision.decision_hash == decision.content_hash()
                and decision.authority_key_id == self.key_id
                and decision.authority_auth_tag
                and hmac.compare_digest(
                    decision.authority_auth_tag,
                    self._mac(decision.unsigned_hash()),
                )
            )
        except (TypeError, ValueError):
            return False


class GraphEvolutionControllerV52:
    """Code-owned authorizer and executor for one recovery policy."""

    def __init__(
        self,
        *,
        authority: RecoveryAuthorityV52,
        policy: RecoveryPolicyV52,
    ) -> None:
        policy.assert_sealed()
        self.authority = authority
        self.policy = policy

    @staticmethod
    def _allowed_prefixes(action: EvolutionActionV52) -> tuple[str, ...]:
        return {
            "PATCH_SAME_SKELETON": ("src/models/", "checks/", "docs/"),
            "SWITCH_REGISTERED_CANDIDATE": ("docs/",),
            "ADMIT_NEW_CANDIDATE": ("docs/",),
            "REVISE_DATA_CONTRACT": ("data/processed/", "docs/"),
            "INVALIDATE_TASK": (),
            "STOP_SCIENTIFICALLY_REJECTED": (),
        }[action]

    def _assert_budget(
        self,
        state: RecoveryStateV52,
        evidence: RecoveryEvidenceV52,
        action: EvolutionActionV52,
        now: datetime,
    ) -> None:
        budget = self.policy.budget
        elapsed = (now - state.started_at).total_seconds()
        if elapsed > budget.max_wall_time_seconds:
            raise PermissionError("recovery wall-time budget exhausted")
        if state.attempt_count >= budget.max_attempts and action not in {
            "INVALIDATE_TASK",
            "STOP_SCIENTIFICALLY_REJECTED",
        }:
            raise PermissionError("recovery attempt budget exhausted")
        if (
            action == "PATCH_SAME_SKELETON"
            and state.same_skeleton_patch_count
            >= budget.max_same_skeleton_patches
        ):
            raise PermissionError("same-skeleton patch budget exhausted")
        if (
            action == "SWITCH_REGISTERED_CANDIDATE"
            and state.candidate_switch_count >= budget.max_candidate_switches
        ):
            raise PermissionError("candidate-switch budget exhausted")
        if (
            action == "ADMIT_NEW_CANDIDATE"
            and (
                not self.policy.candidate_generation_allowed
                or state.generated_candidate_count
                >= budget.max_generated_candidates
            )
        ):
            raise PermissionError("generated-candidate budget exhausted")
        if (
            state.model_call_count >= budget.max_model_calls
            and action
            not in {"INVALIDATE_TASK", "STOP_SCIENTIFICALLY_REJECTED"}
        ):
            raise PermissionError("model-call budget exhausted")
        repeated = state.repeated_failure_counts.get(
            evidence.failure_fingerprint, 0
        )
        if (
            repeated >= budget.max_repeated_failure
            and action == "PATCH_SAME_SKELETON"
        ):
            raise PermissionError("repeated failure forbids another local patch")

    def authorize(
        self,
        *,
        workspace: StageWorkspaceV50,
        state: RecoveryStateV52,
        evidence: RecoveryEvidenceV52,
        proposal: EvolutionProposalV52,
        registered_candidate_hashes: set[str],
        admitted_candidate_receipt_hashes: set[str],
        now: datetime | None = None,
    ) -> EvolutionDecisionV52:
        now = now or _utc_now()
        state.assert_sealed()
        evidence.assert_sealed()
        proposal.assert_sealed()
        if state.workspace_spec_hash != workspace.spec.spec_hash:
            raise PermissionError("recovery state belongs to another workspace")
        if state.policy_hash != self.policy.policy_hash:
            raise PermissionError("recovery state belongs to another policy")
        if evidence.workspace_spec_hash != workspace.spec.spec_hash:
            raise PermissionError("evidence belongs to another workspace")
        if evidence.evaluator_epoch != self.policy.evaluator_epoch:
            raise PermissionError("evaluator epoch changed")
        current_graph_hash = _graph_state_hash(workspace)
        if evidence.graph_state_hash != current_graph_hash:
            raise PermissionError("recovery evidence is stale")
        if proposal.source_candidate_hash != evidence.selected_candidate_hash:
            raise PermissionError("proposal source candidate differs from evidence")
        if not set(proposal.expected_failure_codes_addressed).issubset(
            set(evidence.failure_codes)
        ):
            raise PermissionError("proposal addresses unobserved failure codes")
        if set(evidence.failure_codes) & set(
            self.policy.task_invalid_failure_codes
        ) and proposal.action != "INVALIDATE_TASK":
            raise PermissionError("task-invalid evidence requires invalidation")
        if proposal.action == "PATCH_SAME_SKELETON" and not set(
            proposal.expected_failure_codes_addressed
        ).issubset(set(self.policy.patchable_failure_codes)):
            raise PermissionError("proposal tries to patch a non-patchable failure")
        if proposal.action == "SWITCH_REGISTERED_CANDIDATE":
            if proposal.target_candidate_hash not in registered_candidate_hashes:
                raise PermissionError("switch target is not registered")
            source_score = evidence.candidate_scores[
                evidence.selected_candidate_hash
            ]
            try:
                target_score = evidence.candidate_scores[
                    str(proposal.target_candidate_hash)
                ]
            except KeyError as exc:
                raise PermissionError(
                    "switch target has no public development score"
                ) from exc
            if (
                target_score - source_score
                < self.policy.minimum_score_improvement
            ):
                raise PermissionError("switch lacks frozen score improvement")
        if proposal.action == "ADMIT_NEW_CANDIDATE":
            if (
                proposal.admitted_candidate_receipt_hash
                not in admitted_candidate_receipt_hashes
            ):
                raise PermissionError("new candidate lacks an admitted receipt")
        prefixes = self._allowed_prefixes(proposal.action)
        for item in proposal.projection_files:
            if not item.relative_path.startswith(prefixes):
                raise PermissionError("transition file is outside action scope")
        self._assert_budget(state, evidence, proposal.action, now)
        stage = {
            "PATCH_SAME_SKELETON": "S3",
            "SWITCH_REGISTERED_CANDIDATE": "S1",
            "ADMIT_NEW_CANDIDATE": "S1",
            "REVISE_DATA_CONTRACT": "S2",
            "INVALIDATE_TASK": None,
            "STOP_SCIENTIFICALLY_REJECTED": None,
        }[proposal.action]
        return self.authority.issue(
            decision_id=f"decision.{proposal.proposal_id}",
            proposal_hash=proposal.proposal_hash,
            evidence_hash=evidence.evidence_hash,
            recovery_state_hash=state.state_hash,
            policy_hash=self.policy.policy_hash,
            before_graph_state_hash=current_graph_hash,
            action=proposal.action,
            earliest_affected_stage=stage,
            source_candidate_hash=proposal.source_candidate_hash,
            target_candidate_hash=proposal.target_candidate_hash,
            patch_manifest_hash=proposal.patch_manifest_hash,
            admitted_candidate_receipt_hash=(
                proposal.admitted_candidate_receipt_hash
            ),
            issued_at=now,
        )

    @staticmethod
    def _apply_files(
        root: Path, files: list[TransitionFileV52]
    ) -> dict[str, str]:
        applied: dict[str, str] = {}
        root = root.resolve()
        for item in files:
            target = (root / PurePosixPath(item.relative_path)).resolve()
            if root not in target.parents:
                raise PermissionError("transition target escaped workspace")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(
                f".{target.name}.v52-{os.getpid()}.tmp"
            )
            temporary.write_text(item.utf8_text, encoding="utf-8", newline="\n")
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != item.content_hash:
                temporary.unlink(missing_ok=True)
                raise OSError("transition write changed bytes")
            os.replace(temporary, target)
            applied[item.relative_path] = item.content_hash
        return dict(sorted(applied.items()))

    def execute(
        self,
        *,
        workspace: StageWorkspaceV50,
        decision: EvolutionDecisionV52,
        proposal: EvolutionProposalV52,
    ) -> EvolutionTransitionReceiptV52:
        if not self.authority.verify(decision):
            raise PermissionError("evolution decision authentication failed")
        proposal.assert_sealed()
        if decision.proposal_hash != proposal.proposal_hash:
            raise PermissionError("decision does not bind supplied proposal")
        before = _graph_state_hash(workspace)
        if before != decision.before_graph_state_hash:
            raise PermissionError("workspace graph changed after authorization")

        if decision.action in {
            "INVALIDATE_TASK",
            "STOP_SCIENTIFICALLY_REJECTED",
        }:
            status: TransitionStatusV52 = (
                "TASK_INVALIDATED"
                if decision.action == "INVALIDATE_TASK"
                else "SCIENTIFICALLY_REJECTED"
            )
            receipt = EvolutionTransitionReceiptV52.seal(
                decision_hash=decision.decision_hash,
                before_graph_state_hash=before,
                after_graph_state_hash=before,
                status=status,
                source_candidate_hash=decision.source_candidate_hash,
                target_candidate_hash=decision.target_candidate_hash,
            )
            workspace.commit_evidence(
                "evolution_decision_v52", decision.model_dump(mode="json")
            )
            workspace.commit_evidence(
                "evolution_transition_receipt_v52",
                receipt.model_dump(mode="json"),
            )
            if not workspace.verify():
                raise RuntimeError(
                    "workspace failed verification after terminal transition"
                )
            return receipt

        stage = decision.earliest_affected_stage
        assert stage is not None
        predecessor = _stage_attempt(workspace, stage)
        affected = workspace.invalidate_from(
            stage,
            reason=(
                f"authenticated V5.2 {decision.action} "
                f"decision {decision.decision_hash}"
            ),
            authority="verifier",
        )
        applied = self._apply_files(workspace.root, proposal.projection_files)
        successor = _stage_attempt(workspace, stage)
        after = _graph_state_hash(workspace)
        receipt = EvolutionTransitionReceiptV52.seal(
            decision_hash=decision.decision_hash,
            before_graph_state_hash=before,
            after_graph_state_hash=after,
            status="ATTEMPT_CREATED",
            earliest_affected_stage=stage,
            predecessor_attempt=predecessor,
            successor_attempt=successor,
            affected_node_hashes=sorted(affected),
            applied_file_hashes=applied,
            source_candidate_hash=decision.source_candidate_hash,
            target_candidate_hash=decision.target_candidate_hash,
        )
        workspace.commit_evidence(
            "evolution_decision_v52", decision.model_dump(mode="json")
        )
        workspace.commit_evidence(
            "evolution_transition_receipt_v52",
            receipt.model_dump(mode="json"),
        )
        if not workspace.verify():
            raise RuntimeError("workspace failed verification after transition")
        return receipt

    def advance_state(
        self,
        state: RecoveryStateV52,
        evidence: RecoveryEvidenceV52,
        decision: EvolutionDecisionV52,
    ) -> RecoveryStateV52:
        state.assert_sealed()
        counts = dict(state.repeated_failure_counts)
        counts[evidence.failure_fingerprint] = (
            counts.get(evidence.failure_fingerprint, 0) + 1
        )
        creates_attempt = decision.earliest_affected_stage is not None
        return RecoveryStateV52.seal(
            workspace_spec_hash=state.workspace_spec_hash,
            policy_hash=state.policy_hash,
            attempt_count=state.attempt_count + int(creates_attempt),
            same_skeleton_patch_count=state.same_skeleton_patch_count
            + int(decision.action == "PATCH_SAME_SKELETON"),
            candidate_switch_count=state.candidate_switch_count
            + int(decision.action == "SWITCH_REGISTERED_CANDIDATE"),
            generated_candidate_count=state.generated_candidate_count
            + int(decision.action == "ADMIT_NEW_CANDIDATE"),
            model_call_count=state.model_call_count,
            repeated_failure_counts=counts,
            started_at=state.started_at,
        )


__all__ = [
    "EvolutionDecisionV52",
    "EvolutionProposalV52",
    "EvolutionTransitionReceiptV52",
    "GraphEvolutionControllerV52",
    "RecoveryAuthorityV52",
    "RecoveryBudgetV52",
    "RecoveryEvidenceV52",
    "RecoveryPolicyV52",
    "RecoveryStateV52",
    "TransitionFileV52",
]
