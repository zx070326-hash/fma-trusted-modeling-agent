"""Typed epistemic graph for controlled cross-branch modelling knowledge.

V5.8 does not replace the V4 authority graph or the V5 S0--S6 workflow.  It is
an additive evidence projection used inside S1.  Model processes may propose
knowledge and transfer hypotheses; the harness owns validation, disclosure,
state persistence, experience eligibility, and graph transitions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256, _assert_timezone


KnowledgeKindV58 = Literal[
    "observation",
    "hypothesis",
    "constraint",
    "failure",
    "method",
    "counterexample",
]
KnowledgeStatusV58 = Literal[
    "proposed",
    "mechanically_valid",
    "independently_supported",
    "contradicted",
    "revoked",
    "cross_task_promoted",
    "stale",
]
TransferVerdictV58 = Literal["ACCEPT_FOR_TEST", "REJECT", "HUMAN"]
EpistemicEventTypeV58 = Literal[
    "knowledge_published",
    "initial_frontier_frozen",
    "disclosure_opened",
    "transfer_proposed",
    "transfer_assessed",
    "experience_projected",
]


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _sorted_unique(values: list[str], name: str) -> None:
    if values != sorted(set(values)):
        raise ValueError(f"{name} must be sorted and unique")


class KnowledgeDraftV58(StrictModel):
    """Untrusted semantic content proposed by one blind branch."""

    unit_id: Identifier
    kind: KnowledgeKindV58
    statement: Annotated[str, Field(min_length=10, max_length=2000)]
    applicability_conditions: Annotated[
        list[Annotated[str, Field(min_length=3, max_length=500)]],
        Field(min_length=1, max_length=12),
    ]
    falsification_test: Annotated[str, Field(min_length=5, max_length=1000)]
    utility_hint: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0.5

    @model_validator(mode="after")
    def validate_draft(self) -> "KnowledgeDraftV58":
        if len(self.applicability_conditions) != len(
            set(self.applicability_conditions)
        ):
            raise ValueError("knowledge applicability conditions must be unique")
        return self


class KnowledgeUnitV58(StrictModel):
    schema_version: Literal["5.8"] = "5.8"
    unit_id: Identifier
    task_id: Identifier
    branch_id: Identifier
    kind: KnowledgeKindV58
    statement: Annotated[str, Field(min_length=10, max_length=2000)]
    applicability_conditions: Annotated[list[str], Field(min_length=1, max_length=12)]
    falsification_test: Annotated[str, Field(min_length=5, max_length=1000)]
    shared_input_hashes: Annotated[list[Sha256], Field(min_length=1)]
    independent_origin_hashes: Annotated[list[Sha256], Field(min_length=1)]
    evidence_refs: Annotated[list[Sha256], Field(min_length=1)]
    ancestor_unit_hashes: list[Sha256] = Field(default_factory=list)
    status: KnowledgeStatusV58
    independent_support_count: Annotated[int, Field(ge=0)] = 0
    utility_hint: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0.5
    privacy_scope: Literal[
        "development_public",
        "public_only",
        "private_excluded",
    ] = "development_public"
    created_by: Literal["model", "harness", "verifier"]
    created_at: datetime
    unit_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_unit(self) -> "KnowledgeUnitV58":
        _assert_timezone(self.created_at, "created_at")
        for field_name in (
            "shared_input_hashes",
            "independent_origin_hashes",
            "evidence_refs",
            "ancestor_unit_hashes",
        ):
            _sorted_unique(list(getattr(self, field_name)), field_name)
        if (
            self.status
            in {
                "independently_supported",
                "cross_task_promoted",
            }
            and self.independent_support_count < 2
        ):
            raise ValueError("supported knowledge needs two independent supports")
        if self.status == "cross_task_promoted" and (
            self.privacy_scope == "private_excluded"
        ):
            raise ValueError("private-excluded knowledge cannot be promoted")
        if self.unit_hash and self.unit_hash != self.content_hash():
            raise ValueError("knowledge unit hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "unit_hash")

    def assert_sealed(self) -> None:
        if not self.unit_hash or self.unit_hash != self.content_hash():
            raise ValueError("knowledge unit is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "KnowledgeUnitV58":
        data.setdefault("created_at", datetime.now(timezone.utc))
        draft = cls(**data)
        payload = draft.model_dump(exclude={"unit_hash"})
        payload["unit_hash"] = draft.content_hash()
        return cls(**payload)


class IndependenceAssessmentV58(StrictModel):
    schema_version: Literal["5.8"] = "5.8"
    # This proves attributable context/origin separation only.  Multiple calls
    # to one model remain correlated reasoners, not scientific replications.
    assesses_only_origin_separation: Literal[True] = True
    scientific_independence_established: Literal[False] = False
    branch_origin_hashes: dict[Identifier, list[Sha256]]
    shared_input_hashes: list[Sha256]
    cross_branch_overlap_pairs: list[str]
    effective_independent_branches: Annotated[int, Field(ge=0)]
    minimum_required_branches: Annotated[int, Field(ge=1)] = 3
    passed: bool
    assessment_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_assessment(self) -> "IndependenceAssessmentV58":
        for branch, values in self.branch_origin_hashes.items():
            _sorted_unique(values, f"branch_origin_hashes[{branch}]")
        _sorted_unique(self.shared_input_hashes, "shared_input_hashes")
        _sorted_unique(self.cross_branch_overlap_pairs, "cross_branch_overlap_pairs")
        expected = (
            not self.cross_branch_overlap_pairs
            and self.effective_independent_branches >= self.minimum_required_branches
        )
        if self.passed != expected:
            raise ValueError("independence verdict differs from ancestry")
        if self.assessment_hash and self.assessment_hash != self.content_hash():
            raise ValueError("independence assessment hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "assessment_hash")

    @classmethod
    def assess(
        cls,
        units: list[KnowledgeUnitV58],
        *,
        minimum_required_branches: int = 3,
    ) -> "IndependenceAssessmentV58":
        by_branch: dict[str, set[str]] = {}
        shared: set[str] = set()
        for unit in units:
            unit.assert_sealed()
            by_branch.setdefault(unit.branch_id, set()).update(
                unit.independent_origin_hashes
            )
            shared.update(unit.shared_input_hashes)
        branches = sorted(by_branch)
        overlaps: list[str] = []
        parents = {branch: branch for branch in branches}

        def find(branch: str) -> str:
            while parents[branch] != branch:
                parents[branch] = parents[parents[branch]]
                branch = parents[branch]
            return branch

        def union(left: str, right: str) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        for index, left in enumerate(branches):
            for right in branches[index + 1 :]:
                if by_branch[left] & by_branch[right]:
                    overlaps.append(f"{left}|{right}")
                    union(left, right)
        effective = len({find(branch) for branch in branches})
        draft = cls(
            branch_origin_hashes={
                branch: sorted(values) for branch, values in by_branch.items()
            },
            shared_input_hashes=sorted(shared),
            cross_branch_overlap_pairs=sorted(overlaps),
            effective_independent_branches=effective,
            minimum_required_branches=minimum_required_branches,
            passed=not overlaps and effective >= minimum_required_branches,
        )
        return cls(
            **draft.model_dump(exclude={"assessment_hash"}),
            assessment_hash=draft.content_hash(),
        )


class DisclosurePacketV58(StrictModel):
    schema_version: Literal["5.8"] = "5.8"
    packet_id: Identifier
    recipient_branch_id: Identifier
    source_graph_hash: Sha256
    disclosed_unit_hashes: Annotated[list[Sha256], Field(min_length=1, max_length=8)]
    source_branch_ids: Annotated[list[Identifier], Field(min_length=1)]
    selection_policy: Literal["diversity_then_utility_v58"] = (
        "diversity_then_utility_v58"
    )
    information_value_score: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    peer_context_only: Literal[True] = True
    packet_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_packet(self) -> "DisclosurePacketV58":
        _sorted_unique(self.disclosed_unit_hashes, "disclosed_unit_hashes")
        _sorted_unique(self.source_branch_ids, "source_branch_ids")
        if self.recipient_branch_id in self.source_branch_ids:
            raise ValueError(
                "a branch cannot receive its own knowledge as peer context"
            )
        if self.packet_hash and self.packet_hash != self.content_hash():
            raise ValueError("disclosure packet hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "packet_hash")

    @classmethod
    def seal(cls, **data: object) -> "DisclosurePacketV58":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"packet_hash"})
        payload["packet_hash"] = draft.content_hash()
        return cls(**payload)


class TransferDraftV58(StrictModel):
    transfer_id: Identifier
    source_unit_ids: Annotated[list[Identifier], Field(min_length=1, max_length=4)]
    target_branch_id: Identifier
    target_interpretation: Annotated[str, Field(min_length=10, max_length=1500)]
    proposed_modification: Annotated[str, Field(min_length=10, max_length=1500)]
    falsification_test: Annotated[str, Field(min_length=5, max_length=1000)]

    @model_validator(mode="after")
    def validate_transfer(self) -> "TransferDraftV58":
        _sorted_unique(self.source_unit_ids, "source_unit_ids")
        return self


class TransferHypothesisV58(StrictModel):
    schema_version: Literal["5.8"] = "5.8"
    transfer_id: Identifier
    source_unit_hashes: Annotated[list[Sha256], Field(min_length=1, max_length=4)]
    source_branch_ids: Annotated[list[Identifier], Field(min_length=1)]
    target_branch_id: Identifier
    target_interpretation: Annotated[str, Field(min_length=10, max_length=1500)]
    proposed_modification: Annotated[str, Field(min_length=10, max_length=1500)]
    falsification_test: Annotated[str, Field(min_length=5, max_length=1000)]
    translator_receipt_hash: Sha256
    status: Literal["proposed"] = "proposed"
    transfer_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_hypothesis(self) -> "TransferHypothesisV58":
        _sorted_unique(self.source_unit_hashes, "source_unit_hashes")
        _sorted_unique(self.source_branch_ids, "source_branch_ids")
        if self.target_branch_id in self.source_branch_ids:
            raise ValueError("translation must cross branch boundaries")
        if self.transfer_hash and self.transfer_hash != self.content_hash():
            raise ValueError("transfer hypothesis hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "transfer_hash")

    def assert_sealed(self) -> None:
        if not self.transfer_hash or self.transfer_hash != self.content_hash():
            raise ValueError("transfer hypothesis is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "TransferHypothesisV58":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"transfer_hash"})
        payload["transfer_hash"] = draft.content_hash()
        return cls(**payload)


class TransferAssessmentDraftV58(StrictModel):
    transfer_id: Identifier
    verdict: TransferVerdictV58
    rationale: Annotated[str, Field(min_length=10, max_length=1500)]
    required_test: Annotated[str, Field(min_length=5, max_length=1000)]


class TransferAssessmentV58(StrictModel):
    schema_version: Literal["5.8"] = "5.8"
    transfer_hash: Sha256
    target_branch_id: Identifier
    verdict: TransferVerdictV58
    rationale: Annotated[str, Field(min_length=10, max_length=1500)]
    required_test: Annotated[str, Field(min_length=5, max_length=1000)]
    assessor_receipt_hash: Sha256
    scientific_support_established: Literal[False] = False
    assessment_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_transfer_assessment(self) -> "TransferAssessmentV58":
        if self.assessment_hash and self.assessment_hash != self.content_hash():
            raise ValueError("transfer assessment hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "assessment_hash")

    @classmethod
    def seal(cls, **data: object) -> "TransferAssessmentV58":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"assessment_hash"})
        payload["assessment_hash"] = draft.content_hash()
        return cls(**payload)


class ExperienceProjectionV58(StrictModel):
    schema_version: Literal["5.8"] = "5.8"
    eligible_unit_hashes: list[Sha256]
    quarantined_unit_hashes: list[Sha256]
    rejection_reasons: dict[Sha256, list[Identifier]]
    promotion_authority_present: bool
    cross_task_use_permitted: bool
    projection_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_projection(self) -> "ExperienceProjectionV58":
        _sorted_unique(self.eligible_unit_hashes, "eligible_unit_hashes")
        _sorted_unique(self.quarantined_unit_hashes, "quarantined_unit_hashes")
        if set(self.eligible_unit_hashes) & set(self.quarantined_unit_hashes):
            raise ValueError("experience projection sets overlap")
        if self.cross_task_use_permitted != bool(
            self.promotion_authority_present and self.eligible_unit_hashes
        ):
            raise ValueError("cross-task permission differs from promotion evidence")
        if self.projection_hash and self.projection_hash != self.content_hash():
            raise ValueError("experience projection hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "projection_hash")

    @classmethod
    def project(
        cls,
        units: list[KnowledgeUnitV58],
        *,
        promotion_authority_present: bool = False,
    ) -> "ExperienceProjectionV58":
        eligible: list[str] = []
        quarantined: list[str] = []
        reasons: dict[str, list[str]] = {}
        for unit in sorted(units, key=lambda item: str(item.unit_hash)):
            unit.assert_sealed()
            unit_reasons: list[str] = []
            if unit.status not in {
                "independently_supported",
                "cross_task_promoted",
            }:
                unit_reasons.append("not_independently_supported")
            if unit.independent_support_count < 2:
                unit_reasons.append("insufficient_independent_support")
            if unit.privacy_scope == "private_excluded":
                unit_reasons.append("private_scope_excluded")
            if not promotion_authority_present:
                unit_reasons.append("promotion_authority_absent")
            assert unit.unit_hash is not None
            if unit_reasons:
                quarantined.append(unit.unit_hash)
                reasons[unit.unit_hash] = sorted(unit_reasons)
            else:
                eligible.append(unit.unit_hash)
        draft = cls(
            eligible_unit_hashes=sorted(eligible),
            quarantined_unit_hashes=sorted(quarantined),
            rejection_reasons=reasons,
            promotion_authority_present=promotion_authority_present,
            cross_task_use_permitted=bool(promotion_authority_present and eligible),
        )
        return cls(
            **draft.model_dump(exclude={"projection_hash"}),
            projection_hash=draft.content_hash(),
        )


class EpistemicEventV58(StrictModel):
    schema_version: Literal["5.8"] = "5.8"
    sequence: Annotated[int, Field(ge=1)]
    previous_event_hash: Sha256 | None
    event_type: EpistemicEventTypeV58
    subject_hashes: list[Sha256]
    actor: Literal["model", "harness", "verifier"]
    details: dict[str, Any] = Field(default_factory=dict)
    recorded_at: datetime
    event_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_event(self) -> "EpistemicEventV58":
        _assert_timezone(self.recorded_at, "recorded_at")
        _sorted_unique(self.subject_hashes, "subject_hashes")
        if self.event_hash and self.event_hash != self.content_hash():
            raise ValueError("epistemic event hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "event_hash")

    @classmethod
    def seal(cls, **data: object) -> "EpistemicEventV58":
        data.setdefault("recorded_at", datetime.now(timezone.utc))
        draft = cls(**data)
        payload = draft.model_dump(exclude={"event_hash"})
        payload["event_hash"] = draft.content_hash()
        return cls(**payload)


class EpistemicGraphV58(StrictModel):
    schema_version: Literal["5.8"] = "5.8"
    task_id: Identifier
    workspace_spec_hash: Sha256
    s0_gate_hash: Sha256
    knowledge_units: list[KnowledgeUnitV58]
    disclosure_packets: list[DisclosurePacketV58]
    transfers: list[TransferHypothesisV58]
    transfer_assessments: list[TransferAssessmentV58]
    independence: IndependenceAssessmentV58
    experience_projection: ExperienceProjectionV58
    events: list[EpistemicEventV58]
    head_event_hash: Sha256 | None
    graph_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_graph(self) -> "EpistemicGraphV58":
        unit_hashes: set[str] = set()
        unit_ids: set[str] = set()
        unit_by_hash: dict[str, KnowledgeUnitV58] = {}
        for unit in self.knowledge_units:
            unit.assert_sealed()
            assert unit.unit_hash is not None
            if unit.unit_hash in unit_hashes or unit.unit_id in unit_ids:
                raise ValueError("epistemic graph contains duplicate knowledge")
            unit_hashes.add(unit.unit_hash)
            unit_ids.add(unit.unit_id)
            unit_by_hash[unit.unit_hash] = unit
        packet_hashes: set[str] = set()
        for packet in self.disclosure_packets:
            if not packet.packet_hash or packet.packet_hash != packet.content_hash():
                raise ValueError("epistemic graph contains unsealed disclosure")
            if packet.packet_hash in packet_hashes:
                raise ValueError("epistemic graph contains duplicate disclosure")
            packet_hashes.add(packet.packet_hash)
            if not set(packet.disclosed_unit_hashes).issubset(unit_hashes):
                raise ValueError("disclosure references unknown knowledge")
            disclosed_branches = {
                unit_by_hash[item].branch_id for item in packet.disclosed_unit_hashes
            }
            if sorted(disclosed_branches) != packet.source_branch_ids:
                raise ValueError("disclosure source branches differ from knowledge")
        transfer_hashes: set[str] = set()
        transfer_by_hash: dict[str, TransferHypothesisV58] = {}
        for transfer in self.transfers:
            transfer.assert_sealed()
            assert transfer.transfer_hash is not None
            if transfer.transfer_hash in transfer_hashes:
                raise ValueError("epistemic graph contains duplicate transfer")
            if not set(transfer.source_unit_hashes).issubset(unit_hashes):
                raise ValueError("transfer references unknown knowledge")
            source_branches = {
                unit_by_hash[item].branch_id for item in transfer.source_unit_hashes
            }
            if sorted(source_branches) != transfer.source_branch_ids:
                raise ValueError("transfer source branches differ from knowledge")
            transfer_hashes.add(transfer.transfer_hash)
            transfer_by_hash[transfer.transfer_hash] = transfer
        assessment_hashes: set[str] = set()
        for assessment in self.transfer_assessments:
            if (
                not assessment.assessment_hash
                or assessment.assessment_hash != assessment.content_hash()
            ):
                raise ValueError("epistemic graph contains unsealed assessment")
            if assessment.assessment_hash in assessment_hashes:
                raise ValueError("epistemic graph contains duplicate assessment")
            transfer = transfer_by_hash.get(assessment.transfer_hash)
            if transfer is None:
                raise ValueError("assessment references unknown transfer")
            if assessment.target_branch_id != transfer.target_branch_id:
                raise ValueError("assessment target differs from transfer")
            assessment_hashes.add(assessment.assessment_hash)
        previous = None
        for index, event in enumerate(self.events, start=1):
            if not event.event_hash or event.event_hash != event.content_hash():
                raise ValueError("epistemic graph contains unsealed event")
            if event.sequence != index or event.previous_event_hash != previous:
                raise ValueError("epistemic event chain differs")
            previous = event.event_hash
        if self.head_event_hash != previous:
            raise ValueError("epistemic graph head differs")
        if self.graph_hash and self.graph_hash != self.content_hash():
            raise ValueError("epistemic graph hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "graph_hash")

    def assert_sealed(self) -> None:
        if not self.graph_hash or self.graph_hash != self.content_hash():
            raise ValueError("epistemic graph is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EpistemicGraphV58":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"graph_hash"})
        payload["graph_hash"] = draft.content_hash()
        return cls(**payload)


class S1ExplorationBudgetV58(StrictModel):
    schema_version: Literal["5.8"] = "5.8"
    branch_ids: Annotated[list[Identifier], Field(min_length=3, max_length=8)] = Field(
        default_factory=lambda: [
            "mechanistic",
            "null_baseline",
            "statistical",
            "system_learning",
        ]
    )
    max_parallel_workers: Annotated[int, Field(ge=1, le=8)] = 4
    disclosure_limit_per_branch: Annotated[int, Field(ge=1, le=8)] = 3
    max_cross_pollination_branches: Annotated[int, Field(ge=1, le=4)] = 2
    max_transfer_hypotheses: Annotated[int, Field(ge=1, le=8)] = 2
    # Thirteen calls cover the happy path, including the pre-freeze auditor.
    # Three additional calls are a shared fail-closed reserve for branch,
    # formalization, and review recovery. The global cap always wins.
    max_model_calls: Annotated[int, Field(ge=8, le=32)] = 16

    @model_validator(mode="after")
    def validate_budget(self) -> "S1ExplorationBudgetV58":
        _sorted_unique(self.branch_ids, "branch_ids")
        if self.max_parallel_workers > len(self.branch_ids) + 1:
            raise ValueError("parallel worker budget exceeds initial packet count")
        if self.max_cross_pollination_branches > len(self.branch_ids):
            raise ValueError("cross-pollination budget exceeds branch count")
        minimum_calls = (
            len(self.branch_ids)
            + 1  # scout
            + self.max_cross_pollination_branches  # isolated translators
            + self.max_cross_pollination_branches
            + 1  # synthesis
            + 2  # independent stage reviews
        )
        if self.max_model_calls < minimum_calls:
            raise ValueError("model-call budget cannot execute the declared workflow")
        return self


class KnowledgeBrokerV58:
    """Harness-owned controlled disclosure and budget selection."""

    _KIND_PRIORITY = {
        "counterexample": 0,
        "failure": 1,
        "constraint": 2,
        "observation": 3,
        "hypothesis": 4,
        "method": 5,
    }

    def disclose(
        self,
        graph: EpistemicGraphV58,
        *,
        recipient_branch_id: str,
        limit: int,
    ) -> DisclosurePacketV58:
        graph.assert_sealed()
        candidates = [
            unit
            for unit in graph.knowledge_units
            if unit.branch_id != recipient_branch_id
            and unit.status not in {"contradicted", "revoked", "stale"}
        ]
        if not candidates:
            raise ValueError("no peer knowledge is eligible for disclosure")
        ranked = sorted(
            candidates,
            key=lambda unit: (
                self._KIND_PRIORITY[unit.kind],
                -unit.utility_hint,
                unit.branch_id,
                unit.unit_id,
            ),
        )
        selected: list[KnowledgeUnitV58] = []
        used_branches: set[str] = set()
        for unit in ranked:
            if len(selected) >= limit:
                break
            if unit.branch_id not in used_branches:
                selected.append(unit)
                used_branches.add(unit.branch_id)
        for unit in ranked:
            if len(selected) >= limit:
                break
            if unit not in selected:
                selected.append(unit)
        disclosed_hashes = sorted(
            str(unit.unit_hash) for unit in selected if unit.unit_hash
        )
        score = sum(unit.utility_hint for unit in selected) + 0.25 * len(
            {unit.branch_id for unit in selected}
        )
        return DisclosurePacketV58.seal(
            packet_id=f"disclose-{recipient_branch_id}-{graph.graph_hash[:10]}",
            recipient_branch_id=recipient_branch_id,
            source_graph_hash=graph.graph_hash,
            disclosed_unit_hashes=disclosed_hashes,
            source_branch_ids=sorted({unit.branch_id for unit in selected}),
            information_value_score=score,
        )

    @staticmethod
    def select_for_cross_pollination(
        packets: list[DisclosurePacketV58],
        *,
        limit: int,
    ) -> list[DisclosurePacketV58]:
        return sorted(
            packets,
            key=lambda item: (
                -item.information_value_score,
                item.recipient_branch_id,
            ),
        )[:limit]


class EpistemicGraphBuilderV58:
    def __init__(
        self,
        *,
        task_id: str,
        workspace_spec_hash: str,
        s0_gate_hash: str,
    ) -> None:
        self.task_id = task_id
        self.workspace_spec_hash = workspace_spec_hash
        self.s0_gate_hash = s0_gate_hash
        self.knowledge_units: list[KnowledgeUnitV58] = []
        self.disclosure_packets: list[DisclosurePacketV58] = []
        self.transfers: list[TransferHypothesisV58] = []
        self.transfer_assessments: list[TransferAssessmentV58] = []
        self.events: list[EpistemicEventV58] = []

    def _event(
        self,
        event_type: EpistemicEventTypeV58,
        *,
        subject_hashes: list[str],
        actor: Literal["model", "harness", "verifier"],
        details: dict[str, Any] | None = None,
    ) -> None:
        previous = self.events[-1].event_hash if self.events else None
        self.events.append(
            EpistemicEventV58.seal(
                sequence=len(self.events) + 1,
                previous_event_hash=previous,
                event_type=event_type,
                subject_hashes=sorted(set(subject_hashes)),
                actor=actor,
                details=details or {},
            )
        )

    def add_units(self, units: list[KnowledgeUnitV58]) -> None:
        for unit in units:
            unit.assert_sealed()
        self.knowledge_units.extend(units)
        self._event(
            "knowledge_published",
            subject_hashes=[str(unit.unit_hash) for unit in units if unit.unit_hash],
            actor="model",
            details={"unit_count": len(units)},
        )

    def freeze_initial_frontier(self) -> None:
        self._event(
            "initial_frontier_frozen",
            subject_hashes=[
                str(unit.unit_hash) for unit in self.knowledge_units if unit.unit_hash
            ],
            actor="harness",
            details={
                "branch_ids": sorted({unit.branch_id for unit in self.knowledge_units})
            },
        )

    def add_disclosures(self, packets: list[DisclosurePacketV58]) -> None:
        self.disclosure_packets.extend(packets)
        self._event(
            "disclosure_opened",
            subject_hashes=[
                str(packet.packet_hash) for packet in packets if packet.packet_hash
            ],
            actor="harness",
            details={"packet_count": len(packets)},
        )

    def add_transfers(self, transfers: list[TransferHypothesisV58]) -> None:
        for transfer in transfers:
            transfer.assert_sealed()
        self.transfers.extend(transfers)
        self._event(
            "transfer_proposed",
            subject_hashes=[
                str(transfer.transfer_hash)
                for transfer in transfers
                if transfer.transfer_hash
            ],
            actor="model",
            details={"transfer_count": len(transfers)},
        )

    def add_transfer_assessments(
        self, assessments: list[TransferAssessmentV58]
    ) -> None:
        self.transfer_assessments.extend(assessments)
        self._event(
            "transfer_assessed",
            subject_hashes=[
                str(item.assessment_hash)
                for item in assessments
                if item.assessment_hash
            ],
            actor="model",
            details={
                "assessment_count": len(assessments),
                "scientific_support_established": False,
            },
        )

    def build(
        self,
        *,
        promotion_authority_present: bool = False,
    ) -> EpistemicGraphV58:
        independence = IndependenceAssessmentV58.assess(self.knowledge_units)
        experience = ExperienceProjectionV58.project(
            self.knowledge_units,
            promotion_authority_present=promotion_authority_present,
        )
        events = list(self.events)
        previous = events[-1].event_hash if events else None
        projection_event = EpistemicEventV58.seal(
            sequence=len(events) + 1,
            previous_event_hash=previous,
            event_type="experience_projected",
            subject_hashes=[str(experience.projection_hash)],
            actor="harness",
            details={
                "eligible_count": len(experience.eligible_unit_hashes),
                "cross_task_use_permitted": experience.cross_task_use_permitted,
            },
        )
        events.append(projection_event)
        return EpistemicGraphV58.seal(
            task_id=self.task_id,
            workspace_spec_hash=self.workspace_spec_hash,
            s0_gate_hash=self.s0_gate_hash,
            knowledge_units=sorted(
                self.knowledge_units, key=lambda item: str(item.unit_hash)
            ),
            disclosure_packets=sorted(
                self.disclosure_packets, key=lambda item: str(item.packet_hash)
            ),
            transfers=sorted(self.transfers, key=lambda item: str(item.transfer_hash)),
            transfer_assessments=sorted(
                self.transfer_assessments,
                key=lambda item: str(item.assessment_hash),
            ),
            independence=independence,
            experience_projection=experience,
            events=events,
            head_event_hash=projection_event.event_hash,
        )


class EpistemicGraphStoreV58:
    """Content-addressed immutable snapshots with a replace-only HEAD pointer."""

    def __init__(self, workspace_root: str | Path) -> None:
        self.root = Path(workspace_root).resolve() / ".fma" / "epistemic" / "v58"
        self.snapshots = self.root / "snapshots"
        self.head = self.root / "HEAD"

    def save(self, graph: EpistemicGraphV58) -> Path:
        graph.assert_sealed()
        self.snapshots.mkdir(parents=True, exist_ok=True)
        assert graph.graph_hash is not None
        path = self.snapshots / f"{graph.graph_hash}.json"
        if not path.exists():
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(
                    json.dumps(
                        graph.model_dump(mode="json"),
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n"
                )
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.root / f".HEAD.{uuid4().hex}.tmp"
        temporary.write_text(graph.graph_hash + "\n", encoding="utf-8")
        temporary.replace(self.head)
        return path

    def load_head(self) -> EpistemicGraphV58 | None:
        if not self.head.is_file():
            return None
        graph_hash = self.head.read_text(encoding="utf-8").strip()
        if not graph_hash:
            return None
        path = self.snapshots / f"{graph_hash}.json"
        graph = EpistemicGraphV58.model_validate_json(path.read_text(encoding="utf-8"))
        graph.assert_sealed()
        return graph

    def summary(self) -> dict[str, Any] | None:
        graph = self.load_head()
        if graph is None:
            return None
        return {
            "schema_version": graph.schema_version,
            "graph_hash": graph.graph_hash,
            "knowledge_unit_count": len(graph.knowledge_units),
            "branch_count": len(graph.independence.branch_origin_hashes),
            "effective_independent_branches": (
                graph.independence.effective_independent_branches
            ),
            "independence_passed": graph.independence.passed,
            "independence_scope": "origin_separation_only",
            "scientific_independence_established": False,
            "disclosure_packet_count": len(graph.disclosure_packets),
            "transfer_count": len(graph.transfers),
            "transfer_assessment_count": len(graph.transfer_assessments),
            "cross_task_experience_count": len(
                graph.experience_projection.eligible_unit_hashes
            ),
            "cross_task_use_permitted": (
                graph.experience_projection.cross_task_use_permitted
            ),
        }


__all__ = [
    "DisclosurePacketV58",
    "EpistemicEventV58",
    "EpistemicGraphBuilderV58",
    "EpistemicGraphStoreV58",
    "EpistemicGraphV58",
    "ExperienceProjectionV58",
    "IndependenceAssessmentV58",
    "KnowledgeBrokerV58",
    "KnowledgeDraftV58",
    "KnowledgeUnitV58",
    "S1ExplorationBudgetV58",
    "TransferAssessmentDraftV58",
    "TransferAssessmentV58",
    "TransferDraftV58",
    "TransferHypothesisV58",
]
