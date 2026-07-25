from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal, Protocol

from pydantic import Field, model_validator

from .schemas import (
    EvidenceSnapshot,
    DiscoveryArtifactRef,
    Identifier,
    MissionContract,
    ProblemHypothesis,
    Sha256,
    StrictModel,
)


class ProblemHypothesisDraft(StrictModel):
    """Typed, untrusted candidate returned by a human, fixture, or future model."""

    schema_version: Literal["2.0"] = "2.0"
    draft_id: Identifier
    mission_spec_hash: Sha256
    evidence_snapshot_hashes: list[Sha256] = Field(min_length=1)
    statement: Annotated[str, Field(min_length=8)]
    observed_symptoms: list[Annotated[str, Field(min_length=3)]] = Field(min_length=1)
    proposed_value: Annotated[str, Field(min_length=3)]
    assumptions: list[Annotated[str, Field(min_length=3)]] = Field(default_factory=list)
    open_questions: list[Annotated[str, Field(min_length=3)]] = Field(
        default_factory=list
    )


class ProblemDiscoveryContext(StrictModel):
    """Small public context for a future problem-discovery provider."""

    schema_version: Literal["2.0"] = "2.0"
    mission_spec_hash: Sha256
    mission_summary: dict[str, object]
    evidence: dict[str, object]
    output_contract: Literal["ProblemHypothesisDraft"] = "ProblemHypothesisDraft"


class ProblemDiscoveryProposal(StrictModel):
    """Provider result before the code-owned admission and ledger transitions."""

    status: Literal["proposed", "no_result", "error"]
    draft: ProblemHypothesisDraft | None = None
    provider_observation_ref: DiscoveryArtifactRef | None = None
    terminal_code: Annotated[str, Field(max_length=128)] = ""

    @model_validator(mode="after")
    def validate_proposal(self) -> "ProblemDiscoveryProposal":
        if self.status == "proposed":
            if self.draft is None or self.provider_observation_ref is None:
                raise ValueError("proposed result needs a draft and provider observation")
            if self.terminal_code:
                raise ValueError("proposed result cannot carry a terminal code")
        elif self.draft is not None or self.provider_observation_ref is None:
            raise ValueError("terminal provider result needs an observation and no draft")
        elif not self.terminal_code:
            raise ValueError("terminal provider result needs a terminal code")
        return self


class ProblemDiscoveryExplorer(Protocol):
    """Future providers may propose drafts, but never admit or seal them."""

    def propose(self, context: ProblemDiscoveryContext) -> ProblemDiscoveryProposal: ...


class ProblemDiscoveryHarness:
    """Code-owned provenance and scope gate for problem-definition candidates."""

    policy_version = "problem_discovery_admission_v1"

    @classmethod
    def build_context(
        cls,
        mission_contract: MissionContract,
        snapshot: EvidenceSnapshot,
        *,
        at: datetime | None = None,
    ) -> ProblemDiscoveryContext:
        mission_contract.assert_active(at)
        snapshot.assert_sealed()
        cls._assert_source_is_approved(mission_contract, snapshot)
        mission_hash = mission_contract.mission.mission_spec_hash
        assert mission_hash is not None
        return ProblemDiscoveryContext(
            mission_spec_hash=mission_hash,
            mission_summary={
                "knowledge_objectives": mission_contract.mission.knowledge_objectives,
                "intended_decisions": mission_contract.mission.intended_decisions,
                "spatial_temporal_scope": mission_contract.mission.spatial_temporal_scope,
            },
            evidence=snapshot.public_context(),
        )

    @classmethod
    def admit(
        cls,
        mission_contract: MissionContract,
        snapshot: EvidenceSnapshot,
        draft: ProblemHypothesisDraft,
        *,
        admitted_at: datetime | None = None,
    ) -> ProblemHypothesis:
        """Seal a candidate's provenance; this does not certify semantic truth."""

        cls.build_context(mission_contract, snapshot, at=admitted_at)
        mission_hash = mission_contract.mission.mission_spec_hash
        snapshot_hash = snapshot.snapshot_hash
        assert mission_hash is not None
        assert snapshot_hash is not None
        if draft.mission_spec_hash != mission_hash:
            raise ValueError("problem hypothesis draft is bound to another mission")
        if set(draft.evidence_snapshot_hashes) != {snapshot_hash}:
            raise ValueError("problem hypothesis draft must bind exactly this evidence snapshot")
        return ProblemHypothesis.seal(
            hypothesis_id=draft.draft_id,
            mission_spec_hash=mission_hash,
            statement=draft.statement,
            observed_symptoms=draft.observed_symptoms,
            proposed_value=draft.proposed_value,
            assumptions=draft.assumptions,
            evidence_refs=[f"evidence_snapshot:{snapshot_hash}"],
            created_at=admitted_at or datetime.now(timezone.utc),
        )

    @staticmethod
    def _assert_source_is_approved(
        mission_contract: MissionContract,
        snapshot: EvidenceSnapshot,
    ) -> None:
        if (
            snapshot.pedigree.source_ref
            not in mission_contract.mission.approved_evidence_sources
        ):
            raise ValueError("evidence source is outside the approved mission scope")
