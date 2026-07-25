from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import Field, model_validator

from fma.hashing import canonical_json, jsonable, sha256_value

from .discovery import (
    ProblemDiscoveryContext,
    ProblemDiscoveryHarness,
    ProblemHypothesisDraft,
)
from .schemas import (
    ApprovalRecord,
    DiscoveryArtifactRef,
    DiscoveryEvent,
    DiscoveryProviderObservation,
    DiscoveryRejectionReceipt,
    EvidenceSnapshot,
    MissionContract,
    MissionSpec,
    ProblemHypothesis,
    Sha256,
    StrictModel,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", run_id):
        raise ValueError("run_id contains unsafe path characters")
    return run_id


class ProblemAdmissionOutcome(StrictModel):
    """Terminal outcome for a recorded draft, not a semantic-validity claim."""

    status: str
    snapshot_ref: DiscoveryArtifactRef
    draft_ref: DiscoveryArtifactRef
    last_event_hash: Sha256
    hypothesis: ProblemHypothesis | None = None
    hypothesis_ref: DiscoveryArtifactRef | None = None
    rejection_receipt: DiscoveryRejectionReceipt | None = None
    rejection_ref: DiscoveryArtifactRef | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> "ProblemAdmissionOutcome":
        if self.status not in {"admitted", "rejected"}:
            raise ValueError("status must be admitted or rejected")
        if self.status == "admitted":
            if self.hypothesis is None or self.hypothesis_ref is None:
                raise ValueError("admitted outcome must contain a hypothesis and reference")
            if self.rejection_receipt is not None or self.rejection_ref is not None:
                raise ValueError("admitted outcome cannot contain a rejection receipt")
        else:
            if self.rejection_receipt is None or self.rejection_ref is None:
                raise ValueError("rejected outcome must contain a receipt and reference")
            if self.hypothesis is not None or self.hypothesis_ref is not None:
                raise ValueError("rejected outcome cannot contain a hypothesis")
        return self


class DiscoveryRunState(StrictModel):
    """Projection rebuilt only from verified artifacts and event history."""

    run_id: str
    mission_spec_hash: Sha256
    approval_record_hash: Sha256
    evidence_snapshot_hashes: list[Sha256] = Field(default_factory=list)
    provider_observation_hashes: list[Sha256] = Field(default_factory=list)
    submitted_draft_artifact_hashes: list[Sha256] = Field(default_factory=list)
    admitted_hypothesis_hashes: list[Sha256] = Field(default_factory=list)
    rejected_draft_artifact_hashes: list[Sha256] = Field(default_factory=list)
    event_count: int = Field(ge=1)
    last_event_hash: Sha256


class DiscoveryRunStore:
    """Narrow append-only ledger for V2 discovery before any model-provider link.

    Every mutation is a content-addressed artifact plus a hash-chained event.
    The projector replays the recorded admission relationship rather than
    trusting an in-memory outcome or a model's self-description.
    """

    artifact_schema_version = "2.0"

    def __init__(self, output_root: str | Path, run_id: str | None = None) -> None:
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.run_id = _validate_run_id(run_id or f"discovery-{uuid4().hex[:12]}")
        self.run_directory = (self.output_root / self.run_id).resolve()
        if not self.run_directory.is_relative_to(self.output_root):
            raise ValueError("run directory escapes the configured output root")
        self.artifact_directory = self.run_directory / "artifacts"
        self.run_directory.mkdir(parents=False, exist_ok=False)
        self.artifact_directory.mkdir()
        self.event_path = self.run_directory / "events.jsonl"
        self._sequence = 0
        self._previous_event_hash: str | None = None

    @classmethod
    def open_existing(cls, run_directory: str | Path) -> "DiscoveryRunStore":
        directory = Path(run_directory).resolve()
        instance = cls.__new__(cls)
        instance.output_root = directory.parent
        instance.run_id = _validate_run_id(directory.name)
        instance.run_directory = directory
        instance.artifact_directory = directory / "artifacts"
        instance.event_path = directory / "events.jsonl"
        if not instance.artifact_directory.is_dir() or not instance.event_path.is_file():
            raise FileNotFoundError(f"not a V2 discovery run directory: {directory}")
        events = instance._verified_events()
        instance._sequence = events[-1].sequence
        instance._previous_event_hash = events[-1].event_hash
        return instance

    def start(
        self,
        mission_contract: MissionContract,
        *,
        occurred_at: datetime | None = None,
    ) -> str:
        if self._sequence:
            raise RuntimeError("discovery run has already been started")
        at = occurred_at or _utc_now()
        mission_contract.assert_active(at)
        mission_ref = self.put_artifact("mission_spec", mission_contract.mission)
        approval_ref = self.put_artifact("approval_record", mission_contract.approval)
        return self._append_event(
            "discovery_run_started", [mission_ref, approval_ref], occurred_at=at
        )

    def ingest_evidence(
        self,
        snapshot: EvidenceSnapshot,
        *,
        occurred_at: datetime | None = None,
    ) -> DiscoveryArtifactRef:
        at = occurred_at or _utc_now()
        mission_contract = self._project_contract()
        ProblemDiscoveryHarness.build_context(mission_contract, snapshot, at=at)
        state = self.project_state()
        assert snapshot.snapshot_hash is not None
        if snapshot.snapshot_hash in state.evidence_snapshot_hashes:
            raise RuntimeError("evidence snapshot has already been ingested into this run")
        snapshot_ref = self.put_artifact("evidence_snapshot", snapshot)
        self._append_event("evidence_ingested", [snapshot_ref], occurred_at=at)
        return snapshot_ref

    def build_problem_discovery_context(
        self,
        snapshot: EvidenceSnapshot,
        *,
        at: datetime | None = None,
    ) -> ProblemDiscoveryContext:
        """Build the sole model-facing context from an already-recorded snapshot."""

        occurred_at = at or _utc_now()
        mission_contract = self._project_contract()
        self._find_snapshot_ref(snapshot)
        return ProblemDiscoveryHarness.build_context(
            mission_contract, snapshot, at=occurred_at
        )

    def assert_allowed_actions(
        self,
        required_actions: set[str],
        *,
        at: datetime | None = None,
    ) -> None:
        """Code-owned permission gate for a narrowly declared provider action."""

        occurred_at = at or _utc_now()
        mission_contract = self._project_contract()
        mission_contract.assert_active(occurred_at)
        allowed = set(mission_contract.mission.allowed_actions)
        forbidden = set(mission_contract.mission.forbidden_actions)
        missing = required_actions - allowed
        conflicts = required_actions & forbidden
        approved_actions = mission_contract.approval.approved_scope.get("allowed_actions")
        if isinstance(approved_actions, list):
            missing |= required_actions - {str(action) for action in approved_actions}
        if missing or conflicts:
            details = []
            if missing:
                details.append("missing approved actions: " + ", ".join(sorted(missing)))
            if conflicts:
                details.append("forbidden actions: " + ", ".join(sorted(conflicts)))
            raise PermissionError("; ".join(details))

    def assert_problem_discovery_context(
        self,
        context: ProblemDiscoveryContext,
        *,
        at: datetime | None = None,
    ) -> tuple[EvidenceSnapshot, ProblemDiscoveryContext]:
        """Reject provider calls whose context is not the run's exact public view."""

        snapshot_hash = context.evidence.get("evidence_snapshot_hash")
        if not isinstance(snapshot_hash, str):
            raise ValueError("problem discovery context omits its evidence snapshot hash")
        snapshot = self._load_ingested_snapshot(snapshot_hash)
        expected = self.build_problem_discovery_context(snapshot, at=at)
        if context.model_dump(mode="json") != expected.model_dump(mode="json"):
            raise ValueError("problem discovery context differs from the approved run view")
        return snapshot, expected

    def record_provider_observation(
        self,
        observation: DiscoveryProviderObservation,
        *,
        occurred_at: datetime | None = None,
    ) -> DiscoveryArtifactRef:
        """Record a provider's proposed, no-result, or failed bounded invocation."""

        at = occurred_at or _utc_now()
        observation.assert_sealed()
        if observation.run_id != self.run_id:
            raise ValueError("provider observation belongs to another discovery run")
        snapshot = self._load_ingested_snapshot(observation.evidence_snapshot_hash)
        context = self.build_problem_discovery_context(snapshot, at=at)
        if observation.mission_spec_hash != context.mission_spec_hash:
            raise ValueError("provider observation is bound to another mission")
        if observation.context_hash != sha256_value(context.model_dump(mode="json")):
            raise ValueError("provider observation is bound to another public context")
        for reference in observation.trace_refs:
            self.load_artifact(reference)
        if observation.status == "proposed":
            assert observation.draft_ref is not None
            if observation.draft_ref.kind != "problem_hypothesis_draft":
                raise ValueError("proposed provider observation must reference a draft artifact")
            ProblemHypothesisDraft.model_validate(self.load_artifact(observation.draft_ref))
        state = self.project_state()
        assert observation.observation_hash is not None
        if observation.observation_hash in state.provider_observation_hashes:
            raise RuntimeError("provider observation has already been recorded")
        observation_ref = self.put_artifact("discovery_provider_observation", observation)
        self._append_event(
            "provider_observation_recorded", [observation_ref], occurred_at=at
        )
        return observation_ref

    def submit_and_admit(
        self,
        snapshot: EvidenceSnapshot,
        draft: ProblemHypothesisDraft,
        *,
        draft_ref: DiscoveryArtifactRef | None = None,
        provider_observation_ref: DiscoveryArtifactRef | None = None,
        occurred_at: datetime | None = None,
    ) -> ProblemAdmissionOutcome:
        at = occurred_at or _utc_now()
        mission_contract = self._project_contract()
        snapshot_ref = self._find_snapshot_ref(snapshot)
        if draft_ref is None and provider_observation_ref is not None:
            observation = DiscoveryProviderObservation.model_validate(
                self.load_artifact(provider_observation_ref)
            )
            if observation.status != "proposed" or observation.draft_ref is None:
                raise ValueError("provider observation does not contain a proposed draft")
            draft_ref = observation.draft_ref
        if draft_ref is None:
            draft_ref = self.put_artifact("problem_hypothesis_draft", draft)
        else:
            if draft_ref.kind != "problem_hypothesis_draft":
                raise ValueError("draft_ref must reference a problem_hypothesis_draft")
            recorded_draft = ProblemHypothesisDraft.model_validate(
                self.load_artifact(draft_ref)
            )
            if recorded_draft.model_dump(mode="json") != draft.model_dump(mode="json"):
                raise ValueError("draft_ref does not contain the submitted draft")
        if draft_ref.sha256 in self.project_state().submitted_draft_artifact_hashes:
            raise RuntimeError("draft artifact has already been submitted into this run")
        event_refs = [snapshot_ref, draft_ref]
        if provider_observation_ref is not None:
            self._assert_provider_observation_for_draft(
                provider_observation_ref, snapshot, draft_ref
            )
            event_refs.append(provider_observation_ref)
        self._append_event(
            "problem_draft_submitted", event_refs, occurred_at=at
        )
        try:
            hypothesis = ProblemDiscoveryHarness.admit(
                mission_contract, snapshot, draft, admitted_at=at
            )
        except ValueError:
            receipt = DiscoveryRejectionReceipt.seal(
                receipt_id=f"{draft.draft_id}_rejection",
                run_id=self.run_id,
                draft_artifact_hash=draft_ref.sha256,
                evidence_snapshot_hash=snapshot.snapshot_hash,
                rejection_code="admission_denied",
                created_at=at,
            )
            rejection_ref = self.put_artifact("discovery_rejection_receipt", receipt)
            event_hash = self._append_event(
                "problem_draft_rejected",
                [snapshot_ref, draft_ref, rejection_ref],
                rejection_code="admission_denied",
                occurred_at=at,
            )
            return ProblemAdmissionOutcome(
                status="rejected",
                snapshot_ref=snapshot_ref,
                draft_ref=draft_ref,
                rejection_receipt=receipt,
                rejection_ref=rejection_ref,
                last_event_hash=event_hash,
            )

        hypothesis_ref = self.put_artifact("problem_hypothesis", hypothesis)
        event_hash = self._append_event(
            "problem_hypothesis_admitted",
            [snapshot_ref, draft_ref, hypothesis_ref],
            occurred_at=at,
        )
        return ProblemAdmissionOutcome(
            status="admitted",
            snapshot_ref=snapshot_ref,
            draft_ref=draft_ref,
            hypothesis=hypothesis,
            hypothesis_ref=hypothesis_ref,
            last_event_hash=event_hash,
        )

    def put_artifact(self, kind: str, payload: object) -> DiscoveryArtifactRef:
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,79}", kind):
            raise ValueError("artifact kind must be a safe lowercase identifier")
        envelope = {
            "artifact_schema": self.artifact_schema_version,
            "kind": kind,
            "payload": jsonable(payload),
        }
        digest = sha256_value(envelope)
        path = self.artifact_directory / f"{digest}.json"
        serialized = json.dumps(
            envelope, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if sha256_value(existing) != digest:
                raise RuntimeError(f"existing artifact failed integrity check: {path}")
        else:
            path.write_text(serialized + "\n", encoding="utf-8")
        return DiscoveryArtifactRef(
            kind=kind,
            sha256=digest,
            relative_path=path.relative_to(self.run_directory).as_posix(),
        )

    def load_artifact(self, reference: DiscoveryArtifactRef) -> object:
        path = (self.run_directory / reference.relative_path).resolve()
        if not path.is_relative_to(self.artifact_directory):
            raise RuntimeError("artifact reference escapes this run's artifact directory")
        if not path.is_file():
            raise RuntimeError(f"referenced artifact is missing: {reference.relative_path}")
        envelope = json.loads(path.read_text(encoding="utf-8"))
        actual = sha256_value(envelope)
        if actual != reference.sha256:
            raise RuntimeError(
                f"artifact integrity failure for {reference.relative_path}: "
                f"expected {reference.sha256}, got {actual}"
            )
        if envelope.get("artifact_schema") != self.artifact_schema_version:
            raise RuntimeError("artifact schema version is not supported")
        if envelope.get("kind") != reference.kind:
            raise RuntimeError("artifact kind does not match its reference")
        return envelope["payload"]

    def verify(self) -> bool:
        try:
            self.project_state()
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
            return False
        return True

    def project_state(self) -> DiscoveryRunState:
        events = self._verified_events()
        if events[0].event_type != "discovery_run_started":
            raise RuntimeError("first event must start a discovery run")
        mission_ref = self._exact_ref(events[0], "mission_spec")
        approval_ref = self._exact_ref(events[0], "approval_record")
        if len(events[0].artifact_refs) != 2:
            raise RuntimeError("run start must reference exactly mission and approval artifacts")
        mission = MissionSpec.model_validate(self.load_artifact(mission_ref))
        approval = ApprovalRecord.model_validate(self.load_artifact(approval_ref))
        mission_contract = MissionContract(mission=mission, approval=approval)
        mission_contract.assert_active(events[0].occurred_at)
        assert mission.mission_spec_hash is not None
        assert approval.approval_record_hash is not None

        snapshots: dict[str, DiscoveryArtifactRef] = {}
        provider_observations: dict[str, DiscoveryProviderObservation] = {}
        submitted: dict[str, tuple[DiscoveryArtifactRef, DiscoveryArtifactRef]] = {}
        admitted_hypotheses: list[str] = []
        rejected_drafts: list[str] = []
        terminal_drafts: set[str] = set()

        for event in events[1:]:
            if event.event_type == "evidence_ingested":
                snapshot_ref = self._only_ref(event, "evidence_snapshot")
                snapshot = EvidenceSnapshot.model_validate(self.load_artifact(snapshot_ref))
                snapshot.assert_sealed()
                assert snapshot.snapshot_hash is not None
                if snapshot.snapshot_hash in snapshots:
                    raise RuntimeError("evidence snapshot was ingested more than once")
                ProblemDiscoveryHarness.build_context(
                    mission_contract, snapshot, at=event.occurred_at
                )
                snapshots[snapshot.snapshot_hash] = snapshot_ref
                continue

            if event.event_type == "provider_observation_recorded":
                observation_ref = self._only_ref(event, "discovery_provider_observation")
                observation = DiscoveryProviderObservation.model_validate(
                    self.load_artifact(observation_ref)
                )
                observation.assert_sealed()
                if observation.run_id != self.run_id:
                    raise RuntimeError("provider observation belongs to another run")
                snapshot_ref = snapshots.get(observation.evidence_snapshot_hash)
                if snapshot_ref is None:
                    raise RuntimeError("provider observation references evidence not ingested")
                snapshot = EvidenceSnapshot.model_validate(self.load_artifact(snapshot_ref))
                expected_context = ProblemDiscoveryHarness.build_context(
                    mission_contract, snapshot, at=event.occurred_at
                )
                if (
                    observation.mission_spec_hash != mission.mission_spec_hash
                    or observation.context_hash
                    != sha256_value(expected_context.model_dump(mode="json"))
                ):
                    raise RuntimeError("provider observation is bound to another context")
                for trace_ref in observation.trace_refs:
                    self.load_artifact(trace_ref)
                if observation.status == "proposed":
                    assert observation.draft_ref is not None
                    if observation.draft_ref.kind != "problem_hypothesis_draft":
                        raise RuntimeError("proposed provider observation has wrong draft kind")
                    ProblemHypothesisDraft.model_validate(
                        self.load_artifact(observation.draft_ref)
                    )
                if observation_ref.sha256 in provider_observations:
                    raise RuntimeError("provider observation was recorded more than once")
                provider_observations[observation_ref.sha256] = observation
                continue

            if event.event_type == "problem_draft_submitted":
                snapshot_ref = self._exact_ref(event, "evidence_snapshot")
                draft_ref = self._exact_ref(event, "problem_hypothesis_draft")
                if len(event.artifact_refs) not in {2, 3}:
                    raise RuntimeError("draft submission needs snapshot, draft, and optional provider observation")
                snapshot = EvidenceSnapshot.model_validate(self.load_artifact(snapshot_ref))
                snapshot.assert_sealed()
                assert snapshot.snapshot_hash is not None
                if snapshots.get(snapshot.snapshot_hash) != snapshot_ref:
                    raise RuntimeError("draft references evidence not ingested into this run")
                ProblemHypothesisDraft.model_validate(self.load_artifact(draft_ref))
                if draft_ref.sha256 in submitted:
                    raise RuntimeError("draft artifact was submitted more than once")
                if len(event.artifact_refs) == 3:
                    observation_ref = self._exact_ref(
                        event, "discovery_provider_observation"
                    )
                    observation = self._load_recorded_provider_observation(
                        provider_observations, observation_ref
                    )
                    if (
                        observation.status != "proposed"
                        or observation.draft_ref != draft_ref
                        or observation.evidence_snapshot_hash != snapshot.snapshot_hash
                    ):
                        raise RuntimeError(
                            "provider observation does not bind this submitted draft"
                        )
                submitted[draft_ref.sha256] = (snapshot_ref, draft_ref)
                continue

            if event.event_type == "problem_hypothesis_admitted":
                snapshot_ref = self._exact_ref(event, "evidence_snapshot")
                draft_ref = self._exact_ref(event, "problem_hypothesis_draft")
                hypothesis_ref = self._exact_ref(event, "problem_hypothesis")
                if len(event.artifact_refs) != 3:
                    raise RuntimeError("admission needs snapshot, draft, and hypothesis")
                self._assert_terminal_input(submitted, terminal_drafts, snapshot_ref, draft_ref)
                snapshot = EvidenceSnapshot.model_validate(self.load_artifact(snapshot_ref))
                draft = ProblemHypothesisDraft.model_validate(self.load_artifact(draft_ref))
                hypothesis = ProblemHypothesis.model_validate(self.load_artifact(hypothesis_ref))
                expected = ProblemDiscoveryHarness.admit(
                    mission_contract, snapshot, draft, admitted_at=event.occurred_at
                )
                if expected.hypothesis_hash != hypothesis.hypothesis_hash:
                    raise RuntimeError("admitted hypothesis does not match replayed admission")
                assert hypothesis.hypothesis_hash is not None
                terminal_drafts.add(draft_ref.sha256)
                admitted_hypotheses.append(hypothesis.hypothesis_hash)
                continue

            if event.event_type == "problem_draft_rejected":
                snapshot_ref = self._exact_ref(event, "evidence_snapshot")
                draft_ref = self._exact_ref(event, "problem_hypothesis_draft")
                receipt_ref = self._exact_ref(event, "discovery_rejection_receipt")
                if len(event.artifact_refs) != 3:
                    raise RuntimeError("rejection needs snapshot, draft, and receipt")
                self._assert_terminal_input(submitted, terminal_drafts, snapshot_ref, draft_ref)
                snapshot = EvidenceSnapshot.model_validate(self.load_artifact(snapshot_ref))
                draft = ProblemHypothesisDraft.model_validate(self.load_artifact(draft_ref))
                try:
                    ProblemDiscoveryHarness.admit(
                        mission_contract, snapshot, draft, admitted_at=event.occurred_at
                    )
                except ValueError:
                    pass
                else:
                    raise RuntimeError("rejected draft would pass replayed admission")
                receipt = DiscoveryRejectionReceipt.model_validate(
                    self.load_artifact(receipt_ref)
                )
                receipt.assert_sealed()
                if (
                    event.rejection_code != "admission_denied"
                    or receipt.run_id != self.run_id
                    or receipt.draft_artifact_hash != draft_ref.sha256
                    or receipt.evidence_snapshot_hash != snapshot.snapshot_hash
                    or receipt.rejection_code != event.rejection_code
                ):
                    raise RuntimeError("rejection receipt does not bind this rejected draft")
                terminal_drafts.add(draft_ref.sha256)
                rejected_drafts.append(draft_ref.sha256)
                continue

            raise RuntimeError(f"unsupported discovery event: {event.event_type}")

        return DiscoveryRunState(
            run_id=self.run_id,
            mission_spec_hash=mission.mission_spec_hash,
            approval_record_hash=approval.approval_record_hash,
            evidence_snapshot_hashes=list(snapshots),
            provider_observation_hashes=[
                observation.observation_hash
                for observation in provider_observations.values()
                if observation.observation_hash is not None
            ],
            submitted_draft_artifact_hashes=list(submitted),
            admitted_hypothesis_hashes=admitted_hypotheses,
            rejected_draft_artifact_hashes=rejected_drafts,
            event_count=len(events),
            last_event_hash=events[-1].event_hash,
        )

    def _project_contract(self) -> MissionContract:
        state = self.project_state()
        events = self._verified_events()
        mission = MissionSpec.model_validate(
            self.load_artifact(self._exact_ref(events[0], "mission_spec"))
        )
        approval = ApprovalRecord.model_validate(
            self.load_artifact(self._exact_ref(events[0], "approval_record"))
        )
        if mission.mission_spec_hash != state.mission_spec_hash:
            raise RuntimeError("projected mission does not match run state")
        return MissionContract(mission=mission, approval=approval)

    def _find_snapshot_ref(self, snapshot: EvidenceSnapshot) -> DiscoveryArtifactRef:
        snapshot.assert_sealed()
        assert snapshot.snapshot_hash is not None
        for event in self._verified_events():
            if event.event_type != "evidence_ingested":
                continue
            reference = self._only_ref(event, "evidence_snapshot")
            recorded = EvidenceSnapshot.model_validate(self.load_artifact(reference))
            if recorded.snapshot_hash == snapshot.snapshot_hash:
                return reference
        raise RuntimeError("evidence snapshot has not been ingested into this run")

    def _load_ingested_snapshot(self, snapshot_hash: str) -> EvidenceSnapshot:
        for event in self._verified_events():
            if event.event_type != "evidence_ingested":
                continue
            reference = self._only_ref(event, "evidence_snapshot")
            snapshot = EvidenceSnapshot.model_validate(self.load_artifact(reference))
            if snapshot.snapshot_hash == snapshot_hash:
                return snapshot
        raise RuntimeError("evidence snapshot has not been ingested into this run")

    def _assert_provider_observation_for_draft(
        self,
        observation_ref: DiscoveryArtifactRef,
        snapshot: EvidenceSnapshot,
        draft_ref: DiscoveryArtifactRef,
    ) -> None:
        if observation_ref.kind != "discovery_provider_observation":
            raise ValueError("provider_observation_ref has the wrong artifact kind")
        if not any(
            event.event_type == "provider_observation_recorded"
            and self._only_ref(event, "discovery_provider_observation") == observation_ref
            for event in self._verified_events()
        ):
            raise RuntimeError("provider observation has not been recorded into this run")
        observation = DiscoveryProviderObservation.model_validate(
            self.load_artifact(observation_ref)
        )
        observation.assert_sealed()
        if (
            observation.status != "proposed"
            or observation.draft_ref != draft_ref
            or observation.evidence_snapshot_hash != snapshot.snapshot_hash
        ):
            raise ValueError("provider observation does not bind the submitted draft")

    @staticmethod
    def _load_recorded_provider_observation(
        observations: dict[str, DiscoveryProviderObservation],
        observation_ref: DiscoveryArtifactRef,
    ) -> DiscoveryProviderObservation:
        observation = DiscoveryProviderObservation.model_validate(
            observations.get(observation_ref.sha256)
        )
        return observation

    def _append_event(
        self,
        event_type: str,
        artifact_refs: list[DiscoveryArtifactRef],
        *,
        rejection_code: str | None = None,
        occurred_at: datetime | None = None,
    ) -> str:
        event = DiscoveryEvent.seal(
            run_id=self.run_id,
            sequence=self._sequence + 1,
            event_type=event_type,
            artifact_refs=artifact_refs,
            rejection_code=rejection_code,
            previous_event_hash=self._previous_event_hash,
            occurred_at=occurred_at or _utc_now(),
        )
        assert event.event_hash is not None
        with self.event_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(event.model_dump(mode="json")) + "\n")
        self._sequence = event.sequence
        self._previous_event_hash = event.event_hash
        return event.event_hash

    def _verified_events(self) -> list[DiscoveryEvent]:
        if not self.event_path.is_file():
            raise RuntimeError("discovery event log is missing")
        events: list[DiscoveryEvent] = []
        previous: str | None = None
        for expected_sequence, line in enumerate(
            self.event_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            raw = json.loads(line)
            event = DiscoveryEvent.model_validate(raw)
            event.assert_sealed()
            if event.run_id != self.run_id:
                raise RuntimeError("event belongs to another run")
            if event.sequence != expected_sequence:
                raise RuntimeError("event sequence is not contiguous")
            if event.previous_event_hash != previous:
                raise RuntimeError("event hash chain is broken")
            assert event.event_hash is not None
            previous = event.event_hash
            events.append(event)
        if not events:
            raise RuntimeError("discovery run has no events")
        return events

    @staticmethod
    def _exact_ref(event: DiscoveryEvent, kind: str) -> DiscoveryArtifactRef:
        matching = [ref for ref in event.artifact_refs if ref.kind == kind]
        if len(matching) != 1:
            raise RuntimeError(f"event needs exactly one {kind} artifact reference")
        return matching[0]

    @classmethod
    def _only_ref(cls, event: DiscoveryEvent, kind: str) -> DiscoveryArtifactRef:
        reference = cls._exact_ref(event, kind)
        if len(event.artifact_refs) != 1:
            raise RuntimeError(f"event needs only one {kind} artifact reference")
        return reference

    @staticmethod
    def _assert_terminal_input(
        submitted: dict[str, tuple[DiscoveryArtifactRef, DiscoveryArtifactRef]],
        terminal_drafts: set[str],
        snapshot_ref: DiscoveryArtifactRef,
        draft_ref: DiscoveryArtifactRef,
    ) -> None:
        recorded = submitted.get(draft_ref.sha256)
        if recorded != (snapshot_ref, draft_ref):
            raise RuntimeError("terminal event does not match a prior draft submission")
        if draft_ref.sha256 in terminal_drafts:
            raise RuntimeError("draft already has a terminal admission outcome")
