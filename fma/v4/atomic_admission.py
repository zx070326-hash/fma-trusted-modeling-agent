from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.epistemic_graph import (
    EpistemicEdgeV22,
    EpistemicGraphStore,
    EpistemicNodeV22,
)
from fma.v2.schemas import Identifier, Sha256, _assert_timezone
from fma.v3.evidence_compiled_growth_v313 import (
    CompiledConceptLibraryV313,
    EvidenceCompiledGrowthReportV313,
    PrivateConceptAdjudicationV313,
)
from fma.v3.evidence_concept_compiler_v313 import (
    ConceptExperienceStoreV313,
    append_concept_experience_event_v313,
)


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class AtomicConceptAdmissionReceiptV40(StrictModel):
    """Global all-gates commit layered over the frozen V3.13 artifacts."""

    schema_version: Literal["4.0"] = "4.0"
    receipt_id: Identifier
    source_schema_version: Literal["3.13"] = "3.13"
    report_hash: Sha256
    adjudication_hash: Sha256
    library_hash: Sha256
    input_experience_store_hash: Sha256
    output_experience_store_hash: Sha256
    decision: Literal["committed", "rejected"]
    staged_concept_versions: dict[Identifier, Annotated[int, Field(ge=1)]]
    committed_concept_versions: dict[Identifier, Annotated[int, Field(ge=1)]]
    failed_gates: list[Identifier]
    compensating_revocation_count: Annotated[int, Field(ge=0)]
    scope: Annotated[str, Field(min_length=3)]
    real_world_execution_permitted: Literal[False] = False
    created_at: datetime
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "AtomicConceptAdmissionReceiptV40":
        _assert_timezone(self.created_at, "created_at")
        if self.failed_gates != sorted(set(self.failed_gates)):
            raise ValueError("failed_gates must be sorted and unique")
        if self.decision == "committed":
            if self.failed_gates:
                raise ValueError("committed admission cannot contain failed gates")
            if self.committed_concept_versions != self.staged_concept_versions:
                raise ValueError("atomic commit must publish the complete staged set")
            if not self.committed_concept_versions:
                raise ValueError("atomic commit needs at least one concept")
            if self.compensating_revocation_count:
                raise ValueError("committed admission cannot contain compensation")
        else:
            if self.committed_concept_versions:
                raise ValueError("rejected admission cannot expose active concepts")
            if not self.failed_gates:
                raise ValueError("rejected admission must identify a failed gate")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match atomic admission receipt")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("atomic admission receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "AtomicConceptAdmissionReceiptV40":
        data.setdefault("created_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class AtomicConceptAdmissionOutcomeV40:
    receipt: AtomicConceptAdmissionReceiptV40
    experience_store: ConceptExperienceStoreV313


def reconcile_atomic_concept_admission_v40(
    report: EvidenceCompiledGrowthReportV313,
    adjudication: PrivateConceptAdjudicationV313,
    library: CompiledConceptLibraryV313,
    experience_store: ConceptExperienceStoreV313,
    *,
    receipt_id: str,
    created_at: datetime,
    scope: str = "V3.13 evidence-compiled concepts only",
) -> AtomicConceptAdmissionOutcomeV40:
    """Apply one atomic global decision without rewriting frozen V3.13 history.

    V3.13 staged per-concept decisions before computing its global report.  If
    any global gate failed, this adapter appends compensating revocations and
    exposes an empty active view.  A passing report must commit the complete
    staged set; partial commit is rejected.
    """

    for item in (report, adjudication, library, experience_store):
        item.assert_sealed()
    if (
        report.adjudication_hash != adjudication.adjudication_hash
        or report.library_hash != library.library_hash
        or report.output_experience_store_hash != experience_store.store_hash
        or report.phase != adjudication.phase
    ):
        raise ValueError("atomic admission source binding differs")
    entries_by_id = {entry.package.concept_id: entry for entry in library.entries}
    if len(entries_by_id) != len(library.entries):
        raise ValueError("compiled concept library contains duplicate concepts")
    for item in adjudication.entries:
        library_entry = entries_by_id.get(item.concept_id)
        if (
            library_entry is None
            or library_entry.package.package_hash != item.package_hash
            or library_entry.compiled.compiled_hash != item.compiled_hash
        ):
            raise ValueError("atomic admission adjudication lineage differs")

    staged = {
        item.concept_id: 1
        for item in adjudication.entries
        if item.status == "privately_admitted"
    }
    failed_gates = sorted(name for name, passed in report.gates.items() if not passed)
    globally_ready = bool(
        report.phase == "confirmation"
        and report.ready_for_concept_admission
        and not failed_gates
        and staged
    )
    output = experience_store
    compensation_count = 0
    if globally_ready:
        if output.active_concept_versions != staged:
            raise ValueError("passing global report does not match staged active view")
        decision = "committed"
        committed = staged
    else:
        decision = "rejected"
        committed = {}
        if not failed_gates:
            failed_gates = [
                "confirmation_phase_required"
                if report.phase != "confirmation"
                else "global_atomic_admission_not_ready"
            ]
        for concept_id in sorted(output.active_concept_versions):
            library_entry = entries_by_id.get(concept_id)
            if library_entry is None:
                raise ValueError("active concept is absent from compiled library")
            output = append_concept_experience_event_v313(
                output,
                event_type="revoked",
                package=library_entry.package,
                compiled=library_entry.compiled,
                phase="post_confirmation",
                created_at=created_at,
                adjudication_hash=adjudication.adjudication_hash,
                private_evaluator_event=True,
            )
            compensation_count += 1
        if output.active_concept_versions:
            raise RuntimeError("atomic rejection failed to clear the active view")

    receipt = AtomicConceptAdmissionReceiptV40.seal(
        receipt_id=receipt_id,
        report_hash=report.report_hash,
        adjudication_hash=adjudication.adjudication_hash,
        library_hash=library.library_hash,
        input_experience_store_hash=experience_store.store_hash,
        output_experience_store_hash=output.store_hash,
        decision=decision,
        staged_concept_versions=staged,
        committed_concept_versions=committed,
        failed_gates=failed_gates,
        compensating_revocation_count=compensation_count,
        scope=scope,
        created_at=created_at,
    )
    return AtomicConceptAdmissionOutcomeV40(receipt, output)


def verify_atomic_concept_admission_v40(
    outcome: AtomicConceptAdmissionOutcomeV40,
    report: EvidenceCompiledGrowthReportV313,
    adjudication: PrivateConceptAdjudicationV313,
    library: CompiledConceptLibraryV313,
    experience_store: ConceptExperienceStoreV313,
) -> bool:
    try:
        outcome.receipt.assert_sealed()
        outcome.experience_store.assert_sealed()
        replay = reconcile_atomic_concept_admission_v40(
            report,
            adjudication,
            library,
            experience_store,
            receipt_id=outcome.receipt.receipt_id,
            created_at=outcome.receipt.created_at,
            scope=outcome.receipt.scope,
        )
        return replay == outcome
    except (KeyError, RuntimeError, TypeError, ValueError):
        return False


def register_atomic_concept_admission_v40(
    graph: EpistemicGraphStore,
    outcome: AtomicConceptAdmissionOutcomeV40,
    library: CompiledConceptLibraryV313,
    *,
    created_at: datetime,
) -> dict[str, str]:
    """Register the atomic verdict; only committed concepts become operators."""

    outcome.receipt.assert_sealed()
    library.assert_sealed()
    verdict = EpistemicNodeV22.seal(
        node_id=f"v4_atomic_{outcome.receipt.receipt_id}",
        node_kind=(
            "validation_report"
            if outcome.receipt.decision == "committed"
            else "failure_signature"
        ),
        artifact_hash=outcome.receipt.receipt_hash,
        intended_uses=[outcome.receipt.scope],
        created_at=created_at,
    )
    graph.add_node(verdict)
    hashes = {"verdict": verdict.node_hash}
    if outcome.receipt.decision == "rejected":
        return hashes
    for entry in library.entries:
        concept_id = entry.package.concept_id
        if concept_id not in outcome.receipt.committed_concept_versions:
            continue
        node = EpistemicNodeV22.seal(
            node_id=f"v4_concept_{concept_id}",
            node_kind="evolution_operator",
            artifact_hash=entry.compiled.compiled_hash,
            intended_uses=[outcome.receipt.scope],
            created_at=created_at,
        )
        graph.add_node(node)
        graph.add_edge(
            EpistemicEdgeV22.seal(
                edge_id=f"v4_atomic_supports_{concept_id}",
                source_node_hash=verdict.node_hash,
                target_node_hash=node.node_hash,
                relation="supports",
                rationale="atomic global admission supports this scoped concept",
                created_at=created_at,
            )
        )
        hashes[concept_id] = node.node_hash
    return hashes
