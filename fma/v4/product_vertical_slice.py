from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore
from fma.v2.epistemic_graph import EpistemicGraphStore
from fma.v2.schemas import Identifier, Sha256, _assert_timezone
from fma.v3.evidence_compiled_growth_v313 import (
    CompiledConceptLibraryV313,
    EvidenceCompiledGrowthReportV313,
    PrivateConceptAdjudicationV313,
    verify_evidence_compiled_growth_run_v313,
)
from fma.v3.evidence_concept_compiler_v313 import ConceptExperienceStoreV313
from fma.v3.open_set_concept_evolution_v312 import verify_concept_evolution_run_v312
from fma.v3.representation_invariant_topology_v311 import (
    verify_representation_topology_run_v311,
)

from .atomic_admission import (
    AtomicConceptAdmissionOutcomeV40,
    AtomicConceptAdmissionReceiptV40,
    reconcile_atomic_concept_admission_v40,
    register_atomic_concept_admission_v40,
    verify_atomic_concept_admission_v40,
)
from .graph_loop import (
    GraphEdgeV40,
    GraphLoopContractV40,
    GraphLoopStoreV40,
    GraphNodeV40,
)


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class ProductVerticalSliceSpecV40(StrictModel):
    schema_version: Literal["4.0"] = "4.0"
    slice_id: Identifier
    evaluator_epoch: Identifier = "v4-anchor-1"
    source_v310_run: Annotated[str, Field(min_length=1)]
    development_v311_run: Annotated[str, Field(min_length=1)]
    confirmation_v311_run: Annotated[str, Field(min_length=1)]
    development_v312_run: Annotated[str, Field(min_length=1)]
    confirmation_v312_run: Annotated[str, Field(min_length=1)]
    development_v313_run: Annotated[str, Field(min_length=1)]
    confirmation_v313_run: Annotated[str, Field(min_length=1)]
    created_at: datetime
    spec_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "ProductVerticalSliceSpecV40":
        _assert_timezone(self.created_at, "created_at")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match product vertical slice")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "spec_hash")

    def assert_sealed(self) -> None:
        if not self.spec_hash or self.spec_hash != self.content_hash():
            raise ValueError("product vertical-slice spec is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ProductVerticalSliceSpecV40":
        data.setdefault("created_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"spec_hash"}),
            spec_hash=draft.content_hash(),
        )


class ProductVerticalSliceReportV40(StrictModel):
    schema_version: Literal["4.0"] = "4.0"
    slice_id: Identifier
    spec_hash: Sha256
    graph_snapshot_hash: Sha256
    source_verifications: dict[Identifier, bool]
    atomic_receipt_hash: Sha256
    atomic_decision: Literal["committed", "rejected"]
    active_concept_versions: dict[Identifier, Annotated[int, Field(ge=1)]]
    epistemic_snapshot_hash: Sha256
    terminal_status: Literal[
        "scientific_concepts_committed_v40",
        "scientific_concepts_rejected_v40",
    ]
    real_world_execution_permitted: Literal[False] = False
    created_at: datetime
    report_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "ProductVerticalSliceReportV40":
        _assert_timezone(self.created_at, "created_at")
        if not all(self.source_verifications.values()):
            raise ValueError("vertical-slice report cannot hide source verification failure")
        expected = (
            "scientific_concepts_committed_v40"
            if self.atomic_decision == "committed"
            else "scientific_concepts_rejected_v40"
        )
        if self.terminal_status != expected:
            raise ValueError("vertical-slice terminal status differs")
        if self.atomic_decision == "rejected" and self.active_concept_versions:
            raise ValueError("rejected vertical slice cannot expose active concepts")
        if self.report_hash and self.report_hash != self.content_hash():
            raise ValueError("report_hash does not match product vertical slice")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "report_hash")

    def assert_sealed(self) -> None:
        if not self.report_hash or self.report_hash != self.content_hash():
            raise ValueError("product vertical-slice report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ProductVerticalSliceReportV40":
        data.setdefault("created_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"report_hash"}),
            report_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class ProductVerticalSliceOutcomeV40:
    graph: GraphLoopStoreV40
    epistemic_graph: EpistemicGraphStore
    report: ProductVerticalSliceReportV40


def default_product_vertical_slice_spec_v40(
    root: str | Path,
    *,
    created_at: datetime | None = None,
    slice_id: str = "v4_product_vertical_slice",
) -> ProductVerticalSliceSpecV40:
    base = Path(root).resolve()
    return ProductVerticalSliceSpecV40.seal(
        slice_id=slice_id,
        source_v310_run=str(base / "experiments/iteration_18/v310_skeleton_factorial"),
        development_v311_run=str(
            base
            / "experiments/iteration_19/v311_representation_topology_development_time_recovered"
        ),
        confirmation_v311_run=str(
            base / "experiments/iteration_19/v311_representation_topology_confirmation"
        ),
        development_v312_run=str(
            base / "experiments/iteration_20/v312_open_set_concept_development"
        ),
        confirmation_v312_run=str(
            base / "experiments/iteration_20/v312_open_set_concept_confirmation"
        ),
        development_v313_run=str(
            base / "experiments/iteration_21/v313_evidence_concept_development"
        ),
        confirmation_v313_run=str(
            base / "experiments/iteration_21/v313_evidence_concept_confirmation"
        ),
        created_at=created_at or datetime.now(timezone.utc),
    )


def _event_tip(run_directory: str | Path) -> str:
    store = RunStore.open_existing(run_directory)
    record = json.loads(
        store.event_path.read_text(encoding="utf-8").splitlines()[-1]
    )
    return str(record["event_hash"])


def _artifact_refs(store: RunStore, kind: str) -> list[ArtifactRef]:
    refs: list[ArtifactRef] = []
    for line in store.event_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event["event_type"] != "artifact_committed":
            continue
        ref = ArtifactRef.model_validate(event["payload"])
        if ref.kind == kind:
            refs.append(ref)
    return refs


def _load_one(run_directory: str | Path, kind: str, model):
    store = RunStore.open_existing(run_directory)
    refs = _artifact_refs(store, kind)
    if len(refs) != 1:
        raise RuntimeError(f"vertical slice needs exactly one {kind}")
    return model.model_validate(store.load_artifact(refs[0]))


def _node(node_id: str, spec: ProductVerticalSliceSpecV40, run: str) -> GraphNodeV40:
    return GraphNodeV40.seal(
        node_id=node_id,
        layer="modeling",
        node_kind="evaluation",
        executor="verifier",
        created_by="harness",
        artifact_hash=sha256_value(
            {"spec_hash": spec.spec_hash, "source_event_tip": _event_tip(run)}
        ),
        purpose=f"independently replay {node_id}",
        created_at=spec.created_at,
    )


def _ensure_graph(
    output_root: str | Path, spec: ProductVerticalSliceSpecV40
) -> GraphLoopStoreV40:
    graph_directory = Path(output_root).resolve() / spec.slice_id
    if graph_directory.is_dir():
        graph = GraphLoopStoreV40.open_existing(graph_directory)
        refs = _artifact_refs(graph.store, "product_vertical_slice_spec_v40")
        if len(refs) != 1:
            raise RuntimeError("resumed graph lacks one vertical-slice spec")
        frozen = ProductVerticalSliceSpecV40.model_validate(
            graph.store.load_artifact(refs[0])
        )
        if frozen != spec:
            raise ValueError("resumed vertical slice uses another frozen spec")
        return graph
    contract = GraphLoopContractV40.seal(
        graph_id=spec.slice_id,
        layer="modeling",
        evaluator_epoch=spec.evaluator_epoch,
        objective="replay V3.11 through V3.13 and atomically adjudicate concepts",
        max_nodes=8,
        max_outcomes=8,
        max_failures=2,
        max_promotions=2,
        created_at=spec.created_at,
    )
    graph = GraphLoopStoreV40(output_root, contract)
    graph.put_output("product_vertical_slice_spec_v40", spec)
    nodes = [
        _node("verify_v311", spec, spec.confirmation_v311_run),
        _node("verify_v312", spec, spec.confirmation_v312_run),
        _node("verify_v313_development", spec, spec.development_v313_run),
        _node("verify_v313_confirmation", spec, spec.confirmation_v313_run),
        _node("atomic_concept_admission", spec, spec.confirmation_v313_run),
    ]
    for node in nodes:
        graph.add_node(node)
    for index, (source, target) in enumerate(zip(nodes, nodes[1:]), start=1):
        graph.add_edge(
            GraphEdgeV40.seal(
                edge_id=f"slice_dependency_{index}",
                layer="modeling",
                source_node_hash=source.node_hash,
                target_node_hash=target.node_hash,
                relation="requires_success",
                rationale="frozen predecessor must independently verify",
                created_at=spec.created_at,
            )
        )
    return graph


def _verification_for_node(
    node_id: str, spec: ProductVerticalSliceSpecV40
) -> bool:
    if node_id == "verify_v311":
        return verify_representation_topology_run_v311(
            spec.confirmation_v311_run,
            source_v310_run_directory=spec.source_v310_run,
            development_run_directory=spec.development_v311_run,
        )
    if node_id == "verify_v312":
        return verify_concept_evolution_run_v312(
            spec.confirmation_v312_run,
            source_run_directory=spec.confirmation_v311_run,
            development_run_directory=spec.development_v312_run,
        )
    if node_id == "verify_v313_development":
        return verify_evidence_compiled_growth_run_v313(
            spec.development_v313_run,
            source_v312_run_directory=spec.confirmation_v312_run,
        )
    if node_id == "verify_v313_confirmation":
        return verify_evidence_compiled_growth_run_v313(
            spec.confirmation_v313_run,
            source_v312_run_directory=spec.confirmation_v312_run,
            development_run_directory=spec.development_v313_run,
        )
    raise KeyError(node_id)


def _ensure_epistemic_graph(
    output_root: str | Path, spec: ProductVerticalSliceSpecV40
) -> EpistemicGraphStore:
    graph_id = f"{spec.slice_id}_epistemic"
    directory = Path(output_root).resolve() / "epistemic" / graph_id
    if directory.is_dir():
        return EpistemicGraphStore.open_existing(directory)
    return EpistemicGraphStore(Path(output_root).resolve() / "epistemic", graph_id)


def run_product_vertical_slice_v40(
    output_root: str | Path,
    spec: ProductVerticalSliceSpecV40,
) -> ProductVerticalSliceOutcomeV40:
    spec.assert_sealed()
    graph = _ensure_graph(output_root, spec)
    verifications: dict[str, bool] = {}
    atomic_outcome = None
    while True:
        state = graph.project_state()
        frontier = [
            state.node_by_hash(node_hash)
            for node_hash in state.snapshot.frontier_node_hashes
        ]
        if not frontier:
            break
        node = sorted(frontier, key=lambda item: item.node_id)[0]
        if node.node_id == "atomic_concept_admission":
            report = _load_one(
                spec.confirmation_v313_run,
                "evidence_compiled_growth_report_v313",
                EvidenceCompiledGrowthReportV313,
            )
            adjudication = _load_one(
                spec.confirmation_v313_run,
                "private_concept_adjudication_v313",
                PrivateConceptAdjudicationV313,
            )
            library = _load_one(
                spec.confirmation_v313_run,
                "compiled_concept_library_v313",
                CompiledConceptLibraryV313,
            )
            experience = _load_one(
                spec.confirmation_v313_run,
                "concept_experience_store_v313",
                ConceptExperienceStoreV313,
            )
            atomic_outcome = reconcile_atomic_concept_admission_v40(
                report,
                adjudication,
                library,
                experience,
                receipt_id=f"{spec.slice_id}_atomic",
                created_at=spec.created_at,
            )
            passed = verify_atomic_concept_admission_v40(
                atomic_outcome, report, adjudication, library, experience
            )
            refs = [
                graph.put_output(
                    "atomic_concept_admission_receipt_v40", atomic_outcome.receipt
                ),
                graph.put_output(
                    "atomic_concept_experience_store_v40",
                    atomic_outcome.experience_store,
                ),
            ]
        else:
            passed = _verification_for_node(node.node_id, spec)
            verifications[node.node_id] = passed
            refs = [
                graph.put_output(
                    "vertical_slice_verification_v40",
                    {
                        "node_id": node.node_id,
                        "passed": passed,
                        "source_artifact_hash": node.artifact_hash,
                    },
                )
            ]
        graph.record_outcome(
            node.node_hash,
            actor="verifier",
            status="succeeded" if passed else "failed",
            output_artifacts=refs,
            summary=(
                f"{node.node_id} independently verified"
                if passed
                else f"{node.node_id} failed independent verification"
            ),
            outcome_id=f"outcome_{node.node_id}",
            started_at=spec.created_at,
            finished_at=spec.created_at,
        )
        if not passed:
            raise RuntimeError(f"vertical slice stopped at {node.node_id}")

    report_refs = _artifact_refs(graph.store, "product_vertical_slice_report_v40")
    epistemic = _ensure_epistemic_graph(output_root, spec)
    if report_refs:
        if len(report_refs) != 1:
            raise RuntimeError("vertical slice has duplicate terminal reports")
        report = ProductVerticalSliceReportV40.model_validate(
            graph.store.load_artifact(report_refs[0])
        )
        return ProductVerticalSliceOutcomeV40(graph, epistemic, report)

    if atomic_outcome is None:
        receipt = _load_one(
            graph.run_directory,
            "atomic_concept_admission_receipt_v40",
            AtomicConceptAdmissionReceiptV40,
        )
        experience = _load_one(
            graph.run_directory,
            "atomic_concept_experience_store_v40",
            ConceptExperienceStoreV313,
        )
        atomic_outcome = AtomicConceptAdmissionOutcomeV40(receipt, experience)

    library = _load_one(
        spec.confirmation_v313_run,
        "compiled_concept_library_v313",
        CompiledConceptLibraryV313,
    )
    if not epistemic.project_state().nodes:
        register_atomic_concept_admission_v40(
            epistemic, atomic_outcome, library, created_at=spec.created_at
        )
    source_verifications = {
        "verify_v311": True,
        "verify_v312": True,
        "verify_v313_development": True,
        "verify_v313_confirmation": True,
    }
    terminal = (
        "scientific_concepts_committed_v40"
        if atomic_outcome.receipt.decision == "committed"
        else "scientific_concepts_rejected_v40"
    )
    report = ProductVerticalSliceReportV40.seal(
        slice_id=spec.slice_id,
        spec_hash=spec.spec_hash,
        graph_snapshot_hash=graph.project_state().snapshot.snapshot_hash,
        source_verifications=source_verifications,
        atomic_receipt_hash=atomic_outcome.receipt.receipt_hash,
        atomic_decision=atomic_outcome.receipt.decision,
        active_concept_versions=atomic_outcome.experience_store.active_concept_versions,
        epistemic_snapshot_hash=epistemic.project_state().snapshot.snapshot_hash,
        terminal_status=terminal,
        created_at=spec.created_at,
    )
    graph.put_output("product_vertical_slice_report_v40", report)
    return ProductVerticalSliceOutcomeV40(graph, epistemic, report)


def verify_product_vertical_slice_v40(
    outcome: ProductVerticalSliceOutcomeV40,
    spec: ProductVerticalSliceSpecV40,
) -> bool:
    try:
        spec.assert_sealed()
        outcome.report.assert_sealed()
        if not outcome.graph.verify() or not outcome.epistemic_graph.verify():
            return False
        state = outcome.graph.project_state()
        return bool(
            outcome.report.spec_hash == spec.spec_hash
            and outcome.report.graph_snapshot_hash == state.snapshot.snapshot_hash
            and all(status == "succeeded" for status in state.snapshot.node_statuses.values())
            and outcome.report.epistemic_snapshot_hash
            == outcome.epistemic_graph.project_state().snapshot.snapshot_hash
        )
    except (KeyError, RuntimeError, TypeError, ValueError):
        return False
