from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore
from fma.v2.schemas import Identifier, Sha256, _assert_timezone

from .graph_loop import (
    GraphEdgeV40,
    GraphLoopContractV40,
    GraphLoopStoreV40,
    GraphNodeV40,
)


EvolutionOperatorKindV41 = Literal[
    "replace_skeleton",
    "add_mechanism",
    "relax_assumption",
    "simplify",
    "regularize",
    "combine",
]
DevelopmentDispositionV41 = Literal["advance", "mutate", "discard"]
EvolutionTerminalStatusV41 = Literal[
    "development_champion_frozen",
    "no_development_candidate",
]
PolicyScalar = float | int | str | bool


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


def _assert_finite_mapping(values: dict[str, float], name: str) -> None:
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError(f"{name} contains a non-finite value")


class ModelEvolutionCampaignSpecV41(StrictModel):
    """Frozen development-only search contract.

    This contract may select a development champion.  It cannot grant scientific
    qualification and it cannot authorize access to a private confirmation set.
    """

    schema_version: Literal["4.1"] = "4.1"
    campaign_id: Identifier
    evaluator_epoch: Identifier
    objective: Annotated[str, Field(min_length=10, max_length=2000)]
    development_data_hash: Sha256
    required_gates: Annotated[
        list[Identifier], Field(min_length=1, max_length=32)
    ]
    evaluation_policy: dict[Identifier, PolicyScalar] = Field(default_factory=dict)
    max_generations: Annotated[int, Field(ge=1, le=32)] = 3
    beam_width: Annotated[int, Field(ge=1, le=64)] = 4
    max_candidates: Annotated[int, Field(ge=1, le=512)] = 16
    selection_policy: Literal["utility_diversity_beam"] = "utility_diversity_beam"
    private_confirmation_policy: Literal["sealed_once_after_development"] = (
        "sealed_once_after_development"
    )
    private_data_access_permitted: Literal[False] = False
    created_at: datetime
    spec_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "ModelEvolutionCampaignSpecV41":
        _assert_timezone(self.created_at, "created_at")
        if self.required_gates != sorted(set(self.required_gates)):
            raise ValueError("required_gates must be sorted and unique")
        if self.beam_width > self.max_candidates:
            raise ValueError("beam_width exceeds the candidate budget")
        for key, value in self.evaluation_policy.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"evaluation policy {key} is non-finite")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match model-evolution campaign")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "spec_hash")

    def assert_sealed(self) -> None:
        if not self.spec_hash or self.spec_hash != self.content_hash():
            raise ValueError("model-evolution campaign is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ModelEvolutionCampaignSpecV41":
        data.setdefault("created_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"spec_hash"}),
            spec_hash=draft.content_hash(),
        )


class ModelCandidateV41(StrictModel):
    schema_version: Literal["4.1"] = "4.1"
    candidate_id: Identifier
    generation: Annotated[int, Field(ge=0, le=32)]
    family: Identifier
    model_spec: dict[str, object]
    parent_candidate_hashes: list[Sha256] = Field(default_factory=list)
    operator_hashes: list[Sha256] = Field(default_factory=list)
    rationale: Annotated[str, Field(min_length=10, max_length=3000)]
    expected_failure_modes: Annotated[
        list[Annotated[str, Field(min_length=3, max_length=500)]],
        Field(min_length=1, max_length=16),
    ]
    proposed_by: Literal["model"] = "model"
    candidate_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_candidate(self) -> "ModelCandidateV41":
        if self.parent_candidate_hashes != sorted(
            set(self.parent_candidate_hashes)
        ):
            raise ValueError("parent_candidate_hashes must be sorted and unique")
        if self.operator_hashes != sorted(set(self.operator_hashes)):
            raise ValueError("operator_hashes must be sorted and unique")
        if self.generation == 0 and (
            self.parent_candidate_hashes or self.operator_hashes
        ):
            raise ValueError("generation-zero candidates cannot have lineage")
        if self.generation > 0 and (
            not self.parent_candidate_hashes or not self.operator_hashes
        ):
            raise ValueError("evolved candidates require parent and operator lineage")
        if self.candidate_hash and self.candidate_hash != self.content_hash():
            raise ValueError("candidate_hash does not match model candidate")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "candidate_hash")

    def assert_sealed(self) -> None:
        if not self.candidate_hash or self.candidate_hash != self.content_hash():
            raise ValueError("model candidate is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ModelCandidateV41":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"candidate_hash"}),
            candidate_hash=draft.content_hash(),
        )


class DevelopmentExecutionV41(StrictModel):
    schema_version: Literal["4.1"] = "4.1"
    candidate_hash: Sha256
    development_data_hash: Sha256
    converged: bool
    metrics: dict[Identifier, Annotated[float, Field(allow_inf_nan=False)]]
    domain_payload: dict[str, object]
    execution_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_execution(self) -> "DevelopmentExecutionV41":
        _assert_finite_mapping(self.metrics, "development execution metrics")
        if self.execution_hash and self.execution_hash != self.content_hash():
            raise ValueError("execution_hash does not match development execution")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "execution_hash")

    def assert_sealed(self) -> None:
        if not self.execution_hash or self.execution_hash != self.content_hash():
            raise ValueError("development execution is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DevelopmentExecutionV41":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"execution_hash"}),
            execution_hash=draft.content_hash(),
        )


class DevelopmentEvaluationV41(StrictModel):
    schema_version: Literal["4.1"] = "4.1"
    candidate_hash: Sha256
    execution_hash: Sha256
    evaluator_epoch: Identifier
    gates: dict[Identifier, bool]
    metrics: dict[
        Identifier, Annotated[float, Field(allow_inf_nan=False)]
    ] = Field(default_factory=dict)
    utility: Annotated[float, Field(allow_inf_nan=False)]
    diagnostic_codes: list[Identifier] = Field(default_factory=list)
    disposition: DevelopmentDispositionV41
    private_data_accessed: Literal[False] = False
    evaluation_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_evaluation(self) -> "DevelopmentEvaluationV41":
        _assert_finite_mapping(self.metrics, "development evaluation metrics")
        if self.diagnostic_codes != sorted(set(self.diagnostic_codes)):
            raise ValueError("diagnostic_codes must be sorted and unique")
        if self.disposition == "advance" and not all(self.gates.values()):
            raise ValueError("advance requires every development gate")
        if self.disposition != "advance" and all(self.gates.values()):
            raise ValueError("a fully passing candidate must advance")
        if self.evaluation_hash and self.evaluation_hash != self.content_hash():
            raise ValueError("evaluation_hash does not match development evaluation")
        return self

    def content_hash(self) -> str:
        data = self.model_dump(mode="json", exclude={"evaluation_hash"})
        # V4.1 runs created before metric persistence omitted this field.
        if not self.metrics:
            data.pop("metrics", None)
        return sha256_value(data)

    def assert_sealed(self) -> None:
        if not self.evaluation_hash or self.evaluation_hash != self.content_hash():
            raise ValueError("development evaluation is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DevelopmentEvaluationV41":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evaluation_hash"}),
            evaluation_hash=draft.content_hash(),
        )


class FailureSignatureV41(StrictModel):
    schema_version: Literal["4.1"] = "4.1"
    candidate_hash: Sha256
    evaluation_hash: Sha256
    failed_gates: Annotated[list[Identifier], Field(min_length=1)]
    diagnostic_codes: Annotated[list[Identifier], Field(min_length=1)]
    sanitized_summary: Annotated[str, Field(min_length=10, max_length=2000)]
    private_data_exposed: Literal[False] = False
    failure_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_failure(self) -> "FailureSignatureV41":
        if self.failed_gates != sorted(set(self.failed_gates)):
            raise ValueError("failed_gates must be sorted and unique")
        if self.diagnostic_codes != sorted(set(self.diagnostic_codes)):
            raise ValueError("diagnostic_codes must be sorted and unique")
        if self.failure_hash and self.failure_hash != self.content_hash():
            raise ValueError("failure_hash does not match failure signature")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "failure_hash")

    def assert_sealed(self) -> None:
        if not self.failure_hash or self.failure_hash != self.content_hash():
            raise ValueError("failure signature is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "FailureSignatureV41":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"failure_hash"}),
            failure_hash=draft.content_hash(),
        )


class EvolutionOperatorV41(StrictModel):
    schema_version: Literal["4.1"] = "4.1"
    operator_id: Identifier
    kind: EvolutionOperatorKindV41
    source_candidate_hash: Sha256
    source_evaluation_hash: Sha256
    failure_signature_hash: Sha256
    diagnostic_codes: Annotated[list[Identifier], Field(min_length=1)]
    target_family: Identifier
    rationale: Annotated[str, Field(min_length=10, max_length=2000)]
    operator_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_operator(self) -> "EvolutionOperatorV41":
        if self.diagnostic_codes != sorted(set(self.diagnostic_codes)):
            raise ValueError("diagnostic_codes must be sorted and unique")
        if self.operator_hash and self.operator_hash != self.content_hash():
            raise ValueError("operator_hash does not match evolution operator")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "operator_hash")

    def assert_sealed(self) -> None:
        if not self.operator_hash or self.operator_hash != self.content_hash():
            raise ValueError("evolution operator is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EvolutionOperatorV41":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"operator_hash"}),
            operator_hash=draft.content_hash(),
        )


class EvolutionProposalV41(StrictModel):
    operator: EvolutionOperatorV41
    candidate: ModelCandidateV41
    priority: Annotated[float, Field(allow_inf_nan=False)] = 0.0

    @model_validator(mode="after")
    def validate_proposal(self) -> "EvolutionProposalV41":
        self.operator.assert_sealed()
        self.candidate.assert_sealed()
        if self.operator.source_candidate_hash not in (
            self.candidate.parent_candidate_hashes
        ):
            raise ValueError("evolved candidate does not name the operator parent")
        if self.operator.operator_hash not in self.candidate.operator_hashes:
            raise ValueError("evolved candidate does not name its operator")
        if self.operator.target_family != self.candidate.family:
            raise ValueError("operator target family differs from candidate family")
        return self


class DevelopmentChampionReceiptV41(StrictModel):
    schema_version: Literal["4.1"] = "4.1"
    campaign_spec_hash: Sha256
    candidate_hash: Sha256
    evaluation_hash: Sha256
    considered_evaluation_hashes: Annotated[list[Sha256], Field(min_length=1)]
    selection_policy: Literal["utility_diversity_beam"]
    status: Literal["development_champion_unqualified"] = (
        "development_champion_unqualified"
    )
    qualification_granted: Literal[False] = False
    private_confirmation_consumed: Literal[False] = False
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "DevelopmentChampionReceiptV41":
        if self.considered_evaluation_hashes != sorted(
            set(self.considered_evaluation_hashes)
        ):
            raise ValueError("considered evaluations must be sorted and unique")
        if self.evaluation_hash not in self.considered_evaluation_hashes:
            raise ValueError("champion evaluation was not considered")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match development champion")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("development champion receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DevelopmentChampionReceiptV41":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


class ModelEvolutionReportV41(StrictModel):
    schema_version: Literal["4.1"] = "4.1"
    campaign_id: Identifier
    campaign_spec_hash: Sha256
    terminal_status: EvolutionTerminalStatusV41
    generation_count: Annotated[int, Field(ge=1)]
    candidate_hashes: Annotated[list[Sha256], Field(min_length=1)]
    execution_hashes: Annotated[list[Sha256], Field(min_length=1)]
    evaluation_hashes: Annotated[list[Sha256], Field(min_length=1)]
    failure_signature_hashes: list[Sha256]
    evolution_operator_hashes: list[Sha256]
    champion_receipt_hash: Sha256 | None = None
    champion_candidate_hash: Sha256 | None = None
    graph_snapshot_hash: Sha256
    graph_replay_verified: bool
    private_confirmation_consumed: Literal[False] = False
    qualification_granted: Literal[False] = False
    real_world_execution_permitted: Literal[False] = False
    created_at: datetime
    report_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "ModelEvolutionReportV41":
        _assert_timezone(self.created_at, "created_at")
        for field_name in (
            "candidate_hashes",
            "execution_hashes",
            "evaluation_hashes",
            "failure_signature_hashes",
            "evolution_operator_hashes",
        ):
            values = getattr(self, field_name)
            if values != sorted(set(values)):
                raise ValueError(f"{field_name} must be sorted and unique")
        has_champion = self.terminal_status == "development_champion_frozen"
        if has_champion != bool(
            self.champion_receipt_hash and self.champion_candidate_hash
        ):
            raise ValueError("terminal status and champion fields differ")
        if not self.graph_replay_verified:
            raise ValueError("model-evolution report requires graph replay")
        if self.report_hash and self.report_hash != self.content_hash():
            raise ValueError("report_hash does not match model-evolution report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "report_hash")

    def assert_sealed(self) -> None:
        if not self.report_hash or self.report_hash != self.content_hash():
            raise ValueError("model-evolution report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ModelEvolutionReportV41":
        data.setdefault("created_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"report_hash"}),
            report_hash=draft.content_hash(),
        )


class ModelEvolutionAdapterV41(Protocol):
    def initial_candidates(
        self, spec: ModelEvolutionCampaignSpecV41
    ) -> list[ModelCandidateV41]: ...

    def execute(
        self,
        spec: ModelEvolutionCampaignSpecV41,
        candidate: ModelCandidateV41,
    ) -> DevelopmentExecutionV41: ...

    def evaluate(
        self,
        spec: ModelEvolutionCampaignSpecV41,
        candidate: ModelCandidateV41,
        execution: DevelopmentExecutionV41,
    ) -> DevelopmentEvaluationV41: ...

    def evolve(
        self,
        spec: ModelEvolutionCampaignSpecV41,
        candidate: ModelCandidateV41,
        evaluation: DevelopmentEvaluationV41,
        failure: FailureSignatureV41,
        next_generation: int,
    ) -> list[EvolutionProposalV41]: ...


@dataclass(frozen=True)
class ModelEvolutionOutcomeV41:
    graph: GraphLoopStoreV40
    spec: ModelEvolutionCampaignSpecV41
    candidates: list[ModelCandidateV41]
    executions: list[DevelopmentExecutionV41]
    evaluations: list[DevelopmentEvaluationV41]
    failures: list[FailureSignatureV41]
    operators: list[EvolutionOperatorV41]
    champion: DevelopmentChampionReceiptV41 | None
    report: ModelEvolutionReportV41


@dataclass(frozen=True)
class _Lineage:
    parent_candidate_hashes: tuple[str, ...] = ()
    operator_node_hashes: tuple[str, ...] = ()


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


def _load_many(store: RunStore, kind: str, model: type[StrictModel]) -> list:
    return [model.model_validate(store.load_artifact(ref)) for ref in _artifact_refs(store, kind)]


def _node_id(prefix: str, content_hash: str) -> str:
    return f"{prefix}.{content_hash[:16]}"


def _node(
    spec: ModelEvolutionCampaignSpecV41,
    *,
    node_id: str,
    node_kind: str,
    executor: str,
    created_by: str,
    artifact_hash: str,
    purpose: str,
) -> GraphNodeV40:
    return GraphNodeV40.seal(
        node_id=node_id,
        layer="modeling",
        node_kind=node_kind,
        executor=executor,
        created_by=created_by,
        artifact_hash=artifact_hash,
        purpose=purpose,
        created_at=spec.created_at,
    )


def _edge(
    spec: ModelEvolutionCampaignSpecV41,
    *,
    edge_id: str,
    source: GraphNodeV40,
    target: GraphNodeV40,
    relation: str,
    rationale: str,
) -> GraphEdgeV40:
    return GraphEdgeV40.seal(
        edge_id=edge_id,
        layer="modeling",
        source_node_hash=source.node_hash,
        target_node_hash=target.node_hash,
        relation=relation,
        rationale=rationale,
        created_at=spec.created_at,
    )


def _finish(
    graph: GraphLoopStoreV40,
    node: GraphNodeV40,
    ref: ArtifactRef,
    *,
    actor: str,
    status: str,
    summary: str,
) -> None:
    graph.record_outcome(
        node.node_hash,
        actor=actor,
        status=status,
        output_artifacts=[ref],
        summary=summary,
        outcome_id=f"{node.node_id}.outcome",
        started_at=graph.contract.created_at,
        finished_at=graph.contract.created_at,
    )


def _select_proposals(
    proposals: list[tuple[EvolutionProposalV41, float]],
    limit: int,
) -> list[EvolutionProposalV41]:
    """Keep graph diversity first, then fill the beam by priority and parent utility."""

    if limit <= 0:
        return []
    ordered = sorted(
        proposals,
        key=lambda item: (
            -item[0].priority,
            -item[1],
            item[0].candidate.candidate_hash,
        ),
    )
    selected: list[EvolutionProposalV41] = []
    seen_families: set[str] = set()
    seen_identities: set[str] = set()
    for proposal, _ in ordered:
        identity = sha256_value(
            {
                "family": proposal.candidate.family,
                "model_spec": proposal.candidate.model_spec,
            }
        )
        if identity in seen_identities:
            continue
        if proposal.candidate.family in seen_families:
            continue
        selected.append(proposal)
        seen_families.add(proposal.candidate.family)
        seen_identities.add(identity)
        if len(selected) == limit:
            return selected
    for proposal, _ in ordered:
        if proposal in selected:
            continue
        identity = sha256_value(
            {
                "family": proposal.candidate.family,
                "model_spec": proposal.candidate.model_spec,
            }
        )
        if identity in seen_identities:
            continue
        selected.append(proposal)
        seen_identities.add(identity)
        if len(selected) == limit:
            break
    return selected


def _open_completed(
    run_directory: Path,
    expected_spec: ModelEvolutionCampaignSpecV41,
) -> ModelEvolutionOutcomeV41 | None:
    graph = GraphLoopStoreV40.open_existing(run_directory)
    report_refs = _artifact_refs(graph.store, "model_evolution_report_v41")
    if not report_refs:
        return None
    if len(report_refs) != 1:
        raise RuntimeError("model-evolution run contains duplicate reports")
    specs = _load_many(
        graph.store, "model_evolution_campaign_spec_v41", ModelEvolutionCampaignSpecV41
    )
    if specs != [expected_spec]:
        raise RuntimeError("resumed model-evolution campaign differs")
    candidates = _load_many(graph.store, "model_candidate_v41", ModelCandidateV41)
    executions = _load_many(
        graph.store, "development_execution_v41", DevelopmentExecutionV41
    )
    evaluations = _load_many(
        graph.store, "development_evaluation_v41", DevelopmentEvaluationV41
    )
    failures = _load_many(graph.store, "failure_signature_v41", FailureSignatureV41)
    operators = _load_many(graph.store, "evolution_operator_v41", EvolutionOperatorV41)
    champions = _load_many(
        graph.store, "development_champion_receipt_v41", DevelopmentChampionReceiptV41
    )
    if len(champions) > 1:
        raise RuntimeError("model-evolution run contains duplicate champions")
    report = ModelEvolutionReportV41.model_validate(
        graph.store.load_artifact(report_refs[0])
    )
    outcome = ModelEvolutionOutcomeV41(
        graph=graph,
        spec=expected_spec,
        candidates=candidates,
        executions=executions,
        evaluations=evaluations,
        failures=failures,
        operators=operators,
        champion=champions[0] if champions else None,
        report=report,
    )
    if not verify_model_evolution_campaign_v41(outcome, expected_spec):
        raise RuntimeError("completed model-evolution run failed verification")
    return outcome


def run_model_evolution_campaign_v41(
    output_root: str | Path,
    spec: ModelEvolutionCampaignSpecV41,
    adapter: ModelEvolutionAdapterV41,
) -> ModelEvolutionOutcomeV41:
    spec.assert_sealed()
    root = Path(output_root).resolve()
    run_directory = root / spec.campaign_id
    if run_directory.is_dir():
        completed = _open_completed(run_directory, spec)
        if completed is not None:
            return completed
        raise RuntimeError(
            "incomplete V4.1 campaign is preserved; automatic recovery is not yet supported"
        )

    max_nodes = spec.max_candidates * 6 + 2
    graph = GraphLoopStoreV40(
        root,
        GraphLoopContractV40.seal(
            graph_id=spec.campaign_id,
            layer="modeling",
            evaluator_epoch=spec.evaluator_epoch,
            objective=spec.objective,
            max_nodes=max_nodes,
            max_outcomes=max_nodes,
            max_failures=spec.max_candidates + 1,
            max_promotions=1,
            created_at=spec.created_at,
        ),
    )
    graph.put_output("model_evolution_campaign_spec_v41", spec)

    candidates: list[ModelCandidateV41] = []
    executions: list[DevelopmentExecutionV41] = []
    evaluations: list[DevelopmentEvaluationV41] = []
    failures: list[FailureSignatureV41] = []
    operators: list[EvolutionOperatorV41] = []
    candidate_nodes: dict[str, GraphNodeV40] = {}
    evaluation_nodes: dict[str, GraphNodeV40] = {}
    lineage: dict[str, _Lineage] = {}
    seen_candidates: set[str] = set()
    seen_model_identities: set[str] = set()
    pending = sorted(
        adapter.initial_candidates(spec),
        key=lambda item: item.candidate_hash,
    )[: spec.beam_width]
    if not pending:
        raise ValueError("model-evolution adapter returned no initial candidates")
    if any(candidate.generation != 0 for candidate in pending):
        raise ValueError("initial candidates must be generation zero")

    generations_executed = 0
    for generation in range(spec.max_generations):
        if not pending or len(candidates) >= spec.max_candidates:
            break
        generations_executed += 1
        next_proposals: list[tuple[EvolutionProposalV41, float]] = []
        for candidate in pending:
            candidate.assert_sealed()
            if candidate.generation != generation:
                raise ValueError("candidate generation differs from campaign frontier")
            if candidate.candidate_hash in seen_candidates:
                continue
            model_identity = sha256_value(
                {"family": candidate.family, "model_spec": candidate.model_spec}
            )
            if model_identity in seen_model_identities:
                continue
            if len(candidates) >= spec.max_candidates:
                break
            seen_candidates.add(candidate.candidate_hash)
            seen_model_identities.add(model_identity)
            candidates.append(candidate)

            candidate_ref = graph.put_output("model_candidate_v41", candidate)
            candidate_node = _node(
                spec,
                node_id=_node_id(f"g{generation}.candidate", candidate.candidate_hash),
                node_kind="model_candidate",
                executor="model",
                created_by="model",
                artifact_hash=candidate.candidate_hash,
                purpose=f"propose generation {generation} model family {candidate.family}",
            )
            graph.add_node(candidate_node)
            candidate_nodes[candidate.candidate_hash] = candidate_node
            candidate_lineage = lineage.get(candidate.candidate_hash, _Lineage())
            for parent_hash in candidate_lineage.parent_candidate_hashes:
                parent = candidate_nodes[parent_hash]
                graph.add_edge(
                    _edge(
                        spec,
                        edge_id=_node_id(
                            "candidate.derived", sha256_value([parent_hash, candidate.candidate_hash])
                        ),
                        source=parent,
                        target=candidate_node,
                        relation="derived_from",
                        rationale="bind evolved candidate to its immutable parent",
                    )
                )
            state_nodes = {
                node.node_hash: node for node in graph.project_state().nodes
            }
            for operator_node_hash in candidate_lineage.operator_node_hashes:
                operator_node = state_nodes[operator_node_hash]
                graph.add_edge(
                    _edge(
                        spec,
                        edge_id=_node_id(
                            "candidate.operator",
                            sha256_value([operator_node_hash, candidate.candidate_hash]),
                        ),
                        source=operator_node,
                        target=candidate_node,
                        relation="requires_success",
                        rationale="candidate requires its selected evolution operator",
                    )
                )
            _finish(
                graph,
                candidate_node,
                candidate_ref,
                actor="model",
                status="succeeded",
                summary=f"sealed model candidate {candidate.family}",
            )

            execution = adapter.execute(spec, candidate)
            execution.assert_sealed()
            if execution.candidate_hash != candidate.candidate_hash:
                raise ValueError("execution belongs to another model candidate")
            if execution.development_data_hash != spec.development_data_hash:
                raise ValueError("execution used another development data snapshot")
            executions.append(execution)
            execution_ref = graph.put_output("development_execution_v41", execution)
            execution_node = _node(
                spec,
                node_id=_node_id("development.execute", execution.execution_hash),
                node_kind="execution",
                executor="harness",
                created_by="harness",
                artifact_hash=execution.execution_hash,
                purpose=f"fit and execute development candidate {candidate.family}",
            )
            graph.add_node(execution_node)
            graph.add_edge(
                _edge(
                    spec,
                    edge_id=_node_id(
                        "candidate.executes",
                        sha256_value([candidate.candidate_hash, execution.execution_hash]),
                    ),
                    source=candidate_node,
                    target=execution_node,
                    relation="requires_success",
                    rationale="only a sealed candidate may enter development execution",
                )
            )
            _finish(
                graph,
                execution_node,
                execution_ref,
                actor="harness",
                status="succeeded",
                summary="development execution completed and recorded",
            )

            evaluation = adapter.evaluate(spec, candidate, execution)
            evaluation.assert_sealed()
            if evaluation.candidate_hash != candidate.candidate_hash:
                raise ValueError("evaluation belongs to another candidate")
            if evaluation.execution_hash != execution.execution_hash:
                raise ValueError("evaluation belongs to another execution")
            if evaluation.evaluator_epoch != spec.evaluator_epoch:
                raise ValueError("evaluation used another evaluator epoch")
            if sorted(evaluation.gates) != spec.required_gates:
                raise ValueError("evaluation gate set differs from the frozen campaign")
            passed = all(evaluation.gates.values())
            if passed != (evaluation.disposition == "advance"):
                raise ValueError("evaluation disposition differs from its gates")
            evaluations.append(evaluation)
            evaluation_ref = graph.put_output("development_evaluation_v41", evaluation)
            evaluation_node = _node(
                spec,
                node_id=_node_id("development.evaluate", evaluation.evaluation_hash),
                node_kind="evaluation",
                executor="verifier",
                created_by="harness",
                artifact_hash=evaluation.evaluation_hash,
                purpose=f"independently evaluate development candidate {candidate.family}",
            )
            graph.add_node(evaluation_node)
            evaluation_nodes[evaluation.evaluation_hash] = evaluation_node
            graph.add_edge(
                _edge(
                    spec,
                    edge_id=_node_id(
                        "execution.evaluation",
                        sha256_value([execution.execution_hash, evaluation.evaluation_hash]),
                    ),
                    source=execution_node,
                    target=evaluation_node,
                    relation="requires_success",
                    rationale="development evaluation requires a recorded execution",
                )
            )
            graph.add_edge(
                _edge(
                    spec,
                    edge_id=_node_id(
                        "candidate.evaluator",
                        sha256_value([candidate.candidate_hash, evaluation.evaluation_hash]),
                    ),
                    source=candidate_node,
                    target=evaluation_node,
                    relation="evaluated_by",
                    rationale="bind the candidate to an independent development evaluator",
                )
            )
            _finish(
                graph,
                evaluation_node,
                evaluation_ref,
                actor="verifier",
                status="succeeded" if passed else "failed",
                summary=(
                    "development gates passed"
                    if passed
                    else "development gates failed; candidate requires mutation"
                ),
            )
            if passed:
                continue

            failed_gates = sorted(
                key for key, gate_passed in evaluation.gates.items() if not gate_passed
            )
            diagnostic_codes = evaluation.diagnostic_codes or [
                f"failed_{failed_gates[0]}"
            ]
            failure = FailureSignatureV41.seal(
                candidate_hash=candidate.candidate_hash,
                evaluation_hash=evaluation.evaluation_hash,
                failed_gates=failed_gates,
                diagnostic_codes=diagnostic_codes,
                sanitized_summary=(
                    "Development-only verifier rejected the candidate on "
                    + ", ".join(failed_gates)
                    + "; no private confirmation data was accessed."
                ),
            )
            failures.append(failure)
            failure_ref = graph.put_output("failure_signature_v41", failure)
            failure_node = _node(
                spec,
                node_id=_node_id("development.failure", failure.failure_hash),
                node_kind="failure",
                executor="verifier",
                created_by="harness",
                artifact_hash=failure.failure_hash,
                purpose="publish a sanitized development failure signature",
            )
            graph.add_node(failure_node)
            graph.add_edge(
                _edge(
                    spec,
                    edge_id=_node_id(
                        "evaluation.failure",
                        sha256_value([evaluation.evaluation_hash, failure.failure_hash]),
                    ),
                    source=evaluation_node,
                    target=failure_node,
                    relation="learned_from_failure",
                    rationale="failure signature is released only after evaluator rejection",
                )
            )
            _finish(
                graph,
                failure_node,
                failure_ref,
                actor="verifier",
                status="succeeded",
                summary="sanitized development failure signature committed",
            )
            if generation + 1 < spec.max_generations:
                for proposal in adapter.evolve(
                    spec,
                    candidate,
                    evaluation,
                    failure,
                    generation + 1,
                ):
                    proposal.operator.assert_sealed()
                    proposal.candidate.assert_sealed()
                    if (
                        proposal.operator.source_evaluation_hash
                        != evaluation.evaluation_hash
                        or proposal.operator.failure_signature_hash
                        != failure.failure_hash
                    ):
                        raise ValueError("evolution proposal is bound to another failure")
                    if proposal.candidate.generation != generation + 1:
                        raise ValueError("evolution proposal targets another generation")
                    if proposal.candidate.candidate_hash in seen_candidates:
                        continue
                    proposal_identity = sha256_value(
                        {
                            "family": proposal.candidate.family,
                            "model_spec": proposal.candidate.model_spec,
                        }
                    )
                    if proposal_identity in seen_model_identities:
                        continue
                    next_proposals.append((proposal, evaluation.utility))

        remaining = spec.max_candidates - len(candidates)
        selected = _select_proposals(
            next_proposals,
            min(spec.beam_width, max(remaining, 0)),
        )
        pending = []
        selected_hashes: set[str] = set()
        for proposal in selected:
            if proposal.candidate.candidate_hash in selected_hashes:
                continue
            selected_hashes.add(proposal.candidate.candidate_hash)
            operator = proposal.operator
            operators.append(operator)
            operator_ref = graph.put_output("evolution_operator_v41", operator)
            operator_node = _node(
                spec,
                node_id=operator.operator_id,
                node_kind="experience",
                executor="harness",
                created_by="harness",
                artifact_hash=operator.operator_hash,
                purpose=f"apply {operator.kind} toward {operator.target_family}",
            )
            graph.add_node(operator_node)
            source_evaluation_node = evaluation_nodes[operator.source_evaluation_hash]
            failure_node = next(
                node
                for node in graph.project_state().nodes
                if node.artifact_hash == operator.failure_signature_hash
            )
            graph.add_edge(
                _edge(
                    spec,
                    edge_id=_node_id(
                        "failure.operator",
                        sha256_value(
                            [operator.failure_signature_hash, operator.operator_hash]
                        ),
                    ),
                    source=failure_node,
                    target=operator_node,
                    relation="requires_success",
                    rationale="operator is grounded in a committed failure signature",
                )
            )
            graph.add_edge(
                _edge(
                    spec,
                    edge_id=_node_id(
                        "evaluation.operator",
                        sha256_value(
                            [source_evaluation_node.node_hash, operator.operator_hash]
                        ),
                    ),
                    source=source_evaluation_node,
                    target=operator_node,
                    relation="requires_terminal",
                    rationale="operator remains bound to the terminal evaluation",
                )
            )
            _finish(
                graph,
                operator_node,
                operator_ref,
                actor="harness",
                status="succeeded",
                summary=f"selected graph evolution operator {operator.kind}",
            )
            lineage[proposal.candidate.candidate_hash] = _Lineage(
                parent_candidate_hashes=tuple(
                    proposal.candidate.parent_candidate_hashes
                ),
                operator_node_hashes=(operator_node.node_hash,),
            )
            pending.append(proposal.candidate)

    passing = [
        evaluation for evaluation in evaluations if all(evaluation.gates.values())
    ]
    champion: DevelopmentChampionReceiptV41 | None = None
    if passing:
        best = sorted(
            passing,
            key=lambda item: (-item.utility, item.candidate_hash),
        )[0]
        champion = DevelopmentChampionReceiptV41.seal(
            campaign_spec_hash=spec.spec_hash,
            candidate_hash=best.candidate_hash,
            evaluation_hash=best.evaluation_hash,
            considered_evaluation_hashes=sorted(
                evaluation.evaluation_hash for evaluation in evaluations
            ),
            selection_policy=spec.selection_policy,
        )
        champion_ref = graph.put_output("development_champion_receipt_v41", champion)
        champion_node = _node(
            spec,
            node_id=_node_id("development.champion", champion.receipt_hash),
            node_kind="decision",
            executor="verifier",
            created_by="harness",
            artifact_hash=champion.receipt_hash,
            purpose="freeze a development champion without scientific qualification",
        )
        graph.add_node(champion_node)
        graph.add_edge(
            _edge(
                spec,
                edge_id=_node_id(
                    "candidate.champion",
                    sha256_value([best.candidate_hash, champion.receipt_hash]),
                ),
                source=candidate_nodes[best.candidate_hash],
                target=champion_node,
                relation="requires_success",
                rationale="champion receipt names a successfully proposed candidate",
            )
        )
        graph.add_edge(
            _edge(
                spec,
                edge_id=_node_id(
                    "evaluation.champion",
                    sha256_value([best.evaluation_hash, champion.receipt_hash]),
                ),
                source=evaluation_nodes[best.evaluation_hash],
                target=champion_node,
                relation="requires_success",
                rationale="champion requires a fully passing development evaluation",
            )
        )
        _finish(
            graph,
            champion_node,
            champion_ref,
            actor="verifier",
            status="succeeded",
            summary="development champion frozen; private confirmation remains sealed",
        )

    replay_verified = graph.verify()
    report = ModelEvolutionReportV41.seal(
        campaign_id=spec.campaign_id,
        campaign_spec_hash=spec.spec_hash,
        terminal_status=(
            "development_champion_frozen"
            if champion is not None
            else "no_development_candidate"
        ),
        generation_count=generations_executed,
        candidate_hashes=sorted(candidate.candidate_hash for candidate in candidates),
        execution_hashes=sorted(execution.execution_hash for execution in executions),
        evaluation_hashes=sorted(
            evaluation.evaluation_hash for evaluation in evaluations
        ),
        failure_signature_hashes=sorted(
            failure.failure_hash for failure in failures
        ),
        evolution_operator_hashes=sorted(
            operator.operator_hash for operator in operators
        ),
        champion_receipt_hash=champion.receipt_hash if champion else None,
        champion_candidate_hash=champion.candidate_hash if champion else None,
        graph_snapshot_hash=graph.project_state().snapshot.snapshot_hash,
        graph_replay_verified=replay_verified,
        created_at=spec.created_at,
    )
    graph.put_output("model_evolution_report_v41", report)
    outcome = ModelEvolutionOutcomeV41(
        graph=graph,
        spec=spec,
        candidates=candidates,
        executions=executions,
        evaluations=evaluations,
        failures=failures,
        operators=operators,
        champion=champion,
        report=report,
    )
    if not verify_model_evolution_campaign_v41(outcome, spec):
        raise RuntimeError("new model-evolution campaign failed verification")
    return outcome


def verify_model_evolution_campaign_v41(
    outcome: ModelEvolutionOutcomeV41,
    expected_spec: ModelEvolutionCampaignSpecV41,
) -> bool:
    try:
        expected_spec.assert_sealed()
        outcome.report.assert_sealed()
        if outcome.spec != expected_spec:
            return False
        for item in (
            outcome.candidates
            + outcome.executions
            + outcome.evaluations
            + outcome.failures
            + outcome.operators
        ):
            item.assert_sealed()
        if outcome.champion:
            outcome.champion.assert_sealed()
        if not outcome.graph.verify():
            return False
        state = outcome.graph.project_state()
        if state.promotions:
            return False
        if any(
            status in {"qualified", "active"}
            for status in state.snapshot.node_statuses.values()
        ):
            return False
        if outcome.report.graph_snapshot_hash != state.snapshot.snapshot_hash:
            return False
        if outcome.report.campaign_spec_hash != expected_spec.spec_hash:
            return False
        if outcome.report.candidate_hashes != sorted(
            item.candidate_hash for item in outcome.candidates
        ):
            return False
        if outcome.report.execution_hashes != sorted(
            item.execution_hash for item in outcome.executions
        ):
            return False
        if outcome.report.evaluation_hashes != sorted(
            item.evaluation_hash for item in outcome.evaluations
        ):
            return False
        if outcome.report.failure_signature_hashes != sorted(
            item.failure_hash for item in outcome.failures
        ):
            return False
        if outcome.report.evolution_operator_hashes != sorted(
            item.operator_hash for item in outcome.operators
        ):
            return False
        if any(
            sorted(evaluation.gates) != expected_spec.required_gates
            or evaluation.private_data_accessed
            for evaluation in outcome.evaluations
        ):
            return False
        if any(failure.private_data_exposed for failure in outcome.failures):
            return False
        if outcome.champion:
            evaluation = next(
                item
                for item in outcome.evaluations
                if item.evaluation_hash == outcome.champion.evaluation_hash
            )
            if not all(evaluation.gates.values()):
                return False
            if outcome.report.champion_candidate_hash != outcome.champion.candidate_hash:
                return False
        elif outcome.report.terminal_status != "no_development_candidate":
            return False
    except (ValueError, RuntimeError, KeyError, StopIteration, OSError):
        return False
    return True
