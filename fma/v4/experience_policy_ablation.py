from __future__ import annotations

import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore
from fma.v2.schemas import Identifier, Sha256, _assert_timezone


RetrievalModeV40 = Literal["no_memory", "vector", "graph"]
FrontierPolicyV40 = Literal["linear", "greedy", "diversity", "search"]


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


def _assert_vector(values: list[float], name: str) -> None:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError(f"{name} must be a non-empty finite vector")


def _cosine(left: list[float], right: list[float]) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape:
        raise ValueError("embedding dimensions differ")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator == 0.0 else float(np.dot(a, b) / denominator)


class ExperienceNodeV40(StrictModel):
    node_id: Identifier
    embedding: list[float]
    outcome: Literal["succeeded", "failed"]
    recommended_family: Identifier | None = None
    utility: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    token_cost: Annotated[int, Field(ge=1)]
    node_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_node(self) -> "ExperienceNodeV40":
        _assert_vector(self.embedding, "experience embedding")
        if self.outcome == "failed" and self.recommended_family is not None:
            raise ValueError("failed experience cannot recommend a family directly")
        if self.node_hash and self.node_hash != self.content_hash():
            raise ValueError("node_hash does not match experience node")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "node_hash")

    @classmethod
    def seal(cls, **data: object) -> "ExperienceNodeV40":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"node_hash"}),
            node_hash=draft.content_hash(),
        )


class ExperienceEdgeV40(StrictModel):
    edge_id: Identifier
    source_node_hash: Sha256
    target_node_hash: Sha256
    relation: Literal["fixed_by", "requires", "similar_to"]
    edge_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_edge(self) -> "ExperienceEdgeV40":
        if self.source_node_hash == self.target_node_hash:
            raise ValueError("experience self-edge is forbidden")
        if self.edge_hash and self.edge_hash != self.content_hash():
            raise ValueError("edge_hash does not match experience edge")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "edge_hash")

    @classmethod
    def seal(cls, **data: object) -> "ExperienceEdgeV40":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"edge_hash"}),
            edge_hash=draft.content_hash(),
        )


class ExperienceGraphV40(StrictModel):
    graph_id: Identifier
    nodes: Annotated[list[ExperienceNodeV40], Field(min_length=1)]
    edges: list[ExperienceEdgeV40]
    graph_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_graph(self) -> "ExperienceGraphV40":
        hashes = [node.node_hash for node in self.nodes]
        ids = [node.node_id for node in self.nodes]
        if len(hashes) != len(set(hashes)) or len(ids) != len(set(ids)):
            raise ValueError("experience graph contains duplicate nodes")
        dimensions = {len(node.embedding) for node in self.nodes}
        if len(dimensions) != 1:
            raise ValueError("experience embeddings need one dimension")
        known = set(hashes)
        for edge in self.edges:
            if edge.source_node_hash not in known or edge.target_node_hash not in known:
                raise ValueError("experience edge references an unknown node")
        if self.graph_hash and self.graph_hash != self.content_hash():
            raise ValueError("graph_hash does not match experience graph")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "graph_hash")

    def assert_sealed(self) -> None:
        if not self.graph_hash or self.graph_hash != self.content_hash():
            raise ValueError("experience graph is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ExperienceGraphV40":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"graph_hash"}),
            graph_hash=draft.content_hash(),
        )


class FrontierOptionV40(StrictModel):
    option_id: Identifier
    family: Identifier
    predicted_utility: Annotated[float, Field(ge=0, le=2, allow_inf_nan=False)]
    cost: Annotated[int, Field(ge=1)]
    diversity_embedding: list[float]

    @model_validator(mode="after")
    def validate_option(self) -> "FrontierOptionV40":
        _assert_vector(self.diversity_embedding, "frontier diversity embedding")
        return self


class ExperiencePolicyCaseV40(StrictModel):
    case_id: Identifier
    query_embedding: list[float]
    options: Annotated[list[FrontierOptionV40], Field(min_length=2)]
    hidden_rewards: dict[Identifier, Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]]
    relevant_memory_node_ids: Annotated[list[Identifier], Field(min_length=1)]
    execution_budget: Annotated[int, Field(ge=1)] = 2

    @model_validator(mode="after")
    def validate_case(self) -> "ExperiencePolicyCaseV40":
        _assert_vector(self.query_embedding, "query embedding")
        option_ids = [item.option_id for item in self.options]
        if len(option_ids) != len(set(option_ids)):
            raise ValueError("frontier option ids must be unique")
        if set(self.hidden_rewards) != set(option_ids):
            raise ValueError("hidden rewards must cover exactly the frontier")
        dimensions = {len(item.diversity_embedding) for item in self.options}
        if len(dimensions) != 1:
            raise ValueError("frontier diversity dimensions differ")
        return self


class ExperiencePolicyBenchmarkV40(StrictModel):
    schema_version: Literal["4.0"] = "4.0"
    benchmark_id: Identifier
    experience_graph: ExperienceGraphV40
    cases: Annotated[list[ExperiencePolicyCaseV40], Field(min_length=4)]
    retrieval_limit: Annotated[int, Field(ge=1, le=8)] = 2
    memory_boost: Annotated[float, Field(ge=0, le=2, allow_inf_nan=False)] = 1.0
    success_threshold: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0.8
    created_at: datetime
    benchmark_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_benchmark(self) -> "ExperiencePolicyBenchmarkV40":
        _assert_timezone(self.created_at, "created_at")
        dimension = len(self.experience_graph.nodes[0].embedding)
        if any(len(case.query_embedding) != dimension for case in self.cases):
            raise ValueError("case query differs from experience embedding dimension")
        known_ids = {node.node_id for node in self.experience_graph.nodes}
        if any(
            memory_id not in known_ids
            for case in self.cases
            for memory_id in case.relevant_memory_node_ids
        ):
            raise ValueError("case references unknown relevant memory")
        if self.benchmark_hash and self.benchmark_hash != self.content_hash():
            raise ValueError("benchmark_hash does not match policy benchmark")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "benchmark_hash")

    def assert_sealed(self) -> None:
        if not self.benchmark_hash or self.benchmark_hash != self.content_hash():
            raise ValueError("experience-policy benchmark is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ExperiencePolicyBenchmarkV40":
        data.setdefault("created_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"benchmark_hash"}),
            benchmark_hash=draft.content_hash(),
        )


class PolicyArmResultV40(StrictModel):
    retrieval: RetrievalModeV40
    frontier_policy: FrontierPolicyV40
    case_count: Annotated[int, Field(ge=1)]
    success_count: Annotated[int, Field(ge=0)]
    success_rate: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    mean_hidden_reward: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    mean_execution_cost: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    relevant_memory_recall: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    selections: dict[Identifier, list[Identifier]]


class ExperiencePolicyAblationReportV40(StrictModel):
    schema_version: Literal["4.0"] = "4.0"
    benchmark_hash: Sha256
    arms: Annotated[list[PolicyArmResultV40], Field(min_length=12, max_length=12)]
    best_success_arms: list[str]
    graph_retrieval_greedy_lift: Annotated[float, Field(allow_inf_nan=False)]
    scope: Literal["synthetic_policy_harness_only"] = "synthetic_policy_harness_only"
    real_world_modeling_claim_permitted: Literal[False] = False
    created_at: datetime
    report_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "ExperiencePolicyAblationReportV40":
        _assert_timezone(self.created_at, "created_at")
        keys = [(item.retrieval, item.frontier_policy) for item in self.arms]
        if len(keys) != len(set(keys)):
            raise ValueError("policy report contains duplicate arms")
        if self.report_hash and self.report_hash != self.content_hash():
            raise ValueError("report_hash does not match policy ablation")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "report_hash")

    def assert_sealed(self) -> None:
        if not self.report_hash or self.report_hash != self.content_hash():
            raise ValueError("experience-policy report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ExperiencePolicyAblationReportV40":
        data.setdefault("created_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"report_hash"}),
            report_hash=draft.content_hash(),
        )


def retrieve_experience_v40(
    graph: ExperienceGraphV40,
    query: list[float],
    mode: RetrievalModeV40,
    *,
    limit: int,
) -> list[ExperienceNodeV40]:
    graph.assert_sealed()
    if mode == "no_memory":
        return []
    ranked = sorted(
        graph.nodes,
        key=lambda node: (-_cosine(query, node.embedding), node.node_id),
    )
    if mode == "vector":
        return ranked[:1]
    selected = [ranked[0]]
    by_hash = {node.node_hash: node for node in graph.nodes}
    neighbor_hashes = sorted(
        {
            edge.target_node_hash
            for edge in graph.edges
            if edge.source_node_hash == ranked[0].node_hash
            and edge.relation in {"fixed_by", "requires"}
        }
    )
    for node_hash in neighbor_hashes:
        if len(selected) >= limit:
            break
        selected.append(by_hash[node_hash])
    for node in ranked[1:]:
        if len(selected) >= limit:
            break
        if node.node_hash not in {item.node_hash for item in selected}:
            selected.append(node)
    return selected


def _adjusted_scores(
    options: list[FrontierOptionV40],
    memories: list[ExperienceNodeV40],
    memory_boost: float,
) -> dict[str, float]:
    boosts: dict[str, float] = {}
    for memory in memories:
        if memory.outcome == "succeeded" and memory.recommended_family:
            boosts[memory.recommended_family] = max(
                boosts.get(memory.recommended_family, 0.0), memory.utility
            )
    return {
        option.option_id: option.predicted_utility
        + memory_boost * boosts.get(option.family, 0.0)
        for option in options
    }


def select_frontier_v40(
    options: list[FrontierOptionV40],
    scores: dict[str, float],
    policy: FrontierPolicyV40,
    *,
    budget: int,
) -> list[FrontierOptionV40]:
    if policy == "linear":
        selected: list[FrontierOptionV40] = []
        spent = 0
        for option in options:
            if spent + option.cost <= budget:
                selected.append(option)
                spent += option.cost
        return selected
    if policy == "greedy":
        ranked = sorted(
            options,
            key=lambda item: (-scores[item.option_id] / item.cost, item.option_id),
        )
        selected = []
        spent = 0
        for option in ranked:
            if spent + option.cost <= budget:
                selected.append(option)
                spent += option.cost
        return selected
    if policy == "diversity":
        remaining = list(options)
        selected = []
        spent = 0
        while remaining:
            feasible = [item for item in remaining if spent + item.cost <= budget]
            if not feasible:
                break
            def value(item: FrontierOptionV40) -> tuple[float, str]:
                redundancy = max(
                    (_cosine(item.diversity_embedding, chosen.diversity_embedding)
                     for chosen in selected),
                    default=0.0,
                )
                return scores[item.option_id] - 0.5 * redundancy, item.option_id
            chosen = sorted(feasible, key=lambda item: (-value(item)[0], value(item)[1]))[0]
            selected.append(chosen)
            spent += chosen.cost
            remaining.remove(chosen)
        return selected
    best: tuple[float, tuple[str, ...], list[FrontierOptionV40]] | None = None
    for length in range(1, len(options) + 1):
        for subset in itertools.combinations(options, length):
            if sum(item.cost for item in subset) > budget:
                continue
            pair_diversity = sum(
                1.0 - _cosine(left.diversity_embedding, right.diversity_embedding)
                for left, right in itertools.combinations(subset, 2)
            )
            objective = sum(scores[item.option_id] for item in subset) + 0.6 * pair_diversity
            key = tuple(sorted(item.option_id for item in subset))
            candidate = (objective, key, list(subset))
            if best is None or objective > best[0] or (
                math.isclose(objective, best[0]) and key < best[1]
            ):
                best = candidate
    return [] if best is None else best[2]


def evaluate_experience_policy_benchmark_v40(
    benchmark: ExperiencePolicyBenchmarkV40,
    *,
    evaluated_at: datetime,
) -> ExperiencePolicyAblationReportV40:
    benchmark.assert_sealed()
    arms: list[PolicyArmResultV40] = []
    for retrieval in ("no_memory", "vector", "graph"):
        for policy in ("linear", "greedy", "diversity", "search"):
            successes = 0
            rewards: list[float] = []
            costs: list[int] = []
            recalls: list[float] = []
            selections: dict[str, list[str]] = {}
            for case in benchmark.cases:
                memories = retrieve_experience_v40(
                    benchmark.experience_graph,
                    case.query_embedding,
                    retrieval,
                    limit=benchmark.retrieval_limit,
                )
                memory_ids = {item.node_id for item in memories}
                recalls.append(
                    len(memory_ids & set(case.relevant_memory_node_ids))
                    / len(case.relevant_memory_node_ids)
                )
                scores = _adjusted_scores(
                    case.options, memories, benchmark.memory_boost
                )
                selected = select_frontier_v40(
                    case.options,
                    scores,
                    policy,
                    budget=case.execution_budget,
                )
                reward = max(
                    (case.hidden_rewards[item.option_id] for item in selected),
                    default=0.0,
                )
                cost = sum(item.cost for item in selected)
                rewards.append(reward)
                costs.append(cost)
                successes += reward >= benchmark.success_threshold
                selections[case.case_id] = [item.option_id for item in selected]
            arms.append(
                PolicyArmResultV40(
                    retrieval=retrieval,
                    frontier_policy=policy,
                    case_count=len(benchmark.cases),
                    success_count=successes,
                    success_rate=successes / len(benchmark.cases),
                    mean_hidden_reward=float(np.mean(rewards)),
                    mean_execution_cost=float(np.mean(costs)),
                    relevant_memory_recall=float(np.mean(recalls)),
                    selections=selections,
                )
            )
    best_rate = max(item.success_rate for item in arms)
    best = sorted(
        f"{item.retrieval}:{item.frontier_policy}"
        for item in arms
        if item.success_rate == best_rate
    )
    by_key = {(item.retrieval, item.frontier_policy): item for item in arms}
    lift = (
        by_key[("graph", "greedy")].success_rate
        - by_key[("no_memory", "greedy")].success_rate
    )
    return ExperiencePolicyAblationReportV40.seal(
        benchmark_hash=benchmark.benchmark_hash,
        arms=arms,
        best_success_arms=best,
        graph_retrieval_greedy_lift=lift,
        created_at=evaluated_at,
    )


def default_experience_policy_benchmark_v40(
    *, created_at: datetime | None = None
) -> ExperiencePolicyBenchmarkV40:
    families = ["family_a", "family_b", "family_c", "family_d"]
    nodes: list[ExperienceNodeV40] = []
    edges: list[ExperienceEdgeV40] = []
    cases: list[ExperiencePolicyCaseV40] = []
    for index in range(12):
        query = [0.0] * 13
        query[index] = 1.0
        failure = ExperienceNodeV40.seal(
            node_id=f"failure_{index:02d}",
            embedding=query,
            outcome="failed",
            utility=0.0,
            token_cost=20,
        )
        fix_embedding = [value * 0.8 for value in query]
        fix_embedding[-1] = 0.6
        correct_family = families[index % len(families)]
        fix = ExperienceNodeV40.seal(
            node_id=f"fix_{index:02d}",
            embedding=fix_embedding,
            outcome="succeeded",
            recommended_family=correct_family,
            utility=0.8,
            token_cost=35,
        )
        nodes.extend([failure, fix])
        edges.append(
            ExperienceEdgeV40.seal(
                edge_id=f"failure_fixed_{index:02d}",
                source_node_hash=failure.node_hash,
                target_node_hash=fix.node_hash,
                relation="fixed_by",
            )
        )
        wrong_one = families[(index + 1) % len(families)]
        wrong_two = families[(index + 2) % len(families)]
        other = families[(index + 3) % len(families)]
        options = [
            FrontierOptionV40(
                option_id=f"case_{index:02d}_decoy_1",
                family=wrong_one,
                predicted_utility=0.9,
                cost=1,
                diversity_embedding=[1.0, 0.0],
            ),
            FrontierOptionV40(
                option_id=f"case_{index:02d}_decoy_2",
                family=wrong_two,
                predicted_utility=0.88,
                cost=1,
                diversity_embedding=[0.98, 0.02],
            ),
            FrontierOptionV40(
                option_id=f"case_{index:02d}_correct",
                family=correct_family,
                predicted_utility=0.55,
                cost=1,
                diversity_embedding=[0.0, 1.0],
            ),
            FrontierOptionV40(
                option_id=f"case_{index:02d}_other",
                family=other,
                predicted_utility=0.3,
                cost=1,
                diversity_embedding=[-1.0, 0.0],
            ),
        ]
        rewards = {item.option_id: 0.1 for item in options}
        rewards[f"case_{index:02d}_correct"] = 1.0
        cases.append(
            ExperiencePolicyCaseV40(
                case_id=f"policy_case_{index:02d}",
                query_embedding=query,
                options=options,
                hidden_rewards=rewards,
                relevant_memory_node_ids=[fix.node_id],
                execution_budget=2,
            )
        )
    graph = ExperienceGraphV40.seal(
        graph_id="synthetic_failure_fix_experience_v40",
        nodes=nodes,
        edges=edges,
    )
    return ExperiencePolicyBenchmarkV40.seal(
        benchmark_id="experience_policy_ablation_v40",
        experience_graph=graph,
        cases=cases,
        created_at=created_at or datetime.now(timezone.utc),
    )


def run_experience_policy_ablation_v40(
    output_root: str | Path,
    benchmark: ExperiencePolicyBenchmarkV40,
    *,
    evaluated_at: datetime,
    run_id: str = "v4_experience_policy_ablation",
) -> tuple[RunStore, ExperiencePolicyAblationReportV40]:
    benchmark.assert_sealed()
    store = RunStore(output_root, run_id=run_id)
    benchmark_ref = store.put_artifact("experience_policy_benchmark_v40", benchmark)
    report = evaluate_experience_policy_benchmark_v40(
        benchmark, evaluated_at=evaluated_at
    )
    report_ref = store.put_artifact("experience_policy_ablation_report_v40", report)
    store.emit(
        "experience_policy_ablation_completed_v40",
        {
            "benchmark_ref": benchmark_ref.model_dump(mode="json"),
            "report_ref": report_ref.model_dump(mode="json"),
        },
    )
    if not verify_experience_policy_ablation_v40(store.run_directory):
        raise RuntimeError("experience-policy ablation failed replay verification")
    return store, report


def verify_experience_policy_ablation_v40(run_directory: str | Path) -> bool:
    try:
        store = RunStore.open_existing(run_directory)
        refs: list[ArtifactRef] = []
        for line in store.event_path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event["event_type"] == "artifact_committed":
                refs.append(ArtifactRef.model_validate(event["payload"]))
        benchmark_refs = [
            ref for ref in refs if ref.kind == "experience_policy_benchmark_v40"
        ]
        report_refs = [
            ref for ref in refs if ref.kind == "experience_policy_ablation_report_v40"
        ]
        if len(benchmark_refs) != 1 or len(report_refs) != 1:
            return False
        benchmark = ExperiencePolicyBenchmarkV40.model_validate(
            store.load_artifact(benchmark_refs[0])
        )
        report = ExperiencePolicyAblationReportV40.model_validate(
            store.load_artifact(report_refs[0])
        )
        benchmark.assert_sealed()
        report.assert_sealed()
        replay = evaluate_experience_policy_benchmark_v40(
            benchmark, evaluated_at=report.created_at
        )
        return replay == report
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        return False
