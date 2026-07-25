from __future__ import annotations

from datetime import datetime, timezone

from fma.v4 import (
    default_experience_policy_benchmark_v40,
    evaluate_experience_policy_benchmark_v40,
    retrieve_experience_v40,
    run_experience_policy_ablation_v40,
    verify_experience_policy_ablation_v40,
)


NOW = datetime(2026, 7, 23, tzinfo=timezone.utc)


def test_graph_retrieval_follows_failure_to_fix_while_vector_does_not() -> None:
    benchmark = default_experience_policy_benchmark_v40(created_at=NOW)
    case = benchmark.cases[0]

    no_memory = retrieve_experience_v40(
        benchmark.experience_graph,
        case.query_embedding,
        "no_memory",
        limit=benchmark.retrieval_limit,
    )
    vector = retrieve_experience_v40(
        benchmark.experience_graph,
        case.query_embedding,
        "vector",
        limit=benchmark.retrieval_limit,
    )
    graph = retrieve_experience_v40(
        benchmark.experience_graph,
        case.query_embedding,
        "graph",
        limit=benchmark.retrieval_limit,
    )

    assert no_memory == []
    assert [node.node_id for node in vector] == ["failure_00"]
    assert [node.node_id for node in graph] == ["failure_00", "fix_00"]


def test_all_twelve_retrieval_policy_arms_share_budget_and_report_scope() -> None:
    benchmark = default_experience_policy_benchmark_v40(created_at=NOW)
    report = evaluate_experience_policy_benchmark_v40(
        benchmark,
        evaluated_at=NOW,
    )
    by_key = {
        (arm.retrieval, arm.frontier_policy): arm
        for arm in report.arms
    }

    assert len(report.arms) == 12
    assert {arm.mean_execution_cost for arm in report.arms} == {2.0}
    assert by_key[("no_memory", "greedy")].success_rate == 0.0
    assert by_key[("vector", "greedy")].success_rate == 0.0
    assert by_key[("graph", "linear")].success_rate == 0.0
    assert by_key[("graph", "greedy")].success_rate == 1.0
    assert by_key[("graph", "diversity")].success_rate == 1.0
    assert by_key[("graph", "search")].success_rate == 1.0
    assert by_key[("graph", "greedy")].relevant_memory_recall == 1.0
    assert report.graph_retrieval_greedy_lift == 1.0
    assert report.scope == "synthetic_policy_harness_only"
    assert report.real_world_modeling_claim_permitted is False
    report.assert_sealed()


def test_ablation_run_is_persisted_and_replay_verified(tmp_path) -> None:
    benchmark = default_experience_policy_benchmark_v40(created_at=NOW)
    store, report = run_experience_policy_ablation_v40(
        tmp_path,
        benchmark,
        evaluated_at=NOW,
        run_id="v4_policy_test",
    )

    assert report.benchmark_hash == benchmark.benchmark_hash
    assert verify_experience_policy_ablation_v40(store.run_directory)
