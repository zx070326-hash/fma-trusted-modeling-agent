from __future__ import annotations

from pathlib import Path

import pytest

from fma.benchmark import (
    ArmMetadata,
    BenchmarkCaseResult,
    BenchmarkRunner,
    HoldoutReport,
    _arm_metadata,
    _guard_benchmark_prompt,
    _score_case,
    aggregate_results,
    compare_finite_semantics,
)
from fma.benchmark_cases import BenchmarkCase, build_fma_bench_v0
from fma.codex_driver import (
    CodexAgentOutcome,
    CodexCLIConfig,
    ExplorationRound,
    ExplorerProblemView,
)
from fma.hashing import canonical_json
from fma.schemas import OptimizationModelIR
from fma.storage import RunStore


def test_suite_is_a_deterministic_six_by_four_matrix() -> None:
    first = build_fma_bench_v0()
    second = build_fma_bench_v0()

    assert first.suite_hash == second.suite_hash
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert len(first.cases) == 24
    matrix = {(case.family, case.task_kind) for case in first.cases}
    assert len(matrix) == 24
    assert {case.family for case in first.cases} == {
        "resource_allocation",
        "knapsack",
        "assignment",
        "transportation",
        "facility_location",
        "set_covering",
    }
    assert {case.task_kind for case in first.cases} == {
        "build",
        "revise",
        "explain",
        "no_result",
    }
    assert first.suite_hash == first.content_hash()
    assert all(case.sealed_hash == case.content_hash() for case in first.cases)


def test_public_projection_excludes_every_private_canary() -> None:
    suite = build_fma_bench_v0()

    for case in suite.cases:
        public = canonical_json(ExplorerProblemView.from_contract(case.contract))
        assert case.privacy_canary not in public
        assert "acceptance_tests" not in public
        assert "frozen_hash" not in public
        assert "source_ref" not in public


def test_prompt_guard_blocks_any_suite_canary_before_inference() -> None:
    suite = build_fma_bench_v0()

    with pytest.raises(PermissionError, match=suite.cases[0].case_id):
        _guard_benchmark_prompt(suite, f"public text {suite.cases[0].privacy_canary}")


def test_arm_hash_binds_driver_budgets_and_fixture_identity() -> None:
    narrow = _arm_metadata("live_single", CodexCLIConfig(max_candidates=1))
    wide = _arm_metadata("live_single", CodexCLIConfig(max_candidates=3))
    fixture = _arm_metadata("fixture_golden", CodexCLIConfig(requested_model="ignored"))

    assert narrow.content_hash() != wide.content_hash()
    assert fixture.requested_model == "fixture-control"
    assert fixture.executable_policy == "per_case_fixture"


def test_assignment_no_result_product_is_not_linear_on_permutations() -> None:
    case = next(case for case in build_fma_bench_v0().cases if case.case_id == "as_n1")
    objectives = {
        test.test_id: test.expected_objective
        for test in case.contract.acceptance_tests
        if test.expected_feasible
    }

    # Every matrix entry occurs once across the three even permutations and
    # once across the three odd permutations. Any linear assignment objective
    # therefore has equal sums on the two groups; the product indicator does not.
    even_sum = sum(
        objectives[f"as_n1_{label}"]
        for label in ("identity", "cycle_123", "cycle_132")
    )
    odd_sum = sum(
        objectives[f"as_n1_{label}"]
        for label in ("swap_23", "swap_12", "reverse")
    )

    assert len(case.contract.decisions) == 9
    assert even_sum == -3
    assert odd_sum == 0
    assert even_sum != odd_sum


def test_every_positive_reference_matches_its_full_finite_holdout() -> None:
    suite = build_fma_bench_v0()

    for case in suite.cases:
        if case.reference_ir is None:
            assert case.expected_status == "no_result"
            continue
        report = compare_finite_semantics(case.reference_ir, case.reference_ir)
        assert report.status == "pass", case.case_id
        assert report.assignment_count == case.oracle_assignment_count


def test_finite_holdout_detects_an_objective_mutation() -> None:
    case = next(case for case in build_fma_bench_v0().cases if case.case_id == "ra_b1")
    assert case.reference_ir is not None
    raw = case.reference_ir.model_dump(mode="python", exclude={"ir_hash"})
    raw["candidate_id"] = "ra_b1_mutant"
    raw["objective"]["constant"] += 1
    mutant = OptimizationModelIR.seal(**raw)

    report = compare_finite_semantics(case.reference_ir, mutant)

    assert report.status == "fail"
    assert report.objective_mismatch_count > 0
    assert report.first_counterexample is not None


def test_case_seal_rejects_expected_label_tampering() -> None:
    case = build_fma_bench_v0().cases[0]
    raw = case.model_dump(mode="python")
    raw["sealed_hash"] = "0" * 64

    with pytest.raises(ValueError, match="sealed_hash mismatch"):
        BenchmarkCase.model_validate(raw)


def test_fixture_golden_runs_the_same_driver_and_private_holdout(tmp_path: Path) -> None:
    summary = BenchmarkRunner(tmp_path).run(
        "fixture_golden", case_ids=["ra_b1", "ra_n1"]
    )

    assert summary.aggregate.run_validity == "partial"
    assert summary.aggregate.control_pass_rate == 1.0
    assert summary.aggregate.false_promotion_count == 0
    assert summary.aggregate.privacy_failure_count == 0
    assert summary.aggregate.runtime_provenance_passed
    assert summary.aggregate.runtime_invocation_count == 2
    assert len(summary.aggregate.runtime_executable_sha256s) == 1
    assert summary.aggregate.harness_integrity_passed
    assert summary.event_chain_verified
    assert Path(summary.report_path).is_file()


def test_fixture_mutant_is_not_promoted(tmp_path: Path) -> None:
    summary = BenchmarkRunner(tmp_path).run(
        "fixture_mutant", case_ids=["ra_b1"]
    )

    assert summary.aggregate.control_pass_rate == 1.0
    assert summary.aggregate.false_promotion_count == 0
    assert summary.aggregate.validated_recall == 0.0
    assert summary.aggregate.harness_integrity_passed


def test_live_arm_requires_explicit_authorization(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="explicit authorization"):
        BenchmarkRunner(tmp_path).run("live_single", case_ids=["ra_b1"])


def test_empty_case_selection_is_rejected_instead_of_expanding_to_full_suite(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="at least one"):
        BenchmarkRunner(tmp_path).run("fixture_golden", case_ids=[])


def test_mutant_driver_error_is_not_a_passing_adversarial_control(tmp_path: Path) -> None:
    case = next(case for case in build_fma_bench_v0().cases if case.case_id == "ra_b1")
    case_output = tmp_path / "case"
    exploration = case_output / "explorations" / "run-error"
    exploration.mkdir(parents=True)
    outcome = CodexAgentOutcome(
        status="driver_error",
        stop_reason="synthetic driver failure",
        driver_error_code="synthetic_failure",
        exploration_directory=str(exploration),
        rounds=[
            ExplorationRound(
                attempt_index=1,
                driver_status="error",
                proposed_candidate_ids=[],
                public_rejections=[],
                private_rejections=[],
                assessed_candidate_ids=[],
                decision_statuses=[],
                feedback_disclosure="none",
            )
        ],
        candidate_outcomes=[],
    )

    result = _score_case(
        benchmark_run_id="run-current",
        suite=build_fma_bench_v0(),
        case=case,
        arm=_arm_metadata("fixture_mutant", CodexCLIConfig()),
        repetition=1,
        outcome=outcome,
        expected_case_output=case_output,
        privacy_passed=True,
        privacy_detail="synthetic",
    )

    assert result.infrastructure_failure
    assert not result.control_passed


def test_outcome_from_another_case_directory_is_rejected(tmp_path: Path) -> None:
    case = next(case for case in build_fma_bench_v0().cases if case.case_id == "ra_n1")
    old_case_output = tmp_path / "old"
    old_exploration = old_case_output / "explorations" / "run-old"
    old_exploration.mkdir(parents=True)
    current_case_output = tmp_path / "current"
    current_case_output.mkdir()
    outcome = CodexAgentOutcome(
        status="no_result",
        stop_reason="synthetic abstention",
        exploration_directory=str(old_exploration),
        rounds=[
            ExplorationRound(
                attempt_index=1,
                driver_status="no_result",
                proposed_candidate_ids=[],
                public_rejections=[],
                private_rejections=[],
                assessed_candidate_ids=[],
                decision_statuses=[],
                feedback_disclosure="none",
            )
        ],
        candidate_outcomes=[],
    )

    result = _score_case(
        benchmark_run_id="run-current",
        suite=build_fma_bench_v0(),
        case=case,
        arm=_arm_metadata("fixture_golden", CodexCLIConfig()),
        repetition=1,
        outcome=outcome,
        expected_case_output=current_case_output,
        privacy_passed=True,
        privacy_detail="synthetic",
    )

    assert result.observed_status == "evidence_invalid"
    assert not result.explicit_first_round_no_result
    assert not result.exact_terminal_match
    assert not result.control_passed


def test_forged_outer_status_cannot_replace_persisted_terminal_outcome(
    tmp_path: Path,
) -> None:
    suite = build_fma_bench_v0()
    case = next(case for case in suite.cases if case.case_id == "ra_n1")
    case_output = tmp_path / "case"
    store = RunStore(case_output / "explorations", run_id="run-persisted")
    persisted = CodexAgentOutcome(
        status="driver_error",
        stop_reason="persisted failure",
        driver_error_code="persisted_failure",
        exploration_directory=str(store.run_directory),
        rounds=[
            ExplorationRound(
                attempt_index=1,
                driver_status="error",
                proposed_candidate_ids=[],
                public_rejections=[],
                private_rejections=[],
                assessed_candidate_ids=[],
                decision_statuses=[],
                feedback_disclosure="none",
            )
        ],
        candidate_outcomes=[],
    )
    outcome_ref = store.put_artifact("codex_agent_outcome", persisted)
    store.emit(
        "codex_agent_stopped",
        {
            "status": persisted.status,
            "stop_reason": persisted.stop_reason,
            "result": outcome_ref.model_dump(mode="json"),
        },
    )
    forged = CodexAgentOutcome(
        status="no_result",
        stop_reason="forged abstention",
        exploration_directory=str(store.run_directory),
        rounds=[
            ExplorationRound(
                attempt_index=1,
                driver_status="no_result",
                proposed_candidate_ids=[],
                public_rejections=[],
                private_rejections=[],
                assessed_candidate_ids=[],
                decision_statuses=[],
                feedback_disclosure="none",
            )
        ],
        candidate_outcomes=[],
    )

    result = _score_case(
        benchmark_run_id="run-current",
        suite=suite,
        case=case,
        arm=_arm_metadata("fixture_golden", CodexCLIConfig()),
        repetition=1,
        outcome=forged,
        expected_case_output=case_output,
        privacy_passed=True,
        privacy_detail="synthetic",
    )

    assert result.observed_status == "evidence_invalid"
    assert not result.exact_terminal_match


def _gold_result(
    *,
    suite_hash: str,
    case: BenchmarkCase,
    arm: ArmMetadata,
    observed: str,
    exact: bool,
    explicit_no_result: bool = False,
    infrastructure: bool = False,
    false_promotion: bool = False,
) -> BenchmarkCaseResult:
    return BenchmarkCaseResult(
        benchmark_run_id="run-gold",
        suite_hash=suite_hash,
        case_id=case.case_id,
        case_hash=str(case.sealed_hash),
        public_hash=case.public_hash(),
        expected_hash=case.expected_hash(),
        family=case.family,
        task_kind=case.task_kind,
        arm_id=arm.arm_id,
        arm_role=arm.role,
        arm_config_hash=arm.content_hash(),
        repetition=1,
        expected_status=case.expected_status,
        observed_status=observed,
        explicit_first_round_no_result=explicit_no_result,
        outer_claimed_validated=observed == "validated",
        evidence_validated=observed == "validated" and not false_promotion,
        evidence_detail="gold",
        holdout=HoldoutReport(
            status="pass" if observed == "validated" and not false_promotion else "not_applicable",
            detail="gold",
        ),
        privacy_passed=True,
        privacy_detail="gold",
        false_promotion=false_promotion,
        exact_terminal_match=exact,
        control_passed=exact,
        infrastructure_failure=infrastructure,
        round_count=1,
        candidate_count=1,
        private_rejection_count=0,
        runtime_invocations=[],
        metrics={"elapsed_ms": 10},
        exploration_directory="",
        candidate_run_directories=[],
        detail="gold",
    )


def test_aggregate_denominators_match_the_hand_calculated_gold_sample() -> None:
    suite = build_fma_bench_v0()
    arm = _arm_metadata("live_single", CodexCLIConfig())
    by_id = {case.case_id: case for case in suite.cases}
    selected_ids = ["ra_b1", "ra_e1", "ra_r1", "kp_b1", "ra_n1", "kp_n1"]
    results = [
        _gold_result(suite_hash=str(suite.suite_hash), case=by_id["ra_b1"], arm=arm, observed="validated", exact=True),
        _gold_result(suite_hash=str(suite.suite_hash), case=by_id["ra_e1"], arm=arm, observed="no_result", exact=False, explicit_no_result=True),
        _gold_result(suite_hash=str(suite.suite_hash), case=by_id["ra_r1"], arm=arm, observed="run_invalid", exact=False),
        _gold_result(suite_hash=str(suite.suite_hash), case=by_id["kp_b1"], arm=arm, observed="driver_error", exact=False, infrastructure=True),
        _gold_result(suite_hash=str(suite.suite_hash), case=by_id["ra_n1"], arm=arm, observed="no_result", exact=True, explicit_no_result=True),
        _gold_result(suite_hash=str(suite.suite_hash), case=by_id["kp_n1"], arm=arm, observed="validated", exact=False, false_promotion=True),
    ]

    aggregate = aggregate_results(
        suite,
        arm,
        list(reversed(results)),
        benchmark_run_id="run-gold",
        selected_case_ids=selected_ids,
        repetitions=1,
    )

    assert aggregate.scheduled == 6
    assert aggregate.infrastructure_errors == 1
    assert aggregate.exact_terminal_correct == 2
    assert aggregate.all_case_accuracy == pytest.approx(2 / 6)
    assert aggregate.eligible_accuracy == pytest.approx(2 / 5)
    assert aggregate.validated_precision == pytest.approx(1 / 2)
    assert aggregate.validated_recall == pytest.approx(1 / 4)
    assert aggregate.answerable_coverage == pytest.approx(1 / 4)
    assert aggregate.no_result_precision == pytest.approx(1 / 2)
    assert aggregate.no_result_recall == pytest.approx(1 / 2)
    assert aggregate.false_promotion_count == 1
    assert aggregate.infrastructure_error_rate == pytest.approx(1 / 6)
    assert not aggregate.harness_integrity_passed


def test_aggregate_rejects_case_identity_tampering() -> None:
    suite = build_fma_bench_v0()
    arm = _arm_metadata("live_single", CodexCLIConfig())
    case = next(case for case in suite.cases if case.case_id == "ra_b1")
    result = _gold_result(
        suite_hash=str(suite.suite_hash),
        case=case,
        arm=arm,
        observed="validated",
        exact=True,
    ).model_copy(update={"case_hash": "0" * 64})

    with pytest.raises(ValueError, match="identity binding"):
        aggregate_results(
            suite,
            arm,
            [result],
            benchmark_run_id="run-gold",
            selected_case_ids=[case.case_id],
            repetitions=1,
        )


def test_aggregate_rejects_forged_derived_flags() -> None:
    suite = build_fma_bench_v0()
    arm = _arm_metadata("live_single", CodexCLIConfig())
    case = next(case for case in suite.cases if case.case_id == "ra_b1")
    forged = _gold_result(
        suite_hash=str(suite.suite_hash),
        case=case,
        arm=arm,
        observed="driver_error",
        exact=False,
        infrastructure=True,
    ).model_copy(
        update={
            "infrastructure_failure": False,
            "exact_terminal_match": True,
            "control_passed": True,
        }
    )

    with pytest.raises(ValueError, match="derived flags"):
        aggregate_results(
            suite,
            arm,
            [forged],
            benchmark_run_id="run-gold",
            selected_case_ids=[case.case_id],
            repetitions=1,
        )
