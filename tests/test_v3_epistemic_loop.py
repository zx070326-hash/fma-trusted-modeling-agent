from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from fma.v2.epistemic_graph import (
    EpistemicGraphStore,
    register_epistemic_loop_run_v30,
    register_epistemic_loop_run_v301,
)
from fma.v3.epistemic_loop import (
    EpisodeProblemContractV30,
    EpistemicActionProposalV30,
    EpistemicStateV30,
    default_problem_reformulation_exploratory_spec_v30,
    decide_epistemic_permission_v30,
    execute_epistemic_tool_v30,
    generate_private_problem_reformulation_worldpack_v30,
    run_problem_reformulation_worldpack_v30,
    verify_problem_reformulation_run_v30,
)
from fma.v3.epistemic_loop_v301 import (
    default_problem_reformulation_confirmation_spec_v301,
    default_problem_reformulation_exploratory_spec_v301,
    run_problem_reformulation_worldpack_v301,
    verify_problem_reformulation_run_v301,
)


AT = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def v30_failure(tmp_path_factory):
    root = tmp_path_factory.mktemp("v30_failure")
    spec, baseline, candidate = default_problem_reformulation_exploratory_spec_v30(
        frozen_at=AT
    )
    return run_problem_reformulation_worldpack_v30(
        root,
        spec=spec,
        baseline_policy=baseline,
        candidate_policy=candidate,
        run_id="v30-exploratory-failure",
        at=AT,
    )


@pytest.fixture(scope="module")
def v301_exploratory(tmp_path_factory, v30_failure):
    root = tmp_path_factory.mktemp("v301_exploratory")
    spec, baseline, candidate = default_problem_reformulation_exploratory_spec_v301(
        prior_failure_report_hash=v30_failure.report.report_hash,
        frozen_at=AT,
    )
    return run_problem_reformulation_worldpack_v301(
        root,
        spec=spec,
        baseline_policy=baseline,
        candidate_policy=candidate,
        run_id="v301-exploratory",
        at=AT,
    )


def test_v30_records_the_one_step_tradeoff_failure(v30_failure) -> None:
    report = v30_failure.report
    assert report.status == "exploratory_only"
    assert report.material_negative_transfer_count > 0
    assert not report.gate_results["negative_transfer_rate_gate"]
    assert v30_failure.qualification is None
    assert verify_problem_reformulation_run_v30(v30_failure.store.run_directory)


def test_v301_evolves_only_the_action_horizon(v301_exploratory, v30_failure) -> None:
    outcome = v301_exploratory
    assert outcome.spec.evolved_component == "epistemic_action_horizon_1_to_2"
    assert outcome.spec.prior_failure_report_hash == v30_failure.report.report_hash
    assert outcome.report.material_negative_transfer_count == 0
    assert outcome.report.candidate_reformulation_count == outcome.report.underspecified_case_count
    assert outcome.report.spurious_reformulation_count == 0
    assert outcome.baseline.total_action_cost == outcome.candidate.total_action_cost
    assert verify_problem_reformulation_run_v301(outcome.store.run_directory)


def test_v301_trace_is_sequential_and_permissioned(v301_exploratory) -> None:
    missing = next(
        receipt
        for receipt in v301_exploratory.candidate.case_receipts
        if receipt.final_state.contract_history[0].semantics_status == "underspecified"
    )
    assert [step.proposal.action_kind for step in missing.steps] == [
        "clarify_loss_semantics",
        "collect_demand_batch",
    ]
    assert all(step.permission.decision == "allow" for step in missing.steps)
    assert all(step.tool_result.status == "success" for step in missing.steps)
    assert missing.final_state.contract_history[1].parent_contract_hash == (
        missing.final_state.contract_history[0].contract_hash
    )
    assert missing.final_state.contract_history[1].triggering_evidence_hash == (
        missing.steps[0].tool_result.evidence.evidence_hash
    )


def test_v301_known_semantics_are_not_reformulated(v301_exploratory) -> None:
    known = [
        receipt
        for receipt in v301_exploratory.candidate.case_receipts
        if receipt.final_state.contract_history[0].semantics_status == "authoritative"
    ]
    assert known
    assert all(receipt.reformulation_count == 0 for receipt in known)
    assert all(
        [step.proposal.action_kind for step in receipt.steps]
        == ["collect_demand_batch", "collect_demand_batch"]
        for receipt in known
    )


def test_child_contract_cannot_omit_triggering_evidence(v30_failure) -> None:
    parent = v30_failure.private_pack.cases[0].public_case.initial_contract
    with pytest.raises(ValidationError):
        EpisodeProblemContractV30.seal(
            contract_id="invalid_child",
            case_id=parent.case_id,
            mission_constitution_hash=parent.mission_constitution_hash,
            version=2,
            parent_contract_hash=parent.contract_hash,
            revision_reason="tries to revise without evidence",
            loss_profile=parent.loss_profile.model_copy(
                update={
                    "source_kind": "authoritative",
                    "source_ref": "invalid:missing_evidence",
                }
            ),
            semantics_status="authoritative",
            unresolved_fields=[],
            frozen_at=AT,
        )


def test_budget_denial_still_returns_a_structured_tool_result(v30_failure) -> None:
    spec = v30_failure.spec
    policy = v30_failure.candidate_policy
    private_case = v30_failure.private_pack.cases[0]
    public = private_case.public_case
    state = EpistemicStateV30.seal(
        case_id=public.case_id,
        mission_constitution_hash=spec.mission_constitution.constitution_hash,
        arm=policy.arm,
        policy_hash=policy.policy_hash,
        contract_history=[public.initial_contract],
        current_contract_hash=public.initial_contract.contract_hash,
        evidence_receipts=[],
        remaining_action_budget=0,
        terminal_status="running",
    )
    proposal = EpistemicActionProposalV30.seal(
        proposal_id="denied_budget_proposal",
        case_id=public.case_id,
        policy_hash=policy.policy_hash,
        action_kind="collect_demand_batch",
        proposed_cost=1,
        decision_spread=0,
        rationale_code="low_decision_spread_collect_more",
    )
    permission = decide_epistemic_permission_v30(
        proposal, state, spec.mission_constitution, decided_at=AT
    )
    result = execute_epistemic_tool_v30(
        private_case, proposal, permission, observed_at=AT
    )
    assert permission.decision == "deny"
    assert permission.policy_rule == "deny_budget_exceeded"
    assert result.status == "denied"
    assert result.error_code == "permission_denied"
    assert result.next_valid_actions == ["stop_needs_evidence"]


def test_v301_confirmation_uses_new_seeds_and_can_qualify(
    tmp_path: Path, v30_failure
) -> None:
    spec, baseline, candidate = default_problem_reformulation_confirmation_spec_v301(
        prior_failure_report_hash=v30_failure.report.report_hash,
        frozen_at=AT,
    )
    outcome = run_problem_reformulation_worldpack_v301(
        tmp_path,
        spec=spec,
        baseline_policy=baseline,
        candidate_policy=candidate,
        run_id="v301-confirmation",
        at=AT,
    )
    assert set(spec.seeds).isdisjoint(v30_failure.spec.seeds)
    assert outcome.report.status == "promoted_for_synthetic_epistemic_loop_v301"
    assert outcome.report.material_negative_transfer_count == 0
    assert outcome.report.negative_transfer_rate_upper < spec.max_negative_transfer_rate
    assert outcome.qualification is not None
    assert not outcome.qualification.real_world_validity_established
    assert not outcome.qualification.independent_problem_discovery_established
    assert verify_problem_reformulation_run_v301(outcome.store.run_directory)


def test_v301_tampering_breaks_replay(tmp_path: Path, v301_exploratory) -> None:
    source = v301_exploratory.store.run_directory
    target = tmp_path / source.name
    import shutil

    shutil.copytree(source, target)
    events = [
        json.loads(line)
        for line in (target / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    report_ref = next(
        event["payload"]
        for event in events
        if event["event_type"] == "artifact_committed"
        and event["payload"]["kind"] == "epistemic_loop_report_v301"
    )
    artifact = target / report_ref["relative_path"]
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["payload"]["macro_regret_improvement"] += 0.01
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    assert not verify_problem_reformulation_run_v301(target)


def test_v3_runs_register_failure_lineage_in_epistemic_graph(
    tmp_path: Path, v30_failure, v301_exploratory
) -> None:
    graph = EpistemicGraphStore(tmp_path, graph_id="v3-graph-test")
    first = register_epistemic_loop_run_v30(
        graph, v30_failure.store.run_directory
    )
    second = register_epistemic_loop_run_v301(
        graph,
        v301_exploratory.store.run_directory,
        prior_failure_node_hash=first["report"],
    )
    state = graph.project_state()
    assert graph.verify()
    assert len(state.nodes) == 14
    assert len(state.edges) == 12
    assert state.snapshot.node_statuses[first["candidate_policy"]] == "refuted"
    assert state.snapshot.node_statuses[second["candidate_policy"]] == "active"
