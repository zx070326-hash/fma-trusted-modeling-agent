from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timezone

import pytest

from fma.v3.model_challenge_v37 import (
    FAMILIES_V37,
    _decision_v37,
    default_model_challenge_exploratory_spec_v37,
    default_model_challenge_method_evidence_v37,
    default_model_portfolio_policies_v37,
    run_model_challenge_worldpack_v37,
    verify_model_challenge_run_v37,
)


@pytest.fixture(scope="module")
def v37_outcome(tmp_path_factory):
    at = datetime(2026, 7, 22, 8, 30, tzinfo=timezone.utc)
    method = default_model_challenge_method_evidence_v37()
    baseline_policy, candidate_policy = default_model_portfolio_policies_v37(
        method.evidence_hash
    )
    spec = default_model_challenge_exploratory_spec_v37(
        method_evidence_hash=method.evidence_hash,
        baseline_policy_hash=baseline_policy.policy_hash,
        candidate_policy_hash=candidate_policy.policy_hash,
        frozen_at=at,
    )
    return run_model_challenge_worldpack_v37(
        tmp_path_factory.mktemp("v37"),
        method_evidence=method,
        spec=spec,
        baseline_policy=baseline_policy,
        candidate_policy=candidate_policy,
        evaluated_at=at,
        run_id="v37_test_run",
    )


def test_v37_protocol_binds_untrusted_method_evidence(v37_outcome) -> None:
    outcome = v37_outcome
    assert outcome.method_evidence.external_content_treated_as_untrusted_data
    assert all(not source.guarantee_transferred for source in outcome.method_evidence.sources)
    assert outcome.spec.method_evidence_hash == outcome.method_evidence.evidence_hash
    assert outcome.spec.baseline_policy_hash == outcome.baseline_policy.policy_hash
    assert outcome.spec.candidate_policy_hash == outcome.candidate_policy.policy_hash


def test_v37_shared_context_contains_no_private_outcome(v37_outcome) -> None:
    baseline_by_id = {
        item.case_id: item for item in v37_outcome.baseline_bundle.case_receipts
    }
    for candidate in v37_outcome.candidate_bundle.case_receipts:
        baseline = baseline_by_id[candidate.case_id]
        assert candidate.observation_hashes == baseline.observation_hashes
        assert candidate.applicability_state.state_hash == baseline.applicability_state.state_hash
        assert [item.challenge_hash for item in candidate.challenges] == [
            item.challenge_hash for item in baseline.challenges
        ]
        state = candidate.applicability_state
        assert not state.private_mechanism_seen
        assert not state.private_probe_seen
        assert not state.private_target_loss_seen
        assert [item.family for item in candidate.challenges] == list(FAMILIES_V37)
        recomputed = _decision_v37(
            v37_outcome.candidate_policy, state, candidate.challenges
        )
        assert recomputed.decision_hash == candidate.decision.decision_hash


def test_v37_quality_failures_abstain_symmetrically(v37_outcome) -> None:
    baseline_by_id = {
        item.case_id: item for item in v37_outcome.baseline_bundle.case_receipts
    }
    sentinels = 0
    for candidate in v37_outcome.candidate_bundle.case_receipts:
        if candidate.applicability_state.quality_flags:
            sentinels += 1
            baseline = baseline_by_id[candidate.case_id]
            assert baseline.decision.reason == "deny_data_quality"
            assert candidate.decision.reason == "deny_data_quality"
            assert baseline.selected_model is None
            assert candidate.selected_model is None
    assert sentinels == v37_outcome.spec.expected_quality_abstention_count


def test_v37_private_report_recomputes_macro_and_never_authorizes(v37_outcome) -> None:
    report = v37_outcome.evolution_report
    expected_macro = sum(report.mechanism_mean_improvements.values()) / len(
        report.mechanism_mean_improvements
    )
    assert math.isclose(report.macro_target_loss_improvement, expected_macro, abs_tol=1e-12)
    assert report.ready_for_non_nested_extension == all(report.gates.values())
    assert not report.router_experiment_permitted
    assert not report.overall_qualification_permitted
    assert not report.confirmation_permitted
    assert not report.real_world_authorization_permitted


def test_v37_run_replays_and_tampering_fails_closed(v37_outcome, tmp_path) -> None:
    assert verify_model_challenge_run_v37(v37_outcome.store.run_directory)
    copied = tmp_path / "tampered"
    shutil.copytree(v37_outcome.store.run_directory, copied)
    events = [json.loads(line) for line in (copied / "events.jsonl").read_text().splitlines()]
    target = next(
        event["payload"]
        for event in events
        if event["event_type"] == "artifact_committed"
        and event["payload"]["kind"] == "model_challenge_candidate_bundle_v37"
    )
    path = copied / target["relative_path"]
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["bundle_id"] = "tampered_bundle"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    assert not verify_model_challenge_run_v37(copied)
