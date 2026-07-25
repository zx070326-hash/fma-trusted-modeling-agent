from __future__ import annotations

from datetime import datetime, timezone

import pytest

from fma.hashing import sha256_value
from fma.v3.controlled_dynamics_loop_v33 import (
    default_controlled_dynamics_exploratory_spec_v33,
    default_controlled_dynamics_policies_v33,
    run_controlled_dynamics_worldpack_v33,
    verify_controlled_dynamics_run_v33,
)


AT33 = datetime(2026, 7, 22, 9, 30, tzinfo=timezone.utc)
PRIOR_V32 = sha256_value("v32-failure")
PRIOR_EPISTEMIC = sha256_value("v301-qualified")
PRIOR_ACTIVE = sha256_value("v261-qualified")
METHOD = sha256_value("iteration11-resource-ledger-evidence")


@pytest.fixture(scope="module")
def v33_outcome(tmp_path_factory):
    root = tmp_path_factory.mktemp("controlled_v33")
    baseline, candidate = default_controlled_dynamics_policies_v33(
        prior_v32_failure_report_hash=PRIOR_V32,
        prior_epistemic_qualification_hash=PRIOR_EPISTEMIC,
        prior_active_design_qualification_hash=PRIOR_ACTIVE,
        method_evidence_hash=METHOD,
    )
    spec = default_controlled_dynamics_exploratory_spec_v33(
        baseline_policy_hash=baseline.policy_hash,
        candidate_policy_hash=candidate.policy_hash,
        method_evidence_hash=METHOD,
        prior_epistemic_qualification_hash=PRIOR_EPISTEMIC,
        prior_active_design_qualification_hash=PRIOR_ACTIVE,
        prior_v32_failure_report_hash=PRIOR_V32,
        frozen_at=AT33,
    )
    return run_controlled_dynamics_worldpack_v33(
        root,
        spec=spec,
        baseline_policy=baseline,
        candidate_policy=candidate,
        evaluated_at=AT33,
        run_id="v33-resource-ledger",
    )


def test_v33_purifies_target_and_resource_comparison(v33_outcome) -> None:
    evolution = v33_outcome.evolution_report
    assert evolution.resource_entitlement_parity
    assert evolution.resource_use_parity
    assert evolution.target_contract_parity
    assert evolution.budget_model_changed
    assert evolution.baseline_target_clarification_aligned
    assert not evolution.acquisition_changed
    assert not evolution.model_router_changed
    assert not evolution.overall_qualification_permitted
    assert not evolution.confirmation_permitted


def test_v33_missing_targets_get_query_plus_three_experiments(v33_outcome) -> None:
    private = {
        item.public_case.case_id: item for item in v33_outcome.private_pack.cases
    }
    for bundle in (v33_outcome.baseline_bundle, v33_outcome.candidate_bundle):
        for receipt in bundle.case_receipts:
            source = private[receipt.case_id]
            if source.performance_eligible and source.target_was_underspecified:
                assert receipt.resource_ledger.clarification_used == 1
                assert receipt.resource_ledger.controlled_experiments_used == 3
                assert [step.action_kind for step in receipt.steps] == [
                    "clarify_target",
                    "controlled_experiment",
                    "controlled_experiment",
                    "controlled_experiment",
                ]


def test_v33_candidate_still_uses_frozen_v32_ranking(v33_outcome) -> None:
    for case in v33_outcome.candidate_bundle.case_receipts:
        for step in case.steps:
            if step.action_kind != "controlled_experiment":
                continue
            selected = next(
                item for item in step.acquisition_receipts
                if item.action_hash == step.selected_action_hash
            )
            assert selected.ranking_score == max(
                item.ranking_score
                for item in step.acquisition_receipts if item.admissible
            )


def test_v33_run_replays_independently(v33_outcome) -> None:
    assert verify_controlled_dynamics_run_v33(v33_outcome.store.run_directory)
