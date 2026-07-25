from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from fma.v2.dynamics_active_design import (
    ActiveDesignWorldPackSpecV26,
    default_active_design_exploratory_spec_v26,
    default_active_design_policies_v26,
    failure_evolved_active_design_exploratory_spec_v26,
    qualify_active_design_policy_v26,
    run_active_design_worldpack_v26,
    verify_active_design_worldpack_run_v26,
)
from fma.v2.dynamics_active_design_effect import (
    default_exploratory_effect_protocol_v261,
    run_active_design_effect_worldpack_v261,
    verify_active_design_effect_run_v261,
)
from fma.v2.epistemic_graph import (
    EpistemicGraphStore,
    register_active_design_effect_run_v261,
    register_active_design_run_v26,
)


NOW = datetime(2026, 7, 22, tzinfo=timezone.utc)
KNOWLEDGE_HASH = "a" * 64
FAILURE_HASH = "b" * 64
METHOD_HASH = "c" * 64


def _policies():
    return default_active_design_policies_v26(
        knowledge_bundle_hash=KNOWLEDGE_HASH,
        prior_failure_report_hash=FAILURE_HASH,
    )


def _small_spec() -> ActiveDesignWorldPackSpecV26:
    baseline, active = _policies()
    base = default_active_design_exploratory_spec_v26(
        knowledge_bundle_hash=KNOWLEDGE_HASH,
        prior_failure_report_hash=FAILURE_HASH,
        method_evidence_hash=METHOD_HASH,
        baseline_policy_hash=baseline.policy_hash,
        active_policy_hash=active.policy_hash,
        frozen_at=NOW,
    )
    payload = base.model_dump(exclude={"spec_hash"})
    payload.update(
        seeds=[6101, 6151],
        candidate_action_count=12,
        action_budget=2,
        ensemble_members=8,
        bootstrap_replicates=200,
    )
    return ActiveDesignWorldPackSpecV26.seal(**payload)


def test_v26_active_design_is_single_component_replayable_and_fail_closed(
    tmp_path,
) -> None:
    baseline, active = _policies()
    spec = _small_spec()
    outcome = run_active_design_worldpack_v26(
        tmp_path,
        spec=spec,
        baseline_policy=baseline,
        active_policy=active,
        evaluated_at=NOW,
        run_id="active_design_fixture_v26",
    )
    assert outcome.report.status == "exploratory_only"
    assert outcome.report.reason_codes == ["exploratory_not_eligible"]
    assert outcome.report.same_action_and_fit_budget
    assert outcome.report.invalid_action_count == 0
    assert len(outcome.report.cases) == 8
    assert outcome.baseline.total_action_budget == outcome.active.total_action_budget == 16
    assert outcome.qualification is None
    assert verify_active_design_worldpack_run_v26(outcome.store.run_directory)
    graph = EpistemicGraphStore(tmp_path / "graph", "active_design_graph_v26")
    hashes = register_active_design_run_v26(graph, outcome.store.run_directory)
    assert hashes["report"]
    assert graph.verify()
    with pytest.raises(ValueError, match="rejected V2.6 active-design"):
        qualify_active_design_policy_v26(active, outcome.report)

    ref = next(
        ref
        for ref in outcome.manifest.artifact_refs
        if ref.kind == "private_active_design_worldpack_v26"
    )
    path = outcome.store.run_directory / ref.relative_path
    envelope = json.loads(path.read_text(encoding="utf-8"))
    case = envelope["payload"]["cases"][0]
    action_id = next(iter(case["action_observations"]))
    case["action_observations"][action_id]["values"][0][0] += 1.0
    path.write_text(json.dumps(envelope), encoding="utf-8")
    assert not verify_active_design_worldpack_run_v26(outcome.store.run_directory)


def test_v26_spec_rejects_unfrozen_seed_family() -> None:
    baseline, active = _policies()
    with pytest.raises(ValueError, match="outside the frozen family"):
        ActiveDesignWorldPackSpecV26.seal(
            experiment_id="bad_seed_v26",
            phase="exploratory",
            mechanisms=[
                "exponential_decay",
                "logistic_growth",
                "damped_oscillator",
                "lotka_volterra",
            ],
            seeds=[1, 2],
            baseline_policy_hash=baseline.policy_hash,
            active_policy_hash=active.policy_hash,
            knowledge_bundle_hash=KNOWLEDGE_HASH,
            prior_failure_report_hash=FAILURE_HASH,
            method_evidence_hash=METHOD_HASH,
            frozen_at=NOW,
        )


def test_v261_bounded_effect_protocol_is_replayable(tmp_path) -> None:
    baseline, active = _policies()
    base = failure_evolved_active_design_exploratory_spec_v26(
        knowledge_bundle_hash=KNOWLEDGE_HASH,
        prior_failure_report_hash=FAILURE_HASH,
        method_evidence_hash=METHOD_HASH,
        baseline_policy_hash=baseline.policy_hash,
        active_policy_hash=active.policy_hash,
        frozen_at=NOW,
    )
    payload = base.model_dump(exclude={"spec_hash"})
    payload.update(
        seeds=[6503, 6551],
        candidate_action_count=12,
        action_budget=2,
        ensemble_members=8,
        bootstrap_replicates=200,
    )
    spec = ActiveDesignWorldPackSpecV26.seal(**payload)
    protocol = default_exploratory_effect_protocol_v261(
        base_spec_hash=spec.spec_hash,
        prior_metric_failure_report_hash="d" * 64,
        frozen_at=NOW,
    )
    outcome = run_active_design_effect_worldpack_v261(
        tmp_path,
        spec=spec,
        protocol=protocol,
        baseline_policy=baseline,
        active_policy=active,
        evaluated_at=NOW,
        run_id="active_design_effect_fixture_v261",
    )
    assert outcome.report.status == "exploratory_only"
    assert outcome.report.same_action_and_fit_budget
    assert outcome.report.invalid_action_count == 0
    assert outcome.qualification is None
    assert verify_active_design_effect_run_v261(outcome.store.run_directory)
    graph = EpistemicGraphStore(tmp_path / "effect_graph", "active_effect_graph_v261")
    hashes = register_active_design_effect_run_v261(
        graph, outcome.store.run_directory
    )
    assert hashes["report"]
    assert graph.verify()
