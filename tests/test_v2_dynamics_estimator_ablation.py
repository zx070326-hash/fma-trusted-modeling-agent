from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from fma.v2.dynamics_estimator_ablation import (
    DynamicsEstimatorExperimentSpecV25,
    default_estimator_exploratory_spec_v25,
    qualify_estimator_policy_v25,
    run_estimator_worldpack_v25,
    verify_estimator_worldpack_run_v25,
)
from fma.v2.dynamics_integral import default_estimator_policy_v25
from fma.v2.dynamics_stability import default_stability_protocol_v25
from fma.v2.dynamics_worldpack import DynamicsWorldPackSpecV24
from fma.v2.epistemic_graph import (
    EpistemicGraphStore,
    register_dynamics_estimator_run_v25,
)


NOW = datetime(2026, 7, 22, tzinfo=timezone.utc)
KNOWLEDGE_HASH = "a" * 64
FAILURE_HASH = "b" * 64
METHOD_HASH = "c" * 64


def _policies():
    point = default_estimator_policy_v25(
        "point_savgol",
        knowledge_bundle_hash=KNOWLEDGE_HASH,
        failure_evidence_hash=FAILURE_HASH,
    )
    integral = default_estimator_policy_v25(
        "window_integral_matching",
        knowledge_bundle_hash=KNOWLEDGE_HASH,
        failure_evidence_hash=FAILURE_HASH,
    )
    return point, integral


def _small_spec() -> DynamicsEstimatorExperimentSpecV25:
    point, integral = _policies()
    base = default_estimator_exploratory_spec_v25(
        knowledge_bundle_hash=KNOWLEDGE_HASH,
        prior_failure_report_hash=FAILURE_HASH,
        method_evidence_hash=METHOD_HASH,
        point_policy_hash=point.policy_hash,
        integral_policy_hash=integral.policy_hash,
        frozen_at=NOW,
    )
    data_payload = base.data_spec.model_dump(exclude={"spec_hash"})
    data_payload["seeds"] = [4001, 4051, 4099, 4153]
    data_spec = DynamicsWorldPackSpecV24.seal(**data_payload)
    payload = base.model_dump(exclude={"spec_hash", "data_spec"})
    return DynamicsEstimatorExperimentSpecV25.seal(
        **payload,
        data_spec=data_spec,
    )


def test_v25_estimator_worldpack_is_single_component_replayable_and_fail_closed(
    tmp_path,
) -> None:
    point, integral = _policies()
    spec = _small_spec()
    stability_protocol = default_stability_protocol_v25(
        experiment_spec_hash=spec.spec_hash,
        frozen_at=NOW,
    )
    outcome = run_estimator_worldpack_v25(
        tmp_path,
        spec=spec,
        point_policy=point,
        integral_policy=integral,
        stability_protocol=stability_protocol,
        evaluated_at=NOW,
        run_id="dynamics_estimator_fixture_v25",
    )
    assert outcome.report.status == "exploratory_only"
    assert outcome.report.reason_codes == ["exploratory_not_eligible"]
    assert outcome.report.single_component_ablation
    assert outcome.report.same_candidate_and_fit_budget
    assert outcome.report.sentinel_false_promotion_count == 0
    assert len(outcome.report.cases) == 16
    assert outcome.qualification is None
    assert outcome.stability_report is not None
    assert outcome.stability_report.estimator_report_hash == outcome.report.report_hash
    assert outcome.stability_report.integral_case_count == 16
    assert verify_estimator_worldpack_run_v25(outcome.store.run_directory)
    graph = EpistemicGraphStore(tmp_path / "graph", "dynamics_estimator_graph_v25")
    nodes = register_dynamics_estimator_run_v25(
        graph, outcome.store.run_directory
    )
    statuses = graph.project_state().snapshot.node_statuses
    expected_policy_status = (
        "refuted"
        if outcome.stability_report.status == "stability_gate_failed"
        else "active"
    )
    assert statuses[nodes["integral_policy"]] == expected_policy_status
    assert statuses[nodes["report"]] == "active"
    assert statuses[nodes["stability"]] == "active"
    assert graph.verify()
    with pytest.raises(ValueError, match="rejected V2.5 estimator"):
        qualify_estimator_policy_v25(integral, outcome.report)

    ref = next(
        ref
        for ref in outcome.manifest.artifact_refs
        if ref.kind == "private_dynamics_worldpack_v24"
    )
    path = outcome.store.run_directory / ref.relative_path
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["cases"][0]["clean_values"][0][0] += 1.0
    path.write_text(json.dumps(envelope), encoding="utf-8")
    assert not verify_estimator_worldpack_run_v25(outcome.store.run_directory)
