from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from fma.v2.dynamics_ir import default_dynamics_arm_policy
from fma.v2.dynamics_worldpack import (
    DynamicsWorldPackSpecV24,
    default_exploratory_dynamics_spec_v24,
    generate_private_dynamics_worldpack,
    qualify_dynamics_policy_v24,
    run_dynamics_worldpack,
    verify_dynamics_worldpack_run,
)


NOW = datetime(2026, 7, 22, tzinfo=timezone.utc)
KNOWLEDGE_HASH = "a" * 64


def _small_spec() -> DynamicsWorldPackSpecV24:
    base = default_exploratory_dynamics_spec_v24(
        knowledge_bundle_hash=KNOWLEDGE_HASH,
        frozen_at=NOW,
    )
    payload = base.model_dump(exclude={"spec_hash"})
    payload["seeds"] = [17, 53, 97, 149]
    return DynamicsWorldPackSpecV24.seal(**payload)


def test_private_dynamics_projection_hides_mechanism_truth_and_outer_values() -> None:
    spec = _small_spec()
    pack = generate_private_dynamics_worldpack(spec, generated_at=NOW)
    case = pack.cases[0]
    public = case.public_projection(spec)
    public_payload = public.model_dump(mode="json")
    assert len(public.values) == spec.training_points + spec.inner_validation_points
    assert "mechanism" not in public_payload
    assert "seed" not in public_payload
    assert "truth_coefficients" not in public_payload
    assert len(case.clean_values) == len(public.values) + spec.outer_holdout_points


def test_dynamics_worldpack_is_fail_closed_replayable_and_same_budget(tmp_path) -> None:
    spec = _small_spec()
    direct = default_dynamics_arm_policy("direct_generation")
    memory = default_dynamics_arm_policy(
        "retrieval_evolution_memory",
        knowledge_bundle_hash=KNOWLEDGE_HASH,
    )
    outcome = run_dynamics_worldpack(
        tmp_path,
        spec=spec,
        direct_policy=direct,
        memory_policy=memory,
        evaluated_at=NOW,
        run_id="dynamics_worldpack_fixture",
    )
    assert outcome.report.status == "exploratory_only"
    assert outcome.report.reason_codes == ["exploratory_not_eligible"]
    assert outcome.report.same_candidate_and_fit_budget
    assert outcome.report.sentinel_false_promotion_count == 0
    assert len(outcome.report.cases) == 16
    assert outcome.qualification is None
    assert verify_dynamics_worldpack_run(outcome.store.run_directory)
    with pytest.raises(ValueError, match="rejected Dynamics policy"):
        qualify_dynamics_policy_v24(memory, outcome.report)


def test_tampered_private_dynamics_pack_breaks_independent_replay(tmp_path) -> None:
    spec = _small_spec()
    outcome = run_dynamics_worldpack(
        tmp_path,
        spec=spec,
        direct_policy=default_dynamics_arm_policy("direct_generation"),
        memory_policy=default_dynamics_arm_policy(
            "retrieval_evolution_memory",
            knowledge_bundle_hash=KNOWLEDGE_HASH,
        ),
        evaluated_at=NOW,
        run_id="dynamics_worldpack_tamper",
    )
    ref = next(
        ref
        for ref in outcome.manifest.artifact_refs
        if ref.kind == "private_dynamics_worldpack_v24"
    )
    path = outcome.store.run_directory / ref.relative_path
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["cases"][0]["clean_values"][0][0] += 1.0
    path.write_text(json.dumps(envelope), encoding="utf-8")
    assert not verify_dynamics_worldpack_run(outcome.store.run_directory)
