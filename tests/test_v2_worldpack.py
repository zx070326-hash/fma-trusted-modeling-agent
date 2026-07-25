from __future__ import annotations

import json
from datetime import datetime, timezone

from fma.v2.worldpack import (
    HiddenWorldCaseV22,
    default_worldpack_spec,
    generate_private_worldpack,
    run_worldpack_ablation,
    select_worldpack_candidate,
    verify_worldpack_run,
)


NOW = datetime(2026, 7, 22, tzinfo=timezone.utc)


def test_public_projection_hides_mechanism_and_outer_holdout() -> None:
    spec = default_worldpack_spec(frozen_at=NOW)
    private_pack = generate_private_worldpack(spec)
    case = private_pack.cases[0]
    public = case.public_projection(spec)
    assert "mechanism" not in public
    assert "seed" not in public
    assert len(public["observed_values"]) == (
        spec.training_points + spec.inner_validation_points
    )
    assert case.values[-spec.outer_holdout_points :] != public["observed_values"][-spec.outer_holdout_points :]


def test_outer_holdout_changes_cannot_change_candidate_selection() -> None:
    spec = default_worldpack_spec(frozen_at=NOW)
    case = generate_private_worldpack(spec).cases[0]
    original_public = case.public_projection(spec)
    direct_before = select_worldpack_candidate(original_public, "direct_generation")
    memory_before = select_worldpack_candidate(
        original_public, "retrieval_evolution_memory"
    )
    observed_count = spec.training_points + spec.inner_validation_points
    changed_values = list(case.values)
    changed_values[observed_count:] = [value + 1000.0 for value in changed_values[observed_count:]]
    changed = HiddenWorldCaseV22.seal(
        case_id=case.case_id,
        pack_spec_hash=case.pack_spec_hash,
        mechanism=case.mechanism,
        seed=case.seed,
        values=changed_values,
    )
    changed_public = changed.public_projection(spec)
    assert changed_public == original_public
    assert (
        select_worldpack_candidate(changed_public, "direct_generation").receipt_hash
        == direct_before.receipt_hash
    )
    assert (
        select_worldpack_candidate(
            changed_public, "retrieval_evolution_memory"
        ).receipt_hash
        == memory_before.receipt_hash
    )


def test_worldpack_ablation_is_same_budget_replayable_and_fail_closed(tmp_path) -> None:
    outcome = run_worldpack_ablation(
        tmp_path,
        spec=default_worldpack_spec(frozen_at=NOW),
        evaluated_at=NOW,
        run_id="worldpack_fixture",
    )
    report = outcome.report
    assert report.same_budget
    assert all(
        result.direct_evaluation_count == result.memory_evaluation_count == 96
        for result in report.cases
    )
    assert report.confidence_lower > 0
    assert report.loss_count == 0
    assert report.negative_transfer_count == 0
    # The pre-frozen 50% case-win gate still rejects 4 wins / 12 cases.  A
    # favorable mean effect cannot silently rewrite that gate after the result.
    assert report.status == "candidate_rejected"
    assert report.reason_codes == ["minimum_win_fraction_not_met"]
    assert verify_worldpack_run(outcome.store.run_directory)


def test_worldpack_artifact_tampering_breaks_replay(tmp_path) -> None:
    outcome = run_worldpack_ablation(
        tmp_path,
        spec=default_worldpack_spec(frozen_at=NOW),
        evaluated_at=NOW,
        run_id="worldpack_tamper",
    )
    ref = next(
        ref
        for ref in outcome.manifest.artifact_refs
        if ref.kind == "worldpack_ablation_report_v22"
    )
    path = outcome.store.run_directory / ref.relative_path
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["status"] = "promoted_for_worldpack_scope"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    assert not verify_worldpack_run(outcome.store.run_directory)
