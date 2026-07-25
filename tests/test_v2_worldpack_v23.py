from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from fma.v2.worldpack import safe_hybrid_worldpack_policy
from fma.v2.worldpack_v23 import (
    INITIAL_CONFIRMATION_SEEDS,
    SAFE_POLICY_CONFIRMATION_SEEDS,
    WorldPackConfirmationSpecV23,
    default_confirmation_spec_v23,
    qualify_worldpack_policy_v23,
    run_worldpack_confirmation_v23,
    safe_policy_confirmation_spec_v23,
    verify_worldpack_confirmation_v23,
)


NOW = datetime(2026, 7, 22, 9, 0, tzinfo=timezone.utc)
PRIOR = "486a35679ec49e1a98d8456507dddc941ea2887006e46db0e702cf94e1decb08"


def test_confirmation_seeds_cannot_reuse_exploratory_worldpack() -> None:
    base = default_confirmation_spec_v23(
        prior_exploratory_report_hash=PRIOR, frozen_at=NOW
    )
    payload = base.model_dump(exclude={"spec_hash"})
    payload["seeds"] = [11, *base.seeds[1:]]
    with pytest.raises(ValueError, match="overlap"):
        WorldPackConfirmationSpecV23.seal(**payload)


def test_v23_confirmation_is_precommitted_replayable_and_fail_closed(tmp_path) -> None:
    outcome = run_worldpack_confirmation_v23(
        tmp_path,
        spec=default_confirmation_spec_v23(
            prior_exploratory_report_hash=PRIOR, frozen_at=NOW
        ),
        evaluated_at=NOW,
        run_id="worldpack_confirmation_fixture",
    )
    report = outcome.report
    assert len(report.cases) == 80
    assert report.same_budget
    assert report.macro_confidence_lower > 0
    assert all(result.noninferior for result in report.mechanism_results)
    assert report.negative_transfer_count == 1
    assert report.negative_transfer_rate_upper > 0.05
    assert report.status == "candidate_rejected_v23"
    assert report.reason_codes == ["negative_transfer_rate_bound_failed"]
    with pytest.raises(ValueError, match="rejected"):
        qualify_worldpack_policy_v23(outcome.memory_policy, report)
    assert verify_worldpack_confirmation_v23(outcome.store.run_directory)

    events = [
        json.loads(line)
        for line in outcome.store.event_path.read_text(encoding="utf-8").splitlines()
    ]
    frozen_sequence = next(
        event["sequence"]
        for event in events
        if event["event_type"] == "worldpack_confirmation_protocol_frozen"
    )
    private_sequence = next(
        event["sequence"]
        for event in events
        if event["event_type"] == "artifact_committed"
        and event["payload"]["kind"] == "private_worldpack_v22"
    )
    assert frozen_sequence < private_sequence


def test_v23_report_tampering_breaks_replay(tmp_path) -> None:
    outcome = run_worldpack_confirmation_v23(
        tmp_path,
        spec=default_confirmation_spec_v23(
            prior_exploratory_report_hash=PRIOR, frozen_at=NOW
        ),
        evaluated_at=NOW,
        run_id="worldpack_confirmation_tamper",
    )
    ref = next(
        ref
        for ref in outcome.manifest.artifact_refs
        if ref.kind == "worldpack_confirmation_report_v23"
    )
    path = outcome.store.run_directory / ref.relative_path
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["negative_transfer_count"] = 0
    path.write_text(json.dumps(envelope), encoding="utf-8")
    assert not verify_worldpack_confirmation_v23(outcome.store.run_directory)


def test_failure_evolved_safe_policy_passes_only_on_disjoint_confirmation(tmp_path) -> None:
    assert not (set(INITIAL_CONFIRMATION_SEEDS) & set(SAFE_POLICY_CONFIRMATION_SEEDS))
    prior_confirmation = "79de2aa5457e2cebbae286aaea535e5de9e3d27f0b1b8405840f8fbd058db73b"
    outcome = run_worldpack_confirmation_v23(
        tmp_path,
        spec=safe_policy_confirmation_spec_v23(
            prior_confirmation_report_hash=prior_confirmation,
            frozen_at=NOW,
        ),
        memory_policy=safe_hybrid_worldpack_policy(),
        evaluated_at=NOW,
        run_id="safe_policy_confirmation_fixture",
    )
    report = outcome.report
    assert report.status == "promoted_for_worldpack_scope_v23"
    assert report.reason_codes == []
    assert report.macro_confidence_lower > 0
    assert all(result.noninferior for result in report.mechanism_results)
    assert report.negative_transfer_count == 0
    assert report.negative_transfer_rate_upper < 0.05
    qualification = qualify_worldpack_policy_v23(outcome.memory_policy, report)
    assert qualification.status == "qualified"
    assert qualification.qualification_scope == "synthetic_forecast_worldpack_v23"
    assert verify_worldpack_confirmation_v23(outcome.store.run_directory)
