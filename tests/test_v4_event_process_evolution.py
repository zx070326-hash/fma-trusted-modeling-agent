from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from fma.v4.event_process_evolution import (
    EventProcessEvolutionPolicyV41,
    run_event_process_evolution_campaign_v41,
)
from fma.v4.model_evolution_graph import verify_model_evolution_campaign_v41
from fma.v4.unseen_event_process import (
    EarthquakeEventV40,
    USGSCatalogQueryV40,
    USGSCatalogSnapshotV40,
)


NOW = datetime(2026, 7, 23, tzinfo=timezone.utc)


def _development_snapshot() -> USGSCatalogSnapshotV40:
    query = USGSCatalogQueryV40.seal(
        phase="development",
        start=datetime(2023, 1, 1, tzinfo=timezone.utc),
        end_exclusive=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    events = []
    for index in range(48):
        origin = query.start + timedelta(days=(index + 1) * 365.0 / 49.0)
        events.append(
            EarthquakeEventV40(
                event_id=f"ci_evolution_{index:03d}",
                origin_time=origin,
                magnitude=2.5,
                latitude=34.0,
                longitude=-118.0,
                depth_km=8.0,
            )
        )
    return USGSCatalogSnapshotV40.seal(
        query=query,
        response_sha256=hashlib.sha256(
            b"synthetic-development-only-catalog"
        ).hexdigest(),
        events=events,
        retrieved_at=NOW,
    )


def test_event_process_failure_switches_model_families_without_private_data(
    tmp_path,
) -> None:
    development = _development_snapshot()
    policy = EventProcessEvolutionPolicyV41.seal(
        split_fraction=0.7,
        minimum_events_per_slice=5,
        minimum_validation_log_score_lift_nat_per_event=0.0,
        minimum_time_rescaling_ks_pvalue=0.01,
        maximum_compensator_count_relative_error=0.9,
        maximum_hawkes_branching_ratio=0.95,
        # The fitted Hawkes decay is bounded below by 1/60 day^-1, so this
        # frozen synthetic policy deterministically exercises family switching.
        maximum_interior_hawkes_decay_rate_per_day=0.01,
    )
    outcome = run_event_process_evolution_campaign_v41(
        tmp_path,
        development,
        campaign_id="event_process_evolution_fixture_v41",
        policy=policy,
    )

    assert outcome.report.generation_count == 2
    assert {candidate.family for candidate in outcome.candidates} == {
        "exponential_hawkes",
        "homogeneous_poisson",
        "weibull_renewal",
    }
    initial = next(
        candidate
        for candidate in outcome.candidates
        if candidate.family == "exponential_hawkes"
    )
    initial_evaluation = next(
        evaluation
        for evaluation in outcome.evaluations
        if evaluation.candidate_hash == initial.candidate_hash
    )
    assert not initial_evaluation.gates["parameter_interior"]
    assert "parameter_boundary" in initial_evaluation.diagnostic_codes
    assert len(outcome.operators) == 2
    assert all(
        evaluation.private_data_accessed is False
        for evaluation in outcome.evaluations
    )
    assert outcome.report.private_confirmation_consumed is False
    assert outcome.report.qualification_granted is False
    assert verify_model_evolution_campaign_v41(outcome, outcome.spec)
