from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from fma.v4 import (
    EventProcessCandidateV40,
    FixtureFrontierTransportV40,
    FrontierProposalV40,
    USGSCatalogQueryV40,
    UnseenEventProcessSpecV40,
    parse_usgs_catalog_response_v40,
    run_unseen_event_process_experiment_v40,
    summarize_development_snapshot_v40,
    verify_unseen_event_process_experiment_v40,
)


NOW = datetime(2026, 7, 23, tzinfo=timezone.utc)


def _query(phase: str, year: int) -> USGSCatalogQueryV40:
    return USGSCatalogQueryV40.seal(
        phase=phase,
        start=datetime(year, 1, 1, tzinfo=timezone.utc),
        end_exclusive=datetime(year + 1, 1, 1, tzinfo=timezone.utc),
    )


def _geojson(query: USGSCatalogQueryV40, count: int) -> bytes:
    features = []
    duration = query.end_exclusive - query.start
    for index in range(count):
        offset = duration * ((index + 1) / (count + 1))
        origin = query.start + offset + timedelta(minutes=index % 7)
        features.append(
            {
                "type": "Feature",
                "id": f"ci{query.phase}{index:04d}",
                "properties": {
                    "time": int(origin.timestamp() * 1000),
                    "mag": 2.5 + 0.01 * (index % 10),
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [-118.0, 34.0, 8.0],
                },
            }
        )
    return json.dumps(
        {"type": "FeatureCollection", "metadata": {"count": count}, "features": features}
    ).encode("utf-8")


def _spec() -> UnseenEventProcessSpecV40:
    return UnseenEventProcessSpecV40.seal(
        experiment_id="unseen_event_process_test",
        development_query=_query("development", 2023),
        confirmation_query=_query("confirmation", 2024),
        minimum_events_per_phase=10,
        maximum_events_per_phase=300,
        created_at=NOW,
    )


def test_usgs_parser_filters_the_exclusive_end_and_seals_public_summary() -> None:
    query = _query("development", 2023)
    document = json.loads(_geojson(query, 20))
    document["features"].append(
        {
            "type": "Feature",
            "id": "ci_end_boundary",
            "properties": {
                "time": int(query.end_exclusive.timestamp() * 1000),
                "mag": 3.0,
            },
            "geometry": {
                "type": "Point",
                "coordinates": [-118.0, 34.0, 7.0],
            },
        }
    )
    body = json.dumps(document).encode("utf-8")
    raw, snapshot = parse_usgs_catalog_response_v40(
        query,
        body,
        retrieved_at=NOW,
    )
    summary = summarize_development_snapshot_v40(snapshot)

    assert len(snapshot.events) == 20
    assert all(event.event_id != "ci_end_boundary" for event in snapshot.events)
    assert raw.response_sha256 == snapshot.response_sha256
    assert summary.event_count == 20
    snapshot.assert_sealed()
    summary.assert_sealed()


def test_invalid_hawkes_initialization_is_rejected_before_execution() -> None:
    with pytest.raises(ValueError):
        EventProcessCandidateV40.seal(
            family="exponential_hawkes",
            hawkes_branching_initial=1.1,
            hawkes_decay_days_initial=7.0,
            rationale="invalid branching ratio negative control",
            expected_failure_modes=["nonstationary process"],
        )


def test_full_unseen_task_chain_is_private_rejected_replayable_and_resumable(
    tmp_path,
) -> None:
    spec = _spec()
    bodies = {
        "development": _geojson(spec.development_query, 20),
        "confirmation": _geojson(spec.confirmation_query, 24),
    }

    def fetch(url: str) -> bytes:
        phase = "development" if "starttime=2023" in url else "confirmation"
        return bodies[phase]

    def propose(request):
        serialized = request.model_dump_json()
        assert "2024-01-01" not in serialized
        assert "confirmation_snapshot" not in serialized
        assert "minimum_log_score" not in serialized
        draft = {
            "family": "homogeneous_poisson",
            "hawkes_branching_initial": 0.5,
            "hawkes_decay_days_initial": 7.0,
            "rationale": "Use the minimum-complexity public baseline for this control.",
            "expected_failure_modes": ["future event rate may shift"],
        }
        return FrontierProposalV40(
            request_hash=request.request_hash,
            action="execute",
            selected_node_hash=request.candidate_nodes[0].node_hash,
            draft=json.dumps(draft),
            rationale="select a strict baseline control candidate",
        )

    transport = FixtureFrontierTransportV40(propose)
    outcome = run_unseen_event_process_experiment_v40(
        tmp_path,
        spec,
        transport,
        fetch_bytes=fetch,
        retrieved_at=NOW,
    )

    assert outcome.report.decision == "rejected"
    assert not outcome.evaluation.gates["confirmation_log_score_lift"]
    assert outcome.report.real_world_execution_permitted is False
    assert verify_unseen_event_process_experiment_v40(outcome, spec)
    before = outcome.graph.store.event_path.read_bytes()

    resumed = run_unseen_event_process_experiment_v40(
        tmp_path,
        spec,
        transport,
        fetch_bytes=lambda _: (_ for _ in ()).throw(AssertionError("refetched")),
        retrieved_at=NOW,
    )
    assert resumed.report == outcome.report
    assert resumed.graph.store.event_path.read_bytes() == before
    assert len(transport.requests) == 1
