from __future__ import annotations

import json
from pathlib import Path

from fma.v2.official_shadow import (
    ITERATION_TIME,
    run_official_shadow_benchmark,
    verify_official_shadow_run,
)


def _bls_response(series_id: str = "CES0000000001") -> bytes:
    data = []
    index = 0
    for year in range(2016, 2026):
        for month in range(1, 13):
            seasonal = [0, 4, 7, 9, 7, 4, 0, -4, -7, -9, -7, -4][month - 1]
            data.append(
                {
                    "year": str(year),
                    "period": f"M{month:02d}",
                    "periodName": "fixture",
                    "value": str(1000 + 0.1 * index + seasonal),
                    "footnotes": [{}],
                }
            )
            index += 1
    data.reverse()
    return json.dumps(
        {
            "status": "REQUEST_SUCCEEDED",
            "message": [],
            "Results": {"series": [{"seriesID": series_id, "data": data}]},
        }
    ).encode("utf-8")


def test_official_shadow_run_persists_raw_source_and_recomputes_every_report(
    tmp_path: Path,
) -> None:
    raw = _bls_response()
    outcome = run_official_shadow_benchmark(
        tmp_path,
        dataset_name="bls_nonfarm_employment",
        fetched_at=ITERATION_TIME,
        fetcher=lambda *_: raw,
    )

    assert outcome.report.real_world_action_authorized is False
    assert outcome.report.warnings == [
        "retrospective_only",
        "published_source_is_revision_prone",
        "no_real_world_decision_contract",
    ]
    assert verify_official_shadow_run(outcome.store.run_directory)

    raw_ref = next(
        ref
        for ref in outcome.manifest.artifact_refs
        if ref.kind == "official_api_raw_response"
    )
    raw_path = outcome.store.run_directory / raw_ref.relative_path
    envelope = json.loads(raw_path.read_text(encoding="utf-8"))
    envelope["payload"]["utf8"] += " "
    raw_path.write_text(json.dumps(envelope), encoding="utf-8")
    assert verify_official_shadow_run(outcome.store.run_directory) is False


def test_failure_evolved_portfolio_is_replayed_on_a_withheld_series(
    tmp_path: Path,
) -> None:
    outcome = run_official_shadow_benchmark(
        tmp_path,
        dataset_name="bls_private_weekly_hours",
        fetched_at=ITERATION_TIME,
        fetcher=lambda *_: _bls_response("CES0500000002"),
    )

    candidate_ids = {
        result.candidate_id for result in outcome.validation_report.results
    }
    assert "local_trend_challenger" in candidate_ids
    assert "exponential_level_challenger" in candidate_ids
    assert verify_official_shadow_run(outcome.store.run_directory)
