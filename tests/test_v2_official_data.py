from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest

from fma.v2.empirical_schemas import TimeSeriesDataContract
from fma.v2.official_data import (
    BLS_ENDPOINT,
    fetch_bls_monthly_series,
    fetch_usgs_daily_values,
)


NOW = datetime(2026, 7, 22, tzinfo=timezone.utc)


def _contract(**updates: object) -> TimeSeriesDataContract:
    data: dict[str, object] = {
        "dataset_id": "official_series",
        "mission_spec_hash": "a" * 64,
        "source_kind": "official_api",
        "source_ref": "bls:CES0000000001:2020:2020",
        "frequency": "monthly",
        "value_unit": "thousand_persons",
        "minimum_points": 12,
        "created_at": NOW,
    }
    data.update(updates)
    return TimeSeriesDataContract.seal(**data)


def test_bls_adapter_binds_request_identity_and_sorts_months() -> None:
    response = {
        "status": "REQUEST_SUCCEEDED",
        "message": [],
        "Results": {
            "series": [
                {
                    "seriesID": "CES0000000001",
                    "data": [
                        {
                            "year": "2020",
                            "period": f"M{month:02d}",
                            "value": str(100 + month),
                            "footnotes": [{}],
                        }
                        for month in range(12, 0, -1)
                    ],
                }
            ]
        },
    }
    calls: list[tuple[str, str, bytes | None]] = []

    def fetcher(method: str, url: str, body: bytes | None) -> bytes:
        calls.append((method, url, body))
        return json.dumps(response).encode("utf-8")

    snapshot, receipt = fetch_bls_monthly_series(
        _contract(),
        series_id="CES0000000001",
        start_year=2020,
        end_year=2020,
        fetched_at=NOW,
        fetcher=fetcher,
    )

    snapshot.assert_sealed()
    receipt.assert_sealed()
    assert calls[0][0:2] == ("POST", BLS_ENDPOINT)
    assert [point.timestamp.month for point in snapshot.points] == list(range(1, 13))
    assert snapshot.snapshot_hash == receipt.data_snapshot_hash
    assert "source_identity_verified" in snapshot.quality_checks
    assert receipt.revision_status == "published_but_revision_prone"


def test_bls_adapter_rejects_an_incomplete_or_mismatched_response() -> None:
    incomplete = {
        "status": "REQUEST_SUCCEEDED",
        "Results": {
            "series": [
                {
                    "seriesID": "CES0000000001",
                    "data": [
                        {
                            "year": "2020",
                            "period": "M01",
                            "value": "100",
                            "footnotes": [{}],
                        }
                    ],
                }
            ]
        },
    }
    with pytest.raises(ValueError, match="incomplete"):
        fetch_bls_monthly_series(
            _contract(),
            series_id="CES0000000001",
            start_year=2020,
            end_year=2020,
            fetched_at=NOW,
            fetcher=lambda *_: json.dumps(incomplete).encode("utf-8"),
        )

    with pytest.raises(ValueError, match="frozen data contract"):
        fetch_bls_monthly_series(
            _contract(source_ref="bls:DIFFERENT:2020:2020"),
            series_id="CES0000000001",
            start_year=2020,
            end_year=2020,
            fetched_at=NOW,
            fetcher=lambda *_: b"{}",
        )


def test_usgs_adapter_verifies_site_parameter_and_daily_completeness() -> None:
    start = date(2024, 1, 1)
    end = date(2024, 1, 12)
    values = [
        {
            "value": str(100 + index),
            "qualifiers": ["A"],
            "dateTime": (start + timedelta(days=index)).isoformat() + "T00:00:00.000",
        }
        for index in range(12)
    ]
    response = {
        "value": {
            "timeSeries": [
                {
                    "sourceInfo": {
                        "siteCode": [{"value": "01646500", "agencyCode": "USGS"}]
                    },
                    "variable": {"variableCode": [{"value": "00060"}]},
                    "values": [{"value": values}],
                }
            ]
        }
    }
    contract = _contract(
        source_ref="usgs:01646500:00060:00003:2024-01-01:2024-01-12",
        frequency="daily",
        value_unit="cubic_feet_per_second",
    )

    snapshot, receipt = fetch_usgs_daily_values(
        contract,
        site_id="01646500",
        start_date=start,
        end_date=end,
        fetched_at=NOW,
        fetcher=lambda *_: json.dumps(response).encode("utf-8"),
    )

    snapshot.assert_sealed()
    receipt.assert_sealed()
    assert len(snapshot.points) == 12
    assert receipt.observed_qualifiers == ["A"]
    assert snapshot.points[0].timestamp.tzinfo == timezone.utc


def test_official_adapter_rejects_wrong_source_identity_before_snapshot() -> None:
    response = {
        "value": {
            "timeSeries": [
                {
                    "sourceInfo": {"siteCode": [{"value": "99999999"}]},
                    "variable": {"variableCode": [{"value": "00060"}]},
                    "values": [{"value": []}],
                }
            ]
        }
    }
    contract = _contract(
        source_ref="usgs:01646500:00060:00003:2024-01-01:2024-01-12",
        frequency="daily",
        value_unit="cubic_feet_per_second",
    )
    with pytest.raises(ValueError, match="site does not match"):
        fetch_usgs_daily_values(
            contract,
            site_id="01646500",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 12),
            fetched_at=NOW,
            fetcher=lambda *_: json.dumps(response).encode("utf-8"),
        )
