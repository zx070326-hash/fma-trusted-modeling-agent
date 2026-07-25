from __future__ import annotations

import json
import math
import re
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from typing import Annotated, Callable, Literal

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel

from .empirical_schemas import (
    TimeSeriesDataContract,
    TimeSeriesPoint,
    TimeSeriesSnapshot,
)
from .schemas import Identifier, Sha256, _assert_timezone
from .timeseries_intake import QUALITY_CHECKS


BLS_ENDPOINT = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
USGS_ENDPOINT = "https://waterservices.usgs.gov/nwis/dv/"
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
OfficialFetcher = Callable[[str, str, bytes | None], bytes]


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class OfficialDataReceipt(StrictModel):
    schema_version: Literal["2.1"] = "2.1"
    receipt_id: Identifier
    provider: Literal["bls", "usgs"]
    endpoint: Literal[
        "https://api.bls.gov/publicAPI/v2/timeseries/data/",
        "https://waterservices.usgs.gov/nwis/dv/",
    ]
    request_hash: Sha256
    response_content_hash: Sha256
    data_snapshot_hash: Sha256
    revision_status: Literal["published_but_revision_prone"] = (
        "published_but_revision_prone"
    )
    observed_qualifiers: list[Annotated[str, Field(min_length=1, max_length=80)]]
    fetched_at: datetime
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "OfficialDataReceipt":
        _assert_timezone(self.fetched_at, "fetched_at")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match official data receipt")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("official data receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "OfficialDataReceipt":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


def fetch_bls_monthly_series(
    contract: TimeSeriesDataContract,
    *,
    series_id: str,
    start_year: int,
    end_year: int,
    fetched_at: datetime | None = None,
    fetcher: OfficialFetcher | None = None,
    _raw_sink: Callable[[bytes], None] | None = None,
) -> tuple[TimeSeriesSnapshot, OfficialDataReceipt]:
    contract.assert_sealed()
    _require_official_contract(contract)
    if not re.fullmatch(r"[A-Z0-9_#-]{3,50}", series_id):
        raise ValueError("BLS series id is not allowlisted syntax")
    if end_year < start_year or end_year - start_year + 1 > 10:
        raise ValueError("unregistered BLS query is limited to a 1-10 year range")
    expected_source = f"bls:{series_id}:{start_year}:{end_year}"
    if contract.source_ref != expected_source or contract.frequency != "monthly":
        raise ValueError("BLS request does not match the frozen data contract")
    request_payload = {
        "seriesid": [series_id],
        "startyear": str(start_year),
        "endyear": str(end_year),
    }
    request_bytes = json.dumps(
        request_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    raw = (fetcher or _fetch_official)("POST", BLS_ENDPOINT, request_bytes)
    if _raw_sink is not None:
        _raw_sink(raw)
    response_hash = sha256_value({"raw_response_utf8": _decode_utf8(raw)})
    payload = _load_json(raw)
    if payload.get("status") != "REQUEST_SUCCEEDED":
        raise ValueError("BLS response status is not REQUEST_SUCCEEDED")
    series = payload.get("Results", {}).get("series", [])
    if len(series) != 1 or series[0].get("seriesID") != series_id:
        raise ValueError("BLS response does not contain the requested single series")
    points: list[TimeSeriesPoint] = []
    qualifiers: set[str] = set()
    for item in series[0].get("data", []):
        period = str(item.get("period", ""))
        if period == "M13":
            continue
        if not re.fullmatch(r"M(0[1-9]|1[0-2])", period):
            raise ValueError("BLS response contains an unexpected period")
        year = int(item["year"])
        month = int(period[1:])
        value = _finite_nonnegative(item.get("value"), "BLS value")
        for footnote in item.get("footnotes", []):
            text = str(footnote.get("text") or "").strip()
            if text:
                qualifiers.add(text)
        points.append(
            TimeSeriesPoint(
                timestamp=datetime(year, month, 1, tzinfo=timezone.utc), value=value
            )
        )
    points.sort(key=lambda point: point.timestamp)
    _assert_monthly_complete(points, start_year, end_year)
    snapshot, receipt_time = _official_snapshot(
        contract,
        points,
        response_hash,
        snapshot_id=f"bls_{series_id.lower()}_{start_year}_{end_year}",
        fetched_at=fetched_at,
    )
    assert snapshot.snapshot_hash is not None
    return snapshot, OfficialDataReceipt.seal(
        receipt_id=f"bls_{series_id.lower()}_receipt",
        provider="bls",
        endpoint=BLS_ENDPOINT,
        request_hash=sha256_value(
            {"method": "POST", "endpoint": BLS_ENDPOINT, "payload": request_payload}
        ),
        response_content_hash=response_hash,
        data_snapshot_hash=snapshot.snapshot_hash,
        observed_qualifiers=sorted(qualifiers) or ["no_nonempty_footnotes"],
        fetched_at=receipt_time,
    )


def fetch_usgs_daily_values(
    contract: TimeSeriesDataContract,
    *,
    site_id: str,
    start_date: date,
    end_date: date,
    parameter_code: str = "00060",
    statistic_code: str = "00003",
    fetched_at: datetime | None = None,
    fetcher: OfficialFetcher | None = None,
    _raw_sink: Callable[[bytes], None] | None = None,
) -> tuple[TimeSeriesSnapshot, OfficialDataReceipt]:
    contract.assert_sealed()
    _require_official_contract(contract)
    if not re.fullmatch(r"[0-9]{8,15}", site_id):
        raise ValueError("USGS site id is not allowlisted syntax")
    if not re.fullmatch(r"[0-9]{5}", parameter_code) or not re.fullmatch(
        r"[0-9]{5}", statistic_code
    ):
        raise ValueError("USGS parameter/statistic codes must be five digits")
    if end_date < start_date or (end_date - start_date).days > 3660:
        raise ValueError("USGS date range must be ordered and no longer than ten years")
    expected_source = (
        f"usgs:{site_id}:{parameter_code}:{statistic_code}:"
        f"{start_date.isoformat()}:{end_date.isoformat()}"
    )
    if contract.source_ref != expected_source or contract.frequency != "daily":
        raise ValueError("USGS request does not match the frozen data contract")
    query = {
        "format": "json",
        "sites": site_id,
        "startDT": start_date.isoformat(),
        "endDT": end_date.isoformat(),
        "parameterCd": parameter_code,
        "statCd": statistic_code,
        "siteStatus": "all",
    }
    request_url = USGS_ENDPOINT + "?" + urllib.parse.urlencode(query)
    raw = (fetcher or _fetch_official)("GET", request_url, None)
    if _raw_sink is not None:
        _raw_sink(raw)
    response_hash = sha256_value({"raw_response_utf8": _decode_utf8(raw)})
    payload = _load_json(raw)
    series = payload.get("value", {}).get("timeSeries", [])
    if len(series) != 1:
        raise ValueError("USGS response must contain exactly one time series")
    source_codes = series[0].get("sourceInfo", {}).get("siteCode", [])
    variable_codes = series[0].get("variable", {}).get("variableCode", [])
    if site_id not in {str(code.get("value")) for code in source_codes}:
        raise ValueError("USGS response site does not match the request")
    if parameter_code not in {str(code.get("value")) for code in variable_codes}:
        raise ValueError("USGS response parameter does not match the request")
    value_groups = series[0].get("values", [])
    if len(value_groups) != 1:
        raise ValueError("USGS response must contain one value group")
    points: list[TimeSeriesPoint] = []
    qualifiers: set[str] = set()
    for item in value_groups[0].get("value", []):
        timestamp_text = str(item.get("dateTime", ""))
        try:
            day = date.fromisoformat(timestamp_text[:10])
        except ValueError as exc:
            raise ValueError("USGS response contains an invalid daily timestamp") from exc
        value = _finite_nonnegative(item.get("value"), "USGS value")
        qualifiers.update(str(value) for value in item.get("qualifiers", []))
        points.append(
            TimeSeriesPoint(
                timestamp=datetime(day.year, day.month, day.day, tzinfo=timezone.utc),
                value=value,
            )
        )
    points.sort(key=lambda point: point.timestamp)
    _assert_daily_complete(points, start_date, end_date)
    snapshot, receipt_time = _official_snapshot(
        contract,
        points,
        response_hash,
        snapshot_id=f"usgs_{site_id}_{start_date.year}_{end_date.year}",
        fetched_at=fetched_at,
    )
    assert snapshot.snapshot_hash is not None
    return snapshot, OfficialDataReceipt.seal(
        receipt_id=f"usgs_{site_id}_receipt",
        provider="usgs",
        endpoint=USGS_ENDPOINT,
        request_hash=sha256_value(
            {"method": "GET", "endpoint": USGS_ENDPOINT, "query": query}
        ),
        response_content_hash=response_hash,
        data_snapshot_hash=snapshot.snapshot_hash,
        observed_qualifiers=sorted(qualifiers) or ["no_qualifiers"],
        fetched_at=receipt_time,
    )


def _fetch_official(method: str, url: str, body: bytes | None) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {
        "api.bls.gov",
        "waterservices.usgs.gov",
    }:
        raise ValueError("official fetch is restricted to the BLS and USGS HTTPS hosts")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "FMA-trusted-modeling-agent/0.2 research-shadow",
        },
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        if response.status != 200:
            raise RuntimeError(f"official API returned HTTP {response.status}")
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("official API response exceeds the size limit")
    return raw


def _official_snapshot(
    contract: TimeSeriesDataContract,
    points: list[TimeSeriesPoint],
    response_hash: str,
    *,
    snapshot_id: str,
    fetched_at: datetime | None,
) -> tuple[TimeSeriesSnapshot, datetime]:
    if len(points) < contract.minimum_points:
        raise ValueError("official series violates the contract minimum length")
    observed_at = fetched_at or datetime.now(timezone.utc)
    assert contract.data_contract_hash is not None
    return (
        TimeSeriesSnapshot.seal(
            snapshot_id=snapshot_id,
            data_contract_hash=contract.data_contract_hash,
            source_content_hash=response_hash,
            points=points,
            quality_checks=[
                *QUALITY_CHECKS,
                "source_identity_verified",
                "frequency_complete",
            ],
            collected_at=observed_at,
        ),
        observed_at,
    )


def _require_official_contract(contract: TimeSeriesDataContract) -> None:
    if contract.source_kind != "official_api":
        raise ValueError("official adapter requires an official_api data contract")


def _load_json(raw: bytes) -> dict:
    try:
        payload = json.loads(_decode_utf8(raw))
    except json.JSONDecodeError as exc:
        raise ValueError("official API response is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("official API response must be a JSON object")
    return payload


def _decode_utf8(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("official API response must be UTF-8") from exc


def _finite_nonnegative(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return number


def _assert_monthly_complete(
    points: list[TimeSeriesPoint], start_year: int, end_year: int
) -> None:
    expected = [
        (year, month)
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
    ]
    observed = [(point.timestamp.year, point.timestamp.month) for point in points]
    if observed != expected:
        raise ValueError("BLS monthly series is incomplete for the frozen date range")


def _assert_daily_complete(
    points: list[TimeSeriesPoint], start_date: date, end_date: date
) -> None:
    expected_count = (end_date - start_date).days + 1
    if len(points) != expected_count:
        raise ValueError("USGS daily series is incomplete for the frozen date range")
    observed_days = [point.timestamp.date() for point in points]
    if observed_days[0] != start_date or observed_days[-1] != end_date:
        raise ValueError("USGS daily series does not cover the frozen date range")
    if any((right - left).days != 1 for left, right in zip(observed_days, observed_days[1:])):
        raise ValueError("USGS daily series contains a gap or duplicate")
