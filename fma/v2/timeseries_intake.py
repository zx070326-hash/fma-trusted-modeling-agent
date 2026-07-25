from __future__ import annotations

import csv
import math
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

from fma.hashing import sha256_value

from .empirical_schemas import (
    TimeSeriesDataContract,
    TimeSeriesPoint,
    TimeSeriesSnapshot,
)


MAX_CSV_BYTES = 4 * 1024 * 1024
QUALITY_CHECKS = [
    "utf8_decodable",
    "required_columns_exact",
    "no_missing_values",
    "finite_nonnegative_values",
    "timestamps_strictly_increasing",
    "minimum_length_met",
]


def ingest_local_timeseries_csv(
    path: str | Path,
    *,
    workspace_root: str | Path,
    contract: TimeSeriesDataContract,
    snapshot_id: str = "historical_demand_snapshot",
    collected_at: datetime | None = None,
) -> TimeSeriesSnapshot:
    """Read a narrow two-column UTF-8 CSV without inferring its semantics."""

    contract.assert_sealed()
    if contract.source_kind != "local_file":
        raise ValueError("local CSV intake requires a local_file data contract")
    root = Path(workspace_root).resolve(strict=True)
    source = Path(path).resolve(strict=True)
    if not source.is_relative_to(root):
        raise ValueError("time-series CSV must resolve inside the workspace root")
    if source.is_symlink() or not source.is_file():
        raise ValueError("time-series source must be a regular non-symlink file")
    if source.suffix.lower() != ".csv":
        raise ValueError("time-series intake accepts only .csv files")
    if source.stat().st_size > MAX_CSV_BYTES:
        raise ValueError("time-series CSV exceeds the intake size limit")
    raw_bytes = source.read_bytes()
    try:
        raw_text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("time-series CSV must be UTF-8") from exc
    points = _parse_csv(raw_text, contract)
    if len(points) < contract.minimum_points:
        raise ValueError(
            f"time-series needs at least {contract.minimum_points} observations"
        )
    return TimeSeriesSnapshot.seal(
        snapshot_id=snapshot_id,
        data_contract_hash=contract.data_contract_hash,
        source_content_hash=sha256_value({"raw_utf8_text": raw_text}),
        points=points,
        quality_checks=QUALITY_CHECKS,
        collected_at=collected_at or datetime.now(timezone.utc),
    )


def _parse_csv(raw_text: str, contract: TimeSeriesDataContract) -> list[TimeSeriesPoint]:
    reader = csv.DictReader(StringIO(raw_text, newline=""))
    expected = [contract.timestamp_column, contract.value_column]
    if reader.fieldnames != expected:
        raise ValueError(f"CSV columns must be exactly {expected}")
    points: list[TimeSeriesPoint] = []
    for row_number, row in enumerate(reader, start=2):
        timestamp_text = (row.get(contract.timestamp_column) or "").strip()
        value_text = (row.get(contract.value_column) or "").strip()
        if not timestamp_text or not value_text:
            raise ValueError(f"missing value at CSV row {row_number}")
        try:
            timestamp = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"invalid ISO timestamp at CSV row {row_number}") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError(f"timestamp needs an explicit UTC offset at row {row_number}")
        try:
            value = float(value_text)
        except ValueError as exc:
            raise ValueError(f"invalid numeric value at CSV row {row_number}") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"value must be finite and nonnegative at row {row_number}")
        points.append(TimeSeriesPoint(timestamp=timestamp, value=value))
    if not points:
        raise ValueError("time-series CSV contains no observations")
    if any(
        right.timestamp <= left.timestamp
        for left, right in zip(points, points[1:])
    ):
        raise ValueError("timestamps must be unique and strictly increasing")
    return points
