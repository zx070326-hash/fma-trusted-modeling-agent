"""V6.2 code-owned acquisition for one registered official public source.

The adapter is intentionally narrow.  It fetches a single World Bank
Indicators series from an exact, contract-derived HTTPS URL, rejects redirects
and malformed responses, materializes the raw response, and replays the parse
before issuing a provenance verification.  It establishes source integrity
for the exact public snapshot; it does not establish scientific qualification
or authorize any real-world action.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Callable, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5_2.ode_system import ODETimeSeriesSnapshotV52


WORLD_BANK_HOST = "api.worldbank.org"
WORLD_BANK_LICENSE_URL = "https://datacatalog.worldbank.org/public-licenses"
SOURCE_CONTRACT_PATH = "docs/source_contract_v62.json"
SOURCE_RECEIPT_PATH = "data/source_provenance_v62/receipt.json"
SOURCE_VERIFICATION_PATH = "data/source_provenance_v62/verification.json"
SOURCE_RAW_PATH = "data/source_provenance_v62/raw_response.json"

_COUNTRY = re.compile(r"^[A-Z]{3}$")
_INDICATOR = re.compile(r"^[A-Z0-9.]{3,40}$")


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WorldBankSourceContractV62(StrictModel):
    """Frozen contract for a single World Bank Indicators time series."""

    schema_version: Literal["6.2-world-bank-source-contract"] = (
        "6.2-world-bank-source-contract"
    )
    contract_id: Identifier
    country_code: str
    indicator_id: str
    start_year: Annotated[int, Field(ge=1900, le=2200)]
    end_year: Annotated[int, Field(ge=1900, le=2200)]
    minimum_observations: Annotated[int, Field(ge=12, le=301)] = 23
    time_unit: Identifier = "year"
    state_unit: Identifier
    source_class: Literal["official_public_api"] = "official_public_api"
    provider: Literal["World Bank Indicators API"] = (
        "World Bank Indicators API"
    )
    dataset_family: Literal["World Development Indicators"] = (
        "World Development Indicators"
    )
    license_id: Literal["CC-BY-4.0-default-open-data"] = (
        "CC-BY-4.0-default-open-data"
    )
    license_url: Literal[
        "https://datacatalog.worldbank.org/public-licenses"
    ] = WORLD_BANK_LICENSE_URL
    attribution: str = Field(min_length=10, max_length=500)
    max_response_bytes: Annotated[
        int, Field(ge=1024, le=8 * 1024 * 1024)
    ] = 2 * 1024 * 1024
    exact_url: str
    fixture_only: bool = False
    private_holdout_in_response_permitted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    contract_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "WorldBankSourceContractV62":
        if not _COUNTRY.fullmatch(self.country_code):
            raise ValueError("World Bank country_code must be ISO3 uppercase")
        if not _INDICATOR.fullmatch(self.indicator_id):
            raise ValueError("World Bank indicator_id is malformed")
        if self.end_year < self.start_year:
            raise ValueError("World Bank source interval is reversed")
        if self.end_year - self.start_year + 1 < self.minimum_observations:
            raise ValueError("World Bank interval cannot meet minimum observations")
        expected_url = world_bank_indicators_url_v62(
            country_code=self.country_code,
            indicator_id=self.indicator_id,
            start_year=self.start_year,
            end_year=self.end_year,
        )
        if self.exact_url != expected_url:
            raise ValueError("World Bank exact_url differs from frozen parameters")
        parsed = urlparse(self.exact_url)
        if parsed.scheme != "https" or parsed.hostname != WORLD_BANK_HOST:
            raise ValueError("World Bank source must use the registered HTTPS host")
        if self.contract_hash and self.contract_hash != self.content_hash():
            raise ValueError("World Bank source contract hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "contract_hash")

    def assert_sealed(self) -> None:
        if not self.contract_hash or self.contract_hash != self.content_hash():
            raise ValueError("World Bank source contract is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "WorldBankSourceContractV62":
        if "exact_url" not in data:
            data["exact_url"] = world_bank_indicators_url_v62(
                country_code=str(data["country_code"]),
                indicator_id=str(data["indicator_id"]),
                start_year=int(data["start_year"]),
                end_year=int(data["end_year"]),
            )
        draft = cls(**data)
        payload = draft.model_dump(exclude={"contract_hash"})
        payload["contract_hash"] = draft.content_hash()
        return cls(**payload)


class WorldBankSourceReceiptV62(StrictModel):
    """Immutable receipt for the exact bytes and parsed public series."""

    schema_version: Literal["6.2-world-bank-source-receipt"] = (
        "6.2-world-bank-source-receipt"
    )
    contract_hash: Sha256
    exact_url: str
    final_url: str
    response_status: Literal[200] = 200
    response_content_type: Literal["application/json"] = "application/json"
    response_bytes_hash: Sha256
    response_size_bytes: Annotated[int, Field(ge=1)]
    metadata_hash: Sha256
    observation_records_hash: Sha256
    snapshot_hash: Sha256
    source_id: str = Field(min_length=10, max_length=300)
    observation_count: Annotated[int, Field(ge=12)]
    first_year: Annotated[int, Field(ge=1900, le=2200)]
    last_year: Annotated[int, Field(ge=1900, le=2200)]
    raw_relative_path: Literal[
        "data/source_provenance_v62/raw_response.json"
    ] = SOURCE_RAW_PATH
    fixture_only: bool
    retrieved_at: datetime
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "WorldBankSourceReceiptV62":
        if self.final_url != self.exact_url:
            raise ValueError("World Bank receipt cannot follow a redirect")
        if self.first_year > self.last_year:
            raise ValueError("World Bank receipt year range is reversed")
        if self.retrieved_at.utcoffset() is None:
            raise ValueError("retrieved_at must be timezone-aware")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("World Bank source receipt hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("World Bank source receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "WorldBankSourceReceiptV62":
        data.setdefault("retrieved_at", _utc_now())
        draft = cls(**data)
        payload = draft.model_dump(exclude={"receipt_hash"})
        payload["receipt_hash"] = draft.content_hash()
        return cls(**payload)


class SourceVerificationV62(StrictModel):
    """Code-owned replay result; never an external qualification certificate."""

    schema_version: Literal["6.2-source-verification"] = (
        "6.2-source-verification"
    )
    contract_hash: Sha256
    receipt_hash: Sha256
    snapshot_hash: Sha256
    status: Literal["PASS", "FAIL"]
    checks: dict[Identifier, bool]
    reason_codes: list[Identifier]
    verified_at: datetime
    fixture_only: bool
    evidence_scope: Literal[
        "official_source_integrity", "fixture_control_integrity"
    ]
    scientific_provenance_status: Literal["FAIL", "NOT_RUN", "HUMAN"]
    independent_source_review: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    verification_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_verification(self) -> "SourceVerificationV62":
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("source verification reasons must be sorted and unique")
        expected = bool(self.checks) and all(self.checks.values())
        if (self.status == "PASS") != expected:
            raise ValueError("source verification status differs from checks")
        expected_scope = (
            "fixture_control_integrity"
            if self.fixture_only
            else "official_source_integrity"
        )
        if self.evidence_scope != expected_scope:
            raise ValueError("source verification scope differs from fixture flag")
        expected_scientific_status = (
            "FAIL"
            if self.status == "FAIL"
            else "NOT_RUN"
            if self.fixture_only
            else "HUMAN"
        )
        if self.scientific_provenance_status != expected_scientific_status:
            raise ValueError(
                "scientific provenance status exceeds source verification"
            )
        if self.verified_at.utcoffset() is None:
            raise ValueError("verified_at must be timezone-aware")
        if (
            self.verification_hash
            and self.verification_hash != self.content_hash()
        ):
            raise ValueError("source verification hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "verification_hash")

    def assert_sealed(self) -> None:
        if (
            not self.verification_hash
            or self.verification_hash != self.content_hash()
        ):
            raise ValueError("source verification is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "SourceVerificationV62":
        data.setdefault("verified_at", _utc_now())
        draft = cls(**data)
        payload = draft.model_dump(exclude={"verification_hash"})
        payload["verification_hash"] = draft.content_hash()
        return cls(**payload)


@dataclass(frozen=True)
class SourceHTTPResponseV62:
    status: int
    final_url: str
    content_type: str
    body: bytes


@dataclass(frozen=True)
class FetchedWorldBankSeriesV62:
    contract: WorldBankSourceContractV62
    receipt: WorldBankSourceReceiptV62
    snapshot: ODETimeSeriesSnapshotV52
    raw_body: bytes
    transport_mode: Literal[
        "live_https_no_redirect", "fixture_injected"
    ]


SourceFetcherV62 = Callable[
    [WorldBankSourceContractV62], SourceHTTPResponseV62
]


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        return None


def world_bank_indicators_url_v62(
    *,
    country_code: str,
    indicator_id: str,
    start_year: int,
    end_year: int,
) -> str:
    query = urlencode(
        [
            ("format", "json"),
            ("date", f"{start_year}:{end_year}"),
            ("per_page", "1000"),
        ]
    )
    return (
        f"https://{WORLD_BANK_HOST}/v2/country/{country_code}/indicator/"
        f"{indicator_id}?{query}"
    )


def _default_fetcher(
    contract: WorldBankSourceContractV62,
) -> SourceHTTPResponseV62:
    opener = build_opener(_NoRedirect())
    request = Request(
        contract.exact_url,
        headers={
            "Accept": "application/json",
            "User-Agent": "FMA-Trusted-Kernel/0.3 source-integrity-adapter",
        },
        method="GET",
    )
    try:
        with opener.open(request, timeout=30) as response:
            body = response.read(contract.max_response_bytes + 1)
            content_type = response.headers.get_content_type()
            return SourceHTTPResponseV62(
                status=int(response.status),
                final_url=str(response.geturl()),
                content_type=content_type,
                body=body,
            )
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError("World Bank source fetch failed closed") from exc


def _parse_world_bank_payload(
    *,
    contract: WorldBankSourceContractV62,
    body: bytes,
) -> tuple[dict[str, object], list[dict[str, object]], list[float], list[float]]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("World Bank response is not UTF-8 JSON") from exc
    if (
        not isinstance(payload, list)
        or len(payload) != 2
        or not isinstance(payload[0], dict)
        or not isinstance(payload[1], list)
    ):
        raise ValueError("World Bank response envelope is malformed")
    metadata = payload[0]
    records = payload[1]
    if metadata.get("page") != 1 or metadata.get("pages") != 1:
        raise ValueError("World Bank response must fit one frozen page")
    if metadata.get("total") != len(records):
        raise ValueError("World Bank response total differs from records")

    by_year: dict[int, float] = {}
    normalized: list[dict[str, object]] = []
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("World Bank observation is not an object")
        indicator = item.get("indicator")
        if (
            not isinstance(indicator, dict)
            or indicator.get("id") != contract.indicator_id
        ):
            raise ValueError("World Bank observation indicator differs")
        if item.get("countryiso3code") != contract.country_code:
            raise ValueError("World Bank observation country differs")
        try:
            year = int(str(item["date"]))
            value = float(item["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("World Bank observation value is missing") from exc
        if year < contract.start_year or year > contract.end_year:
            raise ValueError("World Bank observation lies outside frozen interval")
        if year in by_year:
            raise ValueError("World Bank response contains a duplicate year")
        if not math.isfinite(value) or value <= 0:
            raise ValueError("World Bank series must be finite and positive")
        by_year[year] = value
        normalized.append(
            {
                "countryiso3code": contract.country_code,
                "date": str(year),
                "indicator_id": contract.indicator_id,
                "value": value,
            }
        )

    expected_years = list(range(contract.start_year, contract.end_year + 1))
    if sorted(by_year) != expected_years:
        raise ValueError("World Bank response has missing or unexpected years")
    if len(by_year) < contract.minimum_observations:
        raise ValueError("World Bank response has too few observations")
    times = [float(year) for year in expected_years]
    observations = [by_year[year] for year in expected_years]
    normalized.sort(key=lambda item: int(str(item["date"])))
    return metadata, normalized, times, observations


def fetch_world_bank_series_v62(
    *,
    task_id: str,
    contract: WorldBankSourceContractV62,
    fetcher: SourceFetcherV62 | None = None,
    retrieved_at: datetime | None = None,
) -> FetchedWorldBankSeriesV62:
    contract.assert_sealed()
    if fetcher is not None and not contract.fixture_only:
        raise ValueError(
            "an injected source fetcher requires fixture_only=true"
        )
    if fetcher is None and contract.fixture_only:
        raise ValueError(
            "a fixture source contract requires an injected fetcher"
        )
    response = (fetcher or _default_fetcher)(contract)
    if response.status != 200:
        raise ValueError("World Bank response status must be 200")
    if response.final_url != contract.exact_url:
        raise ValueError("World Bank source redirected away from exact URL")
    if response.content_type.lower() != "application/json":
        raise ValueError("World Bank response content type is not JSON")
    if not response.body or len(response.body) > contract.max_response_bytes:
        raise ValueError("World Bank response size is outside contract")

    metadata, records, times, observations = _parse_world_bank_payload(
        contract=contract,
        body=response.body,
    )
    response_hash = hashlib.sha256(response.body).hexdigest()
    source_id = (
        f"world-bank:{contract.country_code}:{contract.indicator_id}:"
        f"{contract.start_year}-{contract.end_year}"
    )
    snapshot = ODETimeSeriesSnapshotV52.seal(
        task_id=task_id,
        time_unit=contract.time_unit,
        state_unit=contract.state_unit,
        times=times,
        observations=observations,
        source_id=source_id,
        fixture_only=contract.fixture_only,
    )
    receipt = WorldBankSourceReceiptV62.seal(
        contract_hash=contract.contract_hash,
        exact_url=contract.exact_url,
        final_url=response.final_url,
        response_bytes_hash=response_hash,
        response_size_bytes=len(response.body),
        metadata_hash=sha256_value(metadata),
        observation_records_hash=sha256_value(records),
        snapshot_hash=snapshot.snapshot_hash,
        source_id=source_id,
        observation_count=len(observations),
        first_year=contract.start_year,
        last_year=contract.end_year,
        fixture_only=contract.fixture_only,
        retrieved_at=retrieved_at or _utc_now(),
    )
    return FetchedWorldBankSeriesV62(
        contract=contract,
        receipt=receipt,
        snapshot=snapshot,
        raw_body=response.body,
        transport_mode=(
            "fixture_injected"
            if fetcher is not None
            else "live_https_no_redirect"
        ),
    )


def materialize_world_bank_series_v62(
    *,
    workspace_root: str | Path,
    fetched: FetchedWorldBankSeriesV62,
) -> None:
    root = Path(workspace_root).resolve()
    payloads: tuple[tuple[str, bytes], ...] = (
        (
            SOURCE_CONTRACT_PATH,
            (
                json.dumps(
                    fetched.contract.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8"),
        ),
        (SOURCE_RAW_PATH, fetched.raw_body),
        (
            SOURCE_RECEIPT_PATH,
            (
                json.dumps(
                    fetched.receipt.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8"),
        ),
    )
    for relative_path, payload in payloads:
        path = (root / relative_path).resolve()
        if root not in path.parents:
            raise ValueError("source artifact path escapes workspace")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as handle:
                handle.write(payload)
        except FileExistsError as exc:
            raise ValueError(
                f"refusing to overwrite source artifact: {relative_path}"
            ) from exc


def verify_world_bank_source_v62(
    *,
    workspace_root: str | Path,
    contract: WorldBankSourceContractV62,
    receipt: WorldBankSourceReceiptV62,
    snapshot: ODETimeSeriesSnapshotV52,
    verified_at: datetime | None = None,
) -> SourceVerificationV62:
    root = Path(workspace_root).resolve()
    reasons: list[str] = []
    try:
        contract.assert_sealed()
    except ValueError:
        reasons.append("source_contract_invalid")
    try:
        receipt.assert_sealed()
    except ValueError:
        reasons.append("source_receipt_invalid")
    try:
        snapshot.assert_sealed()
    except ValueError:
        reasons.append("source_snapshot_invalid")

    raw_path = (root / receipt.raw_relative_path).resolve()
    raw_path_safe = root in raw_path.parents
    raw_exists = raw_path_safe and raw_path.is_file()
    raw_hash_matches = bool(
        raw_exists
        and _file_hash(raw_path) == receipt.response_bytes_hash
        and raw_path.stat().st_size == receipt.response_size_bytes
    )
    replay_matches = False
    if raw_hash_matches:
        try:
            metadata, records, times, observations = _parse_world_bank_payload(
                contract=contract,
                body=raw_path.read_bytes(),
            )
            replay_snapshot = ODETimeSeriesSnapshotV52.seal(
                task_id=snapshot.task_id,
                time_unit=contract.time_unit,
                state_unit=contract.state_unit,
                times=times,
                observations=observations,
                source_id=receipt.source_id,
                fixture_only=contract.fixture_only,
            )
            replay_matches = bool(
                sha256_value(metadata) == receipt.metadata_hash
                and sha256_value(records) == receipt.observation_records_hash
                and replay_snapshot.snapshot_hash == snapshot.snapshot_hash
                and receipt.snapshot_hash == snapshot.snapshot_hash
                and receipt.contract_hash == contract.contract_hash
                and receipt.exact_url == contract.exact_url
                and receipt.source_id == snapshot.source_id
                and receipt.fixture_only == contract.fixture_only
                and snapshot.fixture_only == contract.fixture_only
            )
        except (OSError, ValueError):
            replay_matches = False
    checks = {
        "contract_sealed": "source_contract_invalid" not in reasons,
        "exact_registered_https_source": (
            receipt.exact_url == contract.exact_url
            and receipt.final_url == contract.exact_url
            and urlparse(contract.exact_url).hostname == WORLD_BANK_HOST
        ),
        "license_contract_recorded": (
            contract.license_url == WORLD_BANK_LICENSE_URL
            and contract.license_id == "CC-BY-4.0-default-open-data"
            and bool(contract.attribution.strip())
        ),
        "raw_path_safe": raw_path_safe,
        "raw_response_present": raw_exists,
        "raw_response_hash_bound": raw_hash_matches,
        "parse_replay_matches_snapshot": replay_matches,
        "fixture_scope_consistent": (
            receipt.fixture_only == contract.fixture_only
            and snapshot.fixture_only == contract.fixture_only
        ),
    }
    reasons.extend(
        check_id for check_id, passed in checks.items() if not passed
    )
    reasons = sorted(set(reasons))
    return SourceVerificationV62.seal(
        contract_hash=(
            contract.contract_hash if contract.contract_hash else "0" * 64
        ),
        receipt_hash=receipt.receipt_hash if receipt.receipt_hash else "0" * 64,
        snapshot_hash=(
            snapshot.snapshot_hash if snapshot.snapshot_hash else "0" * 64
        ),
        status="PASS" if not reasons else "FAIL",
        checks=checks,
        reason_codes=reasons,
        fixture_only=contract.fixture_only,
        evidence_scope=(
            "fixture_control_integrity"
            if contract.fixture_only
            else "official_source_integrity"
        ),
        scientific_provenance_status=(
            "FAIL"
            if reasons
            else "NOT_RUN"
            if contract.fixture_only
            else "HUMAN"
        ),
        verified_at=verified_at or _utc_now(),
    )


__all__ = [
    "FetchedWorldBankSeriesV62",
    "SOURCE_CONTRACT_PATH",
    "SOURCE_RAW_PATH",
    "SOURCE_RECEIPT_PATH",
    "SOURCE_VERIFICATION_PATH",
    "SourceHTTPResponseV62",
    "SourceVerificationV62",
    "WorldBankSourceContractV62",
    "WorldBankSourceReceiptV62",
    "fetch_world_bank_series_v62",
    "materialize_world_bank_series_v62",
    "verify_world_bank_source_v62",
    "world_bank_indicators_url_v62",
]
