"""Blinded World Bank scalar-ODE campaign materialization for V5.5.

The custodian selects the first data-quality-eligible series in a keyed
permutation, encrypts targets and exact source provenance under separate keys,
and releases only a transformed public series plus public commitments.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Callable, Literal

import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import Field, model_validator

from fma.hashing import canonical_json, sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5.external_harness import PrivateTargetV50
from fma.v5_2.ode_system import ODEThresholdsV52, ODETimeSeriesSnapshotV52
from fma.v5_3.custody import PrivateScoreContractV53
from fma.v5_3.ode_forecast import ODEForecastPlanV53, ODEForecastTargetV53

from .campaign_protocol import (
    ProspectiveCampaignProtocolV55,
    materialize_public_launch_v55,
)
from .split_custody import (
    SourceProvenanceDraftV55,
    create_split_custody_envelopes_v55,
)


FetcherV55 = Callable[[str], bytes]
WORLD_BANK_API_BASE_V55 = "https://api.worldbank.org/v2"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


class WorldBankSelectionSpecV55(StrictModel):
    """Frozen before the custodian chooses a source pair."""

    schema_version: Literal["5.5-world-bank-selection-spec"] = (
        "5.5-world-bank-selection-spec"
    )
    selection_spec_id: Identifier
    protocol_hash: Sha256
    api_base: Literal["https://api.worldbank.org/v2"] = WORLD_BANK_API_BASE_V55
    country_codes: Annotated[
        list[Annotated[str, Field(pattern=r"^[A-Z]{3}$")]],
        Field(min_length=4),
    ]
    indicator_codes: Annotated[
        list[Annotated[str, Field(pattern=r"^[A-Z0-9.]{5,40}$")]],
        Field(min_length=3),
    ]
    public_start_year: Annotated[int, Field(ge=1960, le=2100)]
    public_end_year: Annotated[int, Field(ge=1960, le=2100)]
    private_end_year: Annotated[int, Field(ge=1960, le=2100)]
    public_observation_count: Literal[28] = 28
    private_target_count: Literal[4] = 4
    ode_threshold_hash: Sha256
    private_minimum_quality_score: Annotated[
        float,
        Field(ge=0, le=1, allow_inf_nan=False),
    ] = 0.2
    quality_scale_rule: Literal["max_public_iqr_range_over_10_epsilon"] = (
        "max_public_iqr_range_over_10_epsilon"
    )
    selection_rule: Literal[
        "secret_hmac_permutation_first_data_quality_eligible"
    ] = "secret_hmac_permutation_first_data_quality_eligible"
    eligibility_rule: Literal[
        "complete_positive_finite_annual_window_only"
    ] = "complete_positive_finite_annual_window_only"
    value_transform: Literal["positive_scale_to_secret_base_index"] = (
        "positive_scale_to_secret_base_index"
    )
    time_transform: Literal["additive_integer_translation"] = (
        "additive_integer_translation"
    )
    prior_campaign_exclusion_hashes: list[Sha256]
    source_identity_withheld_until_closeout: Literal[True] = True
    selection_uses_no_model_score: Literal[True] = True
    fixture_only: bool
    frozen_at: datetime
    selection_spec_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "WorldBankSelectionSpecV55":
        if self.country_codes != sorted(set(self.country_codes)):
            raise ValueError("country codes must be sorted and unique")
        if self.indicator_codes != sorted(set(self.indicator_codes)):
            raise ValueError("indicator codes must be sorted and unique")
        if self.prior_campaign_exclusion_hashes != sorted(
            set(self.prior_campaign_exclusion_hashes)
        ):
            raise ValueError("prior campaign exclusions must be sorted and unique")
        expected_public = self.public_end_year - self.public_start_year + 1
        if expected_public != self.public_observation_count:
            raise ValueError("public year window must contain exactly 28 years")
        if (
            self.private_end_year - self.public_end_year
            != self.private_target_count
        ):
            raise ValueError("private year window must contain exactly four years")
        if self.frozen_at.utcoffset() is None:
            raise ValueError("selection spec frozen_at must be timezone-aware")
        if (
            self.selection_spec_hash
            and self.selection_spec_hash != self.content_hash()
        ):
            raise ValueError("World Bank selection spec hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "selection_spec_hash")

    def assert_sealed(self) -> None:
        if (
            not self.selection_spec_hash
            or self.selection_spec_hash != self.content_hash()
        ):
            raise ValueError("World Bank selection spec is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "WorldBankSelectionSpecV55":
        data.setdefault("frozen_at", _utc_now())
        draft = cls(**data)
        payload = draft.model_dump(exclude={"selection_spec_hash"})
        payload["selection_spec_hash"] = draft.content_hash()
        return cls(**payload)


class WorldBankPublicObservationV55(StrictModel):
    observation_id: Identifier
    time: Annotated[float, Field(allow_inf_nan=False)]
    value: Annotated[float, Field(gt=0, allow_inf_nan=False)]


class WorldBankPublicTargetV55(StrictModel):
    target_id: Identifier
    horizon_steps: Annotated[int, Field(ge=1, le=4)]
    time: Annotated[float, Field(allow_inf_nan=False)]


class WorldBankPublicTaskPacketV55(StrictModel):
    """Generator-visible task packet without exact source identity."""

    schema_version: Literal["5.5-world-bank-public-task"] = (
        "5.5-world-bank-public-task"
    )
    task_id: Identifier
    protocol_hash: Sha256
    selection_spec_hash: Sha256
    selection_seed_commitment: Sha256
    task_kind: Literal["LOCAL_BLINDED_REAL_DATA_ODE"] = (
        "LOCAL_BLINDED_REAL_DATA_ODE"
    )
    domain_class: Literal["annual_official_scalar_state"] = (
        "annual_official_scalar_state"
    )
    public_observations: Annotated[
        list[WorldBankPublicObservationV55],
        Field(min_length=28, max_length=28),
    ]
    targets: Annotated[
        list[WorldBankPublicTargetV55],
        Field(min_length=4, max_length=4),
    ]
    cadence: Literal["one_year"] = "one_year"
    state_unit: Literal["blinded_positive_relative_index"] = (
        "blinded_positive_relative_index"
    )
    time_unit: Literal["blinded_annual_index"] = "blinded_annual_index"
    source_identity_status: Literal["encrypted_until_closeout"] = (
        "encrypted_until_closeout"
    )
    source_inference_resistance_established: Literal[False] = False
    transform_disclosure: Literal[
        "positive_scaling_and_additive_time_translation_only"
    ] = "positive_scaling_and_additive_time_translation_only"
    fixture_only: bool
    same_host_logical_custody_only: Literal[True] = True
    external_host_established: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    packet_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_packet(self) -> "WorldBankPublicTaskPacketV55":
        observation_ids = [
            item.observation_id for item in self.public_observations
        ]
        observation_times = [item.time for item in self.public_observations]
        target_ids = [item.target_id for item in self.targets]
        target_times = [item.time for item in self.targets]
        if observation_ids != sorted(set(observation_ids)):
            raise ValueError("public observation IDs must be sorted and unique")
        if target_ids != sorted(set(target_ids)):
            raise ValueError("public target IDs must be sorted and unique")
        if any(
            right <= left
            for left, right in zip(observation_times, observation_times[1:])
        ):
            raise ValueError("public observation times must be increasing")
        if any(
            right <= left for left, right in zip(target_times, target_times[1:])
        ):
            raise ValueError("public target times must be increasing")
        if target_times[0] <= observation_times[-1]:
            raise ValueError("private targets must follow public observations")
        if [item.horizon_steps for item in self.targets] != [1, 2, 3, 4]:
            raise ValueError("target horizons must be exactly one through four")
        if self.packet_hash and self.packet_hash != self.content_hash():
            raise ValueError("World Bank public task packet hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "packet_hash")

    def assert_sealed(self) -> None:
        if not self.packet_hash or self.packet_hash != self.content_hash():
            raise ValueError("World Bank public task packet is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "WorldBankPublicTaskPacketV55":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"packet_hash"})
        payload["packet_hash"] = draft.content_hash()
        return cls(**payload)


class WorldBankPublicArtifactV55(StrictModel):
    path: Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]+$")]
    sha256: Sha256
    size_bytes: Annotated[int, Field(ge=1)]


class WorldBankPublicManifestV55(StrictModel):
    schema_version: Literal["5.5-world-bank-public-manifest"] = (
        "5.5-world-bank-public-manifest"
    )
    task_id: Identifier
    protocol_hash: Sha256
    selection_spec_hash: Sha256
    files: Annotated[list[WorldBankPublicArtifactV55], Field(min_length=10)]
    private_target_values_disclosed: Literal[False] = False
    source_provenance_disclosed: Literal[False] = False
    fixture_only: bool
    external_host_established: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "WorldBankPublicManifestV55":
        paths = [item.path for item in self.files]
        if paths != sorted(set(paths)):
            raise ValueError("public manifest paths must be sorted and unique")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("World Bank public manifest hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    @classmethod
    def seal(cls, **data: object) -> "WorldBankPublicManifestV55":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"manifest_hash"})
        payload["manifest_hash"] = draft.content_hash()
        return cls(**payload)


class WorldBankCustodianSummaryV55(StrictModel):
    schema_version: Literal["5.5-world-bank-custodian-summary"] = (
        "5.5-world-bank-custodian-summary"
    )
    task_id: Identifier
    protocol_hash: Sha256
    selection_spec_hash: Sha256
    selection_seed_commitment: Sha256
    public_task_packet_hash: Sha256
    private_target_envelope_hash: Sha256
    source_provenance_envelope_hash: Sha256
    split_custody_attestation_hash: Sha256
    public_manifest_hash: Sha256
    source_identity_disclosed: Literal[False] = False
    private_target_values_disclosed: Literal[False] = False
    fixture_only: bool
    external_host_established: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False


def _default_fetcher(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "FMA-V5.5-custodian/1.0"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        final = urllib.parse.urlsplit(response.geturl())
        if final.scheme != "https" or final.hostname != "api.worldbank.org":
            raise ValueError("World Bank API redirected outside the pinned host")
        content_type = response.headers.get_content_type()
        if content_type not in {"application/json", "text/json", "text/plain"}:
            raise ValueError("World Bank API returned an unexpected media type")
        payload = response.read(4_000_001)
    if len(payload) > 4_000_000:
        raise ValueError("World Bank API response exceeds the size limit")
    return payload


def _source_url(
    *,
    spec: WorldBankSelectionSpecV55,
    country_code: str,
    indicator_code: str,
) -> str:
    query = urllib.parse.urlencode(
        {
            "date": f"{spec.public_start_year}:{spec.private_end_year}",
            "format": "json",
            "per_page": "100",
        }
    )
    return (
        f"{spec.api_base}/country/{country_code}/indicator/"
        f"{indicator_code}?{query}"
    )


def _candidate_order(
    spec: WorldBankSelectionSpecV55,
    selection_seed: bytes,
) -> list[tuple[str, str]]:
    candidates = [
        (country_code, indicator_code)
        for country_code in spec.country_codes
        for indicator_code in spec.indicator_codes
    ]
    return sorted(
        candidates,
        key=lambda item: hmac.new(
            selection_seed,
            f"select|{item[0]}|{item[1]}".encode("ascii"),
            hashlib.sha256,
        ).digest(),
    )


def _parse_complete_series(
    *,
    raw_bytes: bytes,
    spec: WorldBankSelectionSpecV55,
    country_code: str,
    indicator_code: str,
) -> tuple[list[float], str, str] | None:
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
        if (
            not isinstance(payload, list)
            or len(payload) < 2
            or not isinstance(payload[1], list)
        ):
            return None
        by_year: dict[int, float] = {}
        indicator_name = ""
        country_name = ""
        for row in payload[1]:
            if not isinstance(row, dict):
                return None
            year = int(row["date"])
            value = float(row["value"])
            if not math.isfinite(value) or value <= 0 or year in by_year:
                return None
            if str(row.get("countryiso3code", "")).upper() != country_code:
                return None
            indicator = row.get("indicator")
            country = row.get("country")
            if not isinstance(indicator, dict) or not isinstance(country, dict):
                return None
            if str(indicator.get("id", "")) != indicator_code:
                return None
            indicator_name = str(indicator.get("value", "")).strip()
            country_name = str(country.get("value", "")).strip()
            by_year[year] = value
        years = list(range(spec.public_start_year, spec.private_end_year + 1))
        if sorted(by_year) != years or not indicator_name or not country_name:
            return None
        return [by_year[year] for year in years], country_name, indicator_name
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, UnicodeError):
        return None


def _select_source(
    *,
    spec: WorldBankSelectionSpecV55,
    selection_seed: bytes,
    fetcher: FetcherV55,
) -> tuple[list[float], str, str, str, bytes]:
    for country_code, indicator_code in _candidate_order(spec, selection_seed):
        url = _source_url(
            spec=spec,
            country_code=country_code,
            indicator_code=indicator_code,
        )
        try:
            raw_bytes = fetcher(url)
        except (OSError, TimeoutError, ValueError):
            continue
        parsed = _parse_complete_series(
            raw_bytes=raw_bytes,
            spec=spec,
            country_code=country_code,
            indicator_code=indicator_code,
        )
        if parsed is not None:
            values, country_name, indicator_name = parsed
            return values, country_name, indicator_name, url, raw_bytes
    raise ValueError("no source candidate satisfied the frozen data-quality rule")


def _transform_series(
    *,
    raw_values: list[float],
    selection_seed: bytes,
) -> tuple[list[float], list[float]]:
    value_digest = hmac.new(
        selection_seed,
        b"value-transform",
        hashlib.sha256,
    ).digest()
    time_digest = hmac.new(
        selection_seed,
        b"time-transform",
        hashlib.sha256,
    ).digest()
    base_index = 80.0 + int.from_bytes(value_digest[:8], "big") / (2**64) * 40.0
    scale = base_index / raw_values[0]
    time_offset = 20_000 + int.from_bytes(time_digest[:4], "big") % 700_000
    values = [float(value * scale) for value in raw_values]
    times = [float(time_offset + index) for index in range(len(raw_values))]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("blinded values are not positive and finite")
    return times, values


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def _json_bytes(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def materialize_world_bank_campaign_v55(
    *,
    protocol: ProspectiveCampaignProtocolV55,
    selection_spec: WorldBankSelectionSpecV55,
    ode_thresholds: ODEThresholdsV52,
    selection_seed: bytes,
    private_target_key_id: str,
    private_target_key: bytes,
    source_provenance_key_id: str,
    source_provenance_key: bytes,
    custodian_host_id: str,
    coordinator_host_id: str,
    generator_host_id: str,
    custody_key_id: str,
    custody_private_key_pem: bytes,
    output_dir: Path,
    fetcher: FetcherV55 = _default_fetcher,
    retrieved_at: datetime | None = None,
) -> WorldBankCustodianSummaryV55:
    """Select, encrypt, and write only generator-safe public artifacts."""

    protocol.assert_sealed()
    selection_spec.assert_sealed()
    ode_thresholds.assert_sealed()
    if selection_spec.protocol_hash != protocol.protocol_hash:
        raise ValueError("selection spec is bound to another protocol")
    if selection_spec.ode_threshold_hash != ode_thresholds.threshold_hash:
        raise ValueError("selection spec is bound to other ODE thresholds")
    if len(selection_seed) != 32:
        raise ValueError("selection seed must contain exactly 32 bytes")
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")

    raw_values, country_name, indicator_name, source_url, raw_bytes = (
        _select_source(
            spec=selection_spec,
            selection_seed=selection_seed,
            fetcher=fetcher,
        )
    )
    times, transformed = _transform_series(
        raw_values=raw_values,
        selection_seed=selection_seed,
    )
    seed_commitment = hashlib.sha256(selection_seed).hexdigest()
    task_id = f"i34-wb-{seed_commitment[:20]}"
    public_count = selection_spec.public_observation_count
    public_times = times[:public_count]
    public_values = transformed[:public_count]
    private_times = times[public_count:]
    private_values = transformed[public_count:]

    policy, eligibility_contract, public_launch_binding = (
        materialize_public_launch_v55(
            protocol=protocol,
            task_id=task_id,
            eligibility_contract_id=f"{task_id}-eligibility",
            materialized_at=retrieved_at or _utc_now(),
        )
    )
    thresholds = ode_thresholds
    snapshot = ODETimeSeriesSnapshotV52.seal(
        task_id=task_id,
        time_unit="blinded_annual_index",
        state_unit="blinded_positive_relative_index",
        times=public_times,
        observations=public_values,
        source_id="custody-withheld-world-bank-v2",
        fixture_only=selection_spec.fixture_only,
    )
    targets = [
        ODEForecastTargetV53(
            target_id=f"target-h{index}",
            time=private_times[index - 1],
        )
        for index in range(1, 5)
    ]
    forecast_plan = ODEForecastPlanV53.seal(
        plan_id=f"{task_id}-forecast-plan",
        task_id=task_id,
        public_snapshot_hash=snapshot.snapshot_hash,
        threshold_hash=thresholds.threshold_hash,
        targets=targets,
        state_unit=snapshot.state_unit,
        time_unit=snapshot.time_unit,
        frozen_at=retrieved_at or _utc_now(),
    )
    scale = max(
        float(np.quantile(public_values, 0.75) - np.quantile(public_values, 0.25)),
        float(max(public_values) - min(public_values)) / 10.0,
        1e-9,
    )
    score_contract = PrivateScoreContractV53.seal(
        contract_id=f"{task_id}-private-score",
        case_id=task_id,
        protocol_hash=protocol.protocol_hash,
        public_case_hash=snapshot.snapshot_hash,
        forecast_plan_hash=forecast_plan.plan_hash,
        target_ids=[target.target_id for target in targets],
        quality_scale=scale,
        minimum_quality_score=selection_spec.private_minimum_quality_score,
        frozen_at=retrieved_at or _utc_now(),
    )
    source_provenance = SourceProvenanceDraftV55(
        case_id=task_id,
        source_authority="World Bank Indicators API v2",
        source_title=f"{indicator_name} - {country_name}",
        source_locator=source_url,
        table_or_series_id=f"{country_name} | {indicator_name}",
        public_period_start=str(selection_spec.public_start_year),
        public_period_end=str(selection_spec.public_end_year),
        private_period_start=str(selection_spec.public_end_year + 1),
        private_period_end=str(selection_spec.private_end_year),
        source_artifact_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        prior_campaign_exclusion_hashes=(
            selection_spec.prior_campaign_exclusion_hashes
        ),
        retrieved_at=retrieved_at or _utc_now(),
    )
    (
        _capsule,
        _provenance,
        private_target_envelope,
        source_provenance_envelope,
        split_attestation,
    ) = create_split_custody_envelopes_v55(
        protocol=protocol,
        score_contract=score_contract,
        private_targets=[
            PrivateTargetV50(
                target_id=f"target-h{index}",
                value=private_values[index - 1],
            )
            for index in range(1, 5)
        ],
        source_provenance=source_provenance,
        private_target_envelope_id=f"{task_id}-target-envelope",
        source_provenance_envelope_id=f"{task_id}-provenance-envelope",
        private_target_key_id=private_target_key_id,
        private_target_key=private_target_key,
        source_provenance_key_id=source_provenance_key_id,
        source_provenance_key=source_provenance_key,
        custodian_host_id=custodian_host_id,
        coordinator_host_id=coordinator_host_id,
        generator_host_id=generator_host_id,
        attestation_id=f"{task_id}-split-custody",
        custody_key_id=custody_key_id,
        custody_private_key_pem=custody_private_key_pem,
        attested_at=retrieved_at or _utc_now(),
    )
    task_packet = WorldBankPublicTaskPacketV55.seal(
        task_id=task_id,
        protocol_hash=protocol.protocol_hash,
        selection_spec_hash=selection_spec.selection_spec_hash,
        selection_seed_commitment=seed_commitment,
        public_observations=[
            WorldBankPublicObservationV55(
                observation_id=f"obs-{index:02d}",
                time=public_times[index - 1],
                value=public_values[index - 1],
            )
            for index in range(1, public_count + 1)
        ],
        targets=[
            WorldBankPublicTargetV55(
                target_id=f"target-h{index}",
                horizon_steps=index,
                time=private_times[index - 1],
            )
            for index in range(1, 5)
        ],
        fixture_only=selection_spec.fixture_only,
    )
    custody_private_key = serialization.load_pem_private_key(
        custody_private_key_pem,
        password=None,
    )
    if not isinstance(custody_private_key, Ed25519PrivateKey):
        raise TypeError("custody signing key must be Ed25519")
    custody_public_key_pem = custody_private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    output_dir.mkdir(parents=True)
    public_values_by_name: dict[str, bytes] = {
        "campaign_protocol_v55.json": _json_bytes(protocol),
        "candidate_selection_policy_v55.json": _json_bytes(policy),
        "custody_public_key.pem": custody_public_key_pem,
        "ode_forecast_plan_v53.json": _json_bytes(forecast_plan),
        "ode_thresholds_v52.json": _json_bytes(thresholds),
        "private_score_contract_v53.json": _json_bytes(score_contract),
        "private_target_envelope_v55.json": _json_bytes(private_target_envelope),
        "public_eligibility_contract_v54.json": _json_bytes(
            eligibility_contract
        ),
        "public_launch_binding_v55.json": _json_bytes(public_launch_binding),
        "public_snapshot_v52.json": _json_bytes(snapshot),
        "public_task_packet_v55.json": _json_bytes(task_packet),
        "source_provenance_envelope_v55.json": _json_bytes(
            source_provenance_envelope
        ),
        "split_custody_attestation_v55.json": _json_bytes(split_attestation),
        "world_bank_selection_spec_v55.json": _json_bytes(selection_spec),
    }
    for name, payload in sorted(public_values_by_name.items()):
        _write_new(output_dir / name, payload)
    manifest = WorldBankPublicManifestV55.seal(
        task_id=task_id,
        protocol_hash=protocol.protocol_hash,
        selection_spec_hash=selection_spec.selection_spec_hash,
        files=[
            WorldBankPublicArtifactV55(
                path=name,
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
            )
            for name, payload in sorted(public_values_by_name.items())
        ],
        fixture_only=selection_spec.fixture_only,
    )
    _write_new(output_dir / "public_manifest_v55.json", _json_bytes(manifest))
    return WorldBankCustodianSummaryV55(
        task_id=task_id,
        protocol_hash=protocol.protocol_hash,
        selection_spec_hash=selection_spec.selection_spec_hash,
        selection_seed_commitment=seed_commitment,
        public_task_packet_hash=task_packet.packet_hash,
        private_target_envelope_hash=private_target_envelope.envelope_hash,
        source_provenance_envelope_hash=(
            source_provenance_envelope.envelope_hash
        ),
        split_custody_attestation_hash=split_attestation.attestation_hash,
        public_manifest_hash=manifest.manifest_hash,
        fixture_only=selection_spec.fixture_only,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--selection-spec", required=True)
    parser.add_argument("--ode-thresholds", required=True)
    parser.add_argument("--selection-seed", required=True)
    parser.add_argument("--private-target-key-id", required=True)
    parser.add_argument("--private-target-key", required=True)
    parser.add_argument("--source-provenance-key-id", required=True)
    parser.add_argument("--source-provenance-key", required=True)
    parser.add_argument("--custodian-host-id", required=True)
    parser.add_argument("--coordinator-host-id", required=True)
    parser.add_argument("--generator-host-id", required=True)
    parser.add_argument("--custody-key-id", required=True)
    parser.add_argument("--custody-private-key", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    protocol = ProspectiveCampaignProtocolV55.model_validate_json(
        Path(args.protocol).read_text(encoding="utf-8")
    )
    selection_spec = WorldBankSelectionSpecV55.model_validate_json(
        Path(args.selection_spec).read_text(encoding="utf-8")
    )
    ode_thresholds = ODEThresholdsV52.model_validate_json(
        Path(args.ode_thresholds).read_text(encoding="utf-8")
    )
    summary = materialize_world_bank_campaign_v55(
        protocol=protocol,
        selection_spec=selection_spec,
        ode_thresholds=ode_thresholds,
        selection_seed=Path(args.selection_seed).read_bytes(),
        private_target_key_id=args.private_target_key_id,
        private_target_key=Path(args.private_target_key).read_bytes(),
        source_provenance_key_id=args.source_provenance_key_id,
        source_provenance_key=Path(args.source_provenance_key).read_bytes(),
        custodian_host_id=args.custodian_host_id,
        coordinator_host_id=args.coordinator_host_id,
        generator_host_id=args.generator_host_id,
        custody_key_id=args.custody_key_id,
        custody_private_key_pem=Path(args.custody_private_key).read_bytes(),
        output_dir=Path(args.output_dir).resolve(),
    )
    print(canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "WorldBankCustodianSummaryV55",
    "WorldBankPublicManifestV55",
    "WorldBankPublicTaskPacketV55",
    "WorldBankSelectionSpecV55",
    "materialize_world_bank_campaign_v55",
]
