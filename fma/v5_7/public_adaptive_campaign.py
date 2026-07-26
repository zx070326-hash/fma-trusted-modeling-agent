"""Code-owned public campaign runner for the V5.7 adaptive series graph."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator

from fma.hashing import canonical_json, sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5_2.ode_system import ODETimeSeriesSnapshotV52
from fma.v5_3.ode_forecast import ODEForecastTargetV53
from fma.v5_5.public_ode_campaign import verify_public_launch_v55
from fma.v5_6.hybrid_ode import (
    FAMILIES,
    RESIDUAL_MODES,
    HybridODEThresholdsV56,
    _estimate_residual_process,
    _fit_trend,
    _forecast_correction,
    _trend_predict,
)
from fma.v5_6.unseen_source import verify_unseen_world_bank_campaign_v56

from .adaptive_positive_series import (
    GROWTH_MODES,
    AdaptivePositiveSeriesBundleV57,
    AdaptiveReplayAuthorityV57,
    AdaptiveReplayReceiptV57,
    AdaptiveThresholdsV57,
    _estimate_growth_process,
    build_adaptive_positive_series_bundle_v57,
    run_authenticated_adaptive_replays_v57,
)


GateDecisionV57 = Literal["ELIGIBLE", "ABSTAIN"]
PredictionStatusV57 = Literal[
    "REGISTERED_FOR_PRIVATE_EVALUATION",
    "PROVISIONAL_ONLY",
]
PrivateEvaluationStatusV57 = Literal[
    "BLOCKED_EXTERNAL_HOST_NOT_RUN",
    "NOT_AUTHORIZED_NOT_RUN",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


class AdaptiveCampaignProtocolV57(StrictModel):
    schema_version: Literal["5.7-adaptive-campaign-protocol"] = (
        "5.7-adaptive-campaign-protocol"
    )
    protocol_id: Identifier
    v55_protocol_hash: Sha256
    source_registry_hash: Sha256
    primary_threshold_hash: Sha256
    adaptive_threshold_hash: Sha256
    primary_adapter_source_sha256: Sha256
    adaptive_adapter_source_sha256: Sha256
    unseen_source_adapter_source_sha256: Sha256
    unseen_source_core_source_sha256: Sha256
    world_bank_custodian_source_sha256: Sha256
    public_runner_source_sha256: Sha256
    primary_candidate_families: list[Identifier]
    primary_residual_modes: list[Identifier]
    recovery_growth_modes: list[Identifier]
    required_public_levels: list[Literal["L0", "L1", "L2", "L3", "L4"]]
    public_gate_rule: Literal[
        "real_nonfixture_and_all_adaptive_l0_l4_pass"
    ] = "real_nonfixture_and_all_adaptive_l0_l4_pass"
    prediction_rule: Literal[
        "full_public_refit_selected_branch_recursive_four_horizon"
    ] = "full_public_refit_selected_branch_recursive_four_horizon"
    maximum_private_evaluations: Literal[1] = 1
    private_evaluation_requires_separate_external_host: Literal[True] = True
    source_selected_after_adapter_freeze: Literal[True] = True
    thresholds_frozen_before_source_selection: Literal[True] = True
    post_result_threshold_change_allowed: Literal[False] = False
    post_result_candidate_change_allowed: Literal[False] = False
    frozen_at: datetime
    protocol_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_protocol(self) -> "AdaptiveCampaignProtocolV57":
        if self.primary_candidate_families != sorted(FAMILIES):
            raise ValueError("V5.7 primary candidate families differ")
        if self.primary_residual_modes != sorted(RESIDUAL_MODES):
            raise ValueError("V5.7 primary residual modes differ")
        if self.recovery_growth_modes != sorted(GROWTH_MODES):
            raise ValueError("V5.7 recovery growth modes differ")
        if self.required_public_levels != ["L0", "L1", "L2", "L3", "L4"]:
            raise ValueError("V5.7 campaign must require ordered L0-L4")
        if self.frozen_at.utcoffset() is None:
            raise ValueError("V5.7 campaign frozen_at must be timezone-aware")
        if self.protocol_hash and self.protocol_hash != self.content_hash():
            raise ValueError("V5.7 campaign protocol hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "protocol_hash")

    def assert_sealed(self) -> None:
        if not self.protocol_hash or self.protocol_hash != self.content_hash():
            raise ValueError("V5.7 campaign protocol is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "AdaptiveCampaignProtocolV57":
        data.setdefault("frozen_at", _utc_now())
        draft = cls(**data)
        payload = draft.model_dump(exclude={"protocol_hash"})
        payload["protocol_hash"] = draft.content_hash()
        return cls(**payload)


class AdaptiveForecastPlanV57(StrictModel):
    schema_version: Literal["5.7-adaptive-forecast-plan"] = (
        "5.7-adaptive-forecast-plan"
    )
    plan_id: Identifier
    task_id: Identifier
    campaign_protocol_hash: Sha256
    source_campaign_manifest_hash: Sha256
    public_snapshot_hash: Sha256
    primary_threshold_hash: Sha256
    adaptive_threshold_hash: Sha256
    targets: Annotated[list[ODEForecastTargetV53], Field(min_length=4, max_length=4)]
    state_unit: Identifier
    time_unit: Identifier
    frozen_before_public_model_run: Literal[True] = True
    private_target_values_accessed: Literal[False] = False
    frozen_at: datetime
    plan_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_plan(self) -> "AdaptiveForecastPlanV57":
        if [item.target_id for item in self.targets] != [
            "target-h1",
            "target-h2",
            "target-h3",
            "target-h4",
        ]:
            raise ValueError("V5.7 forecast target IDs differ")
        times = [item.time for item in self.targets]
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("V5.7 forecast target times must increase")
        if self.frozen_at.utcoffset() is None:
            raise ValueError("V5.7 forecast plan frozen_at must be timezone-aware")
        if self.plan_hash and self.plan_hash != self.content_hash():
            raise ValueError("V5.7 forecast plan hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "plan_hash")

    def assert_sealed(self) -> None:
        if not self.plan_hash or self.plan_hash != self.content_hash():
            raise ValueError("V5.7 forecast plan is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "AdaptiveForecastPlanV57":
        data.setdefault("frozen_at", _utc_now())
        draft = cls(**data)
        payload = draft.model_dump(exclude={"plan_hash"})
        payload["plan_hash"] = draft.content_hash()
        return cls(**payload)


class AdaptivePredictionPointV57(StrictModel):
    target_id: Identifier
    time: Annotated[float, Field(allow_inf_nan=False)]
    horizon_steps: Annotated[int, Field(ge=1, le=4)]
    value: Annotated[float, Field(gt=0, allow_inf_nan=False)]


class AdaptivePredictionArtifactV57(StrictModel):
    schema_version: Literal["5.7-adaptive-prediction-artifact"] = (
        "5.7-adaptive-prediction-artifact"
    )
    task_id: Identifier
    forecast_plan_hash: Sha256
    scientific_bundle_hash: Sha256
    selected_branch: Literal["hybrid_ode", "log_growth", "unresolved"]
    selected_model_id: Identifier
    full_refit_model_hash: Sha256
    predictions: Annotated[
        list[AdaptivePredictionPointV57],
        Field(min_length=4, max_length=4),
    ]
    status: PredictionStatusV57
    registered_by_code_owned_harness: bool
    diagnostic_fallback_used: bool
    private_holdout_accessed_before_artifact: Literal[False] = False
    source_provenance_plaintext_accessed: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    created_at: datetime
    prediction_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_prediction(self) -> "AdaptivePredictionArtifactV57":
        if [item.target_id for item in self.predictions] != [
            "target-h1",
            "target-h2",
            "target-h3",
            "target-h4",
        ]:
            raise ValueError("V5.7 predictions do not cover exact targets")
        if [item.horizon_steps for item in self.predictions] != [1, 2, 3, 4]:
            raise ValueError("V5.7 prediction horizons differ")
        expected_registered = (
            self.status == "REGISTERED_FOR_PRIVATE_EVALUATION"
        )
        if self.registered_by_code_owned_harness != expected_registered:
            raise ValueError("V5.7 prediction registration status differs")
        if self.diagnostic_fallback_used != (
            self.selected_branch == "unresolved"
        ):
            raise ValueError("V5.7 diagnostic fallback status differs")
        if self.created_at.utcoffset() is None:
            raise ValueError("V5.7 prediction created_at must be timezone-aware")
        if self.prediction_hash and self.prediction_hash != self.content_hash():
            raise ValueError("V5.7 prediction hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "prediction_hash")

    @classmethod
    def seal(cls, **data: object) -> "AdaptivePredictionArtifactV57":
        data.setdefault("created_at", _utc_now())
        draft = cls(**data)
        payload = draft.model_dump(exclude={"prediction_hash"})
        payload["prediction_hash"] = draft.content_hash()
        return cls(**payload)


class AdaptivePublicCampaignResultV57(StrictModel):
    schema_version: Literal["5.7-adaptive-public-campaign-result"] = (
        "5.7-adaptive-public-campaign-result"
    )
    task_id: Identifier
    campaign_protocol_hash: Sha256
    source_campaign_manifest_hash: Sha256
    source_selection_receipt_hash: Sha256
    forecast_plan_hash: Sha256
    scientific_bundle_hash: Sha256
    prediction_hash: Sha256
    selected_branch: Literal["hybrid_ode", "log_growth", "unresolved"]
    selected_model_id: Identifier
    recovery_triggered: bool
    fixture_only: bool
    public_level_statuses: dict[
        Literal["L0", "L1", "L2", "L3", "L4"],
        Literal["PASS", "FAIL", "NOT_RUN", "HUMAN"],
    ]
    public_scientific_acceptance: bool
    public_gate_decision: GateDecisionV57
    prediction_status: PredictionStatusV57
    private_evaluation_status: PrivateEvaluationStatusV57
    maximum_private_evaluations: Literal[1] = 1
    private_evaluations_consumed: Literal[0] = 0
    private_target_plaintext_accessed: Literal[False] = False
    private_target_key_accessed: Literal[False] = False
    source_provenance_plaintext_accessed: Literal[False] = False
    external_host_established: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    result_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_result(self) -> "AdaptivePublicCampaignResultV57":
        expected_acceptance = bool(
            not self.fixture_only
            and all(
                self.public_level_statuses.get(level) == "PASS"
                for level in ("L0", "L1", "L2", "L3", "L4")
            )
        )
        if self.public_scientific_acceptance != expected_acceptance:
            raise ValueError("V5.7 public acceptance differs")
        expected_gate = "ELIGIBLE" if expected_acceptance else "ABSTAIN"
        if self.public_gate_decision != expected_gate:
            raise ValueError("V5.7 public gate differs")
        expected_prediction = (
            "REGISTERED_FOR_PRIVATE_EVALUATION"
            if expected_gate == "ELIGIBLE"
            else "PROVISIONAL_ONLY"
        )
        expected_private = (
            "BLOCKED_EXTERNAL_HOST_NOT_RUN"
            if expected_gate == "ELIGIBLE"
            else "NOT_AUTHORIZED_NOT_RUN"
        )
        if (
            self.prediction_status != expected_prediction
            or self.private_evaluation_status != expected_private
        ):
            raise ValueError("V5.7 public downstream status differs")
        if self.result_hash and self.result_hash != self.content_hash():
            raise ValueError("V5.7 public result hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "result_hash")

    @classmethod
    def seal(cls, **data: object) -> "AdaptivePublicCampaignResultV57":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"result_hash"})
        payload["result_hash"] = draft.content_hash()
        return cls(**payload)


class AdaptiveResultArtifactV57(StrictModel):
    path: Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]+$")]
    sha256: Sha256
    size_bytes: Annotated[int, Field(ge=1)]


class AdaptiveResultManifestV57(StrictModel):
    schema_version: Literal["5.7-adaptive-result-manifest"] = (
        "5.7-adaptive-result-manifest"
    )
    source_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    runner_source_sha256: Sha256
    adaptive_adapter_source_sha256: Sha256
    task_id: Identifier
    result_hash: Sha256
    files: Annotated[list[AdaptiveResultArtifactV57], Field(min_length=8)]
    public_gate_decision: GateDecisionV57
    external_host_established: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "AdaptiveResultManifestV57":
        paths = [item.path for item in self.files]
        if paths != sorted(set(paths)):
            raise ValueError("V5.7 result manifest paths differ")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("V5.7 result manifest hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    @classmethod
    def seal(cls, **data: object) -> "AdaptiveResultManifestV57":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"manifest_hash"})
        payload["manifest_hash"] = draft.content_hash()
        return cls(**payload)


def load_adaptive_thresholds_v57(path: Path) -> AdaptiveThresholdsV57:
    payload = json.loads(path.read_text(encoding="utf-8"))
    thresholds = (
        AdaptiveThresholdsV57.model_validate(payload)
        if payload.get("threshold_hash")
        else AdaptiveThresholdsV57.seal(**payload)
    )
    thresholds.assert_sealed()
    return thresholds


def materialize_adaptive_forecast_plan_v57(
    *,
    unseen_campaign_dir: Path,
    protocol: AdaptiveCampaignProtocolV57,
    primary_thresholds: HybridODEThresholdsV56,
    adaptive_thresholds: AdaptiveThresholdsV57,
    frozen_at: datetime | None = None,
) -> AdaptiveForecastPlanV57:
    protocol.assert_sealed()
    primary_thresholds.assert_sealed()
    adaptive_thresholds.assert_sealed()
    unseen = verify_unseen_world_bank_campaign_v56(unseen_campaign_dir)
    launch = verify_public_launch_v55(unseen.inner_public_dir)
    if (
        protocol.v55_protocol_hash != launch.protocol.protocol_hash
        or protocol.source_registry_hash != unseen.registry.registry_hash
        or protocol.primary_threshold_hash != primary_thresholds.threshold_hash
        or protocol.adaptive_threshold_hash
        != adaptive_thresholds.threshold_hash
    ):
        raise ValueError("V5.7 forecast plan inputs differ from protocol")
    return AdaptiveForecastPlanV57.seal(
        plan_id=f"{launch.snapshot.task_id}-adaptive-plan",
        task_id=launch.snapshot.task_id,
        campaign_protocol_hash=protocol.protocol_hash,
        source_campaign_manifest_hash=unseen.manifest.manifest_hash,
        public_snapshot_hash=launch.snapshot.snapshot_hash,
        primary_threshold_hash=primary_thresholds.threshold_hash,
        adaptive_threshold_hash=adaptive_thresholds.threshold_hash,
        targets=[
            ODEForecastTargetV53(
                target_id=item.target_id,
                time=item.time,
            )
            for item in launch.task_packet.targets
        ],
        state_unit=launch.snapshot.state_unit,
        time_unit=launch.snapshot.time_unit,
        frozen_at=frozen_at or _utc_now(),
    )


def _target_steps(
    snapshot: ODETimeSeriesSnapshotV52,
    plan: AdaptiveForecastPlanV57,
) -> list[int]:
    times = np.asarray(snapshot.times, dtype=float)
    cadence = float(np.median(np.diff(times)))
    if not math.isfinite(cadence) or cadence <= 0:
        raise ValueError("V5.7 public cadence is invalid")
    steps: list[int] = []
    for target in plan.targets:
        raw = (float(target.time) - float(times[-1])) / cadence
        step = int(round(raw))
        if step < 1 or abs(raw - step) > 1e-9:
            raise ValueError("V5.7 target is off public cadence")
        steps.append(step)
    if steps != [1, 2, 3, 4]:
        raise ValueError("V5.7 target horizons differ")
    return steps


def _full_refit_predictions(
    *,
    bundle: AdaptivePositiveSeriesBundleV57,
    snapshot: ODETimeSeriesSnapshotV52,
    plan: AdaptiveForecastPlanV57,
    status: PredictionStatusV57,
    created_at: datetime,
) -> AdaptivePredictionArtifactV57:
    times = np.asarray(snapshot.times, dtype=float)
    values = np.asarray(snapshot.observations, dtype=float)
    steps = _target_steps(snapshot, plan)
    diagnostic = bundle.graph.selected_branch == "unresolved"
    branch = bundle.graph.selected_branch
    selected_model = bundle.graph.selected_model_id
    predictions: list[float] = []
    if branch == "hybrid_ode":
        selected = next(
            item
            for item in bundle.primary_bundle.candidates
            if item.candidate_id == selected_model
        )
        trend = _fit_trend(selected.family, times, values)
        residuals = values - _trend_predict(trend, times)
        process, _ = _estimate_residual_process(
            selected.residual_mode,
            residuals,
        )
        target_times = np.asarray(
            [item.time for item in plan.targets],
            dtype=float,
        )
        predictions = (
            _trend_predict(trend, target_times)
            + _forecast_correction(
                last_residual=float(residuals[-1]),
                phi=process.effective_phi,
                horizon_steps=np.asarray(steps),
            )
        ).tolist()
        model_hash = sha256_value(
            {
                "branch": branch,
                "selected_model_id": selected_model,
                "trend_fit_hash": trend.fit_hash,
                "residual_fit_hash": process.fit_hash,
            }
        )
    else:
        selected_growth = next(
            (
                item
                for item in bundle.growth_candidates
                if item.candidate_id == selected_model
            ),
            None,
        )
        if selected_growth is None:
            raise ValueError("V5.7 growth diagnostic candidate is absent")
        growths = np.diff(np.log(values))
        process, _ = _estimate_growth_process(
            selected_growth.mode,
            growths,
        )
        current_level = float(values[-1])
        current_growth = float(growths[-1])
        by_step: dict[int, float] = {}
        for step in range(1, max(steps) + 1):
            current_growth = float(
                process.mean_log_growth
                + process.effective_phi
                * (current_growth - process.mean_log_growth)
            )
            current_level *= math.exp(current_growth)
            by_step[step] = current_level
        predictions = [by_step[step] for step in steps]
        model_hash = sha256_value(
            {
                "branch": branch,
                "selected_model_id": selected_model,
                "growth_fit_hash": process.fit_hash,
            }
        )
    if any(
        not math.isfinite(value) or value <= 0 for value in predictions
    ):
        raise ValueError("V5.7 public prediction is not positive finite")
    return AdaptivePredictionArtifactV57.seal(
        task_id=snapshot.task_id,
        forecast_plan_hash=plan.plan_hash,
        scientific_bundle_hash=bundle.bundle_hash,
        selected_branch=branch,
        selected_model_id=selected_model,
        full_refit_model_hash=model_hash,
        predictions=[
            AdaptivePredictionPointV57(
                target_id=target.target_id,
                time=target.time,
                horizon_steps=step,
                value=float(value),
            )
            for target, step, value in zip(
                plan.targets,
                steps,
                predictions,
            )
        ],
        status=status,
        registered_by_code_owned_harness=(
            status == "REGISTERED_FOR_PRIVATE_EVALUATION"
        ),
        diagnostic_fallback_used=diagnostic,
        created_at=created_at,
    )


def _json_bytes(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _report(
    result: AdaptivePublicCampaignResultV57,
) -> str:
    return "\n".join(
        [
            "# V5.7 public adaptive positive-series result",
            "",
            f"- Public gate: `{result.public_gate_decision}`",
            f"- Selected branch: `{result.selected_branch}`",
            f"- Selected model: `{result.selected_model_id}`",
            f"- Recovery triggered: `{result.recovery_triggered}`",
            (
                "- L0-L4: `"
                + ", ".join(
                    f"{level}={status}"
                    for level, status in result.public_level_statuses.items()
                )
                + "`"
            ),
            f"- Prediction status: `{result.prediction_status}`",
            f"- Private evaluation: `{result.private_evaluation_status}`",
            "- Private evaluations consumed: `0/1`",
            "- External host established: `false`",
            "- Scientific qualification granted: `false`",
            "- Real-world action authorized: `false`",
            "",
        ]
    )


def run_public_adaptive_campaign_v57(
    *,
    unseen_campaign_dir: Path,
    protocol_path: Path,
    primary_threshold_path: Path,
    adaptive_threshold_path: Path,
    forecast_plan_path: Path,
    replay_secret_path: Path,
    output_dir: Path,
    replay_key_id: str = "adaptive-public-local-replay",
    created_at: datetime | None = None,
) -> AdaptivePublicCampaignResultV57:
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    protocol = AdaptiveCampaignProtocolV57.model_validate_json(
        protocol_path.read_text(encoding="utf-8")
    )
    primary_payload = json.loads(
        primary_threshold_path.read_text(encoding="utf-8")
    )
    primary = (
        HybridODEThresholdsV56.model_validate(primary_payload)
        if primary_payload.get("threshold_hash")
        else HybridODEThresholdsV56.seal(**primary_payload)
    )
    adaptive = load_adaptive_thresholds_v57(adaptive_threshold_path)
    plan = AdaptiveForecastPlanV57.model_validate_json(
        forecast_plan_path.read_text(encoding="utf-8")
    )
    protocol.assert_sealed()
    primary.assert_sealed()
    adaptive.assert_sealed()
    plan.assert_sealed()
    unseen = verify_unseen_world_bank_campaign_v56(unseen_campaign_dir)
    launch = verify_public_launch_v55(unseen.inner_public_dir)
    package = Path(__file__).resolve().parent
    if (
        protocol.v55_protocol_hash != launch.protocol.protocol_hash
        or protocol.source_registry_hash != unseen.registry.registry_hash
        or protocol.primary_threshold_hash != primary.threshold_hash
        or protocol.adaptive_threshold_hash != adaptive.threshold_hash
        or protocol.primary_adapter_source_sha256
        != hashlib.sha256(
            (package.parent / "v5_6" / "hybrid_ode.py").read_bytes()
        ).hexdigest()
        or protocol.adaptive_adapter_source_sha256
        != hashlib.sha256(
            (package / "adaptive_positive_series.py").read_bytes()
        ).hexdigest()
        or protocol.unseen_source_adapter_source_sha256
        != hashlib.sha256(
            (package / "unseen_source.py").read_bytes()
        ).hexdigest()
        or protocol.unseen_source_core_source_sha256
        != hashlib.sha256(
            (package.parent / "v5_6" / "unseen_source.py").read_bytes()
        ).hexdigest()
        or protocol.world_bank_custodian_source_sha256
        != hashlib.sha256(
            (
                package.parent
                / "v5_5"
                / "world_bank_custodian.py"
            ).read_bytes()
        ).hexdigest()
        or protocol.public_runner_source_sha256
        != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        or plan.task_id != launch.snapshot.task_id
        or plan.campaign_protocol_hash != protocol.protocol_hash
        or plan.source_campaign_manifest_hash != unseen.manifest.manifest_hash
        or plan.public_snapshot_hash != launch.snapshot.snapshot_hash
        or plan.primary_threshold_hash != primary.threshold_hash
        or plan.adaptive_threshold_hash != adaptive.threshold_hash
    ):
        raise ValueError("V5.7 public campaign bindings differ")
    final_output = output_dir.resolve()
    final_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{final_output.name}.tmp-",
            dir=final_output.parent,
        )
    )
    try:
        replay_input = temporary / "adaptive_replay_input_v57.json"
        _write_new(
            replay_input,
            _json_bytes(
                {
                    "snapshot": launch.snapshot.model_dump(mode="json"),
                    "primary_thresholds": primary.model_dump(mode="json"),
                    "adaptive_thresholds": adaptive.model_dump(mode="json"),
                }
            ),
        )
        secret = replay_secret_path.read_bytes()
        authority = AdaptiveReplayAuthorityV57(
            key_id=replay_key_id,
            secret=secret,
        )
        receipts = run_authenticated_adaptive_replays_v57(
            replay_input,
            authority=authority,
        )
        bundle = build_adaptive_positive_series_bundle_v57(
            snapshot=launch.snapshot,
            primary_thresholds=primary,
            adaptive_thresholds=adaptive,
            replay_receipts=receipts,
            replay_authority=authority,
        )
        public_acceptance = bool(
            not launch.snapshot.fixture_only
            and all(item.status == "PASS" for item in bundle.levels)
        )
        gate: GateDecisionV57 = (
            "ELIGIBLE" if public_acceptance else "ABSTAIN"
        )
        prediction_status: PredictionStatusV57 = (
            "REGISTERED_FOR_PRIVATE_EVALUATION"
            if gate == "ELIGIBLE"
            else "PROVISIONAL_ONLY"
        )
        prediction = _full_refit_predictions(
            bundle=bundle,
            snapshot=launch.snapshot,
            plan=plan,
            status=prediction_status,
            created_at=created_at or _utc_now(),
        )
        result = AdaptivePublicCampaignResultV57.seal(
            task_id=launch.snapshot.task_id,
            campaign_protocol_hash=protocol.protocol_hash,
            source_campaign_manifest_hash=unseen.manifest.manifest_hash,
            source_selection_receipt_hash=unseen.receipt.receipt_hash,
            forecast_plan_hash=plan.plan_hash,
            scientific_bundle_hash=bundle.bundle_hash,
            prediction_hash=prediction.prediction_hash,
            selected_branch=bundle.graph.selected_branch,
            selected_model_id=bundle.graph.selected_model_id,
            recovery_triggered=bundle.graph.recovery_triggered,
            fixture_only=launch.snapshot.fixture_only,
            public_level_statuses={
                item.level: item.status for item in bundle.levels
            },
            public_scientific_acceptance=public_acceptance,
            public_gate_decision=gate,
            prediction_status=prediction_status,
            private_evaluation_status=(
                "BLOCKED_EXTERNAL_HOST_NOT_RUN"
                if gate == "ELIGIBLE"
                else "NOT_AUTHORIZED_NOT_RUN"
            ),
        )
        source_verification = {
            "schema_version": "5.7-source-launch-verification",
            "source_campaign_manifest_hash": unseen.manifest.manifest_hash,
            "source_registry_hash": unseen.registry.registry_hash,
            "source_selection_receipt_hash": unseen.receipt.receipt_hash,
            "inner_public_manifest_hash": (
                unseen.manifest.inner_public_manifest_hash
            ),
            "exact_file_set_verified": True,
            "source_selection_signature_verified": True,
            "selected_source_not_prior_verified": True,
            "private_target_values_accessed": False,
            "source_identity_disclosed": False,
            "external_host_established": False,
            "scientific_qualification_granted": False,
        }
        source_verification["verification_hash"] = sha256_value(
            source_verification
        )
        artifacts: dict[str, bytes] = {
            "adaptive_campaign_protocol_v57.json": _json_bytes(protocol),
            "adaptive_forecast_plan_v57.json": _json_bytes(plan),
            "adaptive_positive_series_bundle_v57.json": _json_bytes(bundle),
            "adaptive_predictions_v57.json": _json_bytes(prediction),
            "adaptive_public_result_v57.json": _json_bytes(result),
            "adaptive_replay_input_v57.json": replay_input.read_bytes(),
            "adaptive_replay_receipts_v57.json": _json_bytes(receipts),
            "adaptive_thresholds_v57.json": _json_bytes(adaptive),
            "primary_thresholds_v56.json": _json_bytes(primary),
            "source_launch_verification_v57.json": _json_bytes(
                source_verification
            ),
            "REPORT.md": _report(result).encode("utf-8"),
        }
        for name, payload in sorted(artifacts.items()):
            if name == replay_input.name:
                continue
            _write_new(temporary / name, payload)
        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        manifest = AdaptiveResultManifestV57.seal(
            source_commit=source_commit,
            runner_source_sha256=hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            adaptive_adapter_source_sha256=hashlib.sha256(
                (package / "adaptive_positive_series.py").read_bytes()
            ).hexdigest(),
            task_id=result.task_id,
            result_hash=result.result_hash,
            files=[
                AdaptiveResultArtifactV57(
                    path=path.name,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    size_bytes=path.stat().st_size,
                )
                for path in sorted(
                    temporary.iterdir(),
                    key=lambda item: item.name,
                )
                if path.is_file()
            ],
            public_gate_decision=result.public_gate_decision,
        )
        _write_new(
            temporary / "result_manifest_v57.json",
            _json_bytes(manifest),
        )
        os.rename(temporary, final_output)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_public_adaptive_campaign_v57(
    *,
    unseen_campaign_dir: Path,
    output_dir: Path,
) -> AdaptivePublicCampaignResultV57:
    unseen = verify_unseen_world_bank_campaign_v56(unseen_campaign_dir)
    launch = verify_public_launch_v55(unseen.inner_public_dir)
    root = output_dir.resolve()
    manifest = AdaptiveResultManifestV57.model_validate_json(
        (root / "result_manifest_v57.json").read_text(encoding="utf-8")
    )
    if manifest.manifest_hash != manifest.content_hash():
        raise ValueError("V5.7 result manifest is unsealed")
    actual = {
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name != "result_manifest_v57.json"
    }
    declared = {item.path: item for item in manifest.files}
    if actual != set(declared):
        raise ValueError("V5.7 result exact file set differs")
    for name, entry in declared.items():
        payload = (root / name).read_bytes()
        if (
            len(payload) != entry.size_bytes
            or hashlib.sha256(payload).hexdigest() != entry.sha256
        ):
            raise ValueError(f"V5.7 result artifact differs: {name}")
    result = AdaptivePublicCampaignResultV57.model_validate_json(
        (root / "adaptive_public_result_v57.json").read_text(
            encoding="utf-8"
        )
    )
    bundle = AdaptivePositiveSeriesBundleV57.model_validate_json(
        (root / "adaptive_positive_series_bundle_v57.json").read_text(
            encoding="utf-8"
        )
    )
    prediction = AdaptivePredictionArtifactV57.model_validate_json(
        (root / "adaptive_predictions_v57.json").read_text(encoding="utf-8")
    )
    protocol = AdaptiveCampaignProtocolV57.model_validate_json(
        (root / "adaptive_campaign_protocol_v57.json").read_text(
            encoding="utf-8"
        )
    )
    plan = AdaptiveForecastPlanV57.model_validate_json(
        (root / "adaptive_forecast_plan_v57.json").read_text(
            encoding="utf-8"
        )
    )
    primary_payload = json.loads(
        (root / "primary_thresholds_v56.json").read_text(encoding="utf-8")
    )
    primary = HybridODEThresholdsV56.model_validate(primary_payload)
    adaptive = load_adaptive_thresholds_v57(
        root / "adaptive_thresholds_v57.json"
    )
    protocol.assert_sealed()
    plan.assert_sealed()
    primary.assert_sealed()
    adaptive.assert_sealed()
    receipts = [
        AdaptiveReplayReceiptV57.model_validate(item)
        for item in json.loads(
            (root / "adaptive_replay_receipts_v57.json").read_text(
                encoding="utf-8"
            )
        )
    ]
    replay_path = root / "adaptive_replay_input_v57.json"
    replay_input = json.loads(replay_path.read_text(encoding="utf-8"))
    expected_replay_input = {
        "snapshot": launch.snapshot.model_dump(mode="json"),
        "primary_thresholds": primary.model_dump(mode="json"),
        "adaptive_thresholds": adaptive.model_dump(mode="json"),
    }
    replay_semantic_hash = sha256_value(expected_replay_input)
    source_verification = json.loads(
        (root / "source_launch_verification_v57.json").read_text(
            encoding="utf-8"
        )
    )
    source_verification_hash = source_verification.pop("verification_hash")
    package = Path(__file__).resolve().parent
    primary_source_hash = hashlib.sha256(
        (package.parent / "v5_6" / "hybrid_ode.py").read_bytes()
    ).hexdigest()
    adaptive_source_hash = hashlib.sha256(
        (package / "adaptive_positive_series.py").read_bytes()
    ).hexdigest()
    unseen_source_hash = hashlib.sha256(
        (package / "unseen_source.py").read_bytes()
    ).hexdigest()
    unseen_source_core_hash = hashlib.sha256(
        (package.parent / "v5_6" / "unseen_source.py").read_bytes()
    ).hexdigest()
    world_bank_custodian_hash = hashlib.sha256(
        (
            package.parent / "v5_5" / "world_bank_custodian.py"
        ).read_bytes()
    ).hexdigest()
    runner_source_hash = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    recomputed_prediction = _full_refit_predictions(
        bundle=bundle,
        snapshot=launch.snapshot,
        plan=plan,
        status=prediction.status,
        created_at=prediction.created_at,
    )
    if (
        result.result_hash != result.content_hash()
        or bundle.bundle_hash != bundle.content_hash()
        or prediction.prediction_hash != prediction.content_hash()
        or recomputed_prediction.prediction_hash != prediction.prediction_hash
        or protocol.v55_protocol_hash != launch.protocol.protocol_hash
        or protocol.source_registry_hash != unseen.registry.registry_hash
        or protocol.primary_threshold_hash != primary.threshold_hash
        or protocol.adaptive_threshold_hash != adaptive.threshold_hash
        or protocol.primary_adapter_source_sha256 != primary_source_hash
        or protocol.adaptive_adapter_source_sha256 != adaptive_source_hash
        or protocol.unseen_source_adapter_source_sha256 != unseen_source_hash
        or protocol.unseen_source_core_source_sha256
        != unseen_source_core_hash
        or protocol.world_bank_custodian_source_sha256
        != world_bank_custodian_hash
        or protocol.public_runner_source_sha256 != runner_source_hash
        or protocol.protocol_hash != result.campaign_protocol_hash
        or plan.plan_hash != result.forecast_plan_hash
        or plan.task_id != launch.snapshot.task_id
        or plan.campaign_protocol_hash != protocol.protocol_hash
        or plan.source_campaign_manifest_hash != unseen.manifest.manifest_hash
        or plan.public_snapshot_hash != launch.snapshot.snapshot_hash
        or plan.primary_threshold_hash != primary.threshold_hash
        or plan.adaptive_threshold_hash != adaptive.threshold_hash
        or [
            (item.target_id, item.time)
            for item in plan.targets
        ]
        != [
            (item.target_id, item.time)
            for item in launch.task_packet.targets
        ]
        or result.task_id != launch.snapshot.task_id
        or result.task_id != prediction.task_id
        or result.source_campaign_manifest_hash != unseen.manifest.manifest_hash
        or result.source_selection_receipt_hash != unseen.receipt.receipt_hash
        or result.scientific_bundle_hash != bundle.bundle_hash
        or result.prediction_hash != prediction.prediction_hash
        or result.selected_branch != bundle.graph.selected_branch
        or result.selected_model_id != bundle.graph.selected_model_id
        or prediction.selected_branch != bundle.graph.selected_branch
        or prediction.selected_model_id != bundle.graph.selected_model_id
        or prediction.forecast_plan_hash != plan.plan_hash
        or prediction.scientific_bundle_hash != bundle.bundle_hash
        or prediction.status != result.prediction_status
        or result.fixture_only != launch.snapshot.fixture_only
        or bundle.task_id != launch.snapshot.task_id
        or bundle.snapshot_hash != launch.snapshot.snapshot_hash
        or bundle.primary_threshold_hash != primary.threshold_hash
        or bundle.adaptive_threshold_hash != adaptive.threshold_hash
        or bundle.fixture_only != launch.snapshot.fixture_only
        or result.public_level_statuses
        != {item.level: item.status for item in bundle.levels}
        or result.public_scientific_acceptance
        != bool(
            not launch.snapshot.fixture_only
            and bundle.scientific_acceptance
        )
        or replay_input != expected_replay_input
        or len({item.process_id for item in receipts}) != 2
        or len({item.deterministic_output_hash for item in receipts}) != 1
        or [item.receipt_hash for item in receipts]
        != bundle.replay_receipt_hashes
        or any(item.receipt_hash != item.content_hash() for item in receipts)
        or any(
            item.input_bytes_hash
            != hashlib.sha256(replay_path.read_bytes()).hexdigest()
            or item.input_semantic_hash != replay_semantic_hash
            for item in receipts
        )
        or source_verification_hash != sha256_value(source_verification)
        or source_verification
        != {
            "schema_version": "5.7-source-launch-verification",
            "source_campaign_manifest_hash": unseen.manifest.manifest_hash,
            "source_registry_hash": unseen.registry.registry_hash,
            "source_selection_receipt_hash": unseen.receipt.receipt_hash,
            "inner_public_manifest_hash": (
                unseen.manifest.inner_public_manifest_hash
            ),
            "exact_file_set_verified": True,
            "source_selection_signature_verified": True,
            "selected_source_not_prior_verified": True,
            "private_target_values_accessed": False,
            "source_identity_disclosed": False,
            "external_host_established": False,
            "scientific_qualification_granted": False,
        }
        or manifest.task_id != result.task_id
        or manifest.result_hash != result.result_hash
        or manifest.runner_source_sha256 != runner_source_hash
        or manifest.adaptive_adapter_source_sha256 != adaptive_source_hash
        or manifest.public_gate_decision != result.public_gate_decision
    ):
        raise ValueError("V5.7 public result cross-bindings differ")
    return result


__all__ = [
    "AdaptiveCampaignProtocolV57",
    "AdaptiveForecastPlanV57",
    "AdaptivePredictionArtifactV57",
    "AdaptivePredictionPointV57",
    "AdaptivePublicCampaignResultV57",
    "AdaptiveResultManifestV57",
    "load_adaptive_thresholds_v57",
    "materialize_adaptive_forecast_plan_v57",
    "run_public_adaptive_campaign_v57",
    "verify_public_adaptive_campaign_v57",
]
