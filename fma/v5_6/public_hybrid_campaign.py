"""Code-owned public I35 runner for the frozen V5.6 hybrid ODE adapter."""

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
from fma.v5_3.ode_forecast import ODEForecastTargetV53
from fma.v5_5.public_ode_campaign import verify_public_launch_v55

from .hybrid_ode import (
    FAMILIES,
    RESIDUAL_MODES,
    HybridODEThresholdsV56,
    HybridReplayAuthorityV56,
    HybridReplayReceiptV56,
    HybridScientificBundleV56,
    _estimate_residual_process,
    _fit_trend,
    _forecast_correction,
    _trend_predict,
    build_hybrid_ode_bundle_v56,
    run_authenticated_hybrid_replays_v56,
)
from .unseen_source import verify_unseen_world_bank_campaign_v56


GateDecisionV56 = Literal["ELIGIBLE", "ABSTAIN"]
PredictionStatusV56 = Literal[
    "REGISTERED_FOR_PRIVATE_EVALUATION",
    "PROVISIONAL_ONLY",
]
PrivateEvaluationStatusV56 = Literal[
    "BLOCKED_EXTERNAL_HOST_NOT_RUN",
    "NOT_AUTHORIZED_NOT_RUN",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


class HybridCampaignProtocolV56(StrictModel):
    schema_version: Literal["5.6-hybrid-campaign-protocol"] = (
        "5.6-hybrid-campaign-protocol"
    )
    protocol_id: Identifier
    v55_protocol_hash: Sha256
    source_registry_hash: Sha256
    hybrid_threshold_hash: Sha256
    hybrid_adapter_source_sha256: Sha256
    unseen_source_adapter_source_sha256: Sha256
    public_runner_source_sha256: Sha256
    candidate_families: list[Identifier]
    residual_modes: list[Identifier]
    required_public_levels: list[Literal["L0", "L1", "L2", "L3", "L4"]]
    public_gate_rule: Literal[
        "real_nonfixture_and_all_l0_l4_pass"
    ] = "real_nonfixture_and_all_l0_l4_pass"
    prediction_rule: Literal[
        "full_public_refit_selected_trend_plus_recursive_ar1_residual"
    ] = "full_public_refit_selected_trend_plus_recursive_ar1_residual"
    maximum_private_evaluations: Literal[1] = 1
    private_evaluation_requires_separate_external_host: Literal[True] = True
    source_selected_after_adapter_freeze: Literal[True] = True
    thresholds_frozen_before_source_selection: Literal[True] = True
    post_result_threshold_change_allowed: Literal[False] = False
    post_result_candidate_change_allowed: Literal[False] = False
    frozen_at: datetime
    protocol_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_protocol(self) -> "HybridCampaignProtocolV56":
        if self.candidate_families != sorted(FAMILIES):
            raise ValueError("hybrid campaign candidate families differ")
        if self.residual_modes != sorted(RESIDUAL_MODES):
            raise ValueError("hybrid campaign residual modes differ")
        if self.required_public_levels != ["L0", "L1", "L2", "L3", "L4"]:
            raise ValueError("hybrid campaign must require ordered L0-L4")
        if self.frozen_at.utcoffset() is None:
            raise ValueError("hybrid campaign frozen_at must be timezone-aware")
        if self.protocol_hash and self.protocol_hash != self.content_hash():
            raise ValueError("hybrid campaign protocol hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "protocol_hash")

    def assert_sealed(self) -> None:
        if not self.protocol_hash or self.protocol_hash != self.content_hash():
            raise ValueError("hybrid campaign protocol is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "HybridCampaignProtocolV56":
        data.setdefault("frozen_at", _utc_now())
        draft = cls(**data)
        payload = draft.model_dump(exclude={"protocol_hash"})
        payload["protocol_hash"] = draft.content_hash()
        return cls(**payload)


class HybridForecastPlanV56(StrictModel):
    schema_version: Literal["5.6-hybrid-forecast-plan"] = (
        "5.6-hybrid-forecast-plan"
    )
    plan_id: Identifier
    task_id: Identifier
    campaign_protocol_hash: Sha256
    source_campaign_manifest_hash: Sha256
    public_snapshot_hash: Sha256
    hybrid_threshold_hash: Sha256
    targets: Annotated[list[ODEForecastTargetV53], Field(min_length=4, max_length=4)]
    state_unit: Identifier
    time_unit: Identifier
    frozen_before_public_model_run: Literal[True] = True
    private_target_values_accessed: Literal[False] = False
    frozen_at: datetime
    plan_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_plan(self) -> "HybridForecastPlanV56":
        ids = [item.target_id for item in self.targets]
        times = [item.time for item in self.targets]
        if ids != ["target-h1", "target-h2", "target-h3", "target-h4"]:
            raise ValueError("hybrid forecast target IDs differ")
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("hybrid forecast target times must increase")
        if self.frozen_at.utcoffset() is None:
            raise ValueError("hybrid forecast plan frozen_at must be timezone-aware")
        if self.plan_hash and self.plan_hash != self.content_hash():
            raise ValueError("hybrid forecast plan hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "plan_hash")

    def assert_sealed(self) -> None:
        if not self.plan_hash or self.plan_hash != self.content_hash():
            raise ValueError("hybrid forecast plan is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "HybridForecastPlanV56":
        data.setdefault("frozen_at", _utc_now())
        draft = cls(**data)
        payload = draft.model_dump(exclude={"plan_hash"})
        payload["plan_hash"] = draft.content_hash()
        return cls(**payload)


class HybridPredictionPointV56(StrictModel):
    target_id: Identifier
    time: Annotated[float, Field(allow_inf_nan=False)]
    horizon_steps: Annotated[int, Field(ge=1, le=4)]
    value: Annotated[float, Field(gt=0, allow_inf_nan=False)]


class HybridPredictionArtifactV56(StrictModel):
    schema_version: Literal["5.6-hybrid-prediction-artifact"] = (
        "5.6-hybrid-prediction-artifact"
    )
    task_id: Identifier
    forecast_plan_hash: Sha256
    scientific_bundle_hash: Sha256
    selected_candidate_id: Identifier
    full_refit_trend_hash: Sha256
    full_refit_residual_process_hash: Sha256
    full_refit_last_residual: Annotated[float, Field(allow_inf_nan=False)]
    predictions: Annotated[
        list[HybridPredictionPointV56],
        Field(min_length=4, max_length=4),
    ]
    status: PredictionStatusV56
    registered_by_code_owned_harness: bool
    private_holdout_accessed_before_artifact: Literal[False] = False
    source_provenance_plaintext_accessed: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    created_at: datetime
    prediction_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_prediction(self) -> "HybridPredictionArtifactV56":
        ids = [item.target_id for item in self.predictions]
        horizons = [item.horizon_steps for item in self.predictions]
        if ids != ["target-h1", "target-h2", "target-h3", "target-h4"]:
            raise ValueError("hybrid predictions do not cover exact targets")
        if horizons != [1, 2, 3, 4]:
            raise ValueError("hybrid prediction horizons differ")
        expected_registered = (
            self.status == "REGISTERED_FOR_PRIVATE_EVALUATION"
        )
        if self.registered_by_code_owned_harness != expected_registered:
            raise ValueError("hybrid prediction registration status differs")
        if self.created_at.utcoffset() is None:
            raise ValueError("hybrid prediction created_at must be timezone-aware")
        if self.prediction_hash and self.prediction_hash != self.content_hash():
            raise ValueError("hybrid prediction hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "prediction_hash")

    @classmethod
    def seal(cls, **data: object) -> "HybridPredictionArtifactV56":
        data.setdefault("created_at", _utc_now())
        draft = cls(**data)
        payload = draft.model_dump(exclude={"prediction_hash"})
        payload["prediction_hash"] = draft.content_hash()
        return cls(**payload)


class HybridPublicCampaignResultV56(StrictModel):
    schema_version: Literal["5.6-hybrid-public-campaign-result"] = (
        "5.6-hybrid-public-campaign-result"
    )
    task_id: Identifier
    campaign_protocol_hash: Sha256
    source_campaign_manifest_hash: Sha256
    source_selection_receipt_hash: Sha256
    forecast_plan_hash: Sha256
    scientific_bundle_hash: Sha256
    prediction_hash: Sha256
    selected_candidate_id: Identifier
    recovery_triggered: bool
    fixture_only: bool
    public_level_statuses: dict[
        Literal["L0", "L1", "L2", "L3", "L4"],
        Literal["PASS", "FAIL", "NOT_RUN", "HUMAN"],
    ]
    public_scientific_acceptance: bool
    public_gate_decision: GateDecisionV56
    prediction_status: PredictionStatusV56
    private_evaluation_status: PrivateEvaluationStatusV56
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
    def validate_result(self) -> "HybridPublicCampaignResultV56":
        expected_acceptance = bool(
            not self.fixture_only
            and all(
                self.public_level_statuses.get(level) == "PASS"
                for level in ("L0", "L1", "L2", "L3", "L4")
            )
        )
        if self.public_scientific_acceptance != expected_acceptance:
            raise ValueError("hybrid public acceptance differs from L0-L4")
        expected_decision = (
            "ELIGIBLE" if expected_acceptance else "ABSTAIN"
        )
        if self.public_gate_decision != expected_decision:
            raise ValueError("hybrid public gate differs from acceptance")
        expected_prediction = (
            "REGISTERED_FOR_PRIVATE_EVALUATION"
            if expected_decision == "ELIGIBLE"
            else "PROVISIONAL_ONLY"
        )
        expected_private = (
            "BLOCKED_EXTERNAL_HOST_NOT_RUN"
            if expected_decision == "ELIGIBLE"
            else "NOT_AUTHORIZED_NOT_RUN"
        )
        if (
            self.prediction_status != expected_prediction
            or self.private_evaluation_status != expected_private
        ):
            raise ValueError("hybrid public downstream status differs")
        if self.result_hash and self.result_hash != self.content_hash():
            raise ValueError("hybrid public result hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "result_hash")

    @classmethod
    def seal(cls, **data: object) -> "HybridPublicCampaignResultV56":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"result_hash"})
        payload["result_hash"] = draft.content_hash()
        return cls(**payload)


class HybridResultArtifactV56(StrictModel):
    path: Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]+$")]
    sha256: Sha256
    size_bytes: Annotated[int, Field(ge=1)]


class HybridResultManifestV56(StrictModel):
    schema_version: Literal["5.6-hybrid-result-manifest"] = (
        "5.6-hybrid-result-manifest"
    )
    source_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    runner_source_sha256: Sha256
    hybrid_adapter_source_sha256: Sha256
    task_id: Identifier
    result_hash: Sha256
    files: Annotated[list[HybridResultArtifactV56], Field(min_length=8)]
    public_gate_decision: GateDecisionV56
    external_host_established: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "HybridResultManifestV56":
        paths = [item.path for item in self.files]
        if paths != sorted(set(paths)):
            raise ValueError("hybrid result manifest paths differ")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("hybrid result manifest hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    @classmethod
    def seal(cls, **data: object) -> "HybridResultManifestV56":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"manifest_hash"})
        payload["manifest_hash"] = draft.content_hash()
        return cls(**payload)


def load_hybrid_thresholds_v56(path: Path) -> HybridODEThresholdsV56:
    payload = json.loads(path.read_text(encoding="utf-8"))
    thresholds = (
        HybridODEThresholdsV56.model_validate(payload)
        if payload.get("threshold_hash")
        else HybridODEThresholdsV56.seal(**payload)
    )
    thresholds.assert_sealed()
    return thresholds


def materialize_hybrid_forecast_plan_v56(
    *,
    unseen_campaign_dir: Path,
    protocol: HybridCampaignProtocolV56,
    thresholds: HybridODEThresholdsV56,
    frozen_at: datetime | None = None,
) -> HybridForecastPlanV56:
    protocol.assert_sealed()
    thresholds.assert_sealed()
    unseen = verify_unseen_world_bank_campaign_v56(unseen_campaign_dir)
    launch = verify_public_launch_v55(unseen.inner_public_dir)
    if (
        protocol.v55_protocol_hash != launch.protocol.protocol_hash
        or protocol.source_registry_hash != unseen.registry.registry_hash
        or protocol.hybrid_threshold_hash != thresholds.threshold_hash
    ):
        raise ValueError("hybrid forecast plan inputs differ from protocol")
    return HybridForecastPlanV56.seal(
        plan_id=f"{launch.snapshot.task_id}-hybrid-plan",
        task_id=launch.snapshot.task_id,
        campaign_protocol_hash=protocol.protocol_hash,
        source_campaign_manifest_hash=unseen.manifest.manifest_hash,
        public_snapshot_hash=launch.snapshot.snapshot_hash,
        hybrid_threshold_hash=thresholds.threshold_hash,
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


def _full_public_refit_predictions(
    *,
    bundle: HybridScientificBundleV56,
    snapshot: object,
    plan: HybridForecastPlanV56,
    status: PredictionStatusV56,
    created_at: datetime,
) -> HybridPredictionArtifactV56:
    from fma.v5_2.ode_system import ODETimeSeriesSnapshotV52

    typed_snapshot = ODETimeSeriesSnapshotV52.model_validate(snapshot)
    selected = next(
        item
        for item in bundle.candidates
        if item.candidate_id == bundle.selected_candidate_id
    )
    times = np.asarray(typed_snapshot.times, dtype=float)
    values = np.asarray(typed_snapshot.observations, dtype=float)
    trend = _fit_trend(selected.family, times, values)
    fitted = _trend_predict(trend, times)
    residuals = values - fitted
    process, _innovations = _estimate_residual_process(
        selected.residual_mode,
        residuals,
    )
    cadence = float(np.median(np.diff(times)))
    if not math.isfinite(cadence) or cadence <= 0:
        raise ValueError("hybrid public cadence is invalid")
    predictions: list[HybridPredictionPointV56] = []
    for target in plan.targets:
        raw_steps = (float(target.time) - float(times[-1])) / cadence
        steps = int(round(raw_steps))
        if steps < 1 or abs(raw_steps - steps) > 1e-9:
            raise ValueError("hybrid target is off the public cadence")
        value = float(
            _trend_predict(
                trend,
                np.asarray([float(target.time)]),
            )[0]
            + _forecast_correction(
                last_residual=float(residuals[-1]),
                phi=process.effective_phi,
                horizon_steps=np.asarray([steps]),
            )[0]
        )
        if not math.isfinite(value) or value <= 0:
            raise ValueError("hybrid public prediction is not positive finite")
        predictions.append(
            HybridPredictionPointV56(
                target_id=target.target_id,
                time=target.time,
                horizon_steps=steps,
                value=value,
            )
        )
    return HybridPredictionArtifactV56.seal(
        task_id=typed_snapshot.task_id,
        forecast_plan_hash=plan.plan_hash,
        scientific_bundle_hash=bundle.bundle_hash,
        selected_candidate_id=bundle.selected_candidate_id,
        full_refit_trend_hash=trend.fit_hash,
        full_refit_residual_process_hash=process.fit_hash,
        full_refit_last_residual=float(residuals[-1]),
        predictions=predictions,
        status=status,
        registered_by_code_owned_harness=(
            status == "REGISTERED_FOR_PRIVATE_EVALUATION"
        ),
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
    *,
    result: HybridPublicCampaignResultV56,
    bundle: HybridScientificBundleV56,
) -> str:
    return "\n".join(
        [
            "# V5.6 I35 public hybrid ODE result",
            "",
            f"- Public gate: `{result.public_gate_decision}`",
            f"- Selected candidate: `{result.selected_candidate_id}`",
            f"- Recovery triggered: `{bundle.graph.recovery_triggered}`",
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


def run_public_hybrid_campaign_v56(
    *,
    unseen_campaign_dir: Path,
    protocol_path: Path,
    threshold_path: Path,
    forecast_plan_path: Path,
    replay_secret_path: Path,
    output_dir: Path,
    replay_key_id: str = "i35-local-public-hybrid-replay",
    created_at: datetime | None = None,
) -> HybridPublicCampaignResultV56:
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    protocol = HybridCampaignProtocolV56.model_validate_json(
        protocol_path.read_text(encoding="utf-8")
    )
    thresholds = load_hybrid_thresholds_v56(threshold_path)
    plan = HybridForecastPlanV56.model_validate_json(
        forecast_plan_path.read_text(encoding="utf-8")
    )
    protocol.assert_sealed()
    thresholds.assert_sealed()
    plan.assert_sealed()
    unseen = verify_unseen_world_bank_campaign_v56(unseen_campaign_dir)
    launch = verify_public_launch_v55(unseen.inner_public_dir)
    if (
        protocol.v55_protocol_hash != launch.protocol.protocol_hash
        or protocol.source_registry_hash != unseen.registry.registry_hash
        or protocol.hybrid_threshold_hash != thresholds.threshold_hash
        or protocol.hybrid_adapter_source_sha256
        != hashlib.sha256(
            Path(__file__).with_name("hybrid_ode.py").read_bytes()
        ).hexdigest()
        or protocol.unseen_source_adapter_source_sha256
        != hashlib.sha256(
            Path(__file__).with_name("unseen_source.py").read_bytes()
        ).hexdigest()
        or protocol.public_runner_source_sha256
        != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        or plan.task_id != launch.snapshot.task_id
        or plan.campaign_protocol_hash != protocol.protocol_hash
        or plan.source_campaign_manifest_hash != unseen.manifest.manifest_hash
        or plan.public_snapshot_hash != launch.snapshot.snapshot_hash
        or plan.hybrid_threshold_hash != thresholds.threshold_hash
    ):
        raise ValueError("hybrid public campaign bindings differ")

    final_output = output_dir.resolve()
    final_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{final_output.name}.tmp-",
            dir=final_output.parent,
        )
    )
    try:
        replay_input = temporary / "hybrid_replay_input_v56.json"
        _write_new(
            replay_input,
            _json_bytes(
                {
                    "snapshot": launch.snapshot.model_dump(mode="json"),
                    "thresholds": thresholds.model_dump(mode="json"),
                }
            ),
        )
        secret = replay_secret_path.read_bytes()
        if len(secret) < 32:
            raise ValueError("hybrid replay secret needs at least 32 bytes")
        authority = HybridReplayAuthorityV56(
            key_id=replay_key_id,
            secret=secret,
        )
        receipts = run_authenticated_hybrid_replays_v56(
            replay_input,
            authority=authority,
        )
        bundle = build_hybrid_ode_bundle_v56(
            snapshot=launch.snapshot,
            thresholds=thresholds,
            replay_receipts=receipts,
            replay_authority=authority,
        )
        public_acceptance = bool(
            not launch.snapshot.fixture_only
            and all(item.status == "PASS" for item in bundle.levels)
        )
        gate: GateDecisionV56 = (
            "ELIGIBLE" if public_acceptance else "ABSTAIN"
        )
        prediction_status: PredictionStatusV56 = (
            "REGISTERED_FOR_PRIVATE_EVALUATION"
            if gate == "ELIGIBLE"
            else "PROVISIONAL_ONLY"
        )
        timestamp = created_at or _utc_now()
        prediction = _full_public_refit_predictions(
            bundle=bundle,
            snapshot=launch.snapshot,
            plan=plan,
            status=prediction_status,
            created_at=timestamp,
        )
        level_statuses = {
            item.level: item.status for item in bundle.levels
        }
        result = HybridPublicCampaignResultV56.seal(
            task_id=launch.snapshot.task_id,
            campaign_protocol_hash=protocol.protocol_hash,
            source_campaign_manifest_hash=unseen.manifest.manifest_hash,
            source_selection_receipt_hash=unseen.receipt.receipt_hash,
            forecast_plan_hash=plan.plan_hash,
            scientific_bundle_hash=bundle.bundle_hash,
            prediction_hash=prediction.prediction_hash,
            selected_candidate_id=bundle.selected_candidate_id,
            recovery_triggered=bundle.graph.recovery_triggered,
            fixture_only=launch.snapshot.fixture_only,
            public_level_statuses=level_statuses,
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
            "schema_version": "5.6-source-launch-verification",
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
            "hybrid_campaign_protocol_v56.json": _json_bytes(protocol),
            "hybrid_forecast_plan_v56.json": _json_bytes(plan),
            "hybrid_ode_bundle_v56.json": _json_bytes(bundle),
            "hybrid_predictions_v56.json": _json_bytes(prediction),
            "hybrid_public_result_v56.json": _json_bytes(result),
            "hybrid_replay_input_v56.json": replay_input.read_bytes(),
            "hybrid_replay_receipts_v56.json": _json_bytes(receipts),
            "hybrid_thresholds_v56.json": _json_bytes(thresholds),
            "source_launch_verification_v56.json": _json_bytes(
                source_verification
            ),
            "REPORT.md": _report(result=result, bundle=bundle).encode("utf-8"),
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
        manifest = HybridResultManifestV56.seal(
            source_commit=source_commit,
            runner_source_sha256=hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            hybrid_adapter_source_sha256=hashlib.sha256(
                Path(__file__).with_name("hybrid_ode.py").read_bytes()
            ).hexdigest(),
            task_id=result.task_id,
            result_hash=result.result_hash,
            files=[
                HybridResultArtifactV56(
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
            temporary / "result_manifest_v56.json",
            _json_bytes(manifest),
        )
        os.rename(temporary, final_output)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_public_hybrid_campaign_v56(
    *,
    unseen_campaign_dir: Path,
    output_dir: Path,
) -> HybridPublicCampaignResultV56:
    unseen = verify_unseen_world_bank_campaign_v56(unseen_campaign_dir)
    launch = verify_public_launch_v55(unseen.inner_public_dir)
    root = output_dir.resolve()
    manifest = HybridResultManifestV56.model_validate_json(
        (root / "result_manifest_v56.json").read_text(encoding="utf-8")
    )
    if manifest.manifest_hash != manifest.content_hash():
        raise ValueError("hybrid result manifest is unsealed")
    actual = {
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name != "result_manifest_v56.json"
    }
    declared = {item.path: item for item in manifest.files}
    if actual != set(declared):
        raise ValueError("hybrid result exact file set differs")
    for name, entry in declared.items():
        payload = (root / name).read_bytes()
        if (
            len(payload) != entry.size_bytes
            or hashlib.sha256(payload).hexdigest() != entry.sha256
        ):
            raise ValueError(f"hybrid result artifact differs: {name}")
    result = HybridPublicCampaignResultV56.model_validate_json(
        (root / "hybrid_public_result_v56.json").read_text(encoding="utf-8")
    )
    bundle = HybridScientificBundleV56.model_validate_json(
        (root / "hybrid_ode_bundle_v56.json").read_text(encoding="utf-8")
    )
    prediction = HybridPredictionArtifactV56.model_validate_json(
        (root / "hybrid_predictions_v56.json").read_text(encoding="utf-8")
    )
    protocol = HybridCampaignProtocolV56.model_validate_json(
        (root / "hybrid_campaign_protocol_v56.json").read_text(
            encoding="utf-8"
        )
    )
    plan = HybridForecastPlanV56.model_validate_json(
        (root / "hybrid_forecast_plan_v56.json").read_text(encoding="utf-8")
    )
    thresholds = load_hybrid_thresholds_v56(
        root / "hybrid_thresholds_v56.json"
    )
    protocol.assert_sealed()
    plan.assert_sealed()
    replay_input_path = root / "hybrid_replay_input_v56.json"
    replay_input = json.loads(replay_input_path.read_text(encoding="utf-8"))
    replay_semantic_hash = sha256_value(
        {
            "snapshot": launch.snapshot.model_dump(mode="json"),
            "thresholds": thresholds.model_dump(mode="json"),
        }
    )
    source_verification = json.loads(
        (root / "source_launch_verification_v56.json").read_text(
            encoding="utf-8"
        )
    )
    source_verification_hash = source_verification.pop("verification_hash")
    receipts = [
        HybridReplayReceiptV56.model_validate(item)
        for item in json.loads(
            (root / "hybrid_replay_receipts_v56.json").read_text(
                encoding="utf-8"
            )
        )
    ]
    if (
        result.result_hash != result.content_hash()
        or bundle.bundle_hash != bundle.content_hash()
        or prediction.prediction_hash != prediction.content_hash()
        or manifest.task_id != result.task_id
        or manifest.result_hash != result.result_hash
        or result.source_campaign_manifest_hash != unseen.manifest.manifest_hash
        or result.source_selection_receipt_hash != unseen.receipt.receipt_hash
        or protocol.protocol_hash != result.campaign_protocol_hash
        or protocol.v55_protocol_hash != launch.protocol.protocol_hash
        or protocol.source_registry_hash != unseen.registry.registry_hash
        or protocol.hybrid_threshold_hash != thresholds.threshold_hash
        or plan.plan_hash != result.forecast_plan_hash
        or plan.task_id != launch.snapshot.task_id
        or plan.campaign_protocol_hash != protocol.protocol_hash
        or plan.source_campaign_manifest_hash != unseen.manifest.manifest_hash
        or plan.public_snapshot_hash != launch.snapshot.snapshot_hash
        or plan.hybrid_threshold_hash != thresholds.threshold_hash
        or [
            (item.target_id, item.time)
            for item in plan.targets
        ]
        != [
            (item.target_id, item.time)
            for item in launch.task_packet.targets
        ]
        or result.scientific_bundle_hash != bundle.bundle_hash
        or result.prediction_hash != prediction.prediction_hash
        or result.selected_candidate_id != bundle.selected_candidate_id
        or result.prediction_status != prediction.status
        or result.task_id != prediction.task_id
        or prediction.forecast_plan_hash != plan.plan_hash
        or prediction.scientific_bundle_hash != bundle.bundle_hash
        or prediction.selected_candidate_id != bundle.selected_candidate_id
        or result.fixture_only != launch.snapshot.fixture_only
        or result.public_level_statuses
        != {item.level: item.status for item in bundle.levels}
        or result.public_scientific_acceptance
        != bool(
            not launch.snapshot.fixture_only
            and bundle.scientific_acceptance
        )
        or replay_input
        != {
            "snapshot": launch.snapshot.model_dump(mode="json"),
            "thresholds": thresholds.model_dump(mode="json"),
        }
        or len({item.process_id for item in receipts}) != 2
        or len({item.deterministic_output_hash for item in receipts}) != 1
        or [item.receipt_hash for item in receipts]
        != bundle.replay_receipt_hashes
        or any(item.receipt_hash != item.content_hash() for item in receipts)
        or any(
            item.input_bytes_hash
            != hashlib.sha256(replay_input_path.read_bytes()).hexdigest()
            or item.input_semantic_hash != replay_semantic_hash
            for item in receipts
        )
        or source_verification_hash != sha256_value(source_verification)
        or source_verification["source_campaign_manifest_hash"]
        != unseen.manifest.manifest_hash
        or source_verification["source_selection_receipt_hash"]
        != unseen.receipt.receipt_hash
    ):
        raise ValueError("hybrid public result cross-bindings differ")
    return result


__all__ = [
    "HybridCampaignProtocolV56",
    "HybridForecastPlanV56",
    "HybridPredictionArtifactV56",
    "HybridPredictionPointV56",
    "HybridPublicCampaignResultV56",
    "HybridResultManifestV56",
    "load_hybrid_thresholds_v56",
    "materialize_hybrid_forecast_plan_v56",
    "run_public_hybrid_campaign_v56",
    "verify_public_hybrid_campaign_v56",
]
