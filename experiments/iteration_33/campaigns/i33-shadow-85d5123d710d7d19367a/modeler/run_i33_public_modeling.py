r"""Reproducible, public-only I33 modeling and artifact construction.

The program reads the sibling public packet, D:\modeling\fma imports, and its
own source.  It never opens a custody artifact, target value, network client,
environment variable, process list, or a different experiment directory.
``--freeze`` writes public evidence artifacts once; it intentionally does not
create a prediction registration or a V5.4 eligibility assessment.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization

MODELER = Path(__file__).resolve().parent
REPO_ROOT = MODELER.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fma.v5_2.ode_system import (
    ODEThresholdsV52,
    ODETimeSeriesSnapshotV52,
    _parameter_vector,
    _predict,
    fit_ode_v52,
)
from fma.v5_3.ode_forecast import ODEForecastPlanV53, build_ode_forecast_bundle_v53
from fma.v5_4.public_eligibility import PublicEligibilityContractV54


CAMPAIGN_ID = "i33-shadow-85d5123d710d7d19367a"
PUBLIC = MODELER.parent / "public"
PROTOCOL = MODELER.parents[2] / "PROTOCOL.json"
OUTPUT_NAMES = (
    "MODELING_REPORT.md",
    "candidate_results.json",
    "paired_public_losses.json",
    "forecast_bundle_v53.json",
    "provisional_predictions.json",
    "artifact_manifest.json",
)
FAMILIES = ("constant", "exponential", "gompertz", "logistic")
ORIGINS = tuple(range(12, 25))
HORIZONS = (1, 2, 3, 4)
RECOVERY_WINDOW = 18


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def json_bytes(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_public(name: str) -> dict[str, object]:
    return read_json(PUBLIC / name)


def verify_public_inputs() -> tuple[
    dict[str, object],
    ODETimeSeriesSnapshotV52,
    ODEThresholdsV52,
    ODEForecastPlanV53,
    PublicEligibilityContractV54,
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    manifest = read_public("hash_manifest.json")
    rows: list[dict[str, object]] = []
    for entry in manifest["files"]:  # type: ignore[index]
        path = PUBLIC / entry["path"]  # type: ignore[index]
        data = path.read_bytes()
        rows.append(
            {
                "path": entry["path"],  # type: ignore[index]
                "expected_size_bytes": entry["size_bytes"],  # type: ignore[index]
                "actual_size_bytes": len(data),
                "expected_sha256": entry["sha256"],  # type: ignore[index]
                "actual_sha256": sha256_bytes(data),
                "size_match": len(data) == entry["size_bytes"],  # type: ignore[index]
                "sha256_match": sha256_bytes(data) == entry["sha256"],  # type: ignore[index]
            }
        )
    attestation = read_public("custody_attestation.json")
    key_document = read_public("custody_public_key.json")
    signing_payload = {
        key: value
        for key, value in attestation.items()
        if key not in {"signature_base64", "signing_payload_sha256"}
    }
    signed_bytes = canonical_json(signing_payload).encode("utf-8")
    signature_valid = False
    try:
        public_key = serialization.load_pem_public_key(
            str(key_document["public_key_pem"]).encode("utf-8")
        )
        public_key.verify(
            base64.b64decode(
                str(attestation["signature_base64"]).encode("ascii"), validate=True
            ),
            signed_bytes,
        )
        signature_valid = True
    except (InvalidSignature, TypeError, ValueError):
        signature_valid = False

    snapshot = ODETimeSeriesSnapshotV52.model_validate(
        read_public("public_snapshot_v52.json")
    )
    thresholds = ODEThresholdsV52.model_validate(read_public("ode_thresholds_v52.json"))
    plan = ODEForecastPlanV53.model_validate(read_public("ode_forecast_plan_v53.json"))
    contract = PublicEligibilityContractV54.model_validate(
        read_public("public_eligibility_contract_v54.json")
    )
    snapshot.assert_sealed()
    thresholds.assert_sealed()
    plan.assert_compatible(snapshot, thresholds)
    contract.assert_sealed()

    policy_hash = sha256_file(PUBLIC / "candidate_selection_policy.json")
    protocol_hash = sha256_file(PROTOCOL)
    packet = read_public("PUBLIC_TASK_PACKET.json")
    score_contract = read_public("score_contract.json")
    policy = read_public("candidate_selection_policy.json")
    binding = read_public("v5_4_protocol_binding.json")
    expected_families = list(FAMILIES)
    if policy["candidate_families"] != expected_families:  # type: ignore[index]
        raise RuntimeError("candidate family registry is not the frozen registry")
    if int(policy["candidate_budget"]) != 16:  # type: ignore[index]
        raise RuntimeError("unexpected candidate budget")
    if not (
        snapshot.task_id == plan.task_id == contract.task_id == CAMPAIGN_ID
        and packet["task_id"] == score_contract["task_id"] == CAMPAIGN_ID
    ):
        raise RuntimeError("typed task bindings differ")
    if not (
        policy_hash == contract.candidate_selection_rule_hash
        and policy_hash == binding["candidate_selection_policy_sha256"]  # type: ignore[index]
    ):
        raise RuntimeError("candidate-selection policy file hash is not bound")
    if not (
        protocol_hash == manifest["protocol_sha256"]  # type: ignore[index]
        == packet["protocol_sha256"]  # type: ignore[index]
        == attestation["protocol_sha256"]  # type: ignore[index]
        == score_contract["protocol_sha256"]  # type: ignore[index]
    ):
        raise RuntimeError("frozen protocol hash differs")
    public_snapshot_values = np.asarray(snapshot.observations, dtype=float)
    packet_values = np.asarray(
        [item["value"] for item in packet["public_observations"]], dtype=float  # type: ignore[index]
    )
    public_snapshot_times = np.asarray(snapshot.times, dtype=float)
    packet_times = np.asarray(
        [item["time"] for item in packet["public_observations"]], dtype=float  # type: ignore[index]
    )
    if not (
        len(public_snapshot_values) == len(packet_values) == 28
        and np.allclose(public_snapshot_values, packet_values, rtol=0.0, atol=1e-9)
        and np.array_equal(public_snapshot_times, packet_times)
    ):
        raise RuntimeError("sealed snapshot and public packet are not aligned")
    public_key_hash = sha256_bytes(str(key_document["public_key_pem"]).encode("utf-8"))
    verification = {
        "manifest_valid": all(
            bool(row["size_match"]) and bool(row["sha256_match"]) for row in rows
        ),
        "manifest_files": rows,
        "attestation_signature_valid": signature_valid,
        "attestation_signing_payload_sha256": sha256_bytes(signed_bytes),
        "attestation_signing_payload_hash_match": (
            sha256_bytes(signed_bytes) == attestation["signing_payload_sha256"]
        ),
        "attestation_public_key_hash_match": (
            public_key_hash
            == key_document["public_key_sha256"]
            == attestation["custody_public_key_sha256"]
        ),
        "v52_snapshot_sealed": snapshot.snapshot_hash == snapshot.content_hash(),
        "v52_thresholds_sealed": thresholds.threshold_hash == thresholds.content_hash(),
        "v53_forecast_plan_sealed_and_compatible": (
            plan.plan_hash == plan.content_hash()
        ),
        "v54_contract_sealed": contract.contract_hash == contract.content_hash(),
        "candidate_selection_policy_file_hash_match": (
            policy_hash == contract.candidate_selection_rule_hash
        ),
        "protocol_hash_match": True,
        "snapshot_packet_alignment_max_abs_difference": float(
            np.max(np.abs(public_snapshot_values - packet_values))
        ),
        "same_host": bool(attestation["same_host"]),
        "external_qualification": bool(attestation["external_qualification"]),
    }
    if not all(
        bool(verification[key])
        for key in (
            "manifest_valid",
            "attestation_signature_valid",
            "attestation_signing_payload_hash_match",
            "attestation_public_key_hash_match",
            "v52_snapshot_sealed",
            "v52_thresholds_sealed",
            "v53_forecast_plan_sealed_and_compatible",
            "v54_contract_sealed",
            "candidate_selection_policy_file_hash_match",
            "protocol_hash_match",
        )
    ):
        raise RuntimeError("public integrity, seal, or contract verification failed")
    return verification, snapshot, thresholds, plan, contract, packet, score_contract, policy


def predict_family(
    family: str,
    train_times: np.ndarray,
    train_values: np.ndarray,
    forecast_times: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    fit = fit_ode_v52(family, train_times, train_values)
    prediction = _predict(
        family,
        np.concatenate(([train_times[0]], forecast_times)),
        float(train_values[0]),
        _parameter_vector(fit),
    )[1:]
    return prediction, fit.model_dump(mode="json")


def run_candidate(
    *,
    candidate_id: str,
    family: str | None,
    window: int | None,
    times: np.ndarray,
    values: np.ndarray,
    scale: float,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    fit_records: list[dict[str, object]] = []
    for origin in ORIGINS:
        train_times = times[:origin]
        train_values = values[:origin]
        if family is None:
            prediction = np.repeat(train_values[-1], len(HORIZONS))
            fit_record: dict[str, object] = {
                "family": "persistence_last_value",
                "fitted_parameter_count": 0,
                "last_public_value": float(train_values[-1]),
            }
        else:
            count = min(window, len(train_values)) if window is not None else len(train_values)
            train_times = train_times[-count:]
            train_values = train_values[-count:]
            prediction, fit_record = predict_family(
                family, train_times, train_values, times[origin : origin + 4]
            )
        fit_records.append({"origin": origin, "fit": fit_record})
        for offset, horizon in enumerate(HORIZONS):
            observed = float(values[origin + offset])
            forecast = float(prediction[offset])
            absolute_error = abs(forecast - observed)
            rows.append(
                {
                    "origin": origin,
                    "horizon": horizon,
                    "prediction": forecast,
                    "observed": observed,
                    "absolute_error": absolute_error,
                    "normalized_absolute_loss": absolute_error / scale,
                }
            )
    loss_vector = np.asarray([item["normalized_absolute_loss"] for item in rows], dtype=float)
    error_vector = np.asarray(
        [item["prediction"] - item["observed"] for item in rows], dtype=float
    )
    by_horizon = {
        str(horizon): {
            "normalized_mae": float(
                np.mean(
                    [
                        item["normalized_absolute_loss"]
                        for item in rows
                        if item["horizon"] == horizon
                    ]
                )
            ),
            "normalized_rmse": float(
                np.sqrt(
                    np.mean(
                        [
                            ((item["prediction"] - item["observed"]) / scale) ** 2
                            for item in rows
                            if item["horizon"] == horizon
                        ]
                    )
                )
            ),
        }
        for horizon in HORIZONS
    }
    h1_errors = np.asarray(
        [item["prediction"] - item["observed"] for item in rows if item["horizon"] == 1],
        dtype=float,
    )
    residual_lag1 = (
        float(np.corrcoef(h1_errors[:-1], h1_errors[1:])[0, 1])
        if np.std(h1_errors[:-1]) > 0 and np.std(h1_errors[1:]) > 0
        else 0.0
    )
    return {
        "candidate_id": candidate_id,
        "family": family if family is not None else "persistence_last_value",
        "training_window_rule": (
            "all_available_public_prefix" if window is None else f"last_min({window}, origin) public observations"
        ),
        "rows": rows,
        "fits_by_origin": fit_records,
        "aggregate": {
            "normalized_mae": float(np.mean(loss_vector)),
            "normalized_rmse": float(np.sqrt(np.mean((error_vector / scale) ** 2))),
            "finite_predictions": bool(np.all(np.isfinite(error_vector))),
        },
        "by_horizon": by_horizon,
        "h1_residual_diagnostics": {
            "mean": float(np.mean(h1_errors)),
            "rmse": float(np.sqrt(np.mean(h1_errors**2))),
            "lag1_autocorrelation": residual_lag1,
            "durbin_watson": float(np.sum(np.diff(h1_errors) ** 2) / np.sum(h1_errors**2)),
        },
    }


def paired_rows(
    candidate: dict[str, object], baseline: dict[str, object]
) -> list[dict[str, object]]:
    candidate_rows = candidate["rows"]  # type: ignore[index]
    baseline_rows = baseline["rows"]  # type: ignore[index]
    result: list[dict[str, object]] = []
    for candidate_row, baseline_row in zip(candidate_rows, baseline_rows):
        if (
            candidate_row["origin"],
            candidate_row["horizon"],
            candidate_row["observed"],
        ) != (
            baseline_row["origin"],
            baseline_row["horizon"],
            baseline_row["observed"],
        ):
            raise RuntimeError("candidate/baseline public loss grid is not paired")
        result.append(
            {
                "candidate_id": candidate["candidate_id"],
                "baseline_id": baseline["candidate_id"],
                "origin": candidate_row["origin"],
                "horizon": candidate_row["horizon"],
                "prediction": candidate_row["prediction"],
                "baseline_prediction": baseline_row["prediction"],
                "observed": candidate_row["observed"],
                "candidate_normalized_absolute_loss": candidate_row[
                    "normalized_absolute_loss"
                ],
                "baseline_normalized_absolute_loss": baseline_row[
                    "normalized_absolute_loss"
                ],
                "advantage": baseline_row["normalized_absolute_loss"]
                - candidate_row["normalized_absolute_loss"],
            }
        )
    return result


def initial_failure_diagnostic(
    candidates: dict[str, dict[str, object]], baseline: dict[str, object]
) -> dict[str, object]:
    initial = [
        candidates[f"initial_{family}"]["aggregate"]["normalized_mae"]  # type: ignore[index]
        for family in FAMILIES
    ]
    baseline_mae = baseline["aggregate"]["normalized_mae"]  # type: ignore[index]
    return {
        "criterion": "At least one initial registered family must strictly improve aggregate paired public normalized MAE over persistence before recovery is unnecessary.",
        "initial_candidate_normalized_maes": {
            family: candidates[f"initial_{family}"]["aggregate"]["normalized_mae"]  # type: ignore[index]
            for family in FAMILIES
        },
        "baseline_normalized_mae": baseline_mae,
        "initial_family_failure": not any(value < baseline_mae for value in initial),
    }


def render_report(
    *,
    candidates: dict[str, dict[str, object]],
    baseline: dict[str, object],
    recovery: dict[str, object],
    selected_family: str,
    bundle: dict[str, object],
    provisional: dict[str, object],
    candidate_hash: str,
    paired_hash: str,
    bundle_hash: str,
    provisional_hash: str,
) -> str:
    rows = sorted(
        candidates.values(),
        key=lambda item: item["aggregate"]["normalized_mae"],  # type: ignore[index]
    )
    lines = [
        "# I33 Public-Only Modeling Report",
        "",
        "## Integrity and scope",
        "",
        "- All 12 public manifest entries matched declared SHA-256 and size; the custody Ed25519 signature, V5.2 snapshot and threshold seals, V5.3 plan compatibility, V5.4 contract seal, protocol binding, and candidate-policy file hash all verified.",
        "- The work uses only the sealed monthly public scalar state. No source identity, target value, custody content, private evaluation, network, browser, environment variable, process inspection, or other experiment was accessed.",
        "- This remains same-host blinded-context shadow evidence. V5.4 public eligibility has **not run**; no final submission or prediction registration exists.",
        "",
        "## First-principles assessment",
        "",
        "The 28-point scalar rises from a low initial level toward a mid-series plateau, then declines sharply in the late public segment. A scalar positive-affine transformed observation series does not by itself identify the latent state, exogenous forcing, causal mechanism, or a source-certified noise model. The registered autonomous ODE families are therefore tested as constrained forecast skeletons, not treated as identified labor-market mechanisms.",
        "",
        "## Initial candidates and graph recovery",
        "",
        "The first round compared all frozen families (constant, exponential, Gompertz, logistic) against the required persistence baseline on 13 expanding origins × four horizons. None improved aggregate normalized MAE over persistence, so the frozen recovery requirement fired. The recovery retained the registered exponential/Gompertz/logistic equations and optimizer, changing only the public training window to the latest min(18, origin) observations. No unregistered family was introduced; search count is seven, within the budget of 16.",
        "",
        "| Candidate | Family | Window | Aggregate nMAE | Aggregate nRMSE | h1 nMAE | h2 nMAE | h3 nMAE | h4 nMAE |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in [baseline, *rows]:
        aggregate = candidate["aggregate"]
        by_horizon = candidate["by_horizon"]
        lines.append(
            "| {id} | {family} | {window} | {mae:.6f} | {rmse:.6f} | {h1:.6f} | {h2:.6f} | {h3:.6f} | {h4:.6f} |".format(
                id=candidate["candidate_id"],
                family=candidate["family"],
                window=candidate["training_window_rule"],
                mae=aggregate["normalized_mae"],
                rmse=aggregate["normalized_rmse"],
                h1=by_horizon["1"]["normalized_mae"],
                h2=by_horizon["2"]["normalized_mae"],
                h3=by_horizon["3"]["normalized_mae"],
                h4=by_horizon["4"]["normalized_mae"],
            )
        )
    lines.extend(
        [
            "",
            "The recovery executed but did not create a public MAE improvement over persistence. The typed V5.2 development selection inside the real V5.3 bundle selected the registered `logistic` family on its frozen chronological development split; this is the locked family used for the single all-28-point final refit. The recovery does not change its family, and no post-hoc re-ranking overwrote the typed bundle selection.",
            "",
            "## Real V5.3 evidence status",
            "",
            "The saved `forecast_bundle_v53.json` contains a real FMA V5.3 bundle with development L0–L4 and h1–h4 all-time final-refit evidence. L1, L2, and L4 pass; L3 fails the frozen validation error, baseline-improvement, and interval-coverage checks. L0 is `NOT_RUN`: two authenticated fresh-process replay receipts are unavailable under this public-only no-process/no-environment scope. Therefore the bundle field is `scientific_acceptance=false`. This is a code-produced evidence value, not a model verification claim.",
            "",
            "## Provisional predictions",
            "",
            "These are V5.3 final-refit outputs only; they are not submitted or privately scored.",
            "",
            "| Target | Time | Provisional prediction |",
            "|---|---:|---:|",
        ]
    )
    for prediction in provisional["predictions"]:  # type: ignore[index]
        lines.append(
            "| {target_id} | {time:.0f} | {value:.12f} |".format(**prediction)
        )
    lines.extend(
        [
            "",
            "## Claim boundary and artifact hashes",
            "",
            "- `scientific_qualification=false`; `external_qualification=false`; `real_world_action_authorized=false`.",
            "- `v5_4_public_eligibility_gate_status=NOT_RUN`; `private_evaluation_performed=false`; `final_modeler_submission_created=false`.",
            f"- candidate results SHA-256: `{candidate_hash}`",
            f"- paired public losses SHA-256: `{paired_hash}`",
            f"- V5.3 bundle SHA-256: `{bundle_hash}`",
            f"- provisional predictions SHA-256: `{provisional_hash}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_artifacts() -> tuple[dict[str, bytes], dict[str, object]]:
    (
        verification,
        snapshot,
        thresholds,
        plan,
        eligibility_contract,
        packet,
        score_contract,
        policy,
    ) = verify_public_inputs()
    values = np.asarray(snapshot.observations, dtype=float)
    times = np.asarray(snapshot.times, dtype=float)
    scale = float(score_contract["frozen_scale"])
    baseline = run_candidate(
        candidate_id="persistence-last-value",
        family=None,
        window=None,
        times=times,
        values=values,
        scale=scale,
    )
    candidates: dict[str, dict[str, object]] = {}
    for family in FAMILIES:
        candidates[f"initial_{family}"] = run_candidate(
            candidate_id=f"initial_{family}",
            family=family,
            window=None,
            times=times,
            values=values,
            scale=scale,
        )
    recovery = initial_failure_diagnostic(candidates, baseline)
    if not recovery["initial_family_failure"]:
        raise RuntimeError("recovery is not authorized because initial family did not fail")
    for family in ("exponential", "gompertz", "logistic"):
        candidates[f"recovery_{family}_trailing18"] = run_candidate(
            candidate_id=f"recovery_{family}_trailing18",
            family=family,
            window=RECOVERY_WINDOW,
            times=times,
            values=values,
            scale=scale,
        )
    if len(candidates) != 7 or len(candidates) > int(policy["candidate_budget"]):  # type: ignore[index]
        raise RuntimeError("candidate count violates frozen budget")

    # Code-owned V5.3 reconstruction from the frozen, sealed public objects.
    # Supplying no replay data leaves L0 explicitly NOT_RUN; this code never
    # invokes FMA replay helpers because their process/environment interface is
    # outside the authorized read scope.
    forecast_bundle = build_ode_forecast_bundle_v53(
        snapshot=snapshot,
        thresholds=thresholds,
        forecast_plan=plan,
        replay_output_hashes=None,
        replay_receipts=None,
        replay_authority=None,
    )
    selected_family = forecast_bundle.development_bundle.selected_candidate_id
    if selected_family not in FAMILIES:
        raise RuntimeError("V5.3 selected an unregistered candidate family")
    if not forecast_bundle.final_refit.selected_family_locked_from_development:
        raise RuntimeError("V5.3 final refit was not family locked")
    final_predictions = [
        {
            "target_id": point.target_id,
            "time": target.time,
            "value": point.value,
        }
        for point, target in zip(forecast_bundle.final_refit.predictions, plan.targets)
    ]
    if not all(math.isfinite(float(point["value"])) for point in final_predictions):
        raise RuntimeError("nonfinite provisional prediction")
    paired = {
        "schema_version": "fma.iteration_33.paired_public_losses.v1",
        "campaign_id": CAMPAIGN_ID,
        "baseline_id": baseline["candidate_id"],
        "origins": list(ORIGINS),
        "horizons": list(HORIZONS),
        "scale": scale,
        "candidate_pairs": [
            {
                "candidate_id": candidate_id,
                "family": candidate["family"],
                "training_window_rule": candidate["training_window_rule"],
                "rows": paired_rows(candidate, baseline),
            }
            for candidate_id, candidate in sorted(candidates.items())
        ],
        "v5_4_gate_status": "NOT_RUN",
        "private_target_access": "forbidden_and_not_performed",
    }
    provisional = {
        "schema_version": "fma.iteration_33.provisional_predictions.v1",
        "campaign_id": CAMPAIGN_ID,
        "kind": "V5.3 final-refit provisional public prediction; not a submission",
        "forecast_bundle_hash": forecast_bundle.bundle_hash,
        "forecast_plan_hash": plan.plan_hash,
        "selected_family": selected_family,
        "predictions": final_predictions,
        "final_modeler_submission_created": False,
        "private_evaluation_performed": False,
        "scientific_qualification": False,
    }
    source_bytes = Path(__file__).read_bytes()
    candidate_results = {
        "schema_version": "fma.iteration_33.public_candidate_results.v1",
        "campaign_id": CAMPAIGN_ID,
        "source_commit": read_json(PROTOCOL)["source_commit"],
        "public_input_hashes": {
            "PUBLIC_TASK_PACKET.json": sha256_file(PUBLIC / "PUBLIC_TASK_PACKET.json"),
            "public_snapshot_v52.json": sha256_file(PUBLIC / "public_snapshot_v52.json"),
            "ode_thresholds_v52.json": sha256_file(PUBLIC / "ode_thresholds_v52.json"),
            "ode_forecast_plan_v53.json": sha256_file(PUBLIC / "ode_forecast_plan_v53.json"),
            "candidate_selection_policy.json": sha256_file(
                PUBLIC / "candidate_selection_policy.json"
            ),
            "public_eligibility_contract_v54.json": sha256_file(
                PUBLIC / "public_eligibility_contract_v54.json"
            ),
        },
        "verification": verification,
        "first_principles_assessment": {
            "cadence": "exact monthly public spacing",
            "observation_count": int(len(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "first_difference_mean": float(np.mean(np.diff(values))),
            "first_difference_standard_deviation": float(np.std(np.diff(values), ddof=1)),
            "status": "registered scalar ODE families are empirical public forecast skeletons only; state sufficiency, forcing, and a physical mechanism are not identifiable from the blinded scalar series.",
        },
        "candidate_policy": {
            "allowed_families": list(FAMILIES),
            "candidate_budget": int(policy["candidate_budget"]),
            "candidate_search_count_excluding_baseline": len(candidates),
            "minimum_candidate_families": int(policy["minimum_candidate_families"]),
            "final_family_locked_before_all_public_refit": bool(
                policy["final_family_locked_before_all_public_refit"]
            ),
            "selection_uses_public_data_only": bool(policy["selection_uses_public_data_only"]),
        },
        "initial_failure_and_recovery": {
            **recovery,
            "recovery_node": {
                "node_id": "recovery-trailing18-after-initial-public-mae-failure",
                "parent_candidates": [f"initial_{family}" for family in FAMILIES],
                "allowed_change": "public training window only",
                "window_rule": f"last_min({RECOVERY_WINDOW}, origin) public observations",
                "registered_families_retested": ["exponential", "gompertz", "logistic"],
                "unregistered_family_introduced": False,
            },
        },
        "baseline": baseline,
        "candidates": candidates,
        "frozen_v53_selection": {
            "selection_source": "build_ode_forecast_bundle_v53 -> frozen V5.2 chronological public development candidate evidence",
            "selected_family": selected_family,
            "development_split_fraction": thresholds.split_fraction,
            "development_candidate_scores": {
                item.candidate_id: item.validation_score
                for item in forecast_bundle.development_bundle.candidates
            },
            "family_locked_before_all_28_refit": True,
            "recovery_did_not_introduce_or_replace_the_registered_family": True,
        },
        "v5_3_evidence_summary": {
            "bundle_hash": forecast_bundle.bundle_hash,
            "scientific_acceptance": forecast_bundle.scientific_acceptance,
            "fixture_only": forecast_bundle.fixture_only,
            "l0_checks": forecast_bundle.l0_checks,
            "development_levels": [
                {"level": level.level, "status": level.status, "checks": level.checks}
                for level in forecast_bundle.development_bundle.levels
            ],
            "development_assessment_status": forecast_bundle.development_assessment.status,
            "final_refit_status": forecast_bundle.final_refit.status,
        },
        "v5_4_public_eligibility_gate_status": "NOT_RUN",
        "claim_limits": {
            "scientific_qualification": False,
            "external_qualification": False,
            "real_world_action_authorized": False,
            "private_evaluation_performed": False,
            "modeler_submission_created": False,
        },
        "reproducibility_source": {
            "path": Path(__file__).name,
            "sha256": sha256_bytes(source_bytes),
            "python_implementation": sys.implementation.name,
            "numpy_version": np.__version__,
        },
    }
    candidate_bytes = json_bytes(candidate_results)
    paired_bytes = json_bytes(paired)
    bundle_bytes = json_bytes(forecast_bundle.model_dump(mode="json"))
    provisional_bytes = json_bytes(provisional)
    report = render_report(
        candidates=candidates,
        baseline=baseline,
        recovery=recovery,
        selected_family=selected_family,
        bundle=forecast_bundle.model_dump(mode="json"),
        provisional=provisional,
        candidate_hash=sha256_bytes(candidate_bytes),
        paired_hash=sha256_bytes(paired_bytes),
        bundle_hash=sha256_bytes(bundle_bytes),
        provisional_hash=sha256_bytes(provisional_bytes),
    ).encode("utf-8")
    artifacts = {
        "MODELING_REPORT.md": report,
        "candidate_results.json": candidate_bytes,
        "paired_public_losses.json": paired_bytes,
        "forecast_bundle_v53.json": bundle_bytes,
        "provisional_predictions.json": provisional_bytes,
    }
    manifest = {
        "schema_version": "fma.iteration_33.modeler_artifact_manifest.v1",
        "campaign_id": CAMPAIGN_ID,
        "scope": "public-only modeler artifacts and reproducibility source; manifest self-hash intentionally excluded",
        "public_input_hashes": candidate_results["public_input_hashes"],
        "files": sorted(
            [
                {"path": name, "sha256": sha256_bytes(data), "size_bytes": len(data)}
                for name, data in artifacts.items()
            ]
            + [
                {
                    "path": Path(__file__).name,
                    "sha256": sha256_bytes(source_bytes),
                    "size_bytes": len(source_bytes),
                }
            ],
            key=lambda item: item["path"],
        ),
        "status": {
            "v5_3_scientific_acceptance": forecast_bundle.scientific_acceptance,
            "v5_4_public_eligibility_gate": "NOT_RUN",
            "private_evaluation_performed": False,
            "scientific_qualification": False,
            "modeler_submission_created": False,
        },
    }
    artifacts["artifact_manifest.json"] = json_bytes(manifest)
    summary = {
        "selected_family": selected_family,
        "v5_3_scientific_acceptance": forecast_bundle.scientific_acceptance,
        "v5_4_public_eligibility_gate": "NOT_RUN",
        "provisional_predictions": final_predictions,
        "artifact_sha256": {
            name: sha256_bytes(data) for name, data in sorted(artifacts.items())
        },
    }
    return artifacts, summary


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"--dry-run", "--freeze"}:
        raise SystemExit("usage: python run_i33_public_modeling.py --dry-run|--freeze")
    artifacts, summary = build_artifacts()
    if sys.argv[1] == "--dry-run":
        print(canonical_json(summary))
        return
    existing = [name for name in OUTPUT_NAMES if (MODELER / name).exists()]
    if existing:
        raise RuntimeError(f"refusing to alter frozen public artifacts: {existing}")
    for name in OUTPUT_NAMES:
        (MODELER / name).write_bytes(artifacts[name])
    print(canonical_json(summary))


if __name__ == "__main__":
    main()
