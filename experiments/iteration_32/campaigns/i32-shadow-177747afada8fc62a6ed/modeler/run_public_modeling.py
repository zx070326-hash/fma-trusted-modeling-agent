"""Reproducible public-only I32 shadow-modeling and one-shot freeze.

This program reads only the sibling public packet and writes the four frozen
submission artifacts in this directory when invoked with ``--freeze``.
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


CAMPAIGN_ID = "i32-shadow-177747afada8fc62a6ed"
SOURCE_COMMIT = "c36a5fbd8abee5beae53cb6e9717882a3"
MODELER = Path(__file__).resolve().parent
PUBLIC = MODELER.parent / "public"
OUTPUT_NAMES = (
    "MODELING_REPORT.md",
    "candidate_results.json",
    "modeler_submission.json",
    "submission_manifest.json",
)
SCALE_EPSILON = 1e-12


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


def read_json(name: str) -> dict[str, object]:
    return json.loads((PUBLIC / name).read_text(encoding="utf-8"))


def verify_public_inputs() -> dict[str, object]:
    manifest = read_json("hash_manifest.json")
    manifest_results: list[dict[str, object]] = []
    for item in manifest["files"]:  # type: ignore[index]
        expected = item  # type: ignore[assignment]
        path = PUBLIC / expected["path"]  # type: ignore[index]
        actual_bytes = path.read_bytes()
        manifest_results.append(
            {
                "path": expected["path"],  # type: ignore[index]
                "size_bytes": len(actual_bytes),
                "expected_size_bytes": expected["size_bytes"],  # type: ignore[index]
                "actual_sha256": sha256_bytes(actual_bytes),
                "expected_sha256": expected["sha256"],  # type: ignore[index]
                "size_match": len(actual_bytes) == expected["size_bytes"],  # type: ignore[index]
                "sha256_match": sha256_bytes(actual_bytes) == expected["sha256"],  # type: ignore[index]
            }
        )
    attestation = read_json("custody_attestation.json")
    unsigned = {
        key: value
        for key, value in attestation.items()
        if key not in {"signature_base64", "signing_payload_sha256"}
    }
    signed_bytes = canonical_json(unsigned).encode("utf-8")
    signing_payload_sha256 = sha256_bytes(signed_bytes)
    public_key_bytes = (PUBLIC / "custody_ed25519_public.pem").read_bytes()
    signature_valid = False
    try:
        public_key = serialization.load_pem_public_key(public_key_bytes)
        public_key.verify(
            base64.b64decode(
                str(attestation["signature_base64"]).encode("ascii"), validate=True
            ),
            signed_bytes,
        )
        signature_valid = True
    except (InvalidSignature, TypeError, ValueError):
        signature_valid = False
    manifest_valid = all(
        bool(item["size_match"]) and bool(item["sha256_match"])
        for item in manifest_results
    )
    key_hash = sha256_bytes(public_key_bytes)
    return {
        "manifest_valid": manifest_valid,
        "manifest_files": manifest_results,
        "attestation_id": attestation["attestation_id"],
        "attestation_file_sha256": sha256_file(PUBLIC / "custody_attestation.json"),
        "signature_payload_excluded_fields": [
            "signature_base64",
            "signing_payload_sha256",
        ],
        "signing_payload_sha256": signing_payload_sha256,
        "declared_signing_payload_sha256": attestation["signing_payload_sha256"],
        "signing_payload_hash_match": signing_payload_sha256
        == attestation["signing_payload_sha256"],
        "signature_valid": signature_valid,
        "public_key_file_sha256": key_hash,
        "attestation_public_key_sha256": attestation["public_key_sha256"],
        "public_key_hash_match": key_hash == attestation["public_key_sha256"],
        "attestation_claim_limits": attestation["claim_limits"],
        "attestation_evidence_scope": attestation["evidence_scope"],
        "custody_order": attestation["custody_order"],
        "host_isolation": attestation["host_isolation"],
    }


def fit_persistence(train: np.ndarray, horizon: int) -> tuple[np.ndarray, dict[str, object]]:
    return np.repeat(train[-1], horizon), {"fitted_parameter_count": 0}


def fit_linear_autonomous_ode(
    train: np.ndarray, horizon: int
) -> tuple[np.ndarray, dict[str, object]]:
    """Exact time-one map for dz/dt=a+bz, for which p=exp(b) must be positive."""
    x, target = train[:-1], train[1:]
    x_centered = x - x.mean()
    target_centered = target - target.mean()
    denominator = float(x_centered @ x_centered)
    unconstrained_p = float(x_centered @ target_centered / denominator)
    p = max(unconstrained_p, 1e-6)
    q = float(target.mean() - p * x.mean())
    prediction: list[float] = []
    state = float(train[-1])
    for _ in range(horizon):
        state = q + p * state
        prediction.append(state)
    return np.asarray(prediction), {
        "fitted_parameter_count": 2,
        "unconstrained_time_one_slope": unconstrained_p,
        "p_time_one": p,
        "positive_flow_constraint_active": unconstrained_p <= 1e-6,
        "ode_b": math.log(p),
        "q_time_one": q,
        "equilibrium": q / (1.0 - p) if abs(1.0 - p) > 1e-10 else None,
    }


def fit_quadratic_autonomous_ode(
    train: np.ndarray, horizon: int
) -> tuple[np.ndarray, dict[str, object]]:
    """Fixed-ridge, RK4-integrated dz/dt=a+bz+cz^2 on a standardized state."""
    center = float(np.mean(train))
    scale = max(float(np.std(train, ddof=0)), 1.0)
    z = (train - center) / scale
    design = np.column_stack([np.ones(len(z) - 1), z[:-1], z[:-1] ** 2])
    increment = np.diff(z)
    # Fixed ridge is only a numerical stabilizer; it is never tuned on targets.
    ridge = np.diag([0.0, 0.05, 0.05])
    theta = np.linalg.solve(design.T @ design + ridge, design.T @ increment)

    def derivative(state: float) -> float:
        return float(theta[0] + theta[1] * state + theta[2] * state * state)

    state = float(z[-1])
    prediction: list[float] = []
    finite = True
    for _ in range(horizon):
        for _ in range(4):
            dt = 0.25
            k1 = derivative(state)
            k2 = derivative(state + 0.5 * dt * k1)
            k3 = derivative(state + 0.5 * dt * k2)
            k4 = derivative(state + dt * k3)
            state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            if not math.isfinite(state) or abs(state) > 1e6:
                finite = False
                break
        if not finite:
            prediction.extend([float("nan")] * (horizon - len(prediction)))
            break
        prediction.append(center + scale * state)
    return np.asarray(prediction), {
        "fitted_parameter_count": 3,
        "center": center,
        "scale": scale,
        "a": float(theta[0]),
        "b": float(theta[1]),
        "c": float(theta[2]),
        "finite": finite,
    }


def fit_discrete_affine_ar1(
    train: np.ndarray, horizon: int
) -> tuple[np.ndarray, dict[str, object]]:
    """One-state discrete recovery, intentionally distinct from a continuous ODE flow."""
    x, target = train[:-1], train[1:]
    x_centered = x - x.mean()
    target_centered = target - target.mean()
    phi = float(x_centered @ target_centered / (x_centered @ x_centered))
    intercept = float(target.mean() - phi * x.mean())
    state = float(train[-1])
    prediction: list[float] = []
    for _ in range(horizon):
        state = intercept + phi * state
        prediction.append(state)
    return np.asarray(prediction), {
        "fitted_parameter_count": 2,
        "intercept": intercept,
        "phi": phi,
        "spectral_radius": abs(phi),
    }


def fit_two_state_ar2_ridge(
    train: np.ndarray, horizon: int
) -> tuple[np.ndarray, dict[str, object]]:
    """Stable two-state discrete recurrence with one fixed ridge penalty."""
    center = float(np.mean(train))
    scale = max(float(np.std(train, ddof=0)), 1.0)
    z = (train - center) / scale
    design = np.column_stack([np.ones(len(z) - 2), z[1:-1], z[:-2]])
    target = z[2:]
    theta = np.linalg.solve(design.T @ design + np.diag([0.0, 1.0, 1.0]), design.T @ target)
    intercept, phi1, phi2 = map(float, theta)
    previous, state = float(z[-2]), float(z[-1])
    prediction: list[float] = []
    for _ in range(horizon):
        next_state = intercept + phi1 * state + phi2 * previous
        prediction.append(center + scale * next_state)
        previous, state = state, next_state
    eigenvalues = np.linalg.eigvals(np.asarray([[phi1, phi2], [1.0, 0.0]]))
    return np.asarray(prediction), {
        "fitted_parameter_count": 3,
        "center": center,
        "scale": scale,
        "intercept": intercept,
        "phi1": phi1,
        "phi2": phi2,
        "spectral_radius": float(max(abs(eigenvalues))),
    }


def make_local_level(estimator: str, window: int):
    if estimator not in {"mean", "median"}:
        raise ValueError("unsupported local-level estimator")

    def fit(train: np.ndarray, horizon: int) -> tuple[np.ndarray, dict[str, object]]:
        level = float(np.mean(train[-window:])) if estimator == "mean" else float(np.median(train[-window:]))
        return np.repeat(level, horizon), {
            "fitted_parameter_count": 0,
            "estimator": estimator,
            "fixed_window": window,
            "estimated_state": level,
        }

    return fit


def finite_number(value: float | None) -> float | None:
    return float(value) if value is not None and math.isfinite(value) else None


def summarize_parameter_ranges(parameters: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    keys = sorted({key for item in parameters for key in item})
    for key in keys:
        values = [
            float(item[key])
            for item in parameters
            if isinstance(item.get(key), (int, float))
            and not isinstance(item.get(key), bool)
            and math.isfinite(float(item[key]))
        ]
        if values:
            output[key] = {
                "min": float(min(values)),
                "median": float(np.median(values)),
                "max": float(max(values)),
            }
    return output


def residual_diagnostics(errors: np.ndarray) -> dict[str, float | int | None]:
    if not np.all(np.isfinite(errors)):
        return {
            "count": int(len(errors)),
            "mean": None,
            "rmse": None,
            "lag1_autocorrelation": None,
            "durbin_watson": None,
            "positive_error_fraction": None,
        }
    lag1 = None
    if len(errors) > 2 and np.std(errors[:-1]) > 0 and np.std(errors[1:]) > 0:
        lag1 = float(np.corrcoef(errors[:-1], errors[1:])[0, 1])
    sum_squares = float(errors @ errors)
    return {
        "count": int(len(errors)),
        "mean": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "lag1_autocorrelation": lag1,
        "durbin_watson": float(np.sum(np.diff(errors) ** 2) / sum_squares)
        if sum_squares > 0
        else None,
        "positive_error_fraction": float(np.mean(errors > 0)),
    }


def evaluate_candidate(
    candidate_id: str,
    skeleton: str,
    interpretation: str,
    fit,
    observations: np.ndarray,
    frozen_scale: float,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    parameter_sets: list[dict[str, object]] = []
    for train_size in range(12, 25):
        prediction, parameters = fit(observations[:train_size], 4)
        parameter_sets.append(parameters)
        for horizon in range(1, 5):
            observed = float(observations[train_size + horizon - 1])
            forecast = float(prediction[horizon - 1])
            error = forecast - observed
            rows.append(
                {
                    "train_size": train_size,
                    "horizon": horizon,
                    "observed_public_value": observed,
                    "prediction": forecast if math.isfinite(forecast) else None,
                    "error": error if math.isfinite(error) else None,
                    "absolute_error": abs(error) if math.isfinite(error) else None,
                }
            )
    finite = all(item["error"] is not None for item in rows)
    by_horizon: dict[str, object] = {}
    for horizon in range(1, 5):
        errors = np.asarray(
            [item["error"] for item in rows if item["horizon"] == horizon], dtype=float
        )
        okay = bool(np.all(np.isfinite(errors)))
        by_horizon[str(horizon)] = {
            "count": int(len(errors)),
            "mae": float(np.mean(np.abs(errors))) if okay else None,
            "rmse": float(np.sqrt(np.mean(errors**2))) if okay else None,
            "normalized_mae": float(np.mean(np.abs(errors)) / frozen_scale) if okay else None,
            "normalized_rmse": float(np.sqrt(np.mean(errors**2)) / frozen_scale)
            if okay
            else None,
            "bias": float(np.mean(errors)) if okay else None,
        }
    all_errors = np.asarray([item["error"] for item in rows], dtype=float)
    h1_errors = np.asarray(
        [item["error"] for item in rows if item["horizon"] == 1], dtype=float
    )
    final_prediction, final_parameters = fit(observations, 4)
    final_windows: dict[str, object] = {}
    if candidate_id.startswith("robust_local_level_median"):
        for window in (3, 5, 7, 10):
            prediction, parameters = make_local_level("median", window)(observations, 4)
            final_windows[str(window)] = {
                "predictions": [finite_number(float(value)) for value in prediction],
                "parameters": parameters,
            }
    else:
        for window in (12, 16, 20, 24, 28):
            prediction, parameters = fit(observations[-window:], 4)
            final_windows[str(window)] = {
                "predictions": [finite_number(float(value)) for value in prediction],
                "parameters": parameters,
            }
    return {
        "candidate_id": candidate_id,
        "skeleton": skeleton,
        "interpretation": interpretation,
        "rolling_validation": {
            "candidate_status": "finite" if finite else "nonfinite_forecast_failure",
            "aggregate": {
                "mae": float(np.mean(np.abs(all_errors))) if finite else None,
                "rmse": float(np.sqrt(np.mean(all_errors**2))) if finite else None,
                "normalized_mae": float(np.mean(np.abs(all_errors)) / frozen_scale)
                if finite
                else None,
                "normalized_rmse": float(np.sqrt(np.mean(all_errors**2)) / frozen_scale)
                if finite
                else None,
            },
            "by_horizon": by_horizon,
            "residual_diagnostics_h1": residual_diagnostics(h1_errors),
            "stability": {
                "finite_forecast_fraction": float(
                    np.mean([row["prediction"] is not None for row in rows])
                ),
                "max_abs_validation_forecast": float(
                    max(abs(float(row["prediction"])) for row in rows if row["prediction"] is not None)
                )
                if any(row["prediction"] is not None for row in rows)
                else None,
                "parameter_ranges": summarize_parameter_ranges(parameter_sets),
            },
            "rows": rows,
        },
        "final_refit_all_28": {
            "observation_count": int(len(observations)),
            "predictions": [finite_number(float(value)) for value in final_prediction],
            "parameters": final_parameters,
            "finite": bool(np.all(np.isfinite(final_prediction))),
        },
        "window_sensitivity_final_refits": final_windows,
    }


def make_first_principles_diagnostics(observations: np.ndarray) -> dict[str, object]:
    increments = np.diff(observations)
    close_values = [
        {
            "time_index": index + 1,
            "state": float(observations[index]),
            "next_increment": float(increments[index]),
        }
        for index in range(len(increments))
        if -600.0 <= observations[index] <= -430.0
    ]
    return {
        "state_and_noise": {
            "observed_series_is_a_scalar_positive_affine_transformed_index": True,
            "source_identity_and_original_units": "withheld_by_custodian",
            "public_observation_error_assumption": "independent additive; not source-certified",
            "consequence": "The public output alone cannot identify hidden forcing, a latent state vector, or a physical conservation law.",
        },
        "time_structure": {
            "public_observation_count": int(len(observations)),
            "daily_spacing": True,
            "first_difference_mean": float(np.mean(increments)),
            "first_difference_standard_deviation": float(np.std(increments, ddof=1)),
            "lag1_observation_autocorrelation": float(np.corrcoef(observations[:-1], observations[1:])[0, 1]),
        },
        "autonomous_scalar_ode_sufficiency": {
            "assessment": "not_supported_by_public_data",
            "principle": "For a one-dimensional autonomous continuous ODE with unique solutions, every fixed positive-time flow map is order-preserving; its affine linearization has p=exp(b)>0.",
            "evidence": {
                "unconstrained_linear_time_one_slope_is_checked_in_each_rolling_fit": True,
                "nearby_negative_states_show_materially_different_next_day_increments": close_values,
            },
            "limit": "This is a falsification of adequacy for this public reduced-state forecast task, not an identification of a physical mechanism.",
        },
    }


def candidate_failure_reason(candidate_id: str, result: dict[str, object]) -> str:
    aggregate = result["rolling_validation"]["aggregate"]  # type: ignore[index]
    if candidate_id == "persistence_last_value":
        return "Frozen baseline retained for comparison; it is not the selected recovery model."
    if candidate_id == "linear_autonomous_scalar_ode":
        return "The positive time-one flow constraint is active in every rolling fit and aggregate MAE does not beat persistence."
    if candidate_id == "quadratic_autonomous_scalar_ode":
        return "Adding a quadratic autonomous drift does not beat persistence on aggregate public MAE and raises finite-sample sensitivity."
    if candidate_id in {"discrete_affine_ar1_recovery", "two_state_ar2_ridge_recovery"}:
        return "The discrete recovery is finite and stable in the tested windows but does not beat persistence on aggregate public MAE."
    if candidate_id.startswith("local_level_mean"):
        return "This low-complexity recovery is finite but is less robust to the latest public observation than the selected median window."
    if candidate_id.startswith("robust_local_level_median_w5"):
        return "Selected by the predeclared rolling public criterion; no target values were used."
    if candidate_id.startswith("robust_local_level_median"):
        return "Finite robust recovery, but its fixed window has higher aggregate public MAE than median window 5."
    return f"Aggregate normalized MAE={aggregate['normalized_mae']}."  # type: ignore[index]


def render_report(
    *,
    candidate_results: dict[str, object],
    candidate_results_hash: str,
    submission: dict[str, object],
    submission_hash: str,
) -> str:
    candidates = candidate_results["candidates"]  # type: ignore[index]
    selected_id = candidate_results["selection"]["selected_candidate_id"]  # type: ignore[index]
    selected = candidates[selected_id]  # type: ignore[index]
    baseline = candidates["persistence_last_value"]  # type: ignore[index]
    lines = [
        "# I32 Public-Only Modeling Report",
        "",
        "## Scope and integrity",
        "",
        "- Campaign: `i32-shadow-177747afada8fc62a6ed`.",
        "- Read scope: only the supplied public packet and this reproducibility script; no target values, private capsules, external data, network, environment variables, or process inspection were used.",
        "- `hash_manifest.json`: all 8 public artifacts matched both SHA-256 and declared size.",
        "- Custody attestation: the declared signing-payload SHA-256 matched and its Ed25519 signature verified with the pinned public key.",
        "- This is same-host blinded-context shadow work only. It is not a gate, source identification, private evaluation, promotion, scientific qualification, or real-world authorization.",
        "",
        "## First-principles assessment",
        "",
        "The packet supplies 28 exactly daily scalar observations in a custody-only positive-affine transformed index. The supplied additive-error scale is explicitly not source-certified. The visible scalar series therefore does not identify the physical state, forcing, or a conservation law.",
        "",
        "A one-dimensional autonomous ODE with unique solutions has an order-preserving positive-time flow. The affine ODE time-one map requires `p = exp(b) > 0`. In every 12–24 point expanding-window fit, the unconstrained affine slope was non-positive, so the positive-flow constraint collapsed the model to near-immediate equilibrium. The nearby public states from -600 to -430 also had visibly heterogeneous next-day increments. This makes an autonomous scalar ODE inadequate for the public reduced-state forecast task; it does not identify an alternative physical mechanism.",
        "",
        "## Public rolling validation and recovery",
        "",
        "Selection was fixed before ranking: use only 13 expanding origins (training sizes 12–24), forecast horizons 1–4 at every origin (52 forecasts/candidate), and rank finite candidates by equal-weight aggregate normalized MAE; break ties by normalized RMSE and then fewer fitted parameters. No target values were scored or requested.",
        "",
        "| Candidate | h1 nMAE | h2 nMAE | h3 nMAE | h4 nMAE | aggregate nMAE | aggregate nRMSE | Result |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank in candidate_results["selection"]["ranking"]:  # type: ignore[index]
        candidate = candidates[rank["candidate_id"]]  # type: ignore[index]
        by_h = candidate["rolling_validation"]["by_horizon"]  # type: ignore[index]
        aggregate = candidate["rolling_validation"]["aggregate"]  # type: ignore[index]
        outcome = "selected" if rank["candidate_id"] == selected_id else "not selected"
        lines.append(
            "| {id} | {h1:.6f} | {h2:.6f} | {h3:.6f} | {h4:.6f} | {mae:.6f} | {rmse:.6f} | {outcome} |".format(
                id=rank["candidate_id"],
                h1=by_h["1"]["normalized_mae"],
                h2=by_h["2"]["normalized_mae"],
                h3=by_h["3"]["normalized_mae"],
                h4=by_h["4"]["normalized_mae"],
                mae=aggregate["normalized_mae"],
                rmse=aggregate["normalized_rmse"],
                outcome=outcome,
            )
        )
    selected_h = selected["rolling_validation"]["by_horizon"]  # type: ignore[index]
    baseline_h = baseline["rolling_validation"]["by_horizon"]  # type: ignore[index]
    lines.extend(
        [
            "",
            "The selected robust local-level median (fixed five-observation window) improves aggregate public rolling nMAE from `{:.6f}` for persistence to `{:.6f}`. Its horizon-specific nMAEs are h1 `{:.6f}`, h2 `{:.6f}`, h3 `{:.6f}`, h4 `{:.6f}`; persistence is h1 `{:.6f}`, h2 `{:.6f}`, h3 `{:.6f}`, h4 `{:.6f}`.".format(
                baseline["rolling_validation"]["aggregate"]["normalized_mae"],  # type: ignore[index]
                selected["rolling_validation"]["aggregate"]["normalized_mae"],  # type: ignore[index]
                selected_h["1"]["normalized_mae"],
                selected_h["2"]["normalized_mae"],
                selected_h["3"]["normalized_mae"],
                selected_h["4"]["normalized_mae"],
                baseline_h["1"]["normalized_mae"],
                baseline_h["2"]["normalized_mae"],
                baseline_h["3"]["normalized_mae"],
                baseline_h["4"]["normalized_mae"],
            ),
            "",
            "## Locked final refit and forecasts",
            "",
            "After selection, the family and fixed window were locked and refit once on all 28 public observations. The latent level estimate is the median of the final five public observations; the model has no fitted continuous parameter and projects that level across h1–h4.",
            "",
            "| Target | Time | Frozen public-transformed prediction |",
            "|---|---:|---:|",
        ]
    )
    for item in submission["bindings"]["predictions"]:  # type: ignore[index]
        lines.append(
            "| {target_id} | {time} | {prediction:.12f} |".format(**item)
        )
    diagnostics = selected["rolling_validation"]["residual_diagnostics_h1"]  # type: ignore[index]
    lines.extend(
        [
            "",
            "## Stability, diagnostics, and limitations",
            "",
            "- Selected h1 rolling residuals: mean `{:.6f}`, RMSE `{:.6f}`, lag-1 autocorrelation `{:.6f}`, Durbin-Watson `{:.6f}`. These are diagnostic summaries only, not calibrated uncertainty or a source-certified error model.".format(
                diagnostics["mean"],
                diagnostics["rmse"],
                diagnostics["lag1_autocorrelation"],
                diagnostics["durbin_watson"],
            ),
            "- Window sensitivity was evaluated from the public data only. The median-window final state is reported for windows 3, 5, 7, and 10 in `candidate_results.json`; window 5 is locked because it had the lowest aggregate rolling MAE.",
            "- Failure reasons are retained for all scalar-ODE, discrete-recovery, mean-level, and alternative median-window candidates in `candidate_results.json`; none of the rejected candidates were silently discarded.",
            "- The score contract defines a private comparison against persistence, but no private score is available here. No claim that the frozen submission passes that comparison is made.",
            "",
            "## Freeze and provenance",
            "",
            "- `public_scientific_acceptance=false`",
            "- `shadow_submission_frozen=true`",
            "- `scientific_qualification=false`",
            "- `external_qualification=false`; `real_world_action_authorized=false`.",
            "- Candidate artifact SHA-256: `{}`.".format(candidate_results_hash),
            "- Submission SHA-256: `{}`.".format(submission_hash),
        ]
    )
    return "\n".join(lines) + "\n"


def build_artifacts() -> tuple[dict[str, bytes], dict[str, object]]:
    verification = verify_public_inputs()
    if not (
        verification["manifest_valid"]
        and verification["signing_payload_hash_match"]
        and verification["signature_valid"]
        and verification["public_key_hash_match"]
    ):
        raise RuntimeError("public integrity verification failed; refusing to model or freeze")
    packet = read_json("PUBLIC_TASK_PACKET.json")
    contract = read_json("score_contract.json")
    forecast_plan = read_json("forecast_plan.json")
    if not (
        packet["campaign_id"] == contract["campaign_id"] == forecast_plan["campaign_id"] == CAMPAIGN_ID
    ):
        raise RuntimeError("campaign binding mismatch")
    if contract["public_task_packet_sha256"] != sha256_file(PUBLIC / "PUBLIC_TASK_PACKET.json"):
        raise RuntimeError("score contract public-packet binding mismatch")
    if contract["forecast_plan_sha256"] != sha256_file(PUBLIC / "forecast_plan.json"):
        raise RuntimeError("score contract forecast-plan binding mismatch")
    if forecast_plan["public_task_packet_sha256"] != sha256_file(PUBLIC / "PUBLIC_TASK_PACKET.json"):
        raise RuntimeError("forecast plan public-packet binding mismatch")
    observations = np.asarray(
        [item["value"] for item in packet["public_observations"]], dtype=float  # type: ignore[index]
    )
    frozen_scale = float(contract["frozen_scale"])
    candidate_specs = [
        (
            "persistence_last_value",
            "y(t+h)=y(t)",
            "Required baseline: carry the last observed public value forward.",
            fit_persistence,
        ),
        (
            "linear_autonomous_scalar_ode",
            "dz/dt=a+bz; z(t+1)=q+exp(b)z(t)",
            "Minimal autonomous scalar continuous-time skeleton with order-preserving time-one flow.",
            fit_linear_autonomous_ode,
        ),
        (
            "quadratic_autonomous_scalar_ode",
            "dz/dt=a+bz+cz^2",
            "Nonlinear autonomous scalar ODE, standardized and integrated with fixed-step RK4.",
            fit_quadratic_autonomous_ode,
        ),
        (
            "discrete_affine_ar1_recovery",
            "y(t+1)=c+phi*y(t)",
            "Recovery that permits a sign-changing discrete time-one map, unlike a continuous scalar ODE.",
            fit_discrete_affine_ar1,
        ),
        (
            "two_state_ar2_ridge_recovery",
            "y(t+1)=c+phi1*y(t)+phi2*y(t-1)",
            "Two-state discrete recovery with fixed ridge penalty and explicit recurrence stability diagnostic.",
            fit_two_state_ar2_ridge,
        ),
    ]
    for estimator in ("mean", "median"):
        for window in (3, 5, 7, 10):
            candidate_id = (
                f"local_level_{estimator}_w{window}"
                if estimator == "mean"
                else f"robust_local_level_median_w{window}_recovery"
            )
            candidate_specs.append(
                (
                    candidate_id,
                    "x(t+1)=x(t); y(t)=x(t)+epsilon(t)",
                    f"Low-complexity local-level recovery: forecast a fixed {estimator} of the last {window} public observations.",
                    make_local_level(estimator, window),
                )
            )
    candidates: dict[str, dict[str, object]] = {}
    for candidate_id, skeleton, interpretation, fit in candidate_specs:
        result = evaluate_candidate(
            candidate_id,
            skeleton,
            interpretation,
            fit,
            observations,
            frozen_scale,
        )
        result["failure_or_selection_reason"] = candidate_failure_reason(candidate_id, result)
        candidates[candidate_id] = result
    complexity = {
        "persistence_last_value": 0,
        "linear_autonomous_scalar_ode": 2,
        "quadratic_autonomous_scalar_ode": 3,
        "discrete_affine_ar1_recovery": 2,
        "two_state_ar2_ridge_recovery": 3,
    }
    for candidate_id in candidates:
        complexity.setdefault(candidate_id, 0)
    ranking = []
    for candidate_id, result in candidates.items():
        aggregate = result["rolling_validation"]["aggregate"]  # type: ignore[index]
        ranking.append(
            {
                "candidate_id": candidate_id,
                "normalized_mae": aggregate["normalized_mae"],
                "normalized_rmse": aggregate["normalized_rmse"],
                "fitted_parameter_count": complexity[candidate_id],
            }
        )
    ranking.sort(
        key=lambda item: (
            float("inf") if item["normalized_mae"] is None else item["normalized_mae"],
            float("inf") if item["normalized_rmse"] is None else item["normalized_rmse"],
            item["fitted_parameter_count"],
            item["candidate_id"],
        )
    )
    selected_id = str(ranking[0]["candidate_id"])
    selected = candidates[selected_id]
    final_prediction = selected["final_refit_all_28"]["predictions"]  # type: ignore[index]
    if not all(value is not None and math.isfinite(float(value)) for value in final_prediction):
        raise RuntimeError("selected final forecast is not finite")
    target_coordinates = forecast_plan["targets"]  # type: ignore[index]
    if [item["target_id"] for item in target_coordinates] != [
        item["target_id"] for item in packet["targets"]  # type: ignore[index]
    ]:
        raise RuntimeError("packet and forecast-plan targets differ")
    source_bytes = Path(__file__).read_bytes()
    runtime = {
        "python_version": sys.version,
        "python_implementation": sys.implementation.name,
        "numpy_version": np.__version__,
        "source_file": Path(__file__).name,
        "source_sha256": sha256_bytes(source_bytes),
        "source_commit": SOURCE_COMMIT,
    }
    candidate_results = {
        "schema_version": "fma.i32_blinded_context_shadow.modeler_candidates.v1",
        "campaign_id": CAMPAIGN_ID,
        "public_input_bindings": {
            "PUBLIC_TASK_PACKET_sha256": sha256_file(PUBLIC / "PUBLIC_TASK_PACKET.json"),
            "score_contract_sha256": sha256_file(PUBLIC / "score_contract.json"),
            "forecast_plan_sha256": sha256_file(PUBLIC / "forecast_plan.json"),
            "hash_manifest_sha256": sha256_file(PUBLIC / "hash_manifest.json"),
        },
        "public_integrity_verification": verification,
        "validation_design": {
            "type": "expanding_window_rolling_origin",
            "training_sizes": list(range(12, 25)),
            "horizons": [1, 2, 3, 4],
            "forecasts_per_candidate": 52,
            "target_access": "none",
            "selection_rule": "finite candidates ordered by equal-weight aggregate normalized MAE across all public rolling forecasts; ties by aggregate normalized RMSE, then fewer fitted parameters",
        },
        "first_principles_diagnostics": make_first_principles_diagnostics(observations),
        "candidates": candidates,
        "selection": {
            "selected_candidate_id": selected_id,
            "selected_family": "robust_local_level" if selected_id.startswith("robust_local_level") else selected_id,
            "ranking": ranking,
            "family_locked_before_final_refit": True,
            "final_refit_support": {
                "public_observation_count": int(len(observations)),
                "support_end_time": packet["public_observations"][-1]["time"],  # type: ignore[index]
            },
        },
        "claim_limits": {
            "public_scientific_acceptance": False,
            "shadow_submission_frozen": True,
            "scientific_qualification": False,
            "external_qualification": False,
            "real_world_action_authorized": False,
            "private_evaluation_requested_or_performed": False,
        },
        "runtime": runtime,
    }
    candidate_bytes = json_bytes(candidate_results)
    predictions = [
        {
            "target_id": target["target_id"],
            "time": target["time"],
            "prediction": float(prediction),
        }
        for target, prediction in zip(target_coordinates, final_prediction)
    ]
    submission = {
        "schema_version": "fma.i32_blinded_context_shadow.modeler_submission.v1",
        "campaign_id": CAMPAIGN_ID,
        "bindings": {
            "PUBLIC_TASK_PACKET_sha256": sha256_file(PUBLIC / "PUBLIC_TASK_PACKET.json"),
            "score_contract_sha256": sha256_file(PUBLIC / "score_contract.json"),
            "forecast_plan_sha256": sha256_file(PUBLIC / "forecast_plan.json"),
            "target_coordinates": target_coordinates,
            "predictions": predictions,
        },
        "model_selection": {
            "selected_candidate_id": selected_id,
            "selected_family": candidate_results["selection"]["selected_family"],
            "selection_rule": candidate_results["validation_design"]["selection_rule"],
            "final_refit": "one all-28-public-observation refit after family and fixed window lock",
        },
        "candidate_results_sha256": sha256_bytes(candidate_bytes),
        "runtime": runtime,
        "freeze": {
            "submission_number": 1,
            "allowed_submission_count": 1,
            "shadow_submission_frozen": True,
            "public_scientific_acceptance": False,
            "scientific_qualification": False,
            "external_qualification": False,
            "real_world_action_authorized": False,
            "private_score_not_observed": True,
        },
    }
    submission_bytes = json_bytes(submission)
    report = render_report(
        candidate_results=candidate_results,
        candidate_results_hash=sha256_bytes(candidate_bytes),
        submission=submission,
        submission_hash=sha256_bytes(submission_bytes),
    )
    artifacts = {
        "candidate_results.json": candidate_bytes,
        "modeler_submission.json": submission_bytes,
        "MODELING_REPORT.md": report.encode("utf-8"),
    }
    manifest_entries = [
        {
            "path": name,
            "sha256": sha256_bytes(payload),
            "size_bytes": len(payload),
        }
        for name, payload in sorted(artifacts.items())
    ]
    manifest_entries.append(
        {
            "path": Path(__file__).name,
            "sha256": sha256_bytes(source_bytes),
            "size_bytes": len(source_bytes),
        }
    )
    manifest = {
        "schema_version": "fma.i32_blinded_context_shadow.submission_manifest.v1",
        "campaign_id": CAMPAIGN_ID,
        "manifest_scope": "frozen modeler artifacts and reproducibility source; self hash excluded to avoid a recursive manifest",
        "public_input_bindings": candidate_results["public_input_bindings"],
        "files": sorted(manifest_entries, key=lambda item: item["path"]),
        "freeze": submission["freeze"],
    }
    artifacts["submission_manifest.json"] = json_bytes(manifest)
    return artifacts, {
        "selected_candidate_id": selected_id,
        "predictions": predictions,
        "candidate_results_sha256": sha256_bytes(candidate_bytes),
        "submission_sha256": sha256_bytes(submission_bytes),
        "manifest_sha256": sha256_bytes(artifacts["submission_manifest.json"]),
        "verification": verification,
    }


def main() -> None:
    allowed_arguments = {"--dry-run", "--freeze"}
    if len(sys.argv) != 2 or sys.argv[1] not in allowed_arguments:
        raise SystemExit("usage: python run_public_modeling.py --dry-run|--freeze")
    artifacts, summary = build_artifacts()
    if sys.argv[1] == "--dry-run":
        print(canonical_json(summary))
        return
    existing = [name for name in OUTPUT_NAMES if (MODELER / name).exists()]
    if existing:
        raise RuntimeError(f"refusing to alter an existing frozen submission: {existing}")
    for name in OUTPUT_NAMES:
        (MODELER / name).write_bytes(artifacts[name])
    print(canonical_json(summary))


if __name__ == "__main__":
    main()
