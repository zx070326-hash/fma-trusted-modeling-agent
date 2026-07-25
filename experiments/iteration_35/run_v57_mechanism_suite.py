"""Run and freeze the V5.7 representation-routing mechanism suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from fma.hashing import canonical_json, sha256_value
from fma.v5_2.ode_system import ODETimeSeriesSnapshotV52
from fma.v5_6.hybrid_ode import HybridODEThresholdsV56
from fma.v5_7.adaptive_positive_series import (
    AdaptivePositiveSeriesBundleV57,
    AdaptiveReplayAuthorityV57,
    AdaptiveThresholdsV57,
    build_adaptive_positive_series_bundle_v57,
    run_authenticated_adaptive_replays_v57,
)


ROOT = Path(__file__).resolve().parents[2]
PRIMARY_THRESHOLD_PATH = ROOT / "V5_6_HYBRID_THRESHOLDS.json"
ADAPTIVE_THRESHOLD_PATH = ROOT / "V5_7_ADAPTIVE_THRESHOLDS.json"
I34_SNAPSHOT_PATH = (
    ROOT
    / "experiments"
    / "iteration_34"
    / "campaign_public"
    / "public_snapshot_v52.json"
)
I35_SNAPSHOT_PATH = (
    ROOT
    / "experiments"
    / "iteration_35"
    / "campaign_unseen_v56"
    / "campaign_public_v55"
    / "public_snapshot_v52.json"
)


def _json_bytes(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _snapshot(task_id: str, values: np.ndarray) -> ODETimeSeriesSnapshotV52:
    return ODETimeSeriesSnapshotV52.seal(
        task_id=task_id,
        time_unit="year",
        state_unit="positive_index",
        times=np.arange(len(values), dtype=float).tolist(),
        observations=values.tolist(),
        source_id=f"{task_id}-frozen-fixture",
        fixture_only=True,
    )


def _ode_ar1_snapshot() -> ODETimeSeriesSnapshotV52:
    times = np.arange(72, dtype=float)
    trend = 220.0 / (1.0 + 9.0 * np.exp(-0.11 * times))
    rng = np.random.default_rng(20260726)
    innovations = rng.normal(0.0, 0.5, len(times))
    residuals = np.zeros(len(times), dtype=float)
    for index in range(1, len(times)):
        residuals[index] = 0.90 * residuals[index - 1] + innovations[index]
    return _snapshot("v57-suite-ode-ar1", trend + residuals)


def _growth_snapshot(
    *,
    task_id: str,
    mean: float,
    phi: float,
    sigma: float,
    seed: int,
    structural_break_at: int | None = None,
    scale: float = 1.0,
) -> ODETimeSeriesSnapshotV52:
    rng = np.random.default_rng(seed)
    growths = np.zeros(71, dtype=float)
    growths[0] = mean
    for index in range(1, len(growths)):
        local_mean = (
            0.14
            if structural_break_at is not None
            and index >= structural_break_at
            else mean
        )
        growths[index] = (
            local_mean
            + phi * (growths[index - 1] - local_mean)
            + rng.normal(0.0, sigma)
        )
    values = (
        100.0
        * scale
        * np.exp(np.concatenate(([0.0], np.cumsum(growths))))
    )
    return _snapshot(task_id, values)


def _cases() -> list[
    tuple[str, ODETimeSeriesSnapshotV52, dict[str, object]]
]:
    return [
        (
            "valid-hybrid-ode-preserved",
            _ode_ar1_snapshot(),
            {
                "selected_branch": "hybrid_ode",
                "selected_model_id": "logistic.ar1_residual",
                "l3_status": "PASS",
                "l4_status": "PASS",
                "scientific_acceptance": True,
            },
        ),
        (
            "stationary-log-drift-recovery",
            _growth_snapshot(
                task_id="v57-suite-log-drift",
                mean=0.04,
                phi=0.0,
                sigma=0.01,
                seed=102,
            ),
            {
                "selected_branch": "log_growth",
                "selected_model_id": "log_random_walk_drift",
                "l3_status": "PASS",
                "l4_status": "PASS",
                "scientific_acceptance": True,
            },
        ),
        (
            "stationary-log-growth-ar1-recovery",
            _growth_snapshot(
                task_id="v57-suite-log-growth-ar1",
                mean=0.04,
                phi=0.85,
                sigma=0.02,
                seed=103,
            ),
            {
                "selected_branch": "log_growth",
                "selected_model_id": "log_growth_ar1",
                "l3_status": "PASS",
                "l4_status": "PASS",
                "scientific_acceptance": True,
            },
        ),
        (
            "scaled-log-growth-ar1-recovery",
            _growth_snapshot(
                task_id="v57-suite-scaled-log-growth-ar1",
                mean=0.04,
                phi=0.85,
                sigma=0.02,
                seed=103,
                scale=1000.0,
            ),
            {
                "selected_branch": "log_growth",
                "selected_model_id": "log_growth_ar1",
                "l3_status": "PASS",
                "l4_status": "PASS",
                "scientific_acceptance": True,
            },
        ),
        (
            "growth-structural-break-reject",
            _growth_snapshot(
                task_id="v57-suite-growth-break",
                mean=0.03,
                phi=0.0,
                sigma=0.008,
                seed=104,
                structural_break_at=45,
            ),
            {
                "selected_branch": "unresolved",
                "l3_status": "FAIL",
                "l4_status": "FAIL",
                "scientific_acceptance": False,
            },
        ),
        (
            "i34-retrospective-reject",
            ODETimeSeriesSnapshotV52.model_validate_json(
                I34_SNAPSHOT_PATH.read_text(encoding="utf-8")
            ),
            {
                "selected_branch": "unresolved",
                "selected_model_id": "log_growth_ar1",
                "l3_status": "FAIL",
                "l4_status": "FAIL",
                "scientific_acceptance": False,
            },
        ),
        (
            "i35-retrospective-development-recovery",
            ODETimeSeriesSnapshotV52.model_validate_json(
                I35_SNAPSHOT_PATH.read_text(encoding="utf-8")
            ),
            {
                "selected_branch": "log_growth",
                "selected_model_id": "log_growth_ar1",
                "l3_status": "PASS",
                "l4_status": "PASS",
                "scientific_acceptance": True,
            },
        ),
    ]


def _matches(actual: dict[str, object], expected: dict[str, object]) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def _selected_growth(
    bundle: AdaptivePositiveSeriesBundleV57,
) -> object | None:
    return next(
        (
            item
            for item in bundle.growth_candidates
            if item.candidate_id == bundle.graph.selected_model_id
        ),
        None,
    )


def _report(
    rows: list[dict[str, object]],
    *,
    scale_invariance_passed: bool,
    suite_passed: bool,
) -> str:
    lines = [
        "# V5.7 adaptive positive-series mechanism suite",
        "",
        f"Status: `{'PASS' if suite_passed else 'FAIL'}`",
        "",
        "| Case | Branch | Selected | L3 | L4 | Acceptance | Expected |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {case_id} | {selected_branch} | {selected_model_id} | "
            "{l3_status} | {l4_status} | {scientific_acceptance} | "
            "{expected_outcome_matched} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Mechanism evidence",
            "",
            "- A valid V5.6 hybrid ODE remains on the primary branch.",
            "- Constant-drift and stationary AR(1) log-growth fixtures route to "
            "their registered recovery models.",
            "- Growth structural break and I34 fail closed.",
            (
                "- Positive level scaling invariance: "
                f"`{str(scale_invariance_passed).lower()}`."
            ),
            "- I35 is recovered only as disclosed retrospective development evidence.",
            "",
            "## Claim boundary",
            "",
            "- Fixture scientific acceptance is mechanism evidence only.",
            "- I34 and I35 are disclosed retrospective controls, not unseen tasks.",
            "- No causal mechanism, private qualification, external host, or "
            "real-world action is established.",
            "",
        ]
    )
    return "\n".join(lines)


def run_suite(*, replay_secret_path: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    primary = HybridODEThresholdsV56.seal(
        **json.loads(PRIMARY_THRESHOLD_PATH.read_text(encoding="utf-8"))
    )
    adaptive = AdaptiveThresholdsV57.seal(
        **json.loads(ADAPTIVE_THRESHOLD_PATH.read_text(encoding="utf-8"))
    )
    secret = replay_secret_path.read_bytes()
    authority = AdaptiveReplayAuthorityV57(
        key_id="i35-v57-local-mechanism-replay",
        secret=secret,
    )
    output = output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.tmp-",
            dir=output.parent,
        )
    )
    try:
        rows: list[dict[str, object]] = []
        artifacts: dict[str, bytes] = {}
        bundles: dict[str, AdaptivePositiveSeriesBundleV57] = {}
        for case_id, snapshot, expected in _cases():
            replay_input = {
                "snapshot": snapshot.model_dump(mode="json"),
                "primary_thresholds": primary.model_dump(mode="json"),
                "adaptive_thresholds": adaptive.model_dump(mode="json"),
            }
            replay_path = temporary / f"{case_id}.replay_input.json"
            _write_new(replay_path, _json_bytes(replay_input))
            receipts = run_authenticated_adaptive_replays_v57(
                replay_path,
                authority=authority,
            )
            bundle = build_adaptive_positive_series_bundle_v57(
                snapshot=snapshot,
                primary_thresholds=primary,
                adaptive_thresholds=adaptive,
                replay_receipts=receipts,
                replay_authority=authority,
            )
            bundles[case_id] = bundle
            selected = _selected_growth(bundle)
            actual = {
                "selected_branch": bundle.graph.selected_branch,
                "selected_model_id": bundle.graph.selected_model_id,
                "l3_status": bundle.levels[3].status,
                "l4_status": bundle.levels[4].status,
                "scientific_acceptance": bundle.scientific_acceptance,
            }
            rows.append(
                {
                    "case_id": case_id,
                    **actual,
                    "expected": expected,
                    "expected_outcome_matched": _matches(actual, expected),
                    "primary_selected_candidate_id": (
                        bundle.primary_bundle.selected_candidate_id
                    ),
                    "primary_l3_status": bundle.graph.primary_level_statuses[
                        "L3"
                    ],
                    "selected_growth_validation_relative_rmse": (
                        selected.validation_relative_rmse
                        if selected is not None
                        else None
                    ),
                    "selected_growth_innovation_lag": (
                        selected.absolute_validation_innovation_lag1_correlation
                        if selected is not None
                        else None
                    ),
                    "selected_growth_raw_phi": (
                        selected.process_fit.raw_phi
                        if selected is not None
                        else None
                    ),
                    "bundle_hash": bundle.bundle_hash,
                    "graph_hash": bundle.graph.graph_hash,
                    "replay_receipt_hashes": bundle.replay_receipt_hashes,
                    "fixture_only": snapshot.fixture_only,
                    "retrospective_development_only": case_id.startswith(
                        ("i34-", "i35-")
                    ),
                    "causal_mechanism_identified": False,
                    "scientific_qualification_granted": False,
                    "real_world_action_authorized": False,
                }
            )
            artifacts[f"{case_id}.bundle.json"] = _json_bytes(bundle)
            artifacts[f"{case_id}.replay_receipts.json"] = _json_bytes(
                receipts
            )

        original = bundles["stationary-log-growth-ar1-recovery"]
        scaled = bundles["scaled-log-growth-ar1-recovery"]
        original_selected = _selected_growth(original)
        scaled_selected = _selected_growth(scaled)
        scale_invariance = bool(
            original.graph.selected_model_id == scaled.graph.selected_model_id
            and original_selected is not None
            and scaled_selected is not None
            and np.isclose(
                original_selected.validation_relative_rmse,
                scaled_selected.validation_relative_rmse,
            )
            and np.isclose(
                original_selected.process_fit.mean_log_growth,
                scaled_selected.process_fit.mean_log_growth,
            )
            and np.isclose(
                original_selected.process_fit.raw_phi,
                scaled_selected.process_fit.raw_phi,
            )
            and np.isclose(
                scaled_selected.forecast_value,
                original_selected.forecast_value * 1000.0,
            )
        )
        suite_passed = bool(
            all(row["expected_outcome_matched"] for row in rows)
            and scale_invariance
        )
        result = {
            "schema_version": "5.7-mechanism-suite-result",
            "primary_threshold_file_sha256": hashlib.sha256(
                PRIMARY_THRESHOLD_PATH.read_bytes()
            ).hexdigest(),
            "adaptive_threshold_file_sha256": hashlib.sha256(
                ADAPTIVE_THRESHOLD_PATH.read_bytes()
            ).hexdigest(),
            "primary_threshold_hash": primary.threshold_hash,
            "adaptive_threshold_hash": adaptive.threshold_hash,
            "case_count": len(rows),
            "rows": rows,
            "positive_scale_invariance_passed": scale_invariance,
            "suite_passed": suite_passed,
            "fixture_scientific_qualification_granted": False,
            "retrospective_scientific_qualification_granted": False,
            "causal_mechanism_identified": False,
            "external_host_established": False,
            "real_world_action_authorized": False,
        }
        result["result_hash"] = sha256_value(result)
        artifacts["MECHANISM_SUITE_RESULTS.json"] = _json_bytes(result)
        artifacts["REPORT.md"] = _report(
            rows,
            scale_invariance_passed=scale_invariance,
            suite_passed=suite_passed,
        ).encode("utf-8")
        for name, payload in sorted(artifacts.items()):
            _write_new(temporary / name, payload)
        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        manifest = {
            "schema_version": "5.7-mechanism-suite-manifest",
            "source_commit": source_commit,
            "runner_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "adapter_sha256": hashlib.sha256(
                (
                    ROOT
                    / "fma"
                    / "v5_7"
                    / "adaptive_positive_series.py"
                ).read_bytes()
            ).hexdigest(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "files": [
                {
                    "path": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(
                    temporary.iterdir(),
                    key=lambda item: item.name,
                )
                if path.is_file()
            ],
            "suite_passed": suite_passed,
            "scientific_qualification_granted": False,
            "real_world_action_authorized": False,
        }
        manifest["manifest_hash"] = sha256_value(manifest)
        _write_new(temporary / "MANIFEST.json", _json_bytes(manifest))
        os.rename(temporary, output)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-secret", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_suite(
        replay_secret_path=Path(args.replay_secret),
        output_dir=Path(args.output_dir),
    )
    print(canonical_json(result))
    return 0 if bool(result["suite_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
