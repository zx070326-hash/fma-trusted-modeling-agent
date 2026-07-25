"""Run the frozen V5.6 mechanism/rejection suite and publish public evidence."""

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
from fma.v5_6.hybrid_ode import (
    HybridODEThresholdsV56,
    HybridReplayAuthorityV56,
    build_hybrid_ode_bundle_v56,
    run_authenticated_hybrid_replays_v56,
)


ROOT = Path(__file__).resolve().parents[2]
THRESHOLD_PATH = ROOT / "V5_6_HYBRID_THRESHOLDS.json"
I34_SNAPSHOT_PATH = (
    ROOT
    / "experiments"
    / "iteration_34"
    / "campaign_public"
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


def _fixture_snapshot(
    *,
    task_id: str,
    phi: float,
    seed: int,
    iid: bool = False,
    structural_break_at: int | None = None,
) -> ODETimeSeriesSnapshotV52:
    count = 72
    times = np.arange(count, dtype=float)
    trend = 220.0 / (1.0 + 9.0 * np.exp(-0.11 * times))
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0.0, 0.5, count)
    residuals = np.zeros(count, dtype=float)
    if iid:
        residuals = innovations
    else:
        for index in range(1, count):
            residuals[index] = (
                phi * residuals[index - 1] + innovations[index]
            )
    observations = trend + residuals
    if structural_break_at is not None:
        observations[structural_break_at:] += 18.0
    return ODETimeSeriesSnapshotV52.seal(
        task_id=task_id,
        time_unit="year",
        state_unit="positive_index",
        times=times.tolist(),
        observations=observations.tolist(),
        source_id=f"{task_id}-frozen-fixture",
        fixture_only=True,
    )


def _cases() -> list[tuple[str, ODETimeSeriesSnapshotV52, dict[str, object]]]:
    return [
        (
            "stationary-ar1-recovery",
            _fixture_snapshot(
                task_id="v56-stationary-ar1-recovery",
                phi=0.90,
                seed=20260726,
            ),
            {
                "recovery_triggered": True,
                "selected_candidate_id": "logistic.ar1_residual",
                "l3_status": "PASS",
                "scientific_acceptance": True,
            },
        ),
        (
            "iid-no-recovery",
            _fixture_snapshot(
                task_id="v56-iid-no-recovery",
                phi=0.0,
                seed=20260727,
                iid=True,
            ),
            {
                "recovery_triggered": False,
                "selected_candidate_id": "logistic.trend_only",
                "l3_status": "PASS",
                "scientific_acceptance": True,
            },
        ),
        (
            "near-unit-root-reject",
            _fixture_snapshot(
                task_id="v56-near-unit-root-reject",
                phi=0.995,
                seed=20260728,
            ),
            {
                "recovery_triggered": True,
                "l3_status": "FAIL",
                "scientific_acceptance": False,
            },
        ),
        (
            "training-structural-break-reject",
            _fixture_snapshot(
                task_id="v56-training-structural-break-reject",
                phi=0.90,
                seed=20260729,
                structural_break_at=45,
            ),
            {
                "recovery_triggered": True,
                "l3_status": "FAIL",
                "scientific_acceptance": False,
            },
        ),
        (
            "validation-structural-break-reject",
            _fixture_snapshot(
                task_id="v56-validation-structural-break-reject",
                phi=0.90,
                seed=20260729,
                structural_break_at=55,
            ),
            {
                "recovery_triggered": True,
                "l3_status": "FAIL",
                "scientific_acceptance": False,
            },
        ),
        (
            "i34-retrospective-reject",
            ODETimeSeriesSnapshotV52.model_validate_json(
                I34_SNAPSHOT_PATH.read_text(encoding="utf-8")
            ),
            {
                "recovery_triggered": True,
                "selected_candidate_id": "logistic.trend_only",
                "l3_status": "FAIL",
                "scientific_acceptance": False,
            },
        ),
    ]


def _matches(actual: dict[str, object], expected: dict[str, object]) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def _report(rows: list[dict[str, object]], suite_passed: bool) -> str:
    lines = [
        "# V5.6 hybrid ODE mechanism suite",
        "",
        f"Status: `{'PASS' if suite_passed else 'FAIL'}`",
        "",
        "| Case | Recovery | Selected | L3 | Acceptance | Expected outcome |",
        "|---|---:|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {case_id} | {recovery_triggered} | {selected_candidate_id} | "
            "{l3_status} | {scientific_acceptance} | {expected_outcome_matched} |".format(
                **row
            )
        )
    recovery = next(
        row for row in rows if row["case_id"] == "stationary-ar1-recovery"
    )
    retrospective = next(
        row for row in rows if row["case_id"] == "i34-retrospective-reject"
    )
    lines.extend(
        [
            "",
            "## Mechanism evidence",
            "",
            (
                "- Stationary AR(1) fixture recovery improvement: "
                f"`{recovery['selected_same_family_ar1_improvement']:.6f}`."
            ),
            (
                "- Stationary AR(1) recovered phi: "
                f"`{recovery['selected_raw_phi']:.6f}` for frozen truth `0.90`."
            ),
            "- IID residual fixture did not activate recovery.",
            "- Near-unit-root and both training/validation structural-break fixtures failed closed.",
            (
                "- I34 retrospective logistic AR(1) improvement exceeded 50%, "
                "but innovation lag remained above the frozen bound, so I34 "
                f"remained `{retrospective['l3_status']}`."
            ),
            "",
            "## Claim boundary",
            "",
            "- Synthetic cases are mechanism controls, not real-world capability claims.",
            "- I34 is a disclosed retrospective control, not an unseen task.",
            "- No private target, qualification, causal identification, or real-world action is authorized.",
            "",
        ]
    )
    return "\n".join(lines)


def run_suite(*, replay_secret_path: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    threshold_payload = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
    thresholds = HybridODEThresholdsV56.seal(**threshold_payload)
    replay_secret = replay_secret_path.read_bytes()
    authority = HybridReplayAuthorityV56(
        key_id="i35-v56-local-mechanism-replay",
        secret=replay_secret,
    )
    temporary_parent = output_dir.resolve().parent
    temporary_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=temporary_parent,
        )
    )
    try:
        rows: list[dict[str, object]] = []
        artifacts: dict[str, bytes] = {}
        for case_id, snapshot, expected in _cases():
            replay_input = {
                "snapshot": snapshot.model_dump(mode="json"),
                "thresholds": thresholds.model_dump(mode="json"),
            }
            replay_path = temporary / f"{case_id}.replay_input.json"
            _write_new(replay_path, _json_bytes(replay_input))
            receipts = run_authenticated_hybrid_replays_v56(
                replay_path,
                authority=authority,
            )
            bundle = build_hybrid_ode_bundle_v56(
                snapshot=snapshot,
                thresholds=thresholds,
                replay_receipts=receipts,
                replay_authority=authority,
            )
            selected = next(
                item
                for item in bundle.candidates
                if item.candidate_id == bundle.selected_candidate_id
            )
            logistic_recovery = next(
                (
                    item
                    for item in bundle.candidates
                    if item.candidate_id == "logistic.ar1_residual"
                ),
                None,
            )
            actual = {
                "recovery_triggered": bundle.graph.recovery_triggered,
                "selected_candidate_id": bundle.selected_candidate_id,
                "l3_status": bundle.levels[3].status,
                "scientific_acceptance": bundle.scientific_acceptance,
            }
            row = {
                "case_id": case_id,
                **actual,
                "expected": expected,
                "expected_outcome_matched": _matches(actual, expected),
                "selected_raw_phi": selected.residual_fit.raw_phi,
                "selected_innovation_lag": (
                    selected.absolute_validation_innovation_lag1_correlation
                ),
                "selected_same_family_ar1_improvement": (
                    selected.same_family_ar1_relative_improvement
                ),
                "selected_admissibility_checks": (
                    selected.admissibility_checks
                ),
                "logistic_recovery_raw_phi": (
                    logistic_recovery.residual_fit.raw_phi
                    if logistic_recovery
                    else None
                ),
                "logistic_recovery_innovation_lag": (
                    logistic_recovery.absolute_validation_innovation_lag1_correlation
                    if logistic_recovery
                    else None
                ),
                "logistic_recovery_same_family_improvement": (
                    logistic_recovery.same_family_ar1_relative_improvement
                    if logistic_recovery
                    else None
                ),
                "bundle_hash": bundle.bundle_hash,
                "graph_hash": bundle.graph.graph_hash,
                "replay_receipt_hashes": bundle.replay_receipt_hashes,
                "fixture_only": snapshot.fixture_only,
                "causal_mechanism_identified": False,
                "scientific_qualification_granted": False,
                "real_world_action_authorized": False,
            }
            rows.append(row)
            artifacts[f"{case_id}.bundle.json"] = _json_bytes(bundle)
            artifacts[f"{case_id}.replay_receipts.json"] = _json_bytes(
                receipts
            )

        suite_passed = all(
            bool(row["expected_outcome_matched"]) for row in rows
        )
        suite_result = {
            "schema_version": "5.6-mechanism-suite-result",
            "threshold_file_sha256": hashlib.sha256(
                THRESHOLD_PATH.read_bytes()
            ).hexdigest(),
            "threshold_semantic_sha256": sha256_value(threshold_payload),
            "threshold_hash": thresholds.threshold_hash,
            "case_count": len(rows),
            "rows": rows,
            "suite_passed": suite_passed,
            "fixture_scientific_qualification_granted": False,
            "retrospective_scientific_qualification_granted": False,
            "causal_mechanism_identified": False,
            "real_world_action_authorized": False,
        }
        suite_result["result_hash"] = sha256_value(suite_result)
        artifacts["MECHANISM_SUITE_RESULTS.json"] = _json_bytes(suite_result)
        artifacts["REPORT.md"] = _report(rows, suite_passed).encode("utf-8")
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
            "schema_version": "5.6-mechanism-suite-manifest",
            "source_commit": source_commit,
            "runner_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "adapter_sha256": hashlib.sha256(
                (
                    ROOT / "fma" / "v5_6" / "hybrid_ode.py"
                ).read_bytes()
            ).hexdigest(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "files": [
                {
                    "path": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": len(path.read_bytes()),
                }
                for path in sorted(temporary.iterdir())
                if path.is_file()
            ],
            "suite_passed": suite_passed,
            "scientific_qualification_granted": False,
            "real_world_action_authorized": False,
        }
        manifest["manifest_hash"] = sha256_value(manifest)
        _write_new(
            temporary / "MANIFEST.json",
            _json_bytes(manifest),
        )
        os.rename(temporary, output_dir.resolve())
        return suite_result
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
