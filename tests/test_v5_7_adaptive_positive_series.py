from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fma.hashing import canonical_json, sha256_value
from fma.v5.check_registry import (
    AdapterContextV50,
    CheckRegistryV50,
)
from fma.v5.workspace_schemas import (
    FileBindingV50,
    StageArtifactManifestV50,
    ValidationObligationV50,
)
from fma.v5_2.ode_system import ODETimeSeriesSnapshotV52
from fma.v5_6.hybrid_ode import HybridODEThresholdsV56
from fma.v5_7.adaptive_positive_series import (
    AdaptivePositiveSeriesLevelAdapterV57,
    AdaptiveReplayAuthorityV57,
    AdaptiveThresholdsV57,
    build_adaptive_positive_series_bundle_v57,
    register_adaptive_positive_series_adapters_v57,
    run_authenticated_adaptive_replays_v57,
)


ROOT = Path(__file__).resolve().parents[1]
V57_RAW_SHA256 = (
    "87fc0db0fb210b932c89e45e29b0ea73561303f354ae0f6f096c5c9879124baf"
)
V57_SEMANTIC_SHA256 = (
    "2beec5a8cedb8cc947ee1dd8e24ca1c5f0a6c87b00edd3d8f8c4d8019989db7f"
)


def _primary_thresholds() -> HybridODEThresholdsV56:
    return HybridODEThresholdsV56.seal(
        **json.loads(
            (ROOT / "V5_6_HYBRID_THRESHOLDS.json").read_text(
                encoding="utf-8"
            )
        )
    )


def _thresholds() -> AdaptiveThresholdsV57:
    return AdaptiveThresholdsV57.seal(
        **json.loads(
            (ROOT / "V5_7_ADAPTIVE_THRESHOLDS.json").read_text(
                encoding="utf-8"
            )
        )
    )


def _snapshot(task_id: str, values: np.ndarray) -> ODETimeSeriesSnapshotV52:
    return ODETimeSeriesSnapshotV52.seal(
        task_id=task_id,
        time_unit="year",
        state_unit="positive_index",
        times=np.arange(len(values), dtype=float).tolist(),
        observations=values.tolist(),
        source_id=f"{task_id}-fixture-source",
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
    return _snapshot("v57-ode-ar1", trend + residuals)


def _growth_snapshot(
    *,
    task_id: str,
    mean: float,
    phi: float,
    sigma: float,
    seed: int,
    structural_break_at: int | None = None,
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
    values = 100.0 * np.exp(np.concatenate(([0.0], np.cumsum(growths))))
    return _snapshot(task_id, values)


def test_v57_threshold_bytes_and_semantics_are_frozen() -> None:
    path = ROOT / "V5_7_ADAPTIVE_THRESHOLDS.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == V57_RAW_SHA256
    assert sha256_value(payload) == V57_SEMANTIC_SHA256
    thresholds = AdaptiveThresholdsV57.seal(**payload)
    thresholds.assert_sealed()


def test_v57_keeps_a_scientifically_valid_hybrid_ode_branch() -> None:
    bundle = build_adaptive_positive_series_bundle_v57(
        snapshot=_ode_ar1_snapshot(),
        primary_thresholds=_primary_thresholds(),
        adaptive_thresholds=_thresholds(),
    )
    assert bundle.graph.recovery_triggered is False
    assert bundle.graph.selected_branch == "hybrid_ode"
    assert bundle.graph.selected_model_id == "logistic.ar1_residual"
    assert bundle.growth_candidates == []
    assert [item.status for item in bundle.levels] == [
        "NOT_RUN",
        "PASS",
        "PASS",
        "PASS",
        "PASS",
    ]
    assert bundle.scientific_acceptance is False


def test_v57_recovers_stationary_log_drift_with_authenticated_replays(
    tmp_path: Path,
) -> None:
    snapshot = _growth_snapshot(
        task_id="v57-log-drift",
        mean=0.04,
        phi=0.0,
        sigma=0.01,
        seed=102,
    )
    primary = _primary_thresholds()
    adaptive = _thresholds()
    replay_input = tmp_path / "adaptive-replay.json"
    replay_input.write_text(
        canonical_json(
            {
                "snapshot": snapshot.model_dump(mode="json"),
                "primary_thresholds": primary.model_dump(mode="json"),
                "adaptive_thresholds": adaptive.model_dump(mode="json"),
            }
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    authority = AdaptiveReplayAuthorityV57(
        key_id="v57-fixture-replay",
        secret=b"a" * 32,
    )
    receipts = run_authenticated_adaptive_replays_v57(
        replay_input,
        authority=authority,
    )
    bundle = build_adaptive_positive_series_bundle_v57(
        snapshot=snapshot,
        primary_thresholds=primary,
        adaptive_thresholds=adaptive,
        replay_receipts=receipts,
        replay_authority=authority,
    )
    assert bundle.graph.recovery_triggered is True
    assert bundle.graph.selected_branch == "log_growth"
    assert bundle.graph.selected_model_id == "log_random_walk_drift"
    selected = next(
        item
        for item in bundle.growth_candidates
        if item.candidate_id == bundle.graph.selected_model_id
    )
    assert selected.scientifically_admissible is True
    assert selected.persistence_relative_improvement >= (
        adaptive.minimum_persistence_relative_improvement
    )
    assert len({item.process_id for item in receipts}) == 2
    assert len({item.deterministic_output_hash for item in receipts}) == 1
    assert all(authority.verify(item) for item in receipts)
    assert all(item.status == "PASS" for item in bundle.levels)
    assert bundle.scientific_acceptance is True
    assert bundle.fixture_only is True
    assert bundle.causal_mechanism_identified is False
    assert bundle.scientific_qualification_granted is False
    assert bundle.real_world_action_authorized is False


def test_v57_selects_stationary_ar1_growth_when_materially_better() -> None:
    bundle = build_adaptive_positive_series_bundle_v57(
        snapshot=_growth_snapshot(
            task_id="v57-log-growth-ar1",
            mean=0.04,
            phi=0.85,
            sigma=0.02,
            seed=103,
        ),
        primary_thresholds=_primary_thresholds(),
        adaptive_thresholds=_thresholds(),
    )
    assert bundle.graph.selected_branch == "log_growth"
    assert bundle.graph.selected_model_id == "log_growth_ar1"
    selected = next(
        item
        for item in bundle.growth_candidates
        if item.candidate_id == "log_growth_ar1"
    )
    assert selected.scientifically_admissible is True
    assert selected.process_fit.raw_phi <= (
        _thresholds().maximum_absolute_growth_ar1_phi
    )
    assert selected.same_family_ar1_relative_improvement >= (
        _thresholds().minimum_growth_ar1_validation_relative_improvement
    )


def test_v57_growth_structural_break_fails_closed() -> None:
    bundle = build_adaptive_positive_series_bundle_v57(
        snapshot=_growth_snapshot(
            task_id="v57-growth-break",
            mean=0.03,
            phi=0.0,
            sigma=0.008,
            seed=104,
            structural_break_at=45,
        ),
        primary_thresholds=_primary_thresholds(),
        adaptive_thresholds=_thresholds(),
    )
    assert bundle.graph.recovery_triggered is True
    assert bundle.graph.selected_branch == "unresolved"
    assert bundle.graph.admissible_recovery_candidate_ids == []
    assert bundle.levels[3].status == "FAIL"
    assert bundle.levels[4].status == "FAIL"
    assert bundle.scientific_acceptance is False
    assert all(
        not item.scientifically_admissible
        for item in bundle.growth_candidates
    )


def test_v57_growth_branch_is_positive_scale_invariant() -> None:
    original = _growth_snapshot(
        task_id="v57-scale-original",
        mean=0.04,
        phi=0.85,
        sigma=0.02,
        seed=103,
    )
    scaled = original.model_copy(
        update={
            "task_id": "v57-scale-transformed",
            "observations": [
                value * 1000.0 for value in original.observations
            ],
            "snapshot_hash": None,
        }
    )
    scaled = ODETimeSeriesSnapshotV52.seal(
        **scaled.model_dump(exclude={"snapshot_hash"})
    )
    first = build_adaptive_positive_series_bundle_v57(
        snapshot=original,
        primary_thresholds=_primary_thresholds(),
        adaptive_thresholds=_thresholds(),
    )
    second = build_adaptive_positive_series_bundle_v57(
        snapshot=scaled,
        primary_thresholds=_primary_thresholds(),
        adaptive_thresholds=_thresholds(),
    )
    assert first.graph.selected_branch == second.graph.selected_branch
    assert first.graph.selected_model_id == second.graph.selected_model_id
    first_selected = next(
        item
        for item in first.growth_candidates
        if item.candidate_id == first.graph.selected_model_id
    )
    second_selected = next(
        item
        for item in second.growth_candidates
        if item.candidate_id == second.graph.selected_model_id
    )
    assert np.isclose(
        first_selected.validation_relative_rmse,
        second_selected.validation_relative_rmse,
    )
    assert np.isclose(
        first_selected.process_fit.mean_log_growth,
        second_selected.process_fit.mean_log_growth,
    )
    assert np.isclose(
        first_selected.process_fit.raw_phi,
        second_selected.process_fit.raw_phi,
    )
    assert np.isclose(
        second_selected.forecast_value,
        first_selected.forecast_value * 1000.0,
    )


def test_v57_i35_is_development_only_recovery_not_qualification() -> None:
    snapshot = ODETimeSeriesSnapshotV52.model_validate_json(
        (
            ROOT
            / "experiments"
            / "iteration_35"
            / "campaign_unseen_v56"
            / "campaign_public_v55"
            / "public_snapshot_v52.json"
        ).read_text(encoding="utf-8")
    )
    bundle = build_adaptive_positive_series_bundle_v57(
        snapshot=snapshot,
        primary_thresholds=_primary_thresholds(),
        adaptive_thresholds=_thresholds(),
    )
    assert bundle.graph.recovery_triggered is True
    assert bundle.graph.selected_branch == "log_growth"
    assert bundle.graph.selected_model_id == "log_growth_ar1"
    assert [item.status for item in bundle.levels] == [
        "NOT_RUN",
        "PASS",
        "PASS",
        "PASS",
        "PASS",
    ]
    assert bundle.scientific_acceptance is False
    assert bundle.fixture_only is False
    assert bundle.causal_mechanism_identified is False
    assert bundle.scientific_qualification_granted is False
    assert bundle.real_world_action_authorized is False


def test_v57_adapter_reads_only_frozen_manifest(tmp_path: Path) -> None:
    bundle = build_adaptive_positive_series_bundle_v57(
        snapshot=_growth_snapshot(
            task_id="v57-adapter",
            mean=0.04,
            phi=0.0,
            sigma=0.01,
            seed=102,
        ),
        primary_thresholds=_primary_thresholds(),
        adaptive_thresholds=_thresholds(),
    )
    bundle_path = (
        tmp_path / "results" / "adaptive_positive_series_bundle.json"
    )
    bundle_path.parent.mkdir(parents=True)
    bundle_path.write_text(
        canonical_json(bundle) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    payload = bundle_path.read_bytes()
    manifest = StageArtifactManifestV50.seal(
        workspace_spec_hash="d" * 64,
        stage="S4",
        attempt=1,
        predecessor_gate_hash="e" * 64,
        files=[
            FileBindingV50(
                relative_path=(
                    "results/adaptive_positive_series_bundle.json"
                ),
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
                snapshot_artifact_hash="f" * 64,
            )
        ],
    )
    obligation = ValidationObligationV50(
        check_id="adaptive_positive_series_l3",
        stage="S4",
        level="L3",
        evidence_class="scientific_computation",
        applicability_rule="Adaptive positive-series bundle is present.",
    )
    context = AdapterContextV50(
        workspace_root=tmp_path,
        manifest=manifest,
        obligation=obligation,
    )
    outcome = AdaptivePositiveSeriesLevelAdapterV57("L3").run(context)
    assert outcome.status == "PASS"
    assert outcome.evidence_payloads[0]["causal_mechanism_identified"] is False

    bundle_path.write_text("{}\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="differs from frozen manifest"):
        AdaptivePositiveSeriesLevelAdapterV57("L3").run(context)

    registry = CheckRegistryV50()
    register_adaptive_positive_series_adapters_v57(registry)
    assert sorted(registry._adapters) == [
        "adaptive_positive_series_l0",
        "adaptive_positive_series_l1",
        "adaptive_positive_series_l2",
        "adaptive_positive_series_l3",
        "adaptive_positive_series_l4",
    ]
