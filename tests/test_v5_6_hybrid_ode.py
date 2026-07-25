from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fma.hashing import canonical_json, sha256_value
from fma.v5.check_registry import AdapterContextV50, CheckRegistryV50
from fma.v5.workspace_schemas import (
    FileBindingV50,
    StageArtifactManifestV50,
    ValidationObligationV50,
)
from fma.v5_2.ode_system import ODETimeSeriesSnapshotV52
from fma.v5_6.hybrid_ode import (
    HybridODELevelAdapterV56,
    HybridODEThresholdsV56,
    HybridReplayAuthorityV56,
    build_hybrid_ode_bundle_v56,
    register_hybrid_ode_adapters_v56,
    run_authenticated_hybrid_replays_v56,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN_THRESHOLDS = ROOT / "V5_6_HYBRID_THRESHOLDS.json"


def _thresholds() -> HybridODEThresholdsV56:
    return HybridODEThresholdsV56.seal(
        **json.loads(FROZEN_THRESHOLDS.read_text(encoding="utf-8"))
    )


def _snapshot(
    *,
    task_id: str,
    phi: float,
    seed: int,
    iid: bool = False,
    structural_break_at: int | None = None,
    structural_break_size: float = 18.0,
    time_scale: float = 1.0,
    time_shift: float = 0.0,
    state_scale: float = 1.0,
) -> ODETimeSeriesSnapshotV52:
    count = 72
    base_times = np.arange(count, dtype=float)
    trend = 220.0 / (1.0 + 9.0 * np.exp(-0.11 * base_times))
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
    values = trend + residuals
    if structural_break_at is not None:
        values[structural_break_at:] += structural_break_size
    return ODETimeSeriesSnapshotV52.seal(
        task_id=task_id,
        time_unit="year",
        state_unit="positive_index",
        times=(base_times * time_scale + time_shift).tolist(),
        observations=(values * state_scale).tolist(),
        source_id=f"{task_id}-fixture",
        fixture_only=True,
    )


def _ar1_snapshot() -> ODETimeSeriesSnapshotV52:
    return _snapshot(
        task_id="hybrid-logistic-ar1",
        phi=0.90,
        seed=20260726,
    )


def test_frozen_threshold_file_has_declared_hashes_and_seals() -> None:
    payload = FROZEN_THRESHOLDS.read_bytes()
    parsed = json.loads(payload)
    assert (
        hashlib.sha256(payload).hexdigest()
        == "7ac2797aab17d89abfc7855a293249cae6ae3661d5f4276ac64139dcd6165866"
    )
    assert (
        sha256_value(parsed)
        == "56b022cfe6ee5ccc3b1f534c2e038b651c7862d323bf81fe60a03fbac057eb00"
    )
    thresholds = _thresholds()
    thresholds.assert_sealed()
    with pytest.raises(ValueError, match="not sealed"):
        thresholds.model_copy(
            update={"maximum_absolute_ar1_phi": 0.90}
        ).assert_sealed()


def test_correlated_residual_triggers_recovery_and_authenticated_l0(
    tmp_path: Path,
) -> None:
    snapshot = _ar1_snapshot()
    thresholds = _thresholds()
    replay_input = tmp_path / "hybrid-replay.json"
    replay_input.write_text(
        canonical_json(
            {
                "snapshot": snapshot.model_dump(mode="json"),
                "thresholds": thresholds.model_dump(mode="json"),
            }
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    authority = HybridReplayAuthorityV56(
        key_id="fixture-hybrid-replay",
        secret=b"h" * 32,
    )
    receipts = run_authenticated_hybrid_replays_v56(
        replay_input,
        authority=authority,
    )
    bundle = build_hybrid_ode_bundle_v56(
        snapshot=snapshot,
        thresholds=thresholds,
        replay_receipts=receipts,
        replay_authority=authority,
    )

    assert bundle.graph.recovery_triggered is True
    assert bundle.graph.initial_selected_candidate_id == "logistic.trend_only"
    assert bundle.selected_candidate_id == "logistic.ar1_residual"
    selected = next(
        item
        for item in bundle.candidates
        if item.candidate_id == bundle.selected_candidate_id
    )
    assert selected.scientifically_admissible is True
    assert abs(selected.residual_fit.raw_phi - 0.90) <= 0.25
    assert (
        selected.absolute_validation_innovation_lag1_correlation
        <= thresholds.maximum_innovation_absolute_lag1_correlation
    )
    assert (
        selected.same_family_ar1_relative_improvement
        >= thresholds.minimum_ar1_validation_relative_improvement
    )
    same_family_trend = next(
        item
        for item in bundle.candidates
        if item.candidate_id == "logistic.trend_only"
    )
    assert selected.validation_rmse < same_family_trend.validation_rmse
    assert same_family_trend.scientifically_admissible is False
    assert len({item.process_id for item in receipts}) == 2
    assert len({item.deterministic_output_hash for item in receipts}) == 1
    assert all(authority.verify(item) for item in receipts)
    assert all(item.status == "PASS" for item in bundle.levels)
    assert bundle.scientific_acceptance is True
    assert bundle.fixture_only is True
    assert bundle.causal_mechanism_identified is False
    assert bundle.scientific_qualification_granted is False
    assert bundle.real_world_action_authorized is False

    forged = receipts[0].model_copy(
        update={"deterministic_output_hash": "0" * 64}
    )
    assert authority.verify(forged) is False


def test_iid_residual_does_not_activate_unnecessary_recovery() -> None:
    bundle = build_hybrid_ode_bundle_v56(
        snapshot=_snapshot(
            task_id="hybrid-logistic-iid",
            phi=0.0,
            seed=20260727,
            iid=True,
        ),
        thresholds=_thresholds(),
    )
    assert bundle.graph.recovery_triggered is False
    assert len(bundle.candidates) == 4
    assert bundle.selected_candidate_id == "logistic.trend_only"
    assert bundle.levels[3].status == "PASS"
    assert bundle.levels[0].status == "NOT_RUN"
    assert bundle.scientific_acceptance is False


def test_near_unit_root_residual_fails_closed() -> None:
    bundle = build_hybrid_ode_bundle_v56(
        snapshot=_snapshot(
            task_id="hybrid-near-unit-root",
            phi=0.995,
            seed=20260728,
        ),
        thresholds=_thresholds(),
    )
    assert bundle.graph.recovery_triggered is True
    recovery = [
        item
        for item in bundle.candidates
        if item.residual_mode == "ar1_residual"
    ]
    assert len(recovery) == 4
    assert not any(item.scientifically_admissible for item in recovery)
    assert any(
        not item.admissibility_checks["ar1_stationary_interior"]
        for item in recovery
    )
    assert bundle.levels[3].status == "FAIL"
    assert bundle.scientific_acceptance is False


def test_structural_break_guard_prevents_guardless_false_selection() -> None:
    bundle = build_hybrid_ode_bundle_v56(
        snapshot=_snapshot(
            task_id="hybrid-structural-break",
            phi=0.90,
            seed=20260729,
            structural_break_at=45,
        ),
        thresholds=_thresholds(),
    )
    guardless_best = sorted(
        bundle.candidates,
        key=lambda item: (
            -item.validation_score,
            item.parameter_count,
            item.candidate_id,
        ),
    )[0]
    assert guardless_best.residual_mode == "ar1_residual"
    assert guardless_best.scientifically_admissible is False
    assert any(
        not guardless_best.admissibility_checks[name]
        for name in (
            "ar1_stationary_interior",
            "innovation_mean_shift_bounded",
            "single_innovation_bounded",
        )
    )
    assert bundle.graph.recovery_triggered is True
    assert bundle.levels[3].status == "FAIL"
    assert bundle.scientific_acceptance is False


def test_dimensionless_fit_is_invariant_to_positive_affine_units() -> None:
    original = build_hybrid_ode_bundle_v56(
        snapshot=_ar1_snapshot(),
        thresholds=_thresholds(),
    )
    transformed = build_hybrid_ode_bundle_v56(
        snapshot=_snapshot(
            task_id="hybrid-logistic-ar1-transformed",
            phi=0.90,
            seed=20260726,
            time_scale=7.0,
            time_shift=1000.0,
            state_scale=13.0,
        ),
        thresholds=_thresholds(),
    )
    assert original.selected_candidate_id == transformed.selected_candidate_id
    original_selected = next(
        item
        for item in original.candidates
        if item.candidate_id == original.selected_candidate_id
    )
    transformed_selected = next(
        item
        for item in transformed.candidates
        if item.candidate_id == transformed.selected_candidate_id
    )
    assert np.allclose(
        original_selected.trend_fit.dimensionless_parameter_values,
        transformed_selected.trend_fit.dimensionless_parameter_values,
        rtol=1e-6,
        atol=1e-8,
    )
    assert np.isclose(
        original_selected.trend_fit.dimensionless_parameter_condition_number,
        transformed_selected.trend_fit.dimensionless_parameter_condition_number,
        rtol=1e-5,
    )
    assert np.isclose(
        original_selected.residual_fit.raw_phi,
        transformed_selected.residual_fit.raw_phi,
        rtol=1e-6,
        atol=1e-8,
    )


def test_hybrid_adapter_reads_only_frozen_manifest(tmp_path: Path) -> None:
    bundle = build_hybrid_ode_bundle_v56(
        snapshot=_ar1_snapshot(),
        thresholds=_thresholds(),
    )
    bundle_path = tmp_path / "results" / "hybrid_ode_scientific_bundle.json"
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
                relative_path="results/hybrid_ode_scientific_bundle.json",
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
                snapshot_artifact_hash="f" * 64,
            )
        ],
    )
    obligation = ValidationObligationV50(
        check_id="scalar_hybrid_ode_l3",
        stage="S4",
        level="L3",
        evidence_class="scientific_computation",
        applicability_rule="Hybrid scalar ODE bundle is present.",
    )
    context = AdapterContextV50(
        workspace_root=tmp_path,
        manifest=manifest,
        obligation=obligation,
    )
    outcome = HybridODELevelAdapterV56("L3").run(context)
    assert outcome.status == "PASS"
    assert outcome.evidence_payloads[0]["causal_mechanism_identified"] is False

    bundle_path.write_text("{}\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="differs from frozen manifest"):
        HybridODELevelAdapterV56("L3").run(context)

    registry = CheckRegistryV50()
    register_hybrid_ode_adapters_v56(registry)
    assert sorted(registry._adapters) == [
        "scalar_hybrid_ode_l0",
        "scalar_hybrid_ode_l1",
        "scalar_hybrid_ode_l2",
        "scalar_hybrid_ode_l3",
        "scalar_hybrid_ode_l4",
    ]


def test_i34_retrospective_control_remains_rejected() -> None:
    snapshot = ODETimeSeriesSnapshotV52.model_validate_json(
        (
            ROOT
            / "experiments"
            / "iteration_34"
            / "campaign_public"
            / "public_snapshot_v52.json"
        ).read_text(encoding="utf-8")
    )
    bundle = build_hybrid_ode_bundle_v56(
        snapshot=snapshot,
        thresholds=_thresholds(),
    )
    assert bundle.graph.recovery_triggered is True
    assert bundle.graph.initial_selected_candidate_id == "logistic.trend_only"
    assert bundle.graph.admissible_candidate_ids == []
    assert bundle.selected_candidate_id == "logistic.trend_only"
    logistic_recovery = next(
        item
        for item in bundle.candidates
        if item.candidate_id == "logistic.ar1_residual"
    )
    assert (
        logistic_recovery.same_family_ar1_relative_improvement
        is not None
        and logistic_recovery.same_family_ar1_relative_improvement > 0.50
    )
    assert (
        logistic_recovery.absolute_validation_innovation_lag1_correlation
        > _thresholds().maximum_innovation_absolute_lag1_correlation
    )
    assert logistic_recovery.scientifically_admissible is False
    assert bundle.levels[3].status == "FAIL"
    assert bundle.scientific_acceptance is False
    assert bundle.scientific_qualification_granted is False
