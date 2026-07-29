from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from fma.hashing import sha256_value
from fma.v5_2.ode_system import (
    ODEThresholdsV52,
    ODETimeSeriesSnapshotV52,
    build_ode_bundle_v52,
)
from fma.v6.executable_candidate import (
    SCALAR_ODE_ADAPTER_ID,
    SCALAR_ODE_FAMILIES_V62,
    ExecutableCandidateReceiptV62,
    allowed_family_registry_hash_v62,
)
from fma.v6.external_prediction_runtime import _scalar_ode_predictions
from fma.v6.external_qualification import (
    ExternalForecastInputV63,
    ExternalPredictionVectorV63,
    PredictiveExternalQualificationContractV63,
)
from fma.v6.predictive_quality import (
    freeze_predictive_quality_contract_v65,
)
from fma.v6.scalar_ode_uq import (
    MINIMUM_CALIBRATION_ORIGINS_V65,
    SCALAR_ODE_INTERVAL_ADAPTER_ID_V65,
    SCALAR_ODE_INTERVAL_PROTOCOL_HASH_V65,
    ScalarODEIntervalError,
    _finite_sample_quantile,
    calibrate_scalar_ode_intervals_v65,
    scalar_ode_interval_implementation_manifest_v65,
    verify_scalar_ode_intervals_v65,
)


T0 = datetime(2026, 7, 28, tzinfo=timezone.utc)


def _hash(label: str) -> str:
    return sha256_value({"label": label})


def _chain():
    times = np.arange(36, dtype=float)
    observations = (180.0 / (1.0 + 8.0 * np.exp(-0.16 * times))).tolist()
    snapshot = ODETimeSeriesSnapshotV52.seal(
        task_id="task.uq",
        time_unit="year",
        state_unit="positive_index",
        times=times.tolist(),
        observations=observations,
        source_id="public-training-series",
        fixture_only=True,
    )
    replay_hash = _hash("replay")
    bundle = build_ode_bundle_v52(
        snapshot=snapshot,
        thresholds=ODEThresholdsV52.seal(),
        replay_output_hashes=[replay_hash, replay_hash],
    )
    assert bundle.scientific_acceptance
    receipt = ExecutableCandidateReceiptV62.seal(
        workspace_spec_hash=_hash("workspace"),
        s1_gate_hash=_hash("s1"),
        s2_gate_hash=_hash("s2"),
        s2_attempt=1,
        resolution_hash=_hash("resolution"),
        selected_candidate_structural_hash=_hash("candidate"),
        adapter_id=SCALAR_ODE_ADAPTER_ID,
        allowed_families=list(SCALAR_ODE_FAMILIES_V62),
        allowed_family_registry_hash=allowed_family_registry_hash_v62(
            SCALAR_ODE_ADAPTER_ID
        ),
        evaluated_families=list(SCALAR_ODE_FAMILIES_V62),
        evaluated_model_ids=list(SCALAR_ODE_FAMILIES_V62),
        selected_family=bundle.selected_candidate_id,
        selected_model_id=bundle.selected_candidate_id,
        bundle_schema_version="5.2",
        bundle_task_id=bundle.task_id,
        bundle_hash=bundle.bundle_hash,
        candidate_registry_hash=bundle.candidate_registry_hash,
        candidate_graph_hash=None,
        nested_candidate_graph_hash=None,
        bundle_scientific_acceptance=True,
        fixture_only=True,
    )
    selected_identity = _hash("selected-identity")
    key_ids = {
        "custody": "custody-key",
        "registry": "registry-key",
        "evaluator": "evaluator-key",
        "promotion": "promotion-key",
    }
    fingerprints = {role: _hash(f"fingerprint-{role}") for role in key_ids}
    contract = PredictiveExternalQualificationContractV63.seal(
        qualification_id="qualification.uq",
        task_id=snapshot.task_id,
        local_context_hash=_hash("context"),
        workspace_spec_hash=receipt.workspace_spec_hash,
        v62_report_hash=_hash("v62"),
        s4_gate_hash=_hash("s4"),
        s6_gate_hash=_hash("s6"),
        scientific_bundle_hash=bundle.bundle_hash,
        processed_snapshot_hash=snapshot.snapshot_hash,
        executable_candidate_receipt_hash=receipt.receipt_hash,
        selected_model_id=receipt.selected_model_id,
        selected_model_identity_hash=selected_identity,
        maximum_metric_value=0.20,
        minimum_external_observation_count=3,
        trusted_authority_key_ids=key_ids,
        trusted_authority_key_fingerprints=fingerprints,
        trusted_authority_set_hash=sha256_value(
            {"key_ids": key_ids, "fingerprints": fingerprints}
        ),
        coordinator_host_id="coordinator-host",
        generator_host_id="generator-host",
        frozen_at=T0,
    )
    target_ids = ["target.1", "target.2", "target.3"]
    forecast_input = ExternalForecastInputV63.seal(
        qualification_id=contract.qualification_id,
        task_id=contract.task_id,
        contract_hash=contract.contract_hash,
        local_context_hash=contract.local_context_hash,
        processed_snapshot_hash=snapshot.snapshot_hash,
        target_ids=target_ids,
        target_order_hash=sha256_value(target_ids),
        forecast_times=[36.0, 37.0, 38.0],
        frozen_at=T0,
    )
    predictions = _scalar_ode_predictions(
        snapshot=snapshot,
        receipt=receipt,
        forecast_input=forecast_input,
    )
    vector = ExternalPredictionVectorV63.seal(
        qualification_id=contract.qualification_id,
        local_context_hash=contract.local_context_hash,
        selected_model_identity_hash=selected_identity,
        external_snapshot_hash=forecast_input.input_hash,
        target_ids=target_ids,
        target_order_hash=forecast_input.target_order_hash,
        predictions=predictions,
        prediction_values_hash=sha256_value(predictions),
    )
    quality_contract = freeze_predictive_quality_contract_v65(
        v63_contract=contract,
        quality_overlay_id="quality.uq",
        interval_alpha=0.10,
        minimum_mse_skill=0.10,
        minimum_interval_score_skill=0.05,
        minimum_empirical_coverage=0.75,
        maximum_absolute_coverage_error=0.15,
        maximum_normalized_interval_score=1.0,
        minimum_external_observation_count=3,
        interval_adapter_id=SCALAR_ODE_INTERVAL_ADAPTER_ID_V65,
        interval_adapter_protocol_hash=(SCALAR_ODE_INTERVAL_PROTOCOL_HASH_V65),
        interval_implementation_manifest=(
            scalar_ode_interval_implementation_manifest_v65()
        ),
        frozen_at=T0,
    )
    return (
        quality_contract,
        contract,
        snapshot,
        receipt,
        forecast_input,
        vector,
    )


def test_scalar_ode_uq_uses_training_only_origins_and_replays() -> None:
    (
        quality_contract,
        contract,
        snapshot,
        executable_receipt,
        forecast_input,
        vector,
    ) = _chain()

    calibration, pack = calibrate_scalar_ode_intervals_v65(
        quality_contract=quality_contract,
        v63_contract=contract,
        snapshot=snapshot,
        executable_receipt=executable_receipt,
        forecast_input=forecast_input,
        prediction_vector=vector,
        allow_fixture=True,
        generated_at=T0,
    )

    assert calibration.horizon_steps == [1, 2, 3]
    assert all(
        count >= MINIMUM_CALIBRATION_ORIGINS_V65
        for count in calibration.calibration_counts
    )
    assert calibration.persistence_baseline_point == snapshot.observations[-1]
    assert calibration.fixture_only is True
    assert pack.fixture_only is True
    assert calibration.interval_evidence_kind == (
        "rolling_origin_empirical_diagnostic"
    )
    assert calibration.finite_sample_coverage_guaranteed is False
    assert calibration.temporal_dependence_coverage_guaranteed is False
    assert calibration.post_selection_coverage_guaranteed is False
    assert calibration.model_selection_replayed_per_origin is False
    assert calibration.scientific_qualification_granted is False
    assert calibration.real_world_action_authorized is False
    assert calibration.prefix_fit_attempt_counts == (
        calibration.prefix_fit_success_counts
    )
    assert calibration.prefix_fit_failure_counts == [0, 0, 0]
    assert len(calibration.implementation_source_sha256) == 64
    assert len(calibration.runtime_identity_hash) == 64
    assert pack.interval_calibration_receipt_hash == calibration.receipt_hash
    assert all(
        low <= point <= high
        for low, point, high in zip(
            pack.lower_bounds,
            vector.predictions,
            pack.upper_bounds,
        )
    )
    assert "target_values" not in calibration.model_dump(mode="json")
    assert verify_scalar_ode_intervals_v65(
        receipt=calibration,
        quality_contract=quality_contract,
        v63_contract=contract,
        snapshot=snapshot,
        executable_receipt=executable_receipt,
        forecast_input=forecast_input,
        prediction_vector=vector,
        prediction_pack=pack,
        )


def test_scalar_ode_uq_requires_explicit_fixture_opt_in() -> None:
    (
        quality_contract,
        contract,
        snapshot,
        executable_receipt,
        forecast_input,
        vector,
    ) = _chain()

    with pytest.raises(
        ScalarODEIntervalError,
        match="explicit diagnostic opt-in",
    ):
        calibrate_scalar_ode_intervals_v65(
            quality_contract=quality_contract,
            v63_contract=contract,
            snapshot=snapshot,
            executable_receipt=executable_receipt,
            forecast_input=forecast_input,
            prediction_vector=vector,
            generated_at=T0,
        )


def test_empirical_quantile_fails_when_requested_rank_is_unavailable() -> None:
    with pytest.raises(
        ScalarODEIntervalError,
        match="too few public calibration origins",
    ):
        _finite_sample_quantile([float(index) for index in range(8)], 0.10)

    assert _finite_sample_quantile(
        [float(index) for index in range(9)],
        0.10,
    ) == 8.0


def test_prefix_fit_failure_is_not_silently_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_prefix_fit(**_: object) -> float:
        raise ScalarODEIntervalError("forced fit failure")

    monkeypatch.setattr(
        "fma.v6.scalar_ode_uq._prefix_forecast",
        fail_prefix_fit,
    )
    (
        quality_contract,
        contract,
        snapshot,
        executable_receipt,
        forecast_input,
        vector,
    ) = _chain()
    with pytest.raises(
        ScalarODEIntervalError,
        match=r"failed closed at horizon=1, origin=11",
    ):
        calibrate_scalar_ode_intervals_v65(
            quality_contract=quality_contract,
            v63_contract=contract,
            snapshot=snapshot,
            executable_receipt=executable_receipt,
            forecast_input=forecast_input,
            prediction_vector=vector,
            allow_fixture=True,
            generated_at=T0,
        )


def test_frozen_implementation_rejects_loaded_callable_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        quality_contract,
        contract,
        snapshot,
        executable_receipt,
        forecast_input,
        vector,
    ) = _chain()

    def substituted_prefix_forecast(**_: object) -> float:
        return 1.0

    monkeypatch.setattr(
        "fma.v6.scalar_ode_uq._prefix_forecast",
        substituted_prefix_forecast,
    )
    with pytest.raises(
        ScalarODEIntervalError,
        match="differs from the frozen manifest",
    ):
        calibrate_scalar_ode_intervals_v65(
            quality_contract=quality_contract,
            v63_contract=contract,
            snapshot=snapshot,
            executable_receipt=executable_receipt,
            forecast_input=forecast_input,
            prediction_vector=vector,
            allow_fixture=True,
            generated_at=T0,
        )


def test_scalar_ode_uq_rejects_a_tampered_point_vector() -> None:
    (
        quality_contract,
        contract,
        snapshot,
        executable_receipt,
        forecast_input,
        vector,
    ) = _chain()
    tampered = vector.model_copy(
        update={"predictions": [*vector.predictions[:-1], 999.0]}
    )

    with pytest.raises(ScalarODEIntervalError, match="not sealed"):
        calibrate_scalar_ode_intervals_v65(
            quality_contract=quality_contract,
            v63_contract=contract,
            snapshot=snapshot,
            executable_receipt=executable_receipt,
            forecast_input=forecast_input,
            prediction_vector=tampered,
            generated_at=T0,
        )
