from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

from fma.hashing import canonical_json, sha256_value
from fma.v5.scaffold import scaffold_task_workspace
from fma.v5.stage_workspace import StageWorkspaceV50
from fma.v5.workspace_schemas import TaskWorkspaceSpecV50, WorkflowProfileV50
from fma.v5_2.ode_system import (
    ODEThresholdsV52,
    ODETimeSeriesSnapshotV52,
    _parameter_vector,
    _predict,
    build_ode_bundle_v52,
    fit_ode_v52,
)
from fma.v5_6.hybrid_ode import HybridODEThresholdsV56
from fma.v5_7.adaptive_positive_series import (
    AdaptiveReplayAuthorityV57,
    AdaptiveThresholdsV57,
    _estimate_growth_process,
    build_adaptive_positive_series_bundle_v57,
    run_authenticated_adaptive_replays_v57,
)
from fma.v6.executable_candidate import (
    ADAPTIVE_POSITIVE_SERIES_ADAPTER_ID,
    ADAPTIVE_POSITIVE_SERIES_FAMILIES_V62,
    EXECUTABLE_CANDIDATE_RECEIPT_PATH,
    SCALAR_ODE_ADAPTER_ID,
    SCALAR_ODE_FAMILIES_V62,
    ExecutableCandidateReceiptV62,
    allowed_family_registry_hash_v62,
)
import fma.v6.external_qualification as qualification
from fma.v6.external_prediction_runtime import (
    ExternalPredictionRuntimeError,
    run_current_model_external_prediction_v63,
    verify_current_model_external_prediction_v63,
)
from fma.v6.provenance import PROCESSED_SNAPSHOT_PATH
from fma.v6.scientific_closure import (
    ADAPTIVE_SCIENTIFIC_BUNDLE_ADMISSION_PATH,
    ODE_SCIENTIFIC_BUNDLE_ADMISSION_PATH,
)


UTC = timezone.utc
T0 = datetime(2026, 7, 20, tzinfo=UTC)
AUTHORITY_KEY = b"v63-prediction-runtime-authority-key-0001"
AUTHORITY_KEY_ID = "v63-prediction-runtime-authority"
GATES = {
    "S4": sha256_value({"runtime-test-gate": "S4"}),
    "S6": sha256_value({"runtime-test-gate": "S6"}),
}
ROOT = Path(__file__).resolve().parents[1]


def _key_pair() -> tuple[bytes, bytes]:
    private = Ed25519PrivateKey.generate()
    return (
        private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ),
        private.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ),
    )


def _keys() -> tuple[dict[str, bytes], dict[str, bytes]]:
    private_keys: dict[str, bytes] = {}
    public_keys: dict[str, bytes] = {}
    for role in ("custody", "registry", "evaluator", "promotion"):
        private_key, public_key = _key_pair()
        private_keys[f"{role}-key"] = private_key
        public_keys[f"{role}-key"] = public_key
    return private_keys, public_keys


def _write_model(path: Path, model: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _new_workspace(root: Path) -> StageWorkspaceV50:
    scaffold_task_workspace(
        root,
        "v63-prediction-runtime",
        "Generate a current-model V6.3 prediction from public evidence",
    )
    spec = TaskWorkspaceSpecV50.seal(
        workspace_id="v63-prediction-runtime",
        graph_id="v5-v63-prediction-runtime",
        objective=(
            "Generate a current-model V6.3 prediction from public evidence"
        ),
        mission_hash=sha256_value({"mission": "v63-prediction-runtime"}),
        evidence_snapshot_hash=sha256_value(
            {"evidence": "v63-prediction-runtime"}
        ),
        evaluator_epoch="v63-prediction-runtime-v1",
        profile=WorkflowProfileV50.seal(),
        evidence_scope="development",
        created_at=T0,
    )
    return StageWorkspaceV50.create(
        root,
        spec,
        authority_key=AUTHORITY_KEY,
        authority_key_id=AUTHORITY_KEY_ID,
    )


def _install_real_ode_model(
    workspace: StageWorkspaceV50,
) -> tuple[
    ODETimeSeriesSnapshotV52,
    Any,
    ExecutableCandidateReceiptV62,
    dict[str, Any],
]:
    times = np.arange(36, dtype=float)
    observations = (
        180.0 / (1.0 + 8.0 * np.exp(-0.16 * times))
    ).tolist()
    snapshot = ODETimeSeriesSnapshotV52.seal(
        task_id=workspace.spec.workspace_id,
        time_unit="day",
        state_unit="positive_index",
        times=times.tolist(),
        observations=observations,
        source_id="public-observational-series",
        fixture_only=False,
    )
    replay_hash = sha256_value({"fresh-public-replay": "identical"})
    bundle = build_ode_bundle_v52(
        snapshot=snapshot,
        thresholds=ODEThresholdsV52.seal(),
        replay_output_hashes=[replay_hash, replay_hash],
    )
    assert bundle.scientific_acceptance
    assert bundle.selected_candidate_id == "logistic"
    structural_hash = sha256_value({"candidate": "registered-ode-search"})
    receipt = ExecutableCandidateReceiptV62.seal(
        workspace_spec_hash=workspace.spec.spec_hash,
        s1_gate_hash=sha256_value({"gate": "S1"}),
        s2_gate_hash=sha256_value({"gate": "S2"}),
        s2_attempt=1,
        resolution_hash=sha256_value({"resolution": "scalar-ode"}),
        selected_candidate_structural_hash=structural_hash,
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
        fixture_only=False,
    )
    _write_model(workspace.root / PROCESSED_SNAPSHOT_PATH, snapshot)
    _write_model(
        workspace.root / ODE_SCIENTIFIC_BUNDLE_ADMISSION_PATH,
        bundle,
    )
    _write_model(
        workspace.root / EXECUTABLE_CANDIDATE_RECEIPT_PATH,
        receipt,
    )
    selected_model_identity_hash = sha256_value(
        {
            "selected_model_id": receipt.selected_model_id,
            "selected_candidate_structural_hash": structural_hash,
            "executable_candidate_receipt_hash": receipt.receipt_hash,
            "scientific_bundle_hash": bundle.bundle_hash,
        }
    )
    summary = {
        "schema_version": "6.2",
        "evaluated": True,
        "source_integrity_status": "PASS",
        "scientific_provenance_status": "HUMAN",
        "stage_admission_status": "PASS",
        "closure_verification_status": "PASS",
        "local_evidence_status": "PASS",
        "scientific_closure_status": "NOT_RUN",
        "claim_ceiling": "local_leakage_safe_predictive_evidence",
        "claim_kind": "predictive",
        "fixture_only": False,
        "dimensions": {
            "workflow_integrity": {"status": "PASS"},
            "local_adapter_checks": {"status": "PASS"},
            "leakage_safe_confirmation": {"status": "PASS"},
        },
        "report_hash": sha256_value({"artifact": "v62-report"}),
        "admission_hash": sha256_value({"artifact": "v62-admission"}),
        "verification_hash": sha256_value({"artifact": "v62-verification"}),
        "scientific_bundle_hash": bundle.bundle_hash,
        "processed_snapshot_hash": snapshot.snapshot_hash,
        "executable_candidate_receipt_hash": receipt.receipt_hash,
        "selected_model_id": receipt.selected_model_id,
        "selected_model_identity_hash": selected_model_identity_hash,
        "scientific_qualification_granted": False,
        "real_world_action_authorized": False,
    }
    return snapshot, bundle, receipt, summary


def _install_real_adaptive_model(
    workspace: StageWorkspaceV50,
    replay_root: Path,
) -> tuple[
    ODETimeSeriesSnapshotV52,
    Any,
    ExecutableCandidateReceiptV62,
    dict[str, Any],
]:
    rng = np.random.default_rng(102)
    growths = np.zeros(71, dtype=float)
    growths[0] = 0.04
    for index in range(1, len(growths)):
        growths[index] = 0.04 + rng.normal(0.0, 0.01)
    observations = 100.0 * np.exp(
        np.concatenate(([0.0], np.cumsum(growths)))
    )
    snapshot = ODETimeSeriesSnapshotV52.seal(
        task_id=workspace.spec.workspace_id,
        time_unit="day",
        state_unit="positive_index",
        times=np.arange(len(observations), dtype=float).tolist(),
        observations=observations.tolist(),
        source_id="public-positive-growth-series",
        fixture_only=False,
    )
    primary_thresholds = HybridODEThresholdsV56.seal(
        **json.loads(
            (ROOT / "V5_6_HYBRID_THRESHOLDS.json").read_text(
                encoding="utf-8"
            )
        )
    )
    adaptive_thresholds = AdaptiveThresholdsV57.seal(
        **json.loads(
            (ROOT / "V5_7_ADAPTIVE_THRESHOLDS.json").read_text(
                encoding="utf-8"
            )
        )
    )
    replay_path = replay_root / "adaptive-replay-input.json"
    replay_path.write_text(
        canonical_json(
            {
                "snapshot": snapshot.model_dump(mode="json"),
                "primary_thresholds": primary_thresholds.model_dump(
                    mode="json"
                ),
                "adaptive_thresholds": adaptive_thresholds.model_dump(
                    mode="json"
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    replay_authority = AdaptiveReplayAuthorityV57(
        key_id="v63-adaptive-runtime-replay",
        secret=b"v63-adaptive-runtime-replay-key" * 2,
    )
    replay_receipts = run_authenticated_adaptive_replays_v57(
        replay_path,
        authority=replay_authority,
    )
    bundle = build_adaptive_positive_series_bundle_v57(
        snapshot=snapshot,
        primary_thresholds=primary_thresholds,
        adaptive_thresholds=adaptive_thresholds,
        replay_receipts=replay_receipts,
        replay_authority=replay_authority,
    )
    assert bundle.scientific_acceptance
    assert bundle.fixture_only is False
    assert bundle.graph.selected_branch == "log_growth"
    structural_hash = sha256_value(
        {"candidate": "registered-adaptive-positive-series-search"}
    )
    primary_model_ids = [
        item.candidate_id for item in bundle.primary_bundle.candidates
    ]
    growth_model_ids = [
        item.candidate_id for item in bundle.growth_candidates
    ]
    evaluated_families = sorted(
        {
            *(item.family for item in bundle.primary_bundle.candidates),
            *growth_model_ids,
        }
    )
    receipt = ExecutableCandidateReceiptV62.seal(
        workspace_spec_hash=workspace.spec.spec_hash,
        s1_gate_hash=sha256_value({"gate": "S1"}),
        s2_gate_hash=sha256_value({"gate": "S2"}),
        s2_attempt=1,
        resolution_hash=sha256_value({"resolution": "adaptive-positive"}),
        selected_candidate_structural_hash=structural_hash,
        adapter_id=ADAPTIVE_POSITIVE_SERIES_ADAPTER_ID,
        allowed_families=list(ADAPTIVE_POSITIVE_SERIES_FAMILIES_V62),
        allowed_family_registry_hash=allowed_family_registry_hash_v62(
            ADAPTIVE_POSITIVE_SERIES_ADAPTER_ID
        ),
        evaluated_families=evaluated_families,
        evaluated_model_ids=sorted(
            [*primary_model_ids, *growth_model_ids]
        ),
        selected_family=bundle.graph.selected_model_id,
        selected_model_id=bundle.graph.selected_model_id,
        bundle_schema_version="5.7",
        bundle_task_id=bundle.task_id,
        bundle_hash=bundle.bundle_hash,
        candidate_registry_hash=(
            bundle.primary_bundle.candidate_registry_hash
        ),
        candidate_graph_hash=bundle.graph.graph_hash,
        nested_candidate_graph_hash=(
            bundle.primary_bundle.graph.graph_hash
        ),
        bundle_scientific_acceptance=True,
        fixture_only=False,
    )
    _write_model(workspace.root / PROCESSED_SNAPSHOT_PATH, snapshot)
    _write_model(
        workspace.root / ADAPTIVE_SCIENTIFIC_BUNDLE_ADMISSION_PATH,
        bundle,
    )
    _write_model(
        workspace.root / EXECUTABLE_CANDIDATE_RECEIPT_PATH,
        receipt,
    )
    selected_model_identity_hash = sha256_value(
        {
            "selected_model_id": receipt.selected_model_id,
            "selected_candidate_structural_hash": structural_hash,
            "executable_candidate_receipt_hash": receipt.receipt_hash,
            "scientific_bundle_hash": bundle.bundle_hash,
        }
    )
    summary = {
        "schema_version": "6.2",
        "evaluated": True,
        "source_integrity_status": "PASS",
        "scientific_provenance_status": "HUMAN",
        "stage_admission_status": "PASS",
        "closure_verification_status": "PASS",
        "local_evidence_status": "PASS",
        "scientific_closure_status": "NOT_RUN",
        "claim_ceiling": "local_leakage_safe_predictive_evidence",
        "claim_kind": "predictive",
        "fixture_only": False,
        "dimensions": {
            "workflow_integrity": {"status": "PASS"},
            "local_adapter_checks": {"status": "PASS"},
            "leakage_safe_confirmation": {"status": "PASS"},
        },
        "report_hash": sha256_value({"artifact": "v62-adaptive-report"}),
        "admission_hash": sha256_value(
            {"artifact": "v62-adaptive-admission"}
        ),
        "verification_hash": sha256_value(
            {"artifact": "v62-adaptive-verification"}
        ),
        "scientific_bundle_hash": bundle.bundle_hash,
        "processed_snapshot_hash": snapshot.snapshot_hash,
        "executable_candidate_receipt_hash": receipt.receipt_hash,
        "selected_model_id": receipt.selected_model_id,
        "selected_model_identity_hash": selected_model_identity_hash,
        "scientific_qualification_granted": False,
        "real_world_action_authorized": False,
    }
    return snapshot, bundle, receipt, summary


def _freeze_and_admit(
    *,
    workspace: StageWorkspaceV50,
    summary: dict[str, Any],
    private_keys: dict[str, bytes],
    public_keys: dict[str, bytes],
    monkeypatch: pytest.MonkeyPatch,
    forecast_start: float = 36.0,
) -> tuple[Any, Any, Any]:
    monkeypatch.setattr(
        qualification,
        "scientific_closure_summary_v62",
        lambda _workspace: summary,
    )
    monkeypatch.setattr(
        StageWorkspaceV50,
        "current_gate",
        lambda _workspace, stage: GATES.get(stage),
    )
    contract = (
        qualification.freeze_predictive_external_qualification_contract_v63(
            workspace=workspace,
            qualification_id="qual.prediction-runtime",
            task_id=workspace.spec.workspace_id,
            maximum_metric_value=0.20,
            minimum_external_observation_count=12,
            coordinator_host_id="coordinator-host",
            generator_host_id="generator-host",
            custody_key_id="custody-key",
            registry_key_id="registry-key",
            evaluator_key_id="evaluator-key",
            promotion_key_id="promotion-key",
            trusted_public_keys=public_keys,
            frozen_at=T0,
        )
    )
    target_ids = [f"target.{index:02d}" for index in range(12)]
    forecast_input = qualification.commit_external_forecast_input_v63(
        workspace=workspace,
        contract=contract,
        target_ids=target_ids,
        forecast_times=[forecast_start + index for index in range(12)],
        frozen_at=T0 + timedelta(minutes=1),
    )
    custody = qualification.sign_external_evidence_custody_v63(
        private_key_pem=private_keys["custody-key"],
        qualification_id=contract.qualification_id,
        contract_hash=contract.contract_hash,
        local_context_hash=contract.local_context_hash,
        v62_report_hash=contract.v62_report_hash,
        external_snapshot_hash=forecast_input.input_hash,
        holdout_commitment_hash=sha256_value({"private": "holdout"}),
        normalization_scale_commitment_hash=sha256_value(
            {
                "holdout_observation_count": 12,
                "target_squared_value_sum": 1200.0,
            }
        ),
        target_order_hash=forecast_input.target_order_hash,
        holdout_observation_count=12,
        fixture_only=False,
        measurement_protocol_hash=sha256_value(
            {"measurement": "protocol"}
        ),
        measurement_review_hash=sha256_value(
            {"review": "independent"}
        ),
        external_environment_hash=sha256_value(
            {"environment": "external"}
        ),
        strict_unseen_verified=True,
        independent_measurement_review_passed=True,
        external_environment_verified=True,
        holdout_frozen_before_prediction=True,
        custodian_host_id="custodian-host",
        coordinator_host_id=contract.coordinator_host_id,
        generator_host_id=contract.generator_host_id,
        attested_at=T0 + timedelta(minutes=2),
        custody_key_id="custody-key",
    )
    admission = qualification.admit_external_evidence_custody_v63(
        workspace=workspace,
        contract=contract,
        custody=custody,
        trusted_public_keys=public_keys,
    )
    assert admission.status == "VERIFIED"
    return contract, forecast_input, custody


def _setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    StageWorkspaceV50,
    ODETimeSeriesSnapshotV52,
    Any,
    ExecutableCandidateReceiptV62,
    Any,
    Any,
    Any,
]:
    workspace = _new_workspace(tmp_path / "task")
    snapshot, bundle, receipt, summary = _install_real_ode_model(workspace)
    private_keys, public_keys = _keys()
    contract, forecast_input, custody = _freeze_and_admit(
        workspace=workspace,
        summary=summary,
        private_keys=private_keys,
        public_keys=public_keys,
        monkeypatch=monkeypatch,
    )
    return (
        workspace,
        snapshot,
        bundle,
        receipt,
        contract,
        forecast_input,
        custody,
    )


def test_runtime_generates_exact_current_scalar_ode_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        workspace,
        snapshot,
        _bundle,
        receipt,
        contract,
        forecast_input,
        custody,
    ) = _setup(tmp_path, monkeypatch)

    result = run_current_model_external_prediction_v63(
        workspace=workspace,
        contract=contract,
        forecast_input=forecast_input,
        custody=custody,
    )

    fit = fit_ode_v52(
        receipt.selected_family,
        np.asarray(snapshot.times, dtype=float),
        np.asarray(snapshot.observations, dtype=float),
    )
    expected = _predict(
        receipt.selected_family,
        np.asarray(
            [snapshot.times[0], *forecast_input.forecast_times],
            dtype=float,
        ),
        snapshot.observations[0],
        _parameter_vector(fit),
    )[1:].tolist()
    assert result.resumed is False
    assert result.prediction_vector.predictions == pytest.approx(expected)
    assert result.prediction_vector.target_ids == forecast_input.target_ids
    assert result.prediction_vector.external_snapshot_hash == (
        forecast_input.input_hash
    )
    assert result.execution_receipt.output_artifact_hash == (
        result.binding.prediction_artifact_hash
    )
    assert result.binding.private_holdout_targets_accessed is False
    trace = workspace._artifact_payload_by_hash(
        result.execution_receipt.transport_trace_hash
    )
    assert trace["external_io_performed"] is False
    assert trace["private_holdout_targets_accessed"] is False
    assert "private_target_values" not in forecast_input.model_dump(mode="json")


def test_runtime_is_idempotent_and_restartable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        workspace,
        _snapshot,
        _bundle,
        _receipt,
        contract,
        forecast_input,
        custody,
    ) = _setup(tmp_path, monkeypatch)
    first = run_current_model_external_prediction_v63(
        workspace=workspace,
        contract=contract,
        forecast_input=forecast_input,
        custody=custody,
    )
    reopened = StageWorkspaceV50.open_existing(
        workspace.root,
        authority_key=AUTHORITY_KEY,
        authority_key_id=AUTHORITY_KEY_ID,
    )

    second = run_current_model_external_prediction_v63(
        workspace=reopened,
        contract=contract,
        forecast_input=forecast_input,
        custody=custody,
    )
    verified = verify_current_model_external_prediction_v63(
        workspace=reopened,
        contract=contract,
        forecast_input=forecast_input,
        custody=custody,
    )

    assert second.resumed is True
    assert verified.resumed is True
    assert second.binding == first.binding == verified.binding
    assert second.prediction_vector == first.prediction_vector
    assert len(
        reopened._artifacts_of_kind("external_prediction_vector_v63")
    ) == 1
    assert len(
        reopened._artifacts_of_kind(
            "current_model_prediction_binding_v63"
        )
    ) == 1
    assert len(
        [
            item
            for _, item in reopened._artifacts_of_kind(
                "role_execution_receipt_v50"
            )
            if item["execution_id"].startswith("v63-prediction-")
        ]
    ) == 1


def test_replay_rejects_authority_signed_but_wrong_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        workspace,
        _snapshot,
        _bundle,
        _receipt,
        contract,
        forecast_input,
        custody,
    ) = _setup(tmp_path, monkeypatch)
    wrong_values = [1.0 + index for index in range(12)]
    wrong_vector = qualification.ExternalPredictionVectorV63.seal(
        qualification_id=contract.qualification_id,
        local_context_hash=contract.local_context_hash,
        selected_model_identity_hash=contract.selected_model_identity_hash,
        external_snapshot_hash=forecast_input.input_hash,
        target_ids=forecast_input.target_ids,
        target_order_hash=forecast_input.target_order_hash,
        predictions=wrong_values,
        prediction_values_hash=sha256_value(wrong_values),
    )
    vector_ref = workspace.commit_evidence(
        "external_prediction_vector_v63",
        wrong_vector.model_dump(mode="json"),
    )
    input_authority_hash = sha256_value(
        {
            "contract_hash": contract.contract_hash,
            "local_context_hash": contract.local_context_hash,
            "scientific_bundle_hash": contract.scientific_bundle_hash,
            "processed_snapshot_hash": contract.processed_snapshot_hash,
            "executable_candidate_receipt_hash": (
                contract.executable_candidate_receipt_hash
            ),
            "selected_model_identity_hash": (
                contract.selected_model_identity_hash
            ),
            "external_snapshot_hash": forecast_input.input_hash,
            "holdout_observation_count": len(forecast_input.target_ids),
        }
    )
    trace_ref = workspace.commit_evidence(
        "adversarial_prediction_trace_v63",
        {
            "role": "modeler",
            "subject_id": contract.task_id,
            "input_authority_hash": input_authority_hash,
            "run_id": "adversarial-run",
            "context_id": "adversarial-context",
        },
    )
    role_receipt = workspace.issue_role_execution(
        stage="S4",
        execution_id="adversarial-prediction",
        role="modeler",
        subject_id=contract.task_id,
        input_authority_hash=input_authority_hash,
        run_id="adversarial-run",
        context_id="adversarial-context",
        provider="authority-test",
        model="wrong-but-signed",
        prompt_hash=sha256_value({"prompt": "wrong prediction"}),
        output_schema_hash=contract.prediction_output_schema_hash,
        transport_trace_hash=trace_ref.sha256,
        output_artifact_hash=vector_ref.sha256,
    )
    qualification.issue_current_model_prediction_binding_v63(
        workspace=workspace,
        contract=contract,
        custody=custody,
        prediction_vector=wrong_vector,
        generator_execution_receipt=role_receipt,
    )

    with pytest.raises(
        ExternalPredictionRuntimeError,
        match="differs from deterministic replay",
    ):
        verify_current_model_external_prediction_v63(
            workspace=workspace,
            contract=contract,
            forecast_input=forecast_input,
            custody=custody,
        )
    with pytest.raises(
        ExternalPredictionRuntimeError,
        match="differs from deterministic replay",
    ):
        run_current_model_external_prediction_v63(
            workspace=workspace,
            contract=contract,
            forecast_input=forecast_input,
            custody=custody,
        )


def test_runtime_supports_registered_adaptive_log_growth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _new_workspace(tmp_path / "adaptive-task")
    snapshot, bundle, receipt, summary = _install_real_adaptive_model(
        workspace,
        tmp_path,
    )
    private_keys, public_keys = _keys()
    contract, forecast_input, custody = _freeze_and_admit(
        workspace=workspace,
        summary=summary,
        private_keys=private_keys,
        public_keys=public_keys,
        monkeypatch=monkeypatch,
        forecast_start=float(snapshot.times[-1] + 1.0),
    )

    result = run_current_model_external_prediction_v63(
        workspace=workspace,
        contract=contract,
        forecast_input=forecast_input,
        custody=custody,
    )

    selected = next(
        item
        for item in bundle.growth_candidates
        if item.candidate_id == receipt.selected_model_id
    )
    public_values = np.asarray(snapshot.observations, dtype=float)
    public_growths = np.diff(np.log(public_values))
    full_fit, _innovations = _estimate_growth_process(
        selected.mode,
        public_growths,
    )
    first_growth = (
        full_fit.mean_log_growth
        + full_fit.effective_phi
        * (public_growths[-1] - full_fit.mean_log_growth)
    )
    expected_first = float(public_values[-1] * np.exp(first_growth))
    assert bundle.graph.selected_branch == "log_growth"
    assert result.prediction_vector.predictions[0] == pytest.approx(
        expected_first
    )
    assert len(result.prediction_vector.predictions) == 12
    assert all(
        value > 0 for value in result.prediction_vector.predictions
    )
    assert verify_current_model_external_prediction_v63(
        workspace=workspace,
        contract=contract,
        forecast_input=forecast_input,
        custody=custody,
    ).binding == result.binding


def test_runtime_fails_closed_on_current_model_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        workspace,
        _snapshot,
        _bundle,
        receipt,
        contract,
        forecast_input,
        custody,
    ) = _setup(tmp_path, monkeypatch)
    substituted_payload = receipt.model_dump(exclude={"receipt_hash"})
    substituted_payload["bundle_hash"] = sha256_value(
        {"substituted": "bundle"}
    )
    substituted = ExecutableCandidateReceiptV62.seal(**substituted_payload)
    _write_model(
        workspace.root / EXECUTABLE_CANDIDATE_RECEIPT_PATH,
        substituted,
    )

    with pytest.raises(
        ExternalPredictionRuntimeError,
        match="executable model differs",
    ):
        run_current_model_external_prediction_v63(
            workspace=workspace,
            contract=contract,
            forecast_input=forecast_input,
            custody=custody,
        )
