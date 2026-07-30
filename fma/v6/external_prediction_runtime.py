"""Pure-local V6.3 prediction runtime for the current V6.2 model.

This module is deliberately narrower than the external qualification
protocol.  It reads only the current public V6.2 snapshot, executable
candidate receipt, scientific bundle, and the target coordinates frozen in
``ExternalForecastInputV63``.  It never accepts or reads private target
values.

The runtime is restartable.  Stable operation identities and exact artifact
replay allow a retry to recover after any committed intermediate artifact.
Conflicting artifacts fail closed instead of selecting the latest one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v5.stage_workspace import StageWorkspaceError, StageWorkspaceV50
from fma.v5.workspace_schemas import RoleExecutionReceiptV50
from fma.v5_2.ode_system import (
    ODEScientificBundleV52,
    ODETimeSeriesSnapshotV52,
    _parameter_vector,
    _predict,
    fit_ode_v52,
)
from fma.v5_6.hybrid_ode import (
    _estimate_residual_process,
    _fit_trend,
    _forecast_correction,
    _trend_predict,
)
from fma.v5_7.adaptive_positive_series import (
    AdaptivePositiveSeriesBundleV57,
    _estimate_growth_process,
)

from . import executable_candidate as executable
from .executable_candidate import (
    ADAPTIVE_POSITIVE_SERIES_ADAPTER_ID,
    EXECUTABLE_CANDIDATE_RECEIPT_PATH,
    SCALAR_ODE_ADAPTER_ID,
    ExecutableCandidateReceiptV62,
)
from . import external_qualification as qualification
from .external_qualification import (
    CurrentModelPredictionBindingV63,
    ExternalEvidenceCustodyV63,
    ExternalForecastInputV63,
    ExternalPredictionVectorV63,
    ExternalQualificationError,
    PredictiveExternalQualificationContractV63,
    issue_current_model_prediction_binding_v63,
)
from .provenance import PROCESSED_SNAPSHOT_PATH
from .scientific_closure import (
    ADAPTIVE_SCIENTIFIC_BUNDLE_ADMISSION_PATH,
    ODE_SCIENTIFIC_BUNDLE_ADMISSION_PATH,
)


PREDICTION_VECTOR_KIND_V63 = "external_prediction_vector_v63"
PREDICTION_BINDING_KIND_V63 = "current_model_prediction_binding_v63"
PREDICTION_TRACE_KIND_V63 = "external_prediction_runtime_trace_v63"
PREDICTION_RUNTIME_ADAPTER_ID_V63 = (
    "current_v62_selected_positive_series_forecast_v63"
)
PREDICTION_RUNTIME_PROTOCOL_HASH_V63 = sha256_value(
    {
        "schema_version": "6.3-external-prediction-runtime-protocol",
        "runtime": PREDICTION_RUNTIME_ADAPTER_ID_V63,
        "input_policy": (
            "current_public_v62_snapshot_plus_frozen_future_coordinates"
        ),
        "private_target_values_permitted": False,
        "supported_adapters": [
            SCALAR_ODE_ADAPTER_ID,
            ADAPTIVE_POSITIVE_SERIES_ADAPTER_ID,
        ],
        "selected_family_refit": "full_public_snapshot",
        "adaptive_horizon_policy": "positive_integer_public_cadence_steps",
    }
)


class ExternalPredictionRuntimeError(ExternalQualificationError):
    """Raised when deterministic V6.3 prediction generation fails closed."""


@dataclass(frozen=True)
class ExternalPredictionRuntimeResultV63:
    """Non-authoritative in-process view of the committed prediction chain."""

    forecast_input: ExternalForecastInputV63
    prediction_vector: ExternalPredictionVectorV63
    execution_receipt: RoleExecutionReceiptV50
    binding: CurrentModelPredictionBindingV63
    resumed: bool


def _read_model(path: Path, model_type: type[StrictModel]) -> StrictModel:
    if not path.is_file():
        raise ExternalPredictionRuntimeError(
            f"required current-model artifact is missing: {path.name}"
        )
    try:
        return model_type.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExternalPredictionRuntimeError(
            f"required current-model artifact is invalid: {path.name}"
        ) from exc


def _one_committed(
    *,
    workspace: StageWorkspaceV50,
    kind: str,
    model_type: type[Any] | None,
    predicate: Any,
    label: str,
) -> tuple[Any, Any]:
    try:
        matches = [
            (reference, item)
            for reference, item in workspace._artifacts_of_kind(
                kind,
                model_type,
            )
            if predicate(item)
        ]
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ExternalPredictionRuntimeError(
            f"{label} ledger could not be replayed"
        ) from exc
    if len(matches) != 1:
        raise ExternalPredictionRuntimeError(
            f"{label} is absent, duplicated, or ambiguous"
        )
    return matches[0]


def _current_forecast_input(
    *,
    workspace: StageWorkspaceV50,
    contract: PredictiveExternalQualificationContractV63,
    supplied: ExternalForecastInputV63,
) -> ExternalForecastInputV63:
    try:
        supplied.assert_sealed()
    except ValueError as exc:
        raise ExternalPredictionRuntimeError(
            "supplied external forecast input is not sealed"
        ) from exc
    _reference, committed = _one_committed(
        workspace=workspace,
        kind="external_forecast_input_v63",
        model_type=ExternalForecastInputV63,
        predicate=lambda item: (
            item.qualification_id == contract.qualification_id
            or item.task_id == contract.task_id
        ),
        label="external forecast input",
    )
    if committed != supplied:
        raise ExternalPredictionRuntimeError(
            "supplied external forecast input differs from committed input"
        )
    if (
        committed.qualification_id != contract.qualification_id
        or committed.task_id != contract.task_id
        or committed.contract_hash != contract.contract_hash
        or committed.local_context_hash != contract.local_context_hash
        or committed.processed_snapshot_hash
        != contract.processed_snapshot_hash
        or committed.input_hash is None
    ):
        raise ExternalPredictionRuntimeError(
            "external forecast input differs from the frozen contract"
        )
    return committed


def _load_current_model(
    *,
    workspace: StageWorkspaceV50,
    contract: PredictiveExternalQualificationContractV63,
) -> tuple[
    ODETimeSeriesSnapshotV52,
    ExecutableCandidateReceiptV62,
    ODEScientificBundleV52 | AdaptivePositiveSeriesBundleV57,
]:
    snapshot = cast(
        ODETimeSeriesSnapshotV52,
        _read_model(
            workspace.root / PROCESSED_SNAPSHOT_PATH,
            ODETimeSeriesSnapshotV52,
        ),
    )
    receipt = cast(
        ExecutableCandidateReceiptV62,
        _read_model(
            workspace.root / EXECUTABLE_CANDIDATE_RECEIPT_PATH,
            ExecutableCandidateReceiptV62,
        ),
    )
    try:
        snapshot.assert_sealed()
        receipt.assert_sealed()
    except ValueError as exc:
        raise ExternalPredictionRuntimeError(
            "current V6.2 snapshot or executable receipt is not sealed"
        ) from exc
    if (
        snapshot.snapshot_hash != contract.processed_snapshot_hash
        or receipt.receipt_hash
        != contract.executable_candidate_receipt_hash
        or receipt.bundle_hash != contract.scientific_bundle_hash
        or receipt.selected_model_id != contract.selected_model_id
        or receipt.bundle_task_id != snapshot.task_id
        or receipt.fixture_only
        or not receipt.bundle_scientific_acceptance
    ):
        raise ExternalPredictionRuntimeError(
            "current V6.2 executable model differs from the frozen contract"
        )

    if receipt.adapter_id == SCALAR_ODE_ADAPTER_ID:
        bundle = cast(
            ODEScientificBundleV52,
            _read_model(
                workspace.root / ODE_SCIENTIFIC_BUNDLE_ADMISSION_PATH,
                ODEScientificBundleV52,
            ),
        )
    elif receipt.adapter_id == ADAPTIVE_POSITIVE_SERIES_ADAPTER_ID:
        bundle = cast(
            AdaptivePositiveSeriesBundleV57,
            _read_model(
                workspace.root / ADAPTIVE_SCIENTIFIC_BUNDLE_ADMISSION_PATH,
                AdaptivePositiveSeriesBundleV57,
            ),
        )
    else:
        raise ExternalPredictionRuntimeError(
            f"unsupported current V6.2 adapter: {receipt.adapter_id}"
        )
    try:
        bundle_hash = cast(str | None, bundle.bundle_hash)
        if bundle_hash is None or bundle_hash != bundle.content_hash():
            raise ValueError("scientific bundle is not sealed")
        observed = executable._observe_bundle(
            adapter_id=receipt.adapter_id,
            bundle=bundle,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ExternalPredictionRuntimeError(
            "current V6.2 scientific bundle failed executable replay"
        ) from exc
    observed_bindings = {
        "bundle_schema_version": receipt.bundle_schema_version,
        "task_id": receipt.bundle_task_id,
        "bundle_hash": receipt.bundle_hash,
        "candidate_registry_hash": receipt.candidate_registry_hash,
        "candidate_graph_hash": receipt.candidate_graph_hash,
        "nested_candidate_graph_hash": receipt.nested_candidate_graph_hash,
        "evaluated_families": receipt.evaluated_families,
        "evaluated_model_ids": receipt.evaluated_model_ids,
        "selected_family": receipt.selected_family,
        "selected_model_id": receipt.selected_model_id,
        "scientific_acceptance": receipt.bundle_scientific_acceptance,
        "fixture_only": receipt.fixture_only,
    }
    if observed.model_dump(mode="json") != observed_bindings:
        raise ExternalPredictionRuntimeError(
            "executable receipt differs from the current scientific bundle"
        )
    if (
        bundle.snapshot_hash != snapshot.snapshot_hash
        or bundle.task_id != snapshot.task_id
        or snapshot.fixture_only
        or bundle.fixture_only
    ):
        raise ExternalPredictionRuntimeError(
            "scientific bundle differs from the current public snapshot"
        )
    return snapshot, receipt, bundle


def _scalar_ode_predictions(
    *,
    snapshot: ODETimeSeriesSnapshotV52,
    receipt: ExecutableCandidateReceiptV62,
    forecast_input: ExternalForecastInputV63,
) -> list[float]:
    if any(time <= snapshot.times[-1] for time in forecast_input.forecast_times):
        raise ExternalPredictionRuntimeError(
            "scalar ODE forecast coordinates are not strictly future"
        )
    times = np.asarray(snapshot.times, dtype=float)
    values = np.asarray(snapshot.observations, dtype=float)
    try:
        fit = fit_ode_v52(
            cast(Any, receipt.selected_family),
            times,
            values,
        )
        if not fit.optimizer_converged:
            raise ValueError("full-public selected-family refit did not converge")
        predictions = _predict(
            cast(Any, receipt.selected_family),
            np.concatenate(
                (
                    np.asarray([times[0]], dtype=float),
                    np.asarray(forecast_input.forecast_times, dtype=float),
                )
            ),
            float(values[0]),
            _parameter_vector(fit),
        )[1:]
    except (ArithmeticError, TypeError, ValueError) as exc:
        raise ExternalPredictionRuntimeError(
            "scalar ODE full-public refit or forecast failed"
        ) from exc
    result = [float(value) for value in predictions]
    if any(not math.isfinite(value) or value <= 0 for value in result):
        raise ExternalPredictionRuntimeError(
            "scalar ODE prediction is not positive finite"
        )
    return result


def _cadence_steps(
    *,
    snapshot: ODETimeSeriesSnapshotV52,
    forecast_input: ExternalForecastInputV63,
) -> list[int]:
    times = np.asarray(snapshot.times, dtype=float)
    cadence = float(np.median(np.diff(times)))
    if not math.isfinite(cadence) or cadence <= 0:
        raise ExternalPredictionRuntimeError(
            "adaptive public-series cadence is invalid"
        )
    steps: list[int] = []
    for target_time in forecast_input.forecast_times:
        raw_step = (float(target_time) - float(times[-1])) / cadence
        step = int(round(raw_step))
        if step < 1 or abs(raw_step - step) > 1e-9:
            raise ExternalPredictionRuntimeError(
                "adaptive forecast coordinate is off the public cadence"
            )
        steps.append(step)
    if steps != sorted(set(steps)):
        raise ExternalPredictionRuntimeError(
            "adaptive forecast horizons must be strictly increasing"
        )
    return steps


def _adaptive_predictions(
    *,
    snapshot: ODETimeSeriesSnapshotV52,
    receipt: ExecutableCandidateReceiptV62,
    bundle: AdaptivePositiveSeriesBundleV57,
    forecast_input: ExternalForecastInputV63,
) -> list[float]:
    if (
        bundle.graph.selected_model_id != receipt.selected_model_id
        or bundle.graph.selected_branch == "unresolved"
    ):
        raise ExternalPredictionRuntimeError(
            "adaptive selected branch is absent or unresolved"
        )
    steps = _cadence_steps(
        snapshot=snapshot,
        forecast_input=forecast_input,
    )
    times = np.asarray(snapshot.times, dtype=float)
    values = np.asarray(snapshot.observations, dtype=float)
    try:
        if bundle.graph.selected_branch == "hybrid_ode":
            selected = next(
                item
                for item in bundle.primary_bundle.candidates
                if item.candidate_id == receipt.selected_model_id
            )
            trend = _fit_trend(selected.family, times, values)
            if not trend.optimizer_converged:
                raise ValueError("adaptive trend refit did not converge")
            residuals = values - _trend_predict(trend, times)
            process, _innovations = _estimate_residual_process(
                selected.residual_mode,
                residuals,
            )
            predictions = (
                _trend_predict(
                    trend,
                    np.asarray(forecast_input.forecast_times, dtype=float),
                )
                + _forecast_correction(
                    last_residual=float(residuals[-1]),
                    phi=process.effective_phi,
                    horizon_steps=np.asarray(steps, dtype=int),
                )
            ).tolist()
        else:
            selected_growth = next(
                item
                for item in bundle.growth_candidates
                if item.candidate_id == receipt.selected_model_id
            )
            growths = np.diff(np.log(values))
            process, _innovations = _estimate_growth_process(
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
    except (ArithmeticError, StopIteration, TypeError, ValueError) as exc:
        raise ExternalPredictionRuntimeError(
            "adaptive full-public refit or forecast failed"
        ) from exc
    result = [float(value) for value in predictions]
    if any(not math.isfinite(value) or value <= 0 for value in result):
        raise ExternalPredictionRuntimeError(
            "adaptive prediction is not positive finite"
        )
    return result


def _input_authority_hash(
    *,
    contract: PredictiveExternalQualificationContractV63,
    forecast_input: ExternalForecastInputV63,
) -> str:
    return sha256_value(
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


def _stable_execution_identity(
    *,
    contract: PredictiveExternalQualificationContractV63,
    forecast_input: ExternalForecastInputV63,
) -> tuple[str, str, str]:
    operation_hash = sha256_value(
        {
            "schema_version": "6.3-external-prediction-operation",
            "contract_hash": contract.contract_hash,
            "forecast_input_hash": forecast_input.input_hash,
            "runtime_protocol_hash": PREDICTION_RUNTIME_PROTOCOL_HASH_V63,
        }
    )
    suffix = operation_hash[:20]
    return (
        f"v63-prediction-{suffix}",
        f"v63-run-{suffix}",
        f"v63-context-{suffix}",
    )


def _commit_or_recover_vector(
    *,
    workspace: StageWorkspaceV50,
    contract: PredictiveExternalQualificationContractV63,
    vector: ExternalPredictionVectorV63,
) -> tuple[ExternalPredictionVectorV63, str]:
    try:
        prior = [
            (reference, item)
            for reference, item in workspace._artifacts_of_kind(
                PREDICTION_VECTOR_KIND_V63,
                ExternalPredictionVectorV63,
            )
            if item.qualification_id == contract.qualification_id
        ]
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ExternalPredictionRuntimeError(
            "prediction-vector ledger could not be replayed"
        ) from exc
    exact = [
        (reference, item)
        for reference, item in prior
        if item == vector
    ]
    if exact:
        if len(prior) != 1 or len(exact) != 1:
            raise ExternalPredictionRuntimeError(
                "prediction-vector ledger contains duplicates"
            )
        return exact[0][1], exact[0][0].sha256
    if prior:
        raise ExternalPredictionRuntimeError(
            "a conflicting prediction vector already exists"
        )
    reference = workspace.commit_evidence(
        PREDICTION_VECTOR_KIND_V63,
        vector.model_dump(mode="json"),
    )
    return vector, reference.sha256


def _commit_or_recover_trace(
    *,
    workspace: StageWorkspaceV50,
    contract: PredictiveExternalQualificationContractV63,
    trace: dict[str, object],
) -> str:
    try:
        prior = [
            (reference, item)
            for reference, item in workspace._artifacts_of_kind(
                PREDICTION_TRACE_KIND_V63,
            )
            if isinstance(item, dict)
            and item.get("qualification_id") == contract.qualification_id
        ]
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ExternalPredictionRuntimeError(
            "prediction-runtime trace ledger could not be replayed"
        ) from exc
    exact = [reference for reference, item in prior if item == trace]
    if exact:
        if len(prior) != 1 or len(exact) != 1:
            raise ExternalPredictionRuntimeError(
                "prediction-runtime trace ledger contains duplicates"
            )
        return exact[0].sha256
    if prior:
        raise ExternalPredictionRuntimeError(
            "a conflicting prediction-runtime trace already exists"
        )
    return workspace.commit_evidence(
        PREDICTION_TRACE_KIND_V63,
        trace,
    ).sha256


def _issue_or_recover_role_receipt(
    *,
    workspace: StageWorkspaceV50,
    contract: PredictiveExternalQualificationContractV63,
    execution_id: str,
    run_id: str,
    context_id: str,
    input_authority_hash: str,
    transport_trace_hash: str,
    prediction_artifact_hash: str,
) -> RoleExecutionReceiptV50:
    try:
        prior = [
            item
            for _reference, item in workspace._artifacts_of_kind(
                "role_execution_receipt_v50",
                RoleExecutionReceiptV50,
            )
            if item.execution_id == execution_id
        ]
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ExternalPredictionRuntimeError(
            "prediction role-receipt ledger could not be replayed"
        ) from exc
    if prior:
        if len(prior) != 1:
            raise ExternalPredictionRuntimeError(
                "prediction role-receipt ledger contains duplicates"
            )
        receipt = prior[0]
        if (
            not workspace.verify_role_execution(receipt)
            or receipt.stage != "S4"
            or receipt.role != "modeler"
            or receipt.subject_id != contract.task_id
            or receipt.input_authority_hash != input_authority_hash
            or receipt.run_id != run_id
            or receipt.context_id != context_id
            or receipt.provider != "fma-harness"
            or receipt.model != PREDICTION_RUNTIME_ADAPTER_ID_V63
            or receipt.prompt_hash
            != PREDICTION_RUNTIME_PROTOCOL_HASH_V63
            or receipt.output_schema_hash
            != contract.prediction_output_schema_hash
            or receipt.transport_trace_hash != transport_trace_hash
            or receipt.output_artifact_hash != prediction_artifact_hash
        ):
            raise ExternalPredictionRuntimeError(
                "existing prediction role receipt conflicts with this run"
            )
        return receipt
    receipt = workspace.issue_role_execution(
        stage="S4",
        execution_id=execution_id,
        role="modeler",
        subject_id=contract.task_id,
        input_authority_hash=input_authority_hash,
        run_id=run_id,
        context_id=context_id,
        provider="fma-harness",
        model=PREDICTION_RUNTIME_ADAPTER_ID_V63,
        prompt_hash=PREDICTION_RUNTIME_PROTOCOL_HASH_V63,
        output_schema_hash=contract.prediction_output_schema_hash,
        transport_trace_hash=transport_trace_hash,
        output_artifact_hash=prediction_artifact_hash,
    )
    if not workspace.verify_role_execution(receipt):
        raise ExternalPredictionRuntimeError(
            "new prediction role receipt failed authority verification"
        )
    return receipt


def _runtime_trace_payload(
    *,
    contract: PredictiveExternalQualificationContractV63,
    forecast_input: ExternalForecastInputV63,
    snapshot: ODETimeSeriesSnapshotV52,
    executable_receipt: ExecutableCandidateReceiptV62,
    vector: ExternalPredictionVectorV63,
    vector_artifact_hash: str,
    execution_id: str,
    run_id: str,
    context_id: str,
    input_authority_hash: str,
) -> dict[str, object]:
    return {
        "schema_version": "6.3-external-prediction-runtime-trace",
        "qualification_id": contract.qualification_id,
        "role": "modeler",
        "subject_id": contract.task_id,
        "input_authority_hash": input_authority_hash,
        "run_id": run_id,
        "context_id": context_id,
        "execution_id": execution_id,
        "runtime_adapter_id": PREDICTION_RUNTIME_ADAPTER_ID_V63,
        "runtime_protocol_hash": PREDICTION_RUNTIME_PROTOCOL_HASH_V63,
        "contract_hash": contract.contract_hash,
        "forecast_input_hash": forecast_input.input_hash,
        "processed_snapshot_hash": snapshot.snapshot_hash,
        "executable_candidate_receipt_hash": executable_receipt.receipt_hash,
        "scientific_bundle_hash": executable_receipt.bundle_hash,
        "selected_model_id": executable_receipt.selected_model_id,
        "prediction_vector_hash": vector.vector_hash,
        "prediction_artifact_hash": vector_artifact_hash,
        "external_io_performed": False,
        "private_holdout_targets_accessed": False,
        "real_world_action_authorized": False,
    }


def _recover_completed(
    *,
    workspace: StageWorkspaceV50,
    contract: PredictiveExternalQualificationContractV63,
    custody: ExternalEvidenceCustodyV63,
    forecast_input: ExternalForecastInputV63,
    snapshot: ODETimeSeriesSnapshotV52,
    executable_receipt: ExecutableCandidateReceiptV62,
    expected_vector: ExternalPredictionVectorV63,
) -> ExternalPredictionRuntimeResultV63 | None:
    try:
        bindings = [
            item
            for _reference, item in workspace._artifacts_of_kind(
                PREDICTION_BINDING_KIND_V63,
                CurrentModelPredictionBindingV63,
            )
            if item.qualification_id == contract.qualification_id
        ]
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ExternalPredictionRuntimeError(
            "prediction-binding ledger could not be replayed"
        ) from exc
    if not bindings:
        return None
    if len(bindings) != 1:
        raise ExternalPredictionRuntimeError(
            "prediction-binding ledger contains duplicates"
        )
    binding = bindings[0]
    try:
        vectors = [
            (reference, item)
            for reference, item in workspace._artifacts_of_kind(
                PREDICTION_VECTOR_KIND_V63,
                ExternalPredictionVectorV63,
            )
            if item.qualification_id == contract.qualification_id
        ]
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ExternalPredictionRuntimeError(
            "prediction-vector ledger could not be replayed"
        ) from exc
    if (
        len(vectors) != 1
        or vectors[0][1] != expected_vector
        or vectors[0][1].vector_hash != binding.prediction_vector_hash
        or vectors[0][0].sha256 != binding.prediction_artifact_hash
    ):
        raise ExternalPredictionRuntimeError(
            "committed prediction vector differs from deterministic replay"
        )
    vector = vectors[0][1]
    _receipt_reference, receipt = _one_committed(
        workspace=workspace,
        kind="role_execution_receipt_v50",
        model_type=RoleExecutionReceiptV50,
        predicate=lambda item: (
            item.receipt_hash == binding.generator_execution_receipt_hash
        ),
        label="bound prediction role receipt",
    )
    input_authority_hash = _input_authority_hash(
        contract=contract,
        forecast_input=forecast_input,
    )
    execution_id, run_id, context_id = _stable_execution_identity(
        contract=contract,
        forecast_input=forecast_input,
    )
    expected_trace = _runtime_trace_payload(
        contract=contract,
        forecast_input=forecast_input,
        snapshot=snapshot,
        executable_receipt=executable_receipt,
        vector=vector,
        vector_artifact_hash=vectors[0][0].sha256,
        execution_id=execution_id,
        run_id=run_id,
        context_id=context_id,
        input_authority_hash=input_authority_hash,
    )
    trace_reference, trace = _one_committed(
        workspace=workspace,
        kind=PREDICTION_TRACE_KIND_V63,
        model_type=None,
        predicate=lambda item: (
            item.get("qualification_id") == contract.qualification_id
        ),
        label="bound prediction runtime trace",
    )
    if trace != expected_trace or trace_reference.sha256 != (
        receipt.transport_trace_hash
    ):
        raise ExternalPredictionRuntimeError(
            "prediction runtime trace differs from deterministic replay"
        )
    if (
        not workspace.verify_role_execution(receipt)
        or receipt.execution_id != execution_id
        or receipt.stage != "S4"
        or receipt.role != "modeler"
        or receipt.subject_id != contract.task_id
        or receipt.input_authority_hash != input_authority_hash
        or receipt.run_id != run_id
        or receipt.context_id != context_id
        or receipt.provider != "fma-harness"
        or receipt.model != PREDICTION_RUNTIME_ADAPTER_ID_V63
        or receipt.prompt_hash != PREDICTION_RUNTIME_PROTOCOL_HASH_V63
        or receipt.output_schema_hash
        != contract.prediction_output_schema_hash
        or receipt.output_artifact_hash != vectors[0][0].sha256
    ):
        raise ExternalPredictionRuntimeError(
            "prediction role receipt differs from deterministic runtime"
        )
    qualification._assert_current_model_prediction_binding(
        workspace=workspace,
        contract=contract,
        custody=custody,
        binding=binding,
    )
    return ExternalPredictionRuntimeResultV63(
        forecast_input=forecast_input,
        prediction_vector=vector,
        execution_receipt=receipt,
        binding=binding,
        resumed=True,
    )


def _replay_current_model_vector(
    *,
    workspace: StageWorkspaceV50,
    contract: PredictiveExternalQualificationContractV63,
    forecast_input: ExternalForecastInputV63,
) -> tuple[
    ODETimeSeriesSnapshotV52,
    ExecutableCandidateReceiptV62,
    ExternalPredictionVectorV63,
]:
    snapshot, receipt, bundle = _load_current_model(
        workspace=workspace,
        contract=contract,
    )
    if receipt.adapter_id == SCALAR_ODE_ADAPTER_ID:
        if not isinstance(bundle, ODEScientificBundleV52):
            raise ExternalPredictionRuntimeError(
                "scalar ODE receipt selected a different bundle type"
            )
        predictions = _scalar_ode_predictions(
            snapshot=snapshot,
            receipt=receipt,
            forecast_input=forecast_input,
        )
    elif receipt.adapter_id == ADAPTIVE_POSITIVE_SERIES_ADAPTER_ID:
        if not isinstance(bundle, AdaptivePositiveSeriesBundleV57):
            raise ExternalPredictionRuntimeError(
                "adaptive receipt selected a different bundle type"
            )
        predictions = _adaptive_predictions(
            snapshot=snapshot,
            receipt=receipt,
            bundle=bundle,
            forecast_input=forecast_input,
        )
    else:
        raise ExternalPredictionRuntimeError(
            f"unsupported current V6.2 adapter: {receipt.adapter_id}"
        )
    vector = ExternalPredictionVectorV63.seal(
        qualification_id=contract.qualification_id,
        local_context_hash=contract.local_context_hash,
        selected_model_identity_hash=contract.selected_model_identity_hash,
        external_snapshot_hash=forecast_input.input_hash,
        target_ids=forecast_input.target_ids,
        target_order_hash=forecast_input.target_order_hash,
        predictions=predictions,
        prediction_values_hash=sha256_value(predictions),
    )
    return snapshot, receipt, vector


def _verify_runtime_prerequisites(
    *,
    workspace: StageWorkspaceV50,
    contract: PredictiveExternalQualificationContractV63,
    custody: ExternalEvidenceCustodyV63,
) -> None:
    qualification._assert_contract_current(
        workspace=workspace,
        contract=contract,
    )
    qualification._assert_custody_bound(
        workspace=workspace,
        contract=contract,
        custody=custody,
    )
    qualification._verified_custody_admission(
        workspace=workspace,
        contract=contract,
        custody=custody,
    )


def verify_current_model_external_prediction_v63(
    *,
    workspace: StageWorkspaceV50,
    contract: PredictiveExternalQualificationContractV63,
    forecast_input: ExternalForecastInputV63,
    custody: ExternalEvidenceCustodyV63,
) -> ExternalPredictionRuntimeResultV63:
    """Read-only deterministic replay of a completed runtime prediction."""

    transaction_factory = getattr(
        getattr(getattr(workspace, "graph", None), "store", None),
        "writer_transaction",
        None,
    )
    if not callable(transaction_factory):
        raise ExternalPredictionRuntimeError(
            "workspace does not expose the required single-writer lock"
        )
    try:
        with transaction_factory():
            if not workspace.verify():
                raise ExternalPredictionRuntimeError(
                    "workspace failed verification before prediction replay"
                )
            contract.assert_sealed()
            current_input = _current_forecast_input(
                workspace=workspace,
                contract=contract,
                supplied=forecast_input,
            )
            _verify_runtime_prerequisites(
                workspace=workspace,
                contract=contract,
                custody=custody,
            )
            snapshot, executable_receipt, expected_vector = (
                _replay_current_model_vector(
                    workspace=workspace,
                    contract=contract,
                    forecast_input=current_input,
                )
            )
            completed = _recover_completed(
                workspace=workspace,
                contract=contract,
                custody=custody,
                forecast_input=current_input,
                snapshot=snapshot,
                executable_receipt=executable_receipt,
                expected_vector=expected_vector,
            )
            if completed is None:
                raise ExternalPredictionRuntimeError(
                    "completed current-model prediction binding is absent"
                )
            if not workspace.verify():
                raise ExternalPredictionRuntimeError(
                    "workspace failed verification after prediction replay"
                )
            return completed
    except ExternalPredictionRuntimeError:
        raise
    except (
        ExternalQualificationError,
        StageWorkspaceError,
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise ExternalPredictionRuntimeError(
            "current-model prediction replay failed closed"
        ) from exc


def run_current_model_external_prediction_v63(
    *,
    workspace: StageWorkspaceV50,
    contract: PredictiveExternalQualificationContractV63,
    forecast_input: ExternalForecastInputV63,
    custody: ExternalEvidenceCustodyV63,
) -> ExternalPredictionRuntimeResultV63:
    """Generate or resume one current-model V6.3 prediction chain.

    All mutation is serialized by the workspace writer lock.  The caller must
    pass a freshly opened workspace; the runtime re-verifies that object under
    the lock.  The function performs no network, subprocess, private-holdout,
    or external action.
    """

    transaction_factory = getattr(
        getattr(getattr(workspace, "graph", None), "store", None),
        "writer_transaction",
        None,
    )
    if not callable(transaction_factory):
        raise ExternalPredictionRuntimeError(
            "workspace does not expose the required single-writer lock"
        )
    try:
        with transaction_factory():
            if not workspace.verify():
                raise ExternalPredictionRuntimeError(
                    "workspace failed verification before prediction"
                )
            contract.assert_sealed()
            current_input = _current_forecast_input(
                workspace=workspace,
                contract=contract,
                supplied=forecast_input,
            )
            _verify_runtime_prerequisites(
                workspace=workspace,
                contract=contract,
                custody=custody,
            )
            snapshot, receipt, expected_vector = (
                _replay_current_model_vector(
                    workspace=workspace,
                    contract=contract,
                    forecast_input=current_input,
                )
            )
            completed = _recover_completed(
                workspace=workspace,
                contract=contract,
                custody=custody,
                forecast_input=current_input,
                snapshot=snapshot,
                executable_receipt=receipt,
                expected_vector=expected_vector,
            )
            if completed is not None:
                return completed

            vector = expected_vector
            vector, vector_artifact_hash = _commit_or_recover_vector(
                workspace=workspace,
                contract=contract,
                vector=vector,
            )
            input_authority_hash = _input_authority_hash(
                contract=contract,
                forecast_input=current_input,
            )
            execution_id, run_id, context_id = _stable_execution_identity(
                contract=contract,
                forecast_input=current_input,
            )
            trace = _runtime_trace_payload(
                contract=contract,
                forecast_input=current_input,
                snapshot=snapshot,
                executable_receipt=receipt,
                vector=vector,
                vector_artifact_hash=vector_artifact_hash,
                execution_id=execution_id,
                run_id=run_id,
                context_id=context_id,
                input_authority_hash=input_authority_hash,
            )
            trace_hash = _commit_or_recover_trace(
                workspace=workspace,
                contract=contract,
                trace=trace,
            )
            role_receipt = _issue_or_recover_role_receipt(
                workspace=workspace,
                contract=contract,
                execution_id=execution_id,
                run_id=run_id,
                context_id=context_id,
                input_authority_hash=input_authority_hash,
                transport_trace_hash=trace_hash,
                prediction_artifact_hash=vector_artifact_hash,
            )
            binding = issue_current_model_prediction_binding_v63(
                workspace=workspace,
                contract=contract,
                custody=custody,
                prediction_vector=vector,
                generator_execution_receipt=role_receipt,
            )
            if not workspace.verify():
                raise ExternalPredictionRuntimeError(
                    "workspace failed verification after prediction"
                )
            return ExternalPredictionRuntimeResultV63(
                forecast_input=current_input,
                prediction_vector=vector,
                execution_receipt=role_receipt,
                binding=binding,
                resumed=False,
            )
    except ExternalPredictionRuntimeError:
        raise
    except (
        ExternalQualificationError,
        StageWorkspaceError,
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise ExternalPredictionRuntimeError(
            "current-model external prediction failed closed"
        ) from exc


__all__ = [
    "ExternalPredictionRuntimeError",
    "ExternalPredictionRuntimeResultV63",
    "PREDICTION_RUNTIME_ADAPTER_ID_V63",
    "PREDICTION_RUNTIME_PROTOCOL_HASH_V63",
    "PREDICTION_TRACE_KIND_V63",
    "run_current_model_external_prediction_v63",
    "verify_current_model_external_prediction_v63",
]
