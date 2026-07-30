"""Code-owned V6.8 development catalog for real capability implementations.

The catalog is intentionally separate from the SDK schemas.  It binds actual
Python modules, Pydantic schemas, benchmark commitments, and sealed V6.0
routing packs into V6.8 manifests.  Both initial manifests remain development
only: mechanical conformance can register them in a sandbox, but cannot grant
stage-workflow admission or scientific qualification.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from fma.hashing import canonical_json, sha256_value
from fma.schemas import StrictModel
from fma.v5_2 import ode_system as ode_system_v52

from .capability_runtime_v68 import (
    CapabilityRuntimeCaseBindingV68,
    CapabilityRuntimeCaseContractV68,
    CapabilityRuntimeDefinitionV68,
    CapabilityRuntimeExecutionV68,
    CapabilityRuntimeInvocationV68,
    callable_semantic_hash_v68,
)
from .capability_sdk_v68 import (
    BaselineContractV68,
    BenchmarkContractV68,
    CapabilityConformanceReportV68,
    CapabilityManifestV68,
    CapabilityRegistryV68,
    LevelObligationV68,
    MeasurementApplicabilityV68,
    RecoveryContractV68,
    ResourceEnvelopeV68,
    SemanticImplementationBindingV68,
    TypedModelIRContractV68,
    evaluate_capability_conformance_v68,
    skeleton_subsumption_hash_v68,
)
from .positive_log_increment_v68 import (
    PositiveLogIncrementBundleV68,
    PositiveLogIncrementModelIRV68,
    PositiveLogIncrementReplayAuthorityV68,
    PositiveLogIncrementReplayReceiptV68,
    PositiveLogIncrementThresholdsV68,
    PositiveScalarSeriesSnapshotV68,
    compile_positive_log_increment_ir_v68,
    execute_positive_log_increment_ir_v68,
    run_authenticated_positive_log_increment_replays_v68,
)
from .positive_log_increment_verifier_v68 import (
    recompute_positive_log_increment_level_v68,
)
from .recovery_kernel import CapabilityPackV60
from .scalar_ode_pack_v68 import (
    ScalarAutonomousODEBundleV68,
    ScalarAutonomousODEModelIRV68,
    compile_scalar_autonomous_ode_ir_v68,
    execute_scalar_autonomous_ode_ir_v68,
)
from .scalar_ode_verifier_v68 import (
    recompute_scalar_autonomous_ode_level_v68,
)


SCALAR_ODE_MANIFEST_ID_V68 = "scalar_autonomous_ode_manifest_v68"
POSITIVE_LOG_INCREMENT_MANIFEST_ID_V68 = (
    "positive_log_increment_manifest_v68"
)

_LEVELS = ("L0", "L1", "L2", "L3", "L4")
_ODE_SKELETON_ATOMS = [
    "autonomous_scalar_flow",
    "constant",
    "exponential",
    "gompertz",
    "logistic",
]
_LOG_INCREMENT_SKELETON_ATOMS = [
    "log_growth_ar1",
    "log_random_walk_drift",
    "stochastic_log_increment",
]

_ODE_BENCHMARK_SPEC = {
    "schema_version": "6.8-capability-benchmark-commitment",
    "suite_id": "scalar_ode_development_suite_v68",
    "case_kinds": [
        "baseline_or_no_signal",
        "canonical_gold",
        "external_held_fresh_required",
        "incompatible_or_nonidentifiable",
        "metamorphic_or_boundary",
        "numerical_or_provenance_trap",
        "public_real_retrospective",
    ],
    "minimum_public_cases": 12,
    "minimum_adversarial_cases": 4,
}
_LOG_INCREMENT_BENCHMARK_SPEC = {
    "schema_version": "6.8-capability-benchmark-commitment",
    "suite_id": "positive_log_increment_development_suite_v68",
    "case_ids": [
        "ar1_material_improvement",
        "drift_canonical",
        "fresh_replay",
        "no_signal_abstain",
        "nonpositive_rejected",
        "short_series_rejected",
        "typed_ir_tamper_rejected",
    ],
    "case_kinds": [
        "baseline_or_no_signal",
        "canonical_gold",
        "metamorphic_or_boundary",
        "numerical_or_contract_trap",
        "replay_integrity",
    ],
    "minimum_public_cases": 7,
    "minimum_adversarial_cases": 3,
}


def _schema_hash(model_type: type[StrictModel]) -> str:
    return sha256_value(model_type.model_json_schema())


def _binding(
    implementation_id: str,
    implementation_version: str,
    callable_object: Callable[..., object],
    *,
    level: str | None = None,
) -> SemanticImplementationBindingV68:
    semantic_hash = callable_semantic_hash_v68(callable_object)
    if level is not None:
        semantic_hash = sha256_value(
            {
                "callable_semantic_hash": semantic_hash,
                "level": level,
            }
        )
    return SemanticImplementationBindingV68(
        implementation_id=implementation_id,
        implementation_version=implementation_version,
        entrypoint_ref=(
            f"{callable_object.__module__}.{callable_object.__qualname__}"
        ),
        semantic_hash=semantic_hash,
    )


def _scalar_ode_pack() -> CapabilityPackV60:
    return CapabilityPackV60.seal(
        pack_id="scalar_autonomous_ode_v52",
        pack_version="v5.2",
        state_kinds=["scalar"],
        time_kinds=["continuous"],
        dynamics_kinds=["autonomous"],
        observation_kinds=["complete"],
        minimum_observations=12,
        requires_positive_observations=True,
        requires_strictly_increasing_time=True,
        executor_id="fma_v52_scalar_ode_executor",
        scientific_adapter_ids=["scalar_ode_scientific_adapter"],
        baseline_ids=["constant", "persistence"],
        supported_levels=list(_LEVELS),
    )


def _positive_log_increment_pack() -> CapabilityPackV60:
    return CapabilityPackV60.seal(
        pack_id="positive_log_increment_v68",
        pack_version="v6.8",
        state_kinds=["scalar"],
        time_kinds=["continuous", "discrete"],
        dynamics_kinds=["autonomous", "stochastic"],
        observation_kinds=["complete"],
        minimum_observations=34,
        requires_positive_observations=True,
        requires_strictly_increasing_time=True,
        executor_id="fma_v68_positive_log_increment_executor",
        scientific_adapter_ids=[
            "positive_log_increment_scientific_adapter_v68"
        ],
        baseline_ids=["persistence"],
        supported_levels=list(_LEVELS),
    )


def _scalar_ode_typed_ir_contract() -> TypedModelIRContractV68:
    return TypedModelIRContractV68(
        ir_kind="scalar_autonomous_ode_ir_v68",
        ir_schema_version="v6.8",
        ir_schema_hash=_schema_hash(ScalarAutonomousODEModelIRV68),
        compiler_input_schema_hash=_schema_hash(
            ode_system_v52.ODEThresholdsV52
        ),
        compiler_output_schema_hash=_schema_hash(
            ScalarAutonomousODEModelIRV68
        ),
    )


def _positive_log_increment_typed_ir_contract() -> TypedModelIRContractV68:
    return TypedModelIRContractV68(
        ir_kind="positive_log_increment_ir_v68",
        ir_schema_version="v6.8",
        ir_schema_hash=_schema_hash(PositiveLogIncrementModelIRV68),
        compiler_input_schema_hash=_schema_hash(
            PositiveLogIncrementThresholdsV68
        ),
        compiler_output_schema_hash=_schema_hash(
            PositiveLogIncrementModelIRV68
        ),
    )


def scalar_autonomous_ode_manifest_v68() -> CapabilityManifestV68:
    """Wrap the historical ODE pack without changing its V5.2/V6.0 meaning."""

    atoms = list(_ODE_SKELETON_ATOMS)
    compiler = _binding(
        "scalar_ode_ir_compiler_v68",
        "v6.8",
        compile_scalar_autonomous_ode_ir_v68,
    )
    executor = _binding(
        "fma_v52_scalar_ode_executor",
        "v6.8",
        execute_scalar_autonomous_ode_ir_v68,
    )
    return CapabilityManifestV68.seal(
        manifest_id=SCALAR_ODE_MANIFEST_ID_V68,
        capability_pack=_scalar_ode_pack(),
        supported_claim_kinds=["predictive"],
        claim_ceilings={"predictive": "local_predictive_evidence_only"},
        supported_task_kinds=["prediction"],
        measurement_applicability=MeasurementApplicabilityV68(
            scale_types=["ratio"],
            study_design_types=["time_series"],
            missingness_policies=["reject_incomplete_series"],
            minimum_planned_observations=23,
        ),
        maturity="development_sandbox",
        typed_ir=_scalar_ode_typed_ir_contract(),
        compiler=compiler,
        executor=executor,
        level_obligations=[
            LevelObligationV68(
                level=level,
                obligation_id=f"scalar_ode_{level.lower()}_v52",
                verifier=_binding(
                    f"scalar_ode_{level.lower()}_verifier_v52",
                    "v6.8",
                    recompute_scalar_autonomous_ode_level_v68,
                    level=level,
                ),
                evidence_kind=f"scalar_ode_{level.lower()}_evidence_v52",
            )
            for level in _LEVELS
        ],
        resources=ResourceEnvelopeV68(
            max_wall_seconds=600,
            max_cpu_seconds=600,
            max_memory_megabytes=2048,
            max_artifact_bytes=20_000_000,
            max_model_calls=0,
            max_tool_calls=8,
        ),
        baselines=BaselineContractV68(
            baseline_ids=["constant", "persistence"],
            supported_common_loss_ids=["one_step_rmse"],
        ),
        recovery=RecoveryContractV68(
            allowed_actions=["ABSTAIN", "BRANCH", "PATCH", "RETRY"],
            max_graph_attempts=3,
            max_same_attempt_retries=1,
        ),
        benchmark=BenchmarkContractV68(
            benchmark_suite_id="scalar_ode_development_suite_v68",
            benchmark_suite_version="v1",
            benchmark_suite_hash=sha256_value(_ODE_BENCHMARK_SPEC),
            minimum_public_cases=12,
            minimum_adversarial_cases=4,
            external_private_evaluation_required_for_stage_workflow=True,
        ),
        skeleton_atoms=atoms,
        skeleton_subsumption_hash=skeleton_subsumption_hash_v68(atoms),
    )


def positive_log_increment_manifest_v68() -> CapabilityManifestV68:
    """Bind the new pure statistical pack as development-only capability."""

    atoms = list(_LOG_INCREMENT_SKELETON_ATOMS)
    return CapabilityManifestV68.seal(
        manifest_id=POSITIVE_LOG_INCREMENT_MANIFEST_ID_V68,
        capability_pack=_positive_log_increment_pack(),
        supported_claim_kinds=["predictive"],
        claim_ceilings={"predictive": "local_predictive_evidence_only"},
        supported_task_kinds=["prediction"],
        measurement_applicability=MeasurementApplicabilityV68(
            scale_types=["ratio"],
            study_design_types=["time_series"],
            missingness_policies=["reject_incomplete_series"],
            minimum_planned_observations=34,
        ),
        maturity="development_sandbox",
        typed_ir=_positive_log_increment_typed_ir_contract(),
        compiler=_binding(
            "positive_log_increment_ir_compiler_v68",
            "v6.8",
            compile_positive_log_increment_ir_v68,
        ),
        executor=_binding(
            "fma_v68_positive_log_increment_executor",
            "v6.8",
            execute_positive_log_increment_ir_v68,
        ),
        level_obligations=[
            LevelObligationV68(
                level=level,
                obligation_id=f"positive_log_increment_{level.lower()}_v68",
                verifier=_binding(
                    f"positive_log_increment_{level.lower()}_verifier_v68",
                    "v6.8",
                    recompute_positive_log_increment_level_v68,
                    level=level,
                ),
                evidence_kind=(
                    f"positive_log_increment_{level.lower()}_evidence_v68"
                ),
            )
            for level in _LEVELS
        ],
        resources=ResourceEnvelopeV68(
            max_wall_seconds=600,
            max_cpu_seconds=600,
            max_memory_megabytes=2048,
            max_artifact_bytes=20_000_000,
            max_model_calls=0,
            max_tool_calls=8,
        ),
        baselines=BaselineContractV68(
            baseline_ids=["persistence"],
            supported_common_loss_ids=["one_step_rmse"],
        ),
        recovery=RecoveryContractV68(
            allowed_actions=["ABSTAIN", "BRANCH", "PATCH", "RETRY"],
            max_graph_attempts=3,
            max_same_attempt_retries=1,
        ),
        benchmark=BenchmarkContractV68(
            benchmark_suite_id=(
                "positive_log_increment_development_suite_v68"
            ),
            benchmark_suite_version="v1",
            benchmark_suite_hash=sha256_value(
                _LOG_INCREMENT_BENCHMARK_SPEC
            ),
            minimum_public_cases=7,
            minimum_adversarial_cases=3,
            external_private_evaluation_required_for_stage_workflow=True,
        ),
        skeleton_atoms=atoms,
        skeleton_subsumption_hash=skeleton_subsumption_hash_v68(atoms),
    )


class _PositiveLogIncrementRuntimePayloadV68(StrictModel):
    schema_version: Literal["6.8-positive-log-runtime-payload"] = (
        "6.8-positive-log-runtime-payload"
    )
    snapshot: PositiveScalarSeriesSnapshotV68
    thresholds: PositiveLogIncrementThresholdsV68
    replay_mode: Literal["none", "authenticated_local_fixture"]


class _PositiveLogIncrementRuntimeResultV68(StrictModel):
    schema_version: Literal["6.8-positive-log-runtime-result"] = (
        "6.8-positive-log-runtime-result"
    )
    bundle: PositiveLogIncrementBundleV68
    replay_receipts: list[PositiveLogIncrementReplayReceiptV68]


class _ScalarODERuntimePayloadV68(StrictModel):
    schema_version: Literal["6.8-scalar-ode-runtime-payload"] = (
        "6.8-scalar-ode-runtime-payload"
    )
    snapshot: ode_system_v52.ODETimeSeriesSnapshotV52
    thresholds: ode_system_v52.ODEThresholdsV52


class _ScalarODERuntimeResultV68(StrictModel):
    schema_version: Literal["6.8-scalar-ode-runtime-result"] = (
        "6.8-scalar-ode-runtime-result"
    )
    bundle: ScalarAutonomousODEBundleV68
    replay_output_hashes: list[str]


_POSITIVE_RUNTIME_PAYLOAD_SCHEMA_ID = "positive_log_increment_runtime_payload"
_POSITIVE_RUNTIME_PAYLOAD_SCHEMA_HASH = _schema_hash(
    _PositiveLogIncrementRuntimePayloadV68
)
_POSITIVE_RUNTIME_RESULT_SCHEMA_ID = "positive_log_increment_runtime_result"
_POSITIVE_RUNTIME_RESULT_SCHEMA_HASH = _schema_hash(
    _PositiveLogIncrementRuntimeResultV68
)
_SCALAR_ODE_RUNTIME_PAYLOAD_SCHEMA_ID = "scalar_ode_runtime_payload"
_SCALAR_ODE_RUNTIME_PAYLOAD_SCHEMA_HASH = _schema_hash(
    _ScalarODERuntimePayloadV68
)
_SCALAR_ODE_RUNTIME_RESULT_SCHEMA_ID = "scalar_ode_runtime_result"
_SCALAR_ODE_RUNTIME_RESULT_SCHEMA_HASH = _schema_hash(
    _ScalarODERuntimeResultV68
)
_LOCAL_REPLAY_SECRET = b"v68-local-conformance-fixture-key"


def _runtime_invocation(
    *,
    manifest: CapabilityManifestV68,
    case_id: str,
    payload_schema_id: str,
    payload_schema_hash: str,
    payload: StrictModel,
) -> CapabilityRuntimeInvocationV68:
    return CapabilityRuntimeInvocationV68.seal(
        manifest_id=manifest.manifest_id,
        manifest_hash=manifest.manifest_hash,
        case_id=case_id,
        payload_schema_id=payload_schema_id,
        payload_schema_hash=payload_schema_hash,
        payload=payload.model_dump(mode="json"),
    )


def _growth_snapshot(
    *,
    task_id: str,
    mean: float,
    phi: float,
    sigma: float,
    seed: int,
    count: int = 72,
) -> PositiveScalarSeriesSnapshotV68:
    rng = np.random.default_rng(seed)
    growths = np.zeros(count - 1, dtype=float)
    growths[0] = mean
    for index in range(1, len(growths)):
        growths[index] = (
            mean
            + phi * (growths[index - 1] - mean)
            + rng.normal(0.0, sigma)
        )
    values = 100.0 * np.exp(np.concatenate(([0.0], np.cumsum(growths))))
    return PositiveScalarSeriesSnapshotV68.seal(
        task_id=task_id,
        time_unit="year",
        state_unit="positive_index",
        times=np.arange(count, dtype=float).tolist(),
        observations=values.tolist(),
        source_id=f"{task_id}-source",
        fixture_only=True,
    )


def _positive_invocation(
    *,
    case_id: str,
    snapshot: PositiveScalarSeriesSnapshotV68,
    replay_mode: Literal["none", "authenticated_local_fixture"] = "none",
) -> CapabilityRuntimeInvocationV68:
    payload = _PositiveLogIncrementRuntimePayloadV68(
        snapshot=snapshot,
        thresholds=PositiveLogIncrementThresholdsV68.seal(),
        replay_mode=replay_mode,
    )
    return _runtime_invocation(
        manifest=positive_log_increment_manifest_v68(),
        case_id=case_id,
        payload_schema_id=_POSITIVE_RUNTIME_PAYLOAD_SCHEMA_ID,
        payload_schema_hash=_POSITIVE_RUNTIME_PAYLOAD_SCHEMA_HASH,
        payload=payload,
    )


def _positive_ar1_input() -> CapabilityRuntimeInvocationV68:
    return _positive_invocation(
        case_id="ar1_material_improvement",
        snapshot=_growth_snapshot(
            task_id="v68-runtime-ar1",
            mean=0.04,
            phi=0.85,
            sigma=0.02,
            seed=103,
        ),
    )


def _positive_drift_input() -> CapabilityRuntimeInvocationV68:
    return _positive_invocation(
        case_id="drift_canonical",
        snapshot=_growth_snapshot(
            task_id="v68-runtime-drift",
            mean=0.04,
            phi=0.0,
            sigma=0.01,
            seed=102,
        ),
    )


def _positive_fresh_replay_input() -> CapabilityRuntimeInvocationV68:
    return _positive_invocation(
        case_id="fresh_replay",
        snapshot=_growth_snapshot(
            task_id="v68-runtime-fresh-replay",
            mean=0.04,
            phi=0.0,
            sigma=0.01,
            seed=102,
        ),
        replay_mode="authenticated_local_fixture",
    )


def _positive_no_signal_input() -> CapabilityRuntimeInvocationV68:
    values = np.full(72, 100.0, dtype=float)
    snapshot = PositiveScalarSeriesSnapshotV68.seal(
        task_id="v68-runtime-no-signal",
        time_unit="year",
        state_unit="positive_index",
        times=np.arange(len(values), dtype=float).tolist(),
        observations=values.tolist(),
        source_id="v68-runtime-no-signal-source",
        fixture_only=True,
    )
    return _positive_invocation(
        case_id="no_signal_abstain",
        snapshot=snapshot,
    )


def _positive_tamper_input() -> CapabilityRuntimeInvocationV68:
    return _positive_invocation(
        case_id="typed_ir_tamper_rejected",
        snapshot=_growth_snapshot(
            task_id="v68-runtime-tamper",
            mean=0.04,
            phi=0.0,
            sigma=0.01,
            seed=102,
        ),
    )


def _parse_positive_invocation(
    invocation: CapabilityRuntimeInvocationV68,
) -> _PositiveLogIncrementRuntimePayloadV68:
    invocation.assert_sealed()
    if (
        invocation.payload_schema_id
        != _POSITIVE_RUNTIME_PAYLOAD_SCHEMA_ID
        or invocation.payload_schema_hash
        != _POSITIVE_RUNTIME_PAYLOAD_SCHEMA_HASH
    ):
        raise ValueError("V6.8 positive runtime payload schema differs")
    return _PositiveLogIncrementRuntimePayloadV68.model_validate(
        invocation.payload
    )


def _parse_positive_execution(
    execution: CapabilityRuntimeExecutionV68,
) -> _PositiveLogIncrementRuntimeResultV68:
    execution.assert_sealed()
    if (
        execution.payload_schema_id != _POSITIVE_RUNTIME_RESULT_SCHEMA_ID
        or execution.payload_schema_hash != _POSITIVE_RUNTIME_RESULT_SCHEMA_HASH
    ):
        raise ValueError("V6.8 positive runtime result schema differs")
    return _PositiveLogIncrementRuntimeResultV68.model_validate(
        execution.payload
    )


def _compile_positive_runtime(
    invocation: CapabilityRuntimeInvocationV68,
) -> StrictModel:
    payload = _parse_positive_invocation(invocation)
    return compile_positive_log_increment_ir_v68(payload.thresholds)


def _local_replay_authority() -> PositiveLogIncrementReplayAuthorityV68:
    return PositiveLogIncrementReplayAuthorityV68(
        key_id="v68-local-conformance-fixture",
        secret=_LOCAL_REPLAY_SECRET,
    )


def _execute_positive_runtime(
    invocation: CapabilityRuntimeInvocationV68,
    model_ir: StrictModel,
) -> CapabilityRuntimeExecutionV68:
    payload = _parse_positive_invocation(invocation)
    ir = PositiveLogIncrementModelIRV68.model_validate(
        model_ir.model_dump(mode="json")
    )
    receipts: list[PositiveLogIncrementReplayReceiptV68] = []
    authority: PositiveLogIncrementReplayAuthorityV68 | None = None
    if payload.replay_mode == "authenticated_local_fixture":
        authority = _local_replay_authority()
        with tempfile.TemporaryDirectory(
            prefix="fma-v68-runtime-replay-"
        ) as temporary:
            replay_path = Path(temporary) / "input.json"
            replay_path.write_text(
                canonical_json(
                    {
                        "snapshot": payload.snapshot.model_dump(mode="json"),
                        "thresholds": payload.thresholds.model_dump(mode="json"),
                    }
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            receipts = run_authenticated_positive_log_increment_replays_v68(
                replay_path,
                authority=authority,
            )
    bundle = execute_positive_log_increment_ir_v68(
        model_ir=ir,
        snapshot=payload.snapshot,
        thresholds=payload.thresholds,
        replay_receipts=receipts,
        replay_authority=authority,
    )
    result = _PositiveLogIncrementRuntimeResultV68(
        bundle=bundle,
        replay_receipts=receipts,
    )
    return CapabilityRuntimeExecutionV68.seal(
        manifest_id=invocation.manifest_id,
        manifest_hash=invocation.manifest_hash,
        case_id=invocation.case_id,
        payload_schema_id=_POSITIVE_RUNTIME_RESULT_SCHEMA_ID,
        payload_schema_hash=_POSITIVE_RUNTIME_RESULT_SCHEMA_HASH,
        payload=result.model_dump(mode="json"),
    )


def _verify_positive_runtime_level(
    invocation: CapabilityRuntimeInvocationV68,
    model_ir: StrictModel,
    execution: CapabilityRuntimeExecutionV68,
    level: str,
) -> StrictModel:
    payload = _parse_positive_invocation(invocation)
    result = _parse_positive_execution(execution)
    ir = PositiveLogIncrementModelIRV68.model_validate(
        model_ir.model_dump(mode="json")
    )
    authority = (
        _local_replay_authority()
        if payload.replay_mode == "authenticated_local_fixture"
        else None
    )
    return recompute_positive_log_increment_level_v68(
        bundle=result.bundle,
        level=level,
        model_ir=ir,
        snapshot=payload.snapshot,
        thresholds=payload.thresholds,
        replay_receipts=result.replay_receipts,
        replay_authority=authority,
    )


def _tamper_positive_ir(model_ir: StrictModel) -> StrictModel:
    ir = PositiveLogIncrementModelIRV68.model_validate(
        model_ir.model_dump(mode="json")
    )
    return ir.model_copy(update={"threshold_hash": "f" * 64})


def _probe_positive_nonpositive() -> None:
    values = np.ones(34, dtype=float)
    values[-1] = 0.0
    PositiveScalarSeriesSnapshotV68.seal(
        task_id="v68-runtime-nonpositive",
        time_unit="year",
        state_unit="positive_index",
        times=np.arange(len(values), dtype=float).tolist(),
        observations=values.tolist(),
        source_id="v68-runtime-nonpositive-source",
        fixture_only=True,
    )


def _probe_positive_short() -> None:
    values = np.ones(33, dtype=float)
    PositiveScalarSeriesSnapshotV68.seal(
        task_id="v68-runtime-short",
        time_unit="year",
        state_unit="positive_index",
        times=np.arange(len(values), dtype=float).tolist(),
        observations=values.tolist(),
        source_id="v68-runtime-short-source",
        fixture_only=True,
    )


def _assert_positive_ar1(
    execution: CapabilityRuntimeExecutionV68,
) -> None:
    bundle = _parse_positive_execution(execution).bundle
    if bundle.selected_model_id != "log_growth_ar1":
        raise ValueError("V6.8 AR1 benchmark selected another model")


def _assert_positive_drift(
    execution: CapabilityRuntimeExecutionV68,
) -> None:
    bundle = _parse_positive_execution(execution).bundle
    if bundle.selected_model_id != "log_random_walk_drift":
        raise ValueError("V6.8 drift benchmark selected another model")


def _assert_positive_no_signal(
    execution: CapabilityRuntimeExecutionV68,
) -> None:
    bundle = _parse_positive_execution(execution).bundle
    if bundle.selection_status != "ABSTAIN":
        raise ValueError("V6.8 no-signal benchmark did not abstain")


def _assert_positive_fresh_replay(
    execution: CapabilityRuntimeExecutionV68,
) -> None:
    result = _parse_positive_execution(execution)
    if len(result.replay_receipts) != 2:
        raise ValueError("V6.8 fresh replay benchmark lacks two receipts")
    if [item.status for item in result.bundle.levels] != [
        "NOT_RUN",
        "PASS",
        "PASS",
        "PASS",
        "PASS",
    ]:
        raise ValueError("V6.8 local replay level statuses differ")
    if result.bundle.local_l0_l4_complete:
        raise ValueError("V6.8 local replay improperly closed L0")


def _scalar_ode_snapshot(
    *,
    task_id: str,
) -> ode_system_v52.ODETimeSeriesSnapshotV52:
    times = np.arange(30, dtype=float)
    values = 20.0 * np.exp(0.025 * times)
    return ode_system_v52.ODETimeSeriesSnapshotV52.seal(
        task_id=task_id,
        time_unit="day",
        state_unit="registered_positive_state",
        times=times.tolist(),
        observations=values.tolist(),
        source_id=f"{task_id}-source",
        fixture_only=True,
    )


def _scalar_ode_invocation(
    *,
    case_id: str,
) -> CapabilityRuntimeInvocationV68:
    payload = _ScalarODERuntimePayloadV68(
        snapshot=_scalar_ode_snapshot(task_id=case_id),
        thresholds=ode_system_v52.ODEThresholdsV52.seal(),
    )
    return _runtime_invocation(
        manifest=scalar_autonomous_ode_manifest_v68(),
        case_id=case_id,
        payload_schema_id=_SCALAR_ODE_RUNTIME_PAYLOAD_SCHEMA_ID,
        payload_schema_hash=_SCALAR_ODE_RUNTIME_PAYLOAD_SCHEMA_HASH,
        payload=payload,
    )


def _scalar_ode_canonical_input() -> CapabilityRuntimeInvocationV68:
    return _scalar_ode_invocation(case_id="scalar_ode_canonical_runtime")


def _scalar_ode_tamper_input() -> CapabilityRuntimeInvocationV68:
    return _scalar_ode_invocation(
        case_id="scalar_ode_typed_ir_tamper_rejected"
    )


def _parse_scalar_ode_invocation(
    invocation: CapabilityRuntimeInvocationV68,
) -> _ScalarODERuntimePayloadV68:
    invocation.assert_sealed()
    if (
        invocation.payload_schema_id != _SCALAR_ODE_RUNTIME_PAYLOAD_SCHEMA_ID
        or invocation.payload_schema_hash
        != _SCALAR_ODE_RUNTIME_PAYLOAD_SCHEMA_HASH
    ):
        raise ValueError("V6.8 scalar ODE runtime payload schema differs")
    return _ScalarODERuntimePayloadV68.model_validate(invocation.payload)


def _parse_scalar_ode_execution(
    execution: CapabilityRuntimeExecutionV68,
) -> _ScalarODERuntimeResultV68:
    execution.assert_sealed()
    if (
        execution.payload_schema_id != _SCALAR_ODE_RUNTIME_RESULT_SCHEMA_ID
        or execution.payload_schema_hash
        != _SCALAR_ODE_RUNTIME_RESULT_SCHEMA_HASH
    ):
        raise ValueError("V6.8 scalar ODE runtime result schema differs")
    return _ScalarODERuntimeResultV68.model_validate(execution.payload)


def _compile_scalar_ode_runtime(
    invocation: CapabilityRuntimeInvocationV68,
) -> StrictModel:
    payload = _parse_scalar_ode_invocation(invocation)
    return compile_scalar_autonomous_ode_ir_v68(payload.thresholds)


def _execute_scalar_ode_runtime(
    invocation: CapabilityRuntimeInvocationV68,
    model_ir: StrictModel,
) -> CapabilityRuntimeExecutionV68:
    payload = _parse_scalar_ode_invocation(invocation)
    ir = ScalarAutonomousODEModelIRV68.model_validate(
        model_ir.model_dump(mode="json")
    )
    bundle = execute_scalar_autonomous_ode_ir_v68(
        model_ir=ir,
        snapshot=payload.snapshot,
        thresholds=payload.thresholds,
    )
    result = _ScalarODERuntimeResultV68(
        bundle=bundle,
        replay_output_hashes=[],
    )
    return CapabilityRuntimeExecutionV68.seal(
        manifest_id=invocation.manifest_id,
        manifest_hash=invocation.manifest_hash,
        case_id=invocation.case_id,
        payload_schema_id=_SCALAR_ODE_RUNTIME_RESULT_SCHEMA_ID,
        payload_schema_hash=_SCALAR_ODE_RUNTIME_RESULT_SCHEMA_HASH,
        payload=result.model_dump(mode="json"),
    )


def _verify_scalar_ode_runtime_level(
    invocation: CapabilityRuntimeInvocationV68,
    model_ir: StrictModel,
    execution: CapabilityRuntimeExecutionV68,
    level: str,
) -> StrictModel:
    payload = _parse_scalar_ode_invocation(invocation)
    result = _parse_scalar_ode_execution(execution)
    ir = ScalarAutonomousODEModelIRV68.model_validate(
        model_ir.model_dump(mode="json")
    )
    return recompute_scalar_autonomous_ode_level_v68(
        bundle=result.bundle,
        level=level,
        model_ir=ir,
        snapshot=payload.snapshot,
        thresholds=payload.thresholds,
        replay_output_hashes=result.replay_output_hashes,
    )


def _tamper_scalar_ode_ir(model_ir: StrictModel) -> StrictModel:
    ir = ScalarAutonomousODEModelIRV68.model_validate(
        model_ir.model_dump(mode="json")
    )
    return ir.model_copy(update={"threshold_hash": "f" * 64})


def _probe_scalar_ode_nonpositive() -> None:
    values = np.ones(12, dtype=float)
    values[-1] = 0.0
    ode_system_v52.ODETimeSeriesSnapshotV52.seal(
        task_id="v68-scalar-ode-nonpositive",
        time_unit="day",
        state_unit="registered_positive_state",
        times=np.arange(len(values), dtype=float).tolist(),
        observations=values.tolist(),
        source_id="v68-scalar-ode-nonpositive-source",
        fixture_only=True,
    )


def _assert_scalar_ode_canonical(
    execution: CapabilityRuntimeExecutionV68,
) -> None:
    result = _parse_scalar_ode_execution(execution)
    if result.bundle.legacy_bundle.selected_candidate_id != "exponential":
        raise ValueError("V6.8 scalar ODE benchmark selected another family")


def _case_contract(
    *,
    case_id: str,
    case_kind: str,
    mode: str,
    expected_level_statuses: dict[str, str] | None = None,
    expected_exception_types: list[str] | None = None,
    deterministic: bool = False,
    public: bool = True,
    adversarial: bool = False,
) -> CapabilityRuntimeCaseContractV68:
    expected_outcome = {
        "compile_execute": "EXECUTED",
        "tamper_rejection": "REJECTED",
        "incompatible_rejection": "REJECTED",
        "not_run": "NOT_RUN",
    }[mode]
    return CapabilityRuntimeCaseContractV68.seal(
        case_id=case_id,
        case_kind=case_kind,
        mode=mode,
        expected_outcome=expected_outcome,
        expected_level_statuses=expected_level_statuses or {},
        expected_exception_types=expected_exception_types or [],
        deterministic_execution_required=deterministic,
        public_benchmark_case=public,
        adversarial_case=adversarial,
    )


def _positive_runtime_cases() -> tuple[CapabilityRuntimeCaseBindingV68, ...]:
    no_replay_levels = {
        "L0": "NOT_RUN",
        "L1": "PASS",
        "L2": "PASS",
        "L3": "PASS",
        "L4": "PASS",
    }
    return (
        CapabilityRuntimeCaseBindingV68(
            contract=_case_contract(
                case_id="ar1_material_improvement",
                case_kind="canonical_gold",
                mode="compile_execute",
                expected_level_statuses=no_replay_levels,
            ),
            input_factory=_positive_ar1_input,
            output_assertion=_assert_positive_ar1,
        ),
        CapabilityRuntimeCaseBindingV68(
            contract=_case_contract(
                case_id="drift_canonical",
                case_kind="canonical_gold",
                mode="compile_execute",
                expected_level_statuses=no_replay_levels,
                deterministic=True,
            ),
            input_factory=_positive_drift_input,
            output_assertion=_assert_positive_drift,
        ),
        CapabilityRuntimeCaseBindingV68(
            contract=_case_contract(
                case_id="fresh_replay",
                case_kind="replay_integrity",
                mode="compile_execute",
                expected_level_statuses={
                    "L0": "NOT_RUN",
                    "L1": "PASS",
                    "L2": "PASS",
                    "L3": "PASS",
                    "L4": "PASS",
                },
            ),
            input_factory=_positive_fresh_replay_input,
            output_assertion=_assert_positive_fresh_replay,
        ),
        CapabilityRuntimeCaseBindingV68(
            contract=_case_contract(
                case_id="no_signal_abstain",
                case_kind="baseline_or_no_signal",
                mode="compile_execute",
                expected_level_statuses={
                    "L0": "NOT_RUN",
                    "L1": "PASS",
                    "L2": "PASS",
                    "L3": "FAIL",
                    "L4": "FAIL",
                },
            ),
            input_factory=_positive_no_signal_input,
            output_assertion=_assert_positive_no_signal,
        ),
        CapabilityRuntimeCaseBindingV68(
            contract=_case_contract(
                case_id="nonpositive_rejected",
                case_kind="metamorphic_or_boundary",
                mode="incompatible_rejection",
                expected_exception_types=["ValidationError"],
                adversarial=True,
            ),
            incompatible_probe=_probe_positive_nonpositive,
        ),
        CapabilityRuntimeCaseBindingV68(
            contract=_case_contract(
                case_id="short_series_rejected",
                case_kind="metamorphic_or_boundary",
                mode="incompatible_rejection",
                expected_exception_types=["ValidationError"],
                adversarial=True,
            ),
            incompatible_probe=_probe_positive_short,
        ),
        CapabilityRuntimeCaseBindingV68(
            contract=_case_contract(
                case_id="typed_ir_tamper_rejected",
                case_kind="numerical_or_contract_trap",
                mode="tamper_rejection",
                expected_exception_types=["ValidationError"],
                adversarial=True,
            ),
            input_factory=_positive_tamper_input,
            ir_mutator=_tamper_positive_ir,
        ),
    )


def _scalar_ode_runtime_cases() -> tuple[
    CapabilityRuntimeCaseBindingV68,
    ...,
]:
    return (
        CapabilityRuntimeCaseBindingV68(
            contract=_case_contract(
                case_id="scalar_ode_canonical_runtime",
                case_kind="canonical_runtime",
                mode="compile_execute",
                deterministic=True,
                public=False,
            ),
            input_factory=_scalar_ode_canonical_input,
            output_assertion=_assert_scalar_ode_canonical,
        ),
        CapabilityRuntimeCaseBindingV68(
            contract=_case_contract(
                case_id="scalar_ode_nonpositive_rejected",
                case_kind="boundary_rejection",
                mode="incompatible_rejection",
                expected_exception_types=["ValidationError"],
                public=False,
            ),
            incompatible_probe=_probe_scalar_ode_nonpositive,
        ),
        CapabilityRuntimeCaseBindingV68(
            contract=_case_contract(
                case_id="scalar_ode_typed_ir_tamper_rejected",
                case_kind="contract_tamper",
                mode="tamper_rejection",
                expected_exception_types=["ValidationError"],
                public=False,
            ),
            input_factory=_scalar_ode_tamper_input,
            ir_mutator=_tamper_scalar_ode_ir,
        ),
    )


def _observed_hashes(
    *,
    manifest: CapabilityManifestV68,
    capability_pack: CapabilityPackV60,
    benchmark_spec: dict[str, object],
    typed_ir: TypedModelIRContractV68,
    compiler: Callable[..., object],
    executor: Callable[..., object],
    verifier: Callable[..., object],
    skeleton_atoms: list[str],
) -> dict[str, str]:
    observed = {
        "benchmark": sha256_value(benchmark_spec),
        "capability_pack": str(capability_pack.pack_hash),
        "compiler": callable_semantic_hash_v68(compiler),
        "executor": callable_semantic_hash_v68(executor),
        "skeleton_subsumption": skeleton_subsumption_hash_v68(
            skeleton_atoms
        ),
        "typed_ir": sha256_value(typed_ir.model_dump(mode="json")),
    }
    observed.update(
        {
            f"verifier.{level}": sha256_value(
                {
                    "callable_semantic_hash": (
                        callable_semantic_hash_v68(verifier)
                    ),
                    "level": level,
                }
            )
            for level in _LEVELS
        }
    )
    if set(observed) != set(manifest.expected_semantic_hashes()):
        raise ValueError("V6.8 catalog observed identity set differs")
    return {key: observed[key] for key in sorted(observed)}


def scalar_autonomous_ode_runtime_definition_v68(
) -> CapabilityRuntimeDefinitionV68:
    manifest = scalar_autonomous_ode_manifest_v68()
    return CapabilityRuntimeDefinitionV68(
        definition_id="scalar_autonomous_ode_runtime_v68",
        manifest=manifest,
        observed_semantic_hashes=_observed_hashes(
            manifest=manifest,
            capability_pack=_scalar_ode_pack(),
            benchmark_spec=_ODE_BENCHMARK_SPEC,
            typed_ir=_scalar_ode_typed_ir_contract(),
            compiler=compile_scalar_autonomous_ode_ir_v68,
            executor=execute_scalar_autonomous_ode_ir_v68,
            verifier=recompute_scalar_autonomous_ode_level_v68,
            skeleton_atoms=list(_ODE_SKELETON_ATOMS),
        ),
        benchmark_spec=_ODE_BENCHMARK_SPEC,
        cases=_scalar_ode_runtime_cases(),
        compiler=_compile_scalar_ode_runtime,
        executor=_execute_scalar_ode_runtime,
        level_verifier=_verify_scalar_ode_runtime_level,
    )


def positive_log_increment_runtime_definition_v68(
) -> CapabilityRuntimeDefinitionV68:
    manifest = positive_log_increment_manifest_v68()
    return CapabilityRuntimeDefinitionV68(
        definition_id="positive_log_increment_runtime_v68",
        manifest=manifest,
        observed_semantic_hashes=_observed_hashes(
            manifest=manifest,
            capability_pack=_positive_log_increment_pack(),
            benchmark_spec=_LOG_INCREMENT_BENCHMARK_SPEC,
            typed_ir=_positive_log_increment_typed_ir_contract(),
            compiler=compile_positive_log_increment_ir_v68,
            executor=execute_positive_log_increment_ir_v68,
            verifier=recompute_positive_log_increment_level_v68,
            skeleton_atoms=list(_LOG_INCREMENT_SKELETON_ATOMS),
        ),
        benchmark_spec=_LOG_INCREMENT_BENCHMARK_SPEC,
        cases=_positive_runtime_cases(),
        compiler=_compile_positive_runtime,
        executor=_execute_positive_runtime,
        level_verifier=_verify_positive_runtime_level,
    )


def default_capability_runtime_definitions_v68(
) -> dict[str, CapabilityRuntimeDefinitionV68]:
    """Return the two direct-callable development definitions by manifest ID."""

    definitions = (
        positive_log_increment_runtime_definition_v68(),
        scalar_autonomous_ode_runtime_definition_v68(),
    )
    return {
        item.manifest.manifest_id: item
        for item in sorted(
            definitions,
            key=lambda value: value.manifest.manifest_id,
        )
    }


def capability_runtime_definition_v68(
    *,
    manifest_id: str,
    manifest_hash: str,
) -> CapabilityRuntimeDefinitionV68:
    """Resolve only an exact code-owned manifest; never import an entrypoint."""

    definition = default_capability_runtime_definitions_v68().get(manifest_id)
    if definition is None:
        raise KeyError(f"unknown V6.8 runtime manifest: {manifest_id}")
    definition.assert_exact_manifest(
        manifest_id=manifest_id,
        manifest_hash=manifest_hash,
    )
    return definition


def observe_capability_manifest_v68(
    manifest: CapabilityManifestV68,
) -> dict[str, str]:
    """Recompute code/schema identities instead of copying manifest values."""

    manifest.assert_sealed()
    definition = default_capability_runtime_definitions_v68().get(
        manifest.manifest_id
    )
    if definition is None:
        raise KeyError(
            f"unknown V6.8 catalog manifest: {manifest.manifest_id}"
        )
    definition.assert_exact_manifest(
        manifest_id=manifest.manifest_id,
        manifest_hash=str(manifest.manifest_hash),
    )
    if manifest != definition.manifest:
        raise ValueError("V6.8 catalog manifest differs from code-owned definition")
    return dict(definition.observed_semantic_hashes)


def conform_capability_manifest_v68(
    manifest: CapabilityManifestV68,
) -> CapabilityConformanceReportV68:
    return evaluate_capability_conformance_v68(
        manifest,
        observed_semantic_hashes=observe_capability_manifest_v68(manifest),
    )


def default_development_capability_registry_v68() -> CapabilityRegistryV68:
    """Return the exact two-pack registry; neither pack gains stage authority."""

    registry = CapabilityRegistryV68(runtime_mode="development_sandbox")
    definitions = default_capability_runtime_definitions_v68()
    for definition in definitions.values():
        manifest = definition.manifest
        report = conform_capability_manifest_v68(manifest)
        if report.status != "PASS":
            raise ValueError(
                f"V6.8 catalog conformance failed: {manifest.manifest_id}"
            )
        registry.register(manifest, conformance_report=report)
    return registry


__all__ = [
    "POSITIVE_LOG_INCREMENT_MANIFEST_ID_V68",
    "SCALAR_ODE_MANIFEST_ID_V68",
    "capability_runtime_definition_v68",
    "conform_capability_manifest_v68",
    "default_capability_runtime_definitions_v68",
    "default_development_capability_registry_v68",
    "observe_capability_manifest_v68",
    "positive_log_increment_runtime_definition_v68",
    "positive_log_increment_manifest_v68",
    "scalar_autonomous_ode_runtime_definition_v68",
    "scalar_autonomous_ode_manifest_v68",
]
