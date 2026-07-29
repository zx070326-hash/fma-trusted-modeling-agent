from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from fma.v5.workspace_schemas import (
    CandidateFormalizationV50,
    CandidateSetV50,
    ModelSpecV50,
)
from fma.v5_2.ode_system import (
    ODEScientificBundleV52,
    ODEThresholdsV52,
    ODETimeSeriesSnapshotV52,
    build_ode_bundle_v52,
)
from fma.v5_6.hybrid_ode import HybridODEThresholdsV56
from fma.v5_7.adaptive_positive_series import (
    AdaptiveCandidateGraphV57,
    AdaptivePositiveSeriesBundleV57,
    AdaptiveThresholdsV57,
    build_adaptive_positive_series_bundle_v57,
)
from fma.v6.executable_candidate import (
    ADAPTIVE_POSITIVE_SERIES_ADAPTER_ID,
    ADAPTIVE_POSITIVE_SERIES_FAMILIES_V62,
    SCALAR_ODE_ADAPTER_ID,
    SCALAR_ODE_FAMILIES_V62,
    ExecutableCandidateError,
    RegisteredFamilySearchIRV62,
    RegisteredFamilySearchIntentV62,
    allowed_family_registry_hash_v62,
    build_executable_candidate_receipt_v62,
    compile_registered_family_search_ir_v62,
    resolve_executable_candidate_v62,
    verify_executable_candidate_receipt_v62,
)


ROOT = Path(__file__).resolve().parents[1]


class _Workspace:
    def __init__(
        self,
        *,
        workspace_id: str,
        s1_gate_hash: str = "1" * 64,
        s2_gate_hash: str | None = None,
        s2_attempt: int = 1,
    ) -> None:
        self.spec = SimpleNamespace(
            workspace_id=workspace_id,
            spec_hash="0" * 64,
        )
        self.s1_gate_hash = s1_gate_hash
        self.s2_gate_hash = s2_gate_hash
        self.s2_attempt = s2_attempt

    def current_gate(self, stage: str) -> str | None:
        if stage == "S1":
            return self.s1_gate_hash
        if stage == "S2":
            return self.s2_gate_hash
        return None

    def _latest_attempt(self, stage: str) -> int:
        if stage != "S2":
            raise KeyError(stage)
        return self.s2_attempt


def _candidate() -> CandidateFormalizationV50:
    return CandidateFormalizationV50(
        candidate_id="candidate.branch_a",
        model_family="registered positive scalar time-series search",
        mathematical_form="dx/dt = f(x; theta)",
        assumption_ids=["assumption.positive"],
        symbol_ids=["symbol.state", "symbol.time"],
        data_requirement_ids=["data.positive_series"],
        validation_obligation_ids=["check.l0", "check.l1"],
        abandon_criteria=["Abandon when registered families fail validation."],
        lineage="blind branch_a proposal",
    )


def _intent(
    candidate_id: str = "candidate.branch_a",
) -> RegisteredFamilySearchIntentV62:
    return RegisteredFamilySearchIntentV62(
        candidate_id=candidate_id,
        allowed_adapter_ids=[
            SCALAR_ODE_ADAPTER_ID,
            ADAPTIVE_POSITIVE_SERIES_ADAPTER_ID,
        ],
    )


def _candidate_set_and_model() -> tuple[
    CandidateSetV50,
    ModelSpecV50,
    RegisteredFamilySearchIRV62,
]:
    candidate = _candidate()
    candidate_set = CandidateSetV50(candidates=[candidate])
    model = ModelSpecV50.seal(
        selected_candidate_id=candidate.candidate_id,
        selected_candidate_structural_hash=candidate.structural_hash(),
        selection_rationale="Best registered and falsifiable search contract.",
        assumption_ids=candidate.assumption_ids,
        symbol_ids=candidate.symbol_ids,
        data_requirement_ids=candidate.data_requirement_ids,
        declared_limit_cases=[
            "Constant-state limit.",
            "Small-rate exponential limit.",
        ],
        identifiability_risks=["Short series may not separate saturation."],
    )
    ir = compile_registered_family_search_ir_v62(candidate, _intent())
    return candidate_set, model, ir


def _resolution(
    workspace: _Workspace,
    adapter_id: str,
):
    candidate_set, model, ir = _candidate_set_and_model()
    return resolve_executable_candidate_v62(
        workspace=workspace,
        execution_ir=ir,
        candidate_set=candidate_set,
        model_spec=model,
        adapter_id=adapter_id,
    )


def _snapshot(
    *,
    task_id: str,
    observations: list[float],
) -> ODETimeSeriesSnapshotV52:
    return ODETimeSeriesSnapshotV52.seal(
        task_id=task_id,
        time_unit="year",
        state_unit="positive_index",
        times=np.arange(len(observations), dtype=float).tolist(),
        observations=observations,
        source_id=f"{task_id}-fixture",
        fixture_only=True,
    )


def test_s1_requires_model_owned_typed_intent_and_never_executes_prose() -> None:
    candidate = _candidate()
    intent = _intent()
    ir = compile_registered_family_search_ir_v62(candidate, intent)

    ir.assert_sealed()
    assert ir.model_intent_hash == intent.content_hash()
    assert ir.candidate_structural_hash == candidate.structural_hash()
    payload = ir.model_dump(mode="json")
    assert candidate.model_family not in str(payload)
    assert candidate.mathematical_form not in str(payload)
    assert ir.model_family_text_executable is False
    assert ir.mathematical_form_text_executable is False
    assert ir.arbitrary_code_execution_permitted is False

    with pytest.raises(ExecutableCandidateError, match="another candidate"):
        compile_registered_family_search_ir_v62(
            candidate,
            _intent("candidate.branch_b"),
        )
    with pytest.raises(ValidationError):
        RegisteredFamilySearchIntentV62.model_validate(
            {
                **intent.model_dump(mode="json"),
                "python_source": "exec(model_family)",
            }
        )
    with pytest.raises(ValidationError):
        RegisteredFamilySearchIRV62.model_validate(
            {
                **ir.model_dump(mode="json"),
                "allowed_adapter_ids": [SCALAR_ODE_ADAPTER_ID],
            }
        )


def test_s2_resolution_binds_current_gate_attempt_and_exact_registry() -> None:
    workspace = _Workspace(
        workspace_id="task-resolution",
        s1_gate_hash="2" * 64,
        s2_attempt=3,
    )
    resolution = _resolution(workspace, SCALAR_ODE_ADAPTER_ID)

    resolution.assert_sealed()
    assert resolution.workspace_spec_hash == workspace.spec.spec_hash
    assert resolution.s1_gate_hash == workspace.s1_gate_hash
    assert resolution.s2_attempt == 3
    assert resolution.allowed_families == list(SCALAR_ODE_FAMILIES_V62)
    assert resolution.allowed_family_registry_hash == (
        allowed_family_registry_hash_v62(SCALAR_ODE_ADAPTER_ID)
    )
    assert resolution.free_text_execution_permitted is False

    workspace.s2_gate_hash = "3" * 64
    with pytest.raises(ExecutableCandidateError, match="before the S2 gate"):
        _resolution(workspace, SCALAR_ODE_ADAPTER_ID)


def test_s3_ode_receipt_closes_registry_selection_and_fixture_authority() -> None:
    workspace = _Workspace(
        workspace_id="task-ode-receipt",
        s1_gate_hash="4" * 64,
        s2_attempt=2,
    )
    resolution = _resolution(workspace, SCALAR_ODE_ADAPTER_ID)
    workspace.s2_gate_hash = "5" * 64
    times = np.arange(24, dtype=float)
    observations = (
        180.0 / (1.0 + 8.0 * np.exp(-0.16 * times))
    ).tolist()
    bundle = build_ode_bundle_v52(
        snapshot=_snapshot(
            task_id=workspace.spec.workspace_id,
            observations=observations,
        ),
        thresholds=ODEThresholdsV52.seal(),
    )

    receipt = build_executable_candidate_receipt_v62(
        workspace=workspace,
        resolution=resolution,
        bundle=bundle,
    )

    receipt.assert_sealed()
    assert receipt.evaluated_families == list(SCALAR_ODE_FAMILIES_V62)
    assert receipt.selected_family == bundle.selected_candidate_id
    assert receipt.selected_model_id == bundle.selected_candidate_id
    assert receipt.candidate_registry_hash == bundle.candidate_registry_hash
    assert receipt.candidate_graph_hash is None
    assert receipt.fixture_only is True
    assert receipt.local_execution_status == "PASS"
    assert receipt.scientific_qualification_status == "NOT_RUN"
    assert receipt.scientific_qualification_granted is False
    assert receipt.real_world_action_authorized is False
    assert verify_executable_candidate_receipt_v62(
        workspace=workspace,
        resolution=resolution,
        bundle=bundle,
        receipt=receipt,
    )

    tampered_payload = bundle.model_dump(exclude={"bundle_hash"})
    tampered_payload["candidate_registry_hash"] = "9" * 64
    tampered = ODEScientificBundleV52.seal(**tampered_payload)
    with pytest.raises(ExecutableCandidateError, match="registry hash"):
        build_executable_candidate_receipt_v62(
            workspace=workspace,
            resolution=resolution,
            bundle=tampered,
        )


def _adaptive_bundle(task_id: str) -> AdaptivePositiveSeriesBundleV57:
    rng = np.random.default_rng(102)
    growths = np.zeros(71, dtype=float)
    growths[0] = 0.04
    for index in range(1, len(growths)):
        growths[index] = 0.04 + rng.normal(0.0, 0.01)
    values = 100.0 * np.exp(np.concatenate(([0.0], np.cumsum(growths))))
    primary = HybridODEThresholdsV56.seal(
        **json.loads(
            (ROOT / "V5_6_HYBRID_THRESHOLDS.json").read_text(
                encoding="utf-8"
            )
        )
    )
    adaptive = AdaptiveThresholdsV57.seal(
        **json.loads(
            (ROOT / "V5_7_ADAPTIVE_THRESHOLDS.json").read_text(
                encoding="utf-8"
            )
        )
    )
    return build_adaptive_positive_series_bundle_v57(
        snapshot=_snapshot(
            task_id=task_id,
            observations=values.tolist(),
        ),
        primary_thresholds=primary,
        adaptive_thresholds=adaptive,
    )


def test_s3_adaptive_receipt_binds_actual_families_and_nested_graphs() -> None:
    workspace = _Workspace(
        workspace_id="task-adaptive-receipt",
        s1_gate_hash="6" * 64,
        s2_attempt=4,
    )
    resolution = _resolution(
        workspace,
        ADAPTIVE_POSITIVE_SERIES_ADAPTER_ID,
    )
    workspace.s2_gate_hash = "7" * 64
    bundle = _adaptive_bundle(workspace.spec.workspace_id)

    receipt = build_executable_candidate_receipt_v62(
        workspace=workspace,
        resolution=resolution,
        bundle=bundle,
    )

    assert resolution.allowed_families == list(
        ADAPTIVE_POSITIVE_SERIES_FAMILIES_V62
    )
    assert receipt.evaluated_families == sorted(
        ADAPTIVE_POSITIVE_SERIES_FAMILIES_V62
    )
    assert receipt.selected_model_id == bundle.graph.selected_model_id
    assert receipt.selected_family in resolution.allowed_families
    assert receipt.candidate_registry_hash == (
        bundle.primary_bundle.candidate_registry_hash
    )
    assert receipt.candidate_graph_hash == bundle.graph.graph_hash
    assert receipt.nested_candidate_graph_hash == (
        bundle.primary_bundle.graph.graph_hash
    )
    assert receipt.fixture_only is True
    assert receipt.scientific_qualification_status == "NOT_RUN"
    assert receipt.scientific_qualification_granted is False
    assert verify_executable_candidate_receipt_v62(
        workspace=workspace,
        resolution=resolution,
        bundle=bundle,
        receipt=receipt,
    )

    graph_payload = bundle.graph.model_dump(exclude={"graph_hash"})
    graph_payload["primary_bundle_hash"] = "8" * 64
    tampered_graph = AdaptiveCandidateGraphV57.seal(**graph_payload)
    bundle_payload = bundle.model_dump(exclude={"bundle_hash"})
    bundle_payload["graph"] = tampered_graph
    tampered_bundle = AdaptivePositiveSeriesBundleV57.seal(**bundle_payload)
    with pytest.raises(ExecutableCandidateError, match="outer graph"):
        build_executable_candidate_receipt_v62(
            workspace=workspace,
            resolution=resolution,
            bundle=tampered_bundle,
        )
