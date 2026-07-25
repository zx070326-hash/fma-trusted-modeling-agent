from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from fma.schemas import ArtifactRef
from fma.storage import RunStore
from fma.v3.controlled_dynamics_loop import (
    PrivateControlledDynamicsWorldPackV31,
    generate_private_controlled_dynamics_worldpack_v31,
)
from fma.v3.skeleton_factorial_v310 import (
    FactorialDecisionV310,
    SkeletonFactorialBundleV310,
    SkeletonFactorialEvolutionReportV310,
    SkeletonFactorialManifestV310,
    SkeletonFactorialMethodEvidenceV310,
    SkeletonFactorialPolicyV310,
    SkeletonFactorialWorldPackSpecV310,
    _execute_case_v310,
    _fit_v310,
    _observations_v310,
    default_skeleton_factorial_method_evidence_v310,
    default_skeleton_factorial_policy_v310,
    default_skeleton_factorial_spec_v310,
    verify_skeleton_factorial_run_v310,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_V391 = (
    ROOT / "experiments" / "iteration_17"
    / "v391_evaluator_partition_recovery"
)
VALID_V310 = (
    ROOT / "experiments" / "iteration_18"
    / "v310_skeleton_factorial"
)


@pytest.fixture(scope="module")
def synthetic_protocol():
    evidence = default_skeleton_factorial_method_evidence_v310()
    policy = default_skeleton_factorial_policy_v310(evidence, "0" * 64)
    spec = default_skeleton_factorial_spec_v310(
        evidence, policy, frozen_at=datetime(2026, 7, 22, tzinfo=timezone.utc)
    )
    pack = generate_private_controlled_dynamics_worldpack_v31(
        spec, generated_at=spec.frozen_at
    )
    return evidence, policy, spec, pack


@pytest.fixture(scope="module")
def v310_artifacts():
    store = RunStore.open_existing(VALID_V310)
    events = [
        json.loads(line)
        for line in store.event_path.read_text(encoding="utf-8").splitlines()
    ]
    refs = [
        ArtifactRef.model_validate(event["payload"])
        for event in events if event["event_type"] == "artifact_committed"
    ]

    def load(kind: str, model):
        ref = next(item for item in refs if item.kind == kind)
        return model.model_validate(store.load_artifact(ref))

    return {
        "store": store,
        "evidence": load(
            "skeleton_factorial_method_evidence_v310",
            SkeletonFactorialMethodEvidenceV310,
        ),
        "policy": load(
            "skeleton_factorial_policy_v310", SkeletonFactorialPolicyV310
        ),
        "spec": load(
            "skeleton_factorial_spec_v310", SkeletonFactorialWorldPackSpecV310
        ),
        "pack": load(
            "private_skeleton_factorial_worldpack_v310",
            PrivateControlledDynamicsWorldPackV31,
        ),
        "bundle": load(
            "skeleton_factorial_bundle_v310", SkeletonFactorialBundleV310
        ),
        "report": load(
            "skeleton_factorial_evolution_report_v310",
            SkeletonFactorialEvolutionReportV310,
        ),
        "manifest": load(
            "skeleton_factorial_manifest_v310", SkeletonFactorialManifestV310
        ),
    }


def test_v310_protocol_is_sealed_and_private_blind(synthetic_protocol) -> None:
    evidence, policy, spec, _ = synthetic_protocol
    evidence.assert_sealed()
    policy.assert_sealed()
    spec.assert_sealed()
    assert not policy.private_mechanism_visible
    assert not policy.private_probe_visible
    assert not policy.private_target_loss_visible
    assert not policy.private_performance_eligible_used_for_partition
    assert not policy.task_router_permitted
    assert not policy.real_world_execution_permitted


def test_v310_second_order_skeleton_enforces_kinematics(synthetic_protocol) -> None:
    _, _, spec, pack = synthetic_protocol
    case = next(
        item for item in pack.cases
        if item.mechanism == "duffing_oscillator"
        and item.public_case.case_id.endswith("_25073")
    )
    fit = _fit_v310(
        case,
        _observations_v310(case, spec),
        "second_order_kinematic_force_law",
        "integral_trapezoid_ridge",
        spec,
        suffix="test",
    )
    coefficients = np.asarray(fit.model.coefficient_matrix)
    assert [item.term_id for item in fit.model.basis_terms] == [
        "position", "velocity", "position3"
    ]
    assert np.array_equal(coefficients[0], np.asarray([0.0, 1.0, 0.0]))
    assert fit.rank_ratio == 1.0


def test_v310_case_factorial_uses_public_evidence_only(synthetic_protocol) -> None:
    _, policy, spec, pack = synthetic_protocol
    case = next(
        item for item in pack.cases
        if item.mechanism == "duffing_oscillator"
        and item.public_case.case_id.endswith("_25073")
    )
    receipt = _execute_case_v310(case, policy, spec, spec.frozen_at)
    assert len(receipt.cells) == 16
    assert len(receipt.pairs) == 8
    assert all(
        binding.contract_valid and binding.segment_count == 6
        for cell in receipt.cells
        for binding in cell.input_bindings
    )
    assert all(not cell.private_values_used for cell in receipt.cells)
    assert isinstance(receipt.candidate_decision, FactorialDecisionV310)
    assert not receipt.candidate_decision.private_values_used


def test_v310_formal_report_respects_frozen_gates(v310_artifacts) -> None:
    report = v310_artifacts["report"]
    assert report.performance_case_count == 39
    assert report.quality_case_count == 9
    assert report.valid_input_binding_count == report.expected_input_binding_count == 1872
    assert report.public_quality_partition_only
    assert not report.private_performance_eligible_used_for_partition
    assert report.ready_for_cross_domain_skeleton_confirmation == all(
        report.gates.values()
    )
    assert not report.task_router_permitted
    assert not report.model_qualification_permitted
    assert not report.real_world_execution_permitted


def test_v310_replays_and_tampering_fails_closed(v310_artifacts, tmp_path) -> None:
    assert len(v310_artifacts["manifest"].artifact_refs) == 6
    assert verify_skeleton_factorial_run_v310(
        v310_artifacts["store"].run_directory,
        source_v391_run_directory=SOURCE_V391,
    )
    copied = tmp_path / "tampered_v310"
    shutil.copytree(v310_artifacts["store"].run_directory, copied)
    events = [
        json.loads(line)
        for line in (copied / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    target = next(
        event["payload"]
        for event in events
        if event["event_type"] == "artifact_committed"
        and event["payload"]["kind"] == "skeleton_factorial_bundle_v310"
    )
    path = copied / target["relative_path"]
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["bundle_id"] = "tampered_bundle"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    assert not verify_skeleton_factorial_run_v310(
        copied, source_v391_run_directory=SOURCE_V391
    )
