from __future__ import annotations

import inspect
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fma.schemas import ArtifactRef
from fma.storage import RunStore
from fma.v3.representation_invariant_topology_v311 import (
    PrivateTopologyWorldPackV311,
    PrivateTopologyWorldPackSpecV311,
    PublicTopologyProtocolV311,
    PublicTopologyWorldPackV311,
    RepresentationTopologyManifestV311,
    RepresentationTopologyReportV311,
    TopologyDiscoveryBundleV311,
    TopologyDiscoveryPolicyV311,
    default_private_topology_spec_v311,
    default_public_topology_protocol_v311,
    default_representation_method_evidence_v311,
    default_topology_discovery_policy_v311,
    execute_topology_discovery_v311,
    generate_topology_worldpacks_v311,
    verify_representation_topology_run_v311,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_V310 = (
    ROOT / "experiments" / "iteration_18" / "v310_skeleton_factorial"
)
DEVELOPMENT_V311 = (
    ROOT / "experiments" / "iteration_19"
    / "v311_representation_topology_development_time_recovered"
)
CONFIRMATION_V311 = (
    ROOT / "experiments" / "iteration_19"
    / "v311_representation_topology_confirmation"
)


def _load_artifacts(run_directory: Path) -> dict[str, object]:
    store = RunStore.open_existing(run_directory)
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
        "policy": load(
            "topology_discovery_policy_v311", TopologyDiscoveryPolicyV311
        ),
        "protocol": load(
            "public_topology_protocol_v311", PublicTopologyProtocolV311
        ),
        "spec": load(
            "private_topology_worldpack_spec_v311",
            PrivateTopologyWorldPackSpecV311,
        ),
        "public_pack": load(
            "public_topology_worldpack_v311", PublicTopologyWorldPackV311
        ),
        "private_pack": load(
            "private_topology_worldpack_v311", PrivateTopologyWorldPackV311
        ),
        "bundle": load(
            "topology_discovery_bundle_v311", TopologyDiscoveryBundleV311
        ),
        "report": load(
            "representation_topology_report_v311",
            RepresentationTopologyReportV311,
        ),
        "manifest": load(
            "representation_topology_manifest_v311",
            RepresentationTopologyManifestV311,
        ),
    }


@pytest.fixture(scope="module")
def development_artifacts() -> dict[str, object]:
    return _load_artifacts(DEVELOPMENT_V311)


@pytest.fixture(scope="module")
def confirmation_artifacts() -> dict[str, object]:
    return _load_artifacts(CONFIRMATION_V311)


def test_v311_public_pack_is_anonymous_and_private_blind() -> None:
    evidence = default_representation_method_evidence_v311()
    policy = default_topology_discovery_policy_v311(evidence, "0" * 64)
    protocol = default_public_topology_protocol_v311(
        evidence, policy, frozen_at=datetime(2026, 7, 22, tzinfo=timezone.utc)
    )
    spec = default_private_topology_spec_v311(
        protocol,
        "0" * 64,
        phase="development",
        frozen_at=datetime(2026, 7, 22, 0, 1, tzinfo=timezone.utc),
    )
    public_pack, private_pack = generate_topology_worldpacks_v311(
        spec, protocol, generated_at=spec.frozen_at
    )
    assert all(
        case.state_names == [f"z{index}" for index in range(len(case.state_names))]
        and not case.semantic_state_labels_available
        and not case.representation_metadata_available
        for case in public_pack.cases
    )
    assert {case.representation for case in private_pack.cases} == {
        "anonymous_reference", "anonymous_scaled_permuted"
    }
    assert not policy.private_mechanism_visible
    assert not policy.private_representation_visible
    assert not policy.private_pair_id_visible
    assert not policy.private_probe_visible
    assert not policy.private_loss_visible
    assert not policy.task_router_permitted
    assert not policy.real_world_execution_permitted


def test_v311_executor_contract_excludes_private_worldpack(
    confirmation_artifacts: dict[str, object],
) -> None:
    parameters = inspect.signature(execute_topology_discovery_v311).parameters
    assert list(parameters)[:3] == ["public_protocol", "public_pack", "policy"]
    assert all("private" not in name for name in parameters)
    bundle = confirmation_artifacts["bundle"]
    assert all(
        not receipt.candidate_decision.private_values_used
        and all(
            not challenge.private_values_used
            and not challenge.semantic_state_labels_used
            for challenge in receipt.challenges
        )
        for receipt in bundle.case_receipts
    )


def test_v311_development_is_diagnostic_only(
    development_artifacts: dict[str, object],
) -> None:
    report = development_artifacts["report"]
    assert report.phase == "development"
    assert report.status == "representation_topology_development_diagnostic_v311"
    assert not report.ready_for_next_confirmation
    assert all(report.gates.values())
    assert report.performance_case_count == 50
    assert report.quality_case_count == 10
    assert report.open_set_case_count == 10
    assert report.open_set_abstention_rate >= 0.9


def test_v311_confirmation_respects_all_frozen_gates(
    confirmation_artifacts: dict[str, object],
) -> None:
    report = confirmation_artifacts["report"]
    manifest = confirmation_artifacts["manifest"]
    protocol = confirmation_artifacts["protocol"]
    spec = confirmation_artifacts["spec"]
    assert report.phase == "confirmation"
    assert report.ready_for_next_confirmation == all(report.gates.values())
    assert report.status in {
        "representation_topology_confirmed_v311",
        "representation_topology_refuted_v311",
    }
    assert report.performance_case_count == 70
    assert report.quality_case_count == 10
    assert report.open_set_case_count == 14
    assert not report.task_router_permitted
    assert not report.model_qualification_permitted
    assert not report.real_world_execution_permitted
    assert manifest.terminal_status == report.status
    assert len(manifest.artifact_refs) == 8
    assert protocol.frozen_at <= spec.frozen_at <= report.created_at


def test_v311_replays_and_tampering_fails_closed(
    confirmation_artifacts: dict[str, object], tmp_path: Path,
) -> None:
    assert verify_representation_topology_run_v311(
        CONFIRMATION_V311,
        source_v310_run_directory=SOURCE_V310,
        development_run_directory=DEVELOPMENT_V311,
    )
    copied = tmp_path / "tampered_v311"
    shutil.copytree(confirmation_artifacts["store"].run_directory, copied)
    events = [
        json.loads(line)
        for line in (copied / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    target = next(
        event["payload"]
        for event in events
        if event["event_type"] == "artifact_committed"
        and event["payload"]["kind"] == "topology_discovery_bundle_v311"
    )
    path = copied / target["relative_path"]
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["bundle_id"] = "tampered_bundle"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    assert not verify_representation_topology_run_v311(
        copied,
        source_v310_run_directory=SOURCE_V310,
        development_run_directory=DEVELOPMENT_V311,
    )
