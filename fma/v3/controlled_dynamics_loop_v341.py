from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore
from fma.v2.schemas import Identifier, Sha256, _assert_timezone

from .controlled_dynamics_loop import (
    MECHANISMS_V31,
    PrivateControlledDynamicsWorldPackV31,
    generate_private_controlled_dynamics_worldpack_v31,
)
from .controlled_dynamics_loop_v34 import (
    ControlledDynamicsSelectionBundleV34,
    ControlledDynamicsWorldPackSpecV34,
    InterruptibleRealityEvolutionReportV34,
    InterruptibleRealityPolicyV34,
    _hash_without,
    evaluate_controlled_dynamics_worldpack_v34,
    execute_controlled_dynamics_policy_v34,
)
EXPLORATORY_SEEDS_V341 = (
    15013, 15061, 15107, 15161, 15217, 15263, 15313, 15361,
    15401, 15451, 15511, 15559, 15607, 15661, 15727, 15773,
)


class PersistentMismatchRealityPolicyV341(InterruptibleRealityPolicyV34):
    schema_version: Literal["3.4.1"] = "3.4.1"
    consecutive_exceedances_required: Literal[2] = 2
    confirmation_rule: Literal[
        "two_consecutive_case_local_threshold_exceedances"
    ] = "two_consecutive_case_local_threshold_exceedances"
    prior_v34_failure_report_hash: Sha256


class ControlledDynamicsWorldPackSpecV341(ControlledDynamicsWorldPackSpecV34):
    schema_version: Literal["3.4.1"] = "3.4.1"
    seeds: list[int] = Field(min_length=16, max_length=16)
    bootstrap_seed: Literal[341722] = 341722
    prior_v34_failure_report_hash: Sha256
    frozen_delta: Literal[
        "single_segment_exceedance_to_two_consecutive_exceedances_only"
    ] = "single_segment_exceedance_to_two_consecutive_exceedances_only"

    @model_validator(mode="after")
    def validate_spec(self) -> "ControlledDynamicsWorldPackSpecV341":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.mechanisms != list(MECHANISMS_V31):
            raise ValueError("V3.4.1 requires the frozen mechanism order")
        if tuple(self.seeds) != EXPLORATORY_SEEDS_V341:
            raise ValueError("V3.4.1 seeds do not match the fresh exploratory set")
        if self.goal_initial_state_scales != [0.75, 1.0, 1.25]:
            raise ValueError("V3.4.1 public goal initial-state scales changed")
        if self.maximum_steps != self.action_budget + self.clarification_budget:
            raise ValueError("V3.4.1 resource budgets do not cover maximum steps")
        if not math.isclose(
            (self.trajectory_points - 1) * self.time_step,
            self.segment_count * self.segment_duration,
            abs_tol=1e-12,
        ):
            raise ValueError("V3.4.1 input segments do not cover trajectory")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match V3.4.1 protocol")
        return self

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsWorldPackSpecV341":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"spec_hash"}),
            spec_hash=draft.content_hash(),
        )


class PersistentMismatchEvolutionReportV341(StrictModel):
    schema_version: Literal["3.4.1"] = "3.4.1"
    evolution_id: Identifier
    spec_hash: Sha256
    prior_v34_failure_report_hash: Sha256
    single_component_delta: Literal[
        "single_segment_exceedance_to_two_consecutive_exceedances_only"
    ] = "single_segment_exceedance_to_two_consecutive_exceedances_only"
    base_adapter_report: InterruptibleRealityEvolutionReportV34
    persistent_adapter_ready: bool
    status: Literal[
        "persistent_interaction_adapter_ready_for_acquisition_retest_v341",
        "persistent_interaction_adapter_failed_v341",
    ]
    proposer_changed: Literal[False] = False
    estimator_changed: Literal[False] = False
    reality_adapter_trigger_changed: Literal[True] = True
    exposure_model_changed: Literal[False] = False
    statistical_gates_changed: Literal[False] = False
    router_changed: Literal[False] = False
    overall_qualification_permitted: Literal[False] = False
    confirmation_permitted: Literal[False] = False
    real_world_authorization_permitted: Literal[False] = False
    created_at: datetime
    evolution_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "PersistentMismatchEvolutionReportV341":
        _assert_timezone(self.created_at, "created_at")
        self.base_adapter_report.assert_sealed()
        if self.base_adapter_report.spec_hash != self.spec_hash:
            raise ValueError("V3.4.1 wrapper belongs to another base report")
        ready = self.base_adapter_report.adapter_candidate_ready
        if self.persistent_adapter_ready != ready:
            raise ValueError("V3.4.1 readiness disagrees with frozen gates")
        expected = (
            "persistent_interaction_adapter_ready_for_acquisition_retest_v341"
            if ready else "persistent_interaction_adapter_failed_v341"
        )
        if self.status != expected:
            raise ValueError("V3.4.1 status disagrees with readiness")
        if self.evolution_hash and self.evolution_hash != self.content_hash():
            raise ValueError("evolution_hash does not match V3.4.1 report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evolution_hash")

    def assert_sealed(self) -> None:
        if not self.evolution_hash or self.evolution_hash != self.content_hash():
            raise ValueError("V3.4.1 report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PersistentMismatchEvolutionReportV341":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evolution_hash"}),
            evolution_hash=draft.content_hash(),
        )


class ControlledDynamicsManifestV341(StrictModel):
    schema_version: Literal["3.4.1"] = "3.4.1"
    run_id: Identifier
    artifact_refs: list[ArtifactRef] = Field(min_length=7)
    terminal_status: Literal[
        "persistent_interaction_adapter_ready_for_acquisition_retest_v341",
        "persistent_interaction_adapter_failed_v341",
    ]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "ControlledDynamicsManifestV341":
        _assert_timezone(self.created_at, "created_at")
        if len({item.kind for item in self.artifact_refs}) != len(self.artifact_refs):
            raise ValueError("V3.4.1 manifest artifact kinds must be unique")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match V3.4.1 manifest")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("V3.4.1 manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsManifestV341":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class ControlledDynamicsOutcomeV341:
    store: RunStore
    spec: ControlledDynamicsWorldPackSpecV341
    private_pack: PrivateControlledDynamicsWorldPackV31
    baseline_policy: PersistentMismatchRealityPolicyV341
    candidate_policy: PersistentMismatchRealityPolicyV341
    baseline_bundle: ControlledDynamicsSelectionBundleV34
    candidate_bundle: ControlledDynamicsSelectionBundleV34
    evolution_report: PersistentMismatchEvolutionReportV341
    manifest: ControlledDynamicsManifestV341


def default_controlled_dynamics_policies_v341(
    *,
    prior_v34_failure_report_hash: str,
    prior_v332_failure_report_hash: str,
    prior_epistemic_qualification_hash: str,
    prior_active_design_qualification_hash: str,
    method_evidence_hash: str,
) -> tuple[PersistentMismatchRealityPolicyV341, PersistentMismatchRealityPolicyV341]:
    shared = dict(
        prior_v34_failure_report_hash=prior_v34_failure_report_hash,
        prior_v332_failure_report_hash=prior_v332_failure_report_hash,
        prior_epistemic_qualification_hash=prior_epistemic_qualification_hash,
        prior_active_design_qualification_hash=prior_active_design_qualification_hash,
        method_evidence_hash=method_evidence_hash,
    )
    return (
        PersistentMismatchRealityPolicyV341.seal(
            policy_id="unguarded_paired_advantage_v341",
            arm="unguarded_full_action",
            execution_rule="execute_selected_action_without_online_interruption",
            **shared,
        ),
        PersistentMismatchRealityPolicyV341.seal(
            policy_id="persistent_interruptible_paired_advantage_v341",
            arm="interruptible_online_guard",
            execution_rule="segment_authorization_then_monotone_zero_fallback",
            **shared,
        ),
    )


def default_controlled_dynamics_exploratory_spec_v341(
    *,
    baseline_policy_hash: str,
    candidate_policy_hash: str,
    method_evidence_hash: str,
    prior_epistemic_qualification_hash: str,
    prior_active_design_qualification_hash: str,
    prior_v332_failure_report_hash: str,
    prior_v34_failure_report_hash: str,
    frozen_at: datetime | None = None,
) -> ControlledDynamicsWorldPackSpecV341:
    return ControlledDynamicsWorldPackSpecV341.seal(
        experiment_id="controlled_dynamics_persistent_mismatch_exploratory_v341",
        mechanisms=list(MECHANISMS_V31),
        seeds=list(EXPLORATORY_SEEDS_V341),
        baseline_policy_hash=baseline_policy_hash,
        candidate_policy_hash=candidate_policy_hash,
        method_evidence_hash=method_evidence_hash,
        prior_epistemic_qualification_hash=prior_epistemic_qualification_hash,
        prior_active_design_qualification_hash=prior_active_design_qualification_hash,
        prior_v332_failure_report_hash=prior_v332_failure_report_hash,
        prior_v34_failure_report_hash=prior_v34_failure_report_hash,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


def evaluate_controlled_dynamics_worldpack_v341(
    spec: ControlledDynamicsWorldPackSpecV341,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    baseline: ControlledDynamicsSelectionBundleV34,
    candidate: ControlledDynamicsSelectionBundleV34,
    *,
    evaluated_at: datetime | None = None,
) -> PersistentMismatchEvolutionReportV341:
    at = evaluated_at or datetime.now(timezone.utc)
    base = evaluate_controlled_dynamics_worldpack_v34(
        spec, private_pack, baseline, candidate, evaluated_at=at
    )
    ready = base.adapter_candidate_ready
    return PersistentMismatchEvolutionReportV341.seal(
        evolution_id="controlled_dynamics_persistent_mismatch_exploratory_v341",
        spec_hash=spec.spec_hash,
        prior_v34_failure_report_hash=spec.prior_v34_failure_report_hash,
        base_adapter_report=base,
        persistent_adapter_ready=ready,
        status=(
            "persistent_interaction_adapter_ready_for_acquisition_retest_v341"
            if ready else "persistent_interaction_adapter_failed_v341"
        ),
        created_at=at,
    )


def run_controlled_dynamics_worldpack_v341(
    output_root: str | Path,
    *,
    spec: ControlledDynamicsWorldPackSpecV341,
    baseline_policy: PersistentMismatchRealityPolicyV341,
    candidate_policy: PersistentMismatchRealityPolicyV341,
    evaluated_at: datetime | None = None,
    run_id: str | None = None,
) -> ControlledDynamicsOutcomeV341:
    spec.assert_sealed()
    baseline_policy.assert_sealed()
    candidate_policy.assert_sealed()
    if baseline_policy.policy_hash != spec.baseline_policy_hash:
        raise ValueError("V3.4.1 baseline is not frozen in protocol")
    if candidate_policy.policy_hash != spec.candidate_policy_hash:
        raise ValueError("V3.4.1 candidate is not frozen in protocol")
    at = evaluated_at or datetime.now(timezone.utc)
    store = RunStore(
        output_root,
        run_id=run_id or f"controlled-dynamics-v341-{uuid4().hex[:10]}",
    )
    refs = [
        store.put_artifact("controlled_dynamics_spec_v341", spec),
        store.put_artifact("controlled_dynamics_baseline_policy_v341", baseline_policy),
        store.put_artifact("controlled_dynamics_candidate_policy_v341", candidate_policy),
    ]
    store.emit("controlled_dynamics_v341_protocol_frozen_before_private_pack", {
        "spec_hash": spec.spec_hash,
        "prior_v34_failure_report_hash": spec.prior_v34_failure_report_hash,
        "single_component_delta": spec.frozen_delta,
    })
    private_pack = generate_private_controlled_dynamics_worldpack_v31(
        spec, generated_at=at
    )
    baseline = execute_controlled_dynamics_policy_v34(
        spec, private_pack, baseline_policy, executed_at=at
    )
    candidate = execute_controlled_dynamics_policy_v34(
        spec, private_pack, candidate_policy, executed_at=at
    )
    evolution = evaluate_controlled_dynamics_worldpack_v341(
        spec, private_pack, baseline, candidate, evaluated_at=at
    )
    refs.extend([
        store.put_artifact("private_controlled_dynamics_worldpack_v341", private_pack),
        store.put_artifact("controlled_dynamics_baseline_bundle_v341", baseline),
        store.put_artifact("controlled_dynamics_candidate_bundle_v341", candidate),
        store.put_artifact(
            "controlled_dynamics_base_adapter_report_v341",
            evolution.base_adapter_report,
        ),
        store.put_artifact("controlled_dynamics_evolution_report_v341", evolution),
    ])
    manifest = ControlledDynamicsManifestV341.seal(
        run_id=store.run_id,
        artifact_refs=refs,
        terminal_status=evolution.status,
        created_at=at,
    )
    manifest_ref = store.put_artifact("controlled_dynamics_manifest_v341", manifest)
    store.emit("controlled_dynamics_v341_worldpack_adjudicated", {
        "manifest_ref": manifest_ref.model_dump(mode="json")
    })
    if not verify_controlled_dynamics_run_v341(store.run_directory):
        raise RuntimeError("V3.4.1 run failed independent verification")
    return ControlledDynamicsOutcomeV341(
        store,
        spec,
        private_pack,
        baseline_policy,
        candidate_policy,
        baseline,
        candidate,
        evolution,
        manifest,
    )


def verify_controlled_dynamics_run_v341(run_directory: str | Path) -> bool:
    try:
        store = RunStore.open_existing(run_directory)
        events = [
            json.loads(line)
            for line in store.event_path.read_text(encoding="utf-8").splitlines()
        ]
        committed = [
            ArtifactRef.model_validate(event["payload"])
            for event in events if event["event_type"] == "artifact_committed"
        ]
        for reference in committed:
            store.load_artifact(reference)
        manifest_refs = [
            item for item in committed
            if item.kind == "controlled_dynamics_manifest_v341"
        ]
        if len(manifest_refs) != 1:
            return False
        manifest = ControlledDynamicsManifestV341.model_validate(
            store.load_artifact(manifest_refs[0])
        )
        manifest.assert_sealed()

        def load_one(kind: str, model):
            references = [
                item for item in manifest.artifact_refs if item.kind == kind
            ]
            if len(references) != 1:
                raise RuntimeError(f"V3.4.1 manifest needs exactly one {kind}")
            return model.model_validate(store.load_artifact(references[0]))

        spec = load_one(
            "controlled_dynamics_spec_v341", ControlledDynamicsWorldPackSpecV341
        )
        baseline_policy = load_one(
            "controlled_dynamics_baseline_policy_v341",
            PersistentMismatchRealityPolicyV341,
        )
        candidate_policy = load_one(
            "controlled_dynamics_candidate_policy_v341",
            PersistentMismatchRealityPolicyV341,
        )
        private_pack = load_one(
            "private_controlled_dynamics_worldpack_v341",
            PrivateControlledDynamicsWorldPackV31,
        )
        baseline = load_one(
            "controlled_dynamics_baseline_bundle_v341",
            ControlledDynamicsSelectionBundleV34,
        )
        candidate = load_one(
            "controlled_dynamics_candidate_bundle_v341",
            ControlledDynamicsSelectionBundleV34,
        )
        base_report = load_one(
            "controlled_dynamics_base_adapter_report_v341",
            InterruptibleRealityEvolutionReportV34,
        )
        evolution = load_one(
            "controlled_dynamics_evolution_report_v341",
            PersistentMismatchEvolutionReportV341,
        )
        for artifact in (
            spec, baseline_policy, candidate_policy, private_pack,
            baseline, candidate, base_report, evolution, manifest,
        ):
            artifact.assert_sealed()
        if manifest.run_id != store.run_id:
            return False
        regenerated = generate_private_controlled_dynamics_worldpack_v31(
            spec, generated_at=private_pack.generated_at
        )
        if regenerated.pack_hash != private_pack.pack_hash:
            return False
        executed_at = baseline.case_receipts[0].executed_at
        replay_baseline = execute_controlled_dynamics_policy_v34(
            spec, private_pack, baseline_policy, executed_at=executed_at
        )
        replay_candidate = execute_controlled_dynamics_policy_v34(
            spec, private_pack, candidate_policy, executed_at=executed_at
        )
        if replay_baseline.bundle_hash != baseline.bundle_hash or (
            replay_candidate.bundle_hash != candidate.bundle_hash
        ):
            return False
        recomputed = evaluate_controlled_dynamics_worldpack_v341(
            spec,
            private_pack,
            baseline,
            candidate,
            evaluated_at=evolution.created_at,
        )
        if recomputed.evolution_hash != evolution.evolution_hash or (
            evolution.base_adapter_report.evolution_hash != base_report.evolution_hash
        ):
            return False
        if any(
            "qualification" in item.kind or "confirmation" in item.kind
            for item in manifest.artifact_refs
        ):
            return False
        freezes = [
            event for event in events
            if event["event_type"]
            == "controlled_dynamics_v341_protocol_frozen_before_private_pack"
        ]
        private_events = [
            event for event in events
            if event["event_type"] == "artifact_committed"
            and event["payload"]["kind"] == "private_controlled_dynamics_worldpack_v341"
        ]
        return (
            len(freezes) == 1
            and len(private_events) == 1
            and freezes[0]["sequence"] < private_events[0]["sequence"]
            and store.verify_event_chain()
        )
    except (
        OSError, RuntimeError, ValueError, KeyError, TypeError,
        json.JSONDecodeError, FloatingPointError,
    ):
        return False
