from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore
from fma.v2.schemas import Identifier, Sha256, _assert_timezone

from .controlled_dynamics_loop import (
    ArmV31,
    CONFIRMATION_SEEDS_V31,
    MECHANISMS_V31,
    ControlledDynamicsPolicyV31,
    ControlledDynamicsReportV31,
    ControlledDynamicsSelectionBundleV31,
    PrivateControlledDynamicsWorldPackV31,
    _execute_case_v31,
    evaluate_controlled_dynamics_worldpack_v31,
    generate_private_controlled_dynamics_worldpack_v31,
)


EVOLVED_EXPLORATORY_SEEDS_V311 = (
    8501, 8551, 8609, 8653, 8707, 8753, 8803, 8851,
)


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class SequentialControlledDynamicsPolicyV311(StrictModel):
    schema_version: Literal["3.1.1"] = "3.1.1"
    policy_id: Identifier
    arm: ArmV31
    selection_rule: Literal[
        "prefrozen_random_without_replacement",
        "clarify_then_goal_information_risk_utility",
    ]
    may_reformulate_problem: bool
    maximum_actions: Literal[3] = 3
    known_actuator_required: Literal[True] = True
    prior_v31_failure_report_hash: Sha256
    prior_epistemic_qualification_hash: Sha256
    prior_active_design_qualification_hash: Sha256
    method_evidence_hash: Sha256
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "SequentialControlledDynamicsPolicyV311":
        expected = {
            "random_bounded_inputs": (
                "prefrozen_random_without_replacement", False
            ),
            "goal_oriented_epistemic_control": (
                "clarify_then_goal_information_risk_utility", True
            ),
        }[self.arm]
        if (self.selection_rule, self.may_reformulate_problem) != expected:
            raise ValueError("V3.1.1 arm and policy behavior disagree")
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("policy_hash does not match V3.1.1 policy")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("V3.1.1 policy is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "SequentialControlledDynamicsPolicyV311":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"policy_hash"}),
            policy_hash=draft.content_hash(),
        )


class ControlledDynamicsWorldPackSpecV311(StrictModel):
    schema_version: Literal["3.1.1"] = "3.1.1"
    experiment_id: Identifier
    phase: Literal["exploratory", "confirmation"]
    mechanisms: list[Literal[
        "exponential_decay", "logistic_growth", "damped_oscillator", "duffing_oscillator"
    ]] = Field(min_length=4, max_length=4)
    seeds: list[int] = Field(min_length=8, max_length=20)
    action_budget: Literal[3] = 3
    trajectory_points: Literal[49] = 49
    time_step: Literal[0.04] = 0.04
    segment_count: Literal[6] = 6
    segment_duration: Literal[0.32] = 0.32
    input_amplitude: Literal[0.35] = 0.35
    observation_noise_fraction: Literal[0.01] = 0.01
    polynomial_degree: Literal[2] = 2
    savgol_window: Literal[9] = 9
    savgol_order: Literal[3] = 3
    ridge_alpha: Literal[0.0001] = 0.0001
    sparsity_threshold: Literal[0.02] = 0.02
    ensemble_members: Literal[12] = 12
    bootstrap_fraction: Literal[0.8] = 0.8
    maximum_empirical_prediction_risk: Literal[0.25] = 0.25
    model_mismatch_residual_threshold: Literal[0.24] = 0.24
    bootstrap_replicates: Annotated[int, Field(ge=200, le=5000)] = 1200
    bootstrap_seed: int = 311722
    minimum_macro_loss_improvement: Literal[0.0] = 0.0
    maximum_mechanism_regression: Literal[0.02] = 0.02
    material_negative_transfer: Literal[0.02] = 0.02
    maximum_negative_transfer_rate: Literal[0.1] = 0.1
    required_routing_accuracy: Literal[1.0] = 1.0
    baseline_policy_hash: Sha256
    candidate_policy_hash: Sha256
    method_evidence_hash: Sha256
    prior_epistemic_qualification_hash: Sha256
    prior_active_design_qualification_hash: Sha256
    prior_v31_failure_report_hash: Sha256
    frozen_delta: Literal["action_horizon_2_to_3_only"] = "action_horizon_2_to_3_only"
    frozen_at: datetime
    spec_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "ControlledDynamicsWorldPackSpecV311":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.mechanisms != list(MECHANISMS_V31):
            raise ValueError("V3.1.1 requires the frozen mechanism order")
        expected = (
            set(EVOLVED_EXPLORATORY_SEEDS_V311)
            if self.phase == "exploratory"
            else set(CONFIRMATION_SEEDS_V31)
        )
        if set(self.seeds) != expected or len(self.seeds) != len(expected):
            raise ValueError("V3.1.1 seeds do not match the frozen phase")
        if not math.isclose(
            (self.trajectory_points - 1) * self.time_step,
            self.segment_count * self.segment_duration,
            abs_tol=1e-12,
        ):
            raise ValueError("V3.1.1 input segments do not cover the trajectory")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match V3.1.1 protocol")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "spec_hash")

    def assert_sealed(self) -> None:
        if not self.spec_hash or self.spec_hash != self.content_hash():
            raise ValueError("V3.1.1 protocol is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsWorldPackSpecV311":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"spec_hash"}),
            spec_hash=draft.content_hash(),
        )


class ControlledDynamicsEvolutionReportV311(StrictModel):
    schema_version: Literal["3.1.1"] = "3.1.1"
    evolution_id: Identifier
    spec_hash: Sha256
    base_adjudication_report: ControlledDynamicsReportV31
    prior_v31_failure_report_hash: Sha256
    single_component_delta: Literal["action_horizon_2_to_3_only"] = (
        "action_horizon_2_to_3_only"
    )
    baseline_input_experiments_per_clean_case: Literal[3] = 3
    candidate_input_experiments_after_clarification: Literal[2] = 2
    estimator_changed: Literal[False] = False
    acquisition_changed: Literal[False] = False
    risk_gate_changed: Literal[False] = False
    model_router_changed: Literal[False] = False
    created_at: datetime
    evolution_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_evolution(self) -> "ControlledDynamicsEvolutionReportV311":
        _assert_timezone(self.created_at, "created_at")
        self.base_adjudication_report.assert_sealed()
        if self.base_adjudication_report.spec_hash != self.spec_hash:
            raise ValueError("V3.1.1 wrapper is bound to another adjudication")
        if self.evolution_hash and self.evolution_hash != self.content_hash():
            raise ValueError("evolution_hash does not match V3.1.1 report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evolution_hash")

    def assert_sealed(self) -> None:
        if not self.evolution_hash or self.evolution_hash != self.content_hash():
            raise ValueError("V3.1.1 evolution report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsEvolutionReportV311":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evolution_hash"}),
            evolution_hash=draft.content_hash(),
        )


class ControlledDynamicsQualificationV311(StrictModel):
    schema_version: Literal["3.1.1"] = "3.1.1"
    qualification_id: Literal[
        "synthetic_known_actuator_goal_oriented_experiment_routing_v311"
    ] = "synthetic_known_actuator_goal_oriented_experiment_routing_v311"
    policy_hash: Sha256
    evolution_report_hash: Sha256
    base_adjudication_report_hash: Sha256
    scope: Literal[
        "synthetic_known_actuator_three_step_worldpack_v311"
    ] = "synthetic_known_actuator_three_step_worldpack_v311"
    known_actuator_only: Literal[True] = True
    empirical_risk_only: Literal[True] = True
    formal_safety_proven: Literal[False] = False
    structural_identifiability_proven: Literal[False] = False
    real_world_validity_proven: Literal[False] = False
    qualified_at: datetime
    qualification_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_qualification(self) -> "ControlledDynamicsQualificationV311":
        _assert_timezone(self.qualified_at, "qualified_at")
        if self.qualification_hash and self.qualification_hash != self.content_hash():
            raise ValueError("qualification_hash does not match V3.1.1 qualification")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "qualification_hash")

    def assert_sealed(self) -> None:
        if not self.qualification_hash or self.qualification_hash != self.content_hash():
            raise ValueError("V3.1.1 qualification is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsQualificationV311":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"qualification_hash"}),
            qualification_hash=draft.content_hash(),
        )


class ControlledDynamicsManifestV311(StrictModel):
    schema_version: Literal["3.1.1"] = "3.1.1"
    run_id: Identifier
    artifact_refs: list[ArtifactRef] = Field(min_length=8)
    terminal_status: Literal[
        "exploratory_only_v31",
        "candidate_rejected_v31",
        "promoted_for_synthetic_controlled_epistemic_worldpack_v31",
    ]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "ControlledDynamicsManifestV311":
        _assert_timezone(self.created_at, "created_at")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match V3.1.1 manifest")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("V3.1.1 manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsManifestV311":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class ControlledDynamicsOutcomeV311:
    store: RunStore
    spec: ControlledDynamicsWorldPackSpecV311
    private_pack: PrivateControlledDynamicsWorldPackV31
    baseline_policy: SequentialControlledDynamicsPolicyV311
    candidate_policy: SequentialControlledDynamicsPolicyV311
    baseline_bundle: ControlledDynamicsSelectionBundleV31
    candidate_bundle: ControlledDynamicsSelectionBundleV31
    evolution_report: ControlledDynamicsEvolutionReportV311
    qualification: ControlledDynamicsQualificationV311 | None
    manifest: ControlledDynamicsManifestV311


def default_controlled_dynamics_policies_v311(
    *,
    prior_v31_failure_report_hash: str,
    prior_epistemic_qualification_hash: str,
    prior_active_design_qualification_hash: str,
    method_evidence_hash: str,
) -> tuple[SequentialControlledDynamicsPolicyV311, SequentialControlledDynamicsPolicyV311]:
    shared = dict(
        prior_v31_failure_report_hash=prior_v31_failure_report_hash,
        prior_epistemic_qualification_hash=prior_epistemic_qualification_hash,
        prior_active_design_qualification_hash=prior_active_design_qualification_hash,
        method_evidence_hash=method_evidence_hash,
    )
    return (
        SequentialControlledDynamicsPolicyV311.seal(
            policy_id="random_bounded_inputs_v311",
            arm="random_bounded_inputs",
            selection_rule="prefrozen_random_without_replacement",
            may_reformulate_problem=False,
            **shared,
        ),
        SequentialControlledDynamicsPolicyV311.seal(
            policy_id="goal_oriented_epistemic_control_v311",
            arm="goal_oriented_epistemic_control",
            selection_rule="clarify_then_goal_information_risk_utility",
            may_reformulate_problem=True,
            **shared,
        ),
    )


def _default_spec_v311(
    phase: Literal["exploratory", "confirmation"],
    *,
    baseline_policy_hash: str,
    candidate_policy_hash: str,
    method_evidence_hash: str,
    prior_epistemic_qualification_hash: str,
    prior_active_design_qualification_hash: str,
    prior_v31_failure_report_hash: str,
    frozen_at: datetime | None = None,
) -> ControlledDynamicsWorldPackSpecV311:
    return ControlledDynamicsWorldPackSpecV311.seal(
        experiment_id=f"controlled_dynamics_{phase}_v311",
        phase=phase,
        mechanisms=list(MECHANISMS_V31),
        seeds=list(
            EVOLVED_EXPLORATORY_SEEDS_V311
            if phase == "exploratory"
            else CONFIRMATION_SEEDS_V31
        ),
        baseline_policy_hash=baseline_policy_hash,
        candidate_policy_hash=candidate_policy_hash,
        method_evidence_hash=method_evidence_hash,
        prior_epistemic_qualification_hash=prior_epistemic_qualification_hash,
        prior_active_design_qualification_hash=prior_active_design_qualification_hash,
        prior_v31_failure_report_hash=prior_v31_failure_report_hash,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


def default_controlled_dynamics_exploratory_spec_v311(**kwargs: object) -> ControlledDynamicsWorldPackSpecV311:
    return _default_spec_v311("exploratory", **kwargs)


def default_controlled_dynamics_confirmation_spec_v311(**kwargs: object) -> ControlledDynamicsWorldPackSpecV311:
    return _default_spec_v311("confirmation", **kwargs)


def execute_controlled_dynamics_policy_v311(
    spec: ControlledDynamicsWorldPackSpecV311,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    policy: SequentialControlledDynamicsPolicyV311,
    *,
    executed_at: datetime | None = None,
) -> ControlledDynamicsSelectionBundleV31:
    spec.assert_sealed()
    private_pack.assert_sealed()
    policy.assert_sealed()
    if private_pack.spec_hash != spec.spec_hash:
        raise ValueError("V3.1.1 private pack belongs to another protocol")
    expected = spec.baseline_policy_hash if policy.arm == "random_bounded_inputs" else spec.candidate_policy_hash
    if policy.policy_hash != expected:
        raise ValueError("V3.1.1 policy is not frozen in the protocol")
    at = executed_at or datetime.now(timezone.utc)
    receipts = [
        _execute_case_v31(spec, private_case, policy, executed_at=at)
        for private_case in private_pack.cases
    ]
    return ControlledDynamicsSelectionBundleV31.seal(
        spec_hash=spec.spec_hash,
        private_pack_hash=private_pack.pack_hash,
        policy_hash=policy.policy_hash,
        arm=policy.arm,
        case_receipts=receipts,
        total_action_budget_consumed=sum(item.action_budget_consumed for item in receipts),
        total_abstentions=sum(item.abstention_count for item in receipts),
    )


def evaluate_controlled_dynamics_worldpack_v311(
    spec: ControlledDynamicsWorldPackSpecV311,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    baseline: ControlledDynamicsSelectionBundleV31,
    candidate: ControlledDynamicsSelectionBundleV31,
    *,
    evaluated_at: datetime | None = None,
) -> ControlledDynamicsEvolutionReportV311:
    report = evaluate_controlled_dynamics_worldpack_v31(
        spec, private_pack, baseline, candidate, evaluated_at=evaluated_at
    )
    return ControlledDynamicsEvolutionReportV311.seal(
        evolution_id=f"controlled_dynamics_evolution_{spec.phase}_v311",
        spec_hash=spec.spec_hash,
        base_adjudication_report=report,
        prior_v31_failure_report_hash=spec.prior_v31_failure_report_hash,
        created_at=evaluated_at or datetime.now(timezone.utc),
    )


def run_controlled_dynamics_worldpack_v311(
    output_root: str | Path,
    *,
    spec: ControlledDynamicsWorldPackSpecV311,
    baseline_policy: SequentialControlledDynamicsPolicyV311,
    candidate_policy: SequentialControlledDynamicsPolicyV311,
    evaluated_at: datetime | None = None,
    run_id: str | None = None,
) -> ControlledDynamicsOutcomeV311:
    spec.assert_sealed()
    baseline_policy.assert_sealed()
    candidate_policy.assert_sealed()
    if baseline_policy.policy_hash != spec.baseline_policy_hash:
        raise ValueError("V3.1.1 baseline is not frozen in the protocol")
    if candidate_policy.policy_hash != spec.candidate_policy_hash:
        raise ValueError("V3.1.1 candidate is not frozen in the protocol")
    at = evaluated_at or datetime.now(timezone.utc)
    store = RunStore(
        output_root,
        run_id=run_id or f"controlled-dynamics-v311-{uuid4().hex[:10]}",
    )
    refs = [
        store.put_artifact("controlled_dynamics_spec_v311", spec),
        store.put_artifact("controlled_dynamics_baseline_policy_v311", baseline_policy),
        store.put_artifact("controlled_dynamics_candidate_policy_v311", candidate_policy),
    ]
    store.emit("controlled_dynamics_v311_protocol_frozen_before_private_pack", {
        "spec_hash": spec.spec_hash,
        "prior_v31_failure_report_hash": spec.prior_v31_failure_report_hash,
        "single_component_delta": spec.frozen_delta,
    })
    private_pack = generate_private_controlled_dynamics_worldpack_v31(spec, generated_at=at)
    baseline = execute_controlled_dynamics_policy_v311(
        spec, private_pack, baseline_policy, executed_at=at
    )
    candidate = execute_controlled_dynamics_policy_v311(
        spec, private_pack, candidate_policy, executed_at=at
    )
    evolution = evaluate_controlled_dynamics_worldpack_v311(
        spec, private_pack, baseline, candidate, evaluated_at=at
    )
    refs.extend([
        store.put_artifact("private_controlled_dynamics_worldpack_v311", private_pack),
        store.put_artifact("controlled_dynamics_baseline_bundle_v311", baseline),
        store.put_artifact("controlled_dynamics_candidate_bundle_v311", candidate),
        store.put_artifact("controlled_dynamics_base_report_v311", evolution.base_adjudication_report),
        store.put_artifact("controlled_dynamics_evolution_report_v311", evolution),
    ])
    qualification = None
    if evolution.base_adjudication_report.status == "promoted_for_synthetic_controlled_epistemic_worldpack_v31":
        qualification = ControlledDynamicsQualificationV311.seal(
            policy_hash=candidate_policy.policy_hash,
            evolution_report_hash=evolution.evolution_hash,
            base_adjudication_report_hash=evolution.base_adjudication_report.report_hash,
            qualified_at=at,
        )
        refs.append(store.put_artifact("controlled_dynamics_qualification_v311", qualification))
    manifest = ControlledDynamicsManifestV311.seal(
        run_id=store.run_id,
        artifact_refs=refs,
        terminal_status=evolution.base_adjudication_report.status,
        created_at=at,
    )
    manifest_ref = store.put_artifact("controlled_dynamics_manifest_v311", manifest)
    store.emit("controlled_dynamics_v311_worldpack_adjudicated", {
        "manifest_ref": manifest_ref.model_dump(mode="json")
    })
    if not verify_controlled_dynamics_run_v311(store.run_directory):
        raise RuntimeError("V3.1.1 controlled-dynamics run failed independent verification")
    return ControlledDynamicsOutcomeV311(
        store, spec, private_pack, baseline_policy, candidate_policy,
        baseline, candidate, evolution, qualification, manifest,
    )


def verify_controlled_dynamics_run_v311(run_directory: str | Path) -> bool:
    try:
        store = RunStore.open_existing(run_directory)
        events = [json.loads(line) for line in store.event_path.read_text(encoding="utf-8").splitlines()]
        committed = [
            ArtifactRef.model_validate(event["payload"])
            for event in events if event["event_type"] == "artifact_committed"
        ]
        for reference in committed:
            store.load_artifact(reference)
        manifest_refs = [item for item in committed if item.kind == "controlled_dynamics_manifest_v311"]
        if len(manifest_refs) != 1:
            return False
        manifest = ControlledDynamicsManifestV311.model_validate(store.load_artifact(manifest_refs[0]))
        manifest.assert_sealed()

        def load_one(kind: str, model):
            references = [item for item in manifest.artifact_refs if item.kind == kind]
            if len(references) != 1:
                raise RuntimeError(f"V3.1.1 manifest needs exactly one {kind}")
            return model.model_validate(store.load_artifact(references[0]))

        spec = load_one("controlled_dynamics_spec_v311", ControlledDynamicsWorldPackSpecV311)
        baseline_policy = load_one("controlled_dynamics_baseline_policy_v311", SequentialControlledDynamicsPolicyV311)
        candidate_policy = load_one("controlled_dynamics_candidate_policy_v311", SequentialControlledDynamicsPolicyV311)
        private_pack = load_one("private_controlled_dynamics_worldpack_v311", PrivateControlledDynamicsWorldPackV31)
        baseline = load_one("controlled_dynamics_baseline_bundle_v311", ControlledDynamicsSelectionBundleV31)
        candidate = load_one("controlled_dynamics_candidate_bundle_v311", ControlledDynamicsSelectionBundleV31)
        base_report = load_one("controlled_dynamics_base_report_v311", ControlledDynamicsReportV31)
        evolution = load_one("controlled_dynamics_evolution_report_v311", ControlledDynamicsEvolutionReportV311)
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
        replay_baseline = execute_controlled_dynamics_policy_v311(
            spec, private_pack, baseline_policy, executed_at=executed_at
        )
        replay_candidate = execute_controlled_dynamics_policy_v311(
            spec, private_pack, candidate_policy, executed_at=executed_at
        )
        if replay_baseline.bundle_hash != baseline.bundle_hash or replay_candidate.bundle_hash != candidate.bundle_hash:
            return False
        recomputed = evaluate_controlled_dynamics_worldpack_v311(
            spec, private_pack, baseline, candidate, evaluated_at=base_report.evaluated_at
        )
        if recomputed.evolution_hash != evolution.evolution_hash:
            return False
        qualifications = [item for item in manifest.artifact_refs if item.kind == "controlled_dynamics_qualification_v311"]
        promoted = base_report.status == "promoted_for_synthetic_controlled_epistemic_worldpack_v31"
        if promoted:
            if len(qualifications) != 1:
                return False
            qualification = ControlledDynamicsQualificationV311.model_validate(
                store.load_artifact(qualifications[0])
            )
            qualification.assert_sealed()
            if qualification.evolution_report_hash != evolution.evolution_hash:
                return False
        elif qualifications:
            return False
        freeze_events = [
            event for event in events
            if event["event_type"] == "controlled_dynamics_v311_protocol_frozen_before_private_pack"
        ]
        return len(freeze_events) == 1 and store.verify_event_chain()
    except (
        OSError, RuntimeError, ValueError, KeyError, TypeError,
        json.JSONDecodeError, FloatingPointError,
    ):
        return False
