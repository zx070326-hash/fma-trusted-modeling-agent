from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Annotated, Literal
from uuid import uuid4

import numpy as np
from pydantic import Field, model_validator
from scipy.stats import beta

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore

from .dynamics_active_design import (
    ActiveDesignPolicyV26,
    ActiveDesignSelectionBundleV26,
    ActiveDesignWorldPackSpecV26,
    PrivateActiveDesignWorldPackV26,
    _hidden_model_metrics_v26,
    _joint_loss,
    assert_single_component_active_design_v26,
    execute_active_design_policy_v26,
    generate_private_active_design_worldpack_v26,
)
from .dynamics_worldpack import Mechanism
from .schemas import Identifier, Sha256, _assert_timezone


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class ActiveDesignEffectProtocolV261(StrictModel):
    schema_version: Literal["2.6.1"] = "2.6.1"
    protocol_id: Identifier
    phase: Literal["exploratory", "confirmation"]
    base_spec_hash: Sha256
    prior_metric_failure_report_hash: Sha256
    prior_effect_report_hash: Sha256 | None = None
    effect_measure: Literal["bounded_joint_loss_absolute_difference"] = (
        "bounded_joint_loss_absolute_difference"
    )
    practical_negative_margin: Annotated[
        float, Field(gt=0, le=0.1, allow_inf_nan=False)
    ] = 0.01
    minimum_macro_improvement: Literal[0.0] = 0.0
    maximum_negative_transfer_rate: Annotated[
        float, Field(gt=0, le=0.5, allow_inf_nan=False)
    ] = 0.25
    maximum_mechanism_regression: Annotated[
        float, Field(ge=0, le=0.05, allow_inf_nan=False)
    ] = 0.01
    require_positive_information_gain: Literal[True] = True
    frozen_at: datetime
    protocol_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_protocol(self) -> "ActiveDesignEffectProtocolV261":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.phase == "exploratory" and self.prior_effect_report_hash is not None:
            raise ValueError("V2.6.1 exploratory protocol cannot bind its own result")
        if self.phase == "confirmation" and self.prior_effect_report_hash is None:
            raise ValueError("V2.6.1 confirmation requires exploratory effect evidence")
        if self.protocol_hash and self.protocol_hash != self.content_hash():
            raise ValueError("protocol_hash does not match V2.6.1 effect protocol")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "protocol_hash")

    def assert_sealed(self) -> None:
        if not self.protocol_hash or self.protocol_hash != self.content_hash():
            raise ValueError("V2.6.1 effect protocol is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ActiveDesignEffectProtocolV261":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"protocol_hash"}),
            protocol_hash=draft.content_hash(),
        )


class ActiveDesignEffectCaseResultV261(StrictModel):
    case_id: Identifier
    mechanism: Mechanism
    baseline_parameter_error: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    active_parameter_error: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    baseline_support_f1: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    active_support_f1: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    baseline_probe_nrmse: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    active_probe_nrmse: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    baseline_joint_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    active_joint_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    joint_loss_improvement: Annotated[float, Field(allow_inf_nan=False)]
    log_condition_improvement: Annotated[float, Field(allow_inf_nan=False)]
    material_negative_transfer: bool


class ActiveDesignEffectMechanismResultV261(StrictModel):
    mechanism: Mechanism
    case_count: Annotated[int, Field(ge=2)]
    mean_joint_loss_improvement: Annotated[float, Field(allow_inf_nan=False)]
    mean_log_condition_improvement: Annotated[float, Field(allow_inf_nan=False)]


class ActiveDesignEffectReportV261(StrictModel):
    schema_version: Literal["2.6.1"] = "2.6.1"
    base_spec_hash: Sha256
    protocol_hash: Sha256
    private_pack_hash: Sha256
    baseline_bundle_hash: Sha256
    active_bundle_hash: Sha256
    cases: list[ActiveDesignEffectCaseResultV261] = Field(min_length=8)
    mechanisms: list[ActiveDesignEffectMechanismResultV261] = Field(
        min_length=4, max_length=4
    )
    same_action_and_fit_budget: bool
    invalid_action_count: Annotated[int, Field(ge=0)]
    material_negative_transfer_count: Annotated[int, Field(ge=0)]
    macro_joint_loss_improvement: Annotated[float, Field(allow_inf_nan=False)]
    macro_joint_loss_improvement_ci_lower: Annotated[
        float, Field(allow_inf_nan=False)
    ]
    macro_joint_loss_improvement_ci_upper: Annotated[
        float, Field(allow_inf_nan=False)
    ]
    negative_transfer_rate_upper: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ]
    macro_log_condition_improvement: Annotated[float, Field(allow_inf_nan=False)]
    status: Literal[
        "exploratory_only",
        "candidate_rejected_active_design_v261",
        "promoted_for_synthetic_active_design_worldpack_v261",
    ]
    reason_codes: list[str]
    evaluated_at: datetime
    report_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "ActiveDesignEffectReportV261":
        _assert_timezone(self.evaluated_at, "evaluated_at")
        if self.report_hash and self.report_hash != self.content_hash():
            raise ValueError("report_hash does not match V2.6.1 effect report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "report_hash")

    def assert_sealed(self) -> None:
        if not self.report_hash or self.report_hash != self.content_hash():
            raise ValueError("V2.6.1 effect report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ActiveDesignEffectReportV261":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"report_hash"}),
            report_hash=draft.content_hash(),
        )


class ActiveDesignEffectQualificationV261(StrictModel):
    schema_version: Literal["2.6.1"] = "2.6.1"
    qualification_id: Identifier
    qualification_scope: Literal["synthetic_safe_initial_condition_design_v261"] = (
        "synthetic_safe_initial_condition_design_v261"
    )
    active_policy_hash: Sha256
    report_hash: Sha256
    real_world_validity_established: Literal[False] = False
    structural_identifiability_proven: Literal[False] = False
    continuous_control_qualified: Literal[False] = False
    qualified_at: datetime
    qualification_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_qualification(self) -> "ActiveDesignEffectQualificationV261":
        _assert_timezone(self.qualified_at, "qualified_at")
        if self.qualification_hash and self.qualification_hash != self.content_hash():
            raise ValueError("qualification_hash does not match V2.6.1 qualification")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "qualification_hash")

    def assert_sealed(self) -> None:
        if not self.qualification_hash or self.qualification_hash != self.content_hash():
            raise ValueError("V2.6.1 qualification is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ActiveDesignEffectQualificationV261":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"qualification_hash"}),
            qualification_hash=draft.content_hash(),
        )


class ActiveDesignEffectManifestV261(StrictModel):
    schema_version: Literal["2.6.1"] = "2.6.1"
    run_id: Identifier
    artifact_refs: list[ArtifactRef] = Field(min_length=8)
    terminal_status: Literal[
        "exploratory_only",
        "candidate_rejected_active_design_v261",
        "promoted_for_synthetic_active_design_worldpack_v261",
    ]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "ActiveDesignEffectManifestV261":
        _assert_timezone(self.created_at, "created_at")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match V2.6.1 manifest")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("V2.6.1 manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ActiveDesignEffectManifestV261":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class ActiveDesignEffectOutcomeV261:
    store: RunStore
    spec: ActiveDesignWorldPackSpecV26
    protocol: ActiveDesignEffectProtocolV261
    private_pack: PrivateActiveDesignWorldPackV26
    baseline_policy: ActiveDesignPolicyV26
    active_policy: ActiveDesignPolicyV26
    baseline: ActiveDesignSelectionBundleV26
    active: ActiveDesignSelectionBundleV26
    report: ActiveDesignEffectReportV261
    qualification: ActiveDesignEffectQualificationV261 | None
    manifest: ActiveDesignEffectManifestV261


def default_exploratory_effect_protocol_v261(
    *,
    base_spec_hash: str,
    prior_metric_failure_report_hash: str,
    frozen_at: datetime | None = None,
) -> ActiveDesignEffectProtocolV261:
    return ActiveDesignEffectProtocolV261.seal(
        protocol_id="bounded_effect_exploratory_v261",
        phase="exploratory",
        base_spec_hash=base_spec_hash,
        prior_metric_failure_report_hash=prior_metric_failure_report_hash,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


def default_confirmation_effect_protocol_v261(
    *,
    base_spec_hash: str,
    prior_metric_failure_report_hash: str,
    prior_effect_report_hash: str,
    frozen_at: datetime | None = None,
) -> ActiveDesignEffectProtocolV261:
    return ActiveDesignEffectProtocolV261.seal(
        protocol_id="bounded_effect_confirmation_v261",
        phase="confirmation",
        base_spec_hash=base_spec_hash,
        prior_metric_failure_report_hash=prior_metric_failure_report_hash,
        prior_effect_report_hash=prior_effect_report_hash,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


def evaluate_active_design_effect_v261(
    spec: ActiveDesignWorldPackSpecV26,
    protocol: ActiveDesignEffectProtocolV261,
    private_pack: PrivateActiveDesignWorldPackV26,
    baseline: ActiveDesignSelectionBundleV26,
    active: ActiveDesignSelectionBundleV26,
    *,
    evaluated_at: datetime | None = None,
) -> ActiveDesignEffectReportV261:
    spec.assert_sealed()
    protocol.assert_sealed()
    private_pack.assert_sealed()
    baseline.assert_sealed()
    active.assert_sealed()
    if protocol.base_spec_hash != spec.spec_hash or protocol.phase != spec.phase:
        raise ValueError("V2.6.1 protocol is bound to another base experiment")
    if baseline.private_pack_hash != private_pack.pack_hash:
        raise ValueError("V2.6.1 baseline is bound to another private pack")
    if active.private_pack_hash != private_pack.pack_hash:
        raise ValueError("V2.6.1 active bundle is bound to another private pack")
    baseline_by_case = {item.case_id: item for item in baseline.case_receipts}
    active_by_case = {item.case_id: item for item in active.case_receipts}
    results: list[ActiveDesignEffectCaseResultV261] = []
    grouped_improvement: dict[str, list[float]] = {
        mechanism: [] for mechanism in spec.mechanisms
    }
    grouped_condition: dict[str, list[float]] = {
        mechanism: [] for mechanism in spec.mechanisms
    }
    for case in private_pack.cases:
        case_id = case.public_case.case_id
        base_receipt = baseline_by_case[case_id]
        active_receipt = active_by_case[case_id]
        base_metrics = _hidden_model_metrics_v26(case, base_receipt.final_model, spec)
        active_metrics = _hidden_model_metrics_v26(case, active_receipt.final_model, spec)
        base_loss = _joint_loss(*base_metrics)
        active_loss = _joint_loss(*active_metrics)
        improvement = base_loss - active_loss
        condition_improvement = math.log1p(
            base_receipt.final_model.normalized_condition_number
        ) - math.log1p(active_receipt.final_model.normalized_condition_number)
        negative = active_loss - base_loss > protocol.practical_negative_margin
        result = ActiveDesignEffectCaseResultV261(
            case_id=case_id,
            mechanism=case.public_case.mechanism,
            baseline_parameter_error=base_metrics[0],
            active_parameter_error=active_metrics[0],
            baseline_support_f1=base_metrics[1],
            active_support_f1=active_metrics[1],
            baseline_probe_nrmse=base_metrics[2],
            active_probe_nrmse=active_metrics[2],
            baseline_joint_loss=base_loss,
            active_joint_loss=active_loss,
            joint_loss_improvement=improvement,
            log_condition_improvement=condition_improvement,
            material_negative_transfer=negative,
        )
        results.append(result)
        grouped_improvement[result.mechanism].append(improvement)
        grouped_condition[result.mechanism].append(condition_improvement)
    mechanisms = [
        ActiveDesignEffectMechanismResultV261(
            mechanism=mechanism,
            case_count=len(grouped_improvement[mechanism]),
            mean_joint_loss_improvement=float(np.mean(grouped_improvement[mechanism])),
            mean_log_condition_improvement=float(np.mean(grouped_condition[mechanism])),
        )
        for mechanism in spec.mechanisms
    ]
    macro = float(np.mean([item.mean_joint_loss_improvement for item in mechanisms]))
    macro_condition = float(
        np.mean([item.mean_log_condition_improvement for item in mechanisms])
    )
    draws = _stratified_macro_bootstrap(grouped_improvement, spec)
    alpha = (1.0 - spec.confidence_level) / 2.0
    ci_lower, ci_upper = np.quantile(draws, [alpha, 1.0 - alpha], method="linear")
    negative_count = sum(item.material_negative_transfer for item in results)
    negative_upper = _clopper_pearson_upper(
        negative_count, len(results), spec.confidence_level
    )
    same_budget = (
        baseline.total_action_budget == active.total_action_budget
        and all(
            baseline_by_case[item.case_id].action_budget_consumed
            == active_by_case[item.case_id].action_budget_consumed
            == spec.action_budget
            for item in results
        )
    )
    invalid_count = baseline.invalid_action_count + active.invalid_action_count
    reasons: list[str] = []
    if spec.phase == "exploratory":
        status = "exploratory_only"
        reasons.append("exploratory_not_eligible")
    else:
        if not same_budget:
            reasons.append("unequal_action_or_fit_budget")
        if invalid_count:
            reasons.append("invalid_or_unsafe_action")
        if float(ci_lower) <= protocol.minimum_macro_improvement:
            reasons.append("macro_improvement_gate_failed")
        if negative_upper > protocol.maximum_negative_transfer_rate:
            reasons.append("negative_transfer_gate_failed")
        if any(
            item.mean_joint_loss_improvement < -protocol.maximum_mechanism_regression
            for item in mechanisms
        ):
            reasons.append("mechanism_noninferiority_gate_failed")
        if protocol.require_positive_information_gain and macro_condition <= 0:
            reasons.append("empirical_information_gate_failed")
        status = (
            "candidate_rejected_active_design_v261"
            if reasons
            else "promoted_for_synthetic_active_design_worldpack_v261"
        )
    return ActiveDesignEffectReportV261.seal(
        base_spec_hash=spec.spec_hash,
        protocol_hash=protocol.protocol_hash,
        private_pack_hash=private_pack.pack_hash,
        baseline_bundle_hash=baseline.bundle_hash,
        active_bundle_hash=active.bundle_hash,
        cases=results,
        mechanisms=mechanisms,
        same_action_and_fit_budget=same_budget,
        invalid_action_count=invalid_count,
        material_negative_transfer_count=negative_count,
        macro_joint_loss_improvement=macro,
        macro_joint_loss_improvement_ci_lower=float(ci_lower),
        macro_joint_loss_improvement_ci_upper=float(ci_upper),
        negative_transfer_rate_upper=negative_upper,
        macro_log_condition_improvement=macro_condition,
        status=status,
        reason_codes=reasons,
        evaluated_at=evaluated_at or datetime.now(timezone.utc),
    )


def qualify_active_design_effect_v261(
    active_policy: ActiveDesignPolicyV26,
    report: ActiveDesignEffectReportV261,
    *,
    qualified_at: datetime | None = None,
) -> ActiveDesignEffectQualificationV261:
    active_policy.assert_sealed()
    report.assert_sealed()
    if report.status != "promoted_for_synthetic_active_design_worldpack_v261":
        raise ValueError("cannot qualify a rejected V2.6.1 active-design effect")
    return ActiveDesignEffectQualificationV261.seal(
        qualification_id="synthetic_safe_initial_condition_design_v261",
        active_policy_hash=active_policy.policy_hash,
        report_hash=report.report_hash,
        qualified_at=qualified_at or datetime.now(timezone.utc),
    )


def run_active_design_effect_worldpack_v261(
    output_root: str | Path,
    *,
    spec: ActiveDesignWorldPackSpecV26,
    protocol: ActiveDesignEffectProtocolV261,
    baseline_policy: ActiveDesignPolicyV26,
    active_policy: ActiveDesignPolicyV26,
    evaluated_at: datetime | None = None,
    run_id: str | None = None,
) -> ActiveDesignEffectOutcomeV261:
    spec.assert_sealed()
    protocol.assert_sealed()
    assert_single_component_active_design_v26(baseline_policy, active_policy)
    if protocol.base_spec_hash != spec.spec_hash:
        raise ValueError("V2.6.1 protocol is bound to another base spec")
    if baseline_policy.policy_hash != spec.baseline_policy_hash:
        raise ValueError("V2.6.1 baseline policy is not frozen in the spec")
    if active_policy.policy_hash != spec.active_policy_hash:
        raise ValueError("V2.6.1 active policy is not frozen in the spec")
    at = evaluated_at or datetime.now(timezone.utc)
    store = RunStore(
        output_root, run_id=run_id or f"dynamics-active-effect-{uuid4().hex[:10]}"
    )
    refs = [
        store.put_artifact("active_design_spec_v26", spec),
        store.put_artifact("active_design_effect_protocol_v261", protocol),
        store.put_artifact("active_design_baseline_policy_v26", baseline_policy),
        store.put_artifact("active_design_candidate_policy_v26", active_policy),
    ]
    store.emit(
        "active_design_effect_protocol_frozen_before_private_pack",
        {
            "spec_hash": spec.spec_hash,
            "protocol_hash": protocol.protocol_hash,
            "frozen_delta": "evaluation_effect_measure_only",
        },
    )
    private_pack = generate_private_active_design_worldpack_v26(spec, generated_at=at)
    baseline = execute_active_design_policy_v26(
        spec, private_pack, baseline_policy, selected_at=at
    )
    active = execute_active_design_policy_v26(
        spec, private_pack, active_policy, selected_at=at
    )
    report = evaluate_active_design_effect_v261(
        spec, protocol, private_pack, baseline, active, evaluated_at=at
    )
    refs.extend(
        [
            store.put_artifact("private_active_design_worldpack_v26", private_pack),
            store.put_artifact("active_design_baseline_bundle_v26", baseline),
            store.put_artifact("active_design_candidate_bundle_v26", active),
            store.put_artifact("active_design_effect_report_v261", report),
        ]
    )
    qualification = None
    if report.status == "promoted_for_synthetic_active_design_worldpack_v261":
        qualification = qualify_active_design_effect_v261(
            active_policy, report, qualified_at=at
        )
        refs.append(
            store.put_artifact("active_design_effect_qualification_v261", qualification)
        )
    manifest = ActiveDesignEffectManifestV261.seal(
        run_id=store.run_id,
        artifact_refs=refs,
        terminal_status=report.status,
        created_at=at,
    )
    manifest_ref = store.put_artifact("active_design_effect_manifest_v261", manifest)
    store.emit(
        "active_design_effect_worldpack_adjudicated",
        {"manifest_ref": manifest_ref.model_dump(mode="json")},
    )
    if not verify_active_design_effect_run_v261(store.run_directory):
        raise RuntimeError("V2.6.1 active-design effect run failed verification")
    return ActiveDesignEffectOutcomeV261(
        store,
        spec,
        protocol,
        private_pack,
        baseline_policy,
        active_policy,
        baseline,
        active,
        report,
        qualification,
        manifest,
    )


def verify_active_design_effect_run_v261(run_directory: str | Path) -> bool:
    try:
        store = RunStore.open_existing(run_directory)
        events = [
            json.loads(line)
            for line in store.event_path.read_text(encoding="utf-8").splitlines()
        ]
        committed = [
            ArtifactRef.model_validate(event["payload"])
            for event in events
            if event["event_type"] == "artifact_committed"
        ]
        for ref in committed:
            store.load_artifact(ref)
        manifest_refs = [
            ref for ref in committed if ref.kind == "active_design_effect_manifest_v261"
        ]
        if len(manifest_refs) != 1:
            return False
        manifest = ActiveDesignEffectManifestV261.model_validate(
            store.load_artifact(manifest_refs[0])
        )
        manifest.assert_sealed()
        if manifest.run_id != store.run_id:
            return False

        def load_one(kind: str, model):
            refs = [ref for ref in manifest.artifact_refs if ref.kind == kind]
            if len(refs) != 1:
                raise RuntimeError(f"manifest needs exactly one {kind}")
            return model.model_validate(store.load_artifact(refs[0]))

        spec = load_one("active_design_spec_v26", ActiveDesignWorldPackSpecV26)
        protocol = load_one(
            "active_design_effect_protocol_v261", ActiveDesignEffectProtocolV261
        )
        baseline_policy = load_one(
            "active_design_baseline_policy_v26", ActiveDesignPolicyV26
        )
        active_policy = load_one(
            "active_design_candidate_policy_v26", ActiveDesignPolicyV26
        )
        private_pack = load_one(
            "private_active_design_worldpack_v26", PrivateActiveDesignWorldPackV26
        )
        baseline = load_one(
            "active_design_baseline_bundle_v26", ActiveDesignSelectionBundleV26
        )
        active = load_one(
            "active_design_candidate_bundle_v26", ActiveDesignSelectionBundleV26
        )
        report = load_one(
            "active_design_effect_report_v261", ActiveDesignEffectReportV261
        )
        for item in (
            spec,
            protocol,
            baseline_policy,
            active_policy,
            private_pack,
            baseline,
            active,
            report,
        ):
            item.assert_sealed()
        regenerated = generate_private_active_design_worldpack_v26(
            spec, generated_at=private_pack.generated_at
        )
        if regenerated.pack_hash != private_pack.pack_hash:
            return False
        selected_at = baseline.case_receipts[0].selected_at
        replay_baseline = execute_active_design_policy_v26(
            spec, private_pack, baseline_policy, selected_at=selected_at
        )
        replay_active = execute_active_design_policy_v26(
            spec, private_pack, active_policy, selected_at=selected_at
        )
        if replay_baseline.bundle_hash != baseline.bundle_hash:
            return False
        if replay_active.bundle_hash != active.bundle_hash:
            return False
        recomputed = evaluate_active_design_effect_v261(
            spec,
            protocol,
            private_pack,
            baseline,
            active,
            evaluated_at=report.evaluated_at,
        )
        if recomputed.report_hash != report.report_hash:
            return False
        qualification_refs = [
            ref
            for ref in manifest.artifact_refs
            if ref.kind == "active_design_effect_qualification_v261"
        ]
        if report.status == "promoted_for_synthetic_active_design_worldpack_v261":
            if len(qualification_refs) != 1:
                return False
            qualification = ActiveDesignEffectQualificationV261.model_validate(
                store.load_artifact(qualification_refs[0])
            )
            qualification.assert_sealed()
            if qualification.report_hash != report.report_hash:
                return False
        elif qualification_refs:
            return False
        freeze_events = [
            event
            for event in events
            if event["event_type"]
            == "active_design_effect_protocol_frozen_before_private_pack"
        ]
        return len(freeze_events) == 1 and store.verify_event_chain()
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        FloatingPointError,
    ):
        return False


def _stratified_macro_bootstrap(
    grouped: dict[str, list[float]],
    spec: ActiveDesignWorldPackSpecV26,
) -> np.ndarray:
    random = Random(spec.bootstrap_seed + 261)
    draws = np.empty(spec.bootstrap_replicates, dtype=float)
    for draw in range(spec.bootstrap_replicates):
        mechanism_means = []
        for mechanism in spec.mechanisms:
            values = grouped[mechanism]
            mechanism_means.append(
                sum(values[random.randrange(len(values))] for _ in values) / len(values)
            )
        draws[draw] = sum(mechanism_means) / len(mechanism_means)
    return draws


def _clopper_pearson_upper(successes: int, trials: int, confidence: float) -> float:
    if successes == trials:
        return 1.0
    return float(beta.ppf(confidence, successes + 1, trials - successes))

