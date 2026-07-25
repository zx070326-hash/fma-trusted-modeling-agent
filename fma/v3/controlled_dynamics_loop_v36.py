from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

import numpy as np
from pydantic import Field, model_validator

from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore
from fma.v2.schemas import Identifier, Sha256, _assert_timezone

from .controlled_dynamics_loop import (
    MECHANISMS_V31,
    PrivateControlledDynamicsCaseV31,
    PrivateControlledDynamicsWorldPackV31,
    _fit_model_v31,
    _target_loss_v31,
    generate_private_controlled_dynamics_worldpack_v31,
)
from .controlled_dynamics_loop_v34 import (
    AdapterArmV34,
    ControlledDynamicsCaseReceiptV34,
    ControlledDynamicsSelectionBundleV34,
    PlanAbstentionV34,
    _calibration_v34,
    _hash_without,
    _select_v332_plan_v34,
    _stream_intervention_v34,
)
from .controlled_dynamics_loop_v332 import (
    ControlledDynamicsSelectionBundleV332,
    PairedAdvantageEvolutionReportV332,
    verify_controlled_dynamics_run_v332,
)
from .controlled_dynamics_loop_v35 import (
    ControlledDynamicsWorldPackSpecV35,
    GuardedAcquisitionEvolutionReportV35,
    verify_controlled_dynamics_run_v35,
)


EXPLORATORY_SEEDS_V36 = (
    17011, 17053, 17107, 17159, 17207, 17257, 17317, 17359,
    17417, 17467, 17519, 17573, 17627, 17681, 17729, 17783,
)

CALIBRATION_CUTOFFS_V36 = (0.03, 0.06, 0.09, 0.12, 0.15)

CalibrationRoleV36 = Literal[
    "shared_random_baseline",
    "original_guarded_acquisition",
    "outcome_calibrated_guarded_acquisition",
]


class OutcomeCalibrationRowV36(StrictModel):
    schema_version: Literal["3.6"] = "3.6"
    source_version: Literal["v332", "v35"]
    source_report_hash: Sha256
    case_id: Identifier
    paired_advantage_q20: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    realized_acquisition_gain: Annotated[float, Field(allow_inf_nan=False)]
    material_negative_transfer: bool
    private_training_outcome: Literal[True] = True

    @model_validator(mode="after")
    def validate_row(self) -> "OutcomeCalibrationRowV36":
        if self.material_negative_transfer != (
            self.realized_acquisition_gain < -0.02
        ):
            raise ValueError("V3.6 training row negative-transfer flag disagrees")
        return self


class OutcomeCutoffSummaryV36(StrictModel):
    schema_version: Literal["3.6"] = "3.6"
    q20_cutoff: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    selected_row_count: Annotated[int, Field(ge=1)]
    mean_realized_gain: Annotated[float, Field(allow_inf_nan=False)]
    bootstrap_ci_low: Annotated[float, Field(allow_inf_nan=False)]
    bootstrap_ci_high: Annotated[float, Field(allow_inf_nan=False)]
    material_negative_transfer_count: Annotated[int, Field(ge=0)]


def _cutoff_summaries_v36(
    rows: list[OutcomeCalibrationRowV36],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> list[OutcomeCutoffSummaryV36]:
    summaries: list[OutcomeCutoffSummaryV36] = []
    for index, cutoff in enumerate(CALIBRATION_CUTOFFS_V36):
        values = np.asarray([
            item.realized_acquisition_gain
            for item in rows if item.paired_advantage_q20 >= cutoff
        ], dtype=float)
        if not len(values):
            raise ValueError("V3.6 cutoff has no training rows")
        random = np.random.default_rng(bootstrap_seed + index)
        samples = random.integers(
            0, len(values), size=(bootstrap_replicates, len(values))
        )
        bootstrap = np.mean(values[samples], axis=1)
        low, high = np.quantile(bootstrap, [0.025, 0.975])
        summaries.append(OutcomeCutoffSummaryV36(
            q20_cutoff=cutoff,
            selected_row_count=len(values),
            mean_realized_gain=float(np.mean(values)),
            bootstrap_ci_low=float(low),
            bootstrap_ci_high=float(high),
            material_negative_transfer_count=int(np.sum(values < -0.02)),
        ))
    return summaries


class OutcomeCalibrationLedgerV36(StrictModel):
    schema_version: Literal["3.6"] = "3.6"
    ledger_id: Identifier
    source_v332_report_hash: Sha256
    source_v35_report_hash: Sha256
    source_bundle_hashes: list[Sha256] = Field(min_length=4, max_length=4)
    rows: list[OutcomeCalibrationRowV36] = Field(min_length=30, max_length=30)
    candidate_cutoffs: list[Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ]] = Field(min_length=5, max_length=5)
    bootstrap_replicates: Literal[5000] = 5000
    bootstrap_seed: Literal[360722] = 360722
    cutoff_summaries: list[OutcomeCutoffSummaryV36] = Field(
        min_length=5, max_length=5
    )
    selection_rule: Literal[
        "lowest_positive_bootstrap_lower_bound_with_zero_material_negatives"
    ] = "lowest_positive_bootstrap_lower_bound_with_zero_material_negatives"
    selected_q20_cutoff: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    calibration_claim: Literal[
        "empirical_training_gate_not_probability_calibration"
    ] = "empirical_training_gate_not_probability_calibration"
    exchangeability_assumed: Literal[False] = False
    conformal_guarantee_claimed: Literal[False] = False
    hidden_training_outcomes_exposed_to_episode_policy: Literal[False] = False
    built_at: datetime
    ledger_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_ledger(self) -> "OutcomeCalibrationLedgerV36":
        _assert_timezone(self.built_at, "built_at")
        if self.candidate_cutoffs != list(CALIBRATION_CUTOFFS_V36):
            raise ValueError("V3.6 calibration cutoff grid changed")
        if len({(item.source_version, item.case_id) for item in self.rows}) != 30:
            raise ValueError("V3.6 calibration rows must be source-case unique")
        if any(
            item.source_report_hash != (
                self.source_v332_report_hash
                if item.source_version == "v332" else self.source_v35_report_hash
            )
            for item in self.rows
        ):
            raise ValueError("V3.6 calibration row is not bound to source report")
        recomputed = _cutoff_summaries_v36(
            self.rows,
            bootstrap_replicates=self.bootstrap_replicates,
            bootstrap_seed=self.bootstrap_seed,
        )
        for actual, expected in zip(self.cutoff_summaries, recomputed, strict=True):
            if actual.q20_cutoff != expected.q20_cutoff:
                raise ValueError("V3.6 calibration cutoff summary order changed")
            for field in (
                "mean_realized_gain", "bootstrap_ci_low", "bootstrap_ci_high"
            ):
                if not math.isclose(
                    getattr(actual, field),
                    getattr(expected, field),
                    rel_tol=0,
                    abs_tol=1e-12,
                ):
                    raise ValueError("V3.6 calibration summary does not recompute")
            if (
                actual.selected_row_count != expected.selected_row_count
                or actual.material_negative_transfer_count
                != expected.material_negative_transfer_count
            ):
                raise ValueError("V3.6 calibration counts do not recompute")
        eligible = [
            item.q20_cutoff for item in recomputed
            if item.bootstrap_ci_low > 0
            and item.material_negative_transfer_count == 0
        ]
        if not eligible or self.selected_q20_cutoff != eligible[0]:
            raise ValueError("V3.6 selected cutoff disagrees with training rule")
        if self.ledger_hash and self.ledger_hash != self.content_hash():
            raise ValueError("ledger_hash does not match V3.6 calibration ledger")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "ledger_hash")

    def assert_sealed(self) -> None:
        if not self.ledger_hash or self.ledger_hash != self.content_hash():
            raise ValueError("V3.6 calibration ledger is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "OutcomeCalibrationLedgerV36":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"ledger_hash"}),
            ledger_hash=draft.content_hash(),
        )


def seal_outcome_calibration_ledger_v36(
    *,
    rows: list[OutcomeCalibrationRowV36],
    source_v332_report_hash: str,
    source_v35_report_hash: str,
    source_bundle_hashes: list[str],
    built_at: datetime,
) -> OutcomeCalibrationLedgerV36:
    ordered = sorted(rows, key=lambda item: (item.source_version, item.case_id))
    summaries = _cutoff_summaries_v36(
        ordered, bootstrap_replicates=5000, bootstrap_seed=360722
    )
    selected = next(
        item.q20_cutoff for item in summaries
        if item.bootstrap_ci_low > 0
        and item.material_negative_transfer_count == 0
    )
    return OutcomeCalibrationLedgerV36.seal(
        ledger_id="paired_advantage_outcome_calibration_v36",
        source_v332_report_hash=source_v332_report_hash,
        source_v35_report_hash=source_v35_report_hash,
        source_bundle_hashes=source_bundle_hashes,
        rows=ordered,
        candidate_cutoffs=list(CALIBRATION_CUTOFFS_V36),
        cutoff_summaries=summaries,
        selected_q20_cutoff=selected,
        built_at=built_at,
    )


def _artifact_payload_v36(run_directory: Path, kind: str) -> object:
    store = RunStore.open_existing(run_directory)
    events = [
        json.loads(line)
        for line in store.event_path.read_text(encoding="utf-8").splitlines()
    ]
    refs = [
        ArtifactRef.model_validate(event["payload"])
        for event in events
        if event["event_type"] == "artifact_committed"
        and event["payload"]["kind"] == kind
    ]
    if len(refs) != 1:
        raise RuntimeError(f"V3.6 source run needs exactly one {kind}")
    return store.load_artifact(refs[0])


def build_outcome_calibration_ledger_v36(
    v332_run_directory: str | Path,
    v35_run_directory: str | Path,
    *,
    built_at: datetime | None = None,
) -> OutcomeCalibrationLedgerV36:
    v332_path = Path(v332_run_directory)
    v35_path = Path(v35_run_directory)
    if not verify_controlled_dynamics_run_v332(v332_path):
        raise RuntimeError("V3.6 source V3.3.2 run failed independent verification")
    if not verify_controlled_dynamics_run_v35(v35_path):
        raise RuntimeError("V3.6 source V3.5 run failed independent verification")
    v332_report = PairedAdvantageEvolutionReportV332.model_validate(
        _artifact_payload_v36(v332_path, "controlled_dynamics_evolution_report_v332")
    )
    v332_baseline = ControlledDynamicsSelectionBundleV332.model_validate(
        _artifact_payload_v36(v332_path, "controlled_dynamics_baseline_bundle_v332")
    )
    v332_candidate = ControlledDynamicsSelectionBundleV332.model_validate(
        _artifact_payload_v36(v332_path, "controlled_dynamics_candidate_bundle_v332")
    )
    v35_report = GuardedAcquisitionEvolutionReportV35.model_validate(
        _artifact_payload_v36(v35_path, "controlled_dynamics_evolution_report_v35")
    )
    v35_baseline = ControlledDynamicsSelectionBundleV34.model_validate(
        _artifact_payload_v36(v35_path, "controlled_dynamics_random_bundle_v35")
    )
    v35_diagnostic = ControlledDynamicsSelectionBundleV34.model_validate(
        _artifact_payload_v36(v35_path, "controlled_dynamics_unguarded_bundle_v35")
    )
    for item in (
        v332_report, v332_baseline, v332_candidate,
        v35_report, v35_baseline, v35_diagnostic,
    ):
        item.assert_sealed()
    rows: list[OutcomeCalibrationRowV36] = []
    v332_base = {item.case_id: item for item in v332_baseline.case_receipts}
    for item in v332_candidate.case_receipts:
        trust = item.paired_advantage_trust_decision
        if (
            trust is None or trust.decision != "use_goal_risk"
            or item.target_loss is None
        ):
            continue
        gain = v332_base[item.case_id].target_loss - item.target_loss
        rows.append(OutcomeCalibrationRowV36(
            source_version="v332",
            source_report_hash=v332_report.evolution_hash,
            case_id=item.case_id,
            paired_advantage_q20=trust.paired_advantage_q20,
            realized_acquisition_gain=gain,
            material_negative_transfer=gain < -0.02,
        ))
    v35_base = {item.case_id: item for item in v35_baseline.case_receipts}
    for item in v35_diagnostic.case_receipts:
        trust = item.trust_decision
        baseline = v35_base[item.case_id]
        if (
            trust is None or item.target_loss is None
            or item.selected_action_hash == baseline.selected_action_hash
        ):
            continue
        gain = baseline.target_loss - item.target_loss
        rows.append(OutcomeCalibrationRowV36(
            source_version="v35",
            source_report_hash=v35_report.evolution_hash,
            case_id=item.case_id,
            paired_advantage_q20=trust.paired_advantage_q20,
            realized_acquisition_gain=gain,
            material_negative_transfer=gain < -0.02,
        ))
    return seal_outcome_calibration_ledger_v36(
        rows=rows,
        source_v332_report_hash=v332_report.evolution_hash,
        source_v35_report_hash=v35_report.evolution_hash,
        source_bundle_hashes=[
            v332_baseline.bundle_hash,
            v332_candidate.bundle_hash,
            v35_baseline.bundle_hash,
            v35_diagnostic.bundle_hash,
        ],
        built_at=built_at or datetime.now(timezone.utc),
    )


class OutcomeCalibratedAcquisitionPolicyV36(StrictModel):
    schema_version: Literal["3.6"] = "3.6"
    policy_id: Identifier
    calibration_role: CalibrationRoleV36
    arm: AdapterArmV34
    selection_rule: Literal[
        "prefrozen_random",
        "original_v332_trust_selection",
        "v332_trust_plus_outcome_q20_cutoff",
    ]
    consecutive_exceedances_required: Literal[2] = 2
    outcome_calibration_ledger_hash: Sha256
    selected_q20_cutoff: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    prior_v35_failure_report_hash: Sha256
    prior_v341_adapter_report_hash: Sha256
    prior_v332_failure_report_hash: Sha256
    method_evidence_hash: Sha256
    real_world_execution_permitted: Literal[False] = False
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "OutcomeCalibratedAcquisitionPolicyV36":
        expected = {
            "shared_random_baseline": (
                "unguarded_full_action", "prefrozen_random"
            ),
            "original_guarded_acquisition": (
                "interruptible_online_guard", "original_v332_trust_selection"
            ),
            "outcome_calibrated_guarded_acquisition": (
                "interruptible_online_guard", "v332_trust_plus_outcome_q20_cutoff"
            ),
        }[self.calibration_role]
        if (self.arm, self.selection_rule) != expected:
            raise ValueError("V3.6 policy role disagrees with selection rule")
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("policy_hash does not match V3.6 policy")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("V3.6 policy is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "OutcomeCalibratedAcquisitionPolicyV36":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"policy_hash"}),
            policy_hash=draft.content_hash(),
        )


class ControlledDynamicsWorldPackSpecV36(ControlledDynamicsWorldPackSpecV35):
    schema_version: Literal["3.6"] = "3.6"
    seeds: list[int] = Field(min_length=16, max_length=16)
    bootstrap_seed: Literal[36722] = 36722
    outcome_calibration_ledger_hash: Sha256
    selected_q20_cutoff: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    prior_v35_failure_report_hash: Sha256
    frozen_delta: Literal[
        "original_guarded_acquisition_to_outcome_calibrated_guarded_acquisition"
    ] = "original_guarded_acquisition_to_outcome_calibrated_guarded_acquisition"

    @model_validator(mode="after")
    def validate_spec(self) -> "ControlledDynamicsWorldPackSpecV36":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.mechanisms != list(MECHANISMS_V31):
            raise ValueError("V3.6 requires the frozen mechanism order")
        if tuple(self.seeds) != EXPLORATORY_SEEDS_V36:
            raise ValueError("V3.6 seeds do not match the fresh exploratory set")
        if self.goal_initial_state_scales != [0.75, 1.0, 1.25]:
            raise ValueError("V3.6 public goal initial-state scales changed")
        if not math.isclose(
            (self.trajectory_points - 1) * self.time_step,
            self.segment_count * self.segment_duration,
            abs_tol=1e-12,
        ):
            raise ValueError("V3.6 input segments do not cover trajectory")
        if len({
            self.baseline_policy_hash,
            self.diagnostic_policy_hash,
            self.candidate_policy_hash,
        }) != 3:
            raise ValueError("V3.6 requires three distinct policies")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match V3.6 protocol")
        return self

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsWorldPackSpecV36":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"spec_hash"}),
            spec_hash=draft.content_hash(),
        )


class OutcomeCalibrationEvolutionReportV36(StrictModel):
    schema_version: Literal["3.6"] = "3.6"
    evolution_id: Identifier
    spec_hash: Sha256
    outcome_calibration_ledger_hash: Sha256
    selected_q20_cutoff: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    eligible_case_count: Annotated[int, Field(ge=1)]
    original_acquisition_change_count: Annotated[int, Field(ge=0)]
    calibration_caused_fallback_count: Annotated[int, Field(ge=0)]
    calibrated_guard_interruption_count: Annotated[int, Field(ge=0)]
    random_vs_original_macro_improvement: Annotated[float, Field(allow_inf_nan=False)]
    original_vs_calibrated_macro_improvement: Annotated[float, Field(allow_inf_nan=False)]
    calibrated_package_macro_improvement: Annotated[float, Field(allow_inf_nan=False)]
    calibrated_package_ci_low: Annotated[float, Field(allow_inf_nan=False)]
    calibrated_package_ci_high: Annotated[float, Field(allow_inf_nan=False)]
    calibrated_mechanism_mean_improvements: dict[Identifier, Annotated[
        float, Field(allow_inf_nan=False)
    ]]
    calibrated_material_negative_transfer_count: Annotated[int, Field(ge=0)]
    calibrated_material_negative_transfer_rate: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ]
    random_max_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    calibrated_max_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    gates: dict[Identifier, bool]
    outcome_calibrated_package_ready: bool
    router_experiment_permitted: bool
    status: Literal[
        "outcome_calibrated_package_ready_for_router_experiment_v36",
        "outcome_calibrated_package_failed_v36",
    ]
    probability_calibration_claimed: Literal[False] = False
    router_changed: Literal[False] = False
    overall_qualification_permitted: Literal[False] = False
    confirmation_permitted: Literal[False] = False
    real_world_authorization_permitted: Literal[False] = False
    created_at: datetime
    evolution_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "OutcomeCalibrationEvolutionReportV36":
        _assert_timezone(self.created_at, "created_at")
        ready = all(self.gates.values())
        if self.outcome_calibrated_package_ready != ready:
            raise ValueError("V3.6 readiness disagrees with gates")
        if self.router_experiment_permitted != ready:
            raise ValueError("V3.6 router permission disagrees with readiness")
        expected = (
            "outcome_calibrated_package_ready_for_router_experiment_v36"
            if ready else "outcome_calibrated_package_failed_v36"
        )
        if self.status != expected:
            raise ValueError("V3.6 status disagrees with readiness")
        if self.evolution_hash and self.evolution_hash != self.content_hash():
            raise ValueError("evolution_hash does not match V3.6 report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evolution_hash")

    def assert_sealed(self) -> None:
        if not self.evolution_hash or self.evolution_hash != self.content_hash():
            raise ValueError("V3.6 report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "OutcomeCalibrationEvolutionReportV36":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evolution_hash"}),
            evolution_hash=draft.content_hash(),
        )


class ControlledDynamicsManifestV36(StrictModel):
    schema_version: Literal["3.6"] = "3.6"
    run_id: Identifier
    artifact_refs: list[ArtifactRef] = Field(min_length=9)
    terminal_status: Literal[
        "outcome_calibrated_package_ready_for_router_experiment_v36",
        "outcome_calibrated_package_failed_v36",
    ]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "ControlledDynamicsManifestV36":
        _assert_timezone(self.created_at, "created_at")
        if len({item.kind for item in self.artifact_refs}) != len(self.artifact_refs):
            raise ValueError("V3.6 manifest artifact kinds must be unique")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match V3.6 manifest")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("V3.6 manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ControlledDynamicsManifestV36":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class ControlledDynamicsOutcomeV36:
    store: RunStore
    ledger: OutcomeCalibrationLedgerV36
    spec: ControlledDynamicsWorldPackSpecV36
    private_pack: PrivateControlledDynamicsWorldPackV31
    baseline_policy: OutcomeCalibratedAcquisitionPolicyV36
    diagnostic_policy: OutcomeCalibratedAcquisitionPolicyV36
    candidate_policy: OutcomeCalibratedAcquisitionPolicyV36
    baseline_bundle: ControlledDynamicsSelectionBundleV34
    diagnostic_bundle: ControlledDynamicsSelectionBundleV34
    candidate_bundle: ControlledDynamicsSelectionBundleV34
    evolution_report: OutcomeCalibrationEvolutionReportV36
    manifest: ControlledDynamicsManifestV36


def default_controlled_dynamics_policies_v36(
    *,
    ledger: OutcomeCalibrationLedgerV36,
    prior_v35_failure_report_hash: str,
    prior_v341_adapter_report_hash: str,
    prior_v332_failure_report_hash: str,
    method_evidence_hash: str,
) -> tuple[
    OutcomeCalibratedAcquisitionPolicyV36,
    OutcomeCalibratedAcquisitionPolicyV36,
    OutcomeCalibratedAcquisitionPolicyV36,
]:
    ledger.assert_sealed()
    shared = dict(
        outcome_calibration_ledger_hash=ledger.ledger_hash,
        selected_q20_cutoff=ledger.selected_q20_cutoff,
        prior_v35_failure_report_hash=prior_v35_failure_report_hash,
        prior_v341_adapter_report_hash=prior_v341_adapter_report_hash,
        prior_v332_failure_report_hash=prior_v332_failure_report_hash,
        method_evidence_hash=method_evidence_hash,
    )
    return (
        OutcomeCalibratedAcquisitionPolicyV36.seal(
            policy_id="shared_random_baseline_v36",
            calibration_role="shared_random_baseline",
            arm="unguarded_full_action",
            selection_rule="prefrozen_random",
            **shared,
        ),
        OutcomeCalibratedAcquisitionPolicyV36.seal(
            policy_id="original_guarded_acquisition_v36",
            calibration_role="original_guarded_acquisition",
            arm="interruptible_online_guard",
            selection_rule="original_v332_trust_selection",
            **shared,
        ),
        OutcomeCalibratedAcquisitionPolicyV36.seal(
            policy_id="outcome_calibrated_guarded_acquisition_v36",
            calibration_role="outcome_calibrated_guarded_acquisition",
            arm="interruptible_online_guard",
            selection_rule="v332_trust_plus_outcome_q20_cutoff",
            **shared,
        ),
    )


def default_controlled_dynamics_exploratory_spec_v36(
    *,
    ledger: OutcomeCalibrationLedgerV36,
    baseline_policy_hash: str,
    diagnostic_policy_hash: str,
    candidate_policy_hash: str,
    method_evidence_hash: str,
    prior_epistemic_qualification_hash: str,
    prior_active_design_qualification_hash: str,
    prior_v332_failure_report_hash: str,
    prior_v34_failure_report_hash: str,
    prior_v341_adapter_report_hash: str,
    prior_v35_failure_report_hash: str,
    frozen_at: datetime | None = None,
) -> ControlledDynamicsWorldPackSpecV36:
    ledger.assert_sealed()
    return ControlledDynamicsWorldPackSpecV36.seal(
        experiment_id="controlled_dynamics_outcome_calibration_v36",
        mechanisms=list(MECHANISMS_V31),
        seeds=list(EXPLORATORY_SEEDS_V36),
        baseline_policy_hash=baseline_policy_hash,
        diagnostic_policy_hash=diagnostic_policy_hash,
        candidate_policy_hash=candidate_policy_hash,
        method_evidence_hash=method_evidence_hash,
        prior_epistemic_qualification_hash=prior_epistemic_qualification_hash,
        prior_active_design_qualification_hash=prior_active_design_qualification_hash,
        prior_v332_failure_report_hash=prior_v332_failure_report_hash,
        prior_v34_failure_report_hash=prior_v34_failure_report_hash,
        prior_v341_adapter_report_hash=prior_v341_adapter_report_hash,
        prior_v35_failure_report_hash=prior_v35_failure_report_hash,
        outcome_calibration_ledger_hash=ledger.ledger_hash,
        selected_q20_cutoff=ledger.selected_q20_cutoff,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


def _abstained_case_v36(
    private_case: PrivateControlledDynamicsCaseV31,
    policy: OutcomeCalibratedAcquisitionPolicyV36,
    target,
    clarification_used: bool,
    *,
    data_quality_passed: bool,
    reason: str,
    executed_at: datetime,
) -> ControlledDynamicsCaseReceiptV34:
    public = private_case.public_case
    return ControlledDynamicsCaseReceiptV34.seal(
        receipt_id=f"receipt_{policy.calibration_role}_{public.case_id}_v36",
        case_id=public.case_id,
        public_case_hash=public.public_hash,
        policy_hash=policy.policy_hash,
        arm=policy.arm,
        data_quality_passed=data_quality_passed,
        plan_admissible=False,
        abstention_reason=reason,
        clarification_used=clarification_used,
        decision_target=target,
        anchor_action_ids=[],
        anchor_action_hashes=[],
        anchor_observation_hashes=[],
        segment_receipts=[],
        performance_eligible=False,
        executed_at=executed_at,
    )


def _execute_case_v36(
    spec: ControlledDynamicsWorldPackSpecV36,
    private_case: PrivateControlledDynamicsCaseV31,
    policy: OutcomeCalibratedAcquisitionPolicyV36,
    *,
    executed_at: datetime,
) -> ControlledDynamicsCaseReceiptV34:
    public = private_case.public_case
    target = (
        private_case.true_decision_target
        if public.initial_contract.target_status == "default_unverified"
        else public.initial_contract.decision_target
    )
    clarification_used = public.initial_contract.target_status == "default_unverified"
    if public.pilot.quality_flags:
        return _abstained_case_v36(
            private_case, policy, target, clarification_used,
            data_quality_passed=False,
            reason="pilot_data_quality",
            executed_at=executed_at,
        )
    try:
        target, anchors, anchor_observations, active, fallback, trust = (
            _select_v332_plan_v34(spec, private_case)
        )
    except PlanAbstentionV34 as exc:
        return _abstained_case_v36(
            private_case, policy, target, clarification_used,
            data_quality_passed=True,
            reason=exc.reason,
            executed_at=executed_at,
        )
    calibration = _calibration_v34(
        spec, private_case, anchors, anchor_observations
    )
    if policy.calibration_role == "shared_random_baseline":
        selected = fallback
        selected_mode: Literal["active", "prefrozen_fallback"] = (
            "prefrozen_fallback"
        )
    else:
        use_active = trust.decision == "use_goal_risk"
        if policy.calibration_role == "outcome_calibrated_guarded_acquisition":
            use_active = use_active and (
                trust.paired_advantage_q20 >= policy.selected_q20_cutoff
            )
        selected = active if use_active else fallback
        selected_mode = "active" if use_active else "prefrozen_fallback"
    online_model = _fit_model_v31(public, anchor_observations, spec)
    intervention, observation, ledger, segment_receipts, noise_hash = (
        _stream_intervention_v34(
            spec,
            private_case,
            policy,
            selected,
            selected_mode,
            online_model,
            calibration,
            observed_at=executed_at,
        )
    )
    final_model = _fit_model_v31(
        public, [*anchor_observations, observation], spec
    )
    target_loss = (
        _target_loss_v31(private_case, final_model, spec)
        if private_case.performance_eligible else None
    )
    return ControlledDynamicsCaseReceiptV34.seal(
        receipt_id=f"receipt_{policy.calibration_role}_{public.case_id}_v36",
        case_id=public.case_id,
        public_case_hash=public.public_hash,
        policy_hash=policy.policy_hash,
        arm=policy.arm,
        data_quality_passed=True,
        plan_admissible=True,
        clarification_used=clarification_used,
        decision_target=target,
        anchor_action_ids=[item.action_id for item in anchors],
        anchor_action_hashes=[item.action_hash for item in anchors],
        anchor_observation_hashes=[item.observation_hash for item in anchor_observations],
        trust_decision=trust,
        calibration=calibration,
        selected_action_hash=selected.action_hash,
        selected_mode=selected_mode,
        noise_schedule_hash=noise_hash,
        segment_receipts=segment_receipts,
        executed_intervention=intervention,
        observation=observation,
        exposure_ledger=ledger,
        final_model=final_model,
        target_loss=target_loss,
        performance_eligible=private_case.performance_eligible,
        executed_at=executed_at,
    )


def execute_controlled_dynamics_policy_v36(
    spec: ControlledDynamicsWorldPackSpecV36,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    policy: OutcomeCalibratedAcquisitionPolicyV36,
    *,
    executed_at: datetime | None = None,
) -> ControlledDynamicsSelectionBundleV34:
    spec.assert_sealed()
    private_pack.assert_sealed()
    policy.assert_sealed()
    expected = {
        "shared_random_baseline": spec.baseline_policy_hash,
        "original_guarded_acquisition": spec.diagnostic_policy_hash,
        "outcome_calibrated_guarded_acquisition": spec.candidate_policy_hash,
    }[policy.calibration_role]
    if policy.policy_hash != expected or private_pack.spec_hash != spec.spec_hash:
        raise ValueError("V3.6 policy/private pack is not bound to protocol")
    at = executed_at or datetime.now(timezone.utc)
    receipts = [
        _execute_case_v36(spec, private_case, policy, executed_at=at)
        for private_case in private_pack.cases
    ]
    return ControlledDynamicsSelectionBundleV34.seal(
        spec_hash=spec.spec_hash,
        private_pack_hash=private_pack.pack_hash,
        policy_hash=policy.policy_hash,
        arm=policy.arm,
        case_receipts=receipts,
    )


def evaluate_controlled_dynamics_worldpack_v36(
    spec: ControlledDynamicsWorldPackSpecV36,
    ledger: OutcomeCalibrationLedgerV36,
    private_pack: PrivateControlledDynamicsWorldPackV31,
    baseline: ControlledDynamicsSelectionBundleV34,
    diagnostic: ControlledDynamicsSelectionBundleV34,
    candidate: ControlledDynamicsSelectionBundleV34,
    *,
    evaluated_at: datetime | None = None,
) -> OutcomeCalibrationEvolutionReportV36:
    spec.assert_sealed()
    ledger.assert_sealed()
    private_pack.assert_sealed()
    if (
        spec.outcome_calibration_ledger_hash != ledger.ledger_hash
        or spec.selected_q20_cutoff != ledger.selected_q20_cutoff
    ):
        raise ValueError("V3.6 ledger is not frozen in protocol")
    for bundle in (baseline, diagnostic, candidate):
        bundle.assert_sealed()
        if bundle.spec_hash != spec.spec_hash or (
            bundle.private_pack_hash != private_pack.pack_hash
        ):
            raise ValueError("V3.6 bundle belongs to another experiment")
    random_by = {item.case_id: item for item in baseline.case_receipts}
    original_by = {item.case_id: item for item in diagnostic.case_receipts}
    calibrated_by = {item.case_id: item for item in candidate.case_receipts}
    private_by = {item.public_case.case_id: item for item in private_pack.cases}
    case_ids = list(random_by)

    def same_all(case_id: str, field: str) -> bool:
        values = [
            getattr(items[case_id], field)
            for items in (random_by, original_by, calibrated_by)
        ]
        return values[0] == values[1] == values[2]

    context_parity = all(
        same_all(case_id, "decision_target")
        and same_all(case_id, "anchor_action_hashes")
        and same_all(case_id, "anchor_observation_hashes")
        and same_all(case_id, "noise_schedule_hash")
        and same_all(case_id, "abstention_reason")
        for case_id in case_ids
    )
    trust_calibration_parity = all(
        len({
            None if items[case_id].trust_decision is None
            else items[case_id].trust_decision.trust_hash
            for items in (random_by, original_by, calibrated_by)
        }) == 1
        and len({
            None if items[case_id].calibration is None
            else items[case_id].calibration.calibration_hash
            for items in (random_by, original_by, calibrated_by)
        }) == 1
        for case_id in case_ids
    )
    complete_ids = [x for x in case_ids if random_by[x].plan_admissible]
    binding = True
    calibration_fallbacks = 0
    original_changes = 0
    for case_id in complete_ids:
        left = random_by[case_id]
        middle = original_by[case_id]
        right = calibrated_by[case_id]
        trust = left.trust_decision
        if trust is None:
            binding = False
            continue
        expected_original = trust.selected_action_hash
        expected_calibrated = (
            trust.active_action_hash
            if trust.decision == "use_goal_risk"
            and trust.paired_advantage_q20 >= ledger.selected_q20_cutoff
            else trust.fallback_action_hash
        )
        binding = binding and (
            left.selected_action_hash == trust.fallback_action_hash
            and middle.selected_action_hash == expected_original
            and right.selected_action_hash == expected_calibrated
        )
        original_changes += middle.selected_action_hash != left.selected_action_hash
        calibration_fallbacks += (
            middle.selected_action_hash != right.selected_action_hash
        )
    receipts_complete = all(
        (not item.plan_admissible) or (
            item.observation is not None
            and item.executed_intervention is not None
            and item.exposure_ledger is not None
            and len(item.segment_receipts)
            == item.exposure_ledger.executed_segment_count
        )
        for bundle in (baseline, diagnostic, candidate)
        for item in bundle.case_receipts
    )
    exposure_dominance = all(
        calibrated_by[case_id].exposure_ledger is None or (
            random_by[case_id].exposure_ledger is not None
            and calibrated_by[case_id].exposure_ledger.used_duration
            <= random_by[case_id].exposure_ledger.used_duration + 1e-12
            and calibrated_by[case_id].exposure_ledger.used_energy
            <= random_by[case_id].exposure_ledger.used_energy + 1e-12
            and calibrated_by[case_id].exposure_ledger.used_peak_amplitude
            <= random_by[case_id].exposure_ledger.used_peak_amplitude + 1e-12
            and calibrated_by[case_id].exposure_ledger.used_switch_count
            <= random_by[case_id].exposure_ledger.used_switch_count
        )
        for case_id in case_ids
    )
    state_violations = sum(
        item.exposure_ledger.state_envelope_violation_count
        for bundle in (baseline, diagnostic, candidate)
        for item in bundle.case_receipts
        if item.exposure_ledger is not None
    )
    interruptions = sum(
        item.executed_intervention is not None
        and item.executed_intervention.interrupted
        for item in candidate.case_receipts
    )
    eligible_ids = [
        case_id for case_id in case_ids
        if all(
            items[case_id].target_loss is not None
            for items in (random_by, original_by, calibrated_by)
        )
    ]
    if not eligible_ids:
        raise RuntimeError("V3.6 evaluation has no eligible cases")
    r = np.asarray([random_by[x].target_loss for x in eligible_ids], dtype=float)
    o = np.asarray([original_by[x].target_loss for x in eligible_ids], dtype=float)
    c = np.asarray([calibrated_by[x].target_loss for x in eligible_ids], dtype=float)
    package = r - c
    random = np.random.default_rng(spec.bootstrap_seed)
    samples = random.integers(
        0, len(package), size=(spec.bootstrap_replicates, len(package))
    )
    bootstrap = np.mean(package[samples], axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    mechanism_means = {
        mechanism: float(np.mean([
            package[index]
            for index, case_id in enumerate(eligible_ids)
            if private_by[case_id].mechanism == mechanism
        ]))
        for mechanism in MECHANISMS_V31
        if any(private_by[x].mechanism == mechanism for x in eligible_ids)
    }
    negative_count = int(
        np.sum(package < -spec.material_negative_transfer)
    )
    negative_rate = negative_count / len(package)
    gates = {
        "ledger_policy_spec_binding": (
            spec.outcome_calibration_ledger_hash == ledger.ledger_hash
            and spec.selected_q20_cutoff == ledger.selected_q20_cutoff
        ),
        "shared_context_and_abstention_parity": context_parity,
        "trust_and_mismatch_calibration_parity": trust_calibration_parity,
        "three_arm_action_binding": binding,
        "minimum_original_acquisition_change": original_changes >= 1,
        "minimum_calibration_caused_fallback": calibration_fallbacks >= 1,
        "all_receipts_complete": receipts_complete,
        "calibrated_exposure_dominated_by_random": exposure_dominance,
        "zero_synthetic_state_envelope_violations": state_violations == 0,
        "calibrated_package_macro_lower_bound": float(low) >= 0.0,
        "calibrated_mechanism_non_regression": min(mechanism_means.values())
        >= -spec.maximum_mechanism_regression,
        "calibrated_negative_transfer_upper_bound": negative_rate
        <= spec.maximum_guard_negative_transfer_rate,
        "calibrated_worst_case_loss_non_regression": float(np.max(c))
        <= float(np.max(r)) + 1e-12,
    }
    ready = all(gates.values())
    return OutcomeCalibrationEvolutionReportV36.seal(
        evolution_id="controlled_dynamics_outcome_calibration_v36",
        spec_hash=spec.spec_hash,
        outcome_calibration_ledger_hash=ledger.ledger_hash,
        selected_q20_cutoff=ledger.selected_q20_cutoff,
        eligible_case_count=len(eligible_ids),
        original_acquisition_change_count=original_changes,
        calibration_caused_fallback_count=calibration_fallbacks,
        calibrated_guard_interruption_count=interruptions,
        random_vs_original_macro_improvement=float(np.mean(r - o)),
        original_vs_calibrated_macro_improvement=float(np.mean(o - c)),
        calibrated_package_macro_improvement=float(np.mean(package)),
        calibrated_package_ci_low=float(low),
        calibrated_package_ci_high=float(high),
        calibrated_mechanism_mean_improvements=mechanism_means,
        calibrated_material_negative_transfer_count=negative_count,
        calibrated_material_negative_transfer_rate=negative_rate,
        random_max_target_loss=float(np.max(r)),
        calibrated_max_target_loss=float(np.max(c)),
        gates=gates,
        outcome_calibrated_package_ready=ready,
        router_experiment_permitted=ready,
        status=(
            "outcome_calibrated_package_ready_for_router_experiment_v36"
            if ready else "outcome_calibrated_package_failed_v36"
        ),
        created_at=evaluated_at or datetime.now(timezone.utc),
    )


def run_controlled_dynamics_worldpack_v36(
    output_root: str | Path,
    *,
    ledger: OutcomeCalibrationLedgerV36,
    spec: ControlledDynamicsWorldPackSpecV36,
    baseline_policy: OutcomeCalibratedAcquisitionPolicyV36,
    diagnostic_policy: OutcomeCalibratedAcquisitionPolicyV36,
    candidate_policy: OutcomeCalibratedAcquisitionPolicyV36,
    evaluated_at: datetime | None = None,
    run_id: str | None = None,
) -> ControlledDynamicsOutcomeV36:
    ledger.assert_sealed()
    spec.assert_sealed()
    for policy in (baseline_policy, diagnostic_policy, candidate_policy):
        policy.assert_sealed()
        if (
            policy.outcome_calibration_ledger_hash != ledger.ledger_hash
            or policy.selected_q20_cutoff != ledger.selected_q20_cutoff
        ):
            raise ValueError("V3.6 policy is not bound to calibration ledger")
    if (
        baseline_policy.policy_hash != spec.baseline_policy_hash
        or diagnostic_policy.policy_hash != spec.diagnostic_policy_hash
        or candidate_policy.policy_hash != spec.candidate_policy_hash
    ):
        raise ValueError("V3.6 policies are not frozen in protocol")
    at = evaluated_at or datetime.now(timezone.utc)
    store = RunStore(
        output_root,
        run_id=run_id or f"controlled-dynamics-v36-{uuid4().hex[:10]}",
    )
    refs = [
        store.put_artifact("outcome_calibration_ledger_v36", ledger),
        store.put_artifact("controlled_dynamics_spec_v36", spec),
        store.put_artifact("controlled_dynamics_random_policy_v36", baseline_policy),
        store.put_artifact("controlled_dynamics_original_policy_v36", diagnostic_policy),
        store.put_artifact("controlled_dynamics_calibrated_policy_v36", candidate_policy),
    ]
    store.emit("controlled_dynamics_v36_protocol_frozen_before_private_pack", {
        "spec_hash": spec.spec_hash,
        "ledger_hash": ledger.ledger_hash,
        "selected_q20_cutoff": ledger.selected_q20_cutoff,
    })
    private_pack = generate_private_controlled_dynamics_worldpack_v31(
        spec, generated_at=at
    )
    baseline = execute_controlled_dynamics_policy_v36(
        spec, private_pack, baseline_policy, executed_at=at
    )
    diagnostic = execute_controlled_dynamics_policy_v36(
        spec, private_pack, diagnostic_policy, executed_at=at
    )
    candidate = execute_controlled_dynamics_policy_v36(
        spec, private_pack, candidate_policy, executed_at=at
    )
    evolution = evaluate_controlled_dynamics_worldpack_v36(
        spec, ledger, private_pack, baseline, diagnostic, candidate, evaluated_at=at
    )
    refs.extend([
        store.put_artifact("private_controlled_dynamics_worldpack_v36", private_pack),
        store.put_artifact("controlled_dynamics_random_bundle_v36", baseline),
        store.put_artifact("controlled_dynamics_original_bundle_v36", diagnostic),
        store.put_artifact("controlled_dynamics_calibrated_bundle_v36", candidate),
        store.put_artifact("controlled_dynamics_evolution_report_v36", evolution),
    ])
    manifest = ControlledDynamicsManifestV36.seal(
        run_id=store.run_id,
        artifact_refs=refs,
        terminal_status=evolution.status,
        created_at=at,
    )
    manifest_ref = store.put_artifact("controlled_dynamics_manifest_v36", manifest)
    store.emit("controlled_dynamics_v36_worldpack_adjudicated", {
        "manifest_ref": manifest_ref.model_dump(mode="json")
    })
    if not verify_controlled_dynamics_run_v36(store.run_directory):
        raise RuntimeError("V3.6 run failed independent verification")
    return ControlledDynamicsOutcomeV36(
        store,
        ledger,
        spec,
        private_pack,
        baseline_policy,
        diagnostic_policy,
        candidate_policy,
        baseline,
        diagnostic,
        candidate,
        evolution,
        manifest,
    )


def verify_controlled_dynamics_run_v36(run_directory: str | Path) -> bool:
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
        manifests = [
            item for item in committed
            if item.kind == "controlled_dynamics_manifest_v36"
        ]
        if len(manifests) != 1:
            return False
        manifest = ControlledDynamicsManifestV36.model_validate(
            store.load_artifact(manifests[0])
        )
        manifest.assert_sealed()

        def load_one(kind: str, model):
            references = [item for item in manifest.artifact_refs if item.kind == kind]
            if len(references) != 1:
                raise RuntimeError(f"V3.6 manifest needs exactly one {kind}")
            return model.model_validate(store.load_artifact(references[0]))

        ledger = load_one("outcome_calibration_ledger_v36", OutcomeCalibrationLedgerV36)
        spec = load_one("controlled_dynamics_spec_v36", ControlledDynamicsWorldPackSpecV36)
        baseline_policy = load_one(
            "controlled_dynamics_random_policy_v36", OutcomeCalibratedAcquisitionPolicyV36
        )
        diagnostic_policy = load_one(
            "controlled_dynamics_original_policy_v36", OutcomeCalibratedAcquisitionPolicyV36
        )
        candidate_policy = load_one(
            "controlled_dynamics_calibrated_policy_v36", OutcomeCalibratedAcquisitionPolicyV36
        )
        private_pack = load_one(
            "private_controlled_dynamics_worldpack_v36",
            PrivateControlledDynamicsWorldPackV31,
        )
        baseline = load_one(
            "controlled_dynamics_random_bundle_v36", ControlledDynamicsSelectionBundleV34
        )
        diagnostic = load_one(
            "controlled_dynamics_original_bundle_v36", ControlledDynamicsSelectionBundleV34
        )
        candidate = load_one(
            "controlled_dynamics_calibrated_bundle_v36", ControlledDynamicsSelectionBundleV34
        )
        evolution = load_one(
            "controlled_dynamics_evolution_report_v36",
            OutcomeCalibrationEvolutionReportV36,
        )
        for artifact in (
            ledger, spec, baseline_policy, diagnostic_policy, candidate_policy,
            private_pack, baseline, diagnostic, candidate, evolution, manifest,
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
        replay_baseline = execute_controlled_dynamics_policy_v36(
            spec, private_pack, baseline_policy, executed_at=executed_at
        )
        replay_diagnostic = execute_controlled_dynamics_policy_v36(
            spec, private_pack, diagnostic_policy, executed_at=executed_at
        )
        replay_candidate = execute_controlled_dynamics_policy_v36(
            spec, private_pack, candidate_policy, executed_at=executed_at
        )
        if (
            replay_baseline.bundle_hash != baseline.bundle_hash
            or replay_diagnostic.bundle_hash != diagnostic.bundle_hash
            or replay_candidate.bundle_hash != candidate.bundle_hash
        ):
            return False
        recomputed = evaluate_controlled_dynamics_worldpack_v36(
            spec,
            ledger,
            private_pack,
            baseline,
            diagnostic,
            candidate,
            evaluated_at=evolution.created_at,
        )
        if recomputed.evolution_hash != evolution.evolution_hash:
            return False
        if any(
            "qualification" in item.kind or "confirmation" in item.kind
            for item in manifest.artifact_refs
        ):
            return False
        freezes = [
            event for event in events
            if event["event_type"]
            == "controlled_dynamics_v36_protocol_frozen_before_private_pack"
        ]
        private_events = [
            event for event in events
            if event["event_type"] == "artifact_committed"
            and event["payload"]["kind"] == "private_controlled_dynamics_worldpack_v36"
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
