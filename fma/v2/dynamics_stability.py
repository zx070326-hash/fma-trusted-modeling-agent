from __future__ import annotations

import math
from datetime import datetime, timezone
from random import Random
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel

from .dynamics_estimator_ablation import (
    DynamicsEstimatorAblationReportV25,
    DynamicsEstimatorExperimentSpecV25,
    DynamicsEstimatorSelectionBundleV25,
)
from .dynamics_integral import (
    DynamicsEstimatorPolicyV25,
    DynamicsSelectionReceiptV25,
    build_estimation_system_v25,
    fit_coefficients_v25,
    fit_dynamics_candidate_v25,
)
from .dynamics_worldpack import PrivateDynamicsCaseV24, PrivateDynamicsWorldPackV24
from .schemas import Identifier, Sha256, _assert_timezone


StabilityReason = Literal[
    "selected_fit_not_replayable",
    "no_material_coefficients",
    "support_stability_below_threshold",
    "sign_stability_below_threshold",
    "relative_interval_width_above_threshold",
]


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class DynamicsStabilityProtocolV25(StrictModel):
    schema_version: Literal["2.5"] = "2.5"
    protocol_id: Identifier
    experiment_spec_hash: Sha256
    resampling_method: Literal["moving_block_rows"] = "moving_block_rows"
    bootstrap_replicates: Annotated[int, Field(ge=100, le=2_000)]
    block_length_rows: Annotated[int, Field(ge=2, le=50)]
    interval_level: Annotated[float, Field(gt=0.5, lt=1, allow_inf_nan=False)]
    material_relative_threshold: Annotated[
        float, Field(gt=0, le=0.5, allow_inf_nan=False)
    ]
    material_absolute_floor: Annotated[
        float, Field(gt=0, le=0.1, allow_inf_nan=False)
    ]
    minimum_support_jaccard: Annotated[
        float, Field(gt=0.5, le=1, allow_inf_nan=False)
    ]
    minimum_sign_agreement: Annotated[
        float, Field(gt=0.5, le=1, allow_inf_nan=False)
    ]
    maximum_median_relative_interval_width: Annotated[
        float, Field(gt=0, le=10, allow_inf_nan=False)
    ]
    maximum_unstable_case_rate: Annotated[
        float, Field(ge=0, lt=0.5, allow_inf_nan=False)
    ]
    bootstrap_seed: Annotated[int, Field(ge=0, le=2_147_483_647)]
    interpretation_scope: Literal[
        "dependence_aware_resampling_stability_not_calibrated_uncertainty"
    ] = "dependence_aware_resampling_stability_not_calibrated_uncertainty"
    frozen_at: datetime
    protocol_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_protocol(self) -> "DynamicsStabilityProtocolV25":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.protocol_hash and self.protocol_hash != self.content_hash():
            raise ValueError("protocol_hash does not match V2.5 stability protocol")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "protocol_hash")

    def assert_sealed(self) -> None:
        if not self.protocol_hash or self.protocol_hash != self.content_hash():
            raise ValueError("V2.5 stability protocol is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DynamicsStabilityProtocolV25":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"protocol_hash"}),
            protocol_hash=draft.content_hash(),
        )


class DynamicsCaseStabilityDiagnosticV25(StrictModel):
    case_id: Identifier
    estimator_arm: Literal["point_savgol", "window_integral_matching"]
    policy_hash: Sha256
    selection_receipt_hash: Sha256
    selected_candidate_id: Identifier
    base_model_hash: Sha256
    estimation_row_count: Annotated[int, Field(ge=1)]
    bootstrap_replicates: Annotated[int, Field(ge=100)]
    material_coefficient_count: Annotated[int, Field(ge=0, le=400)]
    mean_support_jaccard: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    mean_sign_agreement: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    median_relative_interval_width: Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ]
    status: Literal["stable", "unstable", "needs_evidence"]
    reason_codes: list[StabilityReason]

    @model_validator(mode="after")
    def validate_diagnostic(self) -> "DynamicsCaseStabilityDiagnosticV25":
        if self.status == "stable" and self.reason_codes:
            raise ValueError("stable V2.5 diagnostic cannot contain failure reasons")
        if self.status != "stable" and not self.reason_codes:
            raise ValueError("non-stable V2.5 diagnostic needs reasons")
        return self


class DynamicsStabilityReportV25(StrictModel):
    schema_version: Literal["2.5"] = "2.5"
    report_id: Identifier
    protocol_hash: Sha256
    experiment_spec_hash: Sha256
    estimator_report_hash: Sha256
    private_pack_hash: Sha256
    point_policy_hash: Sha256
    integral_policy_hash: Sha256
    diagnostics: list[DynamicsCaseStabilityDiagnosticV25] = Field(min_length=32)
    point_case_count: Annotated[int, Field(ge=16)]
    integral_case_count: Annotated[int, Field(ge=16)]
    point_unstable_count: Annotated[int, Field(ge=0)]
    integral_unstable_count: Annotated[int, Field(ge=0)]
    integral_unstable_rate: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    point_median_support_jaccard: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ]
    integral_median_support_jaccard: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ]
    point_median_sign_agreement: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ]
    integral_median_sign_agreement: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ]
    point_median_relative_interval_width: Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ]
    integral_median_relative_interval_width: Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ]
    status: Literal["stability_gate_passed", "stability_gate_failed"]
    limitations: list[Annotated[str, Field(min_length=12)]] = Field(min_length=4)
    evaluated_at: datetime
    report_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "DynamicsStabilityReportV25":
        _assert_timezone(self.evaluated_at, "evaluated_at")
        if self.point_case_count + self.integral_case_count != len(self.diagnostics):
            raise ValueError("V2.5 stability counts do not cover diagnostics")
        if self.report_hash and self.report_hash != self.content_hash():
            raise ValueError("report_hash does not match V2.5 stability report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "report_hash")

    def assert_sealed(self) -> None:
        if not self.report_hash or self.report_hash != self.content_hash():
            raise ValueError("V2.5 stability report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DynamicsStabilityReportV25":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"report_hash"}),
            report_hash=draft.content_hash(),
        )


def default_stability_protocol_v25(
    *,
    experiment_spec_hash: str,
    frozen_at: datetime | None = None,
) -> DynamicsStabilityProtocolV25:
    return DynamicsStabilityProtocolV25.seal(
        protocol_id="dynamics_parameter_stability_protocol_v25",
        experiment_spec_hash=experiment_spec_hash,
        bootstrap_replicates=200,
        block_length_rows=8,
        interval_level=0.90,
        material_relative_threshold=0.05,
        material_absolute_floor=1e-4,
        minimum_support_jaccard=0.80,
        minimum_sign_agreement=0.90,
        maximum_median_relative_interval_width=1.0,
        maximum_unstable_case_rate=0.10,
        bootstrap_seed=273_722,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


def evaluate_dynamics_stability_v25(
    protocol: DynamicsStabilityProtocolV25,
    experiment_spec: DynamicsEstimatorExperimentSpecV25,
    estimator_report: DynamicsEstimatorAblationReportV25,
    private_pack: PrivateDynamicsWorldPackV24,
    point_policy: DynamicsEstimatorPolicyV25,
    integral_policy: DynamicsEstimatorPolicyV25,
    point_selections: DynamicsEstimatorSelectionBundleV25,
    integral_selections: DynamicsEstimatorSelectionBundleV25,
    *,
    evaluated_at: datetime | None = None,
) -> DynamicsStabilityReportV25:
    for item in [
        protocol,
        experiment_spec,
        estimator_report,
        private_pack,
        point_policy,
        integral_policy,
        point_selections,
        integral_selections,
    ]:
        item.assert_sealed()
    if protocol.experiment_spec_hash != experiment_spec.spec_hash:
        raise ValueError("V2.5 stability protocol is bound to another experiment")
    if estimator_report.experiment_spec_hash != experiment_spec.spec_hash:
        raise ValueError("V2.5 stability report input is bound to another experiment")
    if estimator_report.private_pack_hash != private_pack.pack_hash:
        raise ValueError("V2.5 stability report input uses another private pack")
    diagnostics: list[DynamicsCaseStabilityDiagnosticV25] = []
    for arm_index, (policy, bundle) in enumerate(
        ((point_policy, point_selections), (integral_policy, integral_selections))
    ):
        receipts = {receipt.public_data_hash: receipt for receipt in bundle.case_receipts}
        for case_index, case in enumerate(private_pack.cases):
            public = case.public_projection(experiment_spec.data_spec)
            assert public.snapshot_hash is not None
            receipt = receipts[public.snapshot_hash]
            diagnostics.append(
                _case_stability(
                    protocol,
                    case,
                    public,
                    policy,
                    receipt,
                    random_seed=(
                        protocol.bootstrap_seed
                        + arm_index * 10_000_019
                        + case.seed * 101
                        + case_index
                    ),
                )
            )
    point = [item for item in diagnostics if item.estimator_arm == "point_savgol"]
    integral = [
        item
        for item in diagnostics
        if item.estimator_arm == "window_integral_matching"
    ]
    point_unstable = sum(item.status != "stable" for item in point)
    integral_unstable = sum(item.status != "stable" for item in integral)
    integral_unstable_rate = integral_unstable / len(integral)
    integral_median_support = float(
        np.median([item.mean_support_jaccard for item in integral])
    )
    integral_median_sign = float(
        np.median([item.mean_sign_agreement for item in integral])
    )
    integral_median_width = float(
        np.median([item.median_relative_interval_width for item in integral])
    )
    passed = (
        integral_unstable_rate <= protocol.maximum_unstable_case_rate
        and integral_median_support >= protocol.minimum_support_jaccard
        and integral_median_sign >= protocol.minimum_sign_agreement
        and integral_median_width
        <= protocol.maximum_median_relative_interval_width
    )
    assert protocol.protocol_hash is not None
    assert experiment_spec.spec_hash is not None
    assert estimator_report.report_hash is not None
    assert private_pack.pack_hash is not None
    assert point_policy.policy_hash is not None
    assert integral_policy.policy_hash is not None
    return DynamicsStabilityReportV25.seal(
        report_id=f"{experiment_spec.experiment_id}_stability",
        protocol_hash=protocol.protocol_hash,
        experiment_spec_hash=experiment_spec.spec_hash,
        estimator_report_hash=estimator_report.report_hash,
        private_pack_hash=private_pack.pack_hash,
        point_policy_hash=point_policy.policy_hash,
        integral_policy_hash=integral_policy.policy_hash,
        diagnostics=diagnostics,
        point_case_count=len(point),
        integral_case_count=len(integral),
        point_unstable_count=point_unstable,
        integral_unstable_count=integral_unstable,
        integral_unstable_rate=integral_unstable_rate,
        point_median_support_jaccard=float(
            np.median([item.mean_support_jaccard for item in point])
        ),
        integral_median_support_jaccard=integral_median_support,
        point_median_sign_agreement=float(
            np.median([item.mean_sign_agreement for item in point])
        ),
        integral_median_sign_agreement=integral_median_sign,
        point_median_relative_interval_width=float(
            np.median([item.median_relative_interval_width for item in point])
        ),
        integral_median_relative_interval_width=integral_median_width,
        status="stability_gate_passed" if passed else "stability_gate_failed",
        limitations=[
            "moving-block rows approximate dependence but do not prove calibrated coverage",
            "overlapping integral windows share raw observations and effective sample size is unknown",
            "material-coefficient thresholds are frozen engineering diagnostics rather than posterior probabilities",
            "stability is necessary evidence but cannot establish correct structure or real-world validity",
        ],
        evaluated_at=evaluated_at or datetime.now(timezone.utc),
    )


def _case_stability(
    protocol: DynamicsStabilityProtocolV25,
    case: PrivateDynamicsCaseV24,
    public,
    policy: DynamicsEstimatorPolicyV25,
    receipt: DynamicsSelectionReceiptV25,
    *,
    random_seed: int,
) -> DynamicsCaseStabilityDiagnosticV25:
    if receipt.status != "selected" or receipt.selected_candidate_id is None:
        raise ValueError("V2.5 stability requires a selected primary case")
    definition = next(
        candidate
        for candidate in policy.candidates
        if candidate.candidate_id == receipt.selected_candidate_id
    )
    fit = fit_dynamics_candidate_v25(
        public, definition, policy, fitted_at=receipt.selected_at
    )
    assert receipt.receipt_hash is not None
    assert policy.policy_hash is not None
    if fit.status != "fit_succeeded" or fit.model is None:
        return DynamicsCaseStabilityDiagnosticV25(
            case_id=case.case_id,
            estimator_arm=policy.estimator_arm,
            policy_hash=policy.policy_hash,
            selection_receipt_hash=receipt.receipt_hash,
            selected_candidate_id=receipt.selected_candidate_id,
            base_model_hash="0" * 64,
            estimation_row_count=1,
            bootstrap_replicates=protocol.bootstrap_replicates,
            material_coefficient_count=0,
            mean_support_jaccard=0.0,
            mean_sign_agreement=0.0,
            median_relative_interval_width=0.0,
            status="needs_evidence",
            reason_codes=["selected_fit_not_replayable"],
        )
    assert fit.model.model_hash is not None
    design, targets, _ = build_estimation_system_v25(public, definition)
    base = np.asarray(fit.model.coefficient_matrix, dtype=float)
    thresholds = np.maximum(
        protocol.material_absolute_floor,
        protocol.material_relative_threshold * np.max(np.abs(base), axis=1),
    )
    material = np.abs(base) >= thresholds[:, np.newaxis]
    material_count = int(np.count_nonzero(material))
    if material_count == 0:
        return DynamicsCaseStabilityDiagnosticV25(
            case_id=case.case_id,
            estimator_arm=policy.estimator_arm,
            policy_hash=policy.policy_hash,
            selection_receipt_hash=receipt.receipt_hash,
            selected_candidate_id=receipt.selected_candidate_id,
            base_model_hash=fit.model.model_hash,
            estimation_row_count=design.shape[0],
            bootstrap_replicates=protocol.bootstrap_replicates,
            material_coefficient_count=0,
            mean_support_jaccard=0.0,
            mean_sign_agreement=0.0,
            median_relative_interval_width=0.0,
            status="needs_evidence",
            reason_codes=["no_material_coefficients"],
        )
    random = Random(random_seed)
    replicate_coefficients = np.empty(
        (protocol.bootstrap_replicates, *base.shape), dtype=float
    )
    for replicate in range(protocol.bootstrap_replicates):
        indices = _moving_block_indices(
            design.shape[0], protocol.block_length_rows, random
        )
        replicate_coefficients[replicate] = fit_coefficients_v25(
            design[indices], targets[indices], definition
        )
    base_support = material
    jaccards = []
    sign_matches = []
    for coefficients in replicate_coefficients:
        replicate_support = np.abs(coefficients) >= thresholds[:, np.newaxis]
        union = np.count_nonzero(base_support | replicate_support)
        intersection = np.count_nonzero(base_support & replicate_support)
        jaccards.append(1.0 if union == 0 else intersection / union)
        sign_matches.append(
            float(np.mean(np.sign(coefficients[material]) == np.sign(base[material])))
        )
    alpha = (1.0 - protocol.interval_level) / 2.0
    lower = np.quantile(replicate_coefficients, alpha, axis=0, method="linear")
    upper = np.quantile(
        replicate_coefficients, 1.0 - alpha, axis=0, method="linear"
    )
    relative_widths = (upper[material] - lower[material]) / np.maximum(
        np.abs(base[material]), protocol.material_absolute_floor
    )
    support = float(np.mean(jaccards))
    sign = float(np.mean(sign_matches))
    width = float(np.median(relative_widths))
    if not all(math.isfinite(value) for value in (support, sign, width)):
        raise ValueError("V2.5 stability metric is not finite")
    reasons: list[StabilityReason] = []
    if support < protocol.minimum_support_jaccard:
        reasons.append("support_stability_below_threshold")
    if sign < protocol.minimum_sign_agreement:
        reasons.append("sign_stability_below_threshold")
    if width > protocol.maximum_median_relative_interval_width:
        reasons.append("relative_interval_width_above_threshold")
    return DynamicsCaseStabilityDiagnosticV25(
        case_id=case.case_id,
        estimator_arm=policy.estimator_arm,
        policy_hash=policy.policy_hash,
        selection_receipt_hash=receipt.receipt_hash,
        selected_candidate_id=receipt.selected_candidate_id,
        base_model_hash=fit.model.model_hash,
        estimation_row_count=design.shape[0],
        bootstrap_replicates=protocol.bootstrap_replicates,
        material_coefficient_count=material_count,
        mean_support_jaccard=support,
        mean_sign_agreement=sign,
        median_relative_interval_width=width,
        status="stable" if not reasons else "unstable",
        reason_codes=reasons,
    )


def _moving_block_indices(
    row_count: int,
    block_length: int,
    random: Random,
) -> np.ndarray:
    width = min(block_length, row_count)
    maximum_start = row_count - width
    indices: list[int] = []
    while len(indices) < row_count:
        start = random.randrange(maximum_start + 1)
        indices.extend(range(start, start + width))
    return np.asarray(indices[:row_count], dtype=int)
