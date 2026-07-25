"""V5.1.1 receipts for independently executed gold and ablation arms.

This is additive to V5.1.  It closes the gap where a declared arm could reuse
role outputs from another arm and still appear to have a changed path.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256

from .evaluation_harness import MechanismProfileV51, NuisanceIdentityV51


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class ExecutedMechanismRunV511(StrictModel):
    schema_version: Literal["5.1.1"] = "5.1.1"
    run_id: Identifier
    nuisance_identity: NuisanceIdentityV51
    profile: MechanismProfileV51
    process_receipt_hashes: Annotated[list[Sha256], Field(min_length=1)]
    process_run_ids: Annotated[list[Identifier], Field(min_length=1)]
    process_context_ids: Annotated[list[Identifier], Field(min_length=1)]
    observed_mechanism_events: list[Identifier]
    gold_injection_receipt_hash: Sha256 | None = None
    selected_candidate_id: Identifier
    development_score: float
    output_artifact_hash: Sha256
    terminal_state: Literal[
        "SCIENTIFICALLY_REJECTED",
        "HOLDOUT_SCORED_NOT_QUALIFIED",
        "INTEGRITY_FAILURE",
        "NOT_RUN",
    ]
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_run(self) -> "ExecutedMechanismRunV511":
        for values, label in (
            (self.process_receipt_hashes, "process_receipt_hashes"),
            (self.process_run_ids, "process_run_ids"),
            (self.process_context_ids, "process_context_ids"),
            (self.observed_mechanism_events, "observed_mechanism_events"),
        ):
            if values != sorted(set(values)):
                raise ValueError(f"{label} must be sorted and unique")
        if not (
            len(self.process_receipt_hashes)
            == len(self.process_run_ids)
            == len(self.process_context_ids)
        ):
            raise ValueError("executed process identity counts differ")
        gold_enabled = self.profile.gold_through_stage != "NONE"
        if gold_enabled != (self.gold_injection_receipt_hash is not None):
            raise ValueError("gold profile and injection receipt differ")
        if gold_enabled and "gold_stage_injection_executed" not in (
            self.observed_mechanism_events
        ):
            raise ValueError("gold run lacks an observed injection event")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match executed mechanism run")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    @classmethod
    def seal(cls, **data: object) -> "ExecutedMechanismRunV511":
        for field_name in (
            "process_receipt_hashes",
            "process_run_ids",
            "process_context_ids",
            "observed_mechanism_events",
        ):
            data[field_name] = sorted(set(data[field_name]))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


class ExecutedAblationComparisonV511(StrictModel):
    schema_version: Literal["5.1.1"] = "5.1.1"
    mechanism_id: Literal[
        "competition",
        "independent_review",
        "backward_revision",
        "scientific_adapters",
    ]
    control_run_receipt_hash: Sha256
    treatment_run_receipt_hash: Sha256
    same_nuisance_identity: bool
    exactly_one_mechanism_changed: bool
    process_receipts_disjoint: bool
    observed_execution_path_delta: bool
    valid_executed_ablation: bool
    development_score_delta: float
    causal_claim_permitted: Literal[False] = False
    reasons: list[Identifier]
    comparison_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_comparison(self) -> "ExecutedAblationComparisonV511":
        if self.valid_executed_ablation and (
            not self.same_nuisance_identity
            or not self.exactly_one_mechanism_changed
            or not self.process_receipts_disjoint
            or not self.observed_execution_path_delta
            or self.reasons
        ):
            raise ValueError("valid executed ablation contains invalidity")
        if self.comparison_hash and self.comparison_hash != self.content_hash():
            raise ValueError("comparison_hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "comparison_hash")

    @classmethod
    def seal(cls, **data: object) -> "ExecutedAblationComparisonV511":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"comparison_hash"}),
            comparison_hash=draft.content_hash(),
        )


def compare_executed_ablation_v511(
    control: ExecutedMechanismRunV511,
    treatment: ExecutedMechanismRunV511,
    *,
    mechanism_id: Literal[
        "competition",
        "independent_review",
        "backward_revision",
        "scientific_adapters",
    ],
) -> ExecutedAblationComparisonV511:
    same_nuisance = control.nuisance_identity == treatment.nuisance_identity
    control_profile = control.profile.model_dump(mode="json")
    treatment_profile = treatment.profile.model_dump(mode="json")
    changed = sorted(
        key
        for key in control_profile
        if key != "schema_version"
        and control_profile[key] != treatment_profile[key]
    )
    exactly_one = changed == [mechanism_id]
    disjoint = not (
        set(control.process_receipt_hashes)
        & set(treatment.process_receipt_hashes)
    )
    event = f"{mechanism_id}_executed"
    expected_control = bool(control_profile[mechanism_id])
    expected_treatment = bool(treatment_profile[mechanism_id])
    observed_delta = (
        expected_control != expected_treatment
        and (event in control.observed_mechanism_events) == expected_control
        and (event in treatment.observed_mechanism_events) == expected_treatment
        and control.output_artifact_hash != treatment.output_artifact_hash
    )
    reasons: list[str] = []
    if not same_nuisance:
        reasons.append("nuisance_identity_differs")
    if not exactly_one:
        reasons.append("not_exactly_one_mechanism_changed")
    if not disjoint:
        reasons.append("process_receipts_reused_across_arms")
    if not observed_delta:
        reasons.append("execution_path_delta_not_observed")
    return ExecutedAblationComparisonV511.seal(
        mechanism_id=mechanism_id,
        control_run_receipt_hash=control.receipt_hash,
        treatment_run_receipt_hash=treatment.receipt_hash,
        same_nuisance_identity=same_nuisance,
        exactly_one_mechanism_changed=exactly_one,
        process_receipts_disjoint=disjoint,
        observed_execution_path_delta=observed_delta,
        valid_executed_ablation=(
            same_nuisance and exactly_one and disjoint and observed_delta
        ),
        development_score_delta=(
            treatment.development_score - control.development_score
        ),
        reasons=sorted(reasons),
    )


__all__ = [
    "ExecutedAblationComparisonV511",
    "ExecutedMechanismRunV511",
    "compare_executed_ablation_v511",
]
