"""V6.7 persistent handoff from S1 review rejection to bounded repair.

This module is deliberately independent of the Studio runtime exception type.
It records an already-issued independent-review rejection and an already-
executed RecoveryKernel transition.  It does not sign a gate, choose a graph
rollback root, mutate a workspace, or establish any scientific claim.
"""

from __future__ import annotations

from typing import Annotated, Literal, Sequence, cast

from pydantic import Field, model_validator

from fma.hashing import canonical_json, sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256

from .recovery_kernel import (
    FailureDiagnosisV60,
    RecoveryPlanV60,
    RecoveryTransitionReceiptV60,
)


S1_FORMALIZATION_REJECTION_EVIDENCE_PATH_V67 = (
    "checks/s1_formalization_rejection_evidence_v67.json"
)
S1_FORMALIZATION_REJECTION_HANDOFF_PATH_V67 = (
    "checks/s1_formalization_rejection_handoff_v67.json"
)
S1_BOUNDED_REPAIR_CONTEXT_PATH_V67 = (
    "checks/s1_bounded_repair_context_v67.json"
)
S1_FORMALIZATION_FAILURE_CODE_V67 = "s1_formalization_review_rejected"
S1_MAX_DISCLOSED_FINDINGS_V67 = 12
S1_MAX_FINDING_CHARACTERS_V67 = 600
S1_MAX_REPAIR_CONTEXT_CHARACTERS_V67 = 8_000

FindingTextV67 = Annotated[
    str,
    Field(min_length=3, max_length=S1_MAX_FINDING_CHARACTERS_V67),
]


def _hash_without(model: StrictModel, field_name: str) -> str:
    return sha256_value(
        model.model_dump(mode="json", exclude={field_name})
    )


def s1_review_failure_signature_v67() -> str:
    """Return the code-only RecoveryKernel signature.

    Reviewer prose and its text-derived finding signature are intentionally
    absent, so paraphrasing cannot create a new failure class or evade a
    repeated-failure budget.
    """

    return sha256_value(
        {
            "failed_stage": "S1",
            "category": "review_rejection",
            "failure_code": S1_FORMALIZATION_FAILURE_CODE_V67,
        }
    )


def normalize_s1_review_findings_v67(
    findings: Sequence[str],
) -> tuple[str, ...]:
    """Normalize only the bounded findings disclosed to a repair model."""

    normalized = {
        " ".join(str(item).split()).casefold()
        for item in findings
        if str(item).strip()
    }
    if not normalized:
        normalized = {"reviewer rejected without a normalized finding"}
    ordered = tuple(sorted(normalized))
    if len(ordered) > S1_MAX_DISCLOSED_FINDINGS_V67:
        raise ValueError("S1 rejection exceeds the bounded finding count")
    if any(
        len(item) > S1_MAX_FINDING_CHARACTERS_V67 for item in ordered
    ):
        raise ValueError("S1 rejection finding exceeds the disclosure bound")
    return ordered


def s1_reviewer_finding_signature_v67(
    normalized_findings: Sequence[str],
) -> str:
    """Reproduce the runtime's opaque reviewer-finding commitment."""

    values = tuple(normalized_findings)
    if values != tuple(sorted(set(values))) or not values:
        raise ValueError(
            "normalized S1 findings must be sorted, unique, and nonempty"
        )
    return sha256_value(
        {
            "schema_version": "6.7-s1-formalization-rejection",
            "stage": "S1",
            "recovery_category": "review_rejection",
            "normalized_findings": list(values),
        }
    )


class S1RejectionFindingV67(StrictModel):
    """One necessary reviewer finding, without rationale or hidden reasoning."""

    schema_version: Literal["6.7-s1-rejection-finding"] = (
        "6.7-s1-rejection-finding"
    )
    finding_id: Identifier
    normalized_finding: FindingTextV67
    normalized_finding_hash: Sha256
    source: Literal["independent_s1_formalization_reviewer"] = (
        "independent_s1_formalization_reviewer"
    )
    finding_is_scientific_fact: Literal[False] = False
    finding_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_finding(self) -> "S1RejectionFindingV67":
        if self.normalized_finding != (
            " ".join(self.normalized_finding.split()).casefold()
        ):
            raise ValueError("S1 rejection finding is not normalized")
        expected_text_hash = sha256_value(
            {"normalized_finding": self.normalized_finding}
        )
        if self.normalized_finding_hash != expected_text_hash:
            raise ValueError("S1 normalized finding hash differs")
        expected_id = f"s1rf-{expected_text_hash[:24]}"
        if self.finding_id != expected_id:
            raise ValueError("S1 rejection finding ID is not harness-derived")
        if self.finding_hash and self.finding_hash != self.content_hash():
            raise ValueError("S1 rejection finding hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "finding_hash")

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if not self.finding_hash or self.finding_hash != self.content_hash():
            raise ValueError("S1 rejection finding is not sealed")

    @classmethod
    def seal(cls, *, normalized_finding: str) -> "S1RejectionFindingV67":
        text_hash = sha256_value(
            {"normalized_finding": normalized_finding}
        )
        draft = cls(
            finding_id=f"s1rf-{text_hash[:24]}",
            normalized_finding=normalized_finding,
            normalized_finding_hash=text_hash,
        )
        return cls(
            **draft.model_dump(exclude={"finding_hash"}),
            finding_hash=draft.content_hash(),
        )


class S1FormalizationRejectionHandoffV67(StrictModel):
    """Write-ahead S1 rejection record committed before graph mutation."""

    schema_version: Literal["6.7-s1-formalization-rejection-handoff"] = (
        "6.7-s1-formalization-rejection-handoff"
    )
    workspace_spec_hash: Sha256
    s0_gate_hash: Sha256
    predata_protocol_hash: Sha256
    reviewer_receipt_hash: Sha256
    reviewer_finding_signature: Sha256
    failure_category: Literal["review_rejection"] = "review_rejection"
    failure_code: Literal["s1_formalization_review_rejected"] = (
        S1_FORMALIZATION_FAILURE_CODE_V67
    )
    failure_signature: Sha256
    predecessor_attempt: Annotated[int, Field(ge=1)]
    recovery_disposition: Literal["bounded_patch", "terminal_human"]
    expected_information_gain: Literal[0.0, 0.5]
    existing_repair_context_hash: Sha256 | None = None
    base_evidence_refs: list[Sha256] = Field(min_length=1, max_length=5)
    findings: list[S1RejectionFindingV67] = Field(
        min_length=1,
        max_length=S1_MAX_DISCLOSED_FINDINGS_V67,
    )
    source_role: Literal["independent_reviewer"] = "independent_reviewer"
    observation_data_included: Literal[False] = False
    reviewer_rationale_included: Literal[False] = False
    reviewer_uncertainties_included: Literal[False] = False
    private_evidence_used: Literal[False] = False
    holdout_exposed: Literal[False] = False
    rollback_root_selected_by_evidence: Literal[False] = False
    gate_certificate_issued: Literal[False] = False
    scientific_failure_established: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    handoff_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_handoff(
        self,
    ) -> "S1FormalizationRejectionHandoffV67":
        if self.failure_signature != s1_review_failure_signature_v67():
            raise ValueError(
                "S1 rejection handoff failure signature is not code-derived"
            )
        for finding in self.findings:
            finding.assert_sealed()
        keys = [
            (item.normalized_finding, item.finding_id)
            for item in self.findings
        ]
        if keys != sorted(set(keys)):
            raise ValueError(
                "S1 rejection handoff findings must be sorted and unique"
            )
        expected_finding_signature = s1_reviewer_finding_signature_v67(
            [item.normalized_finding for item in self.findings]
        )
        if self.reviewer_finding_signature != expected_finding_signature:
            raise ValueError("S1 rejection handoff finding signature differs")
        expected_refs = sorted(
            {
                self.workspace_spec_hash,
                self.s0_gate_hash,
                self.predata_protocol_hash,
                self.reviewer_receipt_hash,
                self.reviewer_finding_signature,
            }
        )
        if self.base_evidence_refs != expected_refs:
            raise ValueError(
                "S1 rejection handoff evidence refs are not harness-derived"
            )
        if self.recovery_disposition == "bounded_patch":
            if (
                self.expected_information_gain != 0.5
                or self.existing_repair_context_hash is not None
            ):
                raise ValueError(
                    "first S1 rejection must request one bounded patch"
                )
        elif (
            self.expected_information_gain != 0.0
            or self.existing_repair_context_hash is None
        ):
            raise ValueError(
                "repeated S1 rejection must terminate for human review"
            )
        if self.handoff_hash and self.handoff_hash != self.content_hash():
            raise ValueError("S1 rejection handoff hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "handoff_hash")

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if not self.handoff_hash or self.handoff_hash != self.content_hash():
            raise ValueError("S1 rejection handoff is not sealed")


class S1FormalizationRejectionEvidenceV67(StrictModel):
    """Persistent, claim-limited evidence for one rejected S1 attempt."""

    schema_version: Literal["6.7-s1-formalization-rejection-evidence"] = (
        "6.7-s1-formalization-rejection-evidence"
    )
    workspace_spec_hash: Sha256
    s0_gate_hash: Sha256
    predata_protocol_hash: Sha256
    reviewer_receipt_hash: Sha256
    reviewer_finding_signature: Sha256
    failure_category: Literal["review_rejection"] = "review_rejection"
    failure_code: Literal["s1_formalization_review_rejected"] = (
        S1_FORMALIZATION_FAILURE_CODE_V67
    )
    failure_signature: Sha256
    diagnosis_hash: Sha256
    diagnosis_evidence_refs: list[Sha256] = Field(min_length=1)
    recovery_plan_hash: Sha256
    recovery_receipt_hash: Sha256
    predecessor_attempt: Annotated[int, Field(ge=1)]
    successor_attempt: Annotated[int, Field(ge=2)]
    findings: list[S1RejectionFindingV67] = Field(
        min_length=1,
        max_length=S1_MAX_DISCLOSED_FINDINGS_V67,
    )
    source_role: Literal["independent_reviewer"] = "independent_reviewer"
    private_evidence_used: Literal[False] = False
    holdout_exposed: Literal[False] = False
    rollback_root_selected_by_evidence: Literal[False] = False
    gate_certificate_issued: Literal[False] = False
    scientific_failure_established: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_evidence(
        self,
    ) -> "S1FormalizationRejectionEvidenceV67":
        if self.failure_signature != s1_review_failure_signature_v67():
            raise ValueError(
                "S1 rejection failure signature is not code-derived"
            )
        if self.successor_attempt != self.predecessor_attempt + 1:
            raise ValueError("S1 recovery successor must advance one attempt")
        if self.diagnosis_evidence_refs != sorted(
            set(self.diagnosis_evidence_refs)
        ):
            raise ValueError(
                "S1 diagnosis evidence refs must be sorted and unique"
            )
        for finding in self.findings:
            finding.assert_sealed()
        keys = [
            (item.normalized_finding, item.finding_id)
            for item in self.findings
        ]
        if keys != sorted(set(keys)):
            raise ValueError("S1 rejection findings must be sorted and unique")
        expected_finding_signature = s1_reviewer_finding_signature_v67(
            [item.normalized_finding for item in self.findings]
        )
        if self.reviewer_finding_signature != expected_finding_signature:
            raise ValueError("S1 reviewer finding signature differs")
        if self.evidence_hash and self.evidence_hash != self.content_hash():
            raise ValueError("S1 rejection evidence hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evidence_hash")

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if not self.evidence_hash or self.evidence_hash != self.content_hash():
            raise ValueError("S1 rejection evidence is not sealed")


class S1RepairFindingV67(StrictModel):
    """Minimal finding projection disclosed to the successor repair model."""

    finding_id: Identifier
    normalized_finding: FindingTextV67
    source_finding_hash: Sha256

    @model_validator(mode="after")
    def validate_projection(self) -> "S1RepairFindingV67":
        source = S1RejectionFindingV67.seal(
            normalized_finding=self.normalized_finding
        )
        if (
            self.finding_id != source.finding_id
            or self.source_finding_hash != source.finding_hash
        ):
            raise ValueError(
                "S1 repair finding differs from its sealed source projection"
            )
        return self


class S1BoundedRepairContextV67(StrictModel):
    """The complete feedback packet permitted for one successor repair."""

    schema_version: Literal["6.7-s1-bounded-repair-context"] = (
        "6.7-s1-bounded-repair-context"
    )
    workspace_spec_hash: Sha256
    s0_gate_hash: Sha256
    predata_protocol_hash: Sha256
    source_rejection_evidence_hash: Sha256
    reviewer_receipt_hash: Sha256
    reviewer_finding_signature: Sha256
    failure_code: Literal["s1_formalization_review_rejected"] = (
        S1_FORMALIZATION_FAILURE_CODE_V67
    )
    failure_signature: Sha256
    recovery_receipt_hash: Sha256
    predecessor_attempt: Annotated[int, Field(ge=1)]
    successor_attempt: Annotated[int, Field(ge=2)]
    repair_attempts_used: Literal[0] = 0
    maximum_model_repair_attempts: Literal[1] = 1
    remaining_model_repair_attempts: Literal[1] = 1
    allowed_task: Literal[
        "repair_candidate_to_match_frozen_predata_protocol"
    ] = "repair_candidate_to_match_frozen_predata_protocol"
    findings: list[S1RepairFindingV67] = Field(
        min_length=1,
        max_length=S1_MAX_DISCLOSED_FINDINGS_V67,
    )
    disclosure_policy: Literal[
        "necessary_normalized_findings_only"
    ] = "necessary_normalized_findings_only"
    protocol_change_permitted: Literal[False] = False
    adapter_change_permitted: Literal[False] = False
    threshold_change_permitted: Literal[False] = False
    observation_data_included: Literal[False] = False
    reviewer_rationale_included: Literal[False] = False
    reviewer_uncertainties_included: Literal[False] = False
    private_evidence_included: Literal[False] = False
    holdout_evidence_included: Literal[False] = False
    rollback_root_included: Literal[False] = False
    gate_signing_authority: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    context_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_context(self) -> "S1BoundedRepairContextV67":
        if self.failure_signature != s1_review_failure_signature_v67():
            raise ValueError(
                "S1 repair failure signature is not code-derived"
            )
        if self.successor_attempt != self.predecessor_attempt + 1:
            raise ValueError("S1 repair context lineage does not advance once")
        keys = [
            (item.normalized_finding, item.finding_id)
            for item in self.findings
        ]
        if keys != sorted(set(keys)):
            raise ValueError("S1 repair findings must be sorted and unique")
        expected_finding_signature = s1_reviewer_finding_signature_v67(
            [item.normalized_finding for item in self.findings]
        )
        if self.reviewer_finding_signature != expected_finding_signature:
            raise ValueError("S1 repair finding signature differs")
        if (
            len(canonical_json(self.model_dump(mode="json")))
            > S1_MAX_REPAIR_CONTEXT_CHARACTERS_V67
        ):
            raise ValueError("S1 repair context exceeds its disclosure bound")
        if self.context_hash and self.context_hash != self.content_hash():
            raise ValueError("S1 repair context hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "context_hash")

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if not self.context_hash or self.context_hash != self.content_hash():
            raise ValueError("S1 bounded repair context is not sealed")


def _assert_recovery_chain_v67(
    *,
    workspace_spec_hash: str,
    diagnosis: FailureDiagnosisV60,
    plan: RecoveryPlanV60,
    recovery_receipt: RecoveryTransitionReceiptV60,
) -> None:
    if (
        not diagnosis.diagnosis_hash
        or diagnosis.diagnosis_hash != diagnosis.content_hash()
    ):
        raise ValueError("S1 recovery diagnosis is not sealed")
    if not plan.plan_hash or plan.plan_hash != plan.content_hash():
        raise ValueError("S1 recovery plan is not sealed")
    if (
        not recovery_receipt.receipt_hash
        or recovery_receipt.receipt_hash
        != recovery_receipt.content_hash()
    ):
        raise ValueError("S1 recovery receipt is not sealed")
    expected_failure_signature = s1_review_failure_signature_v67()
    if (
        diagnosis.workspace_spec_hash != workspace_spec_hash
        or diagnosis.failed_stage != "S1"
        or diagnosis.category != "review_rejection"
        or diagnosis.failure_code
        != S1_FORMALIZATION_FAILURE_CODE_V67
        or diagnosis.failure_signature != expected_failure_signature
        or diagnosis.earliest_affected_stage != "S1"
        or not diagnosis.retryable
        or diagnosis.candidate_change_required
        or diagnosis.data_change_required
        or diagnosis.holdout_exposed
        or diagnosis.private_evidence_used
    ):
        raise ValueError("S1 recovery diagnosis differs from rejection")
    if (
        plan.diagnosis_hash != diagnosis.diagnosis_hash
        or plan.failure_signature != expected_failure_signature
        or plan.action != "PATCH"
        or plan.revoke_from != "S1"
        or not plan.automatic_execution_permitted
    ):
        raise ValueError("S1 recovery plan differs from bounded patch policy")
    if (
        recovery_receipt.diagnosis_hash != diagnosis.diagnosis_hash
        or recovery_receipt.plan_hash != plan.plan_hash
        or recovery_receipt.status != "ATTEMPT_CREATED"
        or recovery_receipt.failed_stage != "S1"
        or recovery_receipt.revoke_from != "S1"
        or recovery_receipt.predecessor_attempt is None
        or recovery_receipt.successor_attempt is None
        or recovery_receipt.successor_attempt
        != recovery_receipt.predecessor_attempt + 1
    ):
        raise ValueError("S1 recovery receipt differs from graph transition")


def build_s1_formalization_rejection_evidence_v67(
    *,
    workspace_spec_hash: str,
    s0_gate_hash: str,
    predata_protocol_hash: str,
    reviewer_receipt_hash: str,
    findings: Sequence[str],
    reviewer_finding_signature: str | None = None,
    diagnosis: FailureDiagnosisV60,
    plan: RecoveryPlanV60,
    recovery_receipt: RecoveryTransitionReceiptV60,
) -> S1FormalizationRejectionEvidenceV67:
    """Bind a runtime-neutral rejection to an executed graph transition."""

    _assert_recovery_chain_v67(
        workspace_spec_hash=workspace_spec_hash,
        diagnosis=diagnosis,
        plan=plan,
        recovery_receipt=recovery_receipt,
    )
    normalized = normalize_s1_review_findings_v67(findings)
    computed_finding_signature = s1_reviewer_finding_signature_v67(
        normalized
    )
    if (
        reviewer_finding_signature is not None
        and reviewer_finding_signature != computed_finding_signature
    ):
        raise ValueError(
            "supplied S1 reviewer finding signature differs from findings"
        )
    sealed_findings = [
        S1RejectionFindingV67.seal(normalized_finding=item)
        for item in normalized
    ]
    draft = S1FormalizationRejectionEvidenceV67(
        workspace_spec_hash=workspace_spec_hash,
        s0_gate_hash=s0_gate_hash,
        predata_protocol_hash=predata_protocol_hash,
        reviewer_receipt_hash=reviewer_receipt_hash,
        reviewer_finding_signature=computed_finding_signature,
        failure_signature=s1_review_failure_signature_v67(),
        diagnosis_hash=cast(str, diagnosis.diagnosis_hash),
        diagnosis_evidence_refs=diagnosis.evidence_refs,
        recovery_plan_hash=cast(str, plan.plan_hash),
        recovery_receipt_hash=cast(str, recovery_receipt.receipt_hash),
        predecessor_attempt=cast(int, recovery_receipt.predecessor_attempt),
        successor_attempt=cast(int, recovery_receipt.successor_attempt),
        findings=sealed_findings,
    )
    return S1FormalizationRejectionEvidenceV67(
        **draft.model_dump(exclude={"evidence_hash"}),
        evidence_hash=draft.content_hash(),
    )


def build_s1_formalization_rejection_handoff_v67(
    *,
    workspace_spec_hash: str,
    s0_gate_hash: str,
    predata_protocol_hash: str,
    reviewer_receipt_hash: str,
    findings: Sequence[str],
    predecessor_attempt: int,
    existing_repair_context_hash: str | None,
    reviewer_finding_signature: str | None = None,
) -> S1FormalizationRejectionHandoffV67:
    """Seal the exact runtime-neutral failure before recovery mutates the graph."""

    normalized = normalize_s1_review_findings_v67(findings)
    computed_finding_signature = s1_reviewer_finding_signature_v67(
        normalized
    )
    if (
        reviewer_finding_signature is not None
        and reviewer_finding_signature != computed_finding_signature
    ):
        raise ValueError(
            "supplied S1 reviewer finding signature differs from findings"
        )
    sealed_findings = [
        S1RejectionFindingV67.seal(normalized_finding=item)
        for item in normalized
    ]
    disposition: Literal["bounded_patch", "terminal_human"] = (
        "terminal_human"
        if existing_repair_context_hash is not None
        else "bounded_patch"
    )
    draft = S1FormalizationRejectionHandoffV67(
        workspace_spec_hash=workspace_spec_hash,
        s0_gate_hash=s0_gate_hash,
        predata_protocol_hash=predata_protocol_hash,
        reviewer_receipt_hash=reviewer_receipt_hash,
        reviewer_finding_signature=computed_finding_signature,
        failure_signature=s1_review_failure_signature_v67(),
        predecessor_attempt=predecessor_attempt,
        recovery_disposition=disposition,
        expected_information_gain=(
            0.0 if disposition == "terminal_human" else 0.5
        ),
        existing_repair_context_hash=existing_repair_context_hash,
        base_evidence_refs=sorted(
            {
                workspace_spec_hash,
                s0_gate_hash,
                predata_protocol_hash,
                reviewer_receipt_hash,
                computed_finding_signature,
            }
        ),
        findings=sealed_findings,
    )
    return S1FormalizationRejectionHandoffV67(
        **draft.model_dump(exclude={"handoff_hash"}),
        handoff_hash=draft.content_hash(),
    )


def s1_recovery_evidence_refs_v67(
    handoff: S1FormalizationRejectionHandoffV67,
    *,
    handoff_artifact_hash: str,
) -> list[str]:
    """Bind RecoveryKernel diagnosis to the committed write-ahead handoff."""

    handoff.assert_sealed()
    if (
        len(handoff_artifact_hash) != 64
        or any(
            character not in "0123456789abcdef"
            for character in handoff_artifact_hash
        )
    ):
        raise ValueError("S1 handoff artifact hash is not SHA-256")
    return sorted(
        {
            *handoff.base_evidence_refs,
            handoff_artifact_hash,
        }
    )


def build_s1_bounded_repair_context_v67(
    evidence: S1FormalizationRejectionEvidenceV67,
) -> S1BoundedRepairContextV67:
    """Project only necessary findings from sealed rejection evidence."""

    evidence.assert_sealed()
    findings = [
        S1RepairFindingV67(
            finding_id=item.finding_id,
            normalized_finding=item.normalized_finding,
            source_finding_hash=cast(str, item.finding_hash),
        )
        for item in evidence.findings
    ]
    draft = S1BoundedRepairContextV67(
        workspace_spec_hash=evidence.workspace_spec_hash,
        s0_gate_hash=evidence.s0_gate_hash,
        predata_protocol_hash=evidence.predata_protocol_hash,
        source_rejection_evidence_hash=cast(
            str, evidence.evidence_hash
        ),
        reviewer_receipt_hash=evidence.reviewer_receipt_hash,
        reviewer_finding_signature=evidence.reviewer_finding_signature,
        failure_signature=evidence.failure_signature,
        recovery_receipt_hash=evidence.recovery_receipt_hash,
        predecessor_attempt=evidence.predecessor_attempt,
        successor_attempt=evidence.successor_attempt,
        findings=findings,
    )
    return S1BoundedRepairContextV67(
        **draft.model_dump(exclude={"context_hash"}),
        context_hash=draft.content_hash(),
    )


__all__ = [
    "S1_BOUNDED_REPAIR_CONTEXT_PATH_V67",
    "S1_FORMALIZATION_FAILURE_CODE_V67",
    "S1_FORMALIZATION_REJECTION_EVIDENCE_PATH_V67",
    "S1_FORMALIZATION_REJECTION_HANDOFF_PATH_V67",
    "S1_MAX_DISCLOSED_FINDINGS_V67",
    "S1_MAX_FINDING_CHARACTERS_V67",
    "S1_MAX_REPAIR_CONTEXT_CHARACTERS_V67",
    "S1BoundedRepairContextV67",
    "S1FormalizationRejectionEvidenceV67",
    "S1FormalizationRejectionHandoffV67",
    "S1RejectionFindingV67",
    "S1RepairFindingV67",
    "build_s1_bounded_repair_context_v67",
    "build_s1_formalization_rejection_evidence_v67",
    "build_s1_formalization_rejection_handoff_v67",
    "normalize_s1_review_findings_v67",
    "s1_recovery_evidence_refs_v67",
    "s1_review_failure_signature_v67",
    "s1_reviewer_finding_signature_v67",
]
