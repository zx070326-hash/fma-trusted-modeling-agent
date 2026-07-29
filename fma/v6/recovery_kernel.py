"""Graph-native, fail-closed recovery for an existing V5 stage workspace.

The recovery kernel deliberately separates three responsibilities:

* a proposer or caller reports a bounded failure observation;
* code normalizes that observation into an earliest affected stage and action;
* the V5 workspace remains the only authority that can revoke graph nodes and
  create a new attempt lineage.

Recovery never edits an accepted artifact in place.  Files owned by a failed
stage are moved into an attempt-scoped quarantine after the graph transition,
while the V5 content-addressed snapshots preserve the historical evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, model_validator

from fma._file_lock import exclusive_file_lock
from fma.hashing import canonical_json, sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256, _assert_timezone
from fma.v5.stage_workspace import STAGES, StageWorkspaceError, StageWorkspaceV50
from fma.v5.workspace_schemas import StageId


FailureCategoryV60 = Literal[
    "operational_transient",
    "partial_artifact",
    "contract_semantics",
    "review_rejection",
    "data_contract",
    "model_assumption",
    "identifiability",
    "numerical_implementation",
    "uncertainty_calibration",
    "decision_support",
    "paper_consistency",
    "capability_gap",
    "private_holdout_exposed",
]
RecoveryActionV60 = Literal[
    "RETRY",
    "PATCH",
    "BRANCH",
    "ACQUIRE_DATA",
    "ABSTAIN",
    "HUMAN",
]
RecoveryTransitionStatusV60 = Literal[
    "SAME_ATTEMPT_RETRY_READY",
    "ATTEMPT_CREATED",
    "ABSTAINED",
    "HUMAN_REQUIRED",
]

_STAGE_INDEX = {stage: index for index, stage in enumerate(STAGES)}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_CATEGORY_ROOT: dict[FailureCategoryV60, StageId | None] = {
    "operational_transient": None,
    "partial_artifact": None,
    "contract_semantics": "S0",
    "review_rejection": None,
    "data_contract": "S2",
    "model_assumption": "S1",
    "identifiability": "S1",
    "numerical_implementation": "S3",
    "uncertainty_calibration": "S4",
    "decision_support": "S2",
    "paper_consistency": "S6",
    "capability_gap": None,
    "private_holdout_exposed": None,
}

_CATEGORY_ACTION: dict[FailureCategoryV60, RecoveryActionV60] = {
    "operational_transient": "RETRY",
    "partial_artifact": "RETRY",
    "contract_semantics": "PATCH",
    "review_rejection": "PATCH",
    "data_contract": "ACQUIRE_DATA",
    "model_assumption": "BRANCH",
    "identifiability": "BRANCH",
    "numerical_implementation": "PATCH",
    "uncertainty_calibration": "PATCH",
    "decision_support": "ACQUIRE_DATA",
    "paper_consistency": "PATCH",
    "capability_gap": "HUMAN",
    "private_holdout_exposed": "ABSTAIN",
}

# These are ownership paths, not manifest membership.  Downstream manifests
# intentionally include upstream files; quarantining by manifest membership
# would therefore destroy valid predecessor evidence.
_OWNED_PATHS: dict[StageId, tuple[str, ...]] = {
    "S0": (
        "problem/contract.json",
        "problem/decision_function.json",
        "docs/regime.json",
        "docs/s0_evaluation_profile_v66.json",
        "docs/measurement_study_design_contract_v67.json",
        "docs/predata_execution_protocol_v67.json",
    ),
    "S1": (
        "docs/candidates.json",
        "docs/assumptions.json",
        "docs/symbols.json",
        "docs/model_spec.json",
        "docs/validation_plan.json",
        "docs/executable_candidate_intent_v62.json",
        "docs/executable_candidate_ir_v62.json",
        "docs/candidate_execution_binding_v67.json",
    ),
    "S2": (
        "data/raw/ode_series.json",
        "data/source_provenance_v62/raw_response.json",
        "data/source_provenance_v62/receipt.json",
        "data/source_provenance_v62/acquisition_authority_receipt.json",
        "data/source_provenance_v62/verification.json",
        "data/source_provenance_v62/binding.json",
        "data/ledger.json",
        "data/processed/manifest.json",
        "data/processed/ode_snapshot.json",
        "docs/adapter_binding.json",
        "docs/decision_value_contract_v62.json",
        "docs/executable_candidate_resolution_v62.json",
        "docs/measurement_schema_v62.json",
        "docs/scientific_success_contract_v61.json",
        "docs/source_contract_v62.json",
        "checks/s2_data_transform_receipt.json",
        "checks/s2_source_reverification_v62.json",
        "src/models/prepare_ode_data.py",
    ),
    "S3": (
        "results/ode_scientific_bundle.json",
        "results/adaptive_positive_series_bundle.json",
        "results/executable_candidate_receipt_v62.json",
        "results/index.json",
        "results/code_manifest.json",
        "results/environment.json",
        "results/fermi_estimate.json",
        "results/artifacts/forecast.json",
        "results/artifacts/forecast_interval.json",
        "checks/ode_replay_input.json",
        "checks/ode_replay_receipt.json",
        "checks/ode_toy_oracle.json",
        "checks/adaptive_replay_input.json",
        "checks/adaptive_replay_receipt.json",
        "checks/adaptive_replay_receipts.json",
        "src/models/run_scalar_ode.py",
        "src/models/run_adaptive_positive_series.py",
    ),
    "S4": (
        "results/rolling_confirmation_v61.json",
        "results/verification_summary.json",
        "results/uq_summary.json",
    ),
    "S5": (
        "results/decision_dossier.json",
        "results/decision_value_evidence_v62.json",
    ),
    "S6": (
        "results/values.json",
        "paper/main.template.tex",
        "paper/build/main.tex",
        "paper/build/main.pdf",
        "paper/build/build_receipt.json",
    ),
}

_PRESERVED_DATA_EVIDENCE_PATHS = {
    "data/source_provenance_v62/raw_response.json",
    "data/source_provenance_v62/receipt.json",
    "data/source_provenance_v62/acquisition_authority_receipt.json",
    "docs/measurement_schema_v62.json",
    "docs/source_contract_v62.json",
}

# These files are deliberately outside every stage manifest.  They project a
# post-S6 conclusion over the *current* S0-S6 certificate chain and therefore
# become stale whenever recovery creates any new stage binding.  Keep the root
# separate from _OWNED_PATHS so an S6 same-attempt retry cannot accidentally
# make a graph-bound closure look current or treat it as S6 evidence.
_POST_GATE_PROJECTION_ROOTS = (".fma/scientific_closure_v62",)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _graph_state_hash(workspace: StageWorkspaceV50) -> str:
    return sha256_value(workspace.graph.project_state().model_dump(mode="json"))


def _sealed(model_type, data: dict[str, object], hash_field: str):
    draft = model_type(**data)
    payload = draft.model_dump(mode="json", exclude={hash_field})
    payload[hash_field] = draft.content_hash()
    return model_type(**payload)


class ProblemSignatureV60(StrictModel):
    """Code-routable description of the mathematical problem surface."""

    schema_version: Literal["6.0"] = "6.0"
    state_kind: Literal["scalar", "vector", "network", "spatial_field"]
    time_kind: Literal["static", "discrete", "continuous"]
    dynamics_kind: Literal[
        "none", "autonomous", "nonautonomous", "stochastic"
    ]
    observation_kind: Literal[
        "complete", "partial", "irregular", "aggregate"
    ]
    task_kind: Literal[
        "explanation", "prediction", "control", "optimization", "design", "mixed"
    ]
    observation_count: Annotated[int, Field(ge=0)]
    positive_observations: bool
    strictly_increasing_time: bool
    constraints: list[Identifier] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_signature(self) -> "ProblemSignatureV60":
        if self.constraints != sorted(set(self.constraints)):
            raise ValueError("problem constraints must be sorted and unique")
        return self


class CapabilityPackV60(StrictModel):
    """One executable domain pack with explicit applicability boundaries."""

    schema_version: Literal["6.0"] = "6.0"
    pack_id: Identifier
    pack_version: Identifier
    state_kinds: list[Literal["scalar", "vector", "network", "spatial_field"]]
    time_kinds: list[Literal["static", "discrete", "continuous"]]
    dynamics_kinds: list[
        Literal["none", "autonomous", "nonautonomous", "stochastic"]
    ]
    observation_kinds: list[
        Literal["complete", "partial", "irregular", "aggregate"]
    ]
    minimum_observations: Annotated[int, Field(ge=1)]
    requires_positive_observations: bool
    requires_strictly_increasing_time: bool
    executor_id: Identifier
    scientific_adapter_ids: Annotated[list[Identifier], Field(min_length=1)]
    baseline_ids: Annotated[list[Identifier], Field(min_length=1)]
    supported_levels: list[Literal["L0", "L1", "L2", "L3", "L4"]]
    pack_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_pack(self) -> "CapabilityPackV60":
        for label, values in (
            ("state_kinds", self.state_kinds),
            ("time_kinds", self.time_kinds),
            ("dynamics_kinds", self.dynamics_kinds),
            ("observation_kinds", self.observation_kinds),
            ("scientific_adapter_ids", self.scientific_adapter_ids),
            ("baseline_ids", self.baseline_ids),
            ("supported_levels", self.supported_levels),
        ):
            if values != sorted(set(values)):
                raise ValueError(f"{label} must be sorted and unique")
        if self.supported_levels != ["L0", "L1", "L2", "L3", "L4"]:
            raise ValueError("a production capability pack must implement L0-L4")
        if self.pack_hash and self.pack_hash != self.content_hash():
            raise ValueError("capability pack hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "pack_hash")

    @classmethod
    def seal(cls, **data: object) -> "CapabilityPackV60":
        return _sealed(cls, data, "pack_hash")

    def assert_sealed(self) -> None:
        if not self.pack_hash or self.pack_hash != self.content_hash():
            raise ValueError("capability pack is not sealed")

    def incompatibilities(self, signature: ProblemSignatureV60) -> list[str]:
        reasons: list[str] = []
        if signature.state_kind not in self.state_kinds:
            reasons.append(f"state_kind:{signature.state_kind}")
        if signature.time_kind not in self.time_kinds:
            reasons.append(f"time_kind:{signature.time_kind}")
        if signature.dynamics_kind not in self.dynamics_kinds:
            reasons.append(f"dynamics_kind:{signature.dynamics_kind}")
        if signature.observation_kind not in self.observation_kinds:
            reasons.append(f"observation_kind:{signature.observation_kind}")
        if signature.observation_count < self.minimum_observations:
            reasons.append(
                f"observation_count:{signature.observation_count}"
                f"<{self.minimum_observations}"
            )
        if (
            self.requires_positive_observations
            and not signature.positive_observations
        ):
            reasons.append("positive_observations:required")
        if (
            self.requires_strictly_increasing_time
            and not signature.strictly_increasing_time
        ):
            reasons.append("strictly_increasing_time:required")
        return sorted(reasons)


class CapabilityDecisionV60(StrictModel):
    schema_version: Literal["6.0"] = "6.0"
    compatible_pack_ids: list[Identifier]
    incompatibilities: dict[Identifier, list[str]]
    status: Literal["ROUTABLE", "CAPABILITY_GAP"]
    decision_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> "CapabilityDecisionV60":
        if self.compatible_pack_ids != sorted(set(self.compatible_pack_ids)):
            raise ValueError("compatible capability IDs must be sorted and unique")
        if list(self.incompatibilities) != sorted(self.incompatibilities):
            raise ValueError("capability incompatibilities must be key-sorted")
        expected = "ROUTABLE" if self.compatible_pack_ids else "CAPABILITY_GAP"
        if self.status != expected:
            raise ValueError("capability status differs from compatible packs")
        if self.decision_hash and self.decision_hash != self.content_hash():
            raise ValueError("capability decision hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "decision_hash")

    @classmethod
    def seal(cls, **data: object) -> "CapabilityDecisionV60":
        return _sealed(cls, data, "decision_hash")


class CapabilityRegistryV60:
    """Small deterministic router; it never silently substitutes a model."""

    def __init__(self) -> None:
        self._packs: dict[str, CapabilityPackV60] = {}

    def register(self, pack: CapabilityPackV60) -> None:
        pack.assert_sealed()
        if pack.pack_id in self._packs:
            raise ValueError(f"duplicate capability pack: {pack.pack_id}")
        self._packs[pack.pack_id] = pack

    def route(self, signature: ProblemSignatureV60) -> CapabilityDecisionV60:
        compatible: list[str] = []
        failures: dict[str, list[str]] = {}
        for pack_id, pack in sorted(self._packs.items()):
            reasons = pack.incompatibilities(signature)
            if reasons:
                failures[pack_id] = reasons
            else:
                compatible.append(pack_id)
        return CapabilityDecisionV60.seal(
            compatible_pack_ids=compatible,
            incompatibilities=failures,
            status="ROUTABLE" if compatible else "CAPABILITY_GAP",
        )


def default_capability_registry_v60() -> CapabilityRegistryV60:
    """Register only capability packs backed by current executable adapters."""

    registry = CapabilityRegistryV60()
    registry.register(
        CapabilityPackV60.seal(
            pack_id="scalar_autonomous_ode_v52",
            pack_version="v5.2",
            state_kinds=["scalar"],
            time_kinds=["continuous"],
            dynamics_kinds=["autonomous"],
            observation_kinds=["complete"],
            minimum_observations=12,
            requires_positive_observations=True,
            requires_strictly_increasing_time=True,
            executor_id="fma_v52_scalar_ode_executor",
            scientific_adapter_ids=["scalar_ode_scientific_adapter"],
            baseline_ids=["constant", "persistence"],
            supported_levels=["L0", "L1", "L2", "L3", "L4"],
        )
    )
    registry.register(
        CapabilityPackV60.seal(
            pack_id="adaptive_positive_series_v57",
            pack_version="v5.7",
            state_kinds=["scalar"],
            time_kinds=["continuous", "discrete"],
            dynamics_kinds=["autonomous", "stochastic"],
            observation_kinds=["complete"],
            # Frozen 0.7 split and eight-point minimum per slice need at
            # least 26 points before the adapter is scientifically runnable.
            minimum_observations=26,
            requires_positive_observations=True,
            requires_strictly_increasing_time=True,
            executor_id="fma_v57_adaptive_positive_series_executor",
            scientific_adapter_ids=[
                "adaptive_positive_series_scientific_adapter"
            ],
            baseline_ids=["persistence"],
            supported_levels=["L0", "L1", "L2", "L3", "L4"],
        )
    )
    return registry


class FailureDiagnosisV60(StrictModel):
    schema_version: Literal["6.0"] = "6.0"
    workspace_spec_hash: Sha256
    failed_stage: StageId
    category: FailureCategoryV60
    failure_code: Identifier
    evidence_refs: Annotated[list[Sha256], Field(min_length=1)]
    failure_signature: Sha256
    earliest_affected_stage: StageId | None
    retryable: bool
    candidate_change_required: bool
    data_change_required: bool
    holdout_exposed: bool
    private_evidence_used: bool
    diagnosed_at: datetime
    diagnosis_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_diagnosis(self) -> "FailureDiagnosisV60":
        _assert_timezone(self.diagnosed_at, "diagnosed_at")
        if self.evidence_refs != sorted(set(self.evidence_refs)):
            raise ValueError("diagnosis evidence refs must be sorted and unique")
        expected_signature = sha256_value(
            {
                "failed_stage": self.failed_stage,
                "category": self.category,
                "failure_code": self.failure_code,
            }
        )
        if self.failure_signature != expected_signature:
            raise ValueError("failure signature differs from diagnosis evidence")
        if self.category == "private_holdout_exposed" and not self.holdout_exposed:
            raise ValueError("private-holdout category requires exposure flag")
        if self.category == "contract_semantics" and (
            self.failed_stage != "S0" or self.earliest_affected_stage != "S0"
        ):
            raise ValueError(
                "contract-semantic recovery is restricted to the S0 root"
            )
        if (
            self.category == "review_rejection"
            and self.earliest_affected_stage != self.failed_stage
        ):
            raise ValueError(
                "review-rejection recovery must restart the rejected stage"
            )
        if self.holdout_exposed and self.retryable:
            raise ValueError("holdout exposure cannot be adaptively retried")
        if self.earliest_affected_stage is not None and (
            _STAGE_INDEX[self.earliest_affected_stage]
            > _STAGE_INDEX[self.failed_stage]
        ):
            raise ValueError("recovery root cannot be downstream of failure")
        if self.diagnosis_hash and self.diagnosis_hash != self.content_hash():
            raise ValueError("diagnosis hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "diagnosis_hash")

    @classmethod
    def seal(cls, **data: object) -> "FailureDiagnosisV60":
        data.setdefault("diagnosed_at", _utc_now())
        return _sealed(cls, data, "diagnosis_hash")


class RecoveryPlanV60(StrictModel):
    schema_version: Literal["6.0"] = "6.0"
    plan_id: Identifier
    diagnosis_hash: Sha256
    failure_signature: Sha256
    action: RecoveryActionV60
    revoke_from: StageId | None
    automatic_execution_permitted: bool
    expected_information_gain: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ]
    forbidden_evidence_refs: list[Sha256] = Field(default_factory=list)
    stop_conditions: Annotated[list[str], Field(min_length=1)]
    attempt_budget_remaining: Annotated[int, Field(ge=0)]
    plan_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_plan(self) -> "RecoveryPlanV60":
        if self.forbidden_evidence_refs != sorted(
            set(self.forbidden_evidence_refs)
        ):
            raise ValueError("forbidden evidence refs must be sorted and unique")
        if self.action in {"ABSTAIN", "HUMAN"}:
            if self.revoke_from is not None or self.automatic_execution_permitted:
                raise ValueError("terminal recovery action cannot mutate the graph")
        if self.action == "RETRY" and self.revoke_from is not None:
            raise ValueError("same-attempt retry cannot revoke a stage")
        if self.action in {"PATCH", "BRANCH", "ACQUIRE_DATA"} and (
            self.revoke_from is None
        ):
            raise ValueError("scientific recovery action requires a graph root")
        if self.plan_hash and self.plan_hash != self.content_hash():
            raise ValueError("recovery plan hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "plan_hash")

    @classmethod
    def seal(cls, **data: object) -> "RecoveryPlanV60":
        return _sealed(cls, data, "plan_hash")


class RecoveryPolicyV60(StrictModel):
    schema_version: Literal["6.0"] = "6.0"
    policy_id: Identifier = "studio-recovery-v60"
    max_scientific_attempts: Annotated[int, Field(ge=1, le=16)] = 3
    max_same_failure: Annotated[int, Field(ge=1, le=8)] = 2
    minimum_information_gain: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ] = 0.05
    private_adaptive_feedback_permitted: Literal[False] = False
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "RecoveryPolicyV60":
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("recovery policy hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    @classmethod
    def seal(cls, **data: object) -> "RecoveryPolicyV60":
        return _sealed(cls, data, "policy_hash")

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("recovery policy is not sealed")


class RecoveryStateV60(StrictModel):
    schema_version: Literal["6.0"] = "6.0"
    workspace_spec_hash: Sha256
    policy_hash: Sha256
    scientific_attempts_started: Annotated[int, Field(ge=1)] = 1
    same_attempt_retries: Annotated[int, Field(ge=0)] = 0
    failure_counts: dict[Sha256, Annotated[int, Field(ge=1)]] = Field(
        default_factory=dict
    )
    stopped: bool = False
    stop_reason: str | None = None
    human_required: bool = False
    human_reason: str | None = None
    last_action: RecoveryActionV60 | None = None
    last_revoke_from: StageId | None = None
    state_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_state(self) -> "RecoveryStateV60":
        if self.stopped != (self.stop_reason is not None):
            raise ValueError("recovery stopped flag and reason disagree")
        if self.human_required != (self.human_reason is not None):
            raise ValueError("human-required flag and reason disagree")
        if self.stopped and self.human_required:
            raise ValueError("recovery cannot be stopped and paused simultaneously")
        if self.state_hash and self.state_hash != self.content_hash():
            raise ValueError("recovery state hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "state_hash")

    @classmethod
    def seal(cls, **data: object) -> "RecoveryStateV60":
        return _sealed(cls, data, "state_hash")


class RecoveryTransitionReceiptV60(StrictModel):
    schema_version: Literal["6.0"] = "6.0"
    diagnosis_hash: Sha256
    plan_hash: Sha256
    before_graph_state_hash: Sha256
    after_graph_state_hash: Sha256
    status: RecoveryTransitionStatusV60
    failed_stage: StageId
    revoke_from: StageId | None
    predecessor_attempt: Annotated[int, Field(ge=1)] | None = None
    successor_attempt: Annotated[int, Field(ge=2)] | None = None
    affected_node_hashes: list[Sha256] = Field(default_factory=list)
    quarantined_file_hashes: dict[str, Sha256] = Field(default_factory=dict)
    executed_at: datetime
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "RecoveryTransitionReceiptV60":
        _assert_timezone(self.executed_at, "executed_at")
        if self.affected_node_hashes != sorted(set(self.affected_node_hashes)):
            raise ValueError("affected node hashes must be sorted and unique")
        if list(self.quarantined_file_hashes) != sorted(
            self.quarantined_file_hashes
        ):
            raise ValueError("quarantined file hashes must be key-sorted")
        created = self.status == "ATTEMPT_CREATED"
        if created != (
            self.revoke_from is not None
            and self.predecessor_attempt is not None
            and self.successor_attempt is not None
            and bool(self.affected_node_hashes)
        ):
            raise ValueError("attempt fields disagree with transition status")
        if created and self.successor_attempt <= self.predecessor_attempt:
            raise ValueError("successor attempt must advance lineage")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("recovery receipt hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    @classmethod
    def seal(cls, **data: object) -> "RecoveryTransitionReceiptV60":
        data.setdefault("executed_at", _utc_now())
        return _sealed(cls, data, "receipt_hash")


class RecoveryTransitionIntentV66(StrictModel):
    """Authenticated write-ahead intent for one graph-mutating recovery."""

    schema_version: Literal["6.6-recovery-transition-intent"] = (
        "6.6-recovery-transition-intent"
    )
    workspace_spec_hash: Sha256
    policy_hash: Sha256
    diagnosis_hash: Sha256
    plan_hash: Sha256
    failure_signature: Sha256
    action: Literal["PATCH", "BRANCH", "ACQUIRE_DATA"]
    failed_stage: StageId
    revoke_from: StageId
    predecessor_attempt: Annotated[int, Field(ge=1)]
    predecessor_work_node_hash: Sha256
    predecessor_gate_node_hash: Sha256
    before_graph_state_hash: Sha256
    expected_successor_attempt: Annotated[int, Field(ge=2)]
    expected_affected_node_hashes: list[Sha256] = Field(min_length=1)
    quarantine_file_hashes: dict[str, Sha256] = Field(default_factory=dict)
    preserve_raw_data: bool
    invalidate_post_gate_projections: Literal[True] = True
    scientific_attempts_started_before: Annotated[int, Field(ge=1)]
    failure_count_before: Annotated[int, Field(ge=0)]
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    intent_hash: Sha256 | None = None
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_intent(self) -> "RecoveryTransitionIntentV66":
        if self.expected_successor_attempt <= self.predecessor_attempt:
            raise ValueError("recovery intent successor must advance lineage")
        if self.expected_affected_node_hashes != sorted(
            set(self.expected_affected_node_hashes)
        ):
            raise ValueError("recovery intent affected nodes must be unique")
        if list(self.quarantine_file_hashes) != sorted(
            self.quarantine_file_hashes
        ):
            raise ValueError("recovery intent quarantine paths must be sorted")
        for relative_path in self.quarantine_file_hashes:
            pure = PurePosixPath(relative_path)
            if (
                pure.is_absolute()
                or not pure.parts
                or any(part in {"", ".", ".."} for part in pure.parts)
            ):
                raise ValueError("recovery intent has an unsafe quarantine path")
        if self.authority_auth_tag and not self.intent_hash:
            raise ValueError("authenticated recovery intent requires intent_hash")
        if self.intent_hash and self.intent_hash != self.content_hash():
            raise ValueError("recovery intent hash differs")
        return self

    def unsigned_hash(self) -> str:
        return sha256_value(
            self.model_dump(
                mode="json",
                exclude={"authority_auth_tag", "intent_hash"},
            )
        )

    def content_hash(self) -> str:
        return sha256_value(
            self.model_dump(mode="json", exclude={"intent_hash"})
        )

    def assert_sealed(self) -> None:
        if (
            not self.authority_auth_tag
            or not self.intent_hash
            or self.intent_hash != self.content_hash()
        ):
            raise ValueError("recovery transition intent is not sealed")


class RecoveryTransitionCompletionV66(StrictModel):
    """Authenticated completion binding an intent to its V6.0 receipt."""

    schema_version: Literal["6.6-recovery-transition-completion"] = (
        "6.6-recovery-transition-completion"
    )
    workspace_spec_hash: Sha256
    policy_hash: Sha256
    intent_hash: Sha256
    transition_receipt_hash: Sha256
    transition_receipt_artifact_hash: Sha256
    completed_at: datetime
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    completion_hash: Sha256 | None = None
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_completion(self) -> "RecoveryTransitionCompletionV66":
        _assert_timezone(self.completed_at, "completed_at")
        if self.authority_auth_tag and not self.completion_hash:
            raise ValueError(
                "authenticated recovery completion requires completion_hash"
            )
        if (
            self.completion_hash
            and self.completion_hash != self.content_hash()
        ):
            raise ValueError("recovery completion hash differs")
        return self

    def unsigned_hash(self) -> str:
        return sha256_value(
            self.model_dump(
                mode="json",
                exclude={"authority_auth_tag", "completion_hash"},
            )
        )

    def content_hash(self) -> str:
        return sha256_value(
            self.model_dump(mode="json", exclude={"completion_hash"})
        )

    def assert_sealed(self) -> None:
        if (
            not self.authority_auth_tag
            or not self.completion_hash
            or self.completion_hash != self.content_hash()
        ):
            raise ValueError("recovery transition completion is not sealed")


class RecoveryEventV60(StrictModel):
    schema_version: Literal["6.0"] = "6.0"
    sequence: Annotated[int, Field(ge=1)]
    event_type: Literal[
        "diagnosed", "planned", "transitioned", "abstained", "human_required"
    ]
    artifact_hash: Sha256
    recorded_at: datetime
    previous_event_hash: Sha256 | None = None
    event_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_event(self) -> "RecoveryEventV60":
        _assert_timezone(self.recorded_at, "recorded_at")
        if self.event_hash and self.event_hash != self.content_hash():
            raise ValueError("recovery event hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "event_hash")

    @classmethod
    def seal(cls, **data: object) -> "RecoveryEventV60":
        data.setdefault("recorded_at", _utc_now())
        return _sealed(cls, data, "event_hash")


class RecoveryKernelV60:
    """Code-owned diagnosis, planning, quarantine, and graph transition."""

    def __init__(
        self,
        workspace: StageWorkspaceV50,
        *,
        policy: RecoveryPolicyV60 | None = None,
    ) -> None:
        self.workspace = workspace
        self.policy = policy or RecoveryPolicyV60.seal()
        self.policy.assert_sealed()
        self.root = workspace.root.resolve()
        self.control_root = self.root / ".fma" / "recovery_v60"
        self.state_path = self.control_root / "state.json"
        self.events_path = self.control_root / "events.jsonl"
        self.writer_lock_path = self.control_root / "writer.lock"

    def _write_json_atomic(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
                default=str,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)

    def _committed_transition_records(
        self,
    ) -> list[
        tuple[
            str,
            RecoveryTransitionReceiptV60,
            FailureDiagnosisV60,
            RecoveryPlanV60,
        ]
    ]:
        """Replay valid committed transition evidence, not the state cache."""

        diagnoses = {
            str(item.diagnosis_hash): (reference, item)
            for reference, item in self.workspace._artifacts_of_kind(
                "failure_diagnosis_v60",
                FailureDiagnosisV60,
            )
        }
        plans = {
            str(item.plan_hash): (reference, item)
            for reference, item in self.workspace._artifacts_of_kind(
                "recovery_plan_v60",
                RecoveryPlanV60,
            )
        }
        receipt_fields = set(RecoveryTransitionReceiptV60.model_fields)
        found: list[
            tuple[
                str,
                RecoveryTransitionReceiptV60,
                FailureDiagnosisV60,
                RecoveryPlanV60,
            ]
        ] = []
        for reference, payload in self.workspace._artifacts_of_kind(
            "recovery_transition_receipt_v60"
        ):
            if not isinstance(payload, dict):
                continue
            try:
                receipt = RecoveryTransitionReceiptV60.model_validate(
                    {
                        key: value
                        for key, value in payload.items()
                        if key in receipt_fields
                    }
                )
            except ValueError:
                continue
            diagnosis_record = diagnoses.get(str(receipt.diagnosis_hash))
            plan_record = plans.get(str(receipt.plan_hash))
            if diagnosis_record is None or plan_record is None:
                continue
            diagnosis_ref, diagnosis = diagnosis_record
            plan_ref, plan = plan_record
            if (
                payload.get("diagnosis_artifact_hash") != diagnosis_ref.sha256
                or payload.get("plan_artifact_hash") != plan_ref.sha256
                or diagnosis.workspace_spec_hash
                != self.workspace.spec.spec_hash
                or plan.diagnosis_hash != diagnosis.diagnosis_hash
                or plan.failure_signature != diagnosis.failure_signature
                or (
                    receipt.status == "ATTEMPT_CREATED"
                    and plan.action
                    not in {"PATCH", "BRANCH", "ACQUIRE_DATA"}
                )
                or (
                    receipt.status == "SAME_ATTEMPT_RETRY_READY"
                    and plan.action != "RETRY"
                )
                or (
                    receipt.status == "ABSTAINED"
                    and plan.action != "ABSTAIN"
                )
                or (
                    receipt.status == "HUMAN_REQUIRED"
                    and plan.action != "HUMAN"
                )
            ):
                continue
            found.append((reference.sha256, receipt, diagnosis, plan))
        return sorted(
            found,
            key=lambda item: (
                item[1].executed_at,
                str(item[1].receipt_hash),
                item[0],
            ),
        )

    def committed_transition_records(
        self,
    ) -> list[
        tuple[
            str,
            RecoveryTransitionReceiptV60,
            FailureDiagnosisV60,
            RecoveryPlanV60,
        ]
    ]:
        """Return replay-verified transition evidence for recovery adapters."""

        return self._committed_transition_records()

    def completed_transition_records(
        self,
    ) -> list[
        tuple[
            str,
            RecoveryTransitionReceiptV60,
            FailureDiagnosisV60,
            RecoveryPlanV60,
        ]
    ]:
        """Return only transitions whose durable completion is verified.

        Non-mutating transitions are complete once their receipt is committed.
        A graph-mutating ``ATTEMPT_CREATED`` receipt is complete only when one
        authenticated V6.6 intent and one authenticated V6.6 completion bind
        that exact receipt artifact.
        """

        records = self._committed_transition_records()
        intents = self._verified_transition_intents()
        completions = {
            str(item.intent_hash): item
            for item in self._verified_transition_completions()
        }
        completed = []
        for record in records:
            receipt_artifact_hash, receipt, diagnosis, plan = record
            if receipt.status != "ATTEMPT_CREATED":
                completed.append(record)
                continue
            matching_intents = [
                item
                for item in intents
                if item.diagnosis_hash == diagnosis.diagnosis_hash
                and item.plan_hash == plan.plan_hash
            ]
            if len(matching_intents) > 1:
                raise StageWorkspaceError(
                    "recovery transition has multiple authenticated intents"
                )
            if not matching_intents:
                continue
            intent = matching_intents[0]
            completion = completions.get(str(intent.intent_hash))
            if completion is None:
                continue
            if (
                completion.transition_receipt_artifact_hash
                != receipt_artifact_hash
                or completion.transition_receipt_hash != receipt.receipt_hash
            ):
                raise StageWorkspaceError(
                    "recovery completion differs from its transition receipt"
                )
            completed.append(record)
        return completed

    def _project_state_from_committed_transitions(
        self,
        cached: RecoveryStateV60,
    ) -> RecoveryStateV60:
        records = self._committed_transition_records()
        if not records:
            return cached
        counts: dict[str, int] = {}
        attempts = 1
        retries = 0
        for _, receipt, diagnosis, _ in records:
            counts[diagnosis.failure_signature] = (
                counts.get(diagnosis.failure_signature, 0) + 1
            )
            attempts += int(receipt.status == "ATTEMPT_CREATED")
            retries += int(receipt.status == "SAME_ATTEMPT_RETRY_READY")
        _, latest_receipt, latest_diagnosis, latest_plan = records[-1]
        projected = RecoveryStateV60.seal(
            workspace_spec_hash=self.workspace.spec.spec_hash,
            policy_hash=self.policy.policy_hash,
            scientific_attempts_started=attempts,
            same_attempt_retries=retries,
            failure_counts=dict(sorted(counts.items())),
            stopped=latest_receipt.status == "ABSTAINED",
            stop_reason=(
                latest_diagnosis.failure_code
                if latest_receipt.status == "ABSTAINED"
                else None
            ),
            human_required=latest_receipt.status == "HUMAN_REQUIRED",
            human_reason=(
                latest_diagnosis.failure_code
                if latest_receipt.status == "HUMAN_REQUIRED"
                else None
            ),
            last_action=latest_plan.action,
            last_revoke_from=latest_plan.revoke_from,
        )
        if (
            cached.scientific_attempts_started
            > projected.scientific_attempts_started
            or cached.same_attempt_retries > projected.same_attempt_retries
            or any(
                count > projected.failure_counts.get(signature, 0)
                for signature, count in cached.failure_counts.items()
            )
        ):
            raise RuntimeError(
                "recovery state exceeds committed transition evidence"
            )
        return projected

    def load_state(self) -> RecoveryStateV60:
        if not self.state_path.is_file():
            cached = RecoveryStateV60.seal(
                workspace_spec_hash=self.workspace.spec.spec_hash,
                policy_hash=self.policy.policy_hash,
            )
        else:
            cached = RecoveryStateV60.model_validate_json(
                self.state_path.read_text(encoding="utf-8")
            )
        if (
            cached.workspace_spec_hash != self.workspace.spec.spec_hash
            or cached.policy_hash != self.policy.policy_hash
        ):
            raise PermissionError("recovery state belongs to another control plane")
        return self._project_state_from_committed_transitions(cached)

    def _save_state(self, state: RecoveryStateV60) -> None:
        self._write_json_atomic(self.state_path, state.model_dump(mode="json"))

    def _verify_transition_intent(
        self,
        intent: RecoveryTransitionIntentV66,
    ) -> bool:
        try:
            intent.assert_sealed()
        except ValueError:
            return False
        return (
            intent.workspace_spec_hash == self.workspace.spec.spec_hash
            and intent.policy_hash == self.policy.policy_hash
            and intent.authority_key_id == self.workspace.authority_key_id
            and intent.authority_auth_tag
            == self.workspace._mac(
                "recovery_transition_intent_v66",
                intent.unsigned_hash(),
            )
        )

    def _verified_transition_intents(
        self,
    ) -> list[RecoveryTransitionIntentV66]:
        intents = [
            item
            for _, item in self.workspace._artifacts_of_kind(
                "recovery_transition_intent_v66",
                RecoveryTransitionIntentV66,
            )
            if self._verify_transition_intent(item)
        ]
        hashes = [str(item.intent_hash) for item in intents]
        if len(hashes) != len(set(hashes)):
            raise StageWorkspaceError(
                "duplicate authenticated recovery transition intents"
            )
        return intents

    def _transition_record_for_intent(
        self,
        intent: RecoveryTransitionIntentV66,
    ) -> tuple[
        str,
        RecoveryTransitionReceiptV60,
        FailureDiagnosisV60,
        RecoveryPlanV60,
    ] | None:
        matches = [
            item
            for item in self._committed_transition_records()
            if item[1].diagnosis_hash == intent.diagnosis_hash
            and item[1].plan_hash == intent.plan_hash
        ]
        if len(matches) > 1:
            raise StageWorkspaceError(
                "recovery intent has multiple committed transition receipts"
            )
        return matches[0] if matches else None

    def _verify_transition_completion(
        self,
        completion: RecoveryTransitionCompletionV66,
    ) -> bool:
        try:
            completion.assert_sealed()
            if (
                completion.workspace_spec_hash
                != self.workspace.spec.spec_hash
                or completion.policy_hash != self.policy.policy_hash
                or completion.authority_key_id
                != self.workspace.authority_key_id
                or completion.authority_auth_tag
                != self.workspace._mac(
                    "recovery_transition_completion_v66",
                    completion.unsigned_hash(),
                )
                or completion.transition_receipt_artifact_hash
                not in self.workspace._committed_artifact_hashes()
            ):
                return False
            receipt_fields = set(RecoveryTransitionReceiptV60.model_fields)
            payload = self.workspace._artifact_payload_by_hash(
                completion.transition_receipt_artifact_hash
            )
            if not isinstance(payload, dict):
                return False
            receipt = RecoveryTransitionReceiptV60.model_validate(
                {
                    key: value
                    for key, value in payload.items()
                    if key in receipt_fields
                }
            )
            intents = [
                item
                for item in self._verified_transition_intents()
                if item.intent_hash == completion.intent_hash
            ]
            if len(intents) != 1:
                return False
            intent = intents[0]
            return (
                receipt.receipt_hash == completion.transition_receipt_hash
                and receipt.diagnosis_hash == intent.diagnosis_hash
                and receipt.plan_hash == intent.plan_hash
                and receipt.status == "ATTEMPT_CREATED"
                and receipt.failed_stage == intent.failed_stage
                and receipt.revoke_from == intent.revoke_from
                and receipt.before_graph_state_hash
                == intent.before_graph_state_hash
                and receipt.predecessor_attempt
                == intent.predecessor_attempt
                and receipt.successor_attempt
                == intent.expected_successor_attempt
                and receipt.affected_node_hashes
                == intent.expected_affected_node_hashes
                and receipt.quarantined_file_hashes
                == intent.quarantine_file_hashes
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            return False

    def _verified_transition_completions(
        self,
    ) -> list[RecoveryTransitionCompletionV66]:
        completions = [
            item
            for _, item in self.workspace._artifacts_of_kind(
                "recovery_transition_completion_v66",
                RecoveryTransitionCompletionV66,
            )
            if self._verify_transition_completion(item)
        ]
        by_intent: dict[str, RecoveryTransitionCompletionV66] = {}
        for item in completions:
            if item.intent_hash in by_intent:
                raise StageWorkspaceError(
                    "recovery intent has multiple authenticated completions"
                )
            by_intent[item.intent_hash] = item
        return list(by_intent.values())

    def _pending_transition_intent(
        self,
    ) -> RecoveryTransitionIntentV66 | None:
        completed = {
            str(item.intent_hash)
            for item in self._verified_transition_completions()
        }
        pending = [
            item
            for item in self._verified_transition_intents()
            if str(item.intent_hash) not in completed
        ]
        if len(pending) > 1:
            raise StageWorkspaceError(
                "multiple recovery transition intents remain open"
            )
        return pending[0] if pending else None

    def _diagnosis_plan_for_intent(
        self,
        intent: RecoveryTransitionIntentV66,
    ) -> tuple[FailureDiagnosisV60, RecoveryPlanV60]:
        diagnoses = [
            item
            for _, item in self.workspace._artifacts_of_kind(
                "failure_diagnosis_v60",
                FailureDiagnosisV60,
            )
            if item.diagnosis_hash == intent.diagnosis_hash
        ]
        plans = [
            item
            for _, item in self.workspace._artifacts_of_kind(
                "recovery_plan_v60",
                RecoveryPlanV60,
            )
            if item.plan_hash == intent.plan_hash
        ]
        if len(diagnoses) != 1 or len(plans) != 1:
            raise StageWorkspaceError(
                "recovery intent lacks one committed diagnosis and plan"
            )
        diagnosis = diagnoses[0]
        plan = plans[0]
        if (
            diagnosis.workspace_spec_hash != intent.workspace_spec_hash
            or diagnosis.failure_signature != intent.failure_signature
            or plan.diagnosis_hash != intent.diagnosis_hash
            or plan.failure_signature != intent.failure_signature
            or plan.action != intent.action
            or plan.revoke_from != intent.revoke_from
            or diagnosis.failed_stage != intent.failed_stage
        ):
            raise StageWorkspaceError(
                "recovery intent differs from committed diagnosis or plan"
            )
        return diagnosis, plan

    def _commit_transition_completion(
        self,
        intent: RecoveryTransitionIntentV66,
        record: tuple[
            str,
            RecoveryTransitionReceiptV60,
            FailureDiagnosisV60,
            RecoveryPlanV60,
        ],
    ) -> RecoveryTransitionCompletionV66:
        existing = [
            item
            for item in self._verified_transition_completions()
            if item.intent_hash == intent.intent_hash
        ]
        if existing:
            return existing[0]
        receipt_artifact_hash, receipt, _, _ = record
        if receipt.receipt_hash is None or intent.intent_hash is None:
            raise StageWorkspaceError(
                "recovery completion inputs are not sealed"
            )
        unsigned = RecoveryTransitionCompletionV66(
            workspace_spec_hash=self.workspace.spec.spec_hash,
            policy_hash=self.policy.policy_hash,
            intent_hash=intent.intent_hash,
            transition_receipt_hash=receipt.receipt_hash,
            transition_receipt_artifact_hash=receipt_artifact_hash,
            completed_at=_utc_now(),
            authority_key_id=self.workspace.authority_key_id,
        )
        payload = unsigned.model_dump(mode="json")
        payload["authority_auth_tag"] = self.workspace._mac(
            "recovery_transition_completion_v66",
            unsigned.unsigned_hash(),
        )
        payload["completion_hash"] = sha256_value(
            {
                key: value
                for key, value in payload.items()
                if key != "completion_hash"
            }
        )
        completion = RecoveryTransitionCompletionV66.model_validate(payload)
        completion.assert_sealed()
        self.workspace.commit_evidence(
            "recovery_transition_completion_v66",
            completion.model_dump(mode="json"),
        )
        if not self._verify_transition_completion(completion):
            raise StageWorkspaceError(
                "recovery completion failed authority verification"
            )
        return completion

    def events(self) -> list[RecoveryEventV60]:
        if not self.events_path.is_file():
            return []
        found: list[RecoveryEventV60] = []
        previous: str | None = None
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            event = RecoveryEventV60.model_validate_json(line)
            if event.previous_event_hash != previous:
                raise RuntimeError("recovery event chain predecessor differs")
            found.append(event)
            previous = event.event_hash
        return found

    def _append_event(self, event_type: str, artifact_hash: str) -> None:
        events = self.events()
        event = RecoveryEventV60.seal(
            sequence=len(events) + 1,
            event_type=event_type,
            artifact_hash=artifact_hash,
            previous_event_hash=events[-1].event_hash if events else None,
        )
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(event.model_dump(mode="json")) + "\n")

    def diagnose(
        self,
        *,
        failed_stage: StageId,
        category: FailureCategoryV60,
        failure_code: str,
        evidence_refs: list[str],
        holdout_exposed: bool = False,
        private_evidence_used: bool = False,
    ) -> FailureDiagnosisV60:
        if not evidence_refs or any(not _SHA256.fullmatch(item) for item in evidence_refs):
            raise ValueError("diagnosis requires one or more SHA-256 evidence refs")
        root = _CATEGORY_ROOT[category]
        if category == "contract_semantics" and failed_stage != "S0":
            raise ValueError(
                "contract-semantic recovery is restricted to failed stage S0"
            )
        if category == "review_rejection":
            root = failed_stage
        status = self.workspace.status().stage_statuses[failed_stage]
        if category in {"operational_transient", "partial_artifact"}:
            root = None if status in {"frontier", "pending"} else failed_stage
        if root is not None and _STAGE_INDEX[root] > _STAGE_INDEX[failed_stage]:
            raise ValueError("failure category is incompatible with failed stage")
        exposed = holdout_exposed or category == "private_holdout_exposed"
        adaptive_forbidden = exposed or private_evidence_used
        signature = sha256_value(
            {
                "failed_stage": failed_stage,
                "category": category,
                "failure_code": failure_code,
            }
        )
        diagnosis = FailureDiagnosisV60.seal(
            workspace_spec_hash=self.workspace.spec.spec_hash,
            failed_stage=failed_stage,
            category=category,
            failure_code=failure_code,
            evidence_refs=sorted(set(evidence_refs)),
            failure_signature=signature,
            earliest_affected_stage=root,
            retryable=(
                category not in {"capability_gap", "private_holdout_exposed"}
                and not adaptive_forbidden
            ),
            candidate_change_required=category
            in {"model_assumption", "identifiability"},
            data_change_required=category
            in {"data_contract", "decision_support"},
            holdout_exposed=exposed,
            private_evidence_used=private_evidence_used,
        )
        ref = self.workspace.commit_evidence(
            "failure_diagnosis_v60", diagnosis.model_dump(mode="json")
        )
        self._append_event("diagnosed", ref.sha256)
        return diagnosis

    def plan(
        self,
        diagnosis: FailureDiagnosisV60,
        *,
        expected_information_gain: float = 0.5,
    ) -> RecoveryPlanV60:
        if not math.isfinite(expected_information_gain):
            raise ValueError("expected information gain must be finite")
        if diagnosis.workspace_spec_hash != self.workspace.spec.spec_hash:
            raise PermissionError("diagnosis belongs to another workspace")
        state = self.load_state()
        repeated = state.failure_counts.get(diagnosis.failure_signature, 0)
        action = _CATEGORY_ACTION[diagnosis.category]
        revoke_from = diagnosis.earliest_affected_stage
        automatic = action not in {"ABSTAIN", "HUMAN"}
        stop_conditions = [
            "private holdout exposure forbids adaptive recovery",
            "scientific attempt budget exhausted",
            "same failure signature repeated beyond policy",
            "an S0 contract-semantic signature may be patched only once",
            "no registered compatible capability pack",
        ]
        if state.stopped:
            action = "ABSTAIN"
            revoke_from = None
            automatic = False
        elif diagnosis.holdout_exposed or diagnosis.private_evidence_used:
            action = "ABSTAIN"
            revoke_from = None
            automatic = False
        elif diagnosis.category == "contract_semantics" and repeated >= 1:
            # A second semantically equivalent S0 rejection must not become
            # an adaptive prompt-tuning loop.  The failure code is expected
            # to be a code-owned normalized finding signature.
            action = "ABSTAIN"
            revoke_from = None
            automatic = False
        elif repeated >= self.policy.max_same_failure:
            action = "ABSTAIN"
            revoke_from = None
            automatic = False
        elif action in {"PATCH", "BRANCH", "ACQUIRE_DATA"} and (
            state.scientific_attempts_started
            >= self.policy.max_scientific_attempts
        ):
            action = "ABSTAIN"
            revoke_from = None
            automatic = False
        elif (
            action in {"PATCH", "BRANCH", "ACQUIRE_DATA"}
            and expected_information_gain < self.policy.minimum_information_gain
        ):
            action = "HUMAN"
            revoke_from = None
            automatic = False
        elif action == "RETRY" and revoke_from is not None:
            action = "PATCH"
        remaining = max(
            self.policy.max_scientific_attempts
            - state.scientific_attempts_started,
            0,
        )
        plan = RecoveryPlanV60.seal(
            plan_id=f"recovery-{diagnosis.failure_signature[:16]}",
            diagnosis_hash=diagnosis.diagnosis_hash,
            failure_signature=diagnosis.failure_signature,
            action=action,
            revoke_from=revoke_from,
            automatic_execution_permitted=automatic,
            expected_information_gain=max(0.0, min(expected_information_gain, 1.0)),
            forbidden_evidence_refs=(
                diagnosis.evidence_refs
                if diagnosis.holdout_exposed or diagnosis.private_evidence_used
                else []
            ),
            stop_conditions=stop_conditions,
            attempt_budget_remaining=remaining,
        )
        ref = self.workspace.commit_evidence(
            "recovery_plan_v60", plan.model_dump(mode="json")
        )
        self._append_event("planned", ref.sha256)
        return plan

    def _owned_paths_from(self, stage: StageId) -> list[str]:
        paths: list[str] = []
        for item_stage in STAGES[_STAGE_INDEX[stage] :]:
            paths.extend(self._owned_paths_for_stage(item_stage))
        return sorted(set(paths))

    def _owned_paths_for_stage(self, stage: StageId) -> tuple[str, ...]:
        """Project source-contract ownership for additive V6.7 workspaces."""

        paths = list(_OWNED_PATHS[stage])
        predata_active = all(
            (self.root / relative_path).is_file()
            for relative_path in (
                "docs/measurement_study_design_contract_v67.json",
                "docs/predata_execution_protocol_v67.json",
            )
        )
        source_path = "docs/source_contract_v62.json"
        if predata_active and stage == "S0" and source_path not in paths:
            paths.append(source_path)
        if predata_active and stage == "S2":
            paths = [path for path in paths if path != source_path]
        return tuple(paths)

    def evidence_refs_for_stage(self, stage: StageId) -> list[str]:
        """Collect public/current hashes without exposing artifact contents."""

        refs: set[str] = {str(self.workspace.spec.spec_hash)}
        try:
            manifest = self.workspace._manifest_for_stage(stage)
        except StageWorkspaceError:
            manifest = None
        if manifest is not None:
            if manifest.manifest_hash:
                refs.add(str(manifest.manifest_hash))
            refs.update(item.sha256 for item in manifest.files)
            checks = self.workspace._latest_checks(
                stage, str(manifest.manifest_hash)
            )
            refs.update(
                str(item.result_hash)
                for item in checks.values()
                if item.result_hash is not None
            )
        else:
            for relative_path in self._owned_paths_for_stage(stage):
                path = self._safe_existing_file(relative_path)
                if path is not None:
                    refs.add(hashlib.sha256(path.read_bytes()).hexdigest())
        return sorted(refs)

    def _safe_existing_file(self, relative_path: str) -> Path | None:
        pure = PurePosixPath(relative_path)
        if (
            pure.is_absolute()
            or not pure.parts
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise ValueError("recovery path must be canonical workspace-relative")
        path = (self.root / Path(*pure.parts)).resolve(strict=False)
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError("recovery path escapes workspace") from exc
        current = self.root
        for part in pure.parts:
            current = current / part
            if current.is_symlink():
                raise PermissionError("recovery refuses symlink artifacts")
        if not path.exists():
            return None
        if not path.is_file():
            raise PermissionError("recovery owns files, not directories")
        return path

    def _safe_projection_files(self, relative_root: str) -> list[str]:
        """Enumerate a code-owned projection tree without following links."""

        pure = PurePosixPath(relative_root)
        if (
            pure.is_absolute()
            or not pure.parts
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise ValueError(
                "projection root must be canonical workspace-relative"
            )
        root = (self.root / Path(*pure.parts)).resolve(strict=False)
        try:
            root.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError("projection root escapes workspace") from exc
        current = self.root
        for part in pure.parts:
            current = current / part
            if current.is_symlink():
                raise PermissionError("recovery refuses symlink projections")
        if not root.exists():
            return []
        if not root.is_dir():
            raise PermissionError("post-gate projection root must be a directory")

        relative_files: list[str] = []
        for directory, child_directories, filenames in os.walk(
            root, followlinks=False
        ):
            directory_path = Path(directory)
            for child_name in child_directories:
                if (directory_path / child_name).is_symlink():
                    raise PermissionError(
                        "recovery refuses symlink projections"
                    )
            for filename in filenames:
                file_path = directory_path / filename
                if file_path.is_symlink() or not file_path.is_file():
                    raise PermissionError(
                        "recovery refuses non-file projection artifacts"
                    )
                try:
                    relative = file_path.resolve(strict=True).relative_to(
                        self.root
                    )
                except ValueError as exc:
                    raise PermissionError(
                        "projection artifact escapes workspace"
                    ) from exc
                relative_path = relative.as_posix()
                if self._safe_existing_file(relative_path) is None:
                    raise OSError("projection artifact disappeared")
                relative_files.append(relative_path)
        return sorted(relative_files)

    def _remove_empty_projection_roots(self) -> None:
        for relative_root in _POST_GATE_PROJECTION_ROOTS:
            root = self.root / Path(*PurePosixPath(relative_root).parts)
            if not root.exists():
                continue
            directories = sorted(
                (path for path in root.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts),
                reverse=True,
            )
            for directory in directories:
                directory.rmdir()
            root.rmdir()

    def _quarantine_candidates(
        self,
        *,
        recovery_root: StageId,
        preserve_raw_data: bool,
        invalidate_post_gate_projections: bool = False,
    ) -> list[str]:
        candidates = self._owned_paths_from(recovery_root)
        if invalidate_post_gate_projections:
            for projection_root in _POST_GATE_PROJECTION_ROOTS:
                candidates.extend(self._safe_projection_files(projection_root))
        if preserve_raw_data:
            candidates = [
                item
                for item in candidates
                if not item.startswith("data/raw/")
                and item not in _PRESERVED_DATA_EVIDENCE_PATHS
            ]
        return sorted(set(candidates))

    def _quarantine_root(
        self,
        *,
        failed_stage: StageId,
        attempt: int,
    ) -> Path:
        root = (
            self.control_root
            / "attempts"
            / f"a{attempt}"
            / failed_stage.lower()
            / "quarantine"
        ).resolve(strict=False)
        try:
            root.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError("quarantine root escapes workspace") from exc
        return root

    def _quarantine_snapshot(
        self,
        *,
        failed_stage: StageId,
        recovery_root: StageId,
        attempt: int,
        preserve_raw_data: bool,
        invalidate_post_gate_projections: bool = False,
        include_existing_quarantine: bool = False,
    ) -> dict[str, str]:
        candidates = self._quarantine_candidates(
            recovery_root=recovery_root,
            preserve_raw_data=preserve_raw_data,
            invalidate_post_gate_projections=(
                invalidate_post_gate_projections
            ),
        )
        quarantine_root = self._quarantine_root(
            failed_stage=failed_stage,
            attempt=attempt,
        )
        snapshot: dict[str, str] = {}
        for relative_path in candidates:
            source = self._safe_existing_file(relative_path)
            destination = (
                quarantine_root / Path(*PurePosixPath(relative_path).parts)
            ).resolve(strict=False)
            try:
                destination.relative_to(quarantine_root)
            except ValueError as exc:
                raise PermissionError("quarantine path escapes attempt") from exc
            if source is not None and destination.exists():
                raise StageWorkspaceError(
                    "recovery artifact exists in both active and quarantine "
                    f"locations: {relative_path}"
                )
            if source is not None:
                snapshot[relative_path] = hashlib.sha256(
                    source.read_bytes()
                ).hexdigest()
            elif include_existing_quarantine and destination.exists():
                if destination.is_symlink() or not destination.is_file():
                    raise PermissionError(
                        "recovery refuses non-file quarantine artifacts"
                    )
                snapshot[relative_path] = hashlib.sha256(
                    destination.read_bytes()
                ).hexdigest()
        return dict(sorted(snapshot.items()))

    def _quarantine_expected(
        self,
        *,
        failed_stage: StageId,
        recovery_root: StageId,
        attempt: int,
        preserve_raw_data: bool,
        invalidate_post_gate_projections: bool,
        expected_file_hashes: dict[str, str],
    ) -> dict[str, str]:
        active_candidates = self._quarantine_candidates(
            recovery_root=recovery_root,
            preserve_raw_data=preserve_raw_data,
            invalidate_post_gate_projections=(
                invalidate_post_gate_projections
            ),
        )
        quarantine_root = self._quarantine_root(
            failed_stage=failed_stage,
            attempt=attempt,
        )
        unexpected_active = [
            relative_path
            for relative_path in active_candidates
            if relative_path not in expected_file_hashes
            and self._safe_existing_file(relative_path) is not None
        ]
        if unexpected_active:
            raise StageWorkspaceError(
                "recovery found active artifacts absent from authenticated "
                "intent"
            )
        if quarantine_root.exists():
            if quarantine_root.is_symlink() or not quarantine_root.is_dir():
                raise PermissionError(
                    "recovery quarantine root must be a real directory"
                )
            quarantined_paths: set[str] = set()
            for directory, child_directories, filenames in os.walk(
                quarantine_root,
                followlinks=False,
            ):
                directory_path = Path(directory)
                if any(
                    (directory_path / child).is_symlink()
                    for child in child_directories
                ):
                    raise PermissionError(
                        "recovery refuses symlink quarantine directories"
                    )
                for filename in filenames:
                    path = directory_path / filename
                    if path.is_symlink() or not path.is_file():
                        raise PermissionError(
                            "recovery refuses non-file quarantine artifacts"
                        )
                    quarantined_paths.add(
                        path.relative_to(quarantine_root).as_posix()
                    )
            if not quarantined_paths.issubset(expected_file_hashes):
                raise StageWorkspaceError(
                    "recovery found quarantine artifacts absent from "
                    "authenticated intent"
                )
        moved: dict[str, str] = {}
        for relative_path, digest in expected_file_hashes.items():
            source = self._safe_existing_file(relative_path)
            destination = (
                quarantine_root / Path(
                    *PurePosixPath(relative_path).parts
                )
            ).resolve(strict=False)
            try:
                destination.relative_to(quarantine_root)
            except ValueError as exc:
                raise PermissionError(
                    "quarantine path escapes attempt"
                ) from exc
            if source is not None and destination.exists():
                raise StageWorkspaceError(
                    "recovery artifact exists in both active and quarantine "
                    f"locations: {relative_path}"
                )
            if source is None and not destination.exists():
                raise StageWorkspaceError(
                    "recovery artifact disappeared during quarantine: "
                    f"{relative_path}"
                )
            if source is not None:
                if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                    raise StageWorkspaceError(
                        "recovery source hash differs from authenticated intent"
                    )
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, destination)
            if destination.is_symlink() or not destination.is_file():
                raise PermissionError(
                    "recovery refuses non-file quarantine artifacts"
                )
            if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                raise OSError("quarantined artifact hash changed")
            moved[relative_path] = digest
        if invalidate_post_gate_projections:
            self._remove_empty_projection_roots()
        return dict(sorted(moved.items()))

    def _quarantine(
        self,
        *,
        failed_stage: StageId,
        recovery_root: StageId,
        attempt: int,
        preserve_raw_data: bool,
        invalidate_post_gate_projections: bool = False,
    ) -> dict[str, str]:
        expected = self._quarantine_snapshot(
            failed_stage=failed_stage,
            recovery_root=recovery_root,
            attempt=attempt,
            preserve_raw_data=preserve_raw_data,
            invalidate_post_gate_projections=(
                invalidate_post_gate_projections
            ),
            include_existing_quarantine=True,
        )
        return self._quarantine_expected(
            failed_stage=failed_stage,
            recovery_root=recovery_root,
            attempt=attempt,
            preserve_raw_data=preserve_raw_data,
            invalidate_post_gate_projections=(
                invalidate_post_gate_projections
            ),
            expected_file_hashes=expected,
        )

    def _create_or_load_transition_intent(
        self,
        diagnosis: FailureDiagnosisV60,
        plan: RecoveryPlanV60,
        state: RecoveryStateV60,
        *,
        before_graph_state_hash: str,
    ) -> RecoveryTransitionIntentV66:
        pending = self._pending_transition_intent()
        if pending is not None:
            if (
                pending.diagnosis_hash != diagnosis.diagnosis_hash
                or pending.plan_hash != plan.plan_hash
            ):
                raise StageWorkspaceError(
                    "another authenticated recovery transition remains open"
                )
            return pending
        if (
            plan.action not in {"PATCH", "BRANCH", "ACQUIRE_DATA"}
            or plan.revoke_from is None
            or plan.plan_hash is None
            or diagnosis.diagnosis_hash is None
        ):
            raise PermissionError(
                "only a sealed graph-mutating plan may create an intent"
            )
        predecessor = self.workspace._latest_attempt(plan.revoke_from)
        work = self.workspace._binding(plan.revoke_from, "work")
        gate = self.workspace._binding(plan.revoke_from, "gate")
        graph_state = self.workspace.graph.project_state()
        affected = sorted(
            self.workspace.graph._revocation_closure(
                str(work.node_hash),
                graph_state.edges,
            )
        )
        expected_successor = (
            max(
                self.workspace._latest_attempt(stage)
                for stage in STAGES[_STAGE_INDEX[plan.revoke_from] :]
            )
            + 1
        )
        preserve_raw_data = plan.action == "BRANCH"
        quarantine = self._quarantine_snapshot(
            failed_stage=diagnosis.failed_stage,
            recovery_root=plan.revoke_from,
            attempt=predecessor,
            preserve_raw_data=preserve_raw_data,
            invalidate_post_gate_projections=True,
        )
        unsigned = RecoveryTransitionIntentV66(
            workspace_spec_hash=self.workspace.spec.spec_hash,
            policy_hash=self.policy.policy_hash,
            diagnosis_hash=diagnosis.diagnosis_hash,
            plan_hash=plan.plan_hash,
            failure_signature=diagnosis.failure_signature,
            action=plan.action,
            failed_stage=diagnosis.failed_stage,
            revoke_from=plan.revoke_from,
            predecessor_attempt=predecessor,
            predecessor_work_node_hash=work.node_hash,
            predecessor_gate_node_hash=gate.node_hash,
            before_graph_state_hash=before_graph_state_hash,
            expected_successor_attempt=expected_successor,
            expected_affected_node_hashes=affected,
            quarantine_file_hashes=quarantine,
            preserve_raw_data=preserve_raw_data,
            scientific_attempts_started_before=(
                state.scientific_attempts_started
            ),
            failure_count_before=state.failure_counts.get(
                diagnosis.failure_signature,
                0,
            ),
            authority_key_id=self.workspace.authority_key_id,
        )
        payload = unsigned.model_dump(mode="json")
        payload["authority_auth_tag"] = self.workspace._mac(
            "recovery_transition_intent_v66",
            unsigned.unsigned_hash(),
        )
        payload["intent_hash"] = sha256_value(
            {
                key: value
                for key, value in payload.items()
                if key != "intent_hash"
            }
        )
        intent = RecoveryTransitionIntentV66.model_validate(payload)
        intent.assert_sealed()
        self.workspace.commit_evidence(
            "recovery_transition_intent_v66",
            intent.model_dump(mode="json"),
        )
        if not self._verify_transition_intent(intent):
            raise StageWorkspaceError(
                "recovery transition intent failed authority verification"
            )
        return intent

    def _execute_transition_intent(
        self,
        intent: RecoveryTransitionIntentV66,
        diagnosis: FailureDiagnosisV60,
        plan: RecoveryPlanV60,
    ) -> tuple[list[str], dict[str, str], int, int]:
        if not self._verify_transition_intent(intent):
            raise PermissionError("recovery transition intent is not authentic")
        if (
            intent.diagnosis_hash != diagnosis.diagnosis_hash
            or intent.plan_hash != plan.plan_hash
            or intent.failure_signature != diagnosis.failure_signature
            or intent.action != plan.action
            or intent.revoke_from != plan.revoke_from
        ):
            raise PermissionError(
                "recovery transition intent differs from supplied plan"
            )
        state = self.workspace.graph.project_state()
        current_attempt = self.workspace._latest_attempt(intent.revoke_from)
        predecessor_status = state.snapshot.node_statuses.get(
            intent.predecessor_work_node_hash
        )
        if current_attempt == intent.predecessor_attempt:
            current_work = self.workspace._binding(intent.revoke_from, "work")
            current_gate = self.workspace._binding(intent.revoke_from, "gate")
            if (
                predecessor_status == "revoked"
                or current_work.node_hash
                != intent.predecessor_work_node_hash
                or current_gate.node_hash
                != intent.predecessor_gate_node_hash
            ):
                raise StageWorkspaceError(
                    "recovery graph mutation is partial and requires human "
                    "reconciliation"
                )
            affected = self.workspace.invalidate_from(
                intent.revoke_from,
                reason=(
                    f"V6 recovery {plan.action} for "
                    f"{diagnosis.failure_code}; diagnosis "
                    f"{diagnosis.diagnosis_hash}; intent "
                    f"{intent.intent_hash}"
                ),
                authority="verifier",
            )
            if (
                sorted(affected) != intent.expected_affected_node_hashes
                or self.workspace._latest_attempt(intent.revoke_from)
                != intent.expected_successor_attempt
            ):
                raise StageWorkspaceError(
                    "recovery graph transition differs from authenticated intent"
                )
        elif current_attempt == intent.expected_successor_attempt:
            state = self.workspace.graph.project_state()
            if (
                predecessor_status != "revoked"
                or any(
                    state.snapshot.node_statuses.get(node_hash) != "revoked"
                    for node_hash in intent.expected_affected_node_hashes
                )
            ):
                raise StageWorkspaceError(
                    "recovery successor lacks the intended revocation closure"
                )
            current_work = self.workspace._binding(intent.revoke_from, "work")
            current_gate = self.workspace._binding(intent.revoke_from, "gate")
            supersedes = {
                (
                    str(edge.source_node_hash),
                    str(edge.target_node_hash),
                )
                for edge in state.edges
                if edge.relation == "supersedes"
            }
            if (
                (
                    intent.predecessor_work_node_hash,
                    str(current_work.node_hash),
                )
                not in supersedes
                or (
                    intent.predecessor_gate_node_hash,
                    str(current_gate.node_hash),
                )
                not in supersedes
            ):
                raise StageWorkspaceError(
                    "recovery successor lineage differs from intent"
                )
            affected = list(intent.expected_affected_node_hashes)
        else:
            raise StageWorkspaceError(
                "recovery graph attempt differs from authenticated intent"
            )
        if not self.workspace.verify():
            raise StageWorkspaceError(
                "workspace graph failed verification during recovery replay"
            )
        quarantined = self._quarantine_expected(
            failed_stage=intent.failed_stage,
            recovery_root=intent.revoke_from,
            attempt=intent.predecessor_attempt,
            preserve_raw_data=intent.preserve_raw_data,
            invalidate_post_gate_projections=(
                intent.invalidate_post_gate_projections
            ),
            expected_file_hashes=intent.quarantine_file_hashes,
        )
        return (
            sorted(affected),
            quarantined,
            intent.predecessor_attempt,
            intent.expected_successor_attempt,
        )

    def execute(
        self,
        diagnosis: FailureDiagnosisV60,
        plan: RecoveryPlanV60,
    ) -> RecoveryTransitionReceiptV60:
        if plan.diagnosis_hash != diagnosis.diagnosis_hash:
            raise PermissionError("plan does not bind supplied diagnosis")
        if plan.failure_signature != diagnosis.failure_signature:
            raise PermissionError("plan failure signature differs")
        pending_intent = self._pending_transition_intent()
        if pending_intent is not None:
            if (
                pending_intent.diagnosis_hash != diagnosis.diagnosis_hash
                or pending_intent.plan_hash != plan.plan_hash
            ):
                raise StageWorkspaceError(
                    "another authenticated recovery transition remains open"
                )
            existing_record = self._transition_record_for_intent(
                pending_intent
            )
            if existing_record is not None:
                self._commit_transition_completion(
                    pending_intent,
                    existing_record,
                )
                projected = self.load_state()
                self._save_state(projected)
                self._append_event("transitioned", existing_record[0])
                if not self.workspace.verify():
                    raise StageWorkspaceError(
                        "workspace failed verification after recovery replay"
                    )
                return existing_record[1]
        state = self.load_state()
        if state.stopped and plan.action not in {"ABSTAIN", "HUMAN"}:
            raise PermissionError("recovery state is stopped")
        if diagnosis.category == "contract_semantics" and plan.action not in {
            "ABSTAIN",
            "HUMAN",
        }:
            if (
                plan.action != "PATCH"
                or plan.revoke_from != "S0"
                or not plan.automatic_execution_permitted
            ):
                raise PermissionError(
                    "contract-semantic recovery requires an authorized S0 patch"
                )
            if state.human_required:
                raise PermissionError(
                    "contract-semantic recovery is paused for human input"
                )
            if state.failure_counts.get(diagnosis.failure_signature, 0) >= 1:
                raise PermissionError(
                    "contract-semantic failure signature was already patched"
                )
            if (
                state.scientific_attempts_started
                >= self.policy.max_scientific_attempts
            ):
                raise PermissionError(
                    "contract-semantic recovery attempt budget is exhausted"
                )
        before = _graph_state_hash(self.workspace)
        affected: list[str] = []
        quarantined: dict[str, str] = {}
        predecessor: int | None = None
        successor: int | None = None

        if plan.action == "ABSTAIN":
            status: RecoveryTransitionStatusV60 = "ABSTAINED"
        elif plan.action == "HUMAN":
            status = "HUMAN_REQUIRED"
        elif plan.action == "RETRY":
            if plan.revoke_from is not None:
                raise PermissionError("same-attempt retry cannot revoke graph state")
            attempt = self.workspace._latest_attempt(diagnosis.failed_stage)
            quarantined = self._quarantine(
                failed_stage=diagnosis.failed_stage,
                recovery_root=diagnosis.failed_stage,
                attempt=attempt,
                preserve_raw_data=diagnosis.failed_stage == "S2",
            )
            status = "SAME_ATTEMPT_RETRY_READY"
        else:
            if not plan.automatic_execution_permitted or plan.revoke_from is None:
                raise PermissionError("plan is not authorized for graph mutation")
            intent = self._create_or_load_transition_intent(
                diagnosis,
                plan,
                state,
                before_graph_state_hash=before,
            )
            before = intent.before_graph_state_hash
            (
                affected,
                quarantined,
                predecessor,
                successor,
            ) = self._execute_transition_intent(
                intent,
                diagnosis,
                plan,
            )
            status = "ATTEMPT_CREATED"

        after = _graph_state_hash(self.workspace)
        receipt = RecoveryTransitionReceiptV60.seal(
            diagnosis_hash=diagnosis.diagnosis_hash,
            plan_hash=plan.plan_hash,
            before_graph_state_hash=before,
            after_graph_state_hash=after,
            status=status,
            failed_stage=diagnosis.failed_stage,
            revoke_from=plan.revoke_from,
            predecessor_attempt=predecessor,
            successor_attempt=successor,
            affected_node_hashes=sorted(affected),
            quarantined_file_hashes=quarantined,
        )
        diagnosis_ref = self.workspace.commit_evidence(
            "failure_diagnosis_v60", diagnosis.model_dump(mode="json")
        )
        plan_ref = self.workspace.commit_evidence(
            "recovery_plan_v60", plan.model_dump(mode="json")
        )
        receipt_ref = self.workspace.commit_evidence(
            "recovery_transition_receipt_v60",
            {
                **receipt.model_dump(mode="json"),
                "diagnosis_artifact_hash": diagnosis_ref.sha256,
                "plan_artifact_hash": plan_ref.sha256,
            },
        )
        if status == "ATTEMPT_CREATED":
            intent = self._pending_transition_intent()
            if intent is None:
                raise StageWorkspaceError(
                    "graph-mutating recovery lost its transition intent"
                )
            record = self._transition_record_for_intent(intent)
            if record is None or record[0] != receipt_ref.sha256:
                raise StageWorkspaceError(
                    "recovery receipt does not complete its transition intent"
                )
            self._commit_transition_completion(intent, record)

        counts = dict(state.failure_counts)
        counts[diagnosis.failure_signature] = (
            counts.get(diagnosis.failure_signature, 0) + 1
        )
        stopped = status == "ABSTAINED"
        human_required = status == "HUMAN_REQUIRED"
        next_state = RecoveryStateV60.seal(
            workspace_spec_hash=state.workspace_spec_hash,
            policy_hash=state.policy_hash,
            scientific_attempts_started=state.scientific_attempts_started
            + int(status == "ATTEMPT_CREATED"),
            same_attempt_retries=state.same_attempt_retries
            + int(status == "SAME_ATTEMPT_RETRY_READY"),
            failure_counts=counts,
            stopped=stopped,
            stop_reason=diagnosis.failure_code if stopped else None,
            human_required=human_required,
            human_reason=(
                diagnosis.failure_code if human_required else None
            ),
            last_action=plan.action,
            last_revoke_from=plan.revoke_from,
        )
        self._save_state(next_state)
        self._append_event(
            {
                "ABSTAINED": "abstained",
                "HUMAN_REQUIRED": "human_required",
            }.get(status, "transitioned"),
            receipt_ref.sha256,
        )
        if not self.workspace.verify():
            raise StageWorkspaceError(
                "workspace failed graph verification after V6 recovery"
            )
        return receipt

    def recover(
        self,
        *,
        failed_stage: StageId,
        category: FailureCategoryV60,
        failure_code: str,
        evidence_refs: list[str],
        expected_information_gain: float = 0.5,
        holdout_exposed: bool = False,
        private_evidence_used: bool = False,
    ) -> tuple[
        FailureDiagnosisV60,
        RecoveryPlanV60,
        RecoveryTransitionReceiptV60,
    ]:
        with exclusive_file_lock(self.writer_lock_path):
            pending = self._pending_transition_intent()
            if pending is not None:
                diagnosis, plan = self._diagnosis_plan_for_intent(pending)
                if (
                    diagnosis.failed_stage != failed_stage
                    or diagnosis.category != category
                    or diagnosis.failure_code != failure_code
                    or diagnosis.evidence_refs
                    != sorted(set(evidence_refs))
                    or diagnosis.holdout_exposed
                    != (
                        holdout_exposed
                        or category == "private_holdout_exposed"
                    )
                    or diagnosis.private_evidence_used
                    != private_evidence_used
                ):
                    raise StageWorkspaceError(
                        "an authenticated recovery transition must be "
                        "reconciled with its original failure evidence"
                    )
                receipt = self.execute(diagnosis, plan)
                return diagnosis, plan, receipt
            diagnosis = self.diagnose(
                failed_stage=failed_stage,
                category=category,
                failure_code=failure_code,
                evidence_refs=evidence_refs,
                holdout_exposed=holdout_exposed,
                private_evidence_used=private_evidence_used,
            )
            plan = self.plan(
                diagnosis,
                expected_information_gain=expected_information_gain,
            )
            receipt = self.execute(diagnosis, plan)
            return diagnosis, plan, receipt

    def summary(self) -> dict[str, object]:
        state = self.load_state()
        events = self.events()
        return {
            "schema_version": "6.0",
            "policy_hash": self.policy.policy_hash,
            "scientific_attempts_started": state.scientific_attempts_started,
            "attempt_budget_remaining": max(
                self.policy.max_scientific_attempts
                - state.scientific_attempts_started,
                0,
            ),
            "same_attempt_retries": state.same_attempt_retries,
            "distinct_failure_signatures": len(state.failure_counts),
            "stopped": state.stopped,
            "stop_reason": state.stop_reason,
            "human_required": state.human_required,
            "human_reason": state.human_reason,
            "last_action": state.last_action,
            "last_revoke_from": state.last_revoke_from,
            "event_count": len(events),
            "last_event_hash": events[-1].event_hash if events else None,
            "private_adaptive_feedback_permitted": False,
            "scientific_qualification_granted": False,
            "real_world_action_authorized": False,
        }


__all__ = [
    "CapabilityDecisionV60",
    "CapabilityPackV60",
    "CapabilityRegistryV60",
    "FailureCategoryV60",
    "FailureDiagnosisV60",
    "ProblemSignatureV60",
    "RecoveryEventV60",
    "RecoveryKernelV60",
    "RecoveryPlanV60",
    "RecoveryPolicyV60",
    "RecoveryStateV60",
    "RecoveryTransitionCompletionV66",
    "RecoveryTransitionIntentV66",
    "RecoveryTransitionReceiptV60",
    "default_capability_registry_v60",
]
