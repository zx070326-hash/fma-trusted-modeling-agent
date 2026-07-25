from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import Field, model_validator

from fma.hashing import canonical_json, sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore
from fma.v2.schemas import Identifier, Sha256, _assert_timezone

from .graph_loop import (
    GraphEdgeV40,
    GraphLoopContractV40,
    GraphLoopStoreV40,
    GraphNodeV40,
)


ProposalSourceV42 = Literal["seed", "prescribed", "generated"]
OperatorChannelV42 = Literal["prescribed", "generated"]
OperatorKindV42 = Literal[
    "replace_skeleton",
    "add_mechanism",
    "relax_assumption",
    "simplify",
    "regularize",
    "combine",
    "invent_structure",
    "reparameterize",
]
DevelopmentDispositionV42 = Literal["advance", "mutate", "discard"]
CheckpointPhaseV42 = Literal[
    "model_space_validated",
    "candidate_admitted",
    "candidate_evaluated",
    "evolution_expanded",
    "reconciled",
    "champion_frozen",
]
FailureStageV42 = Literal["model_space", "development"]
TerminalStatusV42 = Literal[
    "development_champion_frozen",
    "no_development_candidate",
]
PolicyScalar = float | int | str | bool


MODEL_SPACE_CHECKS_V42 = [
    "acyclic",
    "arity_valid",
    "complexity_budget",
    "executable_adapter_available",
    "lineage_valid",
    "no_forbidden_token",
    "primitive_allowed",
    "private_data_absent",
    "symbols_resolved",
    "units_valid",
]


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


def _assert_sorted_unique(values: list[str], name: str) -> None:
    if values != sorted(set(values)):
        raise ValueError(f"{name} must be sorted and unique")


class PrimitiveRuleV42(StrictModel):
    schema_version: Literal["4.2"] = "4.2"
    primitive_id: Identifier
    role: Literal["source", "mechanism", "transform", "combiner"]
    input_units: Annotated[list[Annotated[str, Field(min_length=1)]], Field(max_length=8)]
    output_unit: Annotated[str, Field(min_length=1, max_length=100)]
    max_uses_per_candidate: Annotated[int, Field(ge=1, le=32)] = 8


class OpenModelGrammarV42(StrictModel):
    """Frozen typed construction grammar.

    The grammar constrains primitives and executable adapters, not the set of
    generation>0 family names.  That is the opening through which the model
    space may grow without giving the generator admission authority.
    """

    schema_version: Literal["4.2"] = "4.2"
    grammar_id: Identifier
    seed_families: Annotated[list[Identifier], Field(min_length=1, max_length=64)]
    primitives: Annotated[list[PrimitiveRuleV42], Field(min_length=1, max_length=128)]
    executable_adapter_ids: Annotated[
        list[Identifier], Field(min_length=1, max_length=32)
    ]
    executable_adapter_hashes: dict[Identifier, Sha256]
    forbidden_tokens: list[Annotated[str, Field(min_length=1, max_length=100)]] = (
        Field(default_factory=lambda: ["private_confirmation", "qualification_granted"])
    )
    max_symbols: Annotated[int, Field(ge=1, le=256)] = 64
    max_applications: Annotated[int, Field(ge=1, le=512)] = 128
    grammar_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_grammar(self) -> "OpenModelGrammarV42":
        _assert_sorted_unique(self.seed_families, "seed_families")
        _assert_sorted_unique(
            self.executable_adapter_ids, "executable_adapter_ids"
        )
        if sorted(self.executable_adapter_hashes) != self.executable_adapter_ids:
            raise ValueError(
                "executable_adapter_hashes must exactly cover executable_adapter_ids"
            )
        primitive_ids = [item.primitive_id for item in self.primitives]
        if primitive_ids != sorted(set(primitive_ids)):
            raise ValueError("primitives must be sorted by unique primitive_id")
        lowered = [item.lower() for item in self.forbidden_tokens]
        if lowered != sorted(set(lowered)):
            raise ValueError("forbidden_tokens must be sorted and unique")
        if self.grammar_hash and self.grammar_hash != self.content_hash():
            raise ValueError("grammar_hash does not match open model grammar")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "grammar_hash")

    def assert_sealed(self) -> None:
        if not self.grammar_hash or self.grammar_hash != self.content_hash():
            raise ValueError("open model grammar is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "OpenModelGrammarV42":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"grammar_hash"}),
            grammar_hash=draft.content_hash(),
        )


class OpenEvolutionCampaignSpecV42(StrictModel):
    schema_version: Literal["4.2"] = "4.2"
    campaign_id: Identifier
    evaluator_epoch: Identifier
    objective: Annotated[str, Field(min_length=10, max_length=3000)]
    development_data_hash: Sha256
    grammar_hash: Sha256
    required_development_gates: Annotated[
        list[Identifier], Field(min_length=1, max_length=64)
    ]
    evaluation_policy: dict[Identifier, PolicyScalar] = Field(default_factory=dict)
    max_generations: Annotated[int, Field(ge=1, le=32)] = 3
    max_candidates: Annotated[int, Field(ge=1, le=512)] = 24
    prescribed_quota_per_failure: Annotated[int, Field(ge=0, le=32)] = 2
    generated_quota_per_failure: Annotated[int, Field(ge=0, le=32)] = 2
    max_recovery_attempts: Annotated[int, Field(ge=1, le=32)] = 4
    execution_policy: Literal["local_replay_safe_idempotent"] = (
        "local_replay_safe_idempotent"
    )
    recovery_policy: Literal["reconcile_from_committed_graph"] = (
        "reconcile_from_committed_graph"
    )
    private_data_access_permitted: Literal[False] = False
    created_at: datetime
    spec_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "OpenEvolutionCampaignSpecV42":
        _assert_timezone(self.created_at, "created_at")
        _assert_sorted_unique(
            self.required_development_gates, "required_development_gates"
        )
        if (
            self.prescribed_quota_per_failure
            + self.generated_quota_per_failure
            == 0
            and self.max_generations > 1
        ):
            raise ValueError("multi-generation campaign needs an evolution channel")
        for key, value in self.evaluation_policy.items():
            if isinstance(value, float) and not (-float("inf") < value < float("inf")):
                raise ValueError(f"evaluation policy {key} is non-finite")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match open-evolution campaign")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "spec_hash")

    def assert_sealed(self) -> None:
        if not self.spec_hash or self.spec_hash != self.content_hash():
            raise ValueError("open-evolution campaign is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "OpenEvolutionCampaignSpecV42":
        data.setdefault("created_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"spec_hash"}),
            spec_hash=draft.content_hash(),
        )


class ModelSymbolV42(StrictModel):
    schema_version: Literal["4.2"] = "4.2"
    symbol_id: Identifier
    role: Literal["observed", "parameter", "state", "latent", "derived", "output"]
    unit: Annotated[str, Field(min_length=1, max_length=100)]


class PrimitiveApplicationV42(StrictModel):
    schema_version: Literal["4.2"] = "4.2"
    application_id: Identifier
    primitive_id: Identifier
    inputs: Annotated[list[Identifier], Field(max_length=8)]
    output: Identifier


class OpenModelCandidateV42(StrictModel):
    """Untrusted model-space proposal.

    Structural errors deliberately remain representable so the verifier can
    reject them inside the graph instead of losing the failed proposal outside
    the audit trail.
    """

    schema_version: Literal["4.2"] = "4.2"
    candidate_id: Identifier
    generation: Annotated[int, Field(ge=0, le=32)]
    family: Identifier
    source: ProposalSourceV42
    proposed_by: Literal["model", "harness"]
    executable_adapter_id: Identifier
    symbols: Annotated[list[ModelSymbolV42], Field(min_length=1, max_length=256)]
    applications: Annotated[
        list[PrimitiveApplicationV42], Field(min_length=1, max_length=512)
    ]
    assumptions: Annotated[
        list[Annotated[str, Field(min_length=3, max_length=1000)]],
        Field(min_length=1, max_length=32),
    ]
    parent_candidate_hashes: list[Sha256] = Field(default_factory=list)
    operator_hashes: list[Sha256] = Field(default_factory=list)
    rationale: Annotated[str, Field(min_length=10, max_length=4000)]
    expected_failure_modes: Annotated[
        list[Annotated[str, Field(min_length=3, max_length=500)]],
        Field(min_length=1, max_length=32),
    ]
    private_data_references: list[str] = Field(default_factory=list)
    candidate_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_hash(self) -> "OpenModelCandidateV42":
        if self.candidate_hash and self.candidate_hash != self.content_hash():
            raise ValueError("candidate_hash does not match open model candidate")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "candidate_hash")

    def assert_sealed(self) -> None:
        if not self.candidate_hash or self.candidate_hash != self.content_hash():
            raise ValueError("open model candidate is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "OpenModelCandidateV42":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"candidate_hash"}),
            candidate_hash=draft.content_hash(),
        )


class ModelSpaceValidationV42(StrictModel):
    schema_version: Literal["4.2"] = "4.2"
    candidate_hash: Sha256
    grammar_hash: Sha256
    evaluator_epoch: Identifier
    checks: dict[Identifier, bool]
    admitted: bool
    diagnostic_codes: list[Identifier] = Field(default_factory=list)
    private_data_accessed: Literal[False] = False
    validation_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_assessment(self) -> "ModelSpaceValidationV42":
        if sorted(self.checks) != MODEL_SPACE_CHECKS_V42:
            raise ValueError("model-space validation check set differs from V4.2")
        if self.admitted != all(self.checks.values()):
            raise ValueError("admission result differs from model-space checks")
        _assert_sorted_unique(self.diagnostic_codes, "diagnostic_codes")
        if self.validation_hash and self.validation_hash != self.content_hash():
            raise ValueError("validation_hash does not match model-space validation")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "validation_hash")

    def assert_sealed(self) -> None:
        if not self.validation_hash or self.validation_hash != self.content_hash():
            raise ValueError("model-space validation is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ModelSpaceValidationV42":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"validation_hash"}),
            validation_hash=draft.content_hash(),
        )


class ModelAdmissionReceiptV42(StrictModel):
    schema_version: Literal["4.2"] = "4.2"
    campaign_spec_hash: Sha256
    grammar_hash: Sha256
    candidate_hash: Sha256
    validation_hash: Sha256
    status: Literal["admitted_for_development_unqualified"] = (
        "admitted_for_development_unqualified"
    )
    qualification_granted: Literal[False] = False
    private_confirmation_consumed: Literal[False] = False
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "ModelAdmissionReceiptV42":
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match model admission")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("model admission receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ModelAdmissionReceiptV42":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


class ExecutionAttemptV42(StrictModel):
    schema_version: Literal["4.2"] = "4.2"
    campaign_spec_hash: Sha256
    candidate_hash: Sha256
    development_data_hash: Sha256
    executable_adapter_id: Identifier
    executable_adapter_hash: Sha256
    idempotency_key: Sha256
    replay_safe: Literal[True] = True
    attempt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_attempt(self) -> "ExecutionAttemptV42":
        if self.attempt_hash and self.attempt_hash != self.content_hash():
            raise ValueError("attempt_hash does not match execution attempt")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "attempt_hash")

    def assert_sealed(self) -> None:
        if not self.attempt_hash or self.attempt_hash != self.content_hash():
            raise ValueError("execution attempt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ExecutionAttemptV42":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"attempt_hash"}),
            attempt_hash=draft.content_hash(),
        )


class DevelopmentExecutionV42(StrictModel):
    schema_version: Literal["4.2"] = "4.2"
    candidate_hash: Sha256
    attempt_hash: Sha256
    idempotency_key: Sha256
    development_data_hash: Sha256
    converged: bool
    metrics: dict[Identifier, Annotated[float, Field(allow_inf_nan=False)]]
    domain_payload: dict[str, object]
    execution_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_execution(self) -> "DevelopmentExecutionV42":
        if self.execution_hash and self.execution_hash != self.content_hash():
            raise ValueError("execution_hash does not match development execution")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "execution_hash")

    def assert_sealed(self) -> None:
        if not self.execution_hash or self.execution_hash != self.content_hash():
            raise ValueError("development execution is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DevelopmentExecutionV42":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"execution_hash"}),
            execution_hash=draft.content_hash(),
        )


class DevelopmentEvaluationV42(StrictModel):
    schema_version: Literal["4.2"] = "4.2"
    candidate_hash: Sha256
    execution_hash: Sha256
    evaluator_epoch: Identifier
    gates: dict[Identifier, bool]
    metrics: dict[Identifier, Annotated[float, Field(allow_inf_nan=False)]] = Field(
        default_factory=dict
    )
    utility: Annotated[float, Field(allow_inf_nan=False)]
    diagnostic_codes: list[Identifier] = Field(default_factory=list)
    disposition: DevelopmentDispositionV42
    private_data_accessed: Literal[False] = False
    evaluation_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_evaluation(self) -> "DevelopmentEvaluationV42":
        _assert_sorted_unique(self.diagnostic_codes, "diagnostic_codes")
        passed = all(self.gates.values())
        if passed != (self.disposition == "advance"):
            raise ValueError("development disposition differs from its gates")
        if self.evaluation_hash and self.evaluation_hash != self.content_hash():
            raise ValueError("evaluation_hash does not match development evaluation")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evaluation_hash")

    def assert_sealed(self) -> None:
        if not self.evaluation_hash or self.evaluation_hash != self.content_hash():
            raise ValueError("development evaluation is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DevelopmentEvaluationV42":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evaluation_hash"}),
            evaluation_hash=draft.content_hash(),
        )


class OpenFailureSignatureV42(StrictModel):
    schema_version: Literal["4.2"] = "4.2"
    candidate_hash: Sha256
    stage: FailureStageV42
    source_assessment_hash: Sha256
    failed_gates: Annotated[list[Identifier], Field(min_length=1, max_length=64)]
    diagnostic_codes: Annotated[list[Identifier], Field(min_length=1, max_length=64)]
    sanitized_summary: Annotated[str, Field(min_length=10, max_length=3000)]
    private_data_exposed: Literal[False] = False
    failure_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_failure(self) -> "OpenFailureSignatureV42":
        _assert_sorted_unique(self.failed_gates, "failed_gates")
        _assert_sorted_unique(self.diagnostic_codes, "diagnostic_codes")
        if self.failure_hash and self.failure_hash != self.content_hash():
            raise ValueError("failure_hash does not match failure signature")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "failure_hash")

    def assert_sealed(self) -> None:
        if not self.failure_hash or self.failure_hash != self.content_hash():
            raise ValueError("open failure signature is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "OpenFailureSignatureV42":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"failure_hash"}),
            failure_hash=draft.content_hash(),
        )


class HybridEvolutionOperatorV42(StrictModel):
    schema_version: Literal["4.2"] = "4.2"
    operator_id: Identifier
    channel: OperatorChannelV42
    kind: OperatorKindV42
    proposed_by: Literal["model", "harness"]
    source_candidate_hash: Sha256
    source_evaluation_hash: Sha256
    failure_signature_hash: Sha256
    target_family: Identifier
    transformation_summary: Annotated[str, Field(min_length=10, max_length=3000)]
    rationale: Annotated[str, Field(min_length=10, max_length=3000)]
    operator_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_operator(self) -> "HybridEvolutionOperatorV42":
        expected_actor = "harness" if self.channel == "prescribed" else "model"
        if self.proposed_by != expected_actor:
            raise ValueError("operator channel differs from proposing actor")
        if self.operator_hash and self.operator_hash != self.content_hash():
            raise ValueError("operator_hash does not match hybrid evolution operator")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "operator_hash")

    def assert_sealed(self) -> None:
        if not self.operator_hash or self.operator_hash != self.content_hash():
            raise ValueError("hybrid evolution operator is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "HybridEvolutionOperatorV42":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"operator_hash"}),
            operator_hash=draft.content_hash(),
        )


class GenerationCallEvidenceV42(StrictModel):
    schema_version: Literal["4.2"] = "4.2"
    generator_id: Identifier
    transport: Literal["fixture", "codex_cli"]
    request_hash: Sha256
    response_hash: Sha256
    request_payload: dict[str, object]
    response_payload: dict[str, object]
    private_evidence_exposed: Literal[False] = False
    authority_fields_exposed: Literal[False] = False
    tool_events_observed: Literal[False] = False
    served_model_attested: Literal[False] = False
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> "GenerationCallEvidenceV42":
        if self.request_hash != sha256_value(self.request_payload):
            raise ValueError("generation request hash differs from its payload")
        if self.response_hash != sha256_value(self.response_payload):
            raise ValueError("generation response hash differs from its payload")
        if self.response_payload.get("request_hash") != self.request_hash:
            raise ValueError("generation response is bound to another request")
        for field in (
            "private_evidence_exposed",
            "authority_fields_exposed",
            "tools_permitted",
        ):
            if self.request_payload.get(field) is not False:
                raise ValueError(f"generation request does not freeze {field}=false")
        if self.evidence_hash and self.evidence_hash != self.content_hash():
            raise ValueError("evidence_hash does not match generation call evidence")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evidence_hash")

    def assert_sealed(self) -> None:
        if not self.evidence_hash or self.evidence_hash != self.content_hash():
            raise ValueError("generation call evidence is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "GenerationCallEvidenceV42":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evidence_hash"}),
            evidence_hash=draft.content_hash(),
        )


class OpenEvolutionProposalV42(StrictModel):
    operator: HybridEvolutionOperatorV42
    candidate: OpenModelCandidateV42
    generation_evidence: GenerationCallEvidenceV42 | None = None
    priority: Annotated[float, Field(allow_inf_nan=False)] = 0.0

    @model_validator(mode="after")
    def validate_proposal(self) -> "OpenEvolutionProposalV42":
        self.operator.assert_sealed()
        self.candidate.assert_sealed()
        if self.candidate.source != self.operator.channel:
            raise ValueError("candidate source differs from operator channel")
        if self.operator.target_family != self.candidate.family:
            raise ValueError("operator target family differs from candidate family")
        if self.operator.source_candidate_hash not in self.candidate.parent_candidate_hashes:
            raise ValueError("candidate does not name the operator parent")
        if self.operator.operator_hash not in self.candidate.operator_hashes:
            raise ValueError("candidate does not name its evolution operator")
        if self.operator.channel == "generated":
            if self.generation_evidence is None:
                raise ValueError("generated proposal requires generation call evidence")
            self.generation_evidence.assert_sealed()
        elif self.generation_evidence is not None:
            raise ValueError("prescribed proposal cannot contain generation evidence")
        return self


class InitialFrontierV42(StrictModel):
    schema_version: Literal["4.2"] = "4.2"
    campaign_spec_hash: Sha256
    candidates: Annotated[list[OpenModelCandidateV42], Field(min_length=1, max_length=32)]
    frontier_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_frontier(self) -> "InitialFrontierV42":
        hashes = [item.candidate_hash for item in self.candidates]
        if hashes != sorted(set(hashes)):
            raise ValueError("initial candidates must be sorted and unique")
        if any(
            item.generation != 0
            or item.source != "seed"
            or item.parent_candidate_hashes
            or item.operator_hashes
            for item in self.candidates
        ):
            raise ValueError("initial frontier contains a non-seed candidate")
        if self.frontier_hash and self.frontier_hash != self.content_hash():
            raise ValueError("frontier_hash does not match initial frontier")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "frontier_hash")

    def assert_sealed(self) -> None:
        if not self.frontier_hash or self.frontier_hash != self.content_hash():
            raise ValueError("initial frontier is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "InitialFrontierV42":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"frontier_hash"}),
            frontier_hash=draft.content_hash(),
        )


class HybridEvolutionBatchV42(StrictModel):
    schema_version: Literal["4.2"] = "4.2"
    campaign_spec_hash: Sha256
    source_candidate_hash: Sha256
    source_evaluation_hash: Sha256
    failure_signature_hash: Sha256
    next_generation: Annotated[int, Field(ge=1, le=32)]
    proposals: Annotated[list[OpenEvolutionProposalV42], Field(max_length=64)]
    batch_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_batch(self) -> "HybridEvolutionBatchV42":
        keys = [
            (
                item.operator.channel,
                -item.priority,
                item.candidate.candidate_hash,
            )
            for item in self.proposals
        ]
        if keys != sorted(set(keys)):
            raise ValueError("evolution proposals must be deterministically ordered")
        for item in self.proposals:
            if (
                item.operator.source_candidate_hash != self.source_candidate_hash
                or item.operator.source_evaluation_hash != self.source_evaluation_hash
                or item.operator.failure_signature_hash != self.failure_signature_hash
                or item.candidate.generation != self.next_generation
            ):
                raise ValueError("evolution batch contains a proposal for another failure")
        if self.batch_hash and self.batch_hash != self.content_hash():
            raise ValueError("batch_hash does not match hybrid evolution batch")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "batch_hash")

    def assert_sealed(self) -> None:
        if not self.batch_hash or self.batch_hash != self.content_hash():
            raise ValueError("hybrid evolution batch is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "HybridEvolutionBatchV42":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"batch_hash"}),
            batch_hash=draft.content_hash(),
        )


class CampaignCheckpointV42(StrictModel):
    schema_version: Literal["4.2"] = "4.2"
    campaign_spec_hash: Sha256
    checkpoint_index: Annotated[int, Field(ge=1)]
    phase: CheckpointPhaseV42
    subject_hash: Sha256
    base_snapshot_hash: Sha256
    completed_candidate_hashes: list[Sha256]
    pending_node_hashes: list[Sha256]
    replay_safe: Literal[True] = True
    checkpoint_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_checkpoint(self) -> "CampaignCheckpointV42":
        _assert_sorted_unique(
            self.completed_candidate_hashes, "completed_candidate_hashes"
        )
        _assert_sorted_unique(self.pending_node_hashes, "pending_node_hashes")
        if self.checkpoint_hash and self.checkpoint_hash != self.content_hash():
            raise ValueError("checkpoint_hash does not match campaign checkpoint")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "checkpoint_hash")

    def assert_sealed(self) -> None:
        if not self.checkpoint_hash or self.checkpoint_hash != self.content_hash():
            raise ValueError("campaign checkpoint is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "CampaignCheckpointV42":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"checkpoint_hash"}),
            checkpoint_hash=draft.content_hash(),
        )


class RuntimeIncidentV42(StrictModel):
    schema_version: Literal["4.2"] = "4.2"
    incident_id: Identifier
    campaign_spec_hash: Sha256
    recovery_index: Annotated[int, Field(ge=1, le=32)]
    detected_snapshot_hash: Sha256
    last_checkpoint_hash: Sha256 | None = None
    pending_node_hashes: list[Sha256]
    reason: Literal["incomplete_run_detected"] = "incomplete_run_detected"
    safe_to_retry: Literal[True] = True
    incident_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_incident(self) -> "RuntimeIncidentV42":
        _assert_sorted_unique(self.pending_node_hashes, "pending_node_hashes")
        if self.incident_hash and self.incident_hash != self.content_hash():
            raise ValueError("incident_hash does not match runtime incident")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "incident_hash")

    def assert_sealed(self) -> None:
        if not self.incident_hash or self.incident_hash != self.content_hash():
            raise ValueError("runtime incident is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "RuntimeIncidentV42":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"incident_hash"}),
            incident_hash=draft.content_hash(),
        )


class RecoveryPatchV42(StrictModel):
    schema_version: Literal["4.2"] = "4.2"
    patch_id: Identifier
    campaign_spec_hash: Sha256
    incident_hash: Sha256
    strategy: Literal["reconcile_from_committed_graph"] = (
        "reconcile_from_committed_graph"
    )
    idempotency_scope: Literal["spec_and_candidate"] = "spec_and_candidate"
    invalidated_artifact_hashes: list[Sha256] = Field(default_factory=list)
    prevention_controls: Annotated[
        list[Annotated[str, Field(min_length=3, max_length=500)]],
        Field(min_length=3, max_length=16),
    ]
    external_action_retried: Literal[False] = False
    patch_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_patch(self) -> "RecoveryPatchV42":
        _assert_sorted_unique(
            self.invalidated_artifact_hashes, "invalidated_artifact_hashes"
        )
        if self.patch_hash and self.patch_hash != self.content_hash():
            raise ValueError("patch_hash does not match recovery patch")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "patch_hash")

    def assert_sealed(self) -> None:
        if not self.patch_hash or self.patch_hash != self.content_hash():
            raise ValueError("recovery patch is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "RecoveryPatchV42":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"patch_hash"}),
            patch_hash=draft.content_hash(),
        )


class DevelopmentChampionReceiptV42(StrictModel):
    schema_version: Literal["4.2"] = "4.2"
    campaign_spec_hash: Sha256
    candidate_hash: Sha256
    admission_receipt_hash: Sha256
    evaluation_hash: Sha256
    considered_evaluation_hashes: Annotated[list[Sha256], Field(min_length=1)]
    status: Literal["development_champion_unqualified"] = (
        "development_champion_unqualified"
    )
    qualification_granted: Literal[False] = False
    private_confirmation_consumed: Literal[False] = False
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "DevelopmentChampionReceiptV42":
        _assert_sorted_unique(
            self.considered_evaluation_hashes, "considered_evaluation_hashes"
        )
        if self.evaluation_hash not in self.considered_evaluation_hashes:
            raise ValueError("champion evaluation was not considered")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match development champion")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("development champion receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DevelopmentChampionReceiptV42":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


class OpenEvolutionReportV42(StrictModel):
    schema_version: Literal["4.2"] = "4.2"
    campaign_id: Identifier
    campaign_spec_hash: Sha256
    grammar_hash: Sha256
    terminal_status: TerminalStatusV42
    candidate_hashes: list[Sha256]
    validation_hashes: list[Sha256]
    admission_receipt_hashes: list[Sha256]
    execution_hashes: list[Sha256]
    evaluation_hashes: list[Sha256]
    failure_hashes: list[Sha256]
    operator_hashes: list[Sha256]
    evolution_batch_hashes: list[Sha256]
    generation_call_evidence_hashes: list[Sha256]
    checkpoint_hashes: list[Sha256]
    incident_hashes: list[Sha256]
    recovery_patch_hashes: list[Sha256]
    prescribed_operator_count: Annotated[int, Field(ge=0)]
    generated_operator_count: Annotated[int, Field(ge=0)]
    champion_receipt_hash: Sha256 | None = None
    champion_candidate_hash: Sha256 | None = None
    graph_snapshot_hash: Sha256
    graph_replay_verified: bool
    recovered_from_incomplete_run: bool
    private_confirmation_consumed: Literal[False] = False
    qualification_granted: Literal[False] = False
    real_world_execution_permitted: Literal[False] = False
    created_at: datetime
    report_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "OpenEvolutionReportV42":
        _assert_timezone(self.created_at, "created_at")
        for field_name in (
            "candidate_hashes",
            "validation_hashes",
            "admission_receipt_hashes",
            "execution_hashes",
            "evaluation_hashes",
            "failure_hashes",
            "operator_hashes",
            "evolution_batch_hashes",
            "generation_call_evidence_hashes",
            "checkpoint_hashes",
            "incident_hashes",
            "recovery_patch_hashes",
        ):
            _assert_sorted_unique(getattr(self, field_name), field_name)
        has_champion = self.terminal_status == "development_champion_frozen"
        if has_champion != bool(
            self.champion_receipt_hash and self.champion_candidate_hash
        ):
            raise ValueError("terminal status and champion fields differ")
        if not self.graph_replay_verified:
            raise ValueError("open-evolution report requires graph replay")
        if self.recovered_from_incomplete_run != bool(self.incident_hashes):
            raise ValueError("recovery status differs from incident evidence")
        if self.report_hash and self.report_hash != self.content_hash():
            raise ValueError("report_hash does not match open-evolution report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "report_hash")

    def assert_sealed(self) -> None:
        if not self.report_hash or self.report_hash != self.content_hash():
            raise ValueError("open-evolution report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "OpenEvolutionReportV42":
        data.setdefault("created_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"report_hash"}),
            report_hash=draft.content_hash(),
        )


class OpenEvolutionAdapterV42(Protocol):
    adapter_id: str
    adapter_contract_hash: str

    def initial_candidates(
        self,
        spec: OpenEvolutionCampaignSpecV42,
        grammar: OpenModelGrammarV42,
    ) -> list[OpenModelCandidateV42]: ...

    def supports_candidate(self, candidate: OpenModelCandidateV42) -> bool: ...

    def execute(
        self,
        spec: OpenEvolutionCampaignSpecV42,
        candidate: OpenModelCandidateV42,
        attempt: ExecutionAttemptV42,
    ) -> DevelopmentExecutionV42: ...

    def evaluate(
        self,
        spec: OpenEvolutionCampaignSpecV42,
        candidate: OpenModelCandidateV42,
        execution: DevelopmentExecutionV42,
    ) -> DevelopmentEvaluationV42: ...

    def prescribed_evolve(
        self,
        spec: OpenEvolutionCampaignSpecV42,
        grammar: OpenModelGrammarV42,
        candidate: OpenModelCandidateV42,
        evaluation: DevelopmentEvaluationV42,
        failure: OpenFailureSignatureV42,
        next_generation: int,
    ) -> list[OpenEvolutionProposalV42]: ...

    def generated_evolve(
        self,
        spec: OpenEvolutionCampaignSpecV42,
        grammar: OpenModelGrammarV42,
        candidate: OpenModelCandidateV42,
        evaluation: DevelopmentEvaluationV42,
        failure: OpenFailureSignatureV42,
        next_generation: int,
    ) -> list[OpenEvolutionProposalV42]: ...


@dataclass(frozen=True)
class OpenEvolutionOutcomeV42:
    graph: GraphLoopStoreV40
    spec: OpenEvolutionCampaignSpecV42
    grammar: OpenModelGrammarV42
    initial_frontier: InitialFrontierV42
    candidates: list[OpenModelCandidateV42]
    validations: list[ModelSpaceValidationV42]
    admissions: list[ModelAdmissionReceiptV42]
    attempts: list[ExecutionAttemptV42]
    executions: list[DevelopmentExecutionV42]
    evaluations: list[DevelopmentEvaluationV42]
    failures: list[OpenFailureSignatureV42]
    operators: list[HybridEvolutionOperatorV42]
    evolution_batches: list[HybridEvolutionBatchV42]
    generation_calls: list[GenerationCallEvidenceV42]
    checkpoints: list[CampaignCheckpointV42]
    incidents: list[RuntimeIncidentV42]
    recovery_patches: list[RecoveryPatchV42]
    champion: DevelopmentChampionReceiptV42 | None
    report: OpenEvolutionReportV42


def _artifact_refs(store: RunStore, kind: str) -> list[ArtifactRef]:
    refs: list[ArtifactRef] = []
    seen: set[tuple[str, str]] = set()
    for line in store.event_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event["event_type"] != "artifact_committed":
            continue
        ref = ArtifactRef.model_validate(event["payload"])
        key = (ref.kind, ref.sha256)
        if ref.kind == kind and key not in seen:
            refs.append(ref)
            seen.add(key)
    return refs


def _load_many(store: RunStore, kind: str, model: type[StrictModel]) -> list:
    return [
        model.model_validate(store.load_artifact(ref))
        for ref in _artifact_refs(store, kind)
    ]


def _load_one(store: RunStore, kind: str, model: type[StrictModel]) -> StrictModel:
    items = _load_many(store, kind, model)
    if len(items) != 1:
        raise RuntimeError(f"open-evolution run needs exactly one {kind}")
    return items[0]


def _put_once(graph: GraphLoopStoreV40, kind: str, payload: object) -> ArtifactRef:
    expected = json.loads(canonical_json(payload))
    for ref in _artifact_refs(graph.store, kind):
        if graph.store.load_artifact(ref) == expected:
            return ref
    return graph.put_output(kind, payload)


def _node_id(prefix: str, content_hash: str) -> str:
    return f"{prefix}.{content_hash[:16]}"


def _node(
    spec: OpenEvolutionCampaignSpecV42,
    *,
    node_id: str,
    node_kind: str,
    executor: str,
    created_by: str,
    artifact_hash: str,
    purpose: str,
) -> GraphNodeV40:
    return GraphNodeV40.seal(
        node_id=node_id,
        layer="modeling",
        node_kind=node_kind,
        executor=executor,
        created_by=created_by,
        artifact_hash=artifact_hash,
        purpose=purpose,
        created_at=spec.created_at,
    )


def _ensure_node(graph: GraphLoopStoreV40, node: GraphNodeV40) -> GraphNodeV40:
    state = graph.project_state()
    by_id = next((item for item in state.nodes if item.node_id == node.node_id), None)
    by_hash = next(
        (item for item in state.nodes if item.node_hash == node.node_hash), None
    )
    existing = by_id or by_hash
    if existing is not None:
        if existing != node:
            raise RuntimeError("graph node identity collides with different content")
        return existing
    graph.add_node(node)
    return node


def _edge(
    spec: OpenEvolutionCampaignSpecV42,
    *,
    edge_id: str,
    source: GraphNodeV40,
    target: GraphNodeV40,
    relation: str,
    rationale: str,
) -> GraphEdgeV40:
    return GraphEdgeV40.seal(
        edge_id=edge_id,
        layer="modeling",
        source_node_hash=source.node_hash,
        target_node_hash=target.node_hash,
        relation=relation,
        rationale=rationale,
        created_at=spec.created_at,
    )


def _ensure_edge(graph: GraphLoopStoreV40, edge: GraphEdgeV40) -> GraphEdgeV40:
    state = graph.project_state()
    by_id = next((item for item in state.edges if item.edge_id == edge.edge_id), None)
    by_hash = next(
        (item for item in state.edges if item.edge_hash == edge.edge_hash), None
    )
    existing = by_id or by_hash
    if existing is not None:
        if existing != edge:
            raise RuntimeError("graph edge identity collides with different content")
        return existing
    graph.add_edge(edge)
    return edge


def _ensure_outcome(
    graph: GraphLoopStoreV40,
    node: GraphNodeV40,
    ref: ArtifactRef,
    *,
    actor: str,
    status: str,
    summary: str,
) -> None:
    state = graph.project_state()
    existing = [item for item in state.outcomes if item.node_hash == node.node_hash]
    if existing:
        if len(existing) != 1:
            raise RuntimeError("graph node contains duplicate outcomes")
        outcome = existing[0]
        if (
            outcome.actor != actor
            or outcome.status != status
            or ref.sha256 not in {item.sha256 for item in outcome.output_artifacts}
        ):
            raise RuntimeError("existing graph outcome differs from recovery request")
        return
    graph.record_outcome(
        node.node_hash,
        actor=actor,
        status=status,
        output_artifacts=[ref],
        summary=summary,
        outcome_id=f"{node.node_id}.outcome",
        started_at=graph.contract.created_at,
        finished_at=graph.contract.created_at,
    )


def _node_by_artifact(
    graph: GraphLoopStoreV40,
    artifact_hash: str,
    *,
    node_kind: str | None = None,
) -> GraphNodeV40:
    matches = [
        node
        for node in graph.project_state().nodes
        if node.artifact_hash == artifact_hash
        and (node_kind is None or node.node_kind == node_kind)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one graph node for {artifact_hash}, found {len(matches)}"
        )
    return matches[0]


def _has_cycle(candidate: OpenModelCandidateV42) -> bool:
    producers: dict[str, str] = {}
    duplicate_output = False
    for application in candidate.applications:
        if application.output in producers:
            duplicate_output = True
        producers[application.output] = application.application_id
    if duplicate_output:
        return True
    adjacency: dict[str, set[str]] = {
        application.application_id: set() for application in candidate.applications
    }
    indegree = {application.application_id: 0 for application in candidate.applications}
    for application in candidate.applications:
        for symbol in application.inputs:
            producer = producers.get(symbol)
            if producer and application.application_id not in adjacency[producer]:
                adjacency[producer].add(application.application_id)
                indegree[application.application_id] += 1
    ready = sorted(key for key, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        current = ready.pop(0)
        visited += 1
        for target in sorted(adjacency[current]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    return visited != len(candidate.applications)


def validate_model_space_candidate_v42(
    spec: OpenEvolutionCampaignSpecV42,
    grammar: OpenModelGrammarV42,
    candidate: OpenModelCandidateV42,
    adapter: OpenEvolutionAdapterV42,
) -> ModelSpaceValidationV42:
    """Code-owned quarantine verifier for generated and prescribed proposals."""

    spec.assert_sealed()
    grammar.assert_sealed()
    candidate.assert_sealed()
    primitive_rules = {item.primitive_id: item for item in grammar.primitives}
    symbol_ids = [item.symbol_id for item in candidate.symbols]
    symbol_map = {item.symbol_id: item for item in candidate.symbols}
    application_ids = [item.application_id for item in candidate.applications]
    used_primitives: dict[str, int] = {}
    for application in candidate.applications:
        used_primitives[application.primitive_id] = (
            used_primitives.get(application.primitive_id, 0) + 1
        )

    primitive_allowed = all(
        application.primitive_id in primitive_rules
        for application in candidate.applications
    ) and all(
        used_primitives[primitive_id]
        <= primitive_rules[primitive_id].max_uses_per_candidate
        for primitive_id in used_primitives
        if primitive_id in primitive_rules
    )
    arity_valid = primitive_allowed and all(
        len(application.inputs)
        == len(primitive_rules[application.primitive_id].input_units)
        for application in candidate.applications
    )
    symbols_resolved = (
        len(symbol_ids) == len(set(symbol_ids))
        and len(application_ids) == len(set(application_ids))
        and all(
            application.output in symbol_map
            and all(symbol in symbol_map for symbol in application.inputs)
            for application in candidate.applications
        )
    )
    units_valid = primitive_allowed and arity_valid and symbols_resolved
    if units_valid:
        for application in candidate.applications:
            rule = primitive_rules[application.primitive_id]
            actual_inputs = [symbol_map[item].unit for item in application.inputs]
            if (
                actual_inputs != rule.input_units
                or symbol_map[application.output].unit != rule.output_unit
            ):
                units_valid = False
                break
    expected_actor = "harness" if candidate.source == "prescribed" else "model"
    if candidate.generation == 0:
        lineage_valid = (
            candidate.source == "seed"
            and candidate.family in grammar.seed_families
            and not candidate.parent_candidate_hashes
            and not candidate.operator_hashes
            and candidate.proposed_by == "model"
        )
    else:
        lineage_valid = (
            candidate.source in {"prescribed", "generated"}
            and bool(candidate.parent_candidate_hashes)
            and bool(candidate.operator_hashes)
            and candidate.parent_candidate_hashes
            == sorted(set(candidate.parent_candidate_hashes))
            and candidate.operator_hashes == sorted(set(candidate.operator_hashes))
            and candidate.proposed_by == expected_actor
        )
    serialized = canonical_json(candidate.model_dump(mode="json")).lower()
    no_forbidden_token = not any(
        token.lower() in serialized for token in grammar.forbidden_tokens
    )
    checks = {
        "acyclic": symbols_resolved and not _has_cycle(candidate),
        "arity_valid": arity_valid,
        "complexity_budget": (
            len(candidate.symbols) <= grammar.max_symbols
            and len(candidate.applications) <= grammar.max_applications
        ),
        "executable_adapter_available": (
            candidate.executable_adapter_id in grammar.executable_adapter_ids
            and candidate.executable_adapter_id == adapter.adapter_id
            and grammar.executable_adapter_hashes[candidate.executable_adapter_id]
            == adapter.adapter_contract_hash
            and adapter.supports_candidate(candidate)
        ),
        "lineage_valid": lineage_valid,
        "no_forbidden_token": no_forbidden_token,
        "primitive_allowed": primitive_allowed,
        "private_data_absent": not candidate.private_data_references,
        "symbols_resolved": symbols_resolved,
        "units_valid": units_valid,
    }
    diagnostics = sorted(
        f"failed_{name}" for name, passed in checks.items() if not passed
    )
    return ModelSpaceValidationV42.seal(
        candidate_hash=candidate.candidate_hash,
        grammar_hash=grammar.grammar_hash,
        evaluator_epoch=spec.evaluator_epoch,
        checks=checks,
        admitted=all(checks.values()),
        diagnostic_codes=diagnostics,
    )


def _ensure_checkpoint(
    graph: GraphLoopStoreV40,
    spec: OpenEvolutionCampaignSpecV42,
    *,
    phase: CheckpointPhaseV42,
    subject_hash: str,
    source_node: GraphNodeV40,
    source_relation: Literal["requires_success", "requires_terminal"],
) -> CampaignCheckpointV42:
    checkpoints = _load_many(
        graph.store, "campaign_checkpoint_v42", CampaignCheckpointV42
    )
    matching = [
        item
        for item in checkpoints
        if item.phase == phase and item.subject_hash == subject_hash
    ]
    if matching:
        if len(matching) != 1:
            raise RuntimeError("duplicate campaign checkpoint")
        return matching[0]
    state = graph.project_state()
    completed = sorted(
        {
            item.candidate_hash
            for item in _load_many(
                graph.store,
                "development_evaluation_v42",
                DevelopmentEvaluationV42,
            )
        }
    )
    checkpoint = CampaignCheckpointV42.seal(
        campaign_spec_hash=spec.spec_hash,
        checkpoint_index=len(checkpoints) + 1,
        phase=phase,
        subject_hash=subject_hash,
        base_snapshot_hash=state.snapshot.snapshot_hash,
        completed_candidate_hashes=completed,
        pending_node_hashes=sorted(state.snapshot.frontier_node_hashes),
    )
    ref = _put_once(graph, "campaign_checkpoint_v42", checkpoint)
    node = _ensure_node(
        graph,
        _node(
            spec,
            node_id=_node_id(f"checkpoint.{phase}", checkpoint.checkpoint_hash),
            node_kind="checkpoint",
            executor="harness",
            created_by="harness",
            artifact_hash=checkpoint.checkpoint_hash,
            purpose=f"freeze replay-safe campaign checkpoint after {phase}",
        ),
    )
    _ensure_edge(
        graph,
        _edge(
            spec,
            edge_id=_node_id(
                "checkpoint.source",
                sha256_value([source_node.node_hash, checkpoint.checkpoint_hash]),
            ),
            source=source_node,
            target=node,
            relation=source_relation,
            rationale="checkpoint is causally bound to the completed graph phase",
        ),
    )
    _ensure_outcome(
        graph,
        node,
        ref,
        actor="harness",
        status="succeeded",
        summary=f"replay-safe {phase} checkpoint committed",
    )
    return checkpoint


def _ensure_failure(
    graph: GraphLoopStoreV40,
    spec: OpenEvolutionCampaignSpecV42,
    candidate: OpenModelCandidateV42,
    *,
    stage: FailureStageV42,
    source_hash: str,
    source_node: GraphNodeV40,
    failed_gates: list[str],
    diagnostic_codes: list[str],
) -> OpenFailureSignatureV42:
    failures = _load_many(
        graph.store, "open_failure_signature_v42", OpenFailureSignatureV42
    )
    matching = [
        item
        for item in failures
        if item.candidate_hash == candidate.candidate_hash
        and item.stage == stage
        and item.source_assessment_hash == source_hash
    ]
    if matching:
        if len(matching) != 1:
            raise RuntimeError("duplicate open failure signature")
        return matching[0]
    failure = OpenFailureSignatureV42.seal(
        candidate_hash=candidate.candidate_hash,
        stage=stage,
        source_assessment_hash=source_hash,
        failed_gates=sorted(set(failed_gates)),
        diagnostic_codes=sorted(
            set(diagnostic_codes or [f"failed_{failed_gates[0]}"])
        ),
        sanitized_summary=(
            f"{stage} verifier rejected {candidate.family} on "
            + ", ".join(sorted(set(failed_gates)))
            + "; no private confirmation data was accessed."
        ),
    )
    ref = _put_once(graph, "open_failure_signature_v42", failure)
    node = _ensure_node(
        graph,
        _node(
            spec,
            node_id=_node_id(f"{stage}.failure", failure.failure_hash),
            node_kind="failure",
            executor="verifier",
            created_by="harness",
            artifact_hash=failure.failure_hash,
            purpose=f"publish sanitized {stage} failure evidence",
        ),
    )
    _ensure_edge(
        graph,
        _edge(
            spec,
            edge_id=_node_id(
                "assessment.failure",
                sha256_value([source_node.node_hash, failure.failure_hash]),
            ),
            source=source_node,
            target=node,
            relation="learned_from_failure",
            rationale="failure signature is released only after verifier rejection",
        ),
    )
    _ensure_outcome(
        graph,
        node,
        ref,
        actor="verifier",
        status="succeeded",
        summary=f"sanitized {stage} failure signature committed",
    )
    return failure


def _ensure_initial_frontier(
    graph: GraphLoopStoreV40,
    spec: OpenEvolutionCampaignSpecV42,
    grammar: OpenModelGrammarV42,
    adapter: OpenEvolutionAdapterV42,
) -> tuple[InitialFrontierV42, GraphNodeV40]:
    existing = _load_many(
        graph.store, "initial_frontier_v42", InitialFrontierV42
    )
    if existing:
        if len(existing) != 1:
            raise RuntimeError("duplicate initial frontier")
        frontier = existing[0]
        frontier.assert_sealed()
    else:
        candidates = sorted(
            adapter.initial_candidates(spec, grammar),
            key=lambda item: item.candidate_hash,
        )
        if not candidates:
            raise ValueError("open-evolution adapter returned no initial candidates")
        if len(candidates) > spec.max_candidates:
            raise ValueError("initial frontier exceeds candidate budget")
        for candidate in candidates:
            candidate.assert_sealed()
        frontier = InitialFrontierV42.seal(
            campaign_spec_hash=spec.spec_hash,
            candidates=candidates,
        )
    ref = _put_once(graph, "initial_frontier_v42", frontier)
    node = _ensure_node(
        graph,
        _node(
            spec,
            node_id=_node_id("initial.frontier", frontier.frontier_hash),
            node_kind="workflow_plan",
            executor="harness",
            created_by="harness",
            artifact_hash=frontier.frontier_hash,
            purpose="freeze the generation-zero proposal frontier",
        ),
    )
    _ensure_outcome(
        graph,
        node,
        ref,
        actor="harness",
        status="succeeded",
        summary="initial model-space frontier frozen",
    )
    return frontier, node


def _ensure_candidate_proposal(
    graph: GraphLoopStoreV40,
    spec: OpenEvolutionCampaignSpecV42,
    candidate: OpenModelCandidateV42,
    *,
    batch_node: GraphNodeV40,
) -> GraphNodeV40:
    candidate.assert_sealed()
    ref = _put_once(graph, "open_model_candidate_v42", candidate)
    node = _ensure_node(
        graph,
        _node(
            spec,
            node_id=_node_id(
                f"g{candidate.generation}.model.proposal", candidate.candidate_hash
            ),
            node_kind="model_proposal",
            executor=candidate.proposed_by,
            created_by=candidate.proposed_by,
            artifact_hash=candidate.candidate_hash,
            purpose=(
                f"quarantine {candidate.source} model proposal {candidate.family}"
            ),
        ),
    )
    _ensure_edge(
        graph,
        _edge(
            spec,
            edge_id=_node_id(
                "frontier.proposal",
                sha256_value([batch_node.node_hash, candidate.candidate_hash]),
            ),
            source=batch_node,
            target=node,
            relation="requires_success",
            rationale="proposal belongs to a frozen frontier or evolution batch",
        ),
    )
    for parent_hash in candidate.parent_candidate_hashes:
        parent = _node_by_artifact(
            graph, parent_hash, node_kind="model_proposal"
        )
        _ensure_edge(
            graph,
            _edge(
                spec,
                edge_id=_node_id(
                    "proposal.derived",
                    sha256_value([parent_hash, candidate.candidate_hash]),
                ),
                source=parent,
                target=node,
                relation="derived_from",
                rationale="evolved proposal retains immutable model lineage",
            ),
        )
    for operator_hash in candidate.operator_hashes:
        operator = _node_by_artifact(
            graph, operator_hash, node_kind="evolution_operator"
        )
        _ensure_edge(
            graph,
            _edge(
                spec,
                edge_id=_node_id(
                    "operator.proposal",
                    sha256_value([operator_hash, candidate.candidate_hash]),
                ),
                source=operator,
                target=node,
                relation="requires_success",
                rationale="proposal requires its recorded evolution operator",
            ),
        )
    _ensure_outcome(
        graph,
        node,
        ref,
        actor=candidate.proposed_by,
        status="succeeded",
        summary=f"quarantined {candidate.source} proposal {candidate.family}",
    )
    return node


def _ensure_space_validation(
    graph: GraphLoopStoreV40,
    spec: OpenEvolutionCampaignSpecV42,
    grammar: OpenModelGrammarV42,
    candidate: OpenModelCandidateV42,
    adapter: OpenEvolutionAdapterV42,
) -> tuple[ModelSpaceValidationV42, GraphNodeV40]:
    validations = _load_many(
        graph.store, "model_space_validation_v42", ModelSpaceValidationV42
    )
    matching = [
        item for item in validations if item.candidate_hash == candidate.candidate_hash
    ]
    if matching:
        if len(matching) != 1:
            raise RuntimeError("candidate has duplicate model-space validations")
        validation = matching[0]
    else:
        validation = validate_model_space_candidate_v42(
            spec, grammar, candidate, adapter
        )
    validation.assert_sealed()
    ref = _put_once(graph, "model_space_validation_v42", validation)
    proposal_node = _node_by_artifact(
        graph, candidate.candidate_hash, node_kind="model_proposal"
    )
    node = _ensure_node(
        graph,
        _node(
            spec,
            node_id=_node_id("model.space.validation", validation.validation_hash),
            node_kind="model_validation",
            executor="verifier",
            created_by="harness",
            artifact_hash=validation.validation_hash,
            purpose=f"verify typed model-space proposal {candidate.family}",
        ),
    )
    _ensure_edge(
        graph,
        _edge(
            spec,
            edge_id=_node_id(
                "proposal.validation",
                sha256_value([candidate.candidate_hash, validation.validation_hash]),
            ),
            source=proposal_node,
            target=node,
            relation="evaluated_by",
            rationale="the quarantined proposal cannot validate itself",
        ),
    )
    _ensure_outcome(
        graph,
        node,
        ref,
        actor="verifier",
        status="succeeded" if validation.admitted else "failed",
        summary=(
            "model-space checks passed"
            if validation.admitted
            else "model-space checks rejected the quarantined proposal"
        ),
    )
    _ensure_checkpoint(
        graph,
        spec,
        phase="model_space_validated",
        subject_hash=candidate.candidate_hash,
        source_node=node,
        source_relation="requires_terminal",
    )
    if not validation.admitted:
        failed = sorted(
            name for name, passed in validation.checks.items() if not passed
        )
        _ensure_failure(
            graph,
            spec,
            candidate,
            stage="model_space",
            source_hash=validation.validation_hash,
            source_node=node,
            failed_gates=failed,
            diagnostic_codes=validation.diagnostic_codes,
        )
    return validation, node


def _ensure_admission(
    graph: GraphLoopStoreV40,
    spec: OpenEvolutionCampaignSpecV42,
    grammar: OpenModelGrammarV42,
    candidate: OpenModelCandidateV42,
    validation: ModelSpaceValidationV42,
) -> tuple[ModelAdmissionReceiptV42, GraphNodeV40]:
    if not validation.admitted:
        raise ValueError("rejected model-space proposal cannot be admitted")
    admissions = _load_many(
        graph.store, "model_admission_receipt_v42", ModelAdmissionReceiptV42
    )
    matching = [
        item for item in admissions if item.candidate_hash == candidate.candidate_hash
    ]
    if matching:
        if len(matching) != 1:
            raise RuntimeError("candidate has duplicate model admissions")
        admission = matching[0]
    else:
        admission = ModelAdmissionReceiptV42.seal(
            campaign_spec_hash=spec.spec_hash,
            grammar_hash=grammar.grammar_hash,
            candidate_hash=candidate.candidate_hash,
            validation_hash=validation.validation_hash,
        )
    admission.assert_sealed()
    ref = _put_once(graph, "model_admission_receipt_v42", admission)
    proposal_node = _node_by_artifact(
        graph, candidate.candidate_hash, node_kind="model_proposal"
    )
    validation_node = _node_by_artifact(
        graph, validation.validation_hash, node_kind="model_validation"
    )
    node = _ensure_node(
        graph,
        _node(
            spec,
            node_id=_node_id("model.admission", admission.receipt_hash),
            node_kind="model_admission",
            executor="verifier",
            created_by="harness",
            artifact_hash=admission.receipt_hash,
            purpose="admit a structurally valid proposal for development only",
        ),
    )
    for source, label in (
        (proposal_node, "proposal"),
        (validation_node, "validation"),
    ):
        _ensure_edge(
            graph,
            _edge(
                spec,
                edge_id=_node_id(
                    f"{label}.admission",
                    sha256_value([source.node_hash, admission.receipt_hash]),
                ),
                source=source,
                target=node,
                relation="requires_success",
                rationale="development admission requires proposal and verifier success",
            ),
        )
    _ensure_outcome(
        graph,
        node,
        ref,
        actor="verifier",
        status="succeeded",
        summary="candidate admitted for development without qualification",
    )
    _ensure_checkpoint(
        graph,
        spec,
        phase="candidate_admitted",
        subject_hash=candidate.candidate_hash,
        source_node=node,
        source_relation="requires_success",
    )
    return admission, node


def _ensure_execution(
    graph: GraphLoopStoreV40,
    spec: OpenEvolutionCampaignSpecV42,
    candidate: OpenModelCandidateV42,
    admission: ModelAdmissionReceiptV42,
    adapter: OpenEvolutionAdapterV42,
) -> tuple[ExecutionAttemptV42, DevelopmentExecutionV42, GraphNodeV40]:
    attempts = _load_many(
        graph.store, "execution_attempt_v42", ExecutionAttemptV42
    )
    matching_attempts = [
        item for item in attempts if item.candidate_hash == candidate.candidate_hash
    ]
    expected_key = sha256_value(
        {
            "campaign_spec_hash": spec.spec_hash,
            "candidate_hash": candidate.candidate_hash,
            "operation": "development_execute_v42",
        }
    )
    if matching_attempts:
        if len(matching_attempts) != 1:
            raise RuntimeError("candidate has duplicate execution attempts")
        attempt = matching_attempts[0]
    else:
        attempt = ExecutionAttemptV42.seal(
            campaign_spec_hash=spec.spec_hash,
            candidate_hash=candidate.candidate_hash,
            development_data_hash=spec.development_data_hash,
            executable_adapter_id=candidate.executable_adapter_id,
            executable_adapter_hash=adapter.adapter_contract_hash,
            idempotency_key=expected_key,
        )
    attempt.assert_sealed()
    if attempt.idempotency_key != expected_key:
        raise RuntimeError("execution attempt has an incorrect idempotency key")
    attempt_ref = _put_once(graph, "execution_attempt_v42", attempt)
    admission_node = _node_by_artifact(
        graph, admission.receipt_hash, node_kind="model_admission"
    )
    node = _ensure_node(
        graph,
        _node(
            spec,
            node_id=_node_id("development.attempt", attempt.attempt_hash),
            node_kind="execution",
            executor="harness",
            created_by="harness",
            artifact_hash=attempt.attempt_hash,
            purpose=f"execute {candidate.family} under a stable idempotency key",
        ),
    )
    _ensure_edge(
        graph,
        _edge(
            spec,
            edge_id=_node_id(
                "admission.attempt",
                sha256_value([admission.receipt_hash, attempt.attempt_hash]),
            ),
            source=admission_node,
            target=node,
            relation="requires_success",
            rationale="only an admitted development candidate may execute",
        ),
    )

    executions = _load_many(
        graph.store, "development_execution_v42", DevelopmentExecutionV42
    )
    matching_executions = [
        item for item in executions if item.candidate_hash == candidate.candidate_hash
    ]
    if matching_executions:
        if len(matching_executions) != 1:
            raise RuntimeError("candidate has duplicate development executions")
        execution = matching_executions[0]
    else:
        # The attempt node is deliberately durable before this call.  If the
        # adapter raises, the next run observes the pending node and retries
        # under exactly the same idempotency key.
        execution = adapter.execute(spec, candidate, attempt)
    execution.assert_sealed()
    if (
        execution.candidate_hash != candidate.candidate_hash
        or execution.attempt_hash != attempt.attempt_hash
        or execution.idempotency_key != attempt.idempotency_key
        or execution.development_data_hash != spec.development_data_hash
        or attempt.executable_adapter_hash != adapter.adapter_contract_hash
    ):
        raise ValueError("development execution differs from its frozen attempt")
    execution_ref = _put_once(graph, "development_execution_v42", execution)
    _ensure_outcome(
        graph,
        node,
        execution_ref,
        actor="harness",
        status="succeeded",
        summary="idempotent local development execution completed",
    )
    return attempt, execution, node


def _ensure_development_evaluation(
    graph: GraphLoopStoreV40,
    spec: OpenEvolutionCampaignSpecV42,
    candidate: OpenModelCandidateV42,
    execution: DevelopmentExecutionV42,
    execution_node: GraphNodeV40,
    adapter: OpenEvolutionAdapterV42,
) -> tuple[DevelopmentEvaluationV42, GraphNodeV40]:
    evaluations = _load_many(
        graph.store, "development_evaluation_v42", DevelopmentEvaluationV42
    )
    matching = [
        item for item in evaluations if item.candidate_hash == candidate.candidate_hash
    ]
    if matching:
        if len(matching) != 1:
            raise RuntimeError("candidate has duplicate development evaluations")
        evaluation = matching[0]
    else:
        evaluation = adapter.evaluate(spec, candidate, execution)
    evaluation.assert_sealed()
    if (
        evaluation.candidate_hash != candidate.candidate_hash
        or evaluation.execution_hash != execution.execution_hash
        or evaluation.evaluator_epoch != spec.evaluator_epoch
        or sorted(evaluation.gates) != spec.required_development_gates
    ):
        raise ValueError("development evaluation differs from the frozen campaign")
    passed = all(evaluation.gates.values())
    ref = _put_once(graph, "development_evaluation_v42", evaluation)
    proposal_node = _node_by_artifact(
        graph, candidate.candidate_hash, node_kind="model_proposal"
    )
    node = _ensure_node(
        graph,
        _node(
            spec,
            node_id=_node_id("development.evaluate", evaluation.evaluation_hash),
            node_kind="evaluation",
            executor="verifier",
            created_by="harness",
            artifact_hash=evaluation.evaluation_hash,
            purpose=f"independently evaluate admitted candidate {candidate.family}",
        ),
    )
    _ensure_edge(
        graph,
        _edge(
            spec,
            edge_id=_node_id(
                "execution.evaluation",
                sha256_value([execution.execution_hash, evaluation.evaluation_hash]),
            ),
            source=execution_node,
            target=node,
            relation="requires_success",
            rationale="evaluation requires a committed development execution",
        ),
    )
    _ensure_edge(
        graph,
        _edge(
            spec,
            edge_id=_node_id(
                "proposal.evaluator",
                sha256_value([candidate.candidate_hash, evaluation.evaluation_hash]),
            ),
            source=proposal_node,
            target=node,
            relation="evaluated_by",
            rationale="candidate remains bound to an independent verifier",
        ),
    )
    _ensure_outcome(
        graph,
        node,
        ref,
        actor="verifier",
        status="succeeded" if passed else "failed",
        summary=(
            "development gates passed"
            if passed
            else "development gates failed and may trigger hybrid evolution"
        ),
    )
    _ensure_checkpoint(
        graph,
        spec,
        phase="candidate_evaluated",
        subject_hash=candidate.candidate_hash,
        source_node=node,
        source_relation="requires_terminal",
    )
    if not passed:
        failed = sorted(
            name for name, gate_passed in evaluation.gates.items() if not gate_passed
        )
        _ensure_failure(
            graph,
            spec,
            candidate,
            stage="development",
            source_hash=evaluation.evaluation_hash,
            source_node=node,
            failed_gates=failed,
            diagnostic_codes=evaluation.diagnostic_codes,
        )
    return evaluation, node


def _proposal_identity(candidate: OpenModelCandidateV42) -> str:
    return sha256_value(
        {
            "family": candidate.family,
            "symbols": candidate.symbols,
            "applications": candidate.applications,
            "executable_adapter_id": candidate.executable_adapter_id,
        }
    )


def _select_channel_proposals(
    proposals: list[OpenEvolutionProposalV42],
    *,
    channel: OperatorChannelV42,
    quota: int,
    known_identities: set[str],
) -> list[OpenEvolutionProposalV42]:
    if quota <= 0:
        return []
    selected: list[OpenEvolutionProposalV42] = []
    for proposal in sorted(
        proposals,
        key=lambda item: (-item.priority, item.candidate.candidate_hash),
    ):
        if proposal.operator.channel != channel:
            raise ValueError("evolution method returned a proposal on another channel")
        identity = _proposal_identity(proposal.candidate)
        if identity in known_identities:
            continue
        known_identities.add(identity)
        selected.append(proposal)
        if len(selected) == quota:
            break
    return selected


def _ensure_evolution_batch(
    graph: GraphLoopStoreV40,
    spec: OpenEvolutionCampaignSpecV42,
    grammar: OpenModelGrammarV42,
    candidate: OpenModelCandidateV42,
    evaluation: DevelopmentEvaluationV42,
    failure: OpenFailureSignatureV42,
    adapter: OpenEvolutionAdapterV42,
) -> tuple[HybridEvolutionBatchV42, GraphNodeV40]:
    batches = _load_many(
        graph.store, "hybrid_evolution_batch_v42", HybridEvolutionBatchV42
    )
    matching = [
        item
        for item in batches
        if item.failure_signature_hash == failure.failure_hash
    ]
    if matching:
        if len(matching) != 1:
            raise RuntimeError("failure has duplicate evolution batches")
        batch = matching[0]
    else:
        existing_candidates = _load_many(
            graph.store, "open_model_candidate_v42", OpenModelCandidateV42
        )
        remaining = max(spec.max_candidates - len(existing_candidates), 0)
        known = {_proposal_identity(item) for item in existing_candidates}
        next_generation = candidate.generation + 1
        if next_generation >= spec.max_generations or remaining == 0:
            selected: list[OpenEvolutionProposalV42] = []
        else:
            prescribed = adapter.prescribed_evolve(
                spec,
                grammar,
                candidate,
                evaluation,
                failure,
                next_generation,
            )
            generated = adapter.generated_evolve(
                spec,
                grammar,
                candidate,
                evaluation,
                failure,
                next_generation,
            )
            for proposal in prescribed + generated:
                proposal.operator.assert_sealed()
                proposal.candidate.assert_sealed()
                if (
                    proposal.operator.source_candidate_hash
                    != candidate.candidate_hash
                    or proposal.operator.source_evaluation_hash
                    != evaluation.evaluation_hash
                    or proposal.operator.failure_signature_hash
                    != failure.failure_hash
                    or proposal.candidate.generation != next_generation
                ):
                    raise ValueError("evolution proposal is bound to another failure")
            prescribed_selected = _select_channel_proposals(
                prescribed,
                channel="prescribed",
                quota=min(spec.prescribed_quota_per_failure, remaining),
                known_identities=known,
            )
            generated_selected = _select_channel_proposals(
                generated,
                channel="generated",
                quota=min(spec.generated_quota_per_failure, remaining),
                known_identities=known,
            )
            if len(prescribed_selected) + len(generated_selected) <= remaining:
                selected = prescribed_selected + generated_selected
            elif prescribed_selected and generated_selected and remaining >= 2:
                selected = [prescribed_selected[0], generated_selected[0]]
                remainder = prescribed_selected[1:] + generated_selected[1:]
                selected.extend(
                    sorted(
                        remainder,
                        key=lambda item: (
                            -item.priority,
                            item.operator.channel,
                            item.candidate.candidate_hash,
                        ),
                    )[: remaining - 2]
                )
            else:
                selected = sorted(
                    prescribed_selected + generated_selected,
                    key=lambda item: (
                        -item.priority,
                        item.operator.channel,
                        item.candidate.candidate_hash,
                    ),
                )[:remaining]
        selected = sorted(
            selected,
            key=lambda item: (
                item.operator.channel,
                -item.priority,
                item.candidate.candidate_hash,
            ),
        )
        batch = HybridEvolutionBatchV42.seal(
            campaign_spec_hash=spec.spec_hash,
            source_candidate_hash=candidate.candidate_hash,
            source_evaluation_hash=evaluation.evaluation_hash,
            failure_signature_hash=failure.failure_hash,
            next_generation=candidate.generation + 1,
            proposals=selected,
        )
    batch.assert_sealed()
    ref = _put_once(graph, "hybrid_evolution_batch_v42", batch)
    failure_node = _node_by_artifact(
        graph, failure.failure_hash, node_kind="failure"
    )
    evaluation_node = _node_by_artifact(
        graph, evaluation.evaluation_hash, node_kind="evaluation"
    )
    node = _ensure_node(
        graph,
        _node(
            spec,
            node_id=_node_id("hybrid.evolution.batch", batch.batch_hash),
            node_kind="workflow_plan",
            executor="harness",
            created_by="harness",
            artifact_hash=batch.batch_hash,
            purpose="freeze prescribed and generated proposals before materialization",
        ),
    )
    _ensure_edge(
        graph,
        _edge(
            spec,
            edge_id=_node_id(
                "failure.batch",
                sha256_value([failure.failure_hash, batch.batch_hash]),
            ),
            source=failure_node,
            target=node,
            relation="requires_success",
            rationale="hybrid evolution batch is grounded in committed failure evidence",
        ),
    )
    _ensure_edge(
        graph,
        _edge(
            spec,
            edge_id=_node_id(
                "evaluation.batch",
                sha256_value([evaluation.evaluation_hash, batch.batch_hash]),
            ),
            source=evaluation_node,
            target=node,
            relation="requires_terminal",
            rationale="hybrid evolution starts only after terminal evaluation",
        ),
    )
    _ensure_outcome(
        graph,
        node,
        ref,
        actor="harness",
        status="succeeded",
        summary=(
            f"froze {len(batch.proposals)} prescribed/generated evolution proposals"
        ),
    )
    return batch, node


def _materialize_evolution_batch(
    graph: GraphLoopStoreV40,
    spec: OpenEvolutionCampaignSpecV42,
    batch: HybridEvolutionBatchV42,
    batch_node: GraphNodeV40,
) -> None:
    failure_node = _node_by_artifact(
        graph, batch.failure_signature_hash, node_kind="failure"
    )
    evaluation_node = _node_by_artifact(
        graph, batch.source_evaluation_hash, node_kind="evaluation"
    )
    for proposal in batch.proposals:
        operator = proposal.operator
        generation_node: GraphNodeV40 | None = None
        if proposal.generation_evidence is not None:
            evidence = proposal.generation_evidence
            evidence_ref = _put_once(
                graph, "generation_call_evidence_v42", evidence
            )
            generation_node = _ensure_node(
                graph,
                _node(
                    spec,
                    node_id=_node_id(
                        "generation.call", evidence.evidence_hash
                    ),
                    node_kind="generation_call",
                    executor="harness",
                    created_by="harness",
                    artifact_hash=evidence.evidence_hash,
                    purpose=(
                        f"record tool-free {evidence.transport} generated-channel "
                        "request and response"
                    ),
                ),
            )
            for source, label in (
                (batch_node, "batch"),
                (failure_node, "failure"),
            ):
                _ensure_edge(
                    graph,
                    _edge(
                        spec,
                        edge_id=_node_id(
                            f"{label}.generation",
                            sha256_value(
                                [source.node_hash, evidence.evidence_hash]
                            ),
                        ),
                        source=source,
                        target=generation_node,
                        relation="requires_success",
                        rationale=(
                            "generation call is bound to its frozen batch and "
                            "sanitized failure"
                        ),
                    ),
                )
            _ensure_outcome(
                graph,
                generation_node,
                evidence_ref,
                actor="harness",
                status="succeeded",
                summary=(
                    "tool-free generated-channel transport evidence committed"
                ),
            )
        operator_ref = _put_once(graph, "hybrid_evolution_operator_v42", operator)
        operator_node = _ensure_node(
            graph,
            _node(
                spec,
                node_id=operator.operator_id,
                node_kind="evolution_operator",
                executor=operator.proposed_by,
                created_by=operator.proposed_by,
                artifact_hash=operator.operator_hash,
                purpose=(
                    f"propose {operator.channel} {operator.kind} toward "
                    f"{operator.target_family}"
                ),
            ),
        )
        for source, relation, label in (
            (batch_node, "requires_success", "batch"),
            (failure_node, "requires_success", "failure"),
            (evaluation_node, "requires_terminal", "evaluation"),
        ):
            _ensure_edge(
                graph,
                _edge(
                    spec,
                    edge_id=_node_id(
                        f"{label}.operator",
                        sha256_value([source.node_hash, operator.operator_hash]),
                    ),
                    source=source,
                    target=operator_node,
                    relation=relation,
                    rationale="operator is bound to its frozen batch and failure context",
                ),
            )
        if generation_node is not None:
            _ensure_edge(
                graph,
                _edge(
                    spec,
                    edge_id=_node_id(
                        "generation.operator",
                        sha256_value(
                            [generation_node.node_hash, operator.operator_hash]
                        ),
                    ),
                    source=generation_node,
                    target=operator_node,
                    relation="requires_success",
                    rationale=(
                        "generated operator requires its recorded transport evidence"
                    ),
                ),
            )
        _ensure_outcome(
            graph,
            operator_node,
            operator_ref,
            actor=operator.proposed_by,
            status="succeeded",
            summary=f"recorded {operator.channel} evolution operator",
        )
        _ensure_candidate_proposal(
            graph,
            spec,
            proposal.candidate,
            batch_node=batch_node,
        )
    _ensure_checkpoint(
        graph,
        spec,
        phase="evolution_expanded",
        subject_hash=batch.failure_signature_hash,
        source_node=batch_node,
        source_relation="requires_success",
    )


def _ensure_recovery_evidence(
    graph: GraphLoopStoreV40,
    spec: OpenEvolutionCampaignSpecV42,
) -> None:
    incidents = _load_many(
        graph.store, "runtime_incident_v42", RuntimeIncidentV42
    )
    if len(incidents) >= spec.max_recovery_attempts:
        raise RuntimeError("open-evolution recovery budget is exhausted")
    checkpoints = _load_many(
        graph.store, "campaign_checkpoint_v42", CampaignCheckpointV42
    )
    state = graph.project_state()
    recovery_index = len(incidents) + 1
    last_checkpoint = (
        sorted(checkpoints, key=lambda item: item.checkpoint_index)[-1]
        if checkpoints
        else None
    )
    incident = RuntimeIncidentV42.seal(
        incident_id=f"recovery.incident.{recovery_index}",
        campaign_spec_hash=spec.spec_hash,
        recovery_index=recovery_index,
        detected_snapshot_hash=state.snapshot.snapshot_hash,
        last_checkpoint_hash=(
            last_checkpoint.checkpoint_hash if last_checkpoint else None
        ),
        pending_node_hashes=sorted(state.snapshot.frontier_node_hashes),
    )
    incident_ref = _put_once(graph, "runtime_incident_v42", incident)
    incident_node = _ensure_node(
        graph,
        _node(
            spec,
            node_id=incident.incident_id,
            node_kind="incident",
            executor="harness",
            created_by="harness",
            artifact_hash=incident.incident_hash,
            purpose="record detection of an incomplete replay-safe campaign",
        ),
    )
    _ensure_outcome(
        graph,
        incident_node,
        incident_ref,
        actor="harness",
        status="failed",
        summary="incomplete run detected; no external action will be retried",
    )

    patch = RecoveryPatchV42.seal(
        patch_id=f"recovery.patch.{recovery_index}",
        campaign_spec_hash=spec.spec_hash,
        incident_hash=incident.incident_hash,
        prevention_controls=[
            "content-addressed artifact reconciliation",
            "idempotency key bound to spec and candidate",
            "pending graph nodes resumed without history overwrite",
        ],
    )
    patch_ref = _put_once(graph, "recovery_patch_v42", patch)
    patch_node = _ensure_node(
        graph,
        _node(
            spec,
            node_id=patch.patch_id,
            node_kind="recovery_patch",
            executor="harness",
            created_by="harness",
            artifact_hash=patch.patch_hash,
            purpose="reconcile committed artifacts and resume only replay-safe work",
        ),
    )
    _ensure_edge(
        graph,
        _edge(
            spec,
            edge_id=_node_id(
                "incident.patch",
                sha256_value([incident.incident_hash, patch.patch_hash]),
            ),
            source=incident_node,
            target=patch_node,
            relation="learned_from_failure",
            rationale="recovery patch is causally grounded in the detected incident",
        ),
    )
    _ensure_outcome(
        graph,
        patch_node,
        patch_ref,
        actor="harness",
        status="succeeded",
        summary="committed graph reconciled without overwriting prior history",
    )
    _ensure_checkpoint(
        graph,
        spec,
        phase="reconciled",
        subject_hash=patch.patch_hash,
        source_node=patch_node,
        source_relation="requires_success",
    )


def _create_graph(
    output_root: Path,
    spec: OpenEvolutionCampaignSpecV42,
    grammar: OpenModelGrammarV42,
) -> GraphLoopStoreV40:
    max_nodes = (
        spec.max_candidates * 12
        + spec.max_recovery_attempts * 3
        + 16
    )
    graph = GraphLoopStoreV40(
        output_root,
        GraphLoopContractV40.seal(
            graph_id=spec.campaign_id,
            layer="modeling",
            evaluator_epoch=spec.evaluator_epoch,
            objective=spec.objective,
            max_nodes=max_nodes,
            max_outcomes=max_nodes,
            max_failures=spec.max_candidates * 2
            + spec.max_recovery_attempts
            + 4,
            max_promotions=1,
            created_at=spec.created_at,
        ),
    )
    spec_ref = _put_once(graph, "open_evolution_campaign_spec_v42", spec)
    spec_node = _ensure_node(
        graph,
        _node(
            spec,
            node_id=_node_id("campaign.contract", spec.spec_hash),
            node_kind="problem_contract",
            executor="harness",
            created_by="harness",
            artifact_hash=spec.spec_hash,
            purpose="freeze open-evolution authority, budget and private-data boundary",
        ),
    )
    _ensure_outcome(
        graph,
        spec_node,
        spec_ref,
        actor="harness",
        status="succeeded",
        summary="V4.2 open-evolution campaign contract frozen",
    )
    grammar_ref = _put_once(graph, "open_model_grammar_v42", grammar)
    grammar_node = _ensure_node(
        graph,
        _node(
            spec,
            node_id=_node_id("model.space", grammar.grammar_hash),
            node_kind="model_space",
            executor="harness",
            created_by="harness",
            artifact_hash=grammar.grammar_hash,
            purpose="freeze typed primitives without freezing future family names",
        ),
    )
    _ensure_edge(
        graph,
        _edge(
            spec,
            edge_id=_node_id(
                "contract.grammar",
                sha256_value([spec.spec_hash, grammar.grammar_hash]),
            ),
            source=spec_node,
            target=grammar_node,
            relation="requires_success",
            rationale="model-space grammar is scoped by the frozen campaign contract",
        ),
    )
    _ensure_outcome(
        graph,
        grammar_node,
        grammar_ref,
        actor="harness",
        status="succeeded",
        summary="typed open model grammar frozen",
    )
    return graph


def _load_outcome(
    graph: GraphLoopStoreV40,
    spec: OpenEvolutionCampaignSpecV42,
    grammar: OpenModelGrammarV42,
) -> OpenEvolutionOutcomeV42:
    report = _load_one(
        graph.store, "open_evolution_report_v42", OpenEvolutionReportV42
    )
    frontier = _load_one(
        graph.store, "initial_frontier_v42", InitialFrontierV42
    )
    champions = _load_many(
        graph.store,
        "development_champion_receipt_v42",
        DevelopmentChampionReceiptV42,
    )
    if len(champions) > 1:
        raise RuntimeError("open-evolution run contains duplicate champions")
    outcome = OpenEvolutionOutcomeV42(
        graph=graph,
        spec=spec,
        grammar=grammar,
        initial_frontier=frontier,
        candidates=_load_many(
            graph.store, "open_model_candidate_v42", OpenModelCandidateV42
        ),
        validations=_load_many(
            graph.store, "model_space_validation_v42", ModelSpaceValidationV42
        ),
        admissions=_load_many(
            graph.store, "model_admission_receipt_v42", ModelAdmissionReceiptV42
        ),
        attempts=_load_many(
            graph.store, "execution_attempt_v42", ExecutionAttemptV42
        ),
        executions=_load_many(
            graph.store, "development_execution_v42", DevelopmentExecutionV42
        ),
        evaluations=_load_many(
            graph.store, "development_evaluation_v42", DevelopmentEvaluationV42
        ),
        failures=_load_many(
            graph.store, "open_failure_signature_v42", OpenFailureSignatureV42
        ),
        operators=_load_many(
            graph.store,
            "hybrid_evolution_operator_v42",
            HybridEvolutionOperatorV42,
        ),
        evolution_batches=_load_many(
            graph.store, "hybrid_evolution_batch_v42", HybridEvolutionBatchV42
        ),
        generation_calls=_load_many(
            graph.store,
            "generation_call_evidence_v42",
            GenerationCallEvidenceV42,
        ),
        checkpoints=_load_many(
            graph.store, "campaign_checkpoint_v42", CampaignCheckpointV42
        ),
        incidents=_load_many(
            graph.store, "runtime_incident_v42", RuntimeIncidentV42
        ),
        recovery_patches=_load_many(
            graph.store, "recovery_patch_v42", RecoveryPatchV42
        ),
        champion=champions[0] if champions else None,
        report=report,
    )
    if not verify_open_evolution_campaign_v42(outcome, spec, grammar):
        raise RuntimeError("completed V4.2 open-evolution run failed verification")
    return outcome


def _open_existing(
    run_directory: Path,
    spec: OpenEvolutionCampaignSpecV42,
    grammar: OpenModelGrammarV42,
) -> tuple[GraphLoopStoreV40, OpenEvolutionOutcomeV42 | None]:
    graph = GraphLoopStoreV40.open_existing(run_directory)
    stored_spec = _load_one(
        graph.store,
        "open_evolution_campaign_spec_v42",
        OpenEvolutionCampaignSpecV42,
    )
    stored_grammar = _load_one(
        graph.store, "open_model_grammar_v42", OpenModelGrammarV42
    )
    if stored_spec != spec or stored_grammar != grammar:
        raise RuntimeError("resumed V4.2 campaign contract or grammar differs")
    reports = _artifact_refs(graph.store, "open_evolution_report_v42")
    if reports:
        if len(reports) != 1:
            raise RuntimeError("open-evolution run contains duplicate reports")
        return graph, _load_outcome(graph, spec, grammar)
    return graph, None


def _ensure_champion(
    graph: GraphLoopStoreV40,
    spec: OpenEvolutionCampaignSpecV42,
) -> DevelopmentChampionReceiptV42 | None:
    champions = _load_many(
        graph.store,
        "development_champion_receipt_v42",
        DevelopmentChampionReceiptV42,
    )
    if champions:
        if len(champions) != 1:
            raise RuntimeError("open-evolution run contains duplicate champions")
        return champions[0]
    evaluations = _load_many(
        graph.store, "development_evaluation_v42", DevelopmentEvaluationV42
    )
    passing = [item for item in evaluations if all(item.gates.values())]
    if not passing:
        return None
    best = sorted(passing, key=lambda item: (-item.utility, item.candidate_hash))[0]
    admission = next(
        item
        for item in _load_many(
            graph.store, "model_admission_receipt_v42", ModelAdmissionReceiptV42
        )
        if item.candidate_hash == best.candidate_hash
    )
    champion = DevelopmentChampionReceiptV42.seal(
        campaign_spec_hash=spec.spec_hash,
        candidate_hash=best.candidate_hash,
        admission_receipt_hash=admission.receipt_hash,
        evaluation_hash=best.evaluation_hash,
        considered_evaluation_hashes=sorted(
            item.evaluation_hash for item in evaluations
        ),
    )
    ref = _put_once(graph, "development_champion_receipt_v42", champion)
    admission_node = _node_by_artifact(
        graph, admission.receipt_hash, node_kind="model_admission"
    )
    evaluation_node = _node_by_artifact(
        graph, best.evaluation_hash, node_kind="evaluation"
    )
    node = _ensure_node(
        graph,
        _node(
            spec,
            node_id=_node_id("development.champion", champion.receipt_hash),
            node_kind="decision",
            executor="verifier",
            created_by="harness",
            artifact_hash=champion.receipt_hash,
            purpose="freeze the best admitted development candidate without qualification",
        ),
    )
    for source, label in (
        (admission_node, "admission"),
        (evaluation_node, "evaluation"),
    ):
        _ensure_edge(
            graph,
            _edge(
                spec,
                edge_id=_node_id(
                    f"{label}.champion",
                    sha256_value([source.node_hash, champion.receipt_hash]),
                ),
                source=source,
                target=node,
                relation="requires_success",
                rationale="development champion requires admission and passing evaluation",
            ),
        )
    _ensure_outcome(
        graph,
        node,
        ref,
        actor="verifier",
        status="succeeded",
        summary="development champion frozen; qualification remains forbidden",
    )
    _ensure_checkpoint(
        graph,
        spec,
        phase="champion_frozen",
        subject_hash=champion.receipt_hash,
        source_node=node,
        source_relation="requires_success",
    )
    return champion


def run_open_evolution_campaign_v42(
    output_root: str | Path,
    spec: OpenEvolutionCampaignSpecV42,
    grammar: OpenModelGrammarV42,
    adapter: OpenEvolutionAdapterV42,
) -> OpenEvolutionOutcomeV42:
    """Run or recover one graph-native open-evolution development campaign."""

    spec.assert_sealed()
    grammar.assert_sealed()
    if spec.grammar_hash != grammar.grammar_hash:
        raise ValueError("campaign spec is bound to another model grammar")
    if adapter.adapter_id not in grammar.executable_adapter_ids:
        raise ValueError("adapter is not registered in the frozen grammar")
    if (
        grammar.executable_adapter_hashes[adapter.adapter_id]
        != adapter.adapter_contract_hash
    ):
        raise ValueError("adapter implementation differs from the frozen grammar")
    root = Path(output_root).resolve()
    run_directory = root / spec.campaign_id
    if run_directory.is_dir():
        graph, completed = _open_existing(run_directory, spec, grammar)
        if completed is not None:
            return completed
        _ensure_recovery_evidence(graph, spec)
    else:
        graph = _create_graph(root, spec, grammar)

    initial_frontier, frontier_node = _ensure_initial_frontier(
        graph, spec, grammar, adapter
    )
    for candidate in initial_frontier.candidates:
        _ensure_candidate_proposal(
            graph,
            spec,
            candidate,
            batch_node=frontier_node,
        )

    processed: set[str] = set()
    while True:
        candidates = sorted(
            _load_many(
                graph.store, "open_model_candidate_v42", OpenModelCandidateV42
            ),
            key=lambda item: (item.generation, item.candidate_hash),
        )
        candidate = next(
            (
                item
                for item in candidates
                if item.candidate_hash not in processed
            ),
            None,
        )
        if candidate is None:
            break
        processed.add(candidate.candidate_hash)
        candidate.assert_sealed()
        validation, _ = _ensure_space_validation(
            graph, spec, grammar, candidate, adapter
        )
        if not validation.admitted:
            continue
        admission, _ = _ensure_admission(
            graph, spec, grammar, candidate, validation
        )
        _, execution, execution_node = _ensure_execution(
            graph, spec, candidate, admission, adapter
        )
        evaluation, _ = _ensure_development_evaluation(
            graph,
            spec,
            candidate,
            execution,
            execution_node,
            adapter,
        )
        if all(evaluation.gates.values()):
            continue
        failure = next(
            item
            for item in _load_many(
                graph.store,
                "open_failure_signature_v42",
                OpenFailureSignatureV42,
            )
            if item.candidate_hash == candidate.candidate_hash
            and item.stage == "development"
        )
        batch, batch_node = _ensure_evolution_batch(
            graph,
            spec,
            grammar,
            candidate,
            evaluation,
            failure,
            adapter,
        )
        _materialize_evolution_batch(graph, spec, batch, batch_node)

    champion = _ensure_champion(graph, spec)
    replay_verified = graph.verify()
    candidates = _load_many(
        graph.store, "open_model_candidate_v42", OpenModelCandidateV42
    )
    validations = _load_many(
        graph.store, "model_space_validation_v42", ModelSpaceValidationV42
    )
    admissions = _load_many(
        graph.store, "model_admission_receipt_v42", ModelAdmissionReceiptV42
    )
    executions = _load_many(
        graph.store, "development_execution_v42", DevelopmentExecutionV42
    )
    evaluations = _load_many(
        graph.store, "development_evaluation_v42", DevelopmentEvaluationV42
    )
    failures = _load_many(
        graph.store, "open_failure_signature_v42", OpenFailureSignatureV42
    )
    operators = _load_many(
        graph.store,
        "hybrid_evolution_operator_v42",
        HybridEvolutionOperatorV42,
    )
    batches = _load_many(
        graph.store, "hybrid_evolution_batch_v42", HybridEvolutionBatchV42
    )
    generation_calls = _load_many(
        graph.store,
        "generation_call_evidence_v42",
        GenerationCallEvidenceV42,
    )
    checkpoints = _load_many(
        graph.store, "campaign_checkpoint_v42", CampaignCheckpointV42
    )
    incidents = _load_many(
        graph.store, "runtime_incident_v42", RuntimeIncidentV42
    )
    patches = _load_many(
        graph.store, "recovery_patch_v42", RecoveryPatchV42
    )
    report = OpenEvolutionReportV42.seal(
        campaign_id=spec.campaign_id,
        campaign_spec_hash=spec.spec_hash,
        grammar_hash=grammar.grammar_hash,
        terminal_status=(
            "development_champion_frozen"
            if champion is not None
            else "no_development_candidate"
        ),
        candidate_hashes=sorted(item.candidate_hash for item in candidates),
        validation_hashes=sorted(item.validation_hash for item in validations),
        admission_receipt_hashes=sorted(
            item.receipt_hash for item in admissions
        ),
        execution_hashes=sorted(item.execution_hash for item in executions),
        evaluation_hashes=sorted(item.evaluation_hash for item in evaluations),
        failure_hashes=sorted(item.failure_hash for item in failures),
        operator_hashes=sorted(item.operator_hash for item in operators),
        evolution_batch_hashes=sorted(item.batch_hash for item in batches),
        generation_call_evidence_hashes=sorted(
            item.evidence_hash for item in generation_calls
        ),
        checkpoint_hashes=sorted(item.checkpoint_hash for item in checkpoints),
        incident_hashes=sorted(item.incident_hash for item in incidents),
        recovery_patch_hashes=sorted(item.patch_hash for item in patches),
        prescribed_operator_count=sum(
            item.channel == "prescribed" for item in operators
        ),
        generated_operator_count=sum(
            item.channel == "generated" for item in operators
        ),
        champion_receipt_hash=champion.receipt_hash if champion else None,
        champion_candidate_hash=champion.candidate_hash if champion else None,
        graph_snapshot_hash=graph.project_state().snapshot.snapshot_hash,
        graph_replay_verified=replay_verified,
        recovered_from_incomplete_run=bool(incidents),
        created_at=spec.created_at,
    )
    _put_once(graph, "open_evolution_report_v42", report)
    return _load_outcome(graph, spec, grammar)


def verify_open_evolution_campaign_v42(
    outcome: OpenEvolutionOutcomeV42,
    expected_spec: OpenEvolutionCampaignSpecV42,
    expected_grammar: OpenModelGrammarV42,
) -> bool:
    try:
        expected_spec.assert_sealed()
        expected_grammar.assert_sealed()
        outcome.initial_frontier.assert_sealed()
        outcome.report.assert_sealed()
        if outcome.spec != expected_spec or outcome.grammar != expected_grammar:
            return False
        if expected_spec.grammar_hash != expected_grammar.grammar_hash:
            return False
        for item in (
            outcome.candidates
            + outcome.validations
            + outcome.admissions
            + outcome.attempts
            + outcome.executions
            + outcome.evaluations
            + outcome.failures
            + outcome.operators
            + outcome.evolution_batches
            + outcome.generation_calls
            + outcome.checkpoints
            + outcome.incidents
            + outcome.recovery_patches
        ):
            item.assert_sealed()
        if outcome.champion:
            outcome.champion.assert_sealed()
        if not outcome.graph.verify():
            return False
        state = outcome.graph.project_state()
        if state.promotions:
            return False
        if any(
            status in {"qualified", "active"}
            for status in state.snapshot.node_statuses.values()
        ):
            return False
        if state.snapshot.frontier_node_hashes:
            return False
        if outcome.report.graph_snapshot_hash != state.snapshot.snapshot_hash:
            return False
        if outcome.report.campaign_spec_hash != expected_spec.spec_hash:
            return False
        if outcome.report.grammar_hash != expected_grammar.grammar_hash:
            return False

        candidates = {item.candidate_hash: item for item in outcome.candidates}
        if len(candidates) != len(outcome.candidates):
            return False
        if len(candidates) > expected_spec.max_candidates:
            return False
        frontier_hashes = {
            item.candidate_hash for item in outcome.initial_frontier.candidates
        }
        if not frontier_hashes.issubset(candidates):
            return False
        validations = {item.candidate_hash: item for item in outcome.validations}
        if set(validations) != set(candidates):
            return False
        admissions = {item.candidate_hash: item for item in outcome.admissions}
        attempts = {item.candidate_hash: item for item in outcome.attempts}
        executions = {item.candidate_hash: item for item in outcome.executions}
        evaluations = {item.candidate_hash: item for item in outcome.evaluations}
        admitted_hashes = {
            candidate_hash
            for candidate_hash, validation in validations.items()
            if validation.admitted
        }
        if set(admissions) != admitted_hashes:
            return False
        if set(attempts) != admitted_hashes:
            return False
        if set(executions) != admitted_hashes:
            return False
        if set(evaluations) != admitted_hashes:
            return False
        if any(
            validation.grammar_hash != expected_grammar.grammar_hash
            or validation.evaluator_epoch != expected_spec.evaluator_epoch
            or validation.private_data_accessed
            for validation in outcome.validations
        ):
            return False
        for candidate_hash in admitted_hashes:
            admission = admissions[candidate_hash]
            validation = validations[candidate_hash]
            attempt = attempts[candidate_hash]
            execution = executions[candidate_hash]
            evaluation = evaluations[candidate_hash]
            expected_key = sha256_value(
                {
                    "campaign_spec_hash": expected_spec.spec_hash,
                    "candidate_hash": candidate_hash,
                    "operation": "development_execute_v42",
                }
            )
            if (
                admission.validation_hash != validation.validation_hash
                or attempt.idempotency_key != expected_key
                or attempt.executable_adapter_hash
                != expected_grammar.executable_adapter_hashes[
                    attempt.executable_adapter_id
                ]
                or execution.attempt_hash != attempt.attempt_hash
                or execution.idempotency_key != expected_key
                or execution.development_data_hash
                != expected_spec.development_data_hash
                or evaluation.execution_hash != execution.execution_hash
                or sorted(evaluation.gates)
                != expected_spec.required_development_gates
                or evaluation.private_data_accessed
            ):
                return False
        if any(item.private_data_exposed for item in outcome.failures):
            return False

        operators = {item.operator_hash: item for item in outcome.operators}
        if len(operators) != len(outcome.operators):
            return False
        for operator in outcome.operators:
            expected_actor = (
                "harness" if operator.channel == "prescribed" else "model"
            )
            if operator.proposed_by != expected_actor:
                return False
        batch_operators = {
            proposal.operator.operator_hash
            for batch in outcome.evolution_batches
            for proposal in batch.proposals
        }
        batch_candidates = {
            proposal.candidate.candidate_hash
            for batch in outcome.evolution_batches
            for proposal in batch.proposals
        }
        if batch_operators != set(operators):
            return False
        if not batch_candidates.issubset(candidates):
            return False
        batch_generation_calls = {
            proposal.generation_evidence.evidence_hash
            for batch in outcome.evolution_batches
            for proposal in batch.proposals
            if proposal.generation_evidence is not None
        }
        if batch_generation_calls != {
            item.evidence_hash for item in outcome.generation_calls
        }:
            return False
        for evidence in outcome.generation_calls:
            generation_node = _node_by_artifact(
                outcome.graph,
                evidence.evidence_hash,
                node_kind="generation_call",
            )
            if state.snapshot.node_statuses[generation_node.node_hash] != "succeeded":
                return False
            if (
                evidence.private_evidence_exposed
                or evidence.authority_fields_exposed
                or evidence.tool_events_observed
            ):
                return False

        if len(outcome.incidents) != len(outcome.recovery_patches):
            return False
        edges = state.edges
        checkpoints_by_subject = {
            (item.phase, item.subject_hash) for item in outcome.checkpoints
        }
        for incident, patch in zip(
            sorted(outcome.incidents, key=lambda item: item.recovery_index),
            sorted(
                outcome.recovery_patches,
                key=lambda item: int(item.patch_id.rsplit(".", 1)[1]),
            ),
        ):
            if patch.incident_hash != incident.incident_hash:
                return False
            incident_node = _node_by_artifact(
                outcome.graph, incident.incident_hash, node_kind="incident"
            )
            patch_node = _node_by_artifact(
                outcome.graph, patch.patch_hash, node_kind="recovery_patch"
            )
            if not any(
                edge.source_node_hash == incident_node.node_hash
                and edge.target_node_hash == patch_node.node_hash
                and edge.relation == "learned_from_failure"
                for edge in edges
            ):
                return False
            if ("reconciled", patch.patch_hash) not in checkpoints_by_subject:
                return False
            if patch.external_action_retried:
                return False

        if outcome.report.candidate_hashes != sorted(candidates):
            return False
        if outcome.report.validation_hashes != sorted(
            item.validation_hash for item in outcome.validations
        ):
            return False
        if outcome.report.admission_receipt_hashes != sorted(
            item.receipt_hash for item in outcome.admissions
        ):
            return False
        if outcome.report.execution_hashes != sorted(
            item.execution_hash for item in outcome.executions
        ):
            return False
        if outcome.report.evaluation_hashes != sorted(
            item.evaluation_hash for item in outcome.evaluations
        ):
            return False
        if outcome.report.failure_hashes != sorted(
            item.failure_hash for item in outcome.failures
        ):
            return False
        if outcome.report.operator_hashes != sorted(operators):
            return False
        if outcome.report.evolution_batch_hashes != sorted(
            item.batch_hash for item in outcome.evolution_batches
        ):
            return False
        if outcome.report.generation_call_evidence_hashes != sorted(
            item.evidence_hash for item in outcome.generation_calls
        ):
            return False
        if outcome.report.checkpoint_hashes != sorted(
            item.checkpoint_hash for item in outcome.checkpoints
        ):
            return False
        if outcome.report.incident_hashes != sorted(
            item.incident_hash for item in outcome.incidents
        ):
            return False
        if outcome.report.recovery_patch_hashes != sorted(
            item.patch_hash for item in outcome.recovery_patches
        ):
            return False
        if outcome.report.prescribed_operator_count != sum(
            item.channel == "prescribed" for item in outcome.operators
        ):
            return False
        if outcome.report.generated_operator_count != sum(
            item.channel == "generated" for item in outcome.operators
        ):
            return False
        if outcome.report.recovered_from_incomplete_run != bool(outcome.incidents):
            return False

        if outcome.champion:
            evaluation = evaluations[outcome.champion.candidate_hash]
            admission = admissions[outcome.champion.candidate_hash]
            if (
                not all(evaluation.gates.values())
                or outcome.champion.evaluation_hash != evaluation.evaluation_hash
                or outcome.champion.admission_receipt_hash != admission.receipt_hash
                or outcome.report.champion_candidate_hash
                != outcome.champion.candidate_hash
            ):
                return False
        elif outcome.report.terminal_status != "no_development_candidate":
            return False
    except (
        ValueError,
        RuntimeError,
        KeyError,
        StopIteration,
        OSError,
        json.JSONDecodeError,
    ):
        return False
    return True
